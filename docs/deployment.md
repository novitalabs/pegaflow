# Deployment

## P/D with NIXL

This example runs GLM-5.2 FP8 with prefill/decode disaggregation, MTP, and
vLLM's `MultiConnector` on both sides.

```text
request -> prefill -- NIXL --> decode
              |                   |
              +---- PegaFlow -----+
```

NIXL transfers the current request's KV cache from prefill to decode.
PegaFlow stores reusable KV cache across requests and nodes:

- The prefill instance uses PegaFlow in `read_write` mode. It can reuse a
  cached prefix, including output KV previously saved by a decode instance.
- The decode instance puts NIXL first and uses PegaFlow in `save_only` mode.
  NIXL owns the current P-to-D load, while PegaFlow persists the resulting KV
  cache without competing for that load.

### Prerequisites

- Run a PegaFlow server beside each vLLM instance and connect those servers to
  the same PegaFlow metaserver.
- Configure NIXL networking and a P/D-aware router for the deployment
  environment.
- Use the same model revision, tokenizer, block size, KV cache dtype,
  `PYTHONHASHSEED`, and compatible KV cache layout on both sides.
- Set `PYTHONHASHSEED` explicitly. Different values produce different vLLM
  block hashes and prevent cross-instance cache reuse.

The following commands assume the local PegaFlow server listens on
`http://127.0.0.1:50055`. Parallelism and memory limits are examples; adjust
them together with the router configuration for the target hardware.

### Prefill

The first child that advertises a cache hit owns the load. NIXL therefore stays
first, while PegaFlow remains available for cross-request reads and saves.

```bash
PREFILL_KV_CONFIG='{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "connectors": [
      {
        "kv_connector": "NixlConnector",
        "kv_role": "kv_producer"
      },
      {
        "kv_connector": "PegaKVConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": "pegaflow.connector",
        "kv_connector_extra_config": {
          "pegaflow.mode": "read_write"
        }
      }
    ]
  }
}'

PYTHONHASHSEED=42 \
PEGAFLOW_HOST=http://127.0.0.1 \
PEGAFLOW_PORT=50055 \
vllm serve GLM-5.2-FP8 \
  --served-model-name glm-5.2 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 2 \
  --enable-prefix-caching \
  --trust-remote-code \
  --kv-transfer-config "$PREFILL_KV_CONFIG"
```

### Decode

The decode side keeps NIXL as the load owner. PegaFlow receives every save but
does not query or load cache entries in `save_only` mode.

```bash
DECODE_KV_CONFIG='{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "connectors": [
      {
        "kv_connector": "NixlConnector",
        "kv_role": "kv_consumer",
        "kv_connector_extra_config": {
          "kv_recompute_threshold": 0
        }
      },
      {
        "kv_connector": "PegaKVConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": "pegaflow.connector",
        "kv_connector_extra_config": {
          "pegaflow.mode": "save_only"
        }
      }
    ]
  }
}'

PYTHONHASHSEED=42 \
PEGAFLOW_HOST=http://127.0.0.1 \
PEGAFLOW_PORT=50055 \
vllm serve GLM-5.2-FP8 \
  --served-model-name glm-5.2 \
  --host 0.0.0.0 \
  --port 8001 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 2 \
  --enable-prefix-caching \
  --trust-remote-code \
  --kv-transfer-config "$DECODE_KV_CONFIG"
```

### Validate D-to-P reuse

Use a two-turn request whose first response is long enough to span several KV
blocks. Send the complete conversation history on the second turn. The prefill
instance never computed the first response, so a PegaFlow hit for those blocks
exercises the D-to-P path.

Verify all of the following:

- `pegaflow_rdma_fetch_total_total{status="ok"}` increases on the prefill-side
  PegaFlow server.
- `query_blocks_for_transfer` requests increase on a decode-side PegaFlow
  server.
- PegaFlow reports all requested blocks as transferred, with no block-count
  mismatch.
- `vllm:pega_load_failure_total`, `vllm:pega_save_failure_total`, and non-OK
  PegaFlow RPC deltas remain zero.
