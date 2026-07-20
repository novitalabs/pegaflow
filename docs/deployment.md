# Deployment

## P/D with NIXL

This GLM-5.2 FP8 example enables MTP and `MultiConnector` on both sides. NIXL
handles P-to-D transfer; PegaFlow uses `read_write` on P and `save_only` on D
to reuse KV produced by earlier requests without competing with NIXL loads.

Run one PegaFlow server beside each vLLM instance, backed by the same metaserver,
and use a P/D-aware NIXL router. P and D must use compatible model, tokenizer,
block size, KV dtype, KV layout, and `PYTHONHASHSEED`. The example uses seed 42
and a local PegaFlow server at `http://127.0.0.1:50055`.

### Prefill

```bash
PYTHONHASHSEED=42 \
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
  --kv-transfer-config '{
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
            "pegaflow.host": "http://127.0.0.1",
            "pegaflow.port": 50055,
            "pegaflow.mode": "read_write"
          }
        }
      ]
    }
  }'
```

### Decode

```bash
PYTHONHASHSEED=42 \
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
  --kv-transfer-config '{
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
            "pegaflow.host": "http://127.0.0.1",
            "pegaflow.port": 50055,
            "pegaflow.mode": "save_only"
          }
        }
      ]
    }
  }'
```

### Validate D-to-P reuse

Generate a multi-block response, then send the full history in turn two. A hit
for the first response exercises D-to-P reuse. Verify:

- `pegaflow_rdma_fetch_total_total{status="ok"}` increases on the prefill-side
  server and `query_blocks_for_transfer` increases on decode.
- Every requested block is transferred with no block-count mismatch.
- `vllm:pega_load_failure_total`, `vllm:pega_save_failure_total`, and non-OK
  PegaFlow RPC deltas stay at zero.
