# Deployment

## P/D with NIXL

This GLM-5.2 FP8 example enables MTP and `MultiConnector` on both sides. NIXL
handles P-to-D transfer; PegaFlow uses `read_write` on P and `save_only` on D
to reuse KV produced by earlier requests without competing with NIXL loads.

Run one PegaFlow server beside each vLLM instance, backed by the same metaserver,
and use a P/D-aware NIXL router. Configure each server's routable `--addr`,
`--nics`, and `--metaserver-addr` as described in the [P2P guide](./p2p.md).
P and D must use compatible model, tokenizer, block size, KV dtype, KV layout,
and `PYTHONHASHSEED`.

Replace `<p_node_ip>` and `<d_node_ip>` with the addresses assigned to the P
and D nodes. The example assumes P and D run on separate nodes; to colocate
them on one host, keep the distinct NIXL side-channel ports and also give the
decode-side PegaFlow server and connector a different port, since one PegaFlow
server runs beside each vLLM instance.

### Prefill

```bash
PYTHONHASHSEED=42 \
VLLM_NIXL_SIDE_CHANNEL_HOST=<p_node_ip> \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
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
            "pegaflow.host": "http://<p_node_ip>",
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
VLLM_NIXL_SIDE_CHANNEL_HOST=<d_node_ip> \
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
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
            "pegaflow.host": "http://<d_node_ip>",
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
