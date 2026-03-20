# vLLM Patch for Better I/O Performance

To maximize I/O throughput when using PegaFlow with vLLM, we recommend a small patch to vLLM's KV cache block allocation. Sorting block IDs ensures GPU memory addresses are as sequential and contiguous as possible, which improves DMA/RDMA transfer efficiency.

Upstream RFC: https://github.com/vllm-project/vllm/issues/31371

Locate the file `vllm/v1/core/kv_cache_utils.py` in your vLLM installation (e.g., `.venv/lib/python3.10/site-packages/vllm/v1/core/kv_cache_utils.py`), find the `append_n` method in the `FreeKVCacheBlockQueue` class, and add a sorting line:

```python
def append_n(self, blocks: list[KVCacheBlock]) -> None:
    """Put a list of blocks back into the free list

    Args:
        blocks: The blocks to append.
    """
    if len(blocks) == 0:
        return

    blocks.sort(key=lambda x: x.block_id)  # <-- Add this line
    last_block = self.fake_free_list_tail.prev_free_block
    ...
```

This simple change can noticeably reduce transfer latency by enabling more efficient memory access patterns during KV cache operations.
