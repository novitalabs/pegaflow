# P/D CPU-Staging Push TODO

## P0: Correctness: Repeated P/D Requests

Status: fixed in the current branch and validated on h20 with two consecutive
TP4 P/D requests using the same prompt.

Root cause: P/D egress was coupled to normal async-save intents. On a repeated
prompt, the P side can satisfy the request from its local cache, so no new
save intent is generated and D waits forever for a lease that P never writes.

Fix:

- Generate P/D egress intents independently from normal save intents once the
  full source block span is ready on P.
- Treat locally loaded cache blocks as ready P/D source blocks.
- Before P/D egress, wait for any in-flight source-side load to complete.

## P0: RDMA Write Performance

Status: fixed in the current branch and validated on h20.

Target for the TP4 -> TP4 validation path:

- GPU layout should be NUMA-balanced: P `0,2,4,6`, D `1,3,5,7`.
- NICs: `mlx5_1,mlx5_2,mlx5_3,mlx5_4`.
- Expected fabric class: 400Gbps.
- Acceptable RDMA WRITE throughput: about 70% of line rate, roughly 300Gbps
  for each P rank -> D rank push payload.

Validation:

- CPU RDMA bench baseline on `mlx5_1`: 63MB WRITE p50 about 1.48ms,
  about 357Gbps.
- Minimal GPU-memory verbs bench on GPU/NIC PIX pairs: about 1.42ms,
  about 372Gbps.
- P/D TP4 smoke after caching topology selection:
  - first request pure RDMA write: about 1.46-1.49ms per rank,
    about 355-362Gbps;
  - second request with existing RDMA connection: total egress about
    2.3-2.6ms per rank, pure RDMA write about 355-364Gbps.

Root cause: the hot-path `rdma_write_ms` included per-request topology
detection through `SystemTopology::detect()`, which invokes `nvidia-smi`.
Preferred NIC selection is now cached in the worker instead of being recomputed
inside each transfer.

## P0: Remove Loopback Assumptions

The validation stack should use the host's private 10.x address for PegaFlow
and vLLM endpoints. `127.0.0.1`/`localhost` hides routing, advertise-address,
and RDMA peer-key mistakes.

Status: fixed in the current branch.

- Default the launcher host to `PEGAFLOW_PD_HOST`, then detected 10.x address.
- Keep vLLM bind host as `0.0.0.0`, but use the private IP in router and
  PegaFlow endpoint URLs.
- Verify metaserver advertise paths, router endpoint paths, and D descriptor
  polling all work through the private IP.

## P0: Short Prompt P/D Receive

Status: fixed in the current branch and validated on h20.

Root cause: the first P/D receive path sized D staging from
`len(request.block_hashes)`. vLLM only creates hashes for complete scheduler
blocks, so short prompts and partial tail blocks can have fewer hashes than the
KV blocks that still need to be transferred.

Fix:

- Size P/D receive from token count, matching vLLM P2P remote-prefill behavior:
  transfer `prompt_tokens - 1 - local_hit_tokens`.
- Allocate `ceil(tokens_to_transfer / virtual_block_size)` D staging blocks.
- Treat `block_hashes` as optional prefix metadata; allow the final partial
  receive block to have no hash.
- Return the exact external token count to vLLM after IMM is ready, not
  `num_blocks * block_size`.

Validation:

- Direct TP4 P/D smoke with `hello` completed without crashing. The Qwen3
  tokenizer maps that prompt to one prompt token, so no external KV transfer is
  required.
- Direct TP4 P/D smoke with a 17-token short prompt transferred one receive
  block and completed normally.
- Concurrent TP4 P/D smoke hit the new batch load path:
  `reqs=3 items=3 blocks=213` per D worker rank.

## P1: Failure And Cleanup

- P writes partial data and dies before IMM.
- D observes IMM but H2D load fails.
- Lease expires before P writes.
- Worker tries `LoadPdReceive` after lease expiry.
- Multiple in-flight leases with independent `imm_data`.
- P/D process shutdown while CUDA IPC and staging allocations are still live.

## P1: Concurrency And Admission

- Multi-request P/D pressure test.
- P/D push traffic should share a real admission policy with local save/offload
  and remote cache traffic.
- Pinned CPU staging pool pressure and fragmentation need metrics.

## P2: Heterogeneous TP

First validated path is TP4 == TP4. Later support should cover:

- TP8 -> TP4 fan-in.
- TP4 -> TP8 fan-out.
- Multiple P ranks contributing to one D receive rank.
- Expected IMM count derived from the actual fan-in plan, not only
  `receive_rank_count * nic_count`.

## P2: Delta Transfer

If D already has part of the prompt KV, P should eventually push only the
missing block delta. Leave this behind P2P/cache discovery so the router does
not become a tokenizer or block-manager proxy.

## P3: Direct GPU Push

CUDA IPC imported GPU pointers failed RDMA registration in validation. Direct
GPU push needs registration in the vLLM allocation-owning process and separate
GPU/RDMA ordering validation.
