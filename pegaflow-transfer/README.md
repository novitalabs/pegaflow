# PegaFlow Transfer

RDMA verbs transfer engine for PegaFlow (upstream-derived from `pplx-garden`). Provides a one-sided RDMA WRITE data plane with a UD control plane that establishes RC connections lazily per peer.

## Overview

PegaFlow Transfer moves memory between nodes over InfiniBand/RoCE fabrics. Peers only need to exchange memory-region descriptors out of band (PegaFlow does this over gRPC); device discovery, queue pair setup, and connection establishment are handled internally.

Key capabilities:

- **One-sided RDMA WRITE**: Single, Paged, Scatter, Imm, and Barrier transfer requests
- **Lazy connection setup**: a UD control plane handshakes RC queue pairs on first transfer to a peer — no out-of-band connect step
- **Multi-NIC sharding**: a transfer can shard bytes across all NICs of a domain group (`GroupTransferRouting::AllDomainsShardBytes`)
- **NUMA-aware topology**: `detect_topology()` groups NICs by GPU PCIe locality; `detect_host_topology()` groups NICs by NUMA node for host-memory (CPU) engines
- **Pinned worker threads**: each domain group is driven by one polling worker thread, optionally pinned to a NUMA-local CPU

## Architecture

```
TransferEngine (public API)
  ├── submit_transfer / submit_transfer_async / submit_transfer_atomic
  ├── register_memory_local / register_memory_allow_remote
  └── FabricEngine
        └── one Worker thread per domain group
              └── DomainGroup<VerbsDomain, N>   (N = NICs in the group, 1..=8)
                    ├── UD QP   – control plane (connection handshake)
                    └── RC QPs  – data plane (one-sided WRITE), per peer
```

A `MemoryRegionDescriptor` carries the region base pointer plus one `(DomainAddress, rkey)` pair per NIC; the writer pairs its NICs with the remote NICs by index, so both sides of a transfer must expose matching NIC counts per group.

## Building engines

- `TransferEngineBuilder::new(...).add_gpu_domains(...).build()` — GPU-centric engine, one worker per GPU with its PCIe-local NICs (P/D disaggregation path).
- `TransferEngineBuilder::build_host(domains, pin_worker_cpu)` — host-memory engine driving one NUMA node's NICs (cross-node KV fetch path; `pegaflow-core` creates one per NUMA node).

## CLI Tools

### pegaflow-cpu-bench

RDMA push benchmark over local engines, modelling KV-block transfer workloads:

```bash
# Default: host memory, 50 tasks, 150 blocks x 4MB
cargo run --release --bin pegaflow-cpu-bench

# Single NIC, custom block size
cargo run --release --bin pegaflow-cpu-bench -- --domains mlx5_0 --block-size 2mb
```

Run with `--help` for memory kind, pool size, task count, and NIC selection options.

## Requirements

- Linux with RDMA-capable NICs (InfiniBand or RoCE)
- `libibverbs` installed
- CUDA toolkit (feature `cuda-12` default, or `cuda-13`)

## Environment Variables

- `RUST_LOG`: Control logging level (default: `info,pegaflow_transfer=debug`)
