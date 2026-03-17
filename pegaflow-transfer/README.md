# PegaFlow Transfer

RDMA-based inter-node memory transfer engine for PegaFlow. Provides one-sided RDMA READ/WRITE operations between nodes with automatic session management, NUMA-aware topology detection, and a Mooncake-compatible API.

## Overview

PegaFlow Transfer enables high-bandwidth, low-latency memory transfers between nodes over InfiniBand/RoCE RDMA fabrics. It handles the full RDMA lifecycle — device discovery, queue pair management, connection establishment, and memory registration — behind a simple synchronous API.

Key capabilities:

- **One-sided RDMA READ/WRITE**: Transfer memory directly between nodes without involving the remote CPU
- **Automatic session management**: RC (Reliable Connection) queue pairs are established on-demand on first transfer to a peer
- **Batch operations**: Transfer multiple discontiguous memory regions in a single call with pipelined WR chains
- **NUMA-aware topology**: Detect GPUs, RDMA NICs, and CPUs grouped by NUMA node for optimal NIC selection
- **Mooncake-compatible API**: Drop-in replacement for Mooncake transfer engine in vLLM/SGLang P/D disaggregation

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  MooncakeTransferEngine (public API)                                │
│                                                                     │
│  initialize() ─► register_memory() ─► batch_transfer_sync_read()   │
│                                                                     │
├──────────────────────────────────────────────────────────────────────┤
│  SidewayBackend (internal)                                          │
│                                                                     │
│  ┌─────────────────────┐     ┌──────────────────────────────────┐  │
│  │  Control Plane (UD)  │     │  Data Plane (RC)                 │  │
│  │                      │     │                                   │  │
│  │  UD QP (send/recv)   │     │  Per-peer RC QP sessions         │  │
│  │  LZ4+bincode msgs    │     │  RDMA READ / WRITE               │  │
│  │  ConnectReq/Resp     │     │  Pipelined WR chains             │  │
│  │  Address Handle cache│     │  Dedicated session worker threads │  │
│  └──────────┬───────────┘     └──────────┬───────────────────────┘  │
│             │                             │                          │
│             ▼                             ▼                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  SidewayRuntime                                              │   │
│  │                                                               │   │
│  │  sideway (ibverbs)  ·  Protection Domain  ·  Memory Regions  │   │
│  │  Device Context     ·  GID/LID/Port       ·  CQ polling      │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

### Control Plane

Connection setup uses **Unreliable Datagram (UD)** queue pairs for out-of-band signaling:

1. Initiator sends a `ConnectReq` with its RC endpoint (GID, LID, QPN, PSN) to the peer's UD QP
2. Responder creates a local RC QP, transitions it to RTS, and replies with `ConnectResp` carrying its RC endpoint and registered memory regions (base ptr, length, rkey)
3. Initiator connects its RC QP and caches the remote memory map

Control messages are serialized with bincode, compressed with LZ4, and fit within a single UD packet (4 KB).

### Data Plane

Data transfers use **Reliable Connection (RC)** queue pairs with one-sided RDMA verbs:

- **RDMA WRITE** (`IBV_WR_RDMA_WRITE`): Push local memory to remote
- **RDMA READ** (`IBV_WR_RDMA_READ`): Pull remote memory to local
- Each session has a dedicated worker thread that posts WR chains (up to 4 WRs chained) and polls a private send CQ
- Up to 96 inflight operations per session for maximum PCIe/fabric utilization

### DomainAddress

A 26-byte peer identifier encoding the RDMA endpoint:

| Field  | Bytes | Description                          |
|--------|-------|--------------------------------------|
| GID    | 16    | Global Identifier (IPv6 or IB GID)   |
| LID    | 2     | Local Identifier (IB subnet)         |
| QP Num | 4     | UD Queue Pair number                 |
| QKey   | 4     | Queue Key (`0x11111111`)             |

## Building

```bash
# Build the library
cargo build -p pegaflow-transfer

# Build release
cargo build -p pegaflow-transfer --release

# Run tests
cargo test -p pegaflow-transfer
```

## Public API

### MooncakeTransferEngine

The main entry point. Thread-safe — all methods take `&self`.

```rust
use pegaflow_transfer::{MooncakeTransferEngine, DomainAddress};

// Create and initialize
let mut engine = MooncakeTransferEngine::new();
engine.initialize("mlx5_0", 50055)?;

// Register memory for RDMA access
engine.register_memory(buffer_ptr, buffer_len)?;

// Get local session ID (share with peer out-of-band)
let session_id: DomainAddress = engine.get_session_id();

// Transfer data to/from a peer
let peer_session: DomainAddress = /* received from peer */;
engine.transfer_sync_write(&peer_session, local_ptr, remote_ptr, len)?;
engine.transfer_sync_read(&peer_session, local_ptr, remote_ptr, len)?;

// Batch transfer (multiple discontiguous regions)
engine.batch_transfer_sync_read(
    &peer_session,
    &local_ptrs,   // Vec<u64>
    &remote_ptrs,  // Vec<u64>
    &lens,         // Vec<usize>
)?;

// Cleanup
engine.unregister_memory(buffer_ptr)?;
```

### rdma_topo — System Topology Detection

NUMA-aware detection of GPUs, RDMA NICs, and CPUs via sysfs and nvidia-smi.

```rust
use pegaflow_transfer::rdma_topo::SystemTopology;

let topo = SystemTopology::detect();
topo.log_summary();

// Find NICs co-located with GPU 0 on the same NUMA node
let nics = topo.nics_for_gpu(0);
for nic in nics {
    println!("{} at {} (NUMA {})", nic.name, nic.pci_addr, nic.numa_node);
}
```

## CLI Tools

### pegaflow_topo_cli

Display the full GPU + RDMA NIC + CPU topology grouped by NUMA node, including PCIe hierarchy.

```bash
cargo run --bin pegaflow_topo_cli
```

Example output:

```
=== PegaFlow System Topology ===

NUMA 0:
  GPUs: 0, 1, 2, 3
  NICs: mlx5_0 (0000:19:00.0), mlx5_1 (0000:1a:00.0)
  CPUs: 0-47
  PCIe:
    [0000:00:01.0] GPU 0 (0000:19:00.0), mlx5_0 (0000:19:00.0)

NUMA 1:
  GPUs: 4, 5, 6, 7
  NICs: mlx5_2 (0000:98:00.0), mlx5_3 (0000:99:00.0)
  CPUs: 48-95
```

### pegaflow_cpu_bench

RDMA CPU-memory benchmark measuring per-task latency for block-based transfers. Models realistic KV cache transfer workloads with configurable block sizes and batch counts.

```bash
# Default: 50 tasks, 150 blocks x 4MB, both read and write
cargo run --release --bin pegaflow_cpu_bench

# Custom block size and range
cargo run --release --bin pegaflow_cpu_bench -- \
    --block-size 2mb \
    --blocks-per-task 100-200 \
    --tasks 100 \
    --warmup-tasks 10

# Single NIC, write-only
cargo run --release --bin pegaflow_cpu_bench -- \
    --nic mlx5_0 \
    --mode write

# Filter by NUMA node
cargo run --release --bin pegaflow_cpu_bench -- --numa 0
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--block-size` | `4mb` | Block size (e.g. `2mb`, `4mb`) |
| `--blocks-per-task` | `150` | Blocks per task: fixed (`150`) or range (`100-200`) |
| `--tasks` | `50` | Number of measured tasks |
| `--warmup-tasks` | `5` | Warmup tasks (not measured) |
| `--mode` | `both` | `read`, `write`, or `both` |
| `--base-port` | `56100` | Base RPC port (each engine pair uses 2 consecutive ports) |
| `--nic` | all | Restrict to a single NIC (e.g. `mlx5_0`) |
| `--exclude-nic` | none | Exclude a NIC |
| `--numa` | all | Restrict to a NUMA node |

The benchmark runs single-NIC baselines per NIC, then a multi-NIC aggregate test (blocks distributed round-robin) when multiple NICs are available on the same NUMA node.

## Integration Tests

The integration test requires RDMA hardware and a CUDA GPU:

```bash
# Set the NIC to use
export PEGAFLOW_TRANSFER_IT_NIC=mlx5_0

# Optional configuration
export PEGAFLOW_TRANSFER_IT_BASE_PORT=56050   # default: 56050
export PEGAFLOW_TRANSFER_IT_BYTES=1073741824  # default: 1 GB
export PEGAFLOW_TRANSFER_IT_WARMUP=1          # default: 1
export PEGAFLOW_TRANSFER_IT_ITERS=20          # default: 20

# Run (ignored by default — requires hardware)
cargo test -p pegaflow-transfer -- --ignored
```

The test allocates GPU memory via CUDA, performs RDMA WRITE from one engine to another over GPU buffers, and verifies data integrity byte-by-byte.

## Module Structure

| Module | Visibility | Description |
|--------|-----------|-------------|
| `engine.rs` | pub | `MooncakeTransferEngine` — public API facade |
| `sideway_backend.rs` | internal | RDMA backend: UD/RC QP lifecycle, session management, batch RDMA ops |
| `control_protocol.rs` | internal | Wire protocol: `ConnectReq`/`ConnectResp` messages, bincode+LZ4 codec |
| `domain_address.rs` | pub | `DomainAddress` — 26-byte RDMA peer identifier |
| `rdma_topo.rs` | pub | `SystemTopology`, `GpuInfo`, `RdmaNicInfo`, `NumaGroup` — NUMA topology detection |
| `error.rs` | pub | `TransferError` enum and `Result` type alias |
| `api.rs` | internal | `WorkerConfig` (NIC name + RPC port) |
| `logging.rs` | internal | `logforth`-based logging initialization (respects `RUST_LOG`) |

## Integration with PegaFlow

This crate is consumed by the PegaFlow Python bindings (`python/src/transfer.rs`), which expose it as `pegaflow.TransferEngine` for use in vLLM/SGLang P/D disaggregated serving. The typical flow:

1. Each vLLM worker creates a `TransferEngine` and initializes it with the NUMA-local NIC
2. Workers register their GPU KV cache memory regions
3. Workers exchange `DomainAddress` session IDs via the PegaFlow gRPC server
4. Prefill workers RDMA WRITE KV cache blocks to decode workers
5. Decode workers RDMA READ KV cache blocks from prefill workers

## Environment Variables

- `RUST_LOG`: Control logging level (default: `info,pegaflow_transfer=debug`)

## License

Part of the PegaFlow project. Apache-2.0.
