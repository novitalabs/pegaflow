# PegaFlow PD Connector：RDMA Push + Layer-wise（实验性）

> **⚠️ Experimental** — 与现有 `pd.md`（CPU 中转 + 异步回调）完全独立的一条新链路。
> 单独成文、单独连接器实现、单独配置入口，**不要**回写到现有 `python/pegaflow/connector/`。
> 评估通过后再考虑收编。

## 1. 背景与动机

现有 P/D（见 [`pd.md`](./pd.md)）走的是 "P save→PegaEngine→D load" 的 CPU 中转路径：

```
P GPU ──> Pinned CPU (PegaEngine) ──> D GPU
```

代价：
- KV 必须落 CPU pinned 一次，TTFT 被这一来回拖住，无法和 P 的逐层计算重叠；
- D 启动 decode 必须等到 P 全部 layer save 完毕；
- 单机回路下不亏，但跨节点链路上多了一次 D2H + H2D。

本方案要做的事是把 P→D 路径压成 **一次跨节点 RDMA WRITE**，并让传输与 P 的 prefill 计算 **逐层流水**：

```
P GPU layer_i KV ──RDMA WRITE──> D GPU layer_i KV slot
        (P 继续算 layer_{i+1})            (D 收齐就能起 attention)
```

**Baseline：vLLM 官方 NIXL connector**（`vllm/distributed/kv_transfer/kv_connector/v1/nixl/`）。
NIXL 是 D 端 READ pull 模型、按请求触发异步拉 KV；本方案是 P 端 WRITE push + layer-wise 重叠。NIXL 的代码结构值得参考，但数据面方向和时序不照搬。
具体压测条件、硬件、模型、并发等都由后续 benchmark 计划单独给出，本文不预设。

设计上要赢 NIXL 的核心抓手：
- P 的 prefill wall-time 与 RDMA 推送 wall-time **完全重叠**——RDMA 不延长 P 的关键路径；
- D 的 `start_decode` 时刻 ≤ `P_last_layer_compute_done + ε`（ε ≈ 一层 KV 的 RDMA + sync 时延），而 NIXL 必须等 P 全部 prefill 跑完才开始 pull；
- 控制面 + 数据面合计 RTT 数比 NIXL 少一跳（NIXL 是 D 通知 P → P 算完 → D pull；本方案是 D 通知 P → P 边算边 push）。

## 2. 总体架构

```
                       ┌─────────────────────────────────────┐
                       │ Router (pegaflow-router)            │
                       │  - 选 P, 选 D, 只请求 D       │
                       └────┬────────────────────────┘
                            │ HTTP
                  ┌─────────▼───────┐        ┌─────────────────┐
                  │ D vLLM          │        │ P vLLM          │
                  │  PdConnector    │        │ PdConnector     │
                  │  (role=DECODE)  │        │ (role=PREFILL)  │
                  └────┬────────────┘        └────────┬────────┘
                       │                              │
                       │      ┌─────── OOB ZMQ ──────►│
                       │      │  (prompt/QP/GID/PSN/rkey/addr/req_id)
                       │ allocate
                       │ KV blocks
                       │      │            P 算 layer_i ──┐
                       ▼      │                           │
                  ┌─────────────┐                         ▼
                  │ D GPU KV    │  ◄─── RDMA WRITE ── ┌─────────┐
                  │ (per-layer  │  ◄─── (per layer)── │ P GPU   │
                  │  slots)     │  ◄─── WRITE_IMM ─── │ KV cache│
                  └─────────────┘   last layer done   └─────────┘
                       │
                  D 起 decode
```

角色：
- **Router**：只选 P + D，并把请求转给 D；Router 不直接请求 P。
- **D 节点**：拿到请求后先在自己的 GPU 上分配 KV slot（per layer），把 prompt、P/D role 参数、(addr, rkey, layer 数, block layout, req_id) 通过 OOB 推给 P。
- **P 节点**：拿到 "do remote prefill" 指令后立即调度本地 vLLM 算 prefill；每算完一层就把这层 KV WRITE 到 D；最后一层用 `WRITE_WITH_IMM` 携带 `req_id` 作为完成信号。
- **OOB control plane**：ZMQ ROUTER/DEALER（直接复用 vLLM NIXL connector 那套，详见 §4.1）。控制平面**不**走 Router，P↔D 之间直连。

## 3. RDMA 链路接管（**核心**，pegaflow-transfer v2，verbs only）

### 3.1 当前路线：v2 常驻，verbs only，沿用 sideway/mummy 栈

`pegaflow-transfer` 现在有两条独立链路：

| 链路 | 定位 | 当前状态 |
| --- | --- | --- |
| v1 | 现有 Mooncake-style RDMA READ/WRITE batch API | 保持原 API，不作为本方案改造对象 |
| v2 | PD push / layer-wise RDMA WRITE + IMM substrate | 默认编译，已经有 CPU/GPU memory bench |

v2 **不是 v1 的替代品**。v1 的接口模型是“caller 给一批 desc，内部按 NUMA/NIC/QP 分发”；v2 的接口模型是“caller 显式提交 `TransferRequest`，并用 `DomainGroupRouting` 描述本 engine 内 domain 选择”。这对 PD push 是可接受的，因为 PD connector 本来就会按 req/layer/block 维护自己的调度状态。

底层依赖路线已经从最初的“自 vendor `libibverbs-sys`”调整为：

- `sideway = 0.4.2`，继续作为 verbs 生态入口；
- `rdma-mummy-sys = 0.2.3`，由 sideway 依赖；
- CUDA 绑定走 `cudarc`，不维护第二套 CUDA FFI；
- `tokio` 默认存在，不再通过额外 feature 开关；
- v2 默认存在，不再通过 `v2` feature 开关。

这样做的好处是 v1/v2 都尽量站在同一套 RDMA binding 上，后续 `reg_dmabuf_mr` 补齐也应该落到 sideway/mummy 这条线上。

### 3.2 设计差异：v1 vs v2

| 维度 | pegaflow-transfer v1 | pegaflow-transfer v2 |
| --- | --- | --- |
| 底层绑定 | sideway | sideway + v2 raw verbs glue，后续收敛到 sideway/mummy |
| API 粒度 | `batch_transfer_async(op, remote, descs)` | `TransferRequest::{Single, Paged, Scatter, Imm, Barrier}` |
| NIC 选择 | 内部按 NUMA page 分桶、NIC round-robin | caller 通过 `DomainGroupRouting::{Pinned, RoundRobinSharded}` 指定当前 engine 内路由 |
| 完成机制 | 每 active NIC 返回 receiver | callback / atomic counter / `ImmCounter` |
| PD push 适配 | 不适合 layer-wise push，缺 IMM 一等原语 | 第一版只用 `Single + Imm` |
| 多 NIC 聚合 | v1 内部自动分桶 | 单 engine 多 domain 可 `RoundRobinSharded`；跨 topology group 聚合由上层 bench/connector 并发多个 engine |
| GPU MR | 当前 v1 不承担 PD GPU path | 当前 h20 上 `ibv_reg_mr(cuda_ptr)` 已能跑；正式 DMA-BUF path 等 sideway/mummy 暴露 `reg_dmabuf_mr` |

### 3.3 v2 模块布局（pegaflow-transfer 内部）

```
pegaflow-transfer/
├── Cargo.toml
│   依赖:
│     sideway = "0.4.2"
│     rdma-mummy-sys (via sideway)
│     cudarc
│     anyhow / bytes / crossbeam-channel / dashmap / parking_lot / smallvec / tokio / ...
│   features:
│     default = ["cudarc/cuda-12080"]
│     cuda-13 = ["cudarc/cuda-13000"]
├── src/
│   ├── lib.rs                        # v1 root API + pub mod v2
│   ├── engine.rs                     # v1 public API
│   ├── rc_backend/                   # v1 backend
│   ├── error.rs                      # v1 error
│   ├── rdma_topo.rs                  # 现有, 两版共用 (GPU/NIC NUMA 拓扑)
│   ├── cuda_lib/                     # cudarc-backed CUDA wrappers
│   ├── cuda_sys.rs / cudart_sys.rs   # cudarc re-export
│   ├── gdrapi_sys.rs
│   ├── libibverbs_sys.rs             # 当前 v2 raw glue；后续收敛到 sideway/mummy
│   ├── v2/
│   │   ├── mod.rs                    # pub use transfer_engine::TransferEngine;
│   │   ├── api.rs                    # TransferRequest, MemoryRegionDescriptor, ...
│   │   ├── interface.rs              # public traits + mocks
│   │   ├── provider.rs               # RdmaDomain trait
│   │   ├── provider_dispatch.rs      # verbs-only dispatch
│   │   ├── transfer_engine.rs        # TransferEngine + callback worker thread
│   │   ├── fabric_engine.rs          # FabricEngine (多 GPU/Worker 聚合)
│   │   ├── worker.rs                 # per-GPU Worker
│   │   ├── domain_group.rs           # DomainGroupRouting (multi-NIC)
│   │   ├── imm_count.rs              # ImmCount 计数器
│   │   ├── rdma_op.rs                # SingleWriteOp / PagedWriteOp / ...
│   │   ├── mr.rs                     # MemoryRegion + CUDA pointer / DMA-BUF fd probe
│   │   ├── host_buffer.rs            # 控制消息用的小 host buffer
│   │   ├── topo.rs                   # v2 topology groups
│   │   └── verbs/                    # 唯一 provider
│   │       ├── mod.rs
│   │       ├── verbs_address.rs
│   │       ├── verbs_devinfo.rs
│   │       ├── verbs_domain.rs       # VerbsDomain (RdmaDomain 实现)
│   │       ├── verbs_qp.rs
│   │       └── verbs_rdma_op.rs
│   └── bin/
│       ├── cpu_bench.rs              # v1 CPU bench
│       └── cpu_bench_v2.rs           # v2 host/cuda WRITE+IMM bench
```

要点：
- **v2 常驻默认编译**：不再有 `v2` feature 开关；
- **v1 不迁移目录**：当前不做 v1/v2 目录化重构，v1 API 保持 root re-export；
- **v2 不以补齐 v1 API 为目标**：PD connector 可以直接使用 v2 primitives；
- **GPU bench 已落地**：`pegaflow-cpu-bench-v2 --memory host|cuda` 可测 host/GPU memory WRITE+IMM；
- **DMA-BUF 正式路径等待上游**：sideway #107 / rdma-mummy-sys #21 已记录需求。

### 3.4 用 v2 跑 PD push 的最小 API 表面

PD push connector 最终会用到这些 v2 primitives：

```rust
use pegaflow_transfer::v2::{
    FabricEngine, TransferEngine, TransferEngineBuilder,
    TransferRequest, SingleTransferRequest, ImmTransferRequest,
    TransferCallback, ImmCount,
    MemoryRegion, MemoryRegionDescriptor,
    DomainGroupRouting, PeerGroupHandle,
};

// 启动时
let fabric = FabricEngine::new(workers /* per-GPU */)?;
let engine = Arc::new(TransferEngine::new_with_fabric(fabric)?);

// 注册一层 KV 显存
let mr = engine.register_memory_allow_remote(layer_kv_ptr, layer_kv_bytes)?;
// mr.descriptor() -> MemoryRegionDescriptor (含 (DomainAddress, rkey) 列表), 通过 ZMQ 发给对端

// OOB 拿到对端 descriptor 后, 建 peer group
let peer = engine.add_peer_group(remote_descriptor, DomainGroupRouting::RoundRobinSharded { num_shards })?;

// P 端: 每层算完, push 一个 block
engine.submit_transfer(
    TransferRequest::Single(SingleTransferRequest {
        peer,
        src: local_mr_slice,
        dst: remote_block_addr,
        rkey: remote_rkey,
        bytes,
    }),
    TransferCallback { on_done: Box::new(|| {}), on_error: ... },
)?;

// 最后一层最后一个 block: 带 IMM
engine.submit_transfer(
    TransferRequest::Imm(ImmTransferRequest {
        peer,
        imm_data: encode_req_done(req_id),
        // 可选带 payload, 也可以纯 IMM (零字节 RDMA_WRITE_WITH_IMM)
    }),
    no_cb,
)?;

// D 端: 注册 IMM 监听
engine.register_imm_callback(|imm: u32| {
    let req_id = decode_req_done(imm);
    mark_kv_ready(req_id);
});
```

但 **下一步不直接把这些绑定进 connector**。先在 `pd_connector` connector 内定义一个很薄的 RDMA 端口：

```python
class RdmaPort(Protocol):
    def register_local_layers(self, layers: tuple[LayerRemoteLayout, ...]) -> tuple[LayerRemoteLayout, ...]: ...
    def register_remote(self, req_id: str, handshake: PdHandshake | None = None) -> None: ...
    def push_layer(self, req_id: str, layer_idx: int, blocks: list[BlockSlice]) -> None: ...
    def push_done(self, req_id: str) -> None: ...
    def wait_done(self, req_id: str) -> None: ...
    def pop_finished_sending(self) -> set[str]: ...
    def pop_finished_recving(self) -> set[str]: ...
```

第一版实现：
- `NoopRdmaPort`：只记录调用和状态，不做真实 RDMA；
- `MockRdmaPort`：单进程内把 block copy / mark-ready 模拟出来，用来测 vLLM hook 时序；
- `RealRdmaPort`：下一步接 PyO3 `PdRdmaEngine`，不再另开 sidecar 路径。

这样 connector 的角色状态机、metadata、chunk tracker、layout 先站稳；Python binding 的最小 API 从真实调用点反推，而不是先把 Rust surface 暴露一大片。

当前 Python 侧已经把 PyO3 需要的最小边界收敛成：

| Python 调用 | Rust/PyO3 对应语义 | 备注 |
| --- | --- | --- |
| `register_local_layers(layers)` | 按 layer 注册本地 KV tensor region，返回可发给对端的 MR/rkey/addr 描述 | 输入含 `base_addr`、`block_bytes`、K/V block addr；返回后续可把 `mr_desc` 填实 |
| `register_remote(req_id, handshake)` | 为某个 request 建立 peer group / remote MR view | D 端 wait 时注册自己；P 端 push 时消费 D 的 handshake |
| `push_layer(req_id, layer_idx, blocks)` | 对该 req/layer 的 block slice 提交 RDMA WRITE | `BlockSlice` 当前包含 `block_id/src_offset_bytes/bytes`，remote addr 从 handshake 的同 block layout 查 |
| `push_done(req_id)` | 提交最后一个 `WRITE_WITH_IMM` / done 信号 | 只在最后一层且所有 req block 已推完后调用 |
| `wait_done(req_id)` | D 侧等待或查询 IMM done | mock 下立即完成；真实实现应非阻塞 poll，避免卡住所有 layer |
| `pop_finished_sending/recving()` | 把 RDMA 完成结果汇总给 vLLM `get_finished` | 失败 block 后续通过 `get_block_ids_with_load_errors` 上报 |

所以 P2 需要写的 PyO3 类型不是 v2 全量 API，而是 `RealRdmaPort` 背后的一个窄类：`PdRdmaEngine`。它只需要能管理本 worker 的 v2 `TransferEngine`、MR 注册表、peer group、request → remote layout 映射和 IMM 完成队列。

建议的 PyO3 class surface：

```python
class PdRdmaEngine:
    def __init__(
        self,
        *,
        cuda_device: int = 0,
        numa_node: int | None = None,
        domains: list[str] | None = None,
        device: str = "cuda",
    ) -> None: ...

    def register_local_layers(
        self,
        layers: tuple[LayerRemoteLayout, ...],
    ) -> tuple[LayerRemoteLayout, ...]: ...

    def register_remote(
        self,
        req_id: str,
        handshake: PdHandshake,
    ) -> None: ...

    def push_layer(
        self,
        req_id: str,
        layer_idx: int,
        blocks: list[LayerBlockSlices],
    ) -> None: ...

    def push_done(self, req_id: str) -> None: ...
    def pop_finished_sending(self) -> set[str]: ...
    def pop_finished_recving(self) -> set[str]: ...
    def close_request(self, req_id: str) -> None: ...
```

Python `RealRdmaPort` 只做 dataclass ↔ PyO3 参数的薄转换，不承担拓扑、MR、QP、CQ 逻辑。Rust 侧负责：

- 根据 `cuda_device/numa_node` 选择 v2 topology group；
- `register_local_layers` 注册每层 K/V region，并把 `LayerRemoteLayout.mr_desc` 填成可 OOB 传输的描述；
- `register_remote` 反序列化 remote MR/addr/rkey，创建或复用 peer group；
- `push_layer` 把每个 `LayerBlockSlices` 展开为 K/V `TransferRequest::Single`；
- `push_done` 提交 `WRITE_WITH_IMM`；
- completion / error 通过 poll 集合回到 `get_finished` / `get_block_ids_with_load_errors`。

当前实现已经把这个窄类暴露到 `pegaflow.pegaflow.PdRdmaEngine`，并加了
`RealRdmaPort` Python adapter。`LayerRemoteLayout` 的 OOB 契约现在明确包含
`block_ids`，native 返回的 MR 描述不能丢失这组 block id ↔ K/V addr 映射；
否则 P 侧无法把 vLLM 的 logical block 写到 D 侧对应 remote offset。硬件侧第一步
用 `scripts/pd_rdma_binding_probe.py` 在 h20-100 上验证 binding 初始化、topology 选择
和 v2 engine 基本信息。

当前 h20-100 进展：
- `scripts/pd_rdma_binding_probe.py --cuda-device 2 --device cuda` 可以创建
  `PdRdmaEngine`，选中 `cuda:2 numa:0 domains=[mlx5_2]`；
- `scripts/pd_rdma_e2e.py --cuda-device 2 --block-bytes 1048576 --blocks 8`
  可以完成同进程 P/D GPU buffer RDMA WRITE+IMM，并把 D 侧目标 buffer 拷回 host 校验；
- 默认 pin 下 P/D 目前都落在同一组 CPU（cuda:2 上默认 `worker=60, uvm=62`），
  16 MiB 正确性用例约 2.5 Gbps，600 MiB release 用例约 74 Gbps，低于
  `pegaflow-cpu-bench-v2` 的底层 v2 结果；显式把 P/D pin 到不同 CPU 后当前会卡住，
  需要继续调 v2 worker progress / affinity 交互，性能不能按默认 pin 的数字验收。

### 3.5 控制面（OOB）

控制面形态参考 vLLM NIXL：scheduler/worker 之间走 vLLM 自己的 connector metadata，P/D engine 之间用 OOB 直连交换 endpoint、layout、block ids、MR 描述符和完成通知。NIXL 当前用 ZMQ ROUTER/REQ 做远端 metadata 查询；我们第一版可以先把 OOB 抽成 `OobPort`，本地测试用 in-memory/mock，真实跨进程时再接 ZMQ。

序列化格式先用 Python dataclass / msgspec / pickle 里最贴合 vLLM 测试的方案，等 `RealRdmaPort` 接入 Rust v2 后，再决定是否让 Rust 侧描述符直接以 `serde` 格式透出。详细握手时序见 §3.6。

### 3.6 连接 bring-up

每对 (P_worker, D_worker) 一对 RC QP。bring-up 时序（D 侧主动发起）：

```
D                                          P
│ 收到 router 请求, 解析 prompt           │
│ 本地 vllm allocate KV slots              │
│ engine.register_memory_allow_remote()    │
│   -> MemoryRegionDescriptor[num_layers]  │
│                                          │
│ ZMQ PUSH PdHandshake{req_id, mrs_D,      │
│                       peer_conn_D} ────► │ 收到 do_remote_prefill
│                                          │ engine.register_memory_local(P KV)
│ ◄── ZMQ ACK {peer_conn_P}                │ engine.add_peer_group(mrs_D, ...)
│                                          │
│ engine.add_peer_group(peer_conn_P)       │ engine.add_peer_group(peer_conn_D)
│ (QP 内部 INIT→RTR→RTS, fabric-lib 接管)  │ (同上)
│ 等 IMM                                    │ 开始 prefill, 逐层 submit_transfer
```

OOB 上交换的 metadata（直接复用 v2 的 `MemoryRegionDescriptor`，外面套一层）：

```rust
// pegaflow-transfer/src/v2/handshake.rs 或 pd_connector/ 里
pub struct PdHandshake {
    pub req_id: String,
    pub model_id: String,                          // 一致性校验 (TP, num_layers, head_dim)
    pub num_layers: u32,
    pub tp_rank: u32,
    pub kv_layout: KvLayout,                       // separate | merged
    pub block_bytes: u32,
    pub layers: Vec<PerLayerMr>,                   // 每层一条
    pub peer_conn: PeerConnDescriptor,             // fabric-lib 的 QP/GID/PSN 集合体, v2 自带
}

pub struct PerLayerMr {
    pub layer_idx: u32,
    pub k: MemoryRegionDescriptor,                 // v2 已有, 含 (DomainAddress, rkey) 列表
    pub v: Option<MemoryRegionDescriptor>,         // K/V 分离才有
    pub k_block_addrs: Vec<u64>,                   // §3.7 paged KV per-block 地址
    pub v_block_addrs: Option<Vec<u64>>,
}
```

一致性校验在 `add_peer_group` 之前做：`model_id` / `num_layers` / `tp_size` 不一致直接拒绝。

### 3.7 内存注册

当前 v2 已经能在 h20 上用 `ibv_reg_mr(cuda_ptr)` 跑 GPU memory WRITE+IMM bench，这足够支撑 connector skeleton 和早期端到端验证。正式生产路径仍要切到 DMA-BUF MR：

1. `cudaPointerGetAttributes` 验证是 device pointer；
2. `cuMemGetHandleForAddressRange(..., CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, ...)` 拿 fd；
3. `ibv_reg_dmabuf_mr(pd, offset, length, iova, fd, access)`。

`ibv_reg_dmabuf_mr` 需要 sideway / rdma-mummy-sys 暴露；相关 upstream issue 已记录。上游补齐前，真实 RDMA port 先只作为实验链路，不把当前 `reg_mr(cuda_ptr)` 当最终语义。

**注册粒度**：vLLM connector 会在 worker 初始化时拿到 KV cache tensor（`register_kv_caches` / `register_cross_layers_kv_cache`）。我们按实际 tensor region 注册并生成 layer/block 地址表，整生命周期复用。
- D 实例的 MR 数量 = `num_layers * 2`（K/V 分离）或 `num_layers`（合并）；
- P 端同理；
- 不引入 buffer pool / arena——**vLLM 自己管 GPU KV，我们只是借地址**。

⚠️ kernel ≥ 5.12 才支持 DMA-BUF MR。fallback 路径（旧 kernel）走 nv_peer_mem，第一版**不实现**，直接报清楚错误并让 D 本地重算。

### 3.8 传输（per-block scatter WRITE）

`save_kv_layer` hook 内的 P 端伪代码：

```rust
// 在 vllm save_kv_layer(layer_idx, slot_mapping, ...) hook 里调
fn on_layer_done(layer_idx: u32, slot_mapping: &[i64], ctx: &PdContext) {
    use pegaflow_transfer::v2::{TransferRequest, SingleTransferRequest, ImmTransferRequest, TransferCallback};

    let block_size = ctx.block_size;
    let blocks_this_step: Vec<u32> = slot_mapping.iter()
        .filter(|&&s| s >= 0)
        .map(|&s| (s as u64 / block_size as u64) as u32)
        .collect::<HashSet<_>>().into_iter().collect();    // 去重

    let is_last_layer = layer_idx == ctx.num_layers - 1;
    let is_last_chunk = ctx.req_state.is_finished_with_prefill();

    let last_idx = blocks_this_step.len() - 1;
    for (i, b) in blocks_this_step.iter().enumerate() {
        let remote = &ctx.peer_layers[layer_idx as usize];
        let put_imm = is_last_layer && is_last_chunk && i == last_idx;

        // K
        let k_req = TransferRequest::Single(SingleTransferRequest {
            peer: ctx.peer,
            src_mr: ctx.local_layer_mrs[layer_idx as usize].k_handle,
            src_off: (*b as u64) * block_size as u64 * ctx.k_block_stride,
            dst_addr: remote.k_block_addrs[*b as usize],
            rkey: remote.k.rkey_for(ctx.peer),
            bytes: ctx.k_block_bytes,
        });
        ctx.engine.submit_transfer(k_req, TransferCallback::noop())?;

        // V
        let v_req = TransferRequest::Single(/* 同 K, 用 remote.v_block_addrs[*b] */);
        ctx.engine.submit_transfer(v_req, TransferCallback::noop())?;

        if put_imm {
            let imm = encode_req_done(&ctx.req_id);
            ctx.engine.submit_transfer(
                TransferRequest::Imm(ImmTransferRequest { peer: ctx.peer, imm_data: imm }),
                TransferCallback::noop(),
            )?;
        }
    }
}
```

要点：
- 一个 block 的 K + V = 2 个 `Single`，最后一个 chunk 最后一层最后一个 block 之后追加一个 `Imm`（fabric-lib 内部映射成零字节 `IBV_WR_RDMA_WRITE_WITH_IMM`）；
- **不**等 completion 再算下一层——CUDA stream 上 vLLM 继续往下算 layer_{i+1}，host 侧 RDMA 自己推；
- signaled/unsignaled 由 fabric-lib 内部决定（默认每条 WR 都 signaled，见 `pplx-garden/fabric-lib/src/verbs/verbs_rdma_op.rs:56`），第一版**不**做手工 unsignaled 优化，等 P0 benchmark 数据出来再判断 CQE 压力是否成瓶颈；
- 多 NIC 在 `add_peer_group(... DomainGroupRouting::RoundRobinSharded { num_shards })` 里指定，**v2 内部自动分流**，调用方不用感知 NIC 选择。

### 3.9 D 端如何知道单层 KV "到了"

第一版**不需要**逐层感知。原因：
- D 在 D 上算 attention 是从 layer 0 开始的，layer 0 完成最早（P 第一个算完的也是 layer 0）；
- P 推送是顺序的（layer 0 → N-1），网络保序在单 QP RC 上有保证；
- 跨 QP（多 NIC）会乱序，但同一层的 K/V 仍在同一对 QP 上，跨层乱序对 D 也无所谓（D 只在最后一层 IMM 到达时才开 decode）。

D 在收到最后一层 IMM 后才放行 decode（即在 `wait_for_layer_load` 里阻塞所有 layer 直到 IMM 到达）。

> **后续优化**：让 D 在 layer_i 计算前才阻塞等 layer_i 到达，实现真正的 D 侧 layer-wise 流水。需要 P 每层都打 WITH_IMM 携带 `(req_id_hash, layer_idx)`（fabric-lib `ImmCount` 已支持），第一版不做。

### 3.10 Chunked prefill：第一版就支持

#### 为什么必须支持

vLLM 默认开启 chunked prefill（`SchedulerConfig.enable_chunked_prefill`，见 `vllm/engine/arg_utils.py:593, 2318`），prompt 会被切成多个 forward step 喂给 P。关掉的话单次 forward 激活随 prompt 长度线性涨（FlashAttention 干掉了 attention 的 `O(N²)`，但 QKV / MLP 中间张量的 `O(N)` 项跑不掉），长 prompt 直接顶到 OOM 边缘。

第一版直接支持，理由：

1. **paged KV 本来就强制 per-block 推送**。vLLM 的 KV 存在 `kv_cache[layer][block_idx, slot_in_block, head, dim]`，一个 prompt 的 token 散在 `block_table` 指向的多个 block 里，**这些 block 在显存上并不连续**。即使关掉 chunked prefill，一层 KV 也要做 `len(block_table)` 个 WRITE（或 scatter batch）。**"一层一次 WRITE"的假设从来就不成立**。
2. 一旦做了 per-block scatter，"分多次推" vs "一次推完" 的边际复杂度很小：只是 P 端多记一个"本 req 已推过的 chunk 数"的状态。
3. NIXL baseline 默认开 chunked prefill，关了再比赛会被诟病"限制了 prompt 长度"。
4. 直接对齐 vLLM 默认行为，用户无需调启动参数。

#### Push 语义与 layer-wise 的叠加

```
对每个 forward step k:
  对每层 i = 0..L-1:
    on save_kv_layer(name=layer_i, kv, attn_metadata):
      blocks_this_step = unique(attn_metadata.slot_mapping // block_size)
      for b in blocks_this_step:
        remote = peer_mrs[layer_i].block_addr(b)   # 同 logical block 在 D 端的位置
        submit_write(K_slice(b), remote.k, ...)
        submit_write(V_slice(b), remote.v, ...)

      若 i == L-1 且 scheduler 标记本 step 是该 req 的最后一个 chunk:
        最后一个 WRITE 用 WRITE_WITH_IMM, imm = encode_req_done(req_id)
```

要点：
- **寻址来源**：本 step 写到哪些 block，来自 `attn_metadata.slot_mapping`（vLLM forward 时已经填好）。logical block → D 端 remote addr 的映射，握手时一次性传过来（§3.3 `RemoteMrPerLayer` 增加 `block_addrs: Vec<u64>`）。
- **完成信号**：只在"最后一个 chunk 的最后一层"打 IMM。最后一个 chunk 的判定来自 `SchedulerOutput`，参考 `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:399-451` 的累积逻辑（用 `num_scheduled_tokens` + request 的 `num_prompt_tokens` 比对）。
- **保序**：跨 chunk 跨层的 WRITE 仍走同一对 RC QP（同 NIC 路径），网络层保序；多 NIC 时仍按 layer 分流（同层 K/V 同 NIC，跨层乱序对 D 不影响，因为 D 只在 IMM 到达后才放行 decode）。
- **每 req P 端状态**：`chunks_pushed: u32`、`is_finished: bool`。一个 hashmap 足够。

#### 握手 metadata 调整

`RemoteMrPerLayer` 第一版按 prompt 长度上限拍一组 block 地址数组：

```rust
pub struct RemoteMrPerLayer {
    pub layer_idx: u32,
    pub rkey: u32,
    pub bytes_per_block: u32,
    pub k_block_addrs: Vec<u64>,  // 长度 = D 为本 req 预分配的 block 数
    pub v_block_addrs: Vec<u64>,  // 同上（K/V 分离时）
}
```

D 端 `update_state_after_alloc` 拿到 `blocks` 后填这两个数组发给 P。P 端按 logical block index 直接查表。

#### 真正的边界

- **D 必须一次性把 KV slot 全分配好**（即使 P 还没推第一个 chunk）。这要求 D 端在 `update_state_after_alloc` 阶段就知道 prompt 长度，vLLM 直接给。
- **P 端 preemption**：若 P 的 scheduler 把请求 preempt 掉（unlikely，但 vLLM 会），已推的 KV 在 D 上仍是合法的，但 P 重新调度后会**重写**已经推过的 block。这会导致 D 端被多次写同一位置，**幂等**所以 OK，只是浪费带宽。第一版接受这个浪费，不做 preempt-aware 优化。
- **chunked prefill + prefix caching**：若 P 端开 prefix cache，前缀 block 不会重算也不会出现在 `slot_mapping` 里。这时 P 不会推那些 block，**D 端拿不到完整 KV**。第一版要求 **P 端关闭 prefix caching**（`--no-enable-prefix-caching`），把这个变量挡掉。V2 再处理（D 端能告诉 P "这些 prefix 我已经从别处来"，或 P 显式补推 prefix block）。

## 4. vLLM 接入（不 fork vllm）

### 4.0 NIXL connector 调研结论

vLLM NIXL connector 的可参考部分主要是模块边界，而不是传输方向：

```
vllm/distributed/kv_transfer/kv_connector/v1/nixl/
├── connector.py    # KVConnectorBase_V1 facade，只按 role 分发
├── scheduler.py    # request bookkeeping / KVTransferParams / connector metadata
├── worker.py       # register_kv_caches / handshake / async xfer / get_finished
├── metadata.py     # handshake payload, request metadata, compatibility hash
├── tp_mapping.py   # hetero TP / rank mapping
├── stats.py
└── utils.py        # ZMQ helper, device support matrix
```

对 `pd_connector` 的落地建议：

- `connector.py` 保持薄，只负责继承 `KVConnectorBase_V1`、按 `KVConnectorRole` 创建 scheduler/worker，并转发 vLLM hook；
- `scheduler.py` 管请求状态：识别 `kv_transfer_params`、返回 external token 数、在 `update_state_after_alloc` 后记录 D 端 block ids、在 `build_connector_meta` 里把本 step 需要 worker 处理的请求打包；
- `worker.py` 管执行：`register_kv_caches` 生成 layout，`start_load_kv` 在 D 端注册等待，`save_kv_layer` 在 P 端根据 `attn_metadata.slot_mapping` 推 block，`get_finished` 上报 done/failed；
- `metadata.py` 只放 dataclass：`PdConnectorMetadata`、`PdHandshake`、`ReqMeta`、`RemoteMeta`、compatibility hash；
- `rdma.py` 暂时只定义 `RdmaPort` + `NoopRdmaPort` / `MockRdmaPort`，不要把 Rust v2 绑定提前塞进 connector；
- `layout.py` 专门处理 KV cache layout、block id、slot mapping 到 block slice 的转换，避免这些逻辑散在 hook 里。

NIXL 的关键 hook 行为对照：

| hook | NIXL 行为 | `pd_connector` 第一版 |
| --- | --- | --- |
| `get_required_kvcache_layout` | 非 MLA 时偏好 HND | 先沿用 HND 偏好，减少 layout 变量 |
| `get_num_new_matched_tokens` | D 侧看到 `do_remote_prefill` / `do_remote_decode` 后返回可异步拉取 token 数 | D 侧看到 `do_remote_prefill` 后返回 prompt token 数并声明 async |
| `update_state_after_alloc` | 记录 req 和本地 block ids，等待 worker 发起 READ | 记录 D 端 block ids，准备 OOB handshake / mock register |
| `build_connector_meta` | 把 `reqs_to_recv`、`reqs_to_send`、heartbeat 打给 worker | 把 `reqs_to_wait` / `reqs_to_push` / done tracking 打给 worker |
| `register_kv_caches` | 注册 tensor region，计算 per-layer/per-block transfer metadata | 生成 Python layout，真实 RDMA 接入后注册 MR |
| `start_load_kv` | worker 发起非阻塞 READ，并在后续 step poll | D worker 注册等待 done；mock 下可直接状态机验证 |
| `save_kv_layer` | NIXL 不做 layer-wise save | P worker 的核心路径：按 layer/block 调 `RdmaPort.push_layer` |
| `get_finished` | 汇总 async send/recv 完成和失败 block | 汇总 P push 完成、D wait done、失败 block |
| `request_finished` | 返回给下游的 `kv_transfer_params`，并可延迟 free block | P/D 按 req 生命周期释放 metadata/MR 引用 |

这意味着下一步不是在旧 connector 基础上改，而是新建 `pegaflow.pd_connector`，按 NIXL 的文件组织和 hook 分层新写。

### 4.1 角色识别

通过 `--kv-transfer-config` 的 `kv_role`：

```bash
# P 节点
vllm serve $MODEL \
  --no-enable-prefix-caching \               # ← 见 §3.7，第一版禁用 P 端 prefix cache
  --kv-transfer-config '{
    "kv_connector": "PdConnector",
    "kv_connector_module_path": "pegaflow.pd_connector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
      "oob_listen": "0.0.0.0:7100",
      "rdma_devices": ["mlx5_0","mlx5_1"]
    }
  }'

# D 节点
vllm serve $MODEL \
  --kv-transfer-config '{
    "kv_connector": "PdConnector",
    "kv_connector_module_path": "pegaflow.pd_connector",
    "kv_role": "kv_consumer",
    "kv_connector_extra_config": {
      "oob_listen": "0.0.0.0:7100",
      "rdma_devices": ["mlx5_0","mlx5_1"]
    }
  }'
```

### 4.2 Hook 映射

| vLLM hook (`KVConnectorBase_V1`) | P (producer) | D (consumer) |
| --- | --- | --- |
| `get_num_new_matched_tokens` | 返回 `(0, False)` | 若请求带 `do_remote_prefill=True`，返回 `(prompt_len, True)`（异步） |
| `update_state_after_alloc` | 无操作 | 记下 block_ids，触发 `_export_mrs_and_handshake(req)` |
| `build_connector_meta` | 把 "do_remote_prefill" 请求的 `(peer_ep, req_id, mrs_D)` 打包给 worker | 把 "正在等的 req_id" 打包给 worker |
| `start_load_kv` | 无操作 | 无操作（不主动 load，被动等 WRITE） |
| `wait_for_layer_load(name)` | 不调用 | **第一版**：调用 `RdmaPort.wait_done(req_id)`；真实实现应 poll/短等，未 ready 时让第 0 层自然阻塞 |
| `save_kv_layer(name, kv, ...)` | **核心**：见 §3.7。读 `attn_metadata.slot_mapping` 算出本 step 写到的 block 集合，per-block `pd_rdma.submit_write` 推。最后一个 chunk 的最后一层带 IMM | 不调用 |
| `wait_for_save` | 不阻塞（fire-and-forget，IMM 是终点） | 不调用 |
| `get_finished` | 返回 P 已 WRITE 完成（CQ 收到错误）的 req_id（错误集） | 返回收到 IMM 的 req_id |
| `request_finished` | 释放本地 MR 引用 | 释放本地 MR 引用 |

请求携带的 `kv_transfer_params`（参考 NIXL）：

```python
{
  "do_remote_prefill": True,                  # 由 Router 注入到 D 的请求
  "remote_engine_id": "p-node-3:7100",        # P 的 OOB endpoint
  "tp_size": 4,                                # 一致性校验
  "req_id": "<uuid>",                          # 全链路唯一
}
```

### 4.3 Router 改动

`pegaflow-router.rs` 第一版改造：
1. 选 P + 选 D；
2. 给 D 的请求 body 注入 `kv_transfer_params = {do_remote_prefill: true, remote_engine_id: <P 的 OOB>, req_id: ...}`；
3. 只把请求转给 D；
4. D allocate KV blocks 后，通过 OOB 把 prompt + `{do_remote_prefill_sender: true, target_engine_id: <D 的 OOB>, req_id: ...}` + remote layout 发给 P；
5. P 收到 OOB 后开始 prefill 并 RDMA WRITE 到 D；
6. 等 D 的响应作为最终响应返回。

> 注意 P 不返回 token，只做 prefill。这要求 P 端在 `kv_role=producer` 时把 logits 丢弃。第一版直接复用 vLLM NIXL 的 producer-side 早退实现（NIXL connector 已经在 P 端做过这个）。

D→P OOB 现在在 skeleton 里对应 `PdPrefillRequest`：

```python
@dataclass(frozen=True)
class PdPrefillRequest:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    producer_kv_transfer_params: dict[str, Any]
    handshake: PdHandshake
```

D worker 在 `start_load_kv` 里发布这条消息；P 侧后续真实接入时用它向本地 vLLM producer 注入 prefill 请求。

## 5. 代码落点

```
pegaflow/
├── pegaflow-transfer/                 # 现有 crate
│   └── src/
│       ├── v2/                        # 已有 RDMA substrate
│       └── bin/cpu_bench_v2.rs        # host/cuda WRITE+IMM bench
├── python/
│   └── pegaflow/
│       └── pd_connector/              # ★ P1 skeleton
│           ├── __init__.py            # PdConnector facade
│           ├── scheduler.py           # scheduler-side hook state
│           ├── worker.py              # worker-side hook execution
│           ├── proxy.py               # 本地 P/D E2E 调试代理
│           ├── rdma.py                # RdmaPort protocol + Noop/Mock implementation
│           ├── oob.py                 # OOB message shape；先可 mock，不强依赖真实 ZMQ
│           ├── metadata.py            # ConnectorMetadata / PdHandshake Python 镜像
│           ├── layout.py              # slot_mapping -> block_idx / block slice
│           └── chunk_tracker.py       # 每 req: scheduled/pushed/done
└── docs/
    ├── pd.md                          # 现状（CPU 中转）
    └── pd-rdma-push.md                # 本文
```

约束：
- **`pegaflow-transfer` v1 不动**：v2 不是替代 v1；
- **先不写 PyO3**：Python binding 等 `pd_connector` 模块调用面稳定后再收敛；
- **RDMA 先 mock/blank**：connector 先验证 vLLM hook、metadata、chunked prefill 状态机；
- **Python 侧 connector 走全新 module path `pegaflow.pd_connector`**，不挤进现有 `pegaflow.connector`；
- **Router 暂不动**：先本地构造 `kv_transfer_params` 做 connector-level 验证。

### 5.1 本地 P/D proxy 调试

当前先加一个 Python proxy，不接真实 router：

```bash
mkdir -p /tmp/pegaflow-pd-logs

PYTHONPATH=$PWD/python CUDA_VISIBLE_DEVICES=0 vllm serve /data/Qwen3-4B \
  --host 127.0.0.1 --port 8001 \
  --kv-transfer-config '{
    "kv_connector": "PdConnector",
    "kv_connector_module_path": "pegaflow.pd_connector",
    "kv_role": "kv_producer",
    "engine_id": "prefill"
  }' \
  > /tmp/pegaflow-pd-logs/p.log 2>&1

PYTHONPATH=$PWD/python CUDA_VISIBLE_DEVICES=1 vllm serve /data/Qwen3-4B \
  --host 127.0.0.1 --port 8002 \
  --kv-transfer-config '{
    "kv_connector": "PdConnector",
    "kv_connector_module_path": "pegaflow.pd_connector",
    "kv_role": "kv_consumer",
    "engine_id": "decode"
  }' \
  > /tmp/pegaflow-pd-logs/d.log 2>&1

cd python
uv run python -m pegaflow.pd_connector.proxy \
  --listen-host 127.0.0.1 \
  --listen-port 8100 \
  --prefill-url http://127.0.0.1:8001 \
  --decode-url http://127.0.0.1:8002 \
  --done-endpoint tcp://127.0.0.1:7200 \
  --log-file /tmp/pegaflow-pd-logs/proxy.log
```

请求只打 proxy：

```bash
curl -s http://127.0.0.1:8100/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/data/Qwen3-4B",
    "prompt": "Write a short note about RDMA.",
    "max_tokens": 16,
    "temperature": 0
  }'
```

这条路径的语义：
1. proxy 收到请求后生成同一个逻辑 request id；
2. proxy 只向 D 发 decode 请求，注入 `do_remote_prefill=true`、`prefill_url` 和 `done_endpoint`；
3. D 的 connector allocate KV blocks 后进入 async load，并用 proxy 给的 P hint 触发 P prefill；
4. P 的 connector 在最后一层覆盖完目标 blocks 后，后台 ZMQ sender fire-and-forget 发 done；
5. D 的后台 ZMQ receiver 收到 done 后在 `get_finished` 里上报 `finished_recving`，vLLM 才继续 decode。

验收先看三份日志：
- `proxy.log`：应出现 `request=... -> D`、`D completed`，不应出现 `request=... -> P`；
- `p.log`：应出现 `P queued async push`、`P finished fake RDMA push`、`P sent fake RDMA done`；
- `d.log`：应出现 `D queued async wait`、`D -> P prefill request`、`D -> P prefill completed`、`fake RDMA done receiver listening`、`D received fake RDMA done`。

### 5.2 关于 pegainfer 的统一

pegainfer 那边 `pegainfer-comm/crates/pegainfer-comm-fabric-lib` 是 pplx-garden 的另一份 vendor。pegaflow v2 稳定后，pegainfer 应当：
1. 删掉 `pegainfer-comm-fabric-lib/`、`pegainfer-comm-libibverbs-sys/`；
2. 改 `pegainfer-comm` 让 `EpAllToAll` 的 `hw-rdma` 实现底层换成 `pegaflow_transfer::v2::TransferEngine`；
3. 通过 path dep 或私有 registry 拉 pegaflow-transfer。

这一步不在本文档覆盖范围内（pegainfer 仓库独立），但 v2 的 API 设计要预留这个对齐：
- `TransferRequest` enum 保留 `Scatter`（pegainfer EP all-to-all 用得到），即使 PD 不用；
- `Worker` / `FabricEngine` 的多 GPU 抽象不削；
- 不在 v2 公开 API 里塞 PD 专用概念（PD 专用的东西全部放 `pd_connector` 里）。

## 6. 实施阶段

| 阶段 | 内容 | 验收 |
| --- | --- | --- |
| **P0** v2 RDMA substrate | v2 默认编译；host/cuda memory WRITE+IMM bench | 已完成：h20 上 GPU2+mlx5_2，600 MiB/task，约 372.6 Gbps |
| **P1** `pd_connector` connector skeleton | 新增 `pegaflow.pd_connector`；实现 vLLM connector class、Producer/Consumer 状态机、metadata、layout、chunk tracker；RDMA 用 `NoopRdmaPort`/`MockRdmaPort` | 不接真实 RDMA，也能跑通 hook 时序单测：D allocate 后等待，P save layer 后记录 push，最后 done 能释放请求 |
| **P2** 明确 Python/Rust binding surface | 根据 P1 的 `RdmaPort` 调用点，整理最小 Rust API；第一版走 PyO3 `PdRdmaEngine` | 已暴露 `PdRdmaEngine` + `RealRdmaPort`，本地编译和契约单测通过；待 h20-100 probe |
| **P3** 接真实 v2 RDMA | 把 `RealRdmaPort` 接到 `pegaflow-transfer::v2`；先单机 1P+1D、单 GPU、手动注入 `kv_transfer_params` | GPU buffer WRITE+IMM 代替 mock，consumer 收到 done |
| **P4** vLLM 小模型 E2E | 单机 P + D 各占 1 GPU，跑小模型 | 输出 token-level 与 baseline 一致 |
| **P5** Router + 跨节点 benchmark | Router 只转发 D；D 通过 OOB 唤起 P 并携带 prompt/layout；对照 NIXL 同条件压测 | 压测细节后续给出；目标见 §1 设计抓手 |

当前 P1 已经有 skeleton：`pegaflow.pd_connector`、FlashAttention HND layout assert、mock/noop RDMA port、D handshake 发布、P 按 layer/block push、最后一层所有 block 推完后 done。P2 已经开始落地：`PdRdmaEngine` 负责 v2 `TransferEngine`、MR 注册表、remote layout 映射和 IMM 完成队列，`RealRdmaPort` 只做 dataclass ↔ native dict 转换。下一步是在 h20-100 上跑 binding probe，再把 vLLM worker 默认端口从 `NoopRdmaPort` 切到 `RealRdmaPort(PdRdmaEngine(...))`。

## 7. 开放问题（写出来，先不解决）

1. **DMA-BUF kernel 版本**：H800 集群 kernel 是否都 ≥ 5.12？要不要兼容 nv_peer_mem 老路径？
   → 询问 Quiin（QA / 集群侧）。
2. **TP > 1 的握手广播**：D 有 N 个 worker，P 也有 N 个 worker，是 N×N 全互联 RC QP，还是 N 对一一对应？
   → 一一对应（同 rank ↔ 同 rank），不跨 rank 传 KV。
3. **错误恢复**：RDMA WRITE 中途 QP 进入 ERR 状态怎么办？第一版直接把请求标记失败、走 fallback 让 D 本地重算 prefill？还是上层重试？
   → 第一版：标记失败 + D 本地重算，不做 RDMA 层重试。
4. **流控 / 背压**：高并发场景下 P 同时给一个 D 推多个请求的 KV，D 端 KV slot 是否会被吃满？
   → 由 D 的 vLLM scheduler 做 admission control（D allocate 失败就拒绝），P 端不需要单独限流。
5. **跨 instance（不同 TP 拓扑）**：暂不支持。`model_id` 一致性校验直接挡掉。
6. **Prefix caching 与 PD push 的合流**：第一版要求 P 端关 prefix cache（详见 §3.7 "真正的边界"）。这是个真实的功能损失，但绕开了"D 端拿不到 prefix block KV"的语义陷阱。V2 应该让 D 在握手时告诉 P "这些 prefix 块我已经从别处获取"，P 跳过这些 block 的推送；或者 P 显式把命中 prefix cache 的 block 也补推一遍。
7. **TP rank 间通信**：当前设计每个 (P_rank, D_rank) 一对 QP，rank-i ↔ rank-i 同号对应。P 端 TP rank 间是否需要同步 "本 chunk 推送完成"？理论上不需要（每 rank 独立推自己那份 KV，IMM 也独立到 D 对应 rank），但要在实现里验证一下 vLLM 的 forward 同步语义不会被破坏。
8. **Chunked prefill 边界**：当前 skeleton 按 `slot_mapping` 分 chunk 推送，并等目标 block 全覆盖后才 done；它假设 D 端已一次性分配完整 prompt blocks。prefix cache、preemption 重写、P/D block 映射不一致先不优化。

## 8. 不做的事

- 不做 RDMA READ（pull）模式。NIXL 已经覆盖这块，本方案专注 push。
- 不做 SSD / CPU 中转回退。失败就回到 D 本地 prefill。
- 不做 disagg 之外的功能复用（比如 prefix cache、跨节点 prefix 共享），那是 `pegaflow-core` 现有路径的事。
- 不做 IBGDA / GPU-initiated RDMA。Host-initiated + GDR polling 够用，IBGDA 上 H800 也不一定有硬件支持。
- 不做 PyO3 binding 之外的语言接入。

## 9. 参考实现指针

v2 是 `pplx-garden/fabric-lib` 的 vendor（删 efa）。下表路径用上游 pplx-garden 行号方便对照原作意图，搬进 `pegaflow-transfer/src/v2/` 后行号会浮动但文件名一致。同一份代码在 `pegainfer/pegainfer-comm/crates/pegainfer-comm-fabric-lib/` 也能查到（pegainfer 后续会移除该路径，统一用 pegaflow v2）。

| 关心的事 | 看哪 |
| --- | --- |
| RC QP bring-up 完整序列 | `pplx-garden/fabric-lib/src/verbs/verbs_domain.rs:501` (`connect_peer`) |
| DMA-BUF MR 注册 | `pplx-garden/fabric-lib/src/mr.rs:24` |
| WRITE / WRITE_WITH_IMM 提交链路 | `pplx-garden/fabric-lib/src/rdma_op.rs:8` + `transfer_engine.rs:172` |
| CQ poll + callback 异步派发 | `pplx-garden/fabric-lib/src/transfer_engine.rs:414` |
| 多 NIC round-robin | `pplx-garden/fabric-lib/src/api.rs` (`DomainGroupRouting::RoundRobinSharded`) |
| `TransferRequest` enum 五形态 | `pplx-garden/fabric-lib/src/api.rs:176` |
| `ImmCount` 计数器 | `pplx-garden/fabric-lib/src/imm_count.rs` |
| 上游 README + 设计动机 | `pplx-garden/README.md`, 论文 arXiv:2510.27656 |
| OOB ZMQ handshake 格式 | `vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py:40` |
| `KVConnectorBase_V1` 完整 hook 表 | `vllm/distributed/kv_transfer/kv_connector/v1/base.py:171` |
| layer-wise hook 时序 | `vllm/v1/worker/gpu/kv_connector.py:62` (`pre_forward`/`post_forward`) |
| chunked prefill chunk 累积参考 | `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:399-451` |
| RDMA WRITE + 显式 DONE 信号工程实践 | `Mooncake/mooncake-wheel/mooncake/mooncake_connector_v1.py:639` |
