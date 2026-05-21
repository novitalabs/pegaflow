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
NIXL 是 D 端 RDMA READ pull 模型、按整请求粒度拉 KV；本方案是 P 端 push + layer-wise 重叠。
具体压测条件、硬件、模型、并发等都由后续 benchmark 计划单独给出，本文不预设。

设计上要赢 NIXL 的核心抓手：
- P 的 prefill wall-time 与 RDMA 推送 wall-time **完全重叠**——RDMA 不延长 P 的关键路径；
- D 的 `start_decode` 时刻 ≤ `P_last_layer_compute_done + ε`（ε ≈ 一层 KV 的 RDMA + sync 时延），而 NIXL 必须等 P 全部 prefill 跑完才开始 pull；
- 控制面 + 数据面合计 RTT 数比 NIXL 少一跳（NIXL 是 D 通知 P → P 算完 → D pull；本方案是 D 通知 P → P 边算边 push）。

## 2. 总体架构

```
                       ┌─────────────────────────────────────┐
                       │ Router (pegaflow-router)            │
                       │  - 选 P, 选 D, 在 D 请求里塞 P 端点 │
                       └────┬────────────────────────┬───────┘
                            │ HTTP                    │ HTTP
                  ┌─────────▼───────┐        ┌────────▼────────┐
                  │ D vLLM          │        │ P vLLM          │
                  │  PdPushConnector│        │ PdPushConnector │
                  │  (role=DECODE)  │        │ (role=PREFILL)  │
                  └────┬────────────┘        └────────┬────────┘
                       │                              │
                       │      ┌─────── OOB ZMQ ──────►│
                       │      │  (QP/GID/PSN/rkey/addr/req_id)
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
- **Router**：保持现状（参考 `pegaflow-server/src/bin/pegaflow-router.rs`），只多一个职责：在转发给 D 的请求 body 里塞 P 的 OOB endpoint（host:port），并把 prompt **同时**转给 P（不等 P 返回）。
- **D 节点**：拿到请求后先在自己的 GPU 上分配 KV slot（per layer），把 (addr, rkey, layer 数, block layout, req_id) 通过 OOB 推给 P。
- **P 节点**：拿到 "do remote prefill" 指令后立即调度本地 vLLM 算 prefill；每算完一层就把这层 KV WRITE 到 D；最后一层用 `WRITE_WITH_IMM` 携带 `req_id` 作为完成信号。
- **OOB control plane**：ZMQ ROUTER/DEALER（直接复用 vLLM NIXL connector 那套，详见 §4.1）。控制平面**不**走 Router，P↔D 之间直连。

## 3. RDMA 链路接管（**核心**，pegaflow-transfer v2，verbs only）

### 3.1 路线：vendor pplx-garden 的 fabric-lib 设计，砍掉 EFA

底层 RDMA 库的选择有三条路：

| 路线 | 库 | 评价 |
| --- | --- | --- |
| 现状 `pegaflow-transfer` v1 | crates.io `sideway 0.4.1` | libibverbs 安全 wrapper，第三方维护、sync API、单后端，不适合 layer-wise push |
| pplx-garden / `pegainfer-comm-fabric-lib` | 自 vendor `libibverbs-sys` (bindgen + 系统 `libibverbs.so`) + `libfabric-sys` (EFA) | trait + 双后端 + callback + Imm 一等公民，正好对应 PD push 的需求 |
| crates.io 上的 `rdma` / `ibverbs` 等 | 第三方 | 生态绑定，patch 不便，能力不齐 |

**选 pplx-garden 路线**，理由（详见 §3.2 对比表）：
- 异步 callback + `ImmCount` 原语，layer-wise 流水的语义直接对得上；
- `TransferRequest::{Single, Paged, Scatter, Imm, Barrier}` 五形态枚举覆盖未来；
- 自 vendor 的 -sys crate，patch 自由，不锁定生态。

**砍 EFA**：当前部署硬件只有 ConnectX (RoCE/IB)，EFA 路径在 PegaFlow 没场景。所以：
- 只 vendor `libibverbs-sys`，不 vendor `libfabric-sys`；
- 删 `efa/` 目录、`provider_dispatch.rs`、`RdmaDomain` 的 EFA impl；
- `RdmaDomain` trait **保留**——但目的从"多后端 dispatch"变成"为测试可 mock"。如果觉得多此一举，第一版也可以直接砍掉 trait，留 `VerbsDomain` 单结构体。倾向保留，方便单测。

**所有权**：v2 代码归 pegaflow。pegainfer 那边的 `pegainfer-comm-fabric-lib` 之后改为 path-depend pegaflow-transfer，或者干脆 move 过来后从 pegainfer workspace 移除。pegainfer-comm 顶层保留它的 `EpAllToAll` 抽象 surface，只是底层换成 pegaflow-transfer v2 提供。

### 3.2 设计差异：fabric-lib vs 现 pegaflow-transfer

| 维度 | pplx-garden / fabric-lib | pegaflow-transfer v1 (Sideway) | v2 取舍 |
| --- | --- | --- | --- |
| 底层 FFI | 自 vendor `libibverbs-sys` + `libfabric-sys` | crates.io `sideway` | 自 vendor `libibverbs-sys`（搬 fabric-lib 那份），不要 libfabric |
| 后端抽象 | `RdmaDomain` trait + verbs/efa 双实现 | 单 backend，无 trait | **保留 `RdmaDomain` trait + verbs 单实现**（mock 用） |
| 数据面 op | `TransferRequest::{Single, Paged, Scatter, Imm, Barrier}` | RDMA READ/WRITE 两动作，Mooncake 风格 | 全套搬，**第一版只用 Single + Imm**，Paged/Scatter/Barrier 留着备用 |
| 完成机制 | CQ poll 线程 + `TransferCallback{on_done, on_error}` + `ImmCount` | sync `transfer_sync_*` | 全套搬，async/callback |
| 控制面 | UD QP 内嵌握手 (`PeerHandshakeInfo`) | 应用层 TCP/gRPC | **跳过 UD 握手**，pegaflow 用 ZMQ OOB（见 §3.5），握手参数喂给 `connect_peer` |
| 多 NIC 路由 | `DomainGroupRouting::{RoundRobinSharded, Pinned}` | 多 QP per peer，简单 round-robin | 全套搬，PD 按 layer round-robin |
| GPU MR | DMA-BUF GDR，`MemoryRegionDescriptor` 含 rkey | sideway pin pinned memory | 全套搬（DMA-BUF） |
| 控制平面依赖 | 无 OOB 强依赖 | 应用层自带 | **必须 ZMQ**（D 通知 P 起 push） |
| `tokio` feature | 可选 | 弱 | **可选**，主路径不要求 |

### 3.3 v2 模块布局（pegaflow-transfer 内部）

```
pegaflow-transfer/
├── Cargo.toml
│   依赖:
│     v1: sideway = "0.4.1"   (保留, 给 v1 用)
│     v2: libibverbs-sys (path = "./libibverbs-sys")
│         cuda-lib (新增, 或复用 pegaflow-common 的 cuda binding)
│         + 现有 thiserror / serde / crossbeam-channel / dashmap / parking_lot / smallvec
│   features:
│     default = ["v1"]                # 暂保持现状
│     v1 = ["dep:sideway"]
│     v2 = ["dep:libibverbs-sys", ...]  # ★ 新, hw-rdma 等价物
├── libibverbs-sys/                   # ★ 搬自 pplx-garden, 独立 sub-crate
│   ├── Cargo.toml  (links = "ibverbs", bindgen)
│   ├── build.rs
│   ├── wrapper.h
│   └── src/lib.rs
├── src/
│   ├── lib.rs                        # pub use v1::*; pub mod v2;
│   ├── error.rs                      # 现有, 两版共用
│   ├── rdma_topo.rs                  # 现有, 两版共用 (GPU/NIC NUMA 拓扑)
│   ├── v1/                           # ★ 新, 把现有 engine.rs / rc_backend/ 收进来
│   │   ├── mod.rs                    # pub use engine::*;
│   │   ├── engine.rs                 # = 当前 engine.rs
│   │   └── rc_backend/               # = 当前 rc_backend/
│   ├── v2/                           # ★ 新
│   │   ├── mod.rs                    # pub use transfer_engine::TransferEngine;
│   │   ├── api.rs                    # TransferRequest, MemoryRegionDescriptor, ...
│   │   ├── provider.rs               # trait RdmaDomain (只为 mock, 单 verbs 实现)
│   │   ├── transfer_engine.rs        # TransferEngine + callback worker thread
│   │   ├── fabric_engine.rs          # FabricEngine (多 GPU/Worker 聚合)
│   │   ├── worker.rs                 # per-GPU Worker
│   │   ├── domain_group.rs           # DomainGroupRouting (multi-NIC)
│   │   ├── imm_count.rs              # ImmCount 计数器
│   │   ├── rdma_op.rs                # SingleWriteOp / PagedWriteOp / ...
│   │   ├── mr.rs                     # MemoryRegion + DMA-BUF GDR
│   │   ├── host_buffer.rs            # 控制消息用的小 host buffer
│   │   └── verbs/                    # 唯一 provider
│   │       ├── mod.rs
│   │       ├── verbs_address.rs
│   │       ├── verbs_devinfo.rs
│   │       ├── verbs_domain.rs       # VerbsDomain (RdmaDomain 实现)
│   │       ├── verbs_qp.rs
│   │       └── verbs_rdma_op.rs
│   └── bin/                          # 现有 cpu_bench / topo_cli, 保持挂 v1
└── (无 efa/, 无 provider_dispatch.rs, 无 libfabric-sys/)
```

要点：
- **同一个 crate, v2 常驻**：v1 兼容 API 和 v2 RDMA fabric 同时编译，CUDA 版本通过 `default`/`cuda-13` feature 选择，不再用 `v2` feature 做开关；
- **`libibverbs-sys` 作为 sub-crate**：跟 fabric-lib 同布局，path dep，不上 crates.io；
- **共享 `error.rs` / `rdma_topo.rs`**：两版都要的拓扑探测和错误类型；
- **v1 = 把现有 `engine.rs` + `rc_backend/` 整体挪进 `v1/` 子目录**，对外 `pegaflow_transfer::v1::MooncakeTransferEngine` 不变，再加一层 re-export 兼容旧路径（`pub use v1::engine::*` at crate root）；
- 后续 v1 删除时只要去掉 `v1/` 和 sideway dep，crate 形态不变。

### 3.4 用 v2 跑 PD push 的最小 API 表面

PD push connector 用到的就这些（全部在 v2 已有 API 里）：

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

不需要新发明 trait/wrapper。

### 3.5 控制面（OOB）

用 ZMQ（ROUTER/DEALER），跟 vLLM NIXL 一致。序列化用 fabric-lib v2 自带的 `postcard`/`serde`，`MemoryRegionDescriptor` 直接是 `Serialize`。详细握手时序见 §3.6。

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
// pegaflow-transfer/src/v2/handshake.rs 或 pegaflow-pd-push/ 里
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

**GPU Direct RDMA via DMA-BUF**——v2 内部的 `MemoryRegion::new()`（搬自 `pplx-garden/fabric-lib/src/mr.rs:24`）已经包好整套逻辑：

1. `cudaPointerGetAttributes` 验证是 device pointer；
2. `cuMemGetHandleForAddressRange(..., CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, ...)` 拿 fd；
3. `ibv_reg_dmabuf_mr(pd, offset, length, iova, fd, access)`。

**注册粒度**：vLLM 已经把 KV cache 注册成 per-layer tensor（见 `vllm/v1/worker/gpu/kv_connector.py:251` 的 `register_kv_caches`），我们直接 per-layer 调一次 `engine.register_memory_allow_remote(layer_kv_ptr, layer_kv_bytes)` 拿到 `MemoryRegionDescriptor`，整生命周期复用。
- D 实例的 MR 数量 = `num_layers * 2`（K/V 分离）或 `num_layers`（合并）；
- P 端同理；
- 不引入 buffer pool / arena——**vLLM 自己管 GPU KV，我们只是借地址**。

⚠️ kernel ≥ 5.12 才支持 DMA-BUF MR。fallback 路径（旧 kernel）走 nv_peer_mem，第一版**不实现**，直接 panic-with-clear-error。

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

### 4.1 角色识别

通过 `--kv-transfer-config` 的 `kv_role`：

```bash
# P 节点
vllm serve $MODEL \
  --no-enable-prefix-caching \               # ← 见 §3.7，第一版禁用 P 端 prefix cache
  --kv-transfer-config '{
    "kv_connector": "PdPushConnector",
    "kv_connector_module_path": "pegaflow.connectors.pd_push",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
      "oob_listen": "0.0.0.0:7100",
      "rdma_devices": ["mlx5_0","mlx5_1"]
    }
  }'

# D 节点
vllm serve $MODEL \
  --kv-transfer-config '{
    "kv_connector": "PdPushConnector",
    "kv_connector_module_path": "pegaflow.connectors.pd_push",
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
| `wait_for_layer_load(name)` | 不调用 | **第一版**：第 0 层入口 block 等 IMM；其他层立即返回 |
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
3. 给 P 的请求 body 注入 `kv_transfer_params = {do_remote_prefill_sender: true, target_engine_id: <D 的 OOB>, req_id: ...}`；
4. **同时**把请求转给 P 和 D（D 阻塞在 wait_for_layer_load，P 立刻开算）；
5. 等 D 的响应作为最终响应返回。

> 注意 P 不返回 token，只做 prefill。这要求 P 端在 `kv_role=producer` 时把 logits 丢弃。第一版直接复用 vLLM NIXL 的 producer-side 早退实现（NIXL connector 已经在 P 端做过这个）。

## 5. 代码落点

```
pegaflow/
├── pegaflow-transfer/                 # 现有 crate, 内部分 v1/v2
│   ├── Cargo.toml                     # 增 feature "v2" + path dep libibverbs-sys
│   ├── libibverbs-sys/                # ★ 新 sub-crate, 搬自 pplx-garden
│   │   ├── Cargo.toml  (links="ibverbs")
│   │   ├── build.rs
│   │   ├── wrapper.h
│   │   └── src/lib.rs
│   └── src/
│       ├── lib.rs                     # pub use v1 root re-export; pub mod v2;
│       ├── error.rs                   # 共用
│       ├── rdma_topo.rs               # 共用 (NUMA/GPU/NIC 拓扑)
│       ├── v1/                        # ★ 把现有 engine.rs + rc_backend/ 收进来
│       │   ├── mod.rs
│       │   ├── engine.rs
│       │   └── rc_backend/
│       └── v2/                        # ★ 新, 搬自 pplx-garden fabric-lib (去掉 efa)
│           ├── mod.rs                 # 对外 API: TransferEngine 等
│           ├── api.rs
│           ├── provider.rs            # trait RdmaDomain (mock-only)
│           ├── transfer_engine.rs
│           ├── transfer_engine_builder.rs
│           ├── fabric_engine.rs
│           ├── worker.rs
│           ├── domain_group.rs
│           ├── imm_count.rs
│           ├── rdma_op.rs
│           ├── mr.rs
│           ├── host_buffer.rs
│           ├── topo.rs
│           ├── handshake.rs           # ★ 新, PdHandshake / PerLayerMr (PD 专用握手 schema)
│           └── verbs/                 # 唯一 provider
├── pegaflow-pd-push/                  # ★ 新薄 crate
│   ├── Cargo.toml                     # 依赖 pegaflow-transfer (features=v2) + tokio + zmq + cuda-lib
│   └── src/
│       ├── lib.rs
│       ├── role.rs                    # Producer / Consumer 角色状态机
│       ├── oob.rs                     # ZMQ ROUTER/DEALER 握手 (Tokio)
│       ├── chunk_tracker.rs           # 每 req: chunks_pushed, is_finished
│       └── layout.rs                  # vLLM paged KV layout 计算 (slot_mapping -> block_idx)
├── python/
│   ├── src/lib.rs                     # PyO3: export pegaflow_pd_push::{Producer, Consumer}
│   └── pegaflow/
│       └── connectors/                # ★ 新目录, 与现有 connector/ 并列
│           ├── __init__.py
│           └── pd_push/
│               ├── __init__.py        # 暴露 PdPushConnector
│               ├── scheduler.py       # vLLM scheduler 侧
│               ├── worker.py          # vLLM worker 侧
│               ├── oob.py             # ZMQ thin wrapper (大多数逻辑在 Rust 侧)
│               └── meta.py            # ConnectorMetadata / PdHandshake Python 镜像
└── docs/
    ├── pd.md                          # 现状（CPU 中转）
    └── pd-rdma-push.md                # 本文
```

约束：
- **`pegaflow-core` / `pegaflow-server` / `pegaflow-metaserver` 不动**，所有改动局限在 `pegaflow-transfer` (新增 v2 模块) 和新 `pegaflow-pd-push` crate；
- `pegaflow-transfer` v1 公开 API 通过 root re-export 保持向后兼容（`use pegaflow_transfer::MooncakeTransferEngine` 仍然能用）；
- `pegaflow-router.rs` 只加一个 `--mode pd-push` 子命令，原 mode 保持不变；
- Python 侧 connector 走全新 module path `pegaflow.connectors.pd_push`，**不**挤进 `pegaflow.connector`。

### 5.1 关于 pegainfer 的统一

pegainfer 那边 `pegainfer-comm/crates/pegainfer-comm-fabric-lib` 是 pplx-garden 的另一份 vendor。pegaflow v2 稳定后，pegainfer 应当：
1. 删掉 `pegainfer-comm-fabric-lib/`、`pegainfer-comm-libibverbs-sys/`；
2. 改 `pegainfer-comm` 让 `EpAllToAll` 的 `hw-rdma` 实现底层换成 `pegaflow_transfer::v2::TransferEngine`；
3. 通过 path dep 或私有 registry 拉 pegaflow-transfer。

这一步不在本文档覆盖范围内（pegainfer 仓库独立），但 v2 的 API 设计要预留这个对齐：
- `TransferRequest` enum 保留 `Scatter`（pegainfer EP all-to-all 用得到），即使 PD 不用；
- `Worker` / `FabricEngine` 的多 GPU 抽象不削；
- 不在 v2 公开 API 里塞 PD 专用概念（PD 专用的东西全部放 `pegaflow-pd-push` 里）。

## 6. 实施阶段

| 阶段 | 内容 | 验收 |
| --- | --- | --- |
| **P0** v2 模块落地 | 把 pplx-garden `fabric-lib` 内容搬进 `pegaflow-transfer/src/v2/`，删 efa / libfabric，v2 作为默认编译路径常驻 | `cargo check -p pegaflow-transfer` 通过，原 v1 API 路径不退化 |
| **P1** RDMA 自检 | 写一个 Rust bin (`pegaflow-transfer/src/bin/v2_loopback.rs`)，loopback 跑 GPU→GPU WRITE + IMM；单机两进程 + ZMQ 握手 | 1 GB WRITE 走通，IMM 到达，带宽接近 line rate（具体数字由后续 benchmark 给）|
| **P2** PD push crate | 写 `pegaflow-pd-push`，封装 OOB + chunk tracker + KV layout；不接 vLLM，写个 stub 跑通 producer/consumer 协议 | 双进程 stub 模拟"P 逐层 push 1024 个 block + IMM"，consumer 拿到 IMM 后能正确比对 KV |
| **P3** vLLM hook 接入 | Python connector + PyO3 binding，单机 P + D 各占 1 GPU，跑小模型 | E2E 出第一 token，与 NIXL baseline 输出 token-level 一致 |
| **P4** 多 NIC + chunked prefill | 多 NIC round-robin (`DomainGroupRouting::RoundRobinSharded`)；P 端开 chunked prefill | 长 prompt（≥ max_num_batched_tokens 的 2-3 倍）跑通，多 chunk push 后 IMM 触发 |
| **P5** Router 接入 + benchmark | `pegaflow-router --mode pd-push`，跨节点 1P+1D；对照 NIXL 同条件压测 | 压测细节后续给出；目标见 §1 设计抓手 |

P1 之前**不写** Python，不接 vLLM；先把 RDMA 链路本身在小 Rust binary 里跑稳，避免 vLLM/Python/CUDA stack 的噪声盖住 RDMA bug。pplx-garden 的工程经验就是 fabric-lib 独立可测。

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
