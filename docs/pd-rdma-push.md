# PegaFlow PD Connector：RDMA Push + Layer-wise

> 与现有 `pd.md`（CPU 中转 + 异步回调）完全独立的一条新链路。
> 单独成文、单独连接器实现、单独配置入口，**不要**回写到现有 `python/pegaflow/connector/`。

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
                       │      ┌─────── D → P prefill request ──────►│
                       │      │  (prompt + MR/rkey/addr/req_id)
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
- **P/D control plane**：Router 只把用户请求转给 D；D allocate KV 后直接请求 P，
  请求体携带 prompt、producer-side `kv_transfer_params` 和 D 侧 MR/rkey/addr layout。
  完成信号不走 ZMQ，走 RDMA IMM。

## 3. RDMA 链路接管（pegaflow-transfer v2，verbs only）

### 3.1 v2 路线：常驻，verbs only，沿用 sideway/mummy 栈

`pegaflow-transfer` 有两条独立链路：

| 链路 | 定位 | 当前状态 |
| --- | --- | --- |
| v1 | Mooncake-style RDMA READ/WRITE batch API | 保持原 API，不改 |
| v2 | PD push / layer-wise RDMA WRITE + IMM substrate | 默认编译，已有 CPU/GPU bench |

v2 **不是 v1 的替代品**。v1 的接口模型是"caller 给一批 desc，内部按 NUMA/NIC/QP 分发"；v2 的接口模型是"caller 显式提交 `TransferRequest`，并用 `DomainGroupRouting` 描述本 engine 内 domain 选择"。

底层依赖：
- `sideway = 0.4.2`，作为 verbs 生态入口；`rdma-mummy-sys = 0.2.3`，由 sideway 依赖；
- CUDA 绑定走 `cudarc`；`tokio` 默认存在；v2 默认编译，无 feature 开关。

### 3.2 设计差异：v1 vs v2

| 维度 | v1 | v2 |
| --- | --- | --- |
| API 粒度 | `batch_transfer_async(op, remote, descs)` | `TransferRequest::{Single, Paged, Scatter, Imm, Barrier}` |
| NIC 选择 | 内部按 NUMA page 分桶、NIC round-robin | caller 通过 `DomainGroupRouting::{Pinned, RoundRobinSharded}` 指定 |
| 完成机制 | 每 active NIC 返回 receiver | callback / atomic counter / `ImmCounter` |
| PD push 适配 | 缺 IMM 一等原语 | `Single + Imm` / `Scatter + Imm` |
| GPU MR | 不承担 PD GPU path | h20 上 `ibv_reg_mr(cuda_ptr)` 已能跑；DMA-BUF path 等上游 |

### 3.3 v2 模块布局

```
pegaflow-transfer/src/
├── lib.rs / engine.rs / rc_backend/    # v1 API + backend
├── rdma_topo.rs                        # 两版共用 GPU/NIC NUMA 拓扑
├── cuda_lib/ / cuda_sys.rs             # cudarc wrappers
└── v2/
    ├── mod.rs / api.rs                 # TransferRequest, MemoryRegionDescriptor
    ├── transfer_engine.rs              # TransferEngine + callback worker
    ├── fabric_engine.rs                # FabricEngine (多 GPU/Worker 聚合)
    ├── worker.rs / domain_group.rs     # per-GPU Worker + multi-NIC routing
    ├── imm_count.rs                    # ImmCount 计数器
    ├── rdma_op.rs / mr.rs              # RDMA ops + MR 管理
    └── verbs/                          # verbs provider (VerbsDomain)
```

### 3.4 PD push 使用的 v2 API 表面

PD push connector 用到的 v2 primitives：

```rust
use pegaflow_transfer::v2::{
    TransferEngine, TransferRequest, SingleTransferRequest,
    ImmTransferRequest, TransferCallback, ImmCount,
    MemoryRegion, MemoryRegionDescriptor,
    DomainGroupRouting, PeerGroupHandle,
};

// 启动时
let engine = Arc::new(TransferEngine::new_with_fabric(fabric)?);

// 注册一层 KV 显存
let mr = engine.register_memory_allow_remote(layer_kv_ptr, layer_kv_bytes)?;
// mr.descriptor() -> MemoryRegionDescriptor (含 (DomainAddress, rkey) 列表)

// OOB 拿到对端 descriptor 后, 建 peer group
let peer = engine.add_peer_group(remote_descriptor, routing)?;

// P 端: 每层算完, push blocks
engine.submit_transfer(TransferRequest::Single(...), callback)?;

// 最后一层: 带 IMM
engine.submit_transfer(TransferRequest::Imm(ImmTransferRequest {
    peer, imm_data: encode_req_done(req_id),
}), no_cb)?;

// D 端: 注册 IMM 监听
engine.register_imm_callback(|imm: u32| { mark_kv_ready(decode_req_done(imm)); });
```

Python 侧通过 `RdmaPort` protocol 抽象，`RealRdmaPort` 接 PyO3 `PdRdmaEngine`。

当前 PyO3 `PdRdmaEngine` 已完整实现：
- `register_local_layers(layers)` → 注册本地 KV tensor region，返回含 MR 描述的 layout
- `register_remote(req_id, handshake)` → 建立 peer group / remote MR view，预注册 IMM counter
- `push_layer(req_id, layer_idx, blocks)` → 提交 scatter RDMA WRITE，write window 限流
- `wait_for_pushes(req_id)` / `push_done(req_id)` → 等 WRITE 落地 + 发 IMM
- `wait_done(req_id)` / `poll_done(req_id)` → 等/查 IMM 完成
- `pop_finished_sending()` / `pop_finished_recving()` → 完成集合
- `close_request(req_id)` → 清理 per-request 状态

### 3.5 控制面

scheduler/worker 之间走 vLLM 自己的 connector metadata。D worker allocate KV 后构造
`PdHandshake`，通过 D→P prefill HTTP 请求携带 prompt、producer-side
`kv_transfer_params`、layout、block ids 和 MR 描述符。P/D 完成信号走 RDMA IMM，
D 侧后台 waiter 收到 IMM 后由 `get_finished` 上报给 vLLM。

### 3.6 连接 bring-up

每对 (P_worker, D_worker) 一对 RC QP。D 侧主动发起：

```
D                                          P
│ 收到 router 请求, 解析 prompt           │
│ 本地 vllm allocate KV slots              │
│ engine.register_memory_allow_remote()    │
│   -> MemoryRegionDescriptor[num_layers]  │
│                                          │
│ HTTP prefill request {prompt,            │
│   kv_transfer_params, mrs_D} ──────────► │ 收到 do_remote_prefill
│                                          │ engine.register_memory_local(P KV)
│                                          │ engine.add_peer_group(mrs_D, ...)
│                                          │
│ 等 IMM                                    │ 开始 prefill, 逐层 submit_transfer
```

OOB 上交换的 metadata：

```rust
pub struct PdHandshake {
    pub req_id: String,
    pub model_id: String,                          // 一致性校验
    pub num_layers: u32,
    pub tp_rank: u32,
    pub kv_layout: KvLayout,
    pub block_bytes: u32,
    pub layers: Vec<PerLayerMr>,
    pub peer_conn: PeerConnDescriptor,
}

pub struct PerLayerMr {
    pub layer_idx: u32,
    pub k: MemoryRegionDescriptor,
    pub v: Option<MemoryRegionDescriptor>,
    pub k_block_addrs: Vec<u64>,                   // paged KV per-block 地址
    pub v_block_addrs: Option<Vec<u64>>,
}
```

一致性校验在 `add_peer_group` 之前做：`model_id` / `num_layers` / `tp_size` 不一致直接拒绝。

### 3.7 内存注册

当前用 `ibv_reg_mr(cuda_ptr)` 跑 GPU memory WRITE+IMM。正式生产路径仍要切到 DMA-BUF MR（`ibv_reg_dmabuf_mr`），需要 sideway / rdma-mummy-sys 上游暴露。

**注册粒度**：vLLM connector 在 worker 初始化时拿到 KV cache tensor，按实际 tensor region 注册并生成 layer/block 地址表，整生命周期复用。不引入 buffer pool / arena——**vLLM 自己管 GPU KV，我们只是借地址**。

⚠️ kernel ≥ 5.12 才支持 DMA-BUF MR。第一版不实现 nv_peer_mem fallback，直接报错。

### 3.8 传输（per-block scatter WRITE）

`save_kv_layer` hook 内的 P 端逻辑：

```
on_layer_done(layer_idx, slot_mapping, ctx):
    blocks_this_step = unique(slot_mapping / block_size)  # 去重

    for block in blocks_this_step:
        submit_write(K_slice(block), remote.k_addr(block))
        submit_write(V_slice(block), remote.v_addr(block))

    if is_last_layer && is_last_chunk:
        submit_write_with_imm(encode_req_done(req_id))
```

要点：
- 一个 block 的 K + V = 2 个 `Single`，最后一个 chunk 最后一层追加 `Imm`；
- **不**等 completion 再算下一层——CUDA stream 上 vLLM 继续往下算，host 侧 RDMA 自己推；
- 多 NIC 在 `add_peer_group(DomainGroupRouting::RoundRobinSharded)` 里指定，v2 内部自动分流。

### 3.9 D 端完成感知

第一版**不需要**逐层感知。D 在收到最后一层 IMM 后才放行 decode。当前实现用后台 RDMA waiter 等 IMM，`get_finished` 把完成状态上报给 vLLM scheduler；`wait_for_layer_load` 不再承担同步等待。

> **后续优化**：让 D 在 layer_i 计算前才阻塞等 layer_i 到达，实现真正的 D 侧 layer-wise 流水。需要 P 每层都打 WITH_IMM 携带 `(req_id_hash, layer_idx)`，第一版不做。

### 3.10 Chunked prefill：第一版就支持

#### 为什么必须支持

vLLM 默认开启 chunked prefill，关掉的话长 prompt 直接顶到 OOM 边缘。第一版直接支持，理由：

1. **paged KV 本来就强制 per-block 推送**。vLLM 的 KV 散在 `block_table` 指向的多个非连续 block 里，即使关掉 chunked prefill，一层 KV 也要做 `len(block_table)` 个 WRITE。
2. 一旦做了 per-block scatter，"分多次推" vs "一次推完" 的边际复杂度很小：P 端多记一个 chunk 状态。
3. NIXL baseline 默认开 chunked prefill，关了再比赛会被诟病。

#### Push 语义

```
对每个 forward step k:
  对每层 i = 0..L-1:
    on save_kv_layer(layer_i, kv, attn_metadata):
      blocks_this_step = unique(attn_metadata.slot_mapping // block_size)
      for b in blocks_this_step:
        submit_write(K_slice(b), remote.k_addr(b))
        submit_write(V_slice(b), remote.v_addr(b))

      若 i == L-1 且本 step 是该 req 的最后一个 chunk:
        最后一个 WRITE 用 WRITE_WITH_IMM
```

**寻址来源**：本 step 写到哪些 block，来自 `attn_metadata.slot_mapping`。logical block → D 端 remote addr 的映射，握手时一次性传过来。

**完成信号**：只在"最后一个 chunk 的最后一层"打 IMM。

#### 边界约束

- **D 必须一次性把 KV slot 全分配好**（即使 P 还没推第一个 chunk）。
- **P 端 preemption**：已推的 KV 在 D 上仍合法，P 重新调度后会重写同位置，幂等所以 OK，第一版接受带宽浪费。
- **chunked prefill + prefix caching**：P 端开 prefix cache 时，前缀 block 不会出现在 `slot_mapping` 里，D 端拿不到完整 KV。第一版要求 **P 端关闭 prefix caching**。

### 3.11 完成上报：fire-and-forget + `get_finished` 交集

P 主线程不能在 `save_kv_layer` / `wait_for_save` 里同步等 RDMA。等待全部挪到后台。

```
save_kv_layer (主线程):
  - 提交 push (异步 via _push_sender)

last layer of last chunk (主线程):
  - 把 req_id 丢进 _req_pending_finalize（不阻塞）
  - finalize 后台线程：
      wait_for_pushes(req_id)   # 等所有 scatter WRITE 落地
      push_done(req_id)         # IMM
      _completed_pushes.add(req_id)

get_finished(finished_req_ids):
  done = finished_req_ids & _req_pending_finalize & _completed_pushes
  → 回 finished_sending = done
```

要点：
- `wait_for_pushes` 不能砍：多 NIC scatter 跨 QP 不保序，IMM 必须等所有 WRITE 落地才发；但在后台线程做。
- `finished_req_ids & _completed_pushes` 交集是正确性要求：早回 `finished_sending` 会让 vLLM 回收 KV 显存，后台 RDMA 会读到脏数据。
- D 侧 `get_finished` 不对 `_wait_reqs` 跑 `poll_done` for-loop。IMM 完成由后台 waiter 写入 `finished_recving`，主线程直接 `pop_finished_recving()`。
- 超时失败走对称路径上报，触发 D 本地重算。

## 4. vLLM 接入（不 fork vllm）

### 4.0 NIXL connector 调研结论

vLLM NIXL connector 的可参考部分主要是模块边界，而不是传输方向。`pd_connector` 按 NIXL 的文件组织和 hook 分层新写，走全新 module path `pegaflow.pd_connector`。

NIXL 的关键 hook 行为对照：

| hook | NIXL 行为 | `pd_connector` |
| --- | --- | --- |
| `get_required_kvcache_layout` | 非 MLA 时偏好 HND | 沿用 HND 偏好 |
| `get_num_new_matched_tokens` | 返回可异步拉取 token 数 | D 侧返回 prompt token 数并声明 async |
| `update_state_after_alloc` | 记录 req 和本地 block ids | 记录 D 端 block ids，准备 handshake |
| `build_connector_meta` | 打包给 worker | 打包 `reqs_to_wait` / `reqs_to_push` |
| `save_kv_layer` | 不做 layer-wise save | P worker 核心路径：按 layer/block push |
| `get_finished` | 汇总 async 完成和失败 | 汇总 P push 完成、D wait done、失败 |

### 4.1 角色识别 + per-rank NIC/CPU 绑定

通过 `--kv-transfer-config` 的 `kv_role` 区分 P/D。每个 TP rank 的 NIC 和 pin CPU
**显式**写在 `pegaflow.pd.rdma.rank_map` 里——不再依赖自动拓扑探测做隐式选择。

```bash
# P 节点（TP=8）
vllm serve $MODEL \
  --no-enable-prefix-caching \
  --kv-transfer-config '{
    "kv_connector": "PdConnector",
    "kv_connector_module_path": "pegaflow.pd_connector",
    "kv_role": "kv_producer",
    "engine_id": "prefill",
    "kv_connector_extra_config": {
      "pegaflow.pd.rdma.rank_map": {
        "0": {"nic": "mlx5_0", "worker_cpu": 16, "uvm_cpu": 17},
        ...
        "7": {"nic": "mlx5_7", "worker_cpu": 30, "uvm_cpu": 31}
      }
    }
  }'
```

**schema**：
- key 是 TP rank 字符串；`nic` 每 rank 一张 HCA；`worker_cpu` / `uvm_cpu` 两个不同 CPU。
- `cuda_device` 不写，由 `kv_caches.device.index` 推断。

**校验规则**（启动时 fail-fast）：
1. 当前 `tp_rank` 必须在 map 里。
2. 所有 rank 的 `worker_cpu ∪ uvm_cpu` 必须互不相同。
3. CPU 不允许在 `{0..15}`（默认 floor=16，`pegaflow.pd.rdma.reserved_cpu_floor` 可覆盖）。
4. `nic` 必须在 `ibv_get_device_list()` 里。
5. NIC ↔ CPU 的 NUMA 不一致只 warn。

**为什么不靠自动 pin**：实测默认 pin 会撞上 vLLM TP worker 主线程，h20 上 16 MiB 只跑 2.5 Gbps；显式 pin 避开后 24.7 Gbps。

### 4.2 Hook 映射

| vLLM hook | P (producer) | D (consumer) |
| --- | --- | --- |
| `get_num_new_matched_tokens` | 返回 `(0, False)` | 返回 `(prompt_len, True)` |
| `update_state_after_alloc` | 无操作 | 记 block_ids，构造 handshake |
| `build_connector_meta` | 打包 push 请求 | 打包 wait 请求 |
| `save_kv_layer` | 异步提交 push，最后一 chunk 的最后一层进 `_req_pending_finalize` | 不调用 |
| `wait_for_save` | no-op | 不调用 |
| `get_finished` | `finished_req_ids & _req_pending_finalize & _completed_pushes` | 后台 waiter 完成进 `_completed_recvs`，取交集 |
| `request_finished` | 释放 MR 引用 | 释放 MR 引用 |

请求携带的 `kv_transfer_params`：

```python
{
  "do_remote_prefill": True,
  "remote_engine_id": "p-node-3:7100",
  "tp_size": 4,
  "req_id": "<uuid>",
}
```

### 4.3 Router

当前 `pegaflow-router.rs` **尚未实现** PD 路由。开发阶段用 `pd_connector.proxy` 手动构造
`kv_transfer_params` 做 connector-level 验证。

Router 第一版改造计划：
1. 选 P + 选 D，给 D 注入 `kv_transfer_params`，只转发 D；
2. D allocate KV 后通过 HTTP 把 prompt + layout 发给 P；
3. P 做 prefill + RDMA WRITE，不返回 token。

## 5. 代码落点

```
pegaflow/
├── pegaflow-transfer/                 # 现有 crate
│   └── src/v2/                        # RDMA substrate
├── python/
│   ├── src/lib.rs                     # PyO3 PdRdmaEngine binding
│   └── pegaflow/
│       └── pd_connector/
│           ├── __init__.py            # PdConnector facade (KVConnectorBase_V1)
│           ├── scheduler.py           # scheduler-side：状态机、MR cache、prefill dispatch
│           ├── worker.py              # worker-side：push/wait 执行、async 线程
│           ├── metadata.py            # PdHandshake / PdConnectorMetadata / layouts
│           ├── rdma.py                # RdmaPort protocol + MockRdmaPort + RealRdmaPort
│           ├── layout.py              # FlashAttnHndLayout / slot_mapping → block slice
│           ├── chunk_tracker.py       # per-req chunk 进度
│           ├── prefill.py             # AsyncPrefillSender (D→P HTTP)
│           ├── oob.py                 # InMemoryOobPort（进程内 pub/sub 占位）
│           └── proxy.py              # 本地 P/D 调试代理
└── docs/
    ├── pd.md                          # 现状（CPU 中转）
    └── pd-rdma-push.md                # 本文
```

### 5.1 本地 P/D 调试

proxy 只作为手动构造请求的调试工具；生产语义是 Router 只请求 D，D 再向 P 发 prefill 请求。

```bash
# P 节点
PYTHONPATH=$PWD/python CUDA_VISIBLE_DEVICES=0 vllm serve /data/Qwen3-4B \
  --host 127.0.0.1 --port 8001 \
  --kv-transfer-config '{ "kv_connector": "PdConnector",
    "kv_connector_module_path": "pegaflow.pd_connector",
    "kv_role": "kv_producer", "engine_id": "prefill",
    "kv_connector_extra_config": {
      "pegaflow.pd.rdma.rank_map": {
        "0": {"nic": "mlx5_1", "worker_cpu": 60, "uvm_cpu": 62}
      }}}'

# D 节点
PYTHONPATH=$PWD/python CUDA_VISIBLE_DEVICES=1 vllm serve /data/Qwen3-4B \
  --host 127.0.0.1 --port 8002 \
  --kv-transfer-config '{ "kv_connector": "PdConnector",
    "kv_connector_module_path": "pegaflow.pd_connector",
    "kv_role": "kv_consumer", "engine_id": "decode",
    "kv_connector_extra_config": {
      "pegaflow.pd.rdma.rank_map": {
        "0": {"nic": "mlx5_1", "worker_cpu": 64, "uvm_cpu": 66}
      }}}'

# Proxy
cd python && uv run python -m pegaflow.pd_connector.proxy \
  --listen-host 127.0.0.1 --listen-port 8100 \
  --prefill-url http://127.0.0.1:8001 \
  --decode-url http://127.0.0.1:8002

# 请求
curl -s http://127.0.0.1:8100/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/data/Qwen3-4B","prompt":"Write a short note about RDMA.","max_tokens":16,"temperature":0}'
```

### 5.2 关于 pegainfer 的统一

pegainfer 的 `pegainfer-comm-fabric-lib` 是 pplx-garden 的另一份 vendor。pegaflow v2 稳定后，pegainfer 应改用 `pegaflow_transfer::v2::TransferEngine`。v2 的 API 设计预留了这个对齐：`TransferRequest` 保留 `Scatter`，`Worker` / `FabricEngine` 多 GPU 抽象不削，不在 v2 公开 API 里塞 PD 专用概念。

## 6. 实施阶段与当前进展

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| **P0** v2 RDMA substrate | v2 默认编译；host/cuda WRITE+IMM bench | ✅ h20 单 400G HCA 约 372 Gbps |
| **P1** connector skeleton | `pegaflow.pd_connector`；vLLM hook、状态机、metadata、layout、chunk tracker | ✅ MockRdmaPort 单测通过 |
| **P2** PyO3 binding | `PdRdmaEngine` + `RealRdmaPort` | ✅ 编译、契约单测、h20 probe 通过 |
| **P3** 接真实 v2 RDMA | GPU buffer WRITE+IMM 代替 mock | ✅ 同 HCA 小模型 E2E；跨机 TP8 |
| **P4** vLLM 小模型 E2E | 单机/跨机 P+D 跑小模型 | ✅ 同 HCA 和跨机 TP8 均已跑通 |
| **P5** Router + 正式压测 | Router PD 路由；对照 NIXL 同条件压测 | ❌ 未开始 |

### 阶段性压测结果

> ⚠️ 以下数据**不能**直接对比。各行使用了不同的 `max_num_batched_tokens` 和
> vLLM scheduler 配置，仅用于观察优化方向。正式结论前需要统一配置重跑所有 baseline。

条件：Qwen3-8B / 30k prompt / TP8 / prefix cache off / 2-node H20 (P + D)

| 路径 | TTFT | 配置差异 |
| --- | ---: | --- |
| Direct P baseline | 836.9 ms | 默认 `max_num_batched_tokens` |
| NIXL baseline | 1045.9 ms | 默认 `max_num_batched_tokens` |
| PD push（默认 chunked） | 1181.8 ms | 默认配置，4 chunk |
| PD push（`max_num_batched_tokens=32768`） | 590.6 ms | 单 chunk，**配置不同于上面两行** |

RDMA 写本身已不是瓶颈：P 端单层每 rank range WRITE 通常 0.02-0.10 ms，观测带宽数百 Gbps 到 1 Tbps 以上。当前 TTFT 开销主要来自：
1. **P 侧 chunked prefill 调度间隔**：默认配置下 30k prompt 被拆成 4 chunk，每 chunk 间有明显 scheduler/worker 间隔（request forward span 约 642 ms）；设成单 chunk 后降到约 60 ms。
2. **D 侧 TP8 handshake fan-in**：worker export → scheduler dispatch 约 138-163 ms。

## 7. 开放问题

1. **DMA-BUF kernel 版本**：H800 集群 kernel 是否都 ≥ 5.12？→ 待确认。
2. **错误恢复**：RDMA WRITE 中途 QP ERR → 第一版标记失败 + D 本地重算。
3. **流控 / 背压**：由 D 的 vLLM scheduler 做 admission control，P 端不限流。
4. **跨 instance（不同 TP 拓扑）**：暂不支持，`model_id` 一致性校验挡掉。
5. **Prefix caching 与 PD push 的合流**：第一版要求 P 端关 prefix cache。V2 应让 D 告诉 P 哪些 prefix 块已有。
6. **MLA KV layout 支持**：当前只支持 FlashAttention 5D HND layout，MLA（Kimi 等）会被 assert 拦下。压测阶段先用 HND 模型，MLA 作为正式 TODO。
7. **TP8 handshake fan-in**：已改为 D scheduler fan-in 所有 rank handshake + 直接异步触发 P prefill，但 fan-in 成本仍偏大（约 138 ms），后续继续压。
8. **handshake 体积**：HTTP control plane 使用 `block_addr_format=linear` 紧凑表示连续 block layout，生产 native 注册时在本地展开。
9. **跨 HCA 同机路由**：P: GPU0/`mlx5_1`, D: GPU2/`mlx5_2` 会超时。需要多 domain peer/routing，而不是靠选同 HCA GPU。

## 8. 不做的事

- 不做 RDMA READ（pull）模式——NIXL 已覆盖。
- 不做 SSD / CPU 中转回退——失败就 D 本地 prefill。
- 不做 IBGDA / GPU-initiated RDMA。
- 不做 PyO3 binding 之外的语言接入。

## 9. 参考实现指针

v2 是 `pplx-garden/fabric-lib` 的 vendor（删 efa）。

| 关心的事 | 看哪 |
| --- | --- |
| RC QP bring-up | `pplx-garden/fabric-lib/src/verbs/verbs_domain.rs:501` |
| DMA-BUF MR 注册 | `pplx-garden/fabric-lib/src/mr.rs:24` |
| WRITE / WRITE_WITH_IMM | `pplx-garden/fabric-lib/src/rdma_op.rs:8` + `transfer_engine.rs:172` |
| CQ poll + callback | `pplx-garden/fabric-lib/src/transfer_engine.rs:414` |
| 多 NIC round-robin | `pplx-garden/fabric-lib/src/api.rs` (`DomainGroupRouting`) |
| `ImmCount` 计数器 | `pplx-garden/fabric-lib/src/imm_count.rs` |
| `KVConnectorBase_V1` hook 表 | `vllm/distributed/kv_transfer/kv_connector/v1/base.py:171` |
| layer-wise hook 时序 | `vllm/v1/worker/gpu/kv_connector.py:62` |
| chunked prefill 累积参考 | `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:399-451` |
