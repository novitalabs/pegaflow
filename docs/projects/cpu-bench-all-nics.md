# cpu_bench: add whole-machine all-NIC aggregate benchmark

**Status**: Preparation — 等待 review

**TL;DR**: `pegaflow_cpu_bench` 目前只在单个 NUMA group 内做 multi-NIC 聚合，缺少跨 NUMA 全机所有网卡的压测模式。

## Preparation

### 读了什么

- `pegaflow-transfer/src/bin/cpu_bench.rs` — 当前 bench 实现

### 现状分析

当前 `cpu_bench.rs` 的压测结构：

```
for each NUMA group:
    Phase 1: single-NIC baseline (每张卡独立跑)
    Phase 2: multi-NIC aggregate (同一 NUMA group 内所有卡聚合)
```

**缺失**：没有 Phase 3 — 用全机所有网卡（跨 NUMA）创建一个 engine，跑聚合压测。这才是实际部署时的工作模式。

### NIC 选择机制（`rc_backend/mod.rs`）

`batch_transfer_async` 做 NUMA-aware round-robin：
- 用 `move_pages` 查每个 descriptor 的 `local_ptr` 所在 NUMA node
- 路由到同 NUMA 的网卡（`NumaRoundRobin::pick`）
- 如果所有 NIC 在同一 NUMA（`single_numa`），退化为普通 round-robin

**关键约束**：buffer 如果全在一个 NUMA node，所有 desc 会被路由到该 NUMA 的网卡，其他 NUMA 的网卡空转。

### 改动方案

在所有 per-NUMA-group bench 跑完后，新增 "all NICs" phase：

1. 收集全机所有未被 `--nic` / `--exclude-nic` 过滤掉的网卡
2. 仅在网卡跨越多个 NUMA group 时触发（`--nic` / `--numa` 指定时跳过）
3. **按 NUMA 分配 buffer**：每个有网卡的 NUMA node 各分配一对 server/client buffer
4. **交错构造 scatter list**：block i 分配到 NUMA node `i % num_nodes`，使 engine 的 NUMA-aware 路由均匀打到所有网卡
5. 每个 NUMA buffer 大小 = `ceil(max_blocks / num_nodes) * block_size`

改动集中在 `main()` 尾部 + 新增 `build_numa_scatter` 函数，不影响现有逻辑。

## Execution Log

- 实现完成，`cargo check --bin pegaflow-cpu-bench` 通过
- 新增 `build_numa_scatter` 函数：按 NUMA node round-robin 交错分配 block 到对应 buffer
- all-NIC phase 在 per-NUMA bench 后执行，条件：未指定 `--nic`/`--numa` 且网卡跨 >1 NUMA group
- H20 机器实测通过（4×400Gbps RoCE, 2 NUMA, `--exclude-nic mlx5_0`）
  - 拓扑：NUMA0 mlx5_1/mlx5_2, NUMA1 mlx5_3/mlx5_4
  - 单卡：Write ~361 Gbps, Read ~304 Gbps
  - 同 NUMA 2卡：Write ~717 Gbps, Read ~584-606 Gbps
  - **全机 4卡 cross-NUMA：Write 1402 Gbps (163 GiB/s), Read 1177 Gbps (137 GiB/s)**
  - 扩展效率：Write 97%, Read 97%（近线性）
