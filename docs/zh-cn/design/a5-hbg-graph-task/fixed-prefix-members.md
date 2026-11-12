# 普通 SPMD 使用固定前缀成员集合

2026-11-09，基于 `197991d58` 的候选只调整普通、非 `sync_start`、非 MIX SPMD：
将 admission 成员数限制为 `min(block_num, eligible_workers)`，选取类型 bitmap 中最前面的成员。
全局 admission、原子 block 领取、完成计数和 `defer_wait` 路径保持不变。

设备 0 独占，Qwen3-14B 40 层、batch 16、seq_len 3500、seed 1234；
每次上传的全部输入逐项核对冻结哈希，完整 out/K/V 校验通过。
5 次预热后采集 50 个无 profiling 稳态样本；同一次设备分配末尾复测 `d24e9990c` 主线。

| 指标 | 主线复测 | 固定前缀成员候选 |
| ---- | -------- | ---------------- |
| Device 中位数 | 39.570 ms | 44.310 ms |
| Device 均值 | 39.551 ms | 44.334 ms |
| Device 范围 | 39.267–39.858 ms | 44.034–44.754 ms |

独立 L1 采样中，候选的首 kernel 至末 kernel 跨度为 45.480 ms。
按实际 block 执行、数据依赖和同核前驱重建的因果路径包含 20.621 ms kernel 区间与
24.859 ms 调度间隙。此前未限成员的 `197991d58` 采样分别为 16.063 ms 与 23.047 ms；
两次 L1 采样不代替无 profiling 稳态对比。

400 个 6-block AIC 任务的 2,400 次执行从全部 32 个 AIC 缩至固定 6 个 AIC。
原版每核执行 61–84 个 block，候选这 6 核各执行 395–406 个 block，其余核心没有领取这些任务。
这验证了固定前缀放置造成工作集中，不能用减少成员登记数量推断端到端加速。

该候选未进入运行时源码。仅否定固定前缀策略，不否定有负载分布依据的成员选择。
本地证据位于 `outputs/a5-hbg-schedule-8ms/spmd-members/`，包括逐轮日志、完整输入身份、
冻结运行库、L1/L3 采样、`critical-executions-l1.json` 和 `c6-placement.json`；不随仓库分发。
