# DSV4.1 三组配置 L0 平均耗时

实测日期：**2026-11-13**。9 个 workload × 3 种配置，共 **27 项全部通过**；每项 1 次预热、10 次正式测量，逐轮数值校验。以下均为本轮新测的算术平均值，单位 **ms**。

| Case | PyPTO-lib 原版 | DSL 原版 | DSL ≤64 blocks 加 sync_start |
| ---- | -------------- | -------- | ---------------------------- |
| swa | 0.810437 | 1.239594 | 1.526071 |
| c2a_full | 1.400168 | 1.693527 | 1.919586 |
| c2a_reuse | 1.167324 | 1.646076 | 1.873984 |
| c1a_full | 1.385058 | 1.718591 | 1.964377 |
| c1a_reindex | 1.360394 | 1.699443 | 1.874174 |
| c1a_reuse | 1.166869 | 1.611902 | 1.828802 |
| moe | 2.534701 | 1.606016 | 2.173184 |
| engram | 1.131047 | 1.127001 | 1.374100 |
| layers_19_21 | 10.718427 | 9.240956 | 11.501316 |

## 测量口径

- Runtime 基线：`hw-native-sys/simpler@54eaedab7`，Host、AICPU、AICore 均从该提交重新编译；runtime 源码无本地调度修改。
- Runner 和 replay 使用同一基线构建，补齐 benchmark 的 CCU 构建依赖；runner 在两个 rank 都完成 `prepare_native_run` 后对齐启动，将 peer 的 Host 准备排除在 Device 测量之外。
- Ascend950DT_9581（A5）、CANN 9.3.0、驱动 25.6.rc2.b023；默认 `-O3 -g`，L0、resident scheduler mode 1。
- 所有设备通过 `task-submit --device auto` 自动分配。每个 workload 的三种配置依次在同一张卡或同一对卡上运行；本轮单卡分配到 4、5，EP2 分配到 6、7。
- 指标为 `chip.run.runner_run.device_wall`，不含 Host 准备、前缀准备和预热，包含 case 内通信。MoE 和 layers_19_21 为 EP2，每轮取两 rank 的较大耗时，再计算 10 轮平均值。
- 复用原业务 kernel 和固定零权重输入：每 rank 72 个有效 token、补齐至 80 行、64 个本地 expert。按当前 callable ABI 重新打包，逐一核对 kernel 字节与参数签名一致；任务记录的布局和执行数也已验证。
- 三列全部来自本轮测量，共 270 个正式 case 样本；不混用历史批次。结论只覆盖上述源码、输入及运行顺序。

## DSL 同步变体

仅对选定 case 中 `1 < block_num <= 64` 的计算 task 新增 `require_sync_start=true`，保留已有同步标记。block 数、kernel、输入、任务 ID、依赖及切片外前缀不变；单 block 和控制 task 不改。

超过 64 blocks 的任务不新增同步：

- MoE：`dispatch_push`，源任务 57，128 个 AIV blocks。
- Engram：`engram_widen`，源任务 88，500 个 AIV blocks。
- layers_19_21：`dispatch_push`，源任务 1362、1442、1506，各 128 个 AIV blocks。

9 组按单 rank 分别计数后相加，280 个多 block task 中 275 个带同步、5 个不带，相对原配置新增 211 个。MIX 算一个逻辑 task，不按 lane 重复，也不跨用例去重。AIC/MIX 任务均不超过 32 blocks。

用例生成命令见 [README.md](README.md#dsl-bounded-sync-start-variant)，入口为 [production/force_sync_start.py](production/force_sync_start.py)。
