# DSV4.1 18 组用例 L0 baseline 与理论极限

9 个 workload × 两种实现，共 **18 组用例**。HBG 全部 18 项及 percounter 单卡 14 项通过数值校验；percounter 的 4 项多卡配置暂不支持。单位 **ms**，仅列平均值。

| Case | 实现 | HBG | percounter | 理论极限（模型下界） |
| ---- | ---- | --- | ---------- | -------------------- |
| swa | PyPTO-lib | 0.813843 | 0.580336 | 0.388000 |
| swa | DSL 半数同步 | 1.357375 | 1.155841 | 0.918104 |
| c2a_full | PyPTO-lib | 1.406484 | 1.218788 | 0.840052 |
| c2a_full | DSL 半数同步 | 1.836173 | 1.799990 | 1.299201 |
| c2a_reuse | PyPTO-lib | 1.183351 | 0.970064 | 0.771683 |
| c2a_reuse | DSL 半数同步 | 1.778984 | 1.583119 | 1.365545 |
| c1a_full | PyPTO-lib | 1.389396 | 1.105468 | 0.868324 |
| c1a_full | DSL 半数同步 | 1.825318 | 1.711340 | 1.336579 |
| c1a_reindex | PyPTO-lib | 1.369398 | 1.086902 | 0.847620 |
| c1a_reindex | DSL 半数同步 | 1.881647 | 1.667816 | 1.290988 |
| c1a_reuse | PyPTO-lib | 1.151947 | 0.942275 | 0.739888 |
| c1a_reuse | DSL 半数同步 | 1.744648 | 1.564304 | 1.364733 |
| moe | PyPTO-lib | 2.592322 | 不支持（多卡通信） | 不计算（通信） |
| moe | DSL 半数同步 | 1.802757 | 不支持（多卡通信） | 不计算（通信） |
| engram | PyPTO-lib | 1.130147 | 1.022857 | 0.744347 |
| engram | DSL 半数同步 | 1.192794 | 1.041565 | 0.751891 |
| layers_19_21 | PyPTO-lib | 10.752142 | 不支持（多卡通信） | 不计算（通信） |
| layers_19_21 | DSL 半数同步 | 10.272117 | 不支持（多卡通信） | 不计算（通信） |

## 测试方式

- 测试日期：2026-11-13；runtime 源码固定为 `hw-native-sys/main@60a2ba4cf`，HBG 和 percounter 均由该提交重新编译，runtime 源码无本地修改。
- 指标为 `chip.run.runner_run.device_wall`，包含 AICPU 调度、AICore 执行及 case 内通信；不包含 Host 准备、前缀准备和预热。双卡先完成两边 Host 准备再共同启动；每轮取两个 rank 较大值，再计算平均值。
- **纯 L0，1 次预热、10 次正式测量**；不启用逐任务计时或泳道采样，不包含独立 AICore 首尾跨度。HBG 实测模式为 1；percounter 单独加载对应库，其 Host 模式查询值 0 表示未提供观察值。
- 两种 runtime 共用相同 bundle、业务 kernel 和输入，在同一次自动分配的设备上配对运行；各 workload 交替采用 HBG→percounter 或 percounter→HBG 的执行顺序。
- 使用 `task-submit --device auto --device-num N`；本轮单卡实际分配设备 3，双卡设备 4、5，执行与结果记录均使用 `TASK_DEVICE`。
- A5 Ascend950DT_9581（32 AIC / 64 AIV）、CANN 9.3.0、驱动 25.6.rc2.b023，runtime 默认 `-O3 -g`；保留原始业务 kernel 二进制与地址偏移。
- 固定零权重输入，每 rank 72 个有效 token、补齐至 80 行、64 个本地 expert。结论仅覆盖本组输入与配置。
- DSL 保留原有同步标记，按每个 case 最终约半数 SPMD 同步：合计 280 个任务中 143 个带同步，新增 79 个；5 个超容量任务保持不变。生成规则见 [README.md](README.md#dsl-half-spmd-sync-start-policy)。
- percounter 多卡通信尚未支持，MoE 和 layers_19_21 的两种实现不提供平均耗时；错误运行和参考/前缀阶段均不计入成功样本。

## 理论极限列的口径

- 按 [PR #3061](https://github.com/hw-native-sys/simpler/pull/3061) 的实际依赖与 kernel 耗时建模原则，使用相同 bundle、相同 `60a2ba4cf` HBG runtime，单独采集 14 个单卡用例的 L1；1 次预热、10 次正式采样，自动分配的实际设备为 2。**两列 runtime 实测值仍是原来的 L0 数据**。
- 对每个 task，`d = max(最长 block/lane 耗时, AIC 总执行耗时 / 32, AIV 总执行耗时 / 64)`；按实际完成依赖 DAG 计算以 `d` 为权重的关键路径 `CP`。每轮模型下界为 `max(CP, 全图 AIC 总执行耗时 / 32, 全图 AIV 总执行耗时 / 64)`，表中取 10 轮的平均值。
- 假设调度、发布、启动同步及收尾成本为零，放宽资源碎片和额外并发争用；kernel 内已经发生的等待仍计入该 kernel 的实测时长。这是**固定实测 kernel 耗时条件下的模型下界**，不是已实现的调度，也不是硬件吞吐极限。L1 采样和缓存／并发条件会影响输入时长，因此它不保证是另一次 L0 测量的严格下界，不做 L0/L1 比例换算。
- MoE、layers_19_21 的通信场景均不计算。每次计算均检查 block/lane 完整性、跨轮拓扑一致性、依赖端点与无环性，并要求下界不超过该轮 AICore kernel 首尾跨度。详细公式及复算入口见 [README.md](README.md#conditional-ideal-lower-bound)。
