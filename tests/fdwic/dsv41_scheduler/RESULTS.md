# DSV4.1 保留 20 组：同轮 L1 baseline

实测日期：**2026-11-13**。20 组全部通过，每组 1 次 L1 预热、10 次 L1 正式测量，共 **200 个正式 case 样本**。以下只保留本轮算术平均值，单位 **ms**。

| Case | 配置 | device_wall | AICore 首尾跨度 | 零调度开销模型下界 |
| ---- | ---- | ----------- | --------------- | ------------------ |
| swa | DSL 原版 | 1.452544 | 1.218261 | 0.908975 |
| swa | DSL ≤64 sync_start | 1.739990 | 1.490957 | 0.956770 |
| swa | DSL Flat×8 | 10.322717 | 10.075760 | 7.385050 |
| swa | DSL Graph×8 | 10.878415 | 10.606722 | 7.630605 |
| c2a_full | DSL 原版 | 1.930454 | 1.693514 | 1.322808 |
| c2a_full | DSL ≤64 sync_start | 2.143626 | 1.892028 | 1.315141 |
| c1a_reuse | DSL Flat×8 | 13.149425 | 12.904257 | 10.555357 |
| c1a_reuse | DSL Graph×8 | 13.374676 | 13.096056 | 10.474066 |
| c1a_reindex | DSL 原版 | 1.917259 | 1.684010 | 1.301319 |
| c1a_reindex | DSL ≤64 sync_start | 2.076805 | 1.819921 | 1.300288 |
| moe | PyPTO-lib | 3.234042 | 2.958993 | N/A |
| moe | DSL 原版 | 1.890702 | 1.644616 | N/A |
| moe | DSL ≤64 sync_start | 2.886706 | 2.594109 | N/A |
| moe | DSL Graph×1 | 2.015058 | 1.753182 | N/A |
| engram | DSL 原版 | 1.370925 | 1.125376 | 0.750876 |
| engram | DSL ≤64 sync_start | 1.618942 | 1.302128 | 0.751685 |
| layers_19_21 | PyPTO-lib | 12.633813 | 12.291015 | N/A |
| layers_19_21 | DSL 原版 | 10.307781 | 10.051822 | N/A |
| layers_19_21 | DSL ≤64 sync_start | 12.511116 | 12.239416 | N/A |
| layers_19_21 | DSL Graph×1 | 10.660937 | 10.335095 | N/A |

## 测试方式

- 源码基线：`hw-native-sys/simpler@a41c2c1a3`，HBG resident scheduler mode 1。该提交相对 `1e3e82fd6` 仅新增 benchmark skill；`src/`、`cmake/` 和 PTO pin 无差异。本轮复用已核验哈希的相同 Host/AICPU/AICore 运行库，重新构建 runner/replay，并重新执行全部 20 组。
- A5 Ascend950DT_9581，32 AIC、64 AIV；CANN 9.3.0、驱动 25.6.rc2.b023，默认 `-O3 -g`。原业务 kernel、固定零权重输入及参数签名保持不变：每 rank 72 个有效 token，补齐到 80 行，64 个本地 expert。
- **两列实测数据均来自同一次 L1 执行，包含 L1 采集开销。** 没有把 L0 的 device_wall 与另一轮 L1 的 AICore 时间拼在一起。这是一份 L1 baseline，不能直接与此前 L0 数值比较。
- **device_wall** 读取本轮 `chip.run.runner_run.device_wall`，覆盖 AICPU 与 AICore 协同执行的设备窗口；不含 Host 构建、上传、切片外前缀准备及预热。
- **AICore 首尾跨度** 在本 rank 的全部 AIC/AIV/MIX kernel 记录上计算 `(max(kernel_end) - min(kernel_start)) / clock_freq`。它包含首尾之间的调度空隙、依赖等待及通信等待，不是 kernel 时长之和，也不是纯算术时间；不覆盖首个 kernel 之前及末个 kernel 返回之后的启动、异步收尾等时间。
- 所有设备通过 `task-submit --device auto --device-num N` 自动分配。成功的单卡批次使用设备 1，双卡批次使用设备 1、2；同一 workload 的不同配置在同一次独占分配内顺序执行。单卡早期批次遇到执行前的驱动初始化失败，整批重跑；失败批次不进入本表。
- **双卡口径**：每轮先分别计算每个 rank 的 device_wall 和 AICore 首尾跨度，再对每个指标取两个 rank 的最大值，最后平均 10 轮。不跨卡相减绝对时间戳；两列的最大值可能来自不同 rank，因此差值不能解释为纯 AICPU 时间。
- `Flat×8`／`Graph×8` 计时覆盖完整八次串联执行，不除以 8；固定输入与 KV position 不推进。Graph 每次执行录制一份定义、提交八个实例。其他配置执行一个原始切片；MoE 和 layers_19_21 的 DSL 原版同时充当 Flat×1 对照，不重复计数。
- 同步变体仅对 `1 < block_num <= 64` 的计算任务增加 sync_start，保留已有标记。MoE 的 128-block dispatch、Engram 的 500-block widen、跨层中的 128-block dispatch 不新增同步。
- 每轮检查数值、预热标记、L1 级别、调用身份、物理执行数和 block/lane 覆盖。机器初始化失败、缺失记录或不完整 capture 都不能作为有效样本。完整运行清单见 [baseline_matrix.json](baseline_matrix.json)。

## 理论下界的定义与限制

依据 [PR #3061](https://github.com/hw-native-sys/simpler/pull/3061) 的要求，使用**实际依赖 DAG 和本轮 L1 kernel 时长**建立独立模型。这里给的是固定这些 kernel 时长条件下的乐观下界，不是硬件必然能达到的性能，也不是从 device_wall 中减掉一段测得的 scheduler 时间。

对每个任务，将它所有物理 kernel 执行的实测时长记为 `l`，取：

```text
d(task) = max(最大单次 kernel 时长,
              此 task 的 AIC 总工作量 / 32,
              此 task 的 AIV 总工作量 / 64)
finish(task) = d(task) + max(finish(实际前驱))
下界 = max(依赖关键路径长度,
           全部 AIC 总工作量 / 32,
           全部 AIV 总工作量 / 64)
```

模型将调度发布、抢块、组队、依赖交接等开销设为零，保留依赖关系和 AIC/AIV 核数造成的工作量下界；不构造可执行排程，也不精确求解 MIX/SPMD 成员协同、整数分波、内存带宽竞争及重排后的 cache 行为。kernel 内部已测得的等待仍包含在其时长中。Host 已完成的 allocation 经 source-task mapping 核验后作为零时长节点补入，不能把任意缺失前驱直接删掉。

普通任务使用该轮 L1 的 `deps.json`。两种 attention Graph 另采 L4 **仅提取结构**：核验一份定义、八个实例和 native DAG 展开，再投影到实际执行的 kernel。callable 与 plan 哈希必须匹配本轮 bundle；**L4 耗时不参与任何均值或模型权重**。

12 组单卡可闭合此模型。8 组双卡的 L1 没有完整的跨 rank 通知、SDMA 完成及 defer_wait 逻辑完成事件，无法把通信依赖闭合，因此按 PR #3061 要求标记 **N/A**。不能用等待 kernel 的返回替代异步完成，或假设通信零耗时后把结果称为整段理论极限。

输入仍是固定零权重 fixture；本表不能证明任意 checkpoint 的数值精度或所有动态 routing／异步时序的正确性。

## 复现与分析

按 [README.md](README.md) 准备原版 bundle，按 [GRAPH.md](GRAPH.md) 构造 Graph 配对及 Flat×8，按既有 `force_sync_start.py` 构造同步变体。用 [baseline_matrix.json](baseline_matrix.json) 的 `id` 为每个选定 bundle 和输出目录命名。

在自动分配的同一拓扑设备组内执行各配置：

```bash
bash tests/fdwic/dsv41_scheduler/production/run.sh \
  BUNDLE RUNTIME OUTPUT 1 1 10
```

对于两组单卡 Graph，额外运行 `4 0 1` 采集结构。投影命令为：

```bash
PYTHONPATH=.:python python3 tests/fdwic/dsv41_scheduler/production/graph_topology.py \
  BUNDLE L4_OUTPUT/rank0/run0 TOPOLOGY/CASE_ID.json
```

其中 `CASE_ID` 为 `swa--graph8` 或 `c1a_reuse--graph8`。使用完整矩阵生成可审计的逐轮结果：

```bash
PYTHONPATH=.:python python3 tests/fdwic/dsv41_scheduler/production/baseline_report.py \
  BUNDLES CAPTURES TOPOLOGY OUTPUT/results.json
```

`BUNDLES/CASE_ID` 是用例目录，`CAPTURES/CASE_ID/rankN` 是该次采集目录。工具拒绝非 L1 样本、跨调用配对、未核验的 Graph 拓扑、缺失依赖、执行数错误及不同设备上的同 workload 对照。原始样本、capture、结构投影和构建证据留在本机，仓库只保留当前均值表。
