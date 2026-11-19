# DSV4.1 12 组 L0 AICore 性能与同步约束下界

**单位：ms，平均值。** HBG/percounter 保留2026-11-16上一轮12组结果；本轮新增11组 AICPU TMR 实测，新版原生 MoE 不测。同步约束下界沿用原值。

| Case | 实现 | main HBG | main percounter | AICPU TMR | 同步约束下界 | 最优实测距下界 |
| ---- | ---- | -------: | --------------: | --------: | -----------: | -------------: |
| swa | 新版 PyPTO-lib | 0.293834 | 0.183123 | 0.478452 | 0.134880 | 35.77% |
| c2a_full | 初版 PyPTO-lib | 1.159377 | 0.966131 | 1.315182 | 0.843450 | 14.55% |
| c2a_full | DSL 半同步 | 1.521647 | 1.483052 | 1.589281 | 1.341990 | 10.51% |
| c2a_full | DSL 全同步≤64 | 1.603249 | 1.520360 | 1.659525 | 1.354400 | 12.25% |
| c2a_full | 新版 PyPTO-lib | 0.371800 | 0.250790 | 0.563781 | 0.198030 | 26.64% |
| c1a_reindex | 新版 PyPTO-lib | 0.378103 | 0.244629 | 0.564632 | 0.189260 | 29.26% |
| c1a_reuse | 新版 PyPTO-lib | 0.281126 | 0.192529 | 0.457989 | 0.149950 | 28.40% |
| moe | DSL 半同步 | 1.450595 | 1.285974 | 1.572825 | — | — |
| moe | 新版 PyPTO-lib | 0.777624 | 0.550216 | — | — | — |
| engram | 新版 PyPTO-lib | 0.980122 | 0.847579 | 1.188929 | 0.732110† | 15.77%† |
| layers_19_21 | DSL 半同步 | 8.981423 | 8.180093 | 9.519084 | — | — |
| layers_19_21 | 新版 PyPTO-lib | 3.395367 | 2.307145 | 4.499391 | — | — |

## 版本与测试方式

- 测试源码 `f1d91abdb80a1ea455e3239b4d060003170d693d`：当前本地 HBG `3730c4be060a5abc442250d37a1aa89f66afcc7d`，合入 [PR #3114](https://github.com/hw-native-sys/simpler/pull/3114) 的 `a0b90c0be790ddf445e11f50f118f8e37a28f5a5`，并保留独立 MoE 测试入口。两套运行库从这一固定源码构建；业务 kernel 和输入复用原包。表头保留 `main HBG` / `main percounter`，不表示 GitHub 当前 main。
- TMR 列来自带首尾采集的测试运行库：AICPU 回放固定任务流，补充 L0 AICore 首尾采集。采集现已移到 `tmr_runtime.py` 创建的独立源码副本和输出目录；普通 TMR 源码及默认构建不包含这些改动。本次保留原11组数据，隔离构建另做验证。按 TMR Tensor ABI 重编实际使用的 kernel，数值计算源和输入不变。11组全部通过原有回放数值校验。TMR 另批次自动分配设备，三列不是同卡三方配对；不能把差值全部归因于调度算法。
- 纯 L0：每个运行库1次预热、10次正式测量。AICore wall 为首个实际 AIC/AIV kernel 开始至最后一个实际 kernel 结束，包含中间调度、依赖与通信等待，不包含 AICPU 启动和收尾。Device wall 仅作诊断，不填入表格。
- 每个用例的 HBG 与 percounter 在同一张卡或同一对卡上顺序测量，用例间交替执行顺序。双卡每次执行前均完成跨 Device 启动会合；逐轮取两卡各自 AICore 时长的最大值，再取10轮平均，不跨卡相减原始计数器。
- 全程通过 `task-submit --device auto --device-num N` 自动分配，最多8卡并发。A5 Ascend950DT_9581（32 AIC / 64 AIV），CANN 9.3.0。本轮12组、两套运行库的配对测试共耗时220.81秒（包含排队，不含构建）。实际设备号、源码/运行库摘要和双卡会合证据保存在本地 `artifacts/pr3114-refresh/summary.json`、`build-audit.json`、`paired-state.json` 与各用例的 `wall.json`。
- 新版模型固定为 [dev/hbg@a6a9a2d49047](https://github.com/hw-native-sys/pypto-lib/tree/a6a9a2d49047ce67a85857c6b15f9604a7287714/models/deepseek_v4_1_flash)。新版 MoE 执行原始 Host 编排及实际 `get_tensor_data<int32_t>(num_tokens)`，保留原非零权重和强制路由，只测 `72/72`；按性能测试定位跳过数值 golden 与 `0/6`、`6/0` 功能检查，结果标记 `numerical_validation: false`。权重缓存仅保存在本地忽略目录。
- 表内其余11组保留固定回放输入及原有结果校验。回放覆盖捕获控制值下的设备任务和同步通信，不覆盖动态 Host tensor 读取；新版 MoE 的 `recv_count_local` 和 route 仍由 AICore task 读取。跨实现组的变化同时包含算子、融合和缓存布局差异。
- TMR 设备号、逐轮时长和双卡会合证据保存在本地 `artifacts/tmr/summary.json` 和 `tmr-fixed/*/wall.json`。测试专用端点采集尚未单独完成小于1%的开销验收，表内 TMR 时间包含该采集；普通 TMR 不承担这部分开销。
- 本表记录的是 AICore wall；更早的旧表使用包含 AICPU 的 Device wall，两者数值不能直接解释为调度加速比例。

## 同步约束下界

- 8个非通信配置沿用各自已计算的下界，本轮不重采 L1、不重求解。新版下界来自历史 HBG 534c L1，初版/DSL 来自各自历史21d2 L1；半同步和全同步使用独立模型与耗时。用例、kernel、依赖和同步策略保持不变。
- 模型保留任务完成依赖、32 AIC / 64 AIV 容量、不可抢占 block、MIX 同 block 共同启动、sync_start 全 block 共同启动；调度成本为零，放宽 cluster 放置。时长向下量化为100ns，取10轮已证明下界的平均值。
- 下界依赖历史 L1 kernel 时长，是同步约束下的参考模型，不是硬件绝对极限，也不保证是本轮 L0 的严格下界。通信场景不计算。
- 最优实测距下界 = `(min(main HBG, main percounter, 可用的 AICPU TMR) / 同步约束下界 - 1) × 100%`，按未舍入值计算，不截断负值。
- **†**：新版 Engram 的10轮未证明最优；平均已证明下界0.732110ms、可行排程0.735090ms，表中采用前者。其他7个配置均已证明全部轮次最优。
- 用例和复现方式见 [README](README.md)，模型见 [BOUND.md](BOUND.md)。
