# DSV4.1 16 组 L0 AICore 性能与同步约束下界

**单位：ms，平均值。** 16 组 HBG 与 percounter 均在本轮重新测量，使用同卡配对的 L0 AICore wall；同步约束下界沿用原模型。

| Case | 实现 | main HBG | main percounter | 同步约束下界 | 最优实测距下界 |
| ---- | ---- | -------: | --------------: | -----------: | -------------: |
| swa | 新版 PyPTO-lib | 0.317110 | 0.236491 | 0.134880 | 75.33% |
| c2a_full | 初版 PyPTO-lib | 1.183605 | 1.095765 | 0.843450 | 29.91% |
| c2a_full | DSL 半同步 | 1.579532 | 1.701135 | 1.341990 | 17.70% |
| c2a_full | DSL 全同步≤64 | 1.594847 | 2.116128 | 1.354400 | 17.75% |
| c2a_full | 新版 PyPTO-lib | 0.405392 | 0.311436 | 0.198030 | 57.27% |
| c1a_reindex | 新版 PyPTO-lib | 0.398168 | 0.293678 | 0.189260 | 55.17% |
| c1a_reuse | 新版 PyPTO-lib | 0.306163 | 0.236167 | 0.149950 | 57.50% |
| moe | 初版 PyPTO-lib | 3.427724 | 3.080944 | — | — |
| moe | DSL 半同步 | 1.515188 | 1.807762 | — | — |
| moe | DSL 全同步≤64 | 1.557589 | 1.895810 | — | — |
| moe | 新版 PyPTO-lib | 0.839739 | 0.903374 | — | — |
| engram | 新版 PyPTO-lib | 0.992289 | 0.958232 | 0.732110† | 30.89%† |
| layers_19_21 | 初版 PyPTO-lib | 14.474301 | 15.631164 | — | — |
| layers_19_21 | DSL 半同步 | 9.247935 | 10.795979 | — | — |
| layers_19_21 | DSL 全同步≤64 | 9.501243 | 12.176530 | — | — |
| layers_19_21 | 新版 PyPTO-lib | 3.749960 | 4.597247 | — | — |

## 版本与测试方式

- 固定本地测试版本 `6d050abbd10750bc02eeb6641afc800891ac0dab`；两列运行库源码均与该提交一致。复用已核对源码及二进制摘要的 HBG、percounter 构建，以及同一批业务 kernel 和输入。表头保留 `main HBG` / `main percounter`，并不表示两列运行库来自 GitHub 当前 main。
- 纯 L0：每个运行库1次预热、10次正式测量。AICore wall 为首个实际 AIC/AIV kernel 开始至最后一个实际 kernel 结束，包含中间调度、依赖与通信等待，不包含 AICPU 启动和收尾。Device wall 仅作诊断，不填入表格。
- 每个用例的两列在同一张卡或同一对卡上顺序测量，用例间交替先测 HBG 或 percounter。双卡每次执行前均完成跨 Device 启动会合；逐轮取两卡各自 AICore 时长的最大值，再取10轮平均，不跨卡相减原始计数器。
- 全程通过 `task-submit --device auto --device-num N` 自动分配，最多8卡并发。测量日期2026-11-15，A5 Ascend950DT_9581（32 AIC / 64 AIV），CANN 9.3.0。本轮32个运行库/用例组合全部完成，总墙钟时间105.21秒（含提交、排队和回收）。实际设备号与双卡会合证据记录在本地 `artifacts/performance-refresh/summary.json` 及各运行的 `wall.json`。
- 新版模型固定为 [dev/hbg@a6a9a2d49047](https://github.com/hw-native-sys/pypto-lib/tree/a6a9a2d49047ce67a85857c6b15f9604a7287714/models/deepseek_v4_1_flash)。新版 MoE 执行原始 Host 编排及实际 `get_tensor_data<int32_t>(num_tokens)`，保留原非零权重和强制路由，只测 `72/72`；按性能测试定位跳过数值 golden 与 `0/6`、`6/0` 功能检查，结果标记 `numerical_validation: false`。权重缓存仅保存在本地忽略目录。
- 其余15组保留固定回放输入及原有结果校验。回放覆盖捕获控制值下的设备任务和同步通信，不覆盖动态 Host tensor 读取；新版 MoE 的 `recv_count_local` 和 route 仍由 AICore task 读取。跨实现组的变化同时包含算子、融合和缓存布局差异。
- 本表刷新的是 AICore wall；旧表使用包含 AICPU 的 Device wall，两者数值不能直接解释为调度加速比例。

## 同步约束下界

- 8个非通信配置沿用各自已计算的下界，本轮不重采 L1、不重求解。新版下界来自历史 HBG 534c L1，初版/DSL 来自各自历史21d2 L1；半同步和全同步使用独立模型与耗时。用例、kernel、依赖和同步策略保持不变。
- 模型保留任务完成依赖、32 AIC / 64 AIV 容量、不可抢占 block、MIX 同 block 共同启动、sync_start 全 block 共同启动；调度成本为零，放宽 cluster 放置。时长向下量化为100ns，取10轮已证明下界的平均值。
- 下界依赖历史 L1 kernel 时长，是同步约束下的参考模型，不是硬件绝对极限，也不保证是本轮 L0 的严格下界。通信场景不计算。
- 最优实测距下界 = `(min(main HBG, main percounter) / 同步约束下界 - 1) × 100%`，按未舍入值计算，不截断负值。
- **†**：新版 Engram 的10轮未证明最优；平均已证明下界0.732110ms、可行排程0.735090ms，表中采用前者。其他7个配置均已证明全部轮次最优。
- 用例和复现方式见 [README](README.md)，模型见 [BOUND.md](BOUND.md)。
