# DSV4.1 12 组 L0 AICore 性能与同步约束下界

**单位：ms，平均值。** 12 组 HBG 与 percounter 均在本轮重新测量，使用同卡配对的 L0 AICore wall；同步约束下界沿用原模型。

| Case | 实现 | main HBG | main percounter | 同步约束下界 | 最优实测距下界 |
| ---- | ---- | -------: | --------------: | -----------: | -------------: |
| swa | 新版 PyPTO-lib | 0.326833 | 0.230598 | 0.134880 | 70.97% |
| c2a_full | 初版 PyPTO-lib | 1.176845 | 1.090656 | 0.843450 | 29.31% |
| c2a_full | DSL 半同步 | 1.544502 | 1.692580 | 1.341990 | 15.09% |
| c2a_full | DSL 全同步≤64 | 1.562037 | 2.116053 | 1.354400 | 15.33% |
| c2a_full | 新版 PyPTO-lib | 0.401383 | 0.298097 | 0.198030 | 50.53% |
| c1a_reindex | 新版 PyPTO-lib | 0.408772 | 0.285832 | 0.189260 | 51.03% |
| c1a_reuse | 新版 PyPTO-lib | 0.318407 | 0.228398 | 0.149950 | 52.32% |
| moe | DSL 半同步 | 1.472711 | 1.830846 | — | — |
| moe | 新版 PyPTO-lib | 0.797924 | 0.851918 | — | — |
| engram | 新版 PyPTO-lib | 0.982261 | 0.932947 | 0.732110† | 27.43%† |
| layers_19_21 | DSL 半同步 | 8.950110 | 10.701516 | — | — |
| layers_19_21 | 新版 PyPTO-lib | 3.457652 | 4.637452 | — | — |

## 版本与测试方式

- 固定 fwic-graph 的本地 base `5abd1794980eaa493f618067b91aeff117191a63`，合入 [PR #3108](https://github.com/hw-native-sys/simpler/pull/3108) 的 `94c17ffb245ee7712f3d9343bb7ddbc3292485c9`；本轮测试提交为 `a169dff97217113a0cc0583deb1a1ce260c0ae2c`。HBG 使用该 base，percounter 使用 base 加 PR #3108，两列均与此本地源码一致。HBG 复用已核对源码及二进制摘要的构建；percounter 按合入后的源码构建。两列复用同一批业务 kernel 和输入。表头保留 `main HBG` / `main percounter`，并不表示两列运行库来自 GitHub 当前 main。
- 纯 L0：每个运行库1次预热、10次正式测量。AICore wall 为首个实际 AIC/AIV kernel 开始至最后一个实际 kernel 结束，包含中间调度、依赖与通信等待，不包含 AICPU 启动和收尾。Device wall 仅作诊断，不填入表格。
- 每个用例的两列在同一张卡或同一对卡上顺序测量，用例间交替先测 HBG 或 percounter。双卡每次执行前均完成跨 Device 启动会合；逐轮取两卡各自 AICore 时长的最大值，再取10轮平均，不跨卡相减原始计数器。
- 全程通过 `task-submit --device auto --device-num N` 自动分配，最多8卡并发。测量日期2026-11-16，A5 Ascend950DT_9581（32 AIC / 64 AIV），CANN 9.3.0。本轮24个运行库/用例组合全部完成，总墙钟时间135.11秒（含提交、排队和回收）。实际设备号与双卡会合证据记录在本地 `artifacts/pr3108-refresh/summary.json` 及各运行的 `wall.json`。
- 新版模型固定为 [dev/hbg@a6a9a2d49047](https://github.com/hw-native-sys/pypto-lib/tree/a6a9a2d49047ce67a85857c6b15f9604a7287714/models/deepseek_v4_1_flash)。新版 MoE 执行原始 Host 编排及实际 `get_tensor_data<int32_t>(num_tokens)`，保留原非零权重和强制路由，只测 `72/72`；按性能测试定位跳过数值 golden 与 `0/6`、`6/0` 功能检查，结果标记 `numerical_validation: false`。权重缓存仅保存在本地忽略目录。
- 其余11组保留固定回放输入及原有结果校验。回放覆盖捕获控制值下的设备任务和同步通信，不覆盖动态 Host tensor 读取；新版 MoE 的 `recv_count_local` 和 route 仍由 AICore task 读取。跨实现组的变化同时包含算子、融合和缓存布局差异。
- percounter 的6个 AICore 编译单元通过 IR 检查；31个 payload 与3个 scalar-memory CPU 测试通过。12组硬件运行的两列均完成；原生 MoE 仍为仅性能验证。

## 同步约束下界

- 8个非通信配置沿用各自已计算的下界，本轮不重采 L1、不重求解。新版下界来自历史 HBG 534c L1，初版/DSL 来自各自历史21d2 L1；半同步和全同步使用独立模型与耗时。用例、kernel、依赖和同步策略保持不变。
- 模型保留任务完成依赖、32 AIC / 64 AIV 容量、不可抢占 block、MIX 同 block 共同启动、sync_start 全 block 共同启动；调度成本为零，放宽 cluster 放置。时长向下量化为100ns，取10轮已证明下界的平均值。
- 下界依赖历史 L1 kernel 时长，是同步约束下的参考模型，不是硬件绝对极限，也不保证是本轮 L0 的严格下界。通信场景不计算。
- 最优实测距下界 = `(min(main HBG, main percounter) / 同步约束下界 - 1) × 100%`，按未舍入值计算，不截断负值。
- **†**：新版 Engram 的10轮未证明最优；平均已证明下界0.732110ms、可行排程0.735090ms，表中采用前者。其他7个配置均已证明全部轮次最优。
- 用例和复现方式见 [README](README.md)，模型见 [BOUND.md](BOUND.md)。
