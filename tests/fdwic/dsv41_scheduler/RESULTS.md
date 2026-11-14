# DSV4.1 16 组 L0 性能与同步约束下界

**单位：ms，平均值。** 保留初版 PyPTO-lib 3 组、DSL 半同步 3 组、DSL 全同步≤64 3 组、新版 PyPTO-lib 7 组。新版 MoE 已改为原始 Host 编排并重测，其余15行数值原样保留。

| Case | 实现 | main HBG | main percounter | 同步约束下界 | 最优实测距下界 |
| ---- | ---- | -------: | --------------: | -----------: | -------------: |
| swa | 新版 PyPTO-lib | 0.517646 | 0.345087 | 0.134880 | 155.85% |
| c2a_full | 初版 PyPTO-lib | 1.542043 | 1.196598 | 0.843450 | 41.87% |
| c2a_full | DSL 半同步 | 1.823954 | 1.771803 | 1.341990 | 32.03% |
| c2a_full | DSL 全同步≤64 | 1.723550 | 2.210183 | 1.354400 | 27.26% |
| c2a_full | 新版 PyPTO-lib | 0.696364 | 0.415687 | 0.198030 | 109.91% |
| c1a_reindex | 新版 PyPTO-lib | 0.666053 | 0.406814 | 0.189260 | 114.95% |
| c1a_reuse | 新版 PyPTO-lib | 0.510652 | 0.345089 | 0.149950 | 130.14% |
| moe | 初版 PyPTO-lib | 7.635917 | 3.207300 | — | — |
| moe | DSL 半同步 | 2.189010 | 1.999205 | — | — |
| moe | DSL 全同步≤64 | 1.680310 | 2.007643 | — | — |
| moe | 新版 PyPTO-lib | 3.134032 | 1.754210 | — | — |
| engram | 新版 PyPTO-lib | 1.142827 | 1.032040 | 0.732110† | 40.97%† |
| layers_19_21 | 初版 PyPTO-lib | 116.065531 | 15.601330 | — | — |
| layers_19_21 | DSL 半同步 | 17.297568 | 10.843851 | — | — |
| layers_19_21 | DSL 全同步≤64 | 10.592131 | 12.271256 | — | — |
| layers_19_21 | 新版 PyPTO-lib | 13.902196 | 4.603234 | — | — |

## 版本与测试方式

- `main HBG` 对应 `534c5656aede204d270d214656182cb1e3e2f81e`；`main percounter` 对应 `21d2e9a103d5b7372dda4823fb37eab470544d0c`。运行库与已编译业务 kernel 均复用；新版 MoE 复用了27个 kernel 二进制，并按匹配的 Host ABI 重新打包原始编排。
- 新版模型固定为 [dev/hbg@a6a9a2d49047](https://github.com/hw-native-sys/pypto-lib/tree/a6a9a2d49047ce67a85857c6b15f9604a7287714/models/deepseek_v4_1_flash)。新版 MoE 继承 [PR3038](https://github.com/hw-native-sys/simpler/pull/3038)：执行真实 `get_tensor_data<int32_t>(num_tokens)`，同一编译产物在 HBG、percounter 上均通过 `0/6`、`6/0`、`72/72` 三组 token 数和三个输出的原生 numerical golden 校验。
- 纯 L0：1次预热、10次正式测量，指标为包含 AICPU/AICore 的 Device wall；双卡逐轮取两个 rank 的较大值再平均。新版 MoE 另保存 Host 时间及每 rank 样本；Host 控制读取发生于构图，不计入 Device wall。正式性能列不含 L1 插桩。
- 新版 MoE 使用原模型的非零权重及随 token 数变化的强制路由 golden 输入。因此这行是业务输入更新后的基线，不能将它与原冻结零权重 MoE 的差值解释为调度退化。其他15行仍使用原来的固定零权重输入与测量数据。
- 新版 MoE 本轮通过自动分配的两组双卡并发验证：HBG 的 `TASK_DEVICE=0,1`，percounter 为 `2,5`；不是同卡对 A/B。所有用卡均经 `task-submit --device auto --device-num N`。测量日期2026-11-14，A5 Ascend950DT_9581（32 AIC / 64 AIV），CANN 9.3.0 环境。
- 其他保留项沿用先前数据：新版非 MoE 的 HBG/percounter 为同卡配对测量；初版和 DSL 的两列并非同轮配对。跨实现组的变化同时包含算子、融合和缓存布局差异。
- 其余回放用例保留的是捕获控制值下的设备任务与同步通信行为，不覆盖动态 Host tensor 读取。`recv_count_local` 和 route 在新版 MoE 中仍由 AICore task 读取。

## 同步约束下界

- 当前8个非通信配置各自保留已计算的下界：新版采用其 HBG 534c L1 时长，初版/DSL 采用对应的历史21d2 L1时长；不同同步配置不共用模型或耗时。
- 模型保留任务完成依赖、32 AIC / 64 AIV 容量、不可抢占 block、MIX 同 block 共同启动、sync_start 全 block 共同启动；调度成本为零，放宽 cluster 放置。时长向下量化为100ns，取10轮已证明下界的平均值。
- 下界依赖 L1 kernel 时长，不是硬件绝对极限，也不保证是另一轮 L0 的严格下界。通信场景不计算。
- 最优实测距下界 = `(min(main HBG, main percounter) / 同步约束下界 - 1) × 100%`，按未舍入值计算，不截断负值。
- **†**：新版 Engram 的10轮未证明最优；平均已证明下界0.732110ms、可行排程0.735090ms，表中采用前者。其他7个保留配置均已证明全部轮次最优。
- 用例选择、原生 MoE 契约和复现方式见 [README](README.md)，模型见 [BOUND.md](BOUND.md)。
