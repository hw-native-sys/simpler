# DSV4.1 16 组 L0 AICore 性能与同步约束下界

**单位：ms，平均值。** 16 组 HBG/percounter 实测值全部刷新为 **L0 AICore wall**，
即每卡第一个实际 kernel begin 到最后一个实际 kernel end；包含中间的调度、等待和通信空洞。
两列采用下述固定运行库版本。保留初版 PyPTO-lib 3 组、DSL 半同步 3 组、
DSL 全同步≤64 3 组、新版 PyPTO-lib 7 组；8 个非通信配置的下界沿用原值。

| Case | 实现 | main HBG | main percounter | 同步约束下界 | 最优实测距下界 |
| ---- | ---- | -------: | --------------: | -----------: | -------------: |
| swa | 新版 PyPTO-lib | 0.321653 | 0.235504 | 0.134880 | 74.60% |
| c2a_full | 初版 PyPTO-lib | 1.187823 | 1.089796 | 0.843450 | 29.21% |
| c2a_full | DSL 半同步 | 1.570328 | 1.717217 | 1.341990 | 17.01% |
| c2a_full | DSL 全同步≤64 | 1.602462 | 2.148534 | 1.354400 | 18.32% |
| c2a_full | 新版 PyPTO-lib | 0.408528 | 0.303815 | 0.198030 | 53.42% |
| c1a_reindex | 新版 PyPTO-lib | 0.401922 | 0.293938 | 0.189260 | 55.31% |
| c1a_reuse | 新版 PyPTO-lib | 0.306594 | 0.234839 | 0.149950 | 56.61% |
| moe | 初版 PyPTO-lib | 3.442943 | 2.977887 | — | — |
| moe | DSL 半同步 | 1.523743 | 1.846731 | — | — |
| moe | DSL 全同步≤64 | 1.559213 | 1.896224 | — | — |
| moe | 新版 PyPTO-lib | 0.834525 | 0.878857 | — | — |
| engram | 新版 PyPTO-lib | 1.008130 | 0.952684 | 0.732110† | 30.13%† |
| layers_19_21 | 初版 PyPTO-lib | 14.332172 | 15.422858 | — | — |
| layers_19_21 | DSL 半同步 | 9.263651 | 10.759697 | — | — |
| layers_19_21 | DSL 全同步≤64 | 9.534265 | 12.042482 | — | — |
| layers_19_21 | 新版 PyPTO-lib | 3.769646 | 4.573585 | — | — |

## 版本与测试方式

- `main HBG`：`2481bd9431437b665b2ea81c786eef800ea33ab8`；`main percounter`：`fd4d4863a6dfc9386d43dbcc75067948aae34bc7`。
  两者均采用实际 kernel 首尾窗口。HBG 复用与目标提交的 903 个构建相关文件一致的运行库；
  percounter 仅重编运行库，所有业务 kernel 与输入均复用。
- 全部为 L0。HBG 每组 5 次预热、10 次正式测量；percounter 采用已完成的采集开关对照中
  4 个开启批次的全部 40 次正式测量，每批预热 5 次。两列不是同轮、同卡的配对 A/B。
  正式表不含 AICPU 头尾耗时或 L1/L3 插桩耗时。
- 所有双卡运行先在同一 Device stream 上完成双方 epoch rendezvous，再启动被测运行；
  起跑同步不计入窗口。每轮分别计算两卡时长，取较大值后再平均，不跨卡相减原始计数器。
- 使用 `task-submit --device auto --device-num N` 自动分配。HBG 单卡使用设备 2–7、双卡使用
  0/1、2/3、4/5、6/7，最大并发 8 卡；percounter 单卡使用 0/1，双卡使用 0/1、2/3，最大并发 4 卡。
  逐例实际 `TASK_DEVICE` 与原始计数器记录在本地测量结果中。
- 测量日期为 2026-11-15，A5 Ascend950DT（32 AIC / 64 AIV），CANN 9.3.0。
  两列各 16 组全部通过 numerical golden、时间戳及双卡同步检查。
  初始化失败和未完成的对照单独保存，不混入均值。
- percounter 的采集开关对照中，15/16 组 Device wall 平均膨胀低于 1%；
  **新版原生 MoE 的均值差异为 +3.365%，尚未满足 1% 门槛**。该行保留实际测量值，
  不能宣称 percounter 的全量低开销验收已经通过。
- 新版模型固定为 [dev/hbg@a6a9a2d49047](https://github.com/hw-native-sys/pypto-lib/tree/a6a9a2d49047ce67a85857c6b15f9604a7287714/models/deepseek_v4_1_flash)。
  新版 MoE 继承 [PR3038](https://github.com/hw-native-sys/simpler/pull/3038)，执行原生 Host
  `get_tensor_data<int32_t>(num_tokens)` 路径；两种运行库均通过 `0/6`、`6/0`、`72/72`
  三组 token 数和三个输出的原生 golden 校验，性能采用 `72/72`。
  该用例使用非零权重与强制路由输入，其余用例沿用冻结输入。
  `recv_count_local` 与 route 的设备侧读取仍在 AICore task 内。

## 同步约束下界

- 8 个非通信配置沿用各自已计算的下界，本轮未重新测算：新版来自 HBG 534c 的 L1 kernel 时长，
  初版和 DSL 来自历史 21d2 的对应 L1 时长。半同步与全同步配置分别建模，不共用时长或下界。
- 模型保留任务完成依赖、32 AIC / 64 AIV 容量、不可抢占 block、MIX 同 block 共同启动，
  以及 `sync_start` 全 block 共同启动；调度成本为零，放宽 cluster 放置。
  时长向下量化为 100ns，采用 10 轮已证明下界的平均值。通信场景不计算。
- 下界依赖历史 L1 kernel 时长，不是硬件绝对极限，也不保证是本轮 L0 的严格下界。
  表中差距仅反映当前实测值与该模型下界的比例。
- 最优实测距下界 = `(min(main HBG, main percounter) / 同步约束下界 - 1) × 100%`，
  使用未舍入的实测均值计算，保留两位小数，不截断负值。
- **†**：新版 Engram 的 10 轮未证明最优；平均已证明下界为 0.732110ms，可行排程为 0.735090ms，
  表中使用前者。其他 7 个非通信配置均已证明全部轮次最优。
- 用例与复现方式见 [README](README.md)，模型见 [BOUND.md](BOUND.md)。
