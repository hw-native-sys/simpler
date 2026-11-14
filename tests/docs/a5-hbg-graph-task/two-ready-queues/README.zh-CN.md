# 两个 ready 队列：A5 实测与验收记录（2026-11-11）

本轮实现固定 AIC（含所有 MIX）和 AIV 两条业务 ready 队列，删除业务分片、哈希路由和 stealing。
十八种模型功能检查通过；十四种单卡场景 L1 中位数之和从 28.4497 ms
变为 28.2899 ms（-0.56%），按 ±5% 规则属于无明显变化。
本轮确认的是队列结构简化，**不能记作 Device 性能优化收益，也不能据此声称达到 50 µs→5 µs 的 SPMD 切换目标**。
本页记录被测提交的两队列设计与范围；当前执行机制见[目录与 execution token](../../../../docs/zh-cn/design/a5-hbg-graph-task/ready-queue-design.md)。

## 1 实现、对照与环境

| 对照 | 业务 ready 拓扑 | entry 交权 | 用途 |
| ---- | --------------- | ---------- | ---- |
| original | 基线 4/8/4 分片 | 原始 `st_dev` 路径 | 总改动的效果参照，不作为安全发布证明 |
| baseline-atomic | 相同 4/8/4 分片 | 64B 独占 entry、atomic 发布与确认 | 隔离内存协议修正成本 |
| candidate-two-queues | AIC/MIX 与 AIV，各一条 | 与 baseline-atomic 相同 | 隔离拓扑改动效果 |

基线 commit 为 `619051ea29a5804454b66b33b31c5e4b2bff049f`。
三臂均保存冻结源码、dirty patch、编译命令、工具链与二进制 SHA-256，使用 A5 onboard、CANN 9.3.0，
PTO-ISA `90a3796597a5fabc2ce5066e73f785c2e85822f7`。使用当前默认构建的 `-O3 -g`；
`CMAKE_BUILD_TYPE` 为空，未额外启用 `Release/-DNDEBUG`，以实际编译命令为准。
最终 AICore SHA-256 前缀分别为 `e4848082fe7a`、`1294c1b0eded`、`ddb979ea4920`；完整值在 runtime manifest。

上述三臂的生产改动集中在 ready 布局、路由、发布/领取、调用方和 DFX reader。wire 为 v10，Header 64B，
两个 16B descriptor、两个 128B control；每个 entry 64B，head/tail 分行。
消费者只探测本类型一条业务队列，`ld_dev` 是提示，atomic load 确认 ref 后 CAS head 才获得所有权。
普通 direct completion 明确排除 MIX；SPMD admission、GM MIX、Graph control inbox、CHECK/FINISH、
错误与退出协议继续服务。两队列不表示只剩两种运行时状态，不改变用户表达或计算 kernel 算法。
独占 entry 会增加存储，不能把减少队列数解释成总内存必然下降。

本次三臂共同叠加两项验证前置修正：

- 删除 bootstrap/dispatch 内部对 Scalar observer 的提前清空，保留 resident 入口 reset 和 RAII 生命周期。
  此前仅有启动 DCCI/DSB、缺少实际 atomic/ld_dev 的 L3 不完整，全部重新采集；旧 6.6% 膨胀结论作废。
- Host 临时 JSON、orchestration SO 和 kernel 编译临时文件遵循 `TMPDIR`。
  JSON 写出先 close/check，写失败清理并报错。新产物统一写专用磁盘目录。

MoE 原冻结 wait ELF 使用旧 AsyncGroupData 布局；依据同源码重编译 `dispatch_meta_wait`/`combine_wait`，
保持模型算法和源码不变，并在三臂使用同一份新 bundle。
重打包保留原 payload 声明长度和后续 offset，新增零填充，未改其他 kernel 字节。
旧 ABI 设备错误及一次无效打包造成的 registration -1004 保留为失败证据，不计为通过样本。

## 2 功能与计时口径

[生产 dsv41_scheduler](../../../fdwic/dsv41_scheduler/README.md) 使用原始模型任务、依赖、输入边界和 kernel。
原始输入是 72 valid tokens/rank、80 padded rows、零权重；验证覆盖该调度 workload 的数值一致性，
不是已训练 checkpoint 的模型精度验收。
Attention/Engram 只用设备 4；四个含通信的 MoE/19–21 层用设备 6、7，保留真实通信和 defer_wait。
后四项首轮每项做一次功能验证；随后补齐原版／两队列的 L1/L3 参考数据，见下文，不纳入性能考核。另有 MoE 三臂功能对照全部通过。

十四项、三臂、L1/L3 每级 **1 次预热、10 次正式计时**：共 924 次调用，840 个正式样本。
完整输出、执行数量、resident mode 和 epoch 均校验。
单卡 case 对照冻结的原始 prefix+case writable bytes；跨卡 case 对照真实 fused prefix reference。

主指标来自 `chip.run.runner_run.device_wall` 的 `dev_start_cycle`、`dev_end_cycle`、`dev_cnt_hz`：
`(end - start) / frequency`，按 `pid/inv/run_epoch` 与 raw 关联，并核对导出时间。
边界覆盖 AICPU 设备入口至退出；不是首末 kernel 窗口，也不使用 Host 时间或跨核区间相加。
同一场景的 L1/L3 分别取十次中位数。十四项合计只是独立用例中位数之和，不是连续 forward。

| 验证范围 | 结果与边界 |
| -------- | ---------- |
| 调度协议 CPU | 38 个目标、2952 个 GTest 通过 |
| observer 专项 | 新增两项先复现失败；startup 132 项及其他 24 项通过，未将定向复测写成全量重跑 |
| DFX Python | 898 项通过 |
| 临时路径 | Python 28 项；A5/A2A3 loader 各 12 项通过 |
| A5 原语探针 | AIC 8、AIV 16、MIX 8/32 四种形状，显式空洞交错、唯一领取、payload 和边界检查通过 |
| 生产调度器专项 | 24 个真实 pytest 子用例通过，80 次 fresh resident mode 检查；覆盖 SPMD/MIX/Graph/async |
| 同址 run 复用 | 三次新 epoch 复用同一 scheduler state/pool GM，实际 H2D 初始化全 INVALID/零游标，终态排空、数值通过 |
| 模型 | 14 个单卡场景所有计时调用通过；4 个 EP2 场景两 rank 均通过 |

探针的三个逻辑任务共用私有状态与生产新 run 复用是不同证明。
后者使用只读 Host 观测辅助库检查实际 H2D 内容和 GM；不将 prepare 阶段尚未上传的 GM 当作初始化结果，
不手工补写设备状态。该观察引入的 Host 同步仅用于功能证明，没有用于性能采集。
CPU 错误合同和定向上板覆盖不意味着穷尽所有硬件故障交错。

## 3 L1 对比及边界项复测

负数表示耗时减少。±5% 内按用户要求视为无明显变化；不因为小幅负值宣称优化成功。

| 场景 | 原始 L1/ms | Atomic L1/ms | 两队列 L1/ms | 两队列/原始 | 两队列/Atomic |
| ---- | ---------: | -----------: | -----------: | ----------: | ------------: |
| swa / lib | 1.4484 | 1.4476 | 1.4690 | +1.43% | +1.48% |
| swa / DSL | 1.8680 | 1.8892 | 1.8422 | -1.38% | -2.49% |
| c2a_full / lib | 2.2310 | 2.2707 | 2.1918 | -1.76% | -3.47% |
| c2a_full / DSL | 2.3497 | 2.3423 | 2.3028 | -2.00% | -1.69% |
| c2a_reuse / lib | 1.7763 | 1.7538 | 1.7710 | -0.29% | +0.98% |
| c2a_reuse / DSL | 2.1107 | 2.1576 | 2.1490 | +1.81% | -0.40% |
| c1a_full / lib | 2.1007 | 2.1377 | 2.0836 | -0.82% | -2.53% |
| c1a_full / DSL | 2.3901 | 2.3611 | 2.3085 | -3.41% | -2.23% |
| c1a_reindex / lib | 2.1467 | 2.1741 | 2.1208 | -1.21% | -2.45% |
| c1a_reindex / DSL | 2.4009 | 2.3623 | 2.3635 | -1.56% | +0.05% |
| c1a_reuse / lib | 1.7196 | 1.7706 | 1.8096 | +5.23% | +2.20% |
| c1a_reuse / DSL | 2.1462 | 2.1101 | 2.0928 | -2.49% | -0.82% |
| engram / lib | 1.8928 | 1.8902 | 1.9012 | +0.44% | +0.58% |
| engram / DSL | 1.8686 | 1.8624 | 1.8841 | +0.83% | +1.17% |

| 汇总 | 原始 L1/ms | Atomic L1/ms | 两队列 L1/ms | 两队列/原始 |
| ---- | ---------: | -----------: | -----------: | ----------: |
| PyPTO-lib 七项 | 13.3156 | 13.4447 | 13.3470 | +0.24% |
| DSL 七项 | 15.1342 | 15.0849 | 14.9429 | -1.26% |
| 十四项合计 | 28.4497 | 28.5296 | 28.2899 | -0.56% |

初次矩阵中 C1A reuse/lib 相对原始基线 +5.23%，相对相同 atomic 协议基线 +2.20%。
保留该原始结果，并追加同设备 4 的 A/B/A 控制；每臂仍为 1+10 次，没有用复测覆盖首轮数据。

| 控制臂 | 版本 | L1 中位数/ms | 十次范围/ms |
| ------ | ---- | -----------: | ----------: |
| A1 | original | 1.7789 | 1.6999–1.8301 |
| B | candidate-two-queues | 1.8097 | 1.7448–1.8346 |
| A2 | original | 1.7678 | 1.7067–1.8116 |

复测 B/A1 为 +1.74%，B/A2 为 +2.37%，
A2/A1 为 -0.62%。具体差异解释以该控制及 L3 原始证据为界，
不能把初次 5.23% 全部归因于两个队列，也不能只引用更有利的一轮。

首轮 L1 按十次算术均值拆分，原始/Atomic/两队列的首个 kernel 前时间为
190.918/191.379/183.132 µs，末个 kernel 后为 85.237/84.825/81.291 µs，
首末 kernel 窗口为 1458.150/1494.434/1543.489 µs。
首轮增量位于中间执行区，不能归咎于 AICPU 启停；这里使用均值保证三个不重叠区间可加，
不替代上表中位数主指标。L3 的同 case 全程中位数为 1.9751/1.9729/1.9621 ms，
没有复现 L1 的 +5.23%；profiling 会改变调度相位，不能用 L3 较快否定 L1 原始结果。

## 4 L3 完整性、体积与复现

462 份 L3（含预热）逐份检查通过：每份 96 个 worker stream、零 dropped、零 truncated，
committed 与记录数一致，并包含实际 atomic、ld_dev、DCCI、DSB。
`scalar_atomic_claim` 等语义区间不冒充 atomic 指令；atomic-load/CAS 顺序检查仅限可关联的 ready-claim scope，
不从现有无地址/返回值记录推导每次 CAS 是否成功。

| 场景 | 原始 L3/ms | Atomic L3/ms | 两队列 L3/ms | 两队列 L3/L1 增幅 | 原始 raw/MB | 两队列 raw/MB |
| ---- | ---------: | -----------: | -----------: | ----------------: | ----------: | ------------: |
| swa / lib | 1.6438 | 1.6454 | 1.6366 | 11.41% | 81.70 | 55.15 |
| swa / DSL | 2.1276 | 2.1672 | 2.0935 | 13.65% | 118.07 | 73.88 |
| c2a_full / lib | 2.5308 | 2.5245 | 2.4805 | 13.17% | 125.65 | 80.78 |
| c2a_full / DSL | 2.6099 | 2.6569 | 2.5595 | 11.15% | 112.28 | 73.29 |
| c2a_reuse / lib | 1.9840 | 1.9928 | 1.9917 | 12.46% | 89.11 | 60.57 |
| c2a_reuse / DSL | 2.4518 | 2.3935 | 2.3181 | 7.87% | 114.40 | 69.09 |
| c1a_full / lib | 2.3352 | 2.3709 | 2.3614 | 13.33% | 110.68 | 76.10 |
| c1a_full / DSL | 2.6512 | 2.6012 | 2.5678 | 11.23% | 116.42 | 73.71 |
| c1a_reindex / lib | 2.4071 | 2.3788 | 2.3404 | 10.35% | 120.50 | 75.97 |
| c1a_reindex / DSL | 2.6779 | 2.6221 | 2.5834 | 9.30% | 113.56 | 71.42 |
| c1a_reuse / lib | 1.9751 | 1.9729 | 1.9621 | 8.43% | 87.74 | 59.07 |
| c1a_reuse / DSL | 2.4467 | 2.4289 | 2.3671 | 13.11% | 115.91 | 71.80 |
| engram / lib | 2.0595 | 2.0516 | 2.0072 | 5.58% | 89.11 | 60.83 |
| engram / DSL | 2.0608 | 2.0763 | 1.9938 | 5.82% | 89.30 | 60.46 |

raw 为未压缩 `chip_swimlane_records.json` 十次字节数的中位数，MB=1,000,000 bytes。
两队列最大 L3/L1 时间增幅 13.65%，全部低于 50%；十四项 raw 中位数字节数之和变化 -35.18%。
十四项每类原语记录数的十次中位数之和如下（不是时间，允许中位数出现半次）：

| 原语 | original | baseline-atomic | candidate-two-queues |
| ---- | -------: | --------------: | -------------------: |
| ld_dev | 4,502,114 | 4,383,153 | 2,224,042 |
| atomic | 105,116.5 | 108,810 | 116,727 |
| DCCI | 338,414 | 337,915 | 340,165 |
| DSB | 179,109.5 | 178,858.5 | 179,943 |

相同 atomic 协议下，ld_dev 数减少约 49.26%，atomic 数增加约 7.28%；
两队列确实减少了扫描，但没有对应的 L1 显著加速。
共享队头竞争和未改变的 admission/完成汇合可能限制收益，这只是后续定位方向，记录次数不能证明耗时占比。
体积改善伴随实际扫描及记录减少，没有靠丢记录或采样达到。时间优先，文件变小不代表 L1 加速。
原语 issue 时间与结果就绪时间不同，未完成开销可能落在后续 DSB；分析关键路径时不可直接累加各核 phase。

本机证据根记为 `E=.docs/two-ready-queues/final`；该目录映射到专用数据盘，原始大文件不入 Git。

| 产物（相对 E） | 用途 |
| -------------- | ---- |
| `final-runtime-set.json`、`observer-v1/*/runtime-manifest.json` | 三臂源码、编译器与二进制身份 |
| `final-model-matrix-01/matrix-summary.json`、`matrix-cases.csv`、`matrix-samples.csv` | 14 项汇总及全部 924 个样本 |
| `final-model-matrix-01/performance/*/summary.json`、`samples.csv` | 各场景样本、时间边界、raw 哈希与代表轮 |
| `final-model-matrix-01/functional-summary.json` | 四个 EP2 功能结果 |
| `l3-primitive-audit/incremental/summary.json` | 42 组、462 份原始 L3 审核与各组报告哈希 |
| `c1a-reuse-l1-control-01/results.json` | C1A reuse/lib 同卡 A/B/A 原始复测 |
| `c1a-reuse-window-analysis.json` | 首轮非重叠窗口分解；均值与中位数分开报告 |
| `retained-run/candidate-04/result.json` | 生产同址 GM 跨 run 复用证据 |
| `wait-abi-v2/`、`dfx-pr3000/`、`tmpdir-ut/` | 前置修正、失败复现与验证 |
| `final-model-matrix-01/swimlanes/` | 14 场景 × 3 臂 × L1/L3，按场景/实现/版本/级别/rank 命名 |
| `final-model-matrix-01/two-ready-queues-14cases-three-arms-L1-L3.zip` | 上述 84 张 JSON 泳道图及哈希清单 |

泳道图选取各自最接近十次中位数的正式轮；original/atomic 使用冻结 reader，两队列使用当前 reader。
复现需保留冻结输入及相应三臂 runtime，不能用当前 HEAD 替代旧版本。设置专用 `TMPDIR`，架构预检和
`task-submit` 锁定单卡后，分别执行：

```bash
DSV41_RUNNER="$FROZEN_RUNNER" bash tests/fdwic/dsv41_scheduler/production/run.sh \
  "$BUNDLE" "$RUNTIME" "$OUT/L1" 1 1 10
DSV41_RUNNER="$FROZEN_RUNNER" bash tests/fdwic/dsv41_scheduler/production/run.sh \
  "$BUNDLE" "$RUNTIME" "$OUT/L3" 3 1 10
```

同一 reservation 内运行各臂，保存独立输出，分析完整 AICPU 时间。
逐场景 `convert-swimlanes.sh` 保留实际 rank/pid/inv 和 reader 参数；缺少本机冻结产物不能声称重现本次测量。

### 跨卡四项补测：记录数据，不纳入收益考核

按后续要求补齐四项 L1/L3，使用相同冻结的 original 与 candidate-two-queues；设备 2、5，
每版每级 1 次预热、10 次正式运行，保留真实 EP2 通信和 defer_wait。共 176 个双 rank 运行、352 份 raw。
每轮取两个 rank 各自完整 AICPU 设备耗时的较大值，再对十轮取中位数；不跨设备相减绝对时钟。
原始 prefix 与 fused reference 运行只用于准备、校验，明确排除在目标 case 时间之外。单位均为 ms。

| 场景 | 原版 L1 | 两队列 L1 | L1 变化 | 原版 L3 | 两队列 L3 | L3 变化 |
| ---- | ------- | --------- | ------- | ------- | --------- | ------- |
| MoE / PyPTO-lib | 4.1820 | 4.0828 | -2.37% | 5.1180 | 5.4493 | +6.47% |
| MoE / DSL | 2.5135 | 2.5001 | -0.53% | 2.9545 | 2.8364 | -4.00% |
| 19–21 层 / PyPTO-lib | 19.3197 | 17.8918 | -7.39% | 22.9632 | 20.1768 | -12.13% |
| 19–21 层 / DSL | 13.3667 | 13.0548 | -2.33% | 14.9438 | 14.3704 | -3.84% |

| 场景 | 两队列 L1 最小–最大 | 两队列 L3 最小–最大 | 两队列 L3/L1 增幅 |
| ---- | ------------------- | ------------------- | ----------------- |
| MoE / PyPTO-lib | 3.8544–4.7883 | 5.3098–6.0062 | 33.47% |
| MoE / DSL | 2.4571–2.5482 | 2.7079–2.9114 | 13.45% |
| 19–21 层 / PyPTO-lib | 17.4252–20.8467 | 19.7999–20.7424 | 12.77% |
| 19–21 层 / DSL | 12.8134–13.5100 | 14.0214–14.6085 | 10.08% |

全部数值校验和完整设备时间匹配通过。160 份正式 L3 的 producer capture counters 均报告零 dropped、零 truncated。
16 份 L3 预热 raw 中存在截断，涉及四个双 rank 预热轮：原版 MoE/lib、两队列 MoE/DSL、原版及两队列 19–21 层/DSL。
预热不计入十次正式数据；失败记录原样保留。补测没有逐条重做所有 Scalar scope 的因果审核，不能把 capture counter 检查表述为与前述 462 份相同的完整原语审核。
通信四项仅作为参考数据，即使变化超过 5% 也不据此保留或撤回调度器优化；十四项考核表与结论保持独立。

完整样本及每个 rank 时间保存在工作区 `fwic/artifacts/two-ready-queues-20261111/ep2-performance-01/` 下的
`performance-summary.json`、`performance-cases.csv`、`performance-samples.csv` 和各组 `timing-summary.json`。
此前 disk3 原始数据可通过用户目录下 `disk3` 软链接访问；后续新增派生产物和临时文件使用 disk2。

## 5 生产代码文件与函数

对应本轮三个本地提交 `2d7d8f7a7`、`2229f393d`、`836d16e55`，相对 `619051ea`：
生产文件 24 个，新增 380 行、删除 1156 行，净减少 776 行；下表不包含测试与文档文件。
其中 loader 头文件和 overhead 工具的一处变更仅为注释，不计作运行逻辑变化。

前十六行路径相对 `src/a5/runtime/host_build_graph/`；其余路径相对仓库根目录。

| 文件 | 变更函数或结构 | 主要变化 |
| ---- | -------------- | -------- |
| `runtime/scheduler/scheduler_types.h` | `scheduler_task_ready_queue`、`scheduler_plan_layout`、`scheduler_init_data_from_layout`；`SchedulerReadyClass`、`SchedulerReadySource`、`SchedulerRunControl`、`SchedulerWorkerContext` | 两类路由，移除旧 inbox/directory 分配和字段；删除 `scheduler_ready_queue_capacity`、`scheduler_ready_queue_base` |
| `runtime/scheduler/scheduler_ready_pool.h` | `scheduler_ready_pool_queue`、`scheduler_ready_pool_validate`；`SchedulerReadyPoolHeader` | 两队列寻址与 wire v10；删除 `scheduler_ready_pool_groups`、`scheduler_ready_pool_ref_group` |
| `runtime/scheduler/scheduler_ticket_queue.h` | `scheduler_ticket_publish`、`scheduler_ticket_try_pop_batch`；`SchedulerTicketEntry`、`SchedulerTicketQueueView` | 64B 独占 entry、原子发布和权威 atomic 确认 |
| `runtime/scheduler/scheduler_ready.h` | 具体函数见下文 | 移除分片发现及 stealing；统一单队列发布、探测和领取 |
| `runtime/scheduler/scheduler_ready_publication.h` | `scheduler_ready_publication_append`、`scheduler_ready_publication_step` | 区分 AIC/AIV/CONTROL，移除业务 directory 发布阶段 |
| `runtime/scheduler/scheduler_dispatch.h` | `scheduler_ordered_poll_may_progress`、`scheduler_fill_ordered_slots`、`scheduler_fill_own_slots` | 删除 AIC/MIX 双路搜索和 stealing 参数，保留原 admission 额度与协作服务 |
| `runtime/scheduler/scheduler_completion_resolver.h` | `scheduler_resolve_completion_step` | direct handoff 显式排除 MIX，移除旧 MIX candidate 暂存与回填 |
| `runtime/scheduler/scheduler_completion.h` | `scheduler_complete_ordered_task`、`scheduler_service_cluster_completion_slot`、`scheduler_service_cluster_completions` | 删除 `SchedulerRefillCandidates` 及旧 MIX 参数，适配完成调用链 |
| `runtime/scheduler/scheduler_resolve_state.h` | `scheduler_ready_destination`；`SchedulerReadyPublication`、`SchedulerReadyDestination`、`SchedulerReadyPublishPhase` | 独立发布目的地枚举，删除 DIRECTORY 状态 |
| `runtime/scheduler/scheduler_async_layout.h` | `scheduler_async_copy_resolver` | 续行状态复制 `publication.destination` |
| `runtime/scheduler/scheduler_layout.h` | `AicoreSchedulerLayout` | 删除失效 ready inbox/directory offset |
| `runtime/scheduler/scheduler_mix.h`（删除） | `scheduler_dispatch_mix_ready`、`scheduler_fill_cluster_mix_slots`、`scheduler_publish_direct_mix_candidate`、`scheduler_fill_mix_after_completions` | 删除四个旧 MIX 调度函数，当前 GM MIX 保留 |
| `aicore/resident_scheduler_impl.h` | `bootstrap_ready_graph`、`run_ready_dispatch_loop` | 删除失效初始化和刷新；保留入口所绑定的 Scalar observer |
| `aicpu/aicpu_executor.cpp` | 新增 `log_resident_ready_pool`；修改 `log_unfinished_tasks` | 超时诊断读取两队列游标、容量及 head ref |
| `host/ready_pool_plan.h` | `scheduler_plan_ready_pool`、`scheduler_init_ready_pool` | 固定两个 control/descriptor，按逻辑任务统计并初始化 entry |
| `host/runtime_maker.cpp` | `create_scheduler_state`、`publish_aicore_scheduler_profiling`、`create_orch_so_tempfile` | 新布局、删除 Worksteal 事件、临时 SO 遵循 TMPDIR |
| `src/common/aicpu_loader/host/load_aicpu_op.cpp` | `LoadAicpuOp::Init`、`LoadAicpuOp::GenerateAicpuOpJson` | TMPDIR 路径及 JSON close/write 错误处理 |
| `src/common/aicpu_loader/host/load_aicpu_op.h` | 类说明注释 | 更新临时路径说明，无函数逻辑变化 |
| `src/common/host_build_graph/sched_phase_kind.h` | `SchedPhaseKind`、`sched_phase_kind_name` | 删除 Worksteal、ReadyRefresh、ReadyDirectory、PollOwnInbox，保留其他编号 |
| `simpler_setup/kernel_compiler.py` | `KernelCompiler._make_temp_path` | 默认使用 tempfile 配置目录，显式 build directory 仍优先 |
| `simpler_setup/tools/hbg/native_causality.py` | `NativeAudit.execution` | ready source 合法值为 0、2、3，拒绝旧 STOLEN 值 1 |
| `simpler_setup/tools/hbg/phases.py` | `PHASES_BY_PRODUCER`、`SCALAR_PHASES`、`PHASE_COLORS`、`PHASE_REPORT_LABELS`、`PHASE_DISPLAY_NAMES`、`OUTER_PHASES`、`WORK_PHASES` | 同步删除失效阶段和展示映射 |
| `simpler_setup/tools/hbg/scalar_profile.py` | `SCALAR_FUNCTIONS` | 删除三个失效函数映射 |
| `simpler_setup/tools/sched_overhead_analysis.py` | `_scheduler_phases_for_report` 的 docstring | 移除旧 worksteal 顺序描述，无函数逻辑变化 |

`scheduler_ready.h` 中的具体函数如下：

- 修改：`scheduler_initialize_local_config`、`scheduler_ready_queue_holder`、`scheduler_ready_pool_push_batch`、`scheduler_ready_batch_push`、`scheduler_ready_pop_from_queue`、`scheduler_ready_pool_claim`、`scheduler_ready_pool_may_have_work`、`scheduler_claim_ready_for_slot`、`scheduler_resolve_completion`、`scheduler_complete_inline_task`。
- 删除：`scheduler_owner_queue_activate`、`scheduler_owner_queue_deactivate`、`scheduler_ready_inbox_at`、`scheduler_ready_queue_owner_count`、`scheduler_ready_directory_at`、`scheduler_ready_directory_set`、`scheduler_ready_directory_clear`、`scheduler_refresh_ready_inbox_type`、`scheduler_refresh_ready_inbox`、`scheduler_ready_push_target`、`scheduler_ready_pop_from_inbox`、`scheduler_load_ready_directory_shard`、`scheduler_steal_ready_from_shard`、`scheduler_ready_directory_nonempty`、`scheduler_plan_mix_placement`。
- 状态清理：`SchedulerReadyStats`、`SchedulerReadyClaim`、`SchedulerLocalConfig`、`SchedulerLocalState`、`SchedulerCompletionDirect`；`scheduler_ready_batch_push_queue` 更新控制 FIFO 的说明，未重写其算法。
