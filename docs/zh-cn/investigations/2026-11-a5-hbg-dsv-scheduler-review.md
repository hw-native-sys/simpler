# A5 HBG 调度改动取舍与 DSV4.1 回退定位

验收采用 PR #2975 的 18 组 DSV4.1 场景，优先完整三层与 MoE 耗时，同时披露单独 attention 的代价。
保留 defer_wait、MIX、sync_start、跨轮复用和任意合法拓扑；不以删除其他场景需要的逻辑换取性能。
数据结构、构建隔离和解耦的价值单独说明，不将它们一概称为 device 性能优化。

## 保留与撤除的依据

| 改动 | 类型与决定 | 依据及代价 |
| ---- | ---------- | ---------- |
| 共享结构死字段清理、64B 对齐 | 数据结构，保留 | TaskControl 640→384B、WorkerContext 1024→384B、TaskTrace 384→320B、TicketQueue 256→128B；删除字段后重排，不留 reserved 占位 |
| TaskControl 热原子字段隔离 | 性能修复，保留 | next_block、completed_blocks、claim_seal 各占 64B 行；192B 过度压缩版的 Engram/SWA 回退经同锁 A/B/A 定位，见下节 |
| Host CSR 与 device 剩余依赖计数 | 算法与职责分离，保留 | 静态 DAG 的邻接关系在 Host 预计算，device 每条完成边通知一次；增加约 16N+4E 字节状态和旧图兼容路径，不宣称所有场景都更快 |
| 小 grid bounded/rotated 成员 | 性能策略，保留 | 复用 admission，限制无效成员参与并轮转起点；独立去除试验显著损伤 lib MoE 与完整三层，但 DSL SWA 可快约 0.08ms |
| 四个 resident 编译单元 | 构建隔离，保留 | ordinary/native 与 profile 变体共享实现、分别实例化；避免单个巨型 TU 汇聚所有变体 |
| resolver static noinline | 编译成本控制，保留 | 同一个 native_profiled AIV TU，保留时 160.422s 完成，去掉后 300.118s 仍未完成并终止；这不是完整 runtime 编译倍数或 device 加速证明 |
| benchmark Host barrier 去除 sleep | 测量质量，保留 | 原 10ms 轮询睡眠导致两 rank 启动偏差；保留有期限的 busy poll，Host bind 仍可能产生启动差 |
| 历史 DFX reader/test | 撤除 | 当前协议不承担历史或内部实验格式兼容；旧测量保留其原提交和证据，不改变当前运行合同 |
| 独立 block queue/dispatch/completion | 撤除 | 新增逐 block 领取和 completion join 协议，同卡二分证实 Engram/SWA 大幅回退 |
| Host fanout 排序 | 撤除 | 无一致收益；去除后 lib 三层更快，同时解除依赖表与 ready 队列分组耦合 |
| payload prefix writeback | 撤除 | 控制版 35.688ms、候选 35.830ms，无稳定收益，不保留额外回写边界逻辑 |
| 冷错误 noinline、二分 popcount 成员前缀 | 不纳入 | 未建立延迟收益；后者 DSL SWA 约 1.57ms，控制约 1.53–1.55ms，却增加实现复杂度 |

CSR 的 Host-only 开关对照中，设备二进制完全相同：lib 三层 on-a 21.798/22.593ms → off
20.755/20.193ms → on-z 24.773/23.357ms；DSL 三层分别为 12.670/12.985 →
13.458/13.437 → 13.130/13.148ms。控制臂存在漂移，因此其保留依据包括职责分离，
不能据此宣称一致的性能收益。依赖计数的合同与存储见
[dependency-counters.md](../design/a5-hbg-graph-task/dependency-counters.md)。

成员策略独立 A/B/A（两 rank，ms）如下，三臂使用相同 kernel、输入和 Host/AICPU：

| 场景 | bounded-a | full cohort | bounded-z |
| ---- | --------- | ----------- | --------- |
| lib MoE | 3.796837/3.751754 | 6.516412/6.712094 | 3.536971/3.515196 |
| lib 三层 | 21.772138/21.829609 | 28.142897/26.660582 | 20.860764/20.114344 |
| DSL SWA | 1.585238/1.575976 | 1.495002/1.503859 | 1.533414/1.577277 |
| DSL C1A reuse | 1.909473/1.910887 | 1.894070/1.853095 | 1.908473/1.898572 |

## 两类回退的独立定位

首先，独立 block 策略的回退发生在 64B 布局之前。1/7 卡同锁，每臂 warmup 1 次、L0 5 次；
main=18e1841a2，counter=9810fda31，block=866f71c39。表内为 rank0/rank1 中位数，单位 ms。

| 场景 | main-a | counter | block | main-z |
| ---- | ------ | ------- | ----- | ------ |
| lib Engram | 1.473504/1.479678 | 1.457381/1.469396 | 3.343549/3.366643 | 1.459917/1.463373 |
| lib SWA | 1.121061/1.138808 | 1.106926/1.112631 | 1.955807/1.970233 | 1.104292/1.136335 |
| DSL Engram | 1.481160/1.468175 | 1.459901/1.456246 | 3.331382/3.293790 | 1.473253/1.453797 |

Engram 同一逻辑任务有 500 次物理执行。既有 L1 记录中，DSL rank0 任务跨度从
141.397us 增至 1848.441us，而单次 kernel 中位数为 1.310/1.247us。
独立路径每个 block 重新领取共享队头、准备 slot、做原子 completion join；
cooperative 路径按成员分摊 block，复用 admission 与有序执行上下文。
证据定位到运行策略，尚不能将全部损失定量归于某条 CAS 或缓存未命中。

其次，恢复 cooperative 后的 192B TaskControl 仍有约 0.15ms 回退。
将领取、完成、claim seal 三个原子字段分开到 64B 行，保持算法不变，在同锁 A/B/A 中得到：

| 场景 | 紧凑 A ms | 隔离 ms | 紧凑 B ms | 主线 ms |
| ---- | --------- | ------- | --------- | ------- |
| lib Engram | 1.629601/1.652014 | 1.471880/1.479668 | 1.661390/1.641689 | 1.469662/1.465603 |
| lib SWA | 1.235469/1.232291 | 1.115365/1.128077 | 1.244404/1.227050 | 1.131233/1.116264 |
| DSL Engram | 1.636028/1.650243 | 1.474244/1.459054 | 1.637859/1.650236 | 1.477016/1.456045 |

因此保留 384B 布局，不能将所有回退都归于独立 block 策略。
去 fanout 排序的独立通信复测中，lib 三层为 25.263/24.694 → 21.080/20.524ms；
DSL 为 13.519/13.614 → 13.577/13.511ms。这组使用 192B 布局，只用于排序取舍。

## 完整 18 组数据与测量边界

以下是已完成的历史同锁对照：主线 18e1841a2 与取舍后 60c38dcd6，设备 1/7，
统一修正 barrier 的 runner；每版 warmup 1 次、L0 5 次，另采 L1/L3。
72 个 rank 输出检查、144 份诊断捕获审计通过，另有 3 组主线复测。
数据为 L0 device wall 中位数，单位 ms，斜线分隔 rank0/rank1。

这组不能冒充最新主线对照：后续 rebase 基线为 181e24619，已含 #2977 的 exhausted-SPMD
推进优化；最终又同步到含 #2750 参数抓取的 e8b0192f2。主线新功能与收益不计入本改动；新版定向回归与候选对照另列。

| 场景 | main 18e1841a2 | 取舍后 60c38dcd6 |
| ---- | -------------- | ---------------- |
| lib SWA | 1.121611/1.119364 | 1.126575/1.123045 |
| lib C2A full | 1.845301/1.834277 | 1.881933/1.850630 |
| lib C2A reuse | 1.462016/1.417260 | 1.405404/1.456883 |
| lib C1A full | 1.765812/1.784144 | 1.746815/1.768320 |
| lib C1A reindex | 1.775356/1.789816 | 1.780206/1.801590 |
| lib C1A reuse | 1.485954/1.479489 | 1.459855/1.463142 |
| lib MoE | 6.066378/6.650675 | 3.743640/3.842947 |
| lib Engram | 1.497070/1.456956 | 1.470560/1.483091 |
| lib 三层 | 23.896614/27.344131 | 21.680926/21.749939 |
| DSL SWA | 1.514260/1.529573 | 1.576126/1.583034 |
| DSL C2A full | 1.998361/2.001573 | 2.030943/2.065322 |
| DSL C2A reuse | 1.848575/1.818196 | 1.930548/1.904043 |
| DSL C1A full | 1.977742/1.988399 | 2.027971/2.005939 |
| DSL C1A reindex | 2.021139/2.017071 | 2.013787/2.022810 |
| DSL C1A reuse | 1.804542/1.821925 | 1.907282/1.921797 |
| DSL MoE | 2.179868/2.217950 | 2.059842/2.222852 |
| DSL Engram | 1.458264/1.448442 | 1.478270/1.472463 |
| DSL 三层 | 12.239961/12.590887 | 13.383747/13.123507 |

lib MoE/三层改善，部分 DSL attention 回退约 4–6%，DSL 三层也回退；这不是全场景普遍加速。
去除成员策略会损失 lib 宏场景收益，故按“完整三层与 MoE 优先”的验收选择保留，并披露代价。

L0 wall 包含初始化、退出和跨 rank 等待。物理执行级 L1 路径从首 kernel 到末 kernel，
沿数据与同核前驱拆为 kernel+gap，原始 cycles 校验严格等于窗口。
gap 仍包含调度、同步、资源和 peer 等待，不能称为纯 scalar 指令时间，不能用 L1 与 L0 相减。
Host bind 在 barrier 后执行，通信 DFX 采样可放大启动差至几十 ms，不能据其宣称纯调度已低于 8ms。

在 181e24619 基线上，连续拓扑区间掩码候选与现有逐位成员构造作同锁 A/B/A。
候选保持相同成员集合，41,040 组拓扑/跨字/环绕等价检查通过，但宏场景无稳定收益，故不纳入：

| 场景 | 当前 A ms | 候选 ms | 当前 B ms |
| ---- | --------- | ------- | --------- |
| lib MoE | 3.464536/3.454065 | 3.490573/3.815410 | 3.507564/3.731685 |
| lib 三层 | 19.926588/20.931580 | 20.268129/21.115934 | 18.271833/18.235587 |
| DSL MoE | 2.148568/2.049847 | 2.117279/2.125478 | 2.115123/2.222078 |
| DSL 三层 | 12.497706/12.400738 | 12.516264/12.391645 | 12.510566/12.488543 |

24 个 rank 输出与 48 份 L1/L3 审计通过。这组只评估等价位图候选，不是未修改主线对照。

## noinline 与 Issue #2973

[#2973 最新定位](https://github.com/hw-native-sys/simpler/issues/2973#issuecomment-6761549840)
将触发条件收窄为 `alignas(16) SchedulerTaskMetadata` 栈对象通过引用进入外联的
`scheduler_record_dispatch_error`。8B 对齐引用、16B 结构按值、标量按值均通过。
报告观察到调用方额外的栈对齐与地址标记指令，CCEC 内部具体机制仍未完全确认。
不能将其泛化为所有栈引用加 GM 指针，也不能仅凭 static linkage 排除风险。

本改动只新增 resolver 的显式外联，声明与定义上的属性属于同一个函数。
它不接受 SchedulerTaskMetadata 引用；GraphAccess、LocalState、ResolveCursor、各统计对象、
CompletionDirect 及其 ReadyClaim 的自然对齐不超过 8B，并由编译期断言约束。
三个调用来源覆盖普通完成、异步 FINISH、native graph 控制，异步共享 cursor 先复制到局部对象。
GM 数据经明确的 `__gm__` 指针传入；`scheduler_record_dispatch_error` 保持内联。

实际 release `-O3 -g` ELF 有 8 个 LOCAL resolver 实例，没有 dispatch_error 外联符号，
也没有普通 Metadata 引用参数的外联函数符号。CANN 9.3.0/CCEC clang 15.0.5、dav-c310
上运行 Issue 同一 `burst256_mask6`：当前版与连续拓扑位图候选均通过 L0 10 轮和 L3 2 轮。
Issue 的失败组合使用 `-O1 -gline-tables-only` 与更多外联函数，两者不能混为同一构建。

结论是当前保留的边界未命中已知触发条件，且定向上板通过；这不等于证明所有编译器版本或
任意新增外联都安全。本改动关联 #2973，不宣称完整修复或关闭该 Issue。

## 验证与证据入口

- 取舍后历史完整矩阵：72 个 rank 输出、144 份 L1/L3 审计；相关 Python 协议测试 1029 项通过。
- 181e24619 rebase：13 个相关 CPU target 通过，模拟器与最终保留版硬件语义回归各 8 项通过；Issue 定向硬件验证通过。
- e8b0192f2 rebase：20 个相关 CPU target、37 项真实 PTO 头文件异步上下文测试通过。
- 最终 release 在设备 5 再次通过 burst256_mask6 的 L0 10 轮和 L3 2 轮。语义回归硬件 7 个 L2 场景加 1 个独立 sync 检查、模拟器 8 个 L2 场景加 1 个独立检查通过。
- 最终 sim/onboard 均通过参数抓取 full、hybrid 各 4 个场景，以及 burst16_mask6/mask7 的输出快照检查；全修改 pre-commit 通过。
- 新主线加入的 ordered-progress 单测按每任务实际成员选择执行者，保留延迟成员、扫描预算、
  caught-up idle 和待执行第九任务的断言，避免把固定 publisher 假定为轮转策略的执行成员。
- #2750 的参数抓取 helper 与执行前后边界迁入共享 resident 实现，保持上游逻辑；args_dump_address 放入 WorkerContext 的诊断配置行，仍为 384B。
- 主线开始使用原 async_reserved 拒绝不安全的异步输出抓取，因此将其保留并命名 async_output_capture_refused，连同生产者、kernel 检查与每次 dispatch 清零保留，GlobalContext 仍为 48B。
- 新增 bind/抓取单测适配删除的 graph_reserved_address；A5 bind fixture 显式链接 CMAKE_DL_LIBS，覆盖旧 glibc 上的 dlsym 依赖。
- native benchmark runner 补齐最新 worker 所需的 native_submission_state.cpp 链接；模型 bundle
  复用已核对 task-interface、通信设备 ABI 和 PTO-ISA pin 未变。

原始数据与执行记录位于 outputs/a5-hbg-schedule-8ms/dsv-regression：

- final-report.md、final-comparison.csv、final-l0.json、final-physical-path：完整 18 组与物理路径。
- counter-boundary-l0.json、audit-counter：独立 block 接入二分。
- control-lines、aligned-runner：原子行隔离和 Host barrier/fanout 对照。
- full-cohort、counter-off、resolver-retention：成员策略、依赖计数、外联构建成本的独立试验。
- noinline-review：Issue 原文、类型对齐、ELF 符号及编译参数核查。
- pr-rebase：181e24619 构建、定向上板、CPU/模拟器验证和宏场景候选对照。
- pr-final：e8b0192f2 构建、移动代码等价证明、ELF 核查与参数抓取集成回归。

历史分步提交保留在本地备份分支，审阅分支保持一个 commit；用户的 .gitignore 修改不纳入提交。
