# A5 HBG defer_wait 本地代码修改参考

更新日期：2026-11-09。**本地指 `nalinaly/simpler main`；对方指 `poursoul/simpler feat/a5-hbg-async-completion`。** 本文件按用户最新澄清修正本轮范围，可作为后续实施依据。

**本轮只改进 resident／AICore 自调度实现，不为 legacy AICPU 调度增加兼容适配；现有 legacy 代码保留。legacy 将来删除，不属于本轮删除授权。** 不删除原有执行器、回退入口、调度循环、mailbox、等待列表、存储布局或其测试，也不为了删除它们改动 A2/A3、A5 TRB 或共享 runtime。

这一原则适用于 nalinaly 上整轮 defer_wait 开发：hw-sys main 原有的 AICPU 调度功能继续保留，nalinaly 为新增 defer_wait 额外加入的 legacy 适配不作为必须维持的能力。后续若精简这些适配，先按上游基线确认其来源，只撤去新增功能的接线，不删除旧调度器原有实现。

此前按“本轮删除 legacy”实施的源码及测试改动已撤回，未提交或推送；相关隔离构建已停止。本轮已按“先精简生产代码，再验证业务边界”的顺序完成。下述结果来自当前保留源码的重新构建，不沿用撤回实验的产物或测试结论。

## 1. 目标和必须保持的行为

本地起点为 [降低 A5 defer_wait 零登记任务的调度开销][local-head]；对方参考版本为 [共享 SDMA 队列回绕与复用测试][peer-head]。实施时以当前源码为准，并保留其他会话修改。

- resident 覆盖普通任务、SPMD、MIX、sync-start、当前单层 Graph 内的异步任务。
- 普通或手写未知 kernel 无需额外声明，每次物理执行沿用默认 64 条条件预留；不新增资源所有者、绑定或消费准备框架。
- kernel 登记条件并正常返回，空闲 AICore 可跨核 CHECK；条件不满足时交回执行机会，检查工作不固定占住原计算核。
- CHECK／FINISH 为内部工作，不新增业务 TaskId，不占业务协同准入名额，不保存 kernel 栈或增加抢占。
- 整项零登记任务在原生执行全部结束后直接完成并 refill；不能把暂时没有 pending 当作整项同步。
- 保留真实 SDMA、COUNTER、正确的消费者 DCache、实际登记前缀发布和恰好一次完成。
- AICPU 继续承担 resident 路径已有的启动、退出和错误职责；外部 DMA 的最后访问不能由核退出或计算流结束单独证明。

不新增行为开关，不评估 `ld_dev` 差异，不合入其他会话的编译优化，不修改 PyPTO Graph／tensor 格式。legacy 兼容不作为本轮新增设计或验收目标，但也不能据此删除原有实现、专属测试或转移其存储职责。

当前实现与历史验证记录见 [开发参考][design]；固定版本差异见 [实现比较][comparison]。本文对本轮范围的说明优先于此前文档中“先删除 legacy”的建议。

## 2. 本地生产代码修改

本节 `HBG` 表示 `src/a5/runtime/host_build_graph/`。已有能力优先保留，不为吸收对方的命名或结构重写正确代码。

### 2.1 登记主体与上下文

本地 `runtime/async_completion_writer.h` 已有一份登记主体，COUNTER 和 SDMA 可共用。跨 legacy／resident 复用不再作为它的额外优势，但也不成为删除它的理由。

- 在 resident 内审视提交前容量检查、条件写入、计数、粘性错误、封口及必要身份检查，减少真正重复的工作。
- 保持上下文副本共享登记数、首个错误和封口状态；提交 SDMA 前检查最大记录数，返回后检查实际记录数。SDK 提交和登记在一次 kernel 调用中同步完成，期间不调用用户回调，因此不需要跨调用的预约票据、序号或重复写入计数；失败写入共享错误，封口拒绝错误和重复封口。
- `async_kernel_api.h`、`async_execution_context.h`、`common/intrinsic.h`、`dispatch_payload.h` 若需调整，只围绕 resident 的实际需求。原有 legacy 构造、slab 回退和字段不作为本轮删除清单。
- 不为两种调度器继续保持等价而新增桥接层，也不为了拆开现有公共函数再造抽象。没有明确收益时保留当前主体。
- 双方异步上下文 ABI 不同，不能混用异步 kernel 和 runtime 二进制；若调整 resident ABI，相关产物成套构建，并处理仍使用的业务参数布局。

当前生产改动已删除内部预约票据／序号／重复写入计数，writer 保留 `count / error / sealed`；SDK 提交失败仍通过首个错误与既有运行域保活处理，不能把移除预约状态当作外部搬运已完成。已有 writer、SDMA 提交、上下文、CHECK、执行、Host 规划／导出和 native last-use 相关检查通过，AIC／AIV CCEC 语法检查及已有 COUNTER／SDMA kernel 编译通过。当前模拟器的 41 个独立测试和 2 个场景入口通过，包含溢出、漏登记负向校验和永不满足条件；这些结果不代表真实 DMA／DCache 验证。真机正常 O3 构建、相关验收和同条件 L1 对照已完成，见第 6 节。

### 2.2 公共完成流程与零登记快路径

保留本地 `scheduler_ready.h`、`scheduler_completion_resolver.h`、`scheduler_ready_publication.h` 的公共依赖处理主体。普通完成、FINISH、Graph 控制节点都存在于 resident 内部，这项复用与 legacy 无关。

普通完成使用局部游标和大预算，在当前调用中处理到底，保留 direct refill 和 MIX 发布；FINISH／Graph 使用可续接游标分段处理。不能把普通完成强制转换为 FINISH。

保留 `scheduler_async_execution.h`、`scheduler_async_runtime.h` 的一次绑定、返回复用及零登记快路径。`scheduler_async_join.h` 可用 P（待完成）、R（累计实际登记）、closed（原生结束）解释，但不为改名重写已正确的原子协议或增加第二份计数。

### 2.3 必要前缀初始化与上传

这是优先吸收对方的生产改进。本地已经直接遍历 `SchedulerTaskMetadata`，不需要引入另一套 planner 或资源声明。

涉及 `host/async_plan.h/.cpp`、`host/runtime_maker.cpp`、`runtime/scheduler/scheduler_async_layout.h`、`scheduler_async_runtime.h`：

1. 完整保留 Device 条件存储及运行域生命周期，区分总分配大小和初始化／上传大小。
2. 把 SDMA guard 放在条件 entries 之前，控制前缀包含必须初始化的 header、writer、join、lease、progress、目录和 guard。
3. 只初始化和上传控制前缀，跳过尚未写入的条件数组；kernel 写入实际条件并封口后，CHECK 才能读取。
4. 验证 retained storage 跨次运行重新初始化全部必读控制字段，旧 entries 不会被错误 count、目录或 epoch 当作新条件。
5. 同步更新布局校验和私有版本。先只跳过条件数组，不一次延迟全部控制记录的初始化。

当前代码已实现这一改动；相关 C++、模拟器、真机跨次复用和真实 SDMA 验证均已通过。直接收益是 Host 初始化量和 H2D 上传量下降，不能据此推导 Device 预留量或 L1 耗时同比下降。

### 2.4 SDMA 与错误退出

保留本地 `backend/sdma/sdma_submission.h` 的公共提交、提交前容量检查、物理队列保护、SDK 完成记录快照及元数据发布。两类入口都可用于 resident，保留实际使用的薄适配。

- 保留多队列固定锁顺序、提交前失败释放、提交后错误处理及解锁前 SDK 状态发布。
- 保留同步 TPUT 退化、非空 TGET 零 handle 拒绝及实际完成记录检查。
- 不为 legacy 新增 guard 或协议，也不删除现有无 guard 的旧路径兼容分支。
- 保留 resident 的原始错误、核退出和最后使用边界；复用既有 run／domain／buffer owner，不重建资源框架。
- 不按 `aicpu/` 或 `src/common/` 目录整批删除代码。本地 `aicpu_async_transport_enabled()` 本身服务于 resident，不能误认为旧调度器兼容。

现有 legacy Graph／形态回退及其存储保持原样；本轮不为消除回退扩大 Graph 参数区、不移动共享 arena 实现，也不改变原有支持范围。

## 3. 吸收与保留的取舍

| 差异 | 本地决定 | 本轮边界 |
| ---- | -------- | -------- |
| 独立 resident writer | 按实际重复和开销决定是否精简已有主体 | 不因跨调度器复用加分，不删原有 legacy |
| Host 必要前缀上传 | 优先吸收 | 完整分配与生命周期保留，控制区不能遗漏 |
| 普通／FINISH／Graph resolver | 保留本地 | 三类 resident 流程共用主体和发布规则 |
| P／R／closed | 吸收清楚的表达 | 不增加等价状态表或重复计数 |
| SDMA 公共提交、多队列保护 | 保留本地 | 不为 legacy 增加适配，也不删除其现有分支 |
| 冷态 16 次服务机会探测 | 暂保留本地策略 | 有单独发现延迟／空闲成本证据再修改 |
| 活动后启用消费者 DCache | 暂保留本地统一入口 | 先验证换核、predicate-only、Graph EXIT、别名及脏数据 |
| 目录 CAS 与加减 bit | 暂保留本地 | 核对唯一发布／清除约束，性能单独测量 |
| 错误退出 | 保留必要语义 | 可精简 resident 接线，不把核退出当作 DMA 完成 |
| legacy 源码、存储与测试 | 保留现状 | 将来删除由独立任务处理，不计入本轮删行收益 |

未采纳优化和测量限制见 [性能调查][performance]。不把前缀上传、发现策略、DCache 门控和目录算法合为一次大替换；没有证据时不重做已可用部分。

## 4. 业务测试与验证

优先扩展现有测试，吸收能验证外部行为的部分，不复制双方全部内部协议测试或建立新测试框架。

| 对方参考 | 本地落点 | 独立价值 |
| -------- | -------- | -------- |
| `sdma_queue_reuse/` | `async_sdma_reuse/`、`sdma_shared_channel/` | 全 AIV sync-start、双 AIV 对同物理队列累计 66＋2、地址分组、序号和完整数据 |
| `signal_payload_defer/` | `async_peer_counter/test_signal_payload.py`，复用已有 pending-close 验证 | 多缓冲区完整 payload、2／4 rank、缓存预热／脏数据及 pending close |
| `defer_wait_visibility/` | `async_completion/test_predicate_visibility.py` | predicate-only、take／skip、重叠视图、Graph 实例与 EXIT 消费 |
| `sdma_defer_wait/test_error_close.py` | 现有 pending-close／真实 SDMA 失败测试 | post 后登记失败，原始错误、依赖不释放及存储保活 |
| `defer_wait_observer/` | 现有 observer／条件边界测试 | 仅补缺失的有符号 COUNTER 与 64 位完成记录边界 |

三组补充测试均已按精简后的 writer 接口编译通过；predicate-only／重叠视图／Graph EXIT 场景的模拟器运行通过。2／4 卡 payload、共享 SDMA 队列，以及提交后登记失败的原始错误与存储保留检查均已通过真机验收。“66＋2”不证明 66 个同时在途提交或完整 SDK 背压覆盖。保留本地独立多队列、原生失败和 L1 普通／Graph 测试。

legacy 专用、双调度器和原有布局测试保留，不以“将来删除”为理由现在删掉。只去掉本轮新增且冗余的测试，不为 legacy 扩充新的兼容验收矩阵。

验证按改动影响选择：

- 前缀改动复用 `test_hbg_async_plan.cpp`、`test_hbg_async_runtime.cpp` 等，检查 guard、边界、旧存储复用和零登记不读取旧 entries。
- writer、resolver、SDMA 若有实际变化，再运行各自已有相关目标；不无目的地反复全量编译。
- CPU／模拟器不能替代真实 SDMA、跨核缓存和原生失败的真机证据。
- 真机经现有设备隔离工具执行，不覆盖其他会话安装；复用已有 CPU、PTO 与 kernel 缓存，构建产物保持源码身份可追踪。
- 性能只比较同条件 L1 的 Qwen3-14B、CSA、SWA，使用相同输入、golden 和可比较的业务 kernel。分别记录 Host 初始化／上传与 Device 指标。
- 明确区分相关测试、完整测试、初筛、失败和未执行项，不能用撤回实验或旧版本结果证明新组合完成。

## 5. 实施顺序与完成判据

1. 保留原有 legacy 代码，核对当前 resident 改动是否混入旧调度删除或新增兼容接线；发现后撤回该部分。
2. 完成独立的控制前缀初始化／上传及相关边界验证。
3. 保留已有公共 writer、resolver、SDMA 和完成汇合；仅对 resident 内有明确收益的重复逻辑作局部精简。
4. 吸收并去重 payload、predicate、共享队列与提交后失败测试，验证实际外部行为。
5. 完成当前源码的隔离构建、相关 CPU／模拟器／真机验证及 L1 对照，更新实际代码分布和文档。

- [x] 现有 legacy 执行器、回退、调度循环、mailbox、等待列表、存储与测试已恢复。
- [x] 本轮没有为 legacy 新增桥接、资源声明、ABI 适配或新的兼容测试矩阵。
- [x] resident 保留未知 kernel 默认预留、零登记快路径、公共完成、SDMA 保护及必要错误退出语义。
- [x] 必要前缀上传正确，陈旧 entries 不会跨 run 生效。
- [x] 业务测试已吸收去重，相关真机与 L1 结果可追踪，未验证项如实列明。
- [x] 代码统计不把旧调度删除计入本轮收益；文档与实际状态一致。

共享 main 不切分支，不覆盖其他会话修改；提交和推送按另行明确指令执行。本轮已完成上述修正范围；验证产物保存在隔离目录，提交范围限于本轮相关文件。

## 6. 本轮实施结果与验收记录

生产代码相对起点共修改 10 个文件，新增 60 行、删除 185 行，**净减少 125 行**。原有 legacy AICPU 执行器、调度循环和回退入口保留；`src/common/`、A2/A3、`python/simpler/`、`simpler_setup/` 没有本轮生产改动。

| 生产落点（相对 HBG） | 改动 |
| -------------------- | ---- |
| `common/intrinsic.h`、`runtime/async_completion_writer.h` | 删除预约票据、序号与重复写入计数；上下文副本共享 count／error／sealed |
| `runtime/async_execution_context.h`、`runtime/backend/sdma/sdma_submission.h` | 接入简化后的 writer；SDK 提交前检查容量，返回后核对实际数量并登记 |
| `runtime/scheduler/scheduler_async_layout.h`、`scheduler_async_check.h`、`scheduler_async_runtime.h` | 登记快照从 32 缩至 16 字节；合并重复封口，保留零登记快路径与缓存发布顺序 |
| `host/async_plan.cpp/.h`、`host/runtime_maker.cpp` | guard 移至控制前缀，只初始化／上传控制前缀；Device 完整预留不变 |

私有布局版本为 11。`AsyncCtxStorage`、`LocalContext`、`DispatchPayload` 的尺寸和业务参数偏移保持不变；使用异步 API 的 kernel 与 runtime 已成套重编，旧异步二进制不能直接混搭。

测试分布：已有 C++ 单测共修改 8 个文件，净减少 58 行，主要删除预约票据专属断言；场景代码及其说明共涉及 26 个文件，净增加 1,430 行，集中在三个已有测试目录的业务边界，不新增内部协议框架。其他会话原有文件未计入这些数字。

- `async_completion/test_predicate_visibility.py` 与 `kernels/visibility/`：AIC／AIV、普通／Graph、take／skip，覆盖 predicate-only、重叠视图、两个 Graph 实例及 EXIT 后消费。
- `async_peer_counter/test_signal_payload.py` 与 `kernels/payload/`：六个选定的 2／4 卡场景，覆盖四份完整 payload、缓存预热、脏数据释放、AIC／AIV 标量与 TLOAD 消费、作用域退出和 retained 复用。
- `async_sdma_reuse/test_shared_queue_reuse.py` 与 `kernels/shared_queue/`：2／4 卡、诊断级别 0／4，共享队列 66＋2 次提交，以及真实 SDMA 提交后的登记溢出与存储保留。

验证结果：10 个相关 C++ 目标通过；模拟器通过 41 个独立测试及 3 个场景入口；真机通过 51 个独立测试及 5 个场景入口。场景入口内部包含参数组合，未将入口数与组合数混算。**这是相关范围验证，未运行全仓全量测试。** 正常 O3 构建通过，未引入开发期 outlining。初次构建中的旧字段引用及验证环境的旧绑定路径问题已修正；最终产物、源码指纹和结果均可追踪。既有 fork／nanobind 退出警告未作为功能失败处理，失败场景的未确认回收警告保留。

L1 对照在同一张卡上使用相同冻结业务 kernel、输入、golden 和当前 Python 绑定；两边均为正常 O3，每组 10 轮、不预热。全部 golden 通过，60 轮执行数量与逻辑身份一致。

| 模型 | 修改前 Device 中位数 | 修改后 Device 中位数 | 变化 |
| ---- | -------------------- | -------------------- | ---- |
| Qwen3-14B | 42.389515 ms | 42.259975 ms | −0.306% |
| CSA | 2.457039 ms | 2.486706 ms | +1.207% |
| SWA | 0.965093 ms | 0.963367 ms | −0.179% |

`chip.run.publish_image` 的 Host 调用中位数分别为：Qwen 19.044990 → 2.180790 ms、CSA 0.578040 → 0.198910 ms、SWA 0.181540 → 0.146190 ms。该阶段还包含其他发布工作，不能当成孤立的 memset 或 DMA 耗时；每个预规划物理执行少初始化／上传 2,048 字节条件区，Device 预留仍相同。以上为一次同卡顺序对照，不作统计显著性结论。

本机证据目录（相对仓库根目录）：`../logs/defer-wait-local-refactor-20261109/`。

- `production.patch`、`production-source-manifest.json`、`owned-files.json`：生产补丁、已验证源码身份及本轮文件范围。
- `production-runtime-build.log`、`production-runtime-build-result.json`：正常 O3／模拟器构建与实际二进制指纹。
- `writer-simplification-tests.log`、`seal-simplification-tests.log`、`production-native-tests.log`、`final-backend-tests.log`：相关 C++ 检查。
- `production-sim-acceptance/`、`production-sim-errors/`、`supplemental-sim/`：模拟器结果。
- `production-a5-acceptance/`、`production-a5-errors/`、`supplemental-a5/`：真机结果；框架把工作交给子进程，实际通过记录在 `pytest.log` 中，不能只看父进程的空 JUnit。
- `performance/L1-summary.json`、`performance/{control,candidate}/`：原始逐轮数据、golden 结果、身份核对、Host 阶段及 Device 指标。
- `run_production_acceptance.sh`、`run_production_errors.sh`、`run_supplemental_acceptance.sh`、`run_production_l1.sh`：实际验收命令；真机脚本经 `task-submit` 执行。

[local-head]: https://github.com/nalinaly/simpler/commit/7ffb4f74eea98ad6289146671030c5b22a06ae79
[peer-head]: https://github.com/poursoul/simpler/commit/bf0586267d713e9995d4ebb89451a76850117a59
[design]: a5-hbg-aicore-async-design.md
[comparison]: a5-hbg-defer-wait-comparison-for-poursoul.md
[performance]: ../investigations/2026-11-a5-defer-wait-performance.md
