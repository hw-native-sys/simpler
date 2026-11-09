# A5 HBG defer_wait 实现对比与合并取舍（poursoul 分支视角）

本文供双方合并实现时参考。**全文“本方”指 `poursoul/simpler` 的 `feat/a5-hbg-async-completion` 分支；“对方”指 `nalinaly/simpler` 的 `main` 分支。** 比较对象固定在下述提交，不随分支后续更新改变。

**本轮边界：改进 resident／AICore 自调度，不为 legacy AICPU 调度新增兼容适配；现有 legacy 代码与测试仍保留。legacy 将来删除，不属于本轮删除任务。** 下文源码事实和统计描述固定提交，取舍评价面向 resident 的长期维护价值，不能据此删除当前执行器、回退、等待列表或存储。

具体区分上游原有调度功能与新增异步适配：hw-sys main 原有的 AICPU 调度实现继续保留；nalinaly 为本轮 defer_wait 额外增加的 legacy 接线不构成必须维持的能力，可以核对来源后局部精简。这一取舍适用于其整轮 defer_wait 修改，不仅限于下一次提交。

核心判断：两边的异步完成模型已经趋同。本方的代码增量较小，Host 初始化与上传更节省；对方在 AICore 内部依赖释放复用、SDMA 多队列和失败退出方面有可吸收的实现。**跨 legacy／resident 共用 writer 不再单独构成对方的长期优势；普通完成／FINISH／Graph 共用 resolver 的价值仍然成立。** 合并应直接比较 resident 实现，保留必要能力并减少其中的重复主体；原有 legacy 不参与本轮清理。

## 1. 比较范围与已经达成的一致

核实日期：2026-11-09。此次结论来自固定提交的源码、差分和已有验证记录；没有新增真机性能对跑。

| 项目 | 本方：poursoul | 对方：nalinaly |
| ---- | -------------- | -------------- |
| 比较版本 | [Add: cover shared A5 SDMA queue wrap across retained runs][ours-head] | [Update: 降低 A5 defer_wait 零登记任务的调度开销][theirs-head] |
| 自身功能基线 | [Refactor: clarify AICPU initialization responsibilities][ours-base] | [Fix: preserve startup failure details after child exit][theirs-base] |
| 差分入口 | [本方相对基线的修改][ours-diff] | [对方相对基线的修改][theirs-diff] |

双方基线包含的上游历史不同，因此使用各自功能基线统计，不把两棵源码树的直接差分全部归因于 `defer_wait`。不计未提交文件，也不将同一份被互相吸收的测试视为两份独立验证。

评价指标面向 resident 的长期形态，本轮不执行 legacy 删除：

- 正确性：保留 AICore 普通与异步任务、真实 SDMA、跨核消费及失败后的存储保护。
- 维护成本：比较剩余执行路径、状态、独立概念和重复规则，不因“支持两套调度器”额外加分。
- 性能与存储：分别衡量零登记开销、异步发现与完成延迟、整图 L1 时间、Device 预留量和初始化上传量。
- 兼容范围：不为旧调度器新增兼容层或等价性要求；现有旧路径及仍使用的业务 API、kernel 参数 ABI、其他 runtime 共享定义保留，不顺带删除。

本方比上一轮比较版本新增了 16 个提交。以下旧判断已经不适用于本方当前代码：所有预留 group 都计入 pending、零登记任务仍必须进入异步完成、总是发布全部 64 条条件、负数 COUNTER 被直接视为错误。新版本已经调整这些行为。

当前共同模型如下：

- 未知 kernel 无需新增能力或资源声明，Host 默认按每次物理执行预留 64 条条件。
- kernel 登记条件后正常返回，释放执行核；其他空闲 AICore 可以领取内部 CHECK 工作。
- 仅实际非零登记计入异步等待；整项任务零登记时，在原生执行全部释放后走同步完成。
- 条件满足后，通过内部 FINISH 工作释放依赖。CHECK、FINISH 不新增业务 TaskId，不占用业务任务的协同准入名额。
- 覆盖普通任务、SPMD、MIX、sync-start 和当前单层 Graph；不是保存 kernel 栈后恢复执行，也不提供运行中抢占。
- 若所有核都在长时间执行 kernel，无法承诺固定的条件发现延迟；“等待不占住执行核”与“任意时刻都能立即检查”是两回事。
- 有符号 INT32 COUNTER 为负数时保持等待；同步 TPUT 返回的零 handle 可走同步退化，非空 TGET 返回零 handle 则拒绝。
- 均只发布实际登记的条件前缀；消费者的数据可见性仍需由实际消费核保证。

## 2. 代码规模与分布

统计口径为 `git diff --no-renames --numstat <自身基线> <比较版本>` 的物理行，包含注释和空行。下表不是现有文件总行数，也不是历史开发工作量或质量评分。

| 类别 | 本方文件数 | 本方新增 / 删除 / 净增 | 对方文件数 | 对方新增 / 删除 / 净增 |
| ---- | ---------- | ---------------------- | ---------- | ---------------------- |
| 生产代码 | 37 | 2,777 / 293 / **2,484** | 58 | 4,660 / 594 / **4,066** |
| 测试及测试支持 | 103 | 7,971 / 23 / **7,948** | 116 | 12,036 / 33 / **12,003** |
| 文档 | 13 | 906 / 9 / 897 | 8 | 619 / 0 / 619 |
| benchmark 工具及清单 | 2 | 424 / 1 / 423 | 0 | 0 / 0 / 0 |

生产代码进一步按职责归类，均为净增行数：

| 模块 | 本方 | 对方 | 差异主要来源 |
| ---- | ---- | ---- | ------------ |
| 调度器与完成处理 | 1,582 | 2,347 | 发现预算、执行绑定、完成处理及公共 resolver 组织 |
| 条件登记与后端 | 489 | 813 | 公共 writer、SDMA 统一提交与多队列处理 |
| Host 规划与集成 | 284 | 401 | 布局、初始化、上传与运行时接入 |
| AICore 执行器 | 62 | 60 | 上下文绑定、返回处理与消费前可见性 |
| HBG 其他 | 31 | 109 | 公共格式和配套接口 |
| `src/a5/platform/` | 2 | 153 | 原生错误传递与退出结果处理 |
| `src/common/` | 0 | 158 | 退出凭据、Worker 与最后使用检查 |
| DFX 工具 | 34 | 25 | 异步诊断解析 |
| **合计** | **2,484** | **4,066** | |

对方生产代码净增约为本方的 1.64 倍，测试净增约为 1.51 倍。但生产代码多出的 1,582 行中，1,282 行、约 81% 位于 A5 HBG 内部。**不能再把当前主要差异概括为“对方多了一整套外部资源声明框架”。** 对方已经清理那类框架；双方在 `python/simpler/` 下均无生产代码差分。

本方修改较少，有利于阅读和审查；对方在 AICore 内部的流程复用仍有价值，跨旧调度器复用则不再作为长期收益。上述行数是固定提交快照；本轮统计只计算 resident 实际改动，不能把未来删除 legacy 的行数作为本轮收益。

## 3. 主要实现差异及取舍

### 3.1 条件登记：收敛为一份 resident writer

本方在 `async_condition_writer.h` 提供 resident `AsyncWriter`，旧 AICPU 登记仍保留自己的追加路径。当前 `get_async_ctx` 已映射真实的 count、error、entries、capacity，并保留上下文身份检查和粘性错误。[本方登记代码][ours-api]

对方在 `async_completion_writer.h` 共用 reserve、append、commit、finish 等登记函数；resident 另用 seal 封口，旧路径仍按自己的 API 边界发布，并不经过完整 resident 封口与 CHECK 发布流程。resident 上下文副本共享可变预约状态，不能绕过未完成预约。[对方公共 writer][theirs-writer]

**不再因跨 legacy／resident 共用而优先吸收对方 writer。** 两边均应围绕 resident 内的容量预约、实际写入、错误传播和封口评价复杂度，COUNTER、SDMA 等入口仍应共用必要规则。

本方可以保留独立 resident writer，对方也可保留已有公共主体；没有必要为合并名称重写正确代码。现有 legacy 追加路径、slab 发布和上下文回退保留，本轮不为它们新增适配，也不因未来删除计划现在移除。共享可变登记状态和提交真实搬运前确认登记容量仍属于 resident 正确性要求。

### 3.2 ABI：大小相同不代表上下文布局兼容

本方 `GlobalContext` 为 48 字节，其中携带 resident 异步描述；对方保留 4 字节 `GlobalContext`，在 `DispatchPayload` 的 offset 472 放异步上下文地址、offset 480 放 32 字节执行上下文。两边 payload 都为 512 字节，业务参数都从 offset 64 开始。[本方布局][ours-payload]、[对方布局][theirs-payload]

因此使用异步 API 的 kernel 与 runtime 二进制不能直接混搭。公共参数前缀不变，不等于所有纯计算旧 kernel 都不兼容；应区分这两件事。**合并先选定一个内部 ABI，并让相关 kernel 与 runtime 成套构建，不长期保留两套上下文协议。**

本轮只按 resident 实际需要调整上下文和 ABI，不为 legacy 额外设计兼容层；现有旧 slab 回退、双模式识别和业务参数前缀也不因此直接删除。已有接口保持可用，确需调整的 resident 产物成套构建，不把旧调度器迁移混入本轮。

### 3.3 Host 规划与上传：本方的明确优势

本方规划已经从逐物理执行的临时描述，收敛为逐逻辑任务规划。初始化和上传只覆盖 `initialized_bytes` 前缀，包含必要布局、活动标记、映射、任务记录、join、目录、guard 和连续 group header；lease、进度及条件存储的部分后缀在首次发布前初始化。[本方 Host 规划][ours-host]

对方直接遍历已有 `SchedulerTaskMetadata`，同样没有必要引入逐执行临时向量；但当前仍清零、初始化并上传整个 async image。[对方 Host 规划][theirs-host]

**共同实现应保留本方的前缀初始化和上传思路。** 这减少 Host 初始化和 H2D 传输，不要求 PyPTO 补充资源声明，也不改变默认 64 条的预留契约。

移植时有一个具体陷阱：对方 SDMA guard 位于条件 entries 区之后，不能直接把上传长度截到 `entries_offset`。必须先整理布局或单独上传必要控制区域，并保证任何可发现记录在发布前已初始化。可先跳过尚未使用的条件数组，避免一次引入过多延迟初始化状态。

### 3.4 存储：少上传与少预留是不同收益

每次物理执行按 64 条条件计算，以下仅统计固定记录和条件数组，不含任务级记录、目录及区域对齐间隙：

| 组成 | 本方 | 对方 |
| ---- | ---- | ---- |
| 身份、header 与 writer | 128 B | 40 + 64 B |
| lease | 128 B | 128 B |
| CHECK 进度 | 128 B | 32 B |
| 64 条条件 | 2,048 B | 2,048 B |
| **小计** | **2,432 B** | **2,312 B** |

对方这一层每执行少 120 B；本方则减少初始上传量。任务级 FINISH、join、映射组织仍有差别，整图内存还取决于任务与执行数量，不能拿上述小计代替完整内存测量。**保留前缀上传不等于取消 Device 上的预留容量。**

这些记录服务于 resident，应按实际布局改动核对整图预留量、初始化量和访问成本；本轮不计未来 legacy 删除收益。

### 3.5 执行绑定与零登记返回

本方进入 kernel 前按任务映射计算 group，将描述放入上下文，避免提前读取 group header；`get_async_ctx` 或返回 seal 时验证，返回还会重新映射 group。零登记仍写回可变 header 并执行相应屏障。[本方执行控制][ours-control]

对方在最终准入和公共 DCache 处理后绑定一次，记录 group index 与 slot generation；返回时复用绑定结果。零登记保留 writer 错误、未完成预约、sealed 等检查，跳过条件身份与进度处理，并复用已有退出屏障；不可变布局也缓存在核本地。[对方执行处理][theirs-execution]

本方减少进入前的读取，对方减少返回重算和空路径工作。**可吸收“一次绑定、返回复用”和零登记快路径，但不要直接宣称一定更快。** 对方此前尝试过额外 task-map 缓存、固定步长等方案，收益不足，合并不应重新堆叠类似缓存。[对方性能调查][theirs-perf]

这项取舍不受 legacy 删除影响：普通计算任务仍经 AICore 调度，零登记仍需便宜地返回。只精简 resident 内真实重复的工作，保留原有模式选择与回退；有效上下文、容量和未结束预约等检查按实际需要保留。

### 3.6 完成计数：语义已经接近，适合统一表达

本方使用 P（仍待完成）、R（累计实际登记）和 closed（原生计算已结束）：登记时增加 P、R，CHECK 完成减少 P，原生 `TASK_LAST` 关闭计算。R 为零时回到同步完成。[本方 pending][ours-pending]

对方使用 P、原生释放字段、R 和 FAILED；当前原生释放字段初值为 1，物理 block/cohort 计数复用原生机制。仅 `TASK_LAST` 判断实际登记历史，整项零登记走普通 TaskDone/refill，跳过无用 join 更新。[对方 join][theirs-join]

**建议用 P／R／closed 的语言说明协议，保留对方零登记快路径及必要失败状态，最终只维护一份 join。** 不再把“一个原生结束义务”解释成另一套物理 block 计数，也不为同一状态保留多个概念。

### 3.7 依赖释放：对方更充分复用公共流程

本方普通同步任务仍在 `scheduler_ready.h` 保留完整完成循环，FINISH／Graph 控制节点使用有预算的 `scheduler_completion_resolver.h`。底层 ready 发布有共享，但上层依赖释放主体仍有两份。[本方 ready][ours-ready]、[本方 resolver][ours-resolver]

对方普通同步完成以大预算调用公共 step，FINISH 以小预算调用同一主体，同时保留 direct refill、MIX 前缀和 Graph continuation。[对方 ready][theirs-ready]、[对方 resolver][theirs-resolver]

**仍建议吸收对方这一结构。** 普通完成、异步 FINISH 和 Graph 控制节点完成都存在于 AICore 调度器内部，未来删除 legacy 也不会消除这三类调用。本方现状降低了最初改动同步路径的风险，但以后修改依赖释放规则仍需维护两份上层循环。

共用主体时，普通完成使用局部游标和大预算，在当前调用内处理到底；FINISH 使用保存的游标和小预算分段处理。不能把普通完成强制转换成 FINISH 工作。续接的是调度器的依赖释放过程，kernel 栈不参与；应保留 direct refill、MIX 发布及 Graph 行为，并验证公共状态机的同步路径开销。

### 3.8 CHECK／FINISH 发现：少探测与更快发现之间的选择

本方本地异步状态为 56 B。尚未观察到活动、且没有业务返回等触发条件时，每 16 次服务机会进行一次冷态探测；一旦观察到活动便单调记住。单次至多读一个目录字、尝试一个候选，CHECK／FINISH 交替，起始游标已按 worker 分散。[本方调度状态][ours-state]

对方以单轮共享预算限制服务，ready 持续存在时按业务返回计数触发，空闲或准入被拒时也允许服务；预算为 4 个目录字、2 次 lease 探测、2 次 CAS、8 个工作单位。一类为空时可在预算内尝试另一类；冷态没有本方的 16 次间隔。[对方发现流程][theirs-discovery]

本方降低纯同步空闲开销，对方更积极处理稀疏或竞争目录。**16 次是服务机会，不是固定时间。** 延迟还取决于 kernel 长度和可用核。保留哪种策略应单独看首次异步发现延迟、普通 ready 压力下的进展及空闲成本，不增加一批用户可调参数。

### 3.9 目录位更新：有界 CAS 与单次原子更新

本方通过原子读取和有界 strong CAS 实现置位、清位。按 64 位目录的界限，单次最多 127 次 CAS，加初始读取共 128 次原子操作；最后一次 CHECK 清位再发布 FINISH，理论上可达 256 次。这个界限不等于实际常态开销，也不包含在“8 个工作单位”之内。[本方目录][ours-directory]

对方按单次生命周期、唯一贡献者约束，用加减单个 bit 的 `fetch_add` 完成更新，一次 RMW 即可。但重复贡献检查在修改之后，违规可能通过进位、借位破坏邻位，必须使整次运行失败，不能尝试补偿后继续。[对方目录][theirs-directory]

本方更接近通用位操作，对方操作数更固定、依赖的唯一发布／清除约束更强。**不能把对方算法当通用 bitmap 工具移植。** 合并前确认每个目录 bit 在退休前只经历一次合法生命周期，再结合竞争场景决定。

### 3.10 消费者 DCache：统一入口与活动后启用

本方保留运行入口原有 acquire，把每次执行额外的完整 DCache 处理放在异步活动已激活之后；发布侧在 CHECK 可见前发布活动标记并执行屏障，消费侧在执行入口观察该标记。对方在每个非空执行的最终入口统一处理。[本方执行器][ours-executor]、[对方执行器][theirs-executor]

本方可减少全同步场景的额外处理，对方契约更直接。合并若采用活动门控，需要确认 predicate-only tensor、Graph EXIT、换核消费、别名及脏数据所有权场景都受覆盖；**检查核看到条件不替代消费核的 acquire。** 条件读取指令的 `ld_dev` 差异不纳入本轮评估，也不据此提出修改。

### 3.11 SDMA：本方默认通道保护与对方多队列公共提交

本方 resident 默认 SDK 提交由物理簇锁保护：同簇两个 AIV 共享队列状态时串行提交；保护内取得完成记录，成功时发布 SDK 状态，错误时将 guard 标记为不可继续使用。非 resident 旧路径仍沿用原契约。[本方 SDMA][ours-sdma]

对方让两类入口共用 `sdma_submission.h`，支持默认 1 条及多条队列，按实际物理队列组加锁；多锁按固定顺序获取、反序释放，部分获取失败且尚未产生 SDK 副作用时可释放重试。提交前预约条件容量，并在解锁前发布 SDK 元数据。[对方 SDMA][theirs-sdma]

**仍建议吸收对方公共提交主体和物理队列归属处理，保留本方同步 TPUT 退化与共享队列复用测试。** 两类 API 入口分别接受 `AsyncCtx` 或从参数取得上下文，都可用于 resident；入口数量不等于调度器数量。保留有实际调用方的薄适配，提交、保护和 SDK 状态发布只维护一份。

原有 legacy 无 guard 兼容分支保留，不为其新增保护协议；双 AIV 共享 SDK 队列、多队列锁顺序和提交后错误仍按 resident 自身问题处理。同步 TPUT 退化继续保留。

### 3.12 失败与存储释放：对方更明确，本方也已有保护

本方通过既有 CANN／AICPU 错误、Host fence 失败与 domain 隔离保留相关存储。新增 `test_error_close.py` 在真实 TGET 提交后制造登记溢出，检查原始设备错误、未释放依赖、close 的回收未确认、domain 保留以及后续使用被拒绝。[本方失败测试][ours-error-test]

对方进一步区分逻辑失败、AICore 退出凭据和外部访问的最后使用：24 B 凭据带版本、epoch 和退出标志；只有满足指定逻辑失败及有效退出凭据时才允许对应传输层正常返回，逻辑错误仍保留。Host 结合物理完成与成功终态决定可否复用，失败或证明缺失则保留所有者并使域不可用。[对方传输结果][theirs-transport]、[对方退出凭据][theirs-retirement]

**不能说本方没有失败保护，也不能把对方剩余公共代码都认定为冗余资源框架。** 本方依托既有通用失败隔离，对方增加了更清楚的分类与释放条件；代价是跨层接入和维护更多状态。合并应保留必要的最小失败、退出和最后访问处理，继续使用现有运行域所有者，不重新扩展资源声明。

本轮保留 AICPU 调度，同时必须保留 resident 的 AICPU 启动和退出接线。例如对方 `aicpu_async_transport_enabled()` 明确要求 AICore 调度启用，它服务于 resident 运行的错误返回。应按职责判断可删范围，不能按 `aicpu/` 或 `src/common/` 目录整体删除。[对方退出接线][theirs-aicpu-executor]

需要保留的是正确性要求，不必照搬对方当前 24 B 凭据及全部跨层函数的形式。若既有运行域机制可以表达同样的退出和释放边界，应优先复用并减少状态。

双方都不能由“核已退出”或“计算流已 drain”推导出失败的外部 DMA 已经不再访问存储。保守失败路径可能长期保留内存或使运行域不可复用，这是正确性和失败恢复能力之间仍需明确的取舍。

## 4. 测试价值与证据边界

### 4.1 优先合并能验证外部行为的用例

双方已具备普通、SPMD、MIX、sync-start、Graph，以及 0／1／63／64／65 条条件、默认未知 kernel、容量错误、等待进展等覆盖。新增价值主要在真实数据可见性、共享物理队列和失败后的存储生命周期。

| 用例方向 | 本方值得保留的覆盖 | 对方值得吸收的覆盖与合并取舍 |
| -------- | ------------------ | ---------------------------- |
| 完整 payload 可见性 | `signal_payload_defer/`：4 个独立缓冲区，2／4 rank，AIC／AIV 标量及 AIV TLOAD／TSTORE，缓存预热、脏数据、保留环境多轮执行 | 与现有业务依赖测试整合，继续检查完整 payload，不能只看通知值 |
| 条件不在任务数据参数中 | `defer_wait_visibility/`：predicate-only tensor、take／skip、重叠视图、Graph 多实例及 Graph EXIT 同步读取 | 吸收对方 L1 普通可见性和 Graph 实例用例，覆盖公共入口是否遗漏 |
| pending 时关闭 | `signal_payload_defer/test_pending_close.py`：关闭后继续核对完整 payload | 与对方原生失败、最后使用案例互补，区分成功等待结束与失败隔离 |
| 提交后发生错误 | `sdma_defer_wait/test_error_close.py`：真实搬运后登记失败 | 保留对方真实 SDK 失败和退出结果检查，不能仅用伪造错误码替代 |
| 共享队列回绕与复用 | `sdma_queue_reuse/`：全 AIV sync-start，每 AIV 33 次提交，双 AIV 对同一队列累计 66 次；保留环境再各提交一次，即再 2 次 | 对方 `async_sdma_reuse/` 已有 66＋2 累计复用及独立共享通道测试，应合并去重，保留实际记录地址分组、序号连续、完整数据及 guard 检查 |
| 多队列与不同调用入口 | 本方重点覆盖默认共享队列和同步退化 | 保留对方多队列测试，并覆盖继续使用的 resident API 入口；这与默认通道双 AIV 竞争不同 |
| 观察边界 | `defer_wait_observer/`：有符号 COUNTER 与 64 位 post 边界 | 数值边界测试不能代替真实 DMA 与跨核可见性测试 |

本方共享队列用例还覆盖 2／4 rank 和诊断级别 0／4。**“66＋2”证明累计回绕和保留环境中的连续性，不证明同时有 66 个未完成提交，更不能据此宣称 SDK 背压已经验证。** [本方测试目录][ours-tests]、[对方测试目录][theirs-tests]

内部测试只保留能捕获独特协议错误的部分，例如登记溢出、重复完成、漏发布和失败后的错误释放；不把双方所有结构布局断言、源码文本检查、同义状态测试全部并入。合并后的测试应尽量允许实现重构而保持业务期望不变。

原有 legacy 专用测试、双调度器测试和布局断言保留，本轮不扩充新的 legacy 兼容矩阵，也不删除旧实现及其测试。AICore 内部的零登记、普通同步任务、同步 TPUT 退化和启动／退出／错误测试仍需保留；“同步”不等于 legacy。

### 4.2 性能和验证结论的限制

- 本次比较未重新运行 CPU、模拟器或 NPU 测试；新增用例的存在、提交说明中的“通过”及文档中的验收目标，均不当作本轮实测通过。
- 尚无两边最新版本、同机同数据同构建条件的 Qwen3-14B／CSA／SWA 对跑，因此不能认定本方或对方整体更快。
- 对方开发参考中，最近一次三项完整正式对比属于内部 v12：Qwen3-14B 为 52.017 → 42.420 ms，CSA 为 2.257 → 2.485 ms，SWA 为 0.897 → 0.994 ms，口径为同卡十轮 L1 Device 时间中位数。箭头是该轮旧基准到候选版本，**不是本方到对方**。[对方验收记录][theirs-design]
- 对方当前提交来自后续 v14 方案，已有相关功能和 CSA／SWA L1 验证，但最新组合未再做三项完整正式对比；不能用 v12 的完整验证替代当前版本的证明。
- 双方 PTO-ISA 固定版本不同；如果后续开展公平性能对跑，需要固定依赖、编译选项、输入和采样口径。当前对方提交没有合入另一个会话的编译优化，本报告也不建议把它混入异步实现合并。

## 5. 从本方出发的合并建议

### 5.1 resident 改造的取舍边界

| 项目 | 本轮建议 | 边界 |
| ---- | -------- | ---- |
| 条件登记 writer | 比较 resident 主体复杂度，保留有价值的规则复用 | 不因跨调度器复用加分，不删原有 legacy |
| 上下文与 ABI | 只按 resident 需求调整 | 不新增旧路径适配层，不拆除现有回退和业务前缀 |
| 普通完成／FINISH／Graph resolver | 吸收对方公共主体 | 三者均属 resident，保留同步快路径 |
| SDMA 提交及队列保护 | 吸收对方公共提交和多队列处理 | 原有 legacy 分支保留，避免扩充兼容链路 |
| Host 规划与前缀上传 | 保留本方思路 | 全量存储仍预留，必要控制区完整 |
| 完成汇合 | 统一 P／R／closed 的表达 | 不增加重复计数或状态表 |
| 发现、目录、DCache | 分别证明正确性并测量 | 不新增开关，不评估 ld_dev |
| 错误退出 | 保留必要语义，形式可精简 | 不恢复资源声明框架，不把核退出等同于 DMA 完成 |
| 测试 | 吸收独立业务边界，去重本轮新增内容 | 旧测试保留，不新增 legacy 兼容矩阵 |
| 代码规模 | 统计 resident 实际改动 | 不将未来删除 legacy 计入收益 |

### 5.2 建议实施顺序

1. **明确本轮范围。** 现有 legacy 执行器、回退、循环、mailbox、等待列表、共享存储及相关测试保留，不为其新增兼容适配。
2. **整理 resident 公共主体。** 根据实际收益合并 writer、resolver 和 SDMA 提交，保留未知 kernel 默认预留和零登记快路径；已有正确实现不因命名不同重写。
3. **接入前缀上传与必要退出处理。** 控制区完整初始化，复用既有运行域生命周期，不改变原有 legacy Graph／形态回退及其支持范围。
4. **去重并吸收业务测试。** 聚焦换核消费、predicate、真实共享队列、默认未知 kernel、零登记和提交后失败。
5. **独立验证性能。** 用同条件 L1 数据验证 resident 改动，不把多种策略打包替换，不合入其他会话的编译优化。

共同实现应让主要流程直接可读：kernel 登记并返回，AICore 检查条件，满足后复用普通完成流程。本轮不引入资源所有者、绑定、消费准备框架或冗余配置；legacy 的将来删除另行处理。

[ours-head]: https://github.com/poursoul/simpler/commit/bf0586267d713e9995d4ebb89451a76850117a59
[theirs-head]: https://github.com/nalinaly/simpler/commit/7ffb4f74eea98ad6289146671030c5b22a06ae79
[ours-base]: https://github.com/poursoul/simpler/commit/fca73a29b829e2b10d8199330bc337314dd57e02
[theirs-base]: https://github.com/nalinaly/simpler/commit/8001268ba9e82fe11c8627c89297646f09b60e8d
[ours-diff]: https://github.com/poursoul/simpler/compare/fca73a29b829e2b10d8199330bc337314dd57e02..bf0586267d713e9995d4ebb89451a76850117a59
[theirs-diff]: https://github.com/nalinaly/simpler/compare/8001268ba9e82fe11c8627c89297646f09b60e8d..7ffb4f74eea98ad6289146671030c5b22a06ae79
[ours-api]: https://github.com/poursoul/simpler/blob/bf0586267d713e9995d4ebb89451a76850117a59/src/a5/runtime/host_build_graph/runtime/async_kernel_api.h
[theirs-writer]: https://github.com/nalinaly/simpler/blob/7ffb4f74eea98ad6289146671030c5b22a06ae79/src/a5/runtime/host_build_graph/runtime/async_completion_writer.h
[ours-payload]: https://github.com/poursoul/simpler/blob/bf0586267d713e9995d4ebb89451a76850117a59/src/a5/runtime/host_build_graph/runtime/dispatch_payload.h
[theirs-payload]: https://github.com/nalinaly/simpler/blob/7ffb4f74eea98ad6289146671030c5b22a06ae79/src/a5/runtime/host_build_graph/runtime/dispatch_payload.h
[ours-host]: https://github.com/poursoul/simpler/blob/bf0586267d713e9995d4ebb89451a76850117a59/src/a5/runtime/host_build_graph/host/async_completion_plan.cpp
[theirs-host]: https://github.com/nalinaly/simpler/blob/7ffb4f74eea98ad6289146671030c5b22a06ae79/src/a5/runtime/host_build_graph/host/async_plan.cpp
[ours-control]: https://github.com/poursoul/simpler/blob/bf0586267d713e9995d4ebb89451a76850117a59/src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_async_control.h
[theirs-execution]: https://github.com/nalinaly/simpler/blob/7ffb4f74eea98ad6289146671030c5b22a06ae79/src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_async_execution.h
[ours-pending]: https://github.com/poursoul/simpler/blob/bf0586267d713e9995d4ebb89451a76850117a59/src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_async_pending.h
[theirs-join]: https://github.com/nalinaly/simpler/blob/7ffb4f74eea98ad6289146671030c5b22a06ae79/src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_async_join.h
[ours-ready]: https://github.com/poursoul/simpler/blob/bf0586267d713e9995d4ebb89451a76850117a59/src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_ready.h
[ours-resolver]: https://github.com/poursoul/simpler/blob/bf0586267d713e9995d4ebb89451a76850117a59/src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_completion_resolver.h
[theirs-ready]: https://github.com/nalinaly/simpler/blob/7ffb4f74eea98ad6289146671030c5b22a06ae79/src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_ready.h
[theirs-resolver]: https://github.com/nalinaly/simpler/blob/7ffb4f74eea98ad6289146671030c5b22a06ae79/src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_completion_resolver.h
[ours-state]: https://github.com/poursoul/simpler/blob/bf0586267d713e9995d4ebb89451a76850117a59/src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_async_state.h
[theirs-discovery]: https://github.com/nalinaly/simpler/blob/7ffb4f74eea98ad6289146671030c5b22a06ae79/src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_async_discovery.h
[ours-directory]: https://github.com/poursoul/simpler/blob/bf0586267d713e9995d4ebb89451a76850117a59/src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_async_directory.h
[theirs-directory]: https://github.com/nalinaly/simpler/blob/7ffb4f74eea98ad6289146671030c5b22a06ae79/src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_async_directory.h
[ours-executor]: https://github.com/poursoul/simpler/blob/bf0586267d713e9995d4ebb89451a76850117a59/src/a5/runtime/host_build_graph/aicore/aicore_executor.cpp
[theirs-executor]: https://github.com/nalinaly/simpler/blob/7ffb4f74eea98ad6289146671030c5b22a06ae79/src/a5/runtime/host_build_graph/aicore/aicore_executor.cpp
[ours-sdma]: https://github.com/poursoul/simpler/blob/bf0586267d713e9995d4ebb89451a76850117a59/src/a5/runtime/host_build_graph/runtime/backend/sdma/sdma_completion_kernel.h
[theirs-sdma]: https://github.com/nalinaly/simpler/blob/7ffb4f74eea98ad6289146671030c5b22a06ae79/src/a5/runtime/host_build_graph/runtime/backend/sdma/sdma_submission.h
[ours-error-test]: https://github.com/poursoul/simpler/blob/bf0586267d713e9995d4ebb89451a76850117a59/tests/st/a5/host_build_graph/sdma_defer_wait/test_error_close.py
[theirs-transport]: https://github.com/nalinaly/simpler/blob/7ffb4f74eea98ad6289146671030c5b22a06ae79/src/a5/platform/onboard/aicpu/async_transport_result.h
[theirs-retirement]: https://github.com/nalinaly/simpler/blob/7ffb4f74eea98ad6289146671030c5b22a06ae79/src/common/host_build_graph/async_retirement.h
[theirs-aicpu-executor]: https://github.com/nalinaly/simpler/blob/7ffb4f74eea98ad6289146671030c5b22a06ae79/src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp#L774
[ours-tests]: https://github.com/poursoul/simpler/tree/bf0586267d713e9995d4ebb89451a76850117a59/tests/st/a5/host_build_graph
[theirs-tests]: https://github.com/nalinaly/simpler/tree/7ffb4f74eea98ad6289146671030c5b22a06ae79/tests/st/a5/host_build_graph
[theirs-perf]: https://github.com/nalinaly/simpler/blob/7ffb4f74eea98ad6289146671030c5b22a06ae79/docs/investigations/2026-11-a5-defer-wait-performance.md
[theirs-design]: https://github.com/nalinaly/simpler/blob/7ffb4f74eea98ad6289146671030c5b22a06ae79/docs/zh-cn/a5-hbg-aicore-async-design.md
