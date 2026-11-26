# A5 HBG 日历的延迟完成

**共同前提：** 本页的 A5 调度设计遵守 [A5 硬件与调度统一原则](a5-hbg-graph-task/build-execute-separation.md#a5-hardware-principles)。与本原则冲突的旧设计按此修订；当前实现、拟修改布局和未来 STARS 后端分别标明。

A5 `host_build_graph` 的 kernel 可以登记完成条件后返回。物理 lane 继续执行其他任务；
逻辑任务及 Graph shell 的完成计数器，等待所有条件满足后才发布。
用户接口沿用 `get_async_ctx`、`save_expected_notification_counter` 和
`register_completion_condition`，无需预先声明 kernel 是否会 defer。

## 支持范围

| 项目 | 合同 |
| ---- | ---- |
| 任务形状 | 普通 AIC/AIV、SPMD、MIX 掩码 3/5/6/7、sync-start cohort、Graph body。Graph 子任务使用 dispatch predicate 时只支持单 block、单活跃角色。 |
| 每次物理 kernel 调用的条件数 | 每次最多 64 条。各引擎共用这份额度，超出时返回 `ASYNC_WAIT_OVERFLOW` |
| COUNTER | 对齐的 int32 通知，非负阈值；负样本（含 `INT32_MIN`）保持 pending |
| 其他可观察 token | SDMA event record、CCU event，以及 URMA 请求实际提交的 SQ 基本块和 CQ 条目目标 |
| 检查者 | 等待前驱的 AIC/AIV 或阶段出口的 lane 取得记录的所有权。AICPU 监督进度与终止 |
| 用户要求 | kernel 返回后不再修改本次登记；通知必须有可推进的实际生产者 |

HBG 按 SDMA 通道组分配提交互斥字。CPU 和设备使用相同的条件记录格式。

## 数据与资源

每个 lane 有一份 `AsyncGroupData` 注册区。每个注册区容纳 64 条条件。
Host 在每份 prepared payload 的 `GlobalContext` 中预填同一个 lane 的注册区地址、
边界和 run epoch；设备启动时初始化注册区并写回。正常 dispatch 不改这些地址。

注册区只在一次 kernel 调用期间归该 kernel 使用。kernel 返回后，lane 读取登记状态：
没有条件时，沿用同步完成；有条件时，先校验一次，再复制到独立的
`PercounterAsyncCompletion`。复制和注册区复位都完成写回后，注册区可供下一次调用使用。
后续条件检查不占用注册区，因此等待任务的数量不会阻止通知 kernel 执行。

完成记录按本次图的实际 `(task, block, subtask)` 调用数预留。每条记录内联 64 个条目，用一个 64 位位图记录尚未满足的条件。
每次调用至多产生一条记录，因此即使所有调用都 defer，也不会因固定 per-lane 槽数耗尽而等待。
Graph 的调用数来自每个实例的 body 形状；MIX 的每个活跃 subtask 单独计数。
记录只在实际发生 defer 时分配和初始化；Host 上传前缀不包含这些记录。

发送 lane 把完成记录挂到所属任务上。链接使用区域内的索引加一，零为空。
每条记录保存任务和 shell 的通知目标，以及实际发送 AIV 的身份。后继 lane
可以取得这条记录的所有权，继续检查原发送者登记的条件。

## 执行与发布

1. 参数准备、fanin 检查及谓词求值继续使用 prepared payload 协议。
2. kernel 使用既有接口向固定注册区登记；kernel 返回的 store barrier 先于状态读取。
3. 有条件时校验类型、对齐、阈值、保留字段、登记错误及未结束的 reservation。
4. 分配完成记录、复制条件和发布目标、复位注册区，并完成相关 cache writeback。
5. 后继 lane 先检查 DONE 计数。计数不足且存在 defer 记录时，lane 领取记录并检查剩余条件。没有后继的任务由阶段出口检查。
6. 条件全部满足后补上本调用的完成份额；若是 body sink，同时补上 shell 的份额。

COUNTER 的 pending 检查只做一次 GM 样本读取和有符号范围/阈值比较。
URMA 观察者按 token 中的 workspace 找到本次 Run 持有的域。它核对原发送者与行号，
独占该行后才读取 CQE，并更新 SQ tail、CQ tail 和 CQ doorbell。较晚的目标先完成时，
较早的目标会读到已经到达的队尾，两条记录仍分别发布一次完成。
同一个 group 的条件可逐个完成，不必每次重新检查已满足的条件。
不同 group 可乱序结算。

普通调用和 MIX 的每个活跃角色都贡献一份 DONE。登记了条件的角色在 kernel 返回时
留下这份额度，观察者在条件全部满足后补一次 DONE。如果这次调用是 Graph body 的
出口，观察者也在同一时刻补上 shell 的一份完成计数。

GM 同址观察和通知仍有并发串行代价，不能仅由一次检查很短推断全核开销很低。轮询复用已经确认的条件，检查请求率及实际发现延迟；单写者通知可用 st_dev，但跨核份额合并和 task 动态领取保留 atomic。必要 claim 分地址时按 128B 隔离，不能将相邻 uint64 或相邻64B行算作独立竞争单元。相关发布/消费、计数与后续判断的时序依赖都显式 DSB。

上述为后续修改要求，不代表当前 HBG 日历 的共享计数和紧凑链接已经拆散。STARS event/notify 是后续 Wait 的优先方案，空轮询用 nop 限流，并保留 early-notify、epoch、资源分配/释放和 payload 可见性；不把它或无通用分配方案的 SPR 当作 atomic claim 的替代。

## 日历中的等待与进度

Host 按阶段为每个物理 lane 生成执行顺序。一个 GET kernel 返回后，后继消费者在 DONE
处等待，承担独立工作的 lane 可以继续运行。多次调用分别持有完成记录，因此同一个 AIV
可以继续提交，后继也能从别的 lane 检查较早的请求。

如果等待通知的消费者和发通知的任务被排在同一个 AIV 上，按顺序等在消费者处会卡住两者。
对于绑定了 URMA 域、至少含三个任务的阶段，Host 检查各 lane 的排程。如果同一 lane
在两次集合之间有一条带前驱的普通行，且后面还排着普通任务，Host 就为本阶段保留原图的全部输入依赖。当前行受阻且仍有
已登记、未完成的异步调用时，这条 lane 才向后找一条所有输入均已就绪的行执行。
通知发出后，原消费者可以继续。每条行只执行一次。Graph body 先发布全部参数绑定，
各 lane 才能这样检查后面的行。检查到本 lane 的下一个 MIX、`sync-start` 集合或阶段出口为止。
排在消费者之后的集合可以等待消费者，消费者之前的通知仍能发出。

L2 泳道图在实际执行扫描的 lane 上显示 `Select a ready row`。起点是开始查找候选行，
终点是选出可执行行。若选中的是发通知的 kernel，它随后执行，图上分别记录扫描和 kernel 执行时间。

Graph 内的 defer 必须在本次 Graph 调用内完成。下一次 Graph 调用的入口会等待前一个
shell，不能让前一个 Graph 的条件依赖下一次 Graph 才会执行的通知。

## 进度、退出与错误

`async_registered` 在分配完成记录时递增，`async_settled` 在发布该份额后递增。
两者均单调；每条 defer 记录使用这两次共享进度原子操作。
任务和 shell 自身的计数器仍按现有发布规则更新。

AICPU 观察全部顶层 sink 的完成组件，再检查异步计数。
完成组件随本轮上传镜像预置，因此读到正值即属于本轮。
只有全部顶层 sink 已发布且 `async_registered == async_settled` 才允许结束。
登记分配发生在对应完成发布之前，因此末尾任务的条件不会漏查。
各 lane 处理自己的日历行和阶段出口。在收齐执行份额、完成记录全部结算后，AICPU 才结束本次 Run。
看门狗的快路径是 sink 进度与异步计数；两者同时停滞时扫描全图完成组件确认是否真的没有进展，
条件永不满足时报告运行时超时。

登记错误、非法 token、条件溢出沿现有错误通道返回，不能发布为成功。
异常终止不代表外部通信引擎已经完成；通信资源仍遵守平台的 teardown 证明。

如果 AIV 已经发出请求，原 Run 的核全部停下后仍有 CQE 待收，Host 继续独占通信域。
第一次检查时，Host 为这个 Run 启动一个后台线程。线程切到该 Run 的设备上下文，读取各域的软件队尾，
再启动一个 AIV 检查 CQE 的 owner 和状态。确认成功后，后台线程更新 CQ tail，由 AIV 写 CQ 门铃，再更新 SQ tail。
调用方反复 `poll` 时读取后台检查的结果。检查尚未结束就返回 `NOT_READY`，阻塞等待也可以等待这次检查。
设备无法启动这个 AIV、CQE 报错或仍未就绪时，Run 继续持有域和普通张量。
设备被隔离后，后台线程仍可查询已经提交的 AIV 命令。若要强制重置设备，Host 先停止这些线程，等正在执行的 ACL 调用返回后再重置。

HBG 在 bind 时发布 Graph Definition。AICPU 根据本次调用准备参数，日历 lane 执行已就绪的 Graph 行。

## 验收

`tests/st/a5/host_build_graph/defer_wait` 覆盖普通/SPMD/MIX/cohort/Graph/谓词形状、
超过原固定槽数的等待任务，以及 notifier 排在满候选窗口之后的情况。
手动用例使用 `--manual include`。`tests/st/a5/host_build_graph/urma_queues`
复用 TMR 的传输 kernel、完整数据校验和 SQ/CQ 计数断言。`urma_shapes` 检查
SPMD、MIX、Graph 和通知进度，`urma_host` 检查 Host 表初始化、远端读取和 CPU 更新。
CPU 测试还验证末尾 defer 的退出门槛和 group 乱序结算。

性能比较分别报告无 defer 的主线 HBG 日历 与当前实现，以及含 defer 的对应工作负载。
构建版本、设备、预热、轮数和时延层次必须匹配；历史测量不代替当前主线验证。
