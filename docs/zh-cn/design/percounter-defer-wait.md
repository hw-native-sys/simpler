# percounter defer completion

A5 `percounter` 的 kernel 可以登记完成条件后返回。物理 lane 继续执行其他任务；
逻辑任务及 Graph shell 的完成计数器，等待所有条件满足后才发布。
用户接口沿用 `get_async_ctx`、`save_expected_notification_counter` 和
`register_completion_condition`，无需预先声明 kernel 是否会 defer。

## 支持范围

| 项目 | 合同 |
| ---- | ---- |
| 任务形状 | 普通 AIC/AIV、SPMD、MIX 掩码 3/5/6/7、sync-start cohort、Graph body、dispatch predicate |
| 每次 kernel 调用的条件数 | 0 到 4；超出时返回 `ASYNC_WAIT_OVERFLOW` |
| COUNTER | 对齐的 int32 通知，非负阈值；负样本（含 `INT32_MIN`）保持 pending |
| 其他可观察 token | SDMA event record 的单调阈值、CCU event 的消费式检查 |
| 检查者 | 执行该 kernel 的 lane；AICPU 仅监督进度与终止 |
| 用户要求 | kernel 返回后不再修改本次登记；通知必须有可推进的实际生产者 |

SDMA 提交互斥字没有在本 runtime 中配置；能观察一个已有 token 不表示支持
所有产生该 token 的提交接口。CPU 和设备共用当前记录格式，没有历史格式分支。

## 数据与资源

每个 lane 有一份固定的 `AsyncGroupData` 注册区和四个 `DeferredCompletionEntry`。
Host 在每份 prepared payload 的 `GlobalContext` 中预填同一个 lane 的注册区地址、
边界和 run epoch；设备启动时初始化注册区并写回。正常 dispatch 不改这些地址。

注册区只在一次 kernel 调用期间归该 kernel 使用。kernel 返回后，lane 读取登记状态：
没有条件时，沿用同步完成；有条件时，先校验一次，再复制到独立的
`PercounterAsyncCompletion`。复制和注册区复位都完成写回后，注册区可供下一次调用使用。
后续条件检查不占用注册区，因此等待任务的数量不会阻止通知 kernel 执行。

完成记录按本次图的实际 `(task, block, subtask)` 调用数预留，每条 192 字节。
每次调用至多产生一条记录，因此即使所有调用都 defer，也不会因固定 per-lane 槽数耗尽而等待。
Graph 的调用数来自每个实例的 body 形状；MIX 的每个活跃 subtask 单独计数。
记录只在实际发生 defer 时分配和初始化；Host 上传前缀不包含这些记录。

每条记录保存任务、shell、sink 标志、尚未满足的条件位图和条件副本。
每个 lane 独占自己的链表，链接使用区域内的索引加一，零为空。
无跨 lane 的条件记录访问或复用；条件满足时从链表摘除。

## 执行与发布

1. 参数准备、fanin 检查及谓词求值继续使用 prepared payload 协议。
2. kernel 使用既有接口向固定注册区登记；kernel 返回的 store barrier 先于状态读取。
3. 有条件时校验类型、对齐、阈值、保留字段、登记错误及未结束的 reservation。
4. 分配完成记录、复制条件和发布目标、复位注册区，并完成相关 cache writeback。
5. lane 在 checkpoint 轮询自己的链表。无 pending 记录时只检查本地空链表标记。
6. 条件全部满足后补上本调用的完成份额；若是 body sink，同时补上 shell 的份额。

COUNTER 的 pending 检查只做一次 GM 样本读取和有符号范围/阈值比较。
结构和 token 合法性在登记返回时校验一次，轮询时不重复校验。
同一个 group 的条件可逐个完成，不必每次重新检查已满足的条件。
不同 group 可乱序结算。

普通调用直接扣住自己的发布份额。MIX 沿用 rendezvous 的整块发布：
登记了条件的 subtask 在报告 finished 前先减去自己的份额，条件完成时再补回。
cohort 的物理 block 完成与逻辑条件完成分开计数，因此后续 cohort 可以继续。

## 满候选窗口的进度

仅按 b-level 排序并保留有限候选，不能保证 defer 图的执行进度。例如一个 waiter 后面
有许多依赖它的消费者，而独立 notifier 排在消费者之后；所有窗口都被消费者占满后，
notifier 仍未领取。

窗口阻塞且已经观察到本次图使用 defer 时，lane 把一个阻塞位置放到自己的 GM 停放链表，
释放其 prepared payload 槽，继续领取后面的普通任务。停止领取新位置后，lane 重查停放位置，
只恢复已经满足 fanin 的任务。位置始终由原领取 lane 持有，不重复执行；恢复时重新准备参数。

停放链接按实际普通 order 位置预留，每个位置一个 uint64，不上传、不预清零。
相邻位置可能属于不同 lane，链接以原子交换写入，避免 cache line 写回覆盖邻居。
普通图只在原有的周期检查点探测是否出现 defer；没有每次阻塞轮询都读取两个共享计数器。

## 进度、退出与错误

`async_registered` 在分配完成记录时递增，`async_settled` 在发布该份额后递增。
两者均单调；每条 defer 记录使用这两次共享进度原子操作。
任务和 shell 自身的计数器仍按现有发布规则更新。

AICPU 先读取 `completed_block_count`，再检查异步计数。
只有全部物理 block 已运行且 `async_registered == async_settled` 才允许结束。
登记分配发生在对应 block 计数发布之前，因此末尾任务的条件不会漏查。
没有 pending 记录的 lane 可按原 MIX/cohort 退出条件离开；拥有 pending 的 lane 留在 checkpoint。
看门狗使用单调完成进度；条件永不满足时报告运行时超时。

登记错误、非法 token、条件溢出沿现有错误通道返回，不能发布为成功。
异常终止不代表外部通信引擎已经完成；通信资源仍遵守平台的 teardown 证明。

Graph Definition 的设备载体由 percounter 自己的 bind 发布：真机使用 launch envelope，
模拟器使用本次 run 的快照。HBG 的 native Graph 路径不承担 percounter 的 AICPU 物化职责。

## 验收

`tests/st/a5/percounter/defer_wait` 覆盖普通/SPMD/MIX/cohort/Graph/谓词形状、
超过原固定槽数的等待任务，以及 notifier 排在满候选窗口之后的情况。
该 runtime 不参加默认构建；这些 scene 用例需显式构建 percounter，并使用 `--manual include`。
共享形状测试以输入参数分别验证 HBG 的 64 条条件和 percounter 的 4 条条件。
CPU 测试还验证末尾 defer 的退出门槛和 group 乱序结算。

性能比较分别报告无 defer 的主线 percounter 与当前实现，以及含 defer 的对应工作负载。
构建版本、设备、预热、轮数和时延层次必须匹配；历史测量不代替当前主线验证。
