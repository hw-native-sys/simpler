# A5 HBG 前向依赖计数

Host 为整个 resident 图预建 CSR fanout 和每个节点的剩余依赖数。
生产者完成时通知所有后继，最后一个依赖到达的消费者进入现有 ready 发布流程。
普通任务、Graph entry/exit、Graph child 和动态异步完成共享同一套依赖计数。

## Host 布局与初始化

`SchedulerReadyPoolHeader` 为 v8，64 字节对齐、总长 128 字节；dependencies_offset 与队列寻址热字段共用首行。
依赖区包含 64 字节 version 2 header、N 个 uint64 remaining、N+1 个 uint64 CSR offsets 和 E 个 int32 refs，
每段按 64 字节对齐，全部使用 scheduler allocation 相对 offset。
N 是展开后的 resident 节点数，E 的预留容量来自原始顶层 fanin 数和各 Graph 实例的
Definition 内部边、root 数、sink 数之和。空间约为 `16*N + 4*E` 字节加对齐。

初始化使用 `SchedulerGraphAccess` 的统一 fanin view，保留原有 Definition 和引用边界校验。
每个消费者的重复 producer 去重；Host 已完成的生产者和消费者不参与设备通知。
实际边数不得超出预留容量。初始化失败时整个 bind 失败，不发布部分状态。
Graph entry 的外部依赖、root 对 entry 的依赖、exit 对 sink 的依赖和跨实例 completion ref
都由统一 view 提供，不维护另一套 Graph 解析器。

`remaining` 独占原子访问区。Host 在运行前初始化并上传，运行中没有 cached store 或整行回写。
CSR 和 header 在运行期间只读；设备在 READY acquire 后校验其版本、对齐和 allocation 边界。
没有边的图不分配依赖区；没有依赖区的 helper 和兼容调用仍使用 intrusive wake 链。

## Bootstrap 与完成协议

Bootstrap 在 scan gate 前按初始 remaining 判断 READY 或 WAITING。
所有核心完成扫描前不能执行任务或发送依赖通知，因此扫描不与 decrement 竞争。
HostDone 节点仍保留既有 DONE 状态和关闭的 wake head。

统一 `scheduler_resolve_completion_step` 只接受已经 DONE 的生产者。
START 原子关闭其 wake head，作为整次完成遍历的唯一所有权标记；第二个 completion cursor
不能重复发 credit。SELECT 从 CSR 取得后继，以原子 fetch-add(-1) 消耗一份依赖：

- 旧值大于 1：后继继续等待。
- 旧值等于 1：唯一的最后通知者将后继交给 APPEND。
- 旧值为 0、非法引用或越界：记录 run error，停止该 resolver。

每次通知前先保存下一后继，通知后推进 cursor；APPEND、PUBLISH 和 direct handoff
继续遵守原有 bounded budget 和恢复规则。cursor 可在任意预算边界保存或转移，
同一条边不会在恢复时重复递减。任务的 resolved 计数仍发生在整次 resolver 完成之后。

Kernel output completion、TaskDone 发布与 ready 发布的既有顺序保持不变。
同步任务的最后 block、predicate 跳过和异步 FINISH 都经同一 resolver 发布依赖。
`defer_wait` 的返回不等于 TaskDone；只有原有 join/FINISH 条件满足后才发送 credit。
任务领取、SPMD admission、MIX、sync_start、ready 队列及错误退出协议不由依赖计数改写。

## 验收合同

CPU 验证并发 producer 的唯一 ready 转移、重复完成拒绝、每个 budget 边界的恢复、
owner 转移、HostDone/重复 fanin、损坏 offset/ref 和实际 native Graph 边集。
设备验证覆盖 Graph、SPMD/MIX/sync_start、异步 waiter 进展及跨轮复用。
Qwen 使用同一完整输入、计算 kernel 和 out/K/V oracle；L4 独立审计实际依赖与执行序列。
性能分别报告无 DFX 的 device wall，以及 L1 首 kernel 到末 kernel 区间与逐执行关键路径 gap，
不能把所有核心同时空闲的下界当作净调度开销。
