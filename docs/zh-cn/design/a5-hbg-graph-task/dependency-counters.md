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
没有边的图不分配依赖区；不带依赖区的 helper 使用 intrusive wake 链。

## Bootstrap 与完成协议

Bootstrap 在 scan gate 前按初始 remaining 判断 READY 或 WAITING。
所有核心完成扫描前不能执行任务或发送依赖通知，因此扫描不与 decrement 竞争。
HostDone 节点仍保留既有 DONE 状态和关闭的 wake head。

`scheduler_resolve_completion_step` 只接受已经 DONE 的生产者。
同步完成的 CSR overload 由唯一的 TaskDone owner 调用，读取一次后继范围后直接遍历，
不创建 continuation、不摘取 wake list。TaskDone 发布在首个依赖通知前完成排序。
每条边以原子 fetch-add(-1) 消耗一份依赖：

- 旧值大于 1：后继继续等待。
- 旧值等于 1：唯一的最后通知者将后继交给现有 ready 发布或同类型 direct handoff。
- 旧值为 0、非法引用或越界：记录 run error，停止该 resolver。

同步 overload 沿用 ready 批次的容量与目标分组；批次满或队列分组改变时先发布已有批次，
不丢失待加入的后继。其计时行由该 owner 在返回前一次性发布，Host 在退出握手后读取。

需要预算和 owner 转移的 control/异步 FINISH 路径使用 cursor overload。
START 关闭 wake head，防止另一 cursor 重复发 credit；每次通知前保存下一后继，
APPEND、PUBLISH 和 direct handoff 遵守 bounded budget 和恢复规则。
cursor 可在任意预算边界保存或转移，同一条边不会在恢复时重复递减。
任务的 resolved 计数仍发生在整次 resolver 完成之后。

Kernel output completion、TaskDone 发布与 ready 发布的既有顺序保持不变。
同步任务的最后子任务、predicate 跳过和异步 FINISH 都使用同一套后继与剩余计数。
`defer_wait` 的返回不等于 TaskDone；只有原有 join/FINISH 条件满足后才发送 credit。
任务领取、SPMD admission、MIX、sync_start、ready 队列及错误退出协议不由依赖计数改写。

## 验收合同

CPU 验证并发 producer 的唯一 ready 转移、重复完成拒绝、每个 budget 边界的恢复、
owner 转移、HostDone/重复 fanin、损坏 offset/ref 和实际 native Graph 边集。
设备验证覆盖 Graph、SPMD/MIX/sync_start、异步 waiter 进展及跨轮复用。
Qwen 使用同一完整输入、计算 kernel 和 out/K/V oracle；协议事件审计与性能测量分别安排。
性能分别报告无 DFX 的 device wall，以及 L1 首 kernel 到末 kernel 区间与逐执行关键路径 gap，
不能把所有核心同时空闲的下界当作净调度开销。
