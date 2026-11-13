# A5 HBG 依赖解析方案

默认Host maker生成去重的fanout CSR与remaining，使用同步CSR传播完成。
未附依赖表（`dependencies_offset=0`）的有效图使用同步wake-list。格式或范围错误直接拒绝，不在两条路径之间回退来掩盖损坏状态。

## 动态挂链合同

bootstrap 在 scan gate 内将每个消费者挂到第一个未完成前驱，HostDone 前驱跳过。
保存的续行位置还跳过该前驱之后的 HostDone 和同一前驱的重复引用，停在下一个不同的设备前驱或行尾。
每个 waiter 的 `wake_cursor` 是一个 64-bit 原子字：低 32 位为有符号 next task ref，-1 终止链；
高 32 位为下一条尚未检查的 fanin 位置，已经跳过当前挂靠的前驱。
`next_waiter` 单独供 native control inbox 和 Ready 临时批次使用，不承载依赖续行位置。

前驱发布 TaskDone 后原子关闭 wake head。唯一 resolver 取得链，原子读取每个 waiter 的 link/index；
位置已到 fanin 末尾时直接发布 Ready，否则从该位置继续检查前驱。
路由依据单调 CLOSED 的 wake head 判断完成；OPEN 时允许 CAS 挂入，即使前驱正在 TaskDone 与摘链之间。
CAS 前原子写入完整 wake_cursor；CAS 失败仍持有 waiter，可以用返回的实际 head 重试。
CAS 成功后所有权已经交给新前驱，原 resolver 只能使用提前保存的旧 next，不能再访问 waiter。

同步 TaskDone owner 直接遍历摘下的链，不构造可恢复的 phase。它在 Ready 前缀发布期间保留当前 waiter
的 fanin view 与旧 next；重挂链成功即放弃所有权。其 resolve 计时包含摘链、依赖检查与 Ready 发布。

普通kernel、native control和async FINISH都使用同一次调用内的同步传播。
control服务按节点数限制每轮工作；FINISH持有独占lease直到全部后继处理完，不发布或转移resolver续行。
业务同类任务合并发布，控制节点进入 inbox；producer 汇总只在全部后继责任移交后增加。
重复完成由 TaskDone/子任务贡献/服务 lease 的唯一所有权和 CLOSED 检查拒绝。
CPU 合约检查不能证明 A5 cache 与原子可见性，必须配合上板验证。

## 默认 CSR 路径

Host为resident图预建fanout和remaining，普通与native节点共享统一fanin view。
生产者完成后对每条后继边减计数，取到旧值 1 的通知者发布 Ready。

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
CSR 候选没有边时可省略依赖区；挂链候选无论是否有边都不生成此表。

## Bootstrap 与完成协议

Bootstrap 在 scan gate 前按初始 remaining 判断 READY 或 WAITING。
所有核心完成扫描前不能执行任务或发送依赖通知，因此扫描不与 decrement 竞争。
HostDone 节点保留 DONE 状态和关闭的 wake head。

`scheduler_resolve_csr_completion` 只接受已经 DONE 的生产者。
同步 CSR 入口 由唯一的 TaskDone owner 调用，读取一次后继范围后直接遍历，
不创建 continuation、不摘取 wake list。TaskDone 发布在首个依赖通知前完成排序。
每条边以原子 fetch-add(-1) 消耗一份依赖：

- 旧值大于 1：后继继续等待。
- 旧值等于 1：唯一的最后通知者将后继交给现有 ready 发布或同类型 direct handoff。
- 旧值为 0、非法引用或越界：记录 run error，停止该 resolver。

同步入口 沿用 ready 批次的容量与目标分组；批次满或队列分组改变时先发布已有批次，
不丢失待加入的后继。其计时行由该 owner 在返回前一次性发布，Host 在退出握手后读取。

control与异步FINISH的唯一完成owner也调用同步入口。每个owner负责整次传播，任务的resolved计数在全部后继责任处理完成后增加。

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
