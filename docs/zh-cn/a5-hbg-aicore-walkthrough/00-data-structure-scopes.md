# 数据结构：作用范围与生命周期

[总索引](README.md) · [下一段：初始化](01-initialization.md)

同一个结构的多核 reader 不代表每核有一份副本。以下均以一次 run 的所有权解释。
精确 wire 字段和缓存行位置见[数据布局](../design/a5-hbg-graph-task/data-layout.md)。

## Host 发布与整轮 GM 状态

| 数据 | 数量和存放位置 | 写者与生命周期 |
| ---- | -------------- | -------------- |
| TaskImageStatus | 每轮 64 B 首错头 | Host 初始化，AICPU 首错锁存和 Host 结算 |
| TaskImageStorage | 每顶层任务 80 B，后接实用参数/fanin 池 | Host pack 后发布，设备只读参数与依赖 |
| SchedulerNativeGraphHeader / templates | native Graph 每轮一份 128 B 头、每唯一 Graph 一份 64 B 计划 | Host 持有自有字节，发布后只读 |
| SchedulerGraphInstance | 每个 shell invocation 一份 | 只含实例 ref/Graph index 和 args_base，32 B |
| SchedulerRunControl | 每轮一份 | AICPU 发布实际参与者和预算，核按字段协议更新 bootstrap/完成/错误 |
| SchedulerGraphSummary | 每轮一份 | Host 图事实加 AICPU 验证后的拓扑，READY 后只读 |
| SchedulerWorkerContext | 每物理 worker 一个 320 B 记录 | READY 发布前四行；本核写最后一行启动诊断 |
| Handshake | 每 worker 一条握手记录 | 本轮报告、READY、EXIT/ACK，各词有独立交权方向 |

Host 的 rich scratch 不是设备的运行状态。80 B 记录中没有 ChipTaskSlotState，
Graph child 也不另占一个顶层任务镜像条目。

## 每 node 与每真实 execution

| 数据 | 作用范围 | 写者与读者 |
| ---- | -------- | ---------- |
| SchedulerTaskMetadata，16 B | 每 expanded node | Host 只读形状、kernel IDs、trace prefix |
| SchedulerTaskControl，768 B | 每 expanded node | 核以 atomic 管 ticket、gate、park、admission；无后继通知链 |
| SchedulerExecutionPlan，16 B | 每 node | Host 声明该 node 的真实执行区间 |
| SchedulerPreparedCall，512 B | 每 block×active-lane 执行 | Host 唯一写者，设备读取完整函数和 args |
| SchedulerPreparedControl，128 B | 每真实执行 | 单原子 phase/owner/generation；控制 Call 责任 |
| SchedulerExecutionBinding，128 B | 每真实执行 | 两行分别放 LocalContext/GlobalContext，地址稳定至退休 |
| SchedulerTaskTrace，256 B | 每记录的真实执行 | 本核依次发布准备、kernel、完成与执行身份 |
| SchedulerTaskPmuCell，64 B | 每启用 PMU 的执行 | 执行核写入，AICPU 退休后读取 |

logical node 的 gate 可以 DONE，而执行者仍在完成 trace 或 group release。
run 汇总和退出证明分别处理这些剩余责任。

## 核内状态与共享协作

每核一个自然对齐的 SchedulerLocalState：四个普通 execution token、两个 40 B slot、
本地扫描游标、ready mask/head、pending completion 和 cooperative/async 状态引用。
只有本核准备和执行自己的 slot；一个 slot 只保存临近执行状态，不复制整份 Call args。
24 B SchedulerExecutorTaskTrace 暂存 kernel start/end 与 ready-observe，退休时合并到 GM trace。

目录 refs 在整个 run 不变。MIX 的 cluster signal、group reservation/mailbox、lane ack，
sync_start 的 admission log/record、member closure，以及 async condition/group/join/lease
是共享状态，各自有唯一 owner 和 publication 合同；它们不是另一个 ready queue。

Call、Binding、参数、capture backing 和控制表都必须保持至本轮清退被证明完成。
FAILED 或 PARKED 不授权 Host 提前回收；后续 epoch 也不能覆盖仍有 owner 的旧 run。
