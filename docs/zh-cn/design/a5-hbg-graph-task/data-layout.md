# A5 原生 Graph 的数据布局与 Host 绑定

本文描述默认 A5 HBG resident 的当前 wire 合同。Host、AICPU、AICore 必须配套编译。
类型采用固定宽度 POD，内部引用用 offset/index；绝对设备地址以整数保存。
各 wire 类型以 standard-layout、trivially-copyable、sizeof 和关键 offsetof 约束。

## 身份和顶层镜像

设顶层任务数 T、Graph 实例数 G、实例 g 的 child 数 n[g]：

```text
N = T - G + sum(n[g] + 2)
completion_refs[top_id] = 普通节点 ref 或 Graph EXIT ref
child_ref(g, local) = instances[g].child_begin + local
```

Host 按顶层 task ID 分配 expanded ref；一个 Graph 连续分配 ENTER、children、EXIT。
公共 TaskId 保留 GLOBAL/SUB_TASK 的 shell/local 身份，不能拿 expanded ref 直接索引顶层表。

| ref | A → G(c0 → c1) → B 的节点 | 公共身份 |
| --- | ------------------------- | -------- |
| 0 | A | GLOBAL(0) |
| 1 | ENTER G | GLOBAL(1)，带 ENTER kind |
| 2 | c0 | SUB_TASK(parent=1, local=0) |
| 3 | c1 | SUB_TASK(parent=1, local=1) |
| 4 | EXIT G | GLOBAL(1)，带 EXIT kind |
| 5 | B | GLOBAL(2) |

此例 completion_refs=[0,4,5]。控制节点没有 kernel execution identity。
`SchedulerNodeBinding` 为 12 B：source_index/instance_index 各 uint32_t，加独立 kind。
非 Graph 节点的 instance_index 为 UINT32_MAX。

默认任务镜像从 64 B `TaskImageStatus` 开始，保存本轮首错 atomic；随后是至少一个 slot pitch 的
80 B `TaskImageStorage` 数组，以及实用的 fanin、tensor、scalar 池，区段按 64 B 对齐。
每个记录只含 task ID/kernel IDs、参数计数、field-relative pool deltas 和 predicate。
delta 是目标减去字段自身的地址；整份镜像一次搬运不需要设备修复内部指针。
scalar 是原始值位，不当地址重定位；tensor buffer 和 predicate operand 按 heap 映射重定位。

Host 编排可保留公共 rich scratch，设备不接收其 slot/progress/task-state 表。
Percounter 的活跃 Graph materializer 由其 runtime include 路径选择自己的当前布局。

## Resident backing 与缓存所有权

`SchedulerGraphView` 为 56 B，version 6；字段为六个 uint64_t 地址/数量/extent、
uint32_t original_task_count 和两个 uint16_t view_version/view_bytes。
无 Graph 时 native_header_offset 为零；有 Graph 时所有 expanded node 都经 binding 访问。
原始数量还受 TaskId parent 编码容量限制，不能仅检查 uint32_t 是否容纳。

| 类型 | 大小 | 所有权与用途 |
| ---- | ---- | ------------ |
| SchedulerTaskMetadata | 16 B | Host 只读形状、kernel、block 和 trace prefix |
| SchedulerTaskControl | 768 B | 六个独立 128 B 原子区域：claim_tickets、next_block、completion_state、admission_state、claim_seal、parked_count |
| SchedulerWorkerContext | 320 B | 前四个 64 B 行由 READY 发布；最后一行归本核诊断 |
| SchedulerRunControl | 7296 B | 384 B 配置/生命周期/首错前缀及 108 个独占行的 worker 错误信号 |
| SchedulerGraphSummary | 192 B | 本轮只读拓扑和各子区 offset |
| SchedulerTaskTrace | 256 B | publication、kernel timing、dispatch/completion、execution identity 四行 |
| SchedulerExecutorTaskTrace | 24 B | 核内 start/end/ready-observe 三个时间戳 |
| SchedulerAdmissionRecord | 64 B | sync_start 任务的冻结 admission 事实 |
| SchedulerAdmissionControl | 256 B | commit_lock 和 bypass commit_tail 分占 128 B 区域 |
| SchedulerGroupMailbox | 384 B | group descriptor、发布字及三个独立 lane ack |
| SchedulerMixCloseSeal | 64 B | 单个保留的原子 close seal |

TaskControl 的 atomic offset 依次为 0、128、256、384、512、640。
completion_state 编码 total/remaining_compute/debt；具体有效性和 DONE 判据见
[canonical completion gate](dependency-counters.md)。它不包含 wake-list、remaining fanin 或 resolver timing。

WorkerContext 首行是 core_type/physical_core_id/type_rank、run_control/task_controls/state base/
prepared layout、worker_index/worker_contexts_offset；第二行保存完整 graph view 和 summary offset。
第三行从 128 开始保存 metadata、callable 表、runtime_worker_count，以及从 160 开始的 cluster 信息。
trace/activity/PMU offsets 分别位于 208/216/224，args_dump_address 位于 232。
最后一行从 256 开始，保存 bootstrap 统计。
所有核都自行调度，结构中没有另一个 scheduler 角色、索引或数量。

TaskTrace 首行的 valid 由 atomic 发布；kernel timing 行从 64 开始，dispatch/completion 行从 128 开始，
execution identity 从 192 开始。退休时先写回有效计时行并 barrier，再发布 valid。
后续 cached writeback 不能覆盖 valid 所在行；核内 staged trace 保持自然对齐。

backing 基址满足 128 B 对齐。标量缓存维护仍按 64 B 行，不能从原子区域尺寸推导缓存指令粒度。
业务 kernel 不接收调度私有 slot 或 admission 对象的地址。

## Native Graph 与实例参数

`SchedulerNativeGraphHeader` 为 128 B、version 5。magic/version 为两个 uint32_t，
随后依次是 image_bytes、node_count、original_task_count、definition_count、instance_count、
bindings_offset、completion_refs_offset、templates_offset、instances_offset、readonly_end_offset、run_epoch，
最后是 uint32_t header_bytes。没有 control inbox 区。

只读区域按 header、binding、completion refs、template、instance、child plan、fanin CSR、sink 排列。
读取 header 前先以可信 scheduler_state_bytes 检查地址和长度，再核对 header.image_bytes；
不能只相信尚未验证 header 自报的大小。offset/count、对齐、非重叠、epoch 与展开数量均校验。
原始录制 Graph 及其 fanout、root、参数重定位记录、scalar inheritance 不复制进 native GM 镜像。

每个唯一 Graph 对应一份 64 B `SchedulerGraphTemplate`：
child plans、fanin rows/indices、sinks 的四个 offset，以及 task/edge/sink 数和 args_bytes。
fanin CSR 只拷贝实际运行使用的前驱边，sink 由 Host 验证后的拓扑推导。
它表达 child fanin，不是运行时后继通知表。

每 child 有一份 48 B plan：16 B metadata、三个参数 offset、tensor/scalar 数。
每 instance 有一份 32 B 记录：Graph index、原始 shell ID、ENTER/child/EXIT ref 和 args_base。
boundary 参数、heap_base 和完整 Graph 由 Host 绑定器持有；设备不读取它们。
同一 Graph 的拓扑和 child plan 可共享；每个 instance 的参数和控制责任独立。

共享录制结构的类型名是 `Graph`，头文件为 `host_build_graph/graph.h`；
外层 framing 为 `GraphImageHeader`，按值解码结果为 `GraphImageValue`，没有旧类型 alias 或转发头。
共享录制格式的字段与布局保持；A5 native 镜像只接受 v5，旧 native 布局直接拒绝。

| 节点 | fanin 视图 |
| ---- | ---------- |
| TOP_KERNEL / ENTER | 顶层 fanin 经 completion_refs 映射 |
| CHILD，有内部 producer | Graph-local indices 加 child_begin |
| CHILD root | ENTER |
| EXIT | 实例的全部 sink |
| HOST_DONE | 已完成 gate，消费者仍正常检查 |

Graph row 可以有 1023 个 producer，EXIT 可以有 1024 个 sink，计数和游标使用至少 32 bit。
普通顶层的 fanin 上限仍是 128；不能把这个限制套在 Graph 内部。

Host 验证 Graph 的 tensor/scalar 区间连续、覆盖完整参数池；重叠、留洞或未使用尾部均拒绝。
每个 child 的参数 tail 为：

```text
align64(tensor_count * 128)
+ align64(scalar_count * 8)
+ (has_predicate ? 64 : 0)
```

空参数 child 的 tail 可以为零。Graph 总预算为 required_heap + args_bytes，
Host 计算 args_base=heap_base+required_heap；tensor 描述符仍为现行 128 B，不能直接传 96 B Graph record 给 kernel。
所有乘法、加法、align-up、实例区间及 INT32_MAX 限制在发布前检查。
Graph 预算计算一次并缓存，Graph 提交和 Host bind 使用同一算法验证。

PARAM 引用本次 boundary，SUB_TASK 引用本实例业务 heap；固定 scalar 按值复制，inherited scalar 来自 boundary。
predicate 地址采用对应重定位规则，值在实际执行时按依赖顺序读取。
Host 在 orchestration 返回前取得自有 Graph 字节与索引，不能保留 recording 临时容器的指针。

## Prepared execution、预算与生命周期

每 node 一个 16 B `SchedulerExecutionPlan`。每真实 block/lane 执行有以下对象：

| 对象 | 大小 | 内容 |
| ---- | ---- | ---- |
| SchedulerPreparedCall | 512 B | immutable function、epoch、node/block/lane、完整 args |
| SchedulerPreparedControl | 128 B | phase、owner、generation 的单原子字 |
| SchedulerExecutionBinding | 128 B | 各占一行的 LocalContext/GlobalContext |

args[48]/args[49] 指向本次 execution 的 Binding，两核内 slot 复用不覆盖这些参数。
四个 token 与两个 slot 分别保有执行责任和临近派发状态，不能把两种容量当成同一个队列深度。

统计 backing 必须分别计入 metadata/control、Call/control/Binding、native readonly、实际目录 refs、
admission/group、async、args dump、trace/PMU/activity/Scalar 区及对齐空隙。
当前 layout 在 E 非零时按 E、否则按 N 规划并保留任务 trace，不能声称采集关闭就释放该区；PMU、protocol、activity 按请求另计。
每个 trace 的 64 B 减少是布局事实，不是延迟收益证明。

prepare retry 重新用本轮 backing 生成全部地址；publication source 和目标由同一 pipeline slot 持有。
异步上传失败按既有 drain/quarantine 路径保留所有权，不启动不完整的 run。
后继 run 不能借用仍在执行或清退失败 run 的 Graph、参数、控制或 capture backing。
首错、EXIT/ACK 与 post-close release 共同决定是否可回收；DONE 或缺失 ACK 都不足以证明安全退休。
