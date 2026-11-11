# 2. 依赖、ready 队列与 kernel 执行

[总索引](README.md) · [上一段：初始化](01-initialization.md) · [下一段：协作与完成](03-cooperative-completion.md)

结构的作用范围、数量、存放位置、读写者和生命周期可对照 [数据结构总览](00-data-structure-scopes.md)；下面在关键结构的字段表前也说明其范围与用途。

## 2.1 从输入图读什么

[SchedulerGraphView](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_graph.h#L108)（第 108 行） 是本核对图的只读视图：

**范围与用途：** 每个 worker 构造自己的局部视图；视图很小，保存地址、长度和版本，指向的是同一轮 GM 图存储。每核一份视图不等于每核一份图。

| 字段 | 作用 |
| ---- | ---- |
| `storage_address` | 原始 ChipTaskStorage 数组 GM 基址 |
| `task_count` | 当前 resident 节点总数 |
| `task_window_last_index` | 原始存储容量边界；HBG 不对 task_ref 取模 |
| `scheduler_state_address` | 调度状态 GM 基址 |
| `native_header_offset` | native graph 头偏移；0 为普通图 |
| `scheduler_state_bytes` | 调度状态范围 |
| `original_task_count` | 原始任务存储条目数 |
| `view_version / view_bytes` | 视图版本和大小，当前版本为 3，sizeof 为 56B |

[ChipTaskStorage](../../../src/common/host_build_graph/runtime_types.h#L799)（第 799 行） 的三个成员：`task` 是描述符；`slot` 是共享任务存储里的调度信息，resident 可变依赖状态另在 SchedulerTaskControl；`payload` 是参数与依赖区域的入口。当前 sizeof 为 320，不要自行按旧版本跨度寻址。

**范围与用途：** `ChipTaskStorage` 及其 `TaskDescriptor`、`TaskPayload` 属于原始任务存储条目，与 worker/执行 slot 无关。某任务的多个 block 可以反复读取同一份参数源。native child 的描述来自 template/instance，不必各占一个原始 ChipTaskStorage。

[TaskDescriptor](../../../src/common/host_build_graph/runtime_types.h#L269)（第 269 行）：`task_id` 为公开任务身份；`kernel_id[3]` 对应三个子任务入口 ID；`packed_buffer_base/end` 为打包输出区的起止地址；`graph_definition_offset` 为 GRAPH 定义偏移；`reserved[20]` 保留。本次不展开它们的 host 构造或 legacy 管理。

[TaskPayload](../../../src/common/host_build_graph/runtime_types.h#L357)（第 357 行） 中 resident 参数读取关心：

| 字段 | 作用 |
| ---- | ---- |
| `tensor_count / scalar_count` | tensor 描述符和 scalar 数量 |
| `fanin_count` | 前驱依赖数量 |
| `tensors / scalars / fanin` | 三块数据的 SelfRelativePtr；存储的是相对该字段自身的 int32 delta |
| `predicate` | 可选派发条件；resident 在准备执行时求值 |
| `dump_metadata` | 参数 dump 元信息，不控制此次调度 |

其 early-dispatch 区属于跳过的另一套调度协议，不沿该区展开。resident 通过 graph accessor 得到 fanin 和参数，不使用该区发布 kernel。

[SchedulerTaskMetadata](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_types.h#L813)（第 813 行） 是 16B 的紧凑调度入口：

**范围与用途：** 每个 resident 节点一份 GM 元数据，整轮按 task_ref 读取。它回答“这个任务是什么形状、用什么 kernel”，多个 worker 执行该任务的不同 block 时读取同一条元数据。

| 字段 | 作用 |
| ---- | ---- |
| `kernel_ids[3]` | AIC、AIV0、AIV1 的 kernel 地址表索引 |
| `active_mask` | 三个子任务中哪些有效 |
| `flags` | EXECUTABLE、HAS_FANIN、MIX、SPMD、SYNC_START、INLINE、HAS_PREDICATE 位 |
| `trace_index_base` | 本任务各 block/subtask 记录的起始位置 |
| `timing_slot` | 定点计时槽，-1 为未指定，有效范围 0..15 |
| `logical_block_num` | 本任务逻辑 block 数 |

`MIX` 表示多个 active subtask；`SPMD` 表示多个 logical block，二者可同时存在。
`INLINE` 是由调度器直接完成的节点标记；不要与 RunControl 中输入时已完成的 `inline_completed_count` 混同。

## 2.2 依赖不是每轮扫描整图

当前优先使用 Host 预建的静态依赖计数。[bootstrap_route_task](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_ready.h#L982) 读取 remaining，
为 0 才发布初始 ready；非零等待实际前驱完成通知。Host 已去掉输入时完成的节点和重复边。
没有依赖 header 时，才使用下文的 wake-list 兼容路径。

[SchedulerTaskControl](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_types.h#L930)（第 930 行） 为 384B、64B 对齐，每字段的含义：

**范围与用途：** 每个 resident 节点一份 GM 可变控制记录，所有 scheduler 共同推进这一份。它跟随逻辑任务，而不是跟随某个 worker；同一任务被多个核执行时，block join 和依赖完成都必须汇总到这里。

| 字段 | 作用 |
| ---- | ---- |
| `state` | BLOCKED=0 或 DONE=2；READY 由队列/slot 表达，不写成这里的第三个状态 |
| `wake_list_head` | 等待本任务的任务链表头；-1 空且可注册，-2 已关闭 |
| `next_waiter` | 自己在 wake list、ready inbox 或待发布批次中的链指针；同时只属于一处 |
| `claim_tickets` | 协作 block 的原子领票计数，可能包含少量越过有效 block 范围的票 |
| `admission_claim_lock` | 保证任务 admission 初次认领只发生一次 |
| `next_fanin_index` | 本任务扫描依赖的保存位置 |
| `waiting_producer` | 当前挂在哪个前驱上，-1 表示无 |
| `completion_resolve_start_cycles / completion_resolve_end_cycles` | 本任务作为 producer 的依赖传播开始/结束时间 |
| `scheduler_worker_id` | 执行该次完成解析的 worker |
| `completion_resolve_loop_iter` | 对应 scheduler 循环序号 |
| `next_block` | 已确认成功领取的 block 数/领取游标；ticket 分支中不是 next ticket 值 |
| `completed_blocks` | 已完成的逻辑 block 数；MIX 一个 group 汇总后贡献一个 block |
| `admission_state` | UNCLAIMED → CLAIMED → PREPARED → ADMITTED |
| `claim_seal` | 全部有效 block 已确认领取的不可变凭据，值为逻辑 block 总数 |

TaskControl 共六行：首行是原子 state/链指针/claim tickets；第二行是 resolver 的 cached metadata；
`next_block`、`completed_blocks`、`admission_state` 和 `claim_seal` 分别独占后四行。
完成与领取会并发访问，缓存行布局本身就是各字段访问协议的一部分。

[SchedulerDependencyHeader](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_dependency.h#L18) 为一条 64B 只读记录，version=2：

| 字段 | 作用 |
| ---- | ---- |
| `version / task_count` | 格式版本与 resident 节点数 |
| `edge_capacity / edge_count` | 后继边的预留容量和去重后的有效边数 |
| `remaining_offset` | N 个 uint64_t 原子剩余依赖数 |
| `fanout_offsets_offset` | N+1 个 uint64_t CSR 行边界 |
| `fanout_refs_offset` | int32_t 后继 task_ref 数组 |
| `storage_end_offset` | 依赖存储末端，校验所有数组边界 |

例如 `T2` 依赖尚未完成的 `T0,T1`：

```text
Host：remaining[T2]=2，T0 和 T1 的 fanout CSR 各含 T2 一次
bootstrap：T2 不进入 ready
T0 完成解析：remaining[T2] 2→1，仍等待
T1 完成解析：remaining[T2] 1→0，由这次通知的 owner 发布 ready
```

两个前驱可以以任意顺序完成，只有取到旧值 1 的通知者使后继 ready。
CSR 与计数器沿用同一个图访问器的依赖身份和越界检查，不改变 predicate、MIX 或 async 的 DONE 条件。

**无依赖 header 的兼容路径：** bootstrap 为 waiter 找到第一个未完成前驱，用 exchange 挂入 wake list；
producer 完成时摘链，[scheduler_route_task_step](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_ready.h#L847) 继续扫描剩余 fanin，必要时用 CAS 迁移到下一个 producer。
稳态 CAS 路由必须先发布 waiter 的 next 和扫描位置，再挂入新链；遇到 CLOSED 则继续路由。
bootstrap 的 exchange-before-link 只在扫描屏障尚未开放、没有 resolver 并发摘链时使用。

[SchedulerFaninView](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_graph.h#L425)（第 425 行）：
`global_refs` 是普通图 int32 前驱索引数组；`local_refs` 是 native definition 的局部前驱索引；
`completion_refs` 是原始任务到调度完成节点的映射；`base_ref` 是局部引用转换的基数；
`count` 是边数；`source_limit` 用来验证源索引范围。

## 2.3 ready 的两种存储都是 resident

在 [scheduler_claim_ready_for_slot](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_ready.h#L1636)（第 1636 行） 看分支。
类型索引统一为 0=AIC、1=AIV、2=MIX；MIX 由 AIC coordinator 领取。

`ready_pool_offset != 0` 时使用分组 ticket pool：
发布者原子增加 reserved_tail 取得写槽；写入 task_ref 后，该 entry 本轮不再改写。
消费者读 head 对应 entry，成功 CAS head 才拥有任务。
reserved_tail 增加只代表预留，尚不代表 entry 已经发布；NOT_READY 不等于全图结束。
见 [ticket reserve/publish/pop](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_ticket_queue.h#L57)（第 57 行）。

[SchedulerReadyPoolHeader](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_ready_pool.h#L24)（第 24 行）：

**范围与用途：** Header 是整轮一份的只读布局；Descriptor 和 TicketQueue 则按每条分组队列各一份。它们是 GM 共享对象；TicketQueueView、TicketRange 是当前操作的本地访问视图和预留结果，不能把后两者当作新的队列实例。

| 字段 | 作用 |
| ---- | ---- |
| `version` | pool 协议版本，当前 8 |
| `group_width` | 每组目标核数量参数，当前 8 |
| `task_count` | 合法 task_ref 的上界 |
| `group_counts[3]` | AIC/AIV/MIX 各有多少组 |
| `queue_begins[3]` | 三类队列在扁平数组中的起始索引 |
| `descriptors_offset` | 队列描述符数组偏移 |
| `controls_offset` | 队列 head/tail 控制数组偏移 |
| `dependencies_offset` | 静态依赖 header 偏移，0 表示采用 wake-list 兼容路径 |
| `storage_end_offset` | pool 存储的末端边界 |
| `queue_count` | 总队列数 |
| `ref_chunk` | 路由 task_ref 的分块粒度，当前 4 |
| `entry_count` | 全部队列 entry 容量之和 |

相关小结构全部字段：

| 结构 | 字段及作用 |
| ---- | ---------- |
| `SchedulerReadyPoolDescriptor` | `entries_offset`：entry 区偏移；`capacity`：此队列容量 |
| `SchedulerTicketQueue` | `head`：下一消费位置；`reserved_tail`：下一个未预留位置；head 与 reserved_tail 分别以 64B 对齐，总长 128B |
| `SchedulerTicketQueueView` | `control`：控制对象指针；`entries`：entry 指针；`capacity`：队列容量；`task_count`：ref 边界 |
| `SchedulerTicketRange` | `begin/count`：本发布者取得的连续写入区间 |

`ready_pool_offset == 0` 时使用 per-type inbox 和目录：
先看本核同类型 rank 对应 inbox，无任务时按目录尝试其他 inbox。
普通图按本 shard 搜索；native graph 还可搜索远端 shard。目录是搜索提示，CAS 成功才取得所有权；本核 own inbox 不能仅因目录位为 0 就跳过。

| 结构 | 字段及作用 |
| ---- | ---------- |
| `SchedulerReadyInbox` | `endpoints`：低 32 位为 head、高 32 位为 tail，作为一个 64 位 CAS 对象更新；总长/对齐均为 64B |
| `SchedulerReadyDirectoryShard` | `bits`：哪些 owner inbox 可能非空；总长/对齐均为 64B |
| `SchedulerReadyDirectory` | `core_types[3][shard_count]`：按类型和 shard 索引的目录 |
| `SchedulerReadyBatch` | `head/tail`：批次链表首尾；`count`：节点数 |

inbox 的 head=tail>=0 表示恰好一个任务；head=tail=-1 才是空，和常见 ring buffer 约定不同。

[SchedulerReadyClaim](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_ready.h#L113)（第 113 行）：
`task_ref` 是领到的任务；`inbox_index` 是来源队列编号；
`source` 区分 LOCAL/STOLEN/DIRECT_RESOLVE/COOPERATIVE；
`publication_mode` 区分首次 DISPATCH 和完成后 REFILL；
`state_probe_start_cycles/end_cycles` 记录领取探测区间。

## 2.4 本核状态与执行 slot

[SchedulerLocalConfig](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_ready.h#L220)（第 220 行） 缓存 READY 后不变的配置：

**范围与用途：** 每个 worker 的 LocalState 中一份，只由本核使用。它把 GM WorkerContext 中常用的不变配置缓存下来；它里面的地址/偏移仍然可以指向跨核共享表。

| 字段 | 作用 |
| ---- | ---- |
| `shared_context` | 本 worker 的 GM context 指针 |
| `graph` | 本核 GraphView 的指针 |
| `task_controls_offset / task_metadata_offset` | 任务控制/元数据表 |
| `ready_inboxes_offset / ready_directory_offset / ready_pool_offset` | ready 两种后端的定位信息 |
| `control_inboxes_offset` | native 控制节点队列，普通图为 0 |
| `mix_payloads_offset / mix_traces_offset` | 保留 MIX 存储定位；当前 GM MIX 不用远端 SSBUF slot |
| `graph_summary_offset` | 图摘要 |
| `callable_addresses_address / callable_addresses_count` | kernel 入口表及边界 |
| `pmu_cells_offset` | PMU 记录区 |
| `worker_contexts_offset` | 全部 worker context |
| `dispatch_payloads_offset` | 所有 own payload 的起点，从本核 offset 反推得到 |
| `worker_ids[3]` | 本 cluster 的三个 worker |
| `runtime_worker_count` | 总 worker 数 |
| `scheduler_index / scheduler_count` | 本核 scheduler 编号和总数 |
| `cluster_type_ranks[3]` | 本 cluster 三个核各自在本类型内的 rank |
| `cluster_count` | 总 cluster 数 |
| `self_lane` | 本核在 cluster 的位置 |
| `self_core_type` | ready 类型编号，0=AIC，1=AIV |

[SchedulerLocalState](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_ready.h#L296)（第 296 行） 是每核私有的可变账本：

**范围与用途：** 每个 worker 在进入 resident scheduler 时创建一份私有对象，生命周期覆盖本轮循环。它记录“我这个核正在做什么”，其他核不直接修改它；需要跨核协调的内容放在它引用的 GM 控制对象中。

| 字段 | 作用 |
| ---- | ---- |
| `config` | 上述不可变配置 |
| `profiling` | 可选本地采集状态指针 |
| `admission` | 图存在协作任务时的本地有序准入状态 |
| `async` | 图存在异步完成布局时的本地服务状态 |
| `pending_completed` | 已完成依赖传播、尚未累加到全局的节点数 |
| `control_resolve` | native 控制节点的可续行完成解析游标 |
| `last_completion_flush_cycles` | 最近一次相应批量完成刷新时间 |
| `control_peer_cursor` | native 控制 inbox 的轮转搜索位置 |
| `dispatch_sequence` | 本核 own slot 的下一发布 generation 来源 |
| `has_mix` | 图是否含 MIX |
| `slots[2]` | 本核自己的两个执行 slot |
| `local_completed_generations[2]` | executor 返回后记录的完成 generation |
| `consumed_completion_generation` | 已消费的两个 generation，各占 32 位 |
| `local_ready_mask` | 哪些 own slot 已发布 ready |
| `owner_ready_queue_mask` | 本核已激活并负责维护目录的队列类型位 |
| `pool_poll_counts[3]` | ticket pool 各类型轮询节奏状态 |
| `mix_trackers[]` | 非当前 GM MIX 路径的子任务完成聚合槽 |
| `mix_slots[3][2]` | 保留的远端 MIX slot 账本；当前 GM MIX 用 own slots |
| `mix_dispatch_sequences[3]` | 上述 MIX slot 发布序列 |
| `consumed_mix_completion_generations[3]` | 上述 MIX slot 已消费完成序列 |

`SchedulerMixTracker` 的 `task_ref` 定位任务，`active_mask` 表示需要哪些子任务，`completed_mask` 表示哪些已完成。当前 GM MIX 用 mailbox 的 release join，不用此 tracker 做主线汇总。

[SchedulerLocalSlotState](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_ready.h#L196)（第 196 行）：

**范围与用途：** 当前主线每个 worker 有两个 own slot 的本地状态。slot 对象整轮保留，task_ref/block_idx/generation 随执行绑定变化；它与 GM 中同 worker、同 slot 的 DispatchPayload 对应。状态在本核，供 kernel 读取的参数在 GM。

| 字段 | 作用 |
| ---- | ---- |
| `task_ref` | slot 当前归属任务 |
| `generation` | 本次发布身份，0 不是有效发布 |
| `state` | EMPTY/FREE/FILLING/READY |
| `subtask_slot` | 当前逻辑子任务身份 0/1/2 |
| `mix_tracker` | 旧 MIX tracker 索引；当前 GM MIX 设置 UINT8_MAX |
| `cooperative` | 是否属于有序协作任务 |
| `block_idx` | 本次执行的逻辑 block |
| `candidate_index` | 普通 ready 候选在 admission.candidates 中的位置 |
| `predicate` | 条件是否尚未求值、通过或不通过 |
| `args_observed` | 是否已经观察本轮相关参数缓存 |
| `gm_mix` | 是否属于 GM mailbox 协调的 MIX 执行 |

`SchedulerFreeSlotClaim`：`worker_id` 指目标 worker；`slot_index` 指目标 slot；`generation` 是认领时看到的旧 generation，用于校验 slot 没变；`cluster_lane` 指 cluster 位置；`mix` 区分远端 MIX 存储和 own 存储。当前主循环创建 own claim，mix=false。

slot 生命周期为 `EMPTY → FREE → FILLING → READY → FREE`。执行时不另写 RUNNING 状态；ready bit 的消费和 generation 配合标识进度。

## 2.5 把任务变成函数调用

从 [fill_own_slots](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_dispatch.h#L451)（第 451 行） 跟到
[fill_dispatch_slot](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_ready.h#L2000)（第 2000 行）。

没有 admission 时，每轮只领取下一次普通执行需要的任务；即使有两个 slot，也不会无条件把所有 slot 填满后才运行。
有 admission 则转到下一篇的 `fill_ordered_slots`。

`fill_dispatch_slot` 的顺序：

1. 从 metadata 得到任务形状；INLINE 节点直接完成，不调用 kernel。
2. `prepare_dispatch_slot` 校验核类型/subtask 对应关系，从入口表查 kernel 地址。
3. 必要时观察 native 参数并求值 predicate。不通过时把函数地址设为 0，之后仍按正常逻辑完成处理。
4. 填 DispatchPayload，包括本次 logical block、subtask 对应物理 AIV lane。
5. `stage_dispatch_slot` 递增 generation，记录任务身份，写回 payload。
6. cache barrier 后 `commit_dispatch_slot` 设置本核 `local_ready_mask`。

[DispatchPayload](../../../src/a5/runtime/host_build_graph/runtime/dispatch_payload.h#L91)（第 91 行），512B：

**范围与用途：** 这是每个执行 slot 复用的 GM 参数存储，不是每个逻辑任务固定一份。当前主线每个 worker 使用两个 own payload；本核准备后让本核 kernel 读取，下一次绑定新任务时覆盖其内容。TaskPayload 是任务参数源，DispatchPayload 是一次执行 materialize 后的调用包。

| 字段 | 作用 |
| ---- | ---- |
| `function_bin_addr` | 将要间接调用的 kernel 入口；0 表示此次跳过函数调用 |
| `local_context` | 本次逻辑 block 信息 |
| `src_payload` | 另一套 early-dispatch 协议字段；当前 resident own payload 执行不轮询它 |
| `args[50]` | tensor 描述符指针、scalar 值和末尾两个 context 指针 |
| `global_context` | 当前物理 AIV lane 与 resident async 执行身份 |

[materialize](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_graph.h#L667)（第 667 行） 的参数布局：

```text
args[0 .. tensor_count-1]            = Tensor 描述符地址
args[tensor_count .. +scalar_count)  = scalar 的 64 位值
args[48]                            = &payload.local_context
args[49]                            = &payload.global_context
```

tensor 槽保存的是 Tensor 描述符地址，不是直接把数据 buffer 地址放进去。每个 Tensor 描述符跨度为 128B。
`SchedulerTaskPayloadArguments` 的 `tensors/scalars` 是解析出的区域指针，`tensor_count/scalar_count` 是区域元素数。
`SchedulerTaskInfo` 的 `task_ref/kernel_id/subtask_slot/core_type` 分别定位任务、函数、逻辑子任务和执行核类型。
`SchedulerDispatchPredicate` 的 `addr/target/elem_size/op` 分别指定被比较值地址、比较目标、元素字节宽度和比较运算。

[LocalContext 与 GlobalContext](../../../src/a5/runtime/host_build_graph/common/intrinsic.h#L108)（第 108 行）：

**范围与用途：** 两者都嵌在每个 DispatchPayload 里，每个 payload 各有一份。`LocalContext` 说明本次逻辑 block，`GlobalContext` 说明物理 lane 和 async 执行身份；这里的 Global 不表示全芯片单例。地址随 payload 稳定，字段可以随每次派发改变。

| 字段 | 作用 |
| ---- | ---- |
| `LocalContext.block_idx` | 当前任务逻辑 block 序号 |
| `LocalContext.block_num` | 当前任务全部逻辑 block 数，多波执行仍保持总数 |
| `LocalContext.async_ctx` | legacy ABI 字段，不属于 resident 路径，此处不展开 |
| `GlobalContext.sub_block_id` | MIX 中当前 AIV 的物理左右 lane：0/1 |
| `GlobalContext.async_present` | 是否附带有效 resident async writer 身份 |
| `GlobalContext.async_image_address / async_image_bytes` | async 所在 image 的基址/范围 |
| `GlobalContext.async_run_epoch` | 本轮 async 身份 |
| `GlobalContext.async_group_offset / async_group_index` | 当前执行对应的 async group |
| `GlobalContext.async_output_capture_refused` | 参数抓取选中了不能在异步完成前读取的输出时置位；每次 resident 执行重置，kernel 注册入口据此拒绝 |

kernel 用 `get_block_idx(args)`、`get_block_num(args)`、`get_sub_block_id(args)` 读取这些信息；它们与平台入口使用的无参数硬件 intrinsic 不是同一语义。

Tensor 的字段定义在 [TensorData](../../../src/common/host_build_graph/tensor_data.h#L39)（第 39 行）：
`buffer.addr/size` 为数据区地址和字节数；`owner_task_id` 为来源身份；
`start_offset` 为元素偏移；`version` 为依赖分析版本；
`ndims/dtype` 为维数/类型；`manual_dep` 为依赖处理标志；
`is_contiguous` 缓存连续布局判断；`address_space` 为地址所属空间；
`shapes[]/strides[]` 为形状和元素步长；`extent_elem_cache` 缓存最小覆盖跨度。
调度阶段只把描述符地址传给 kernel，不在此复制 tensor 数据。

## 2.6 主循环实际执行次序

回到 [run_ready_dispatch_loop](../../../src/a5/runtime/host_build_graph/aicore/resident_scheduler_impl.h#L448)（第 448 行）：

```text
检查 DMB EXIT；间隔检查共享错误
refresh ready inbox
消费本地完成 generation，处理依赖完成
native 图：服务控制节点
fill own slots：领取/准入/准备任务
pop 最旧 ready generation
最终 admission 许可检查；不允许则把 ready bit 放回
建立本次 async writer
启用参数抓取时完成 BEFORE_DISPATCH 边界
调用 function_bin_addr(args)
seal 本次 async 注册
启用参数抓取时完成 AFTER_COMPLETION 边界
local_completed_generations[slot] = generation
回到循环开头
```

每个核自己准备、自己执行自己的 own payload。调度器和 executor 是同一个常驻 AICore 程序里的两个阶段。
kernel 调用期间该核不会同时运行自己的 scheduler；其他核仍在各自循环运行。

无执行且无调度进展时先刷新 `pending_completed`，再做本核 NOP 自旋；这不是 OS sleep。
异步服务有自己的有界轮询机会，详见第四篇。

缓存相关代码读 [scheduler_memory.h](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_memory.h#L31)（第 31 行）：
`publish_cache_line`=写回+屏障；`observe_cache_line`=失效+屏障；
`gm_query` 用于低成本探测；关键跨核确认按协议使用原子读取；
`gm_compare_exchange` 的返回值是观察到的旧值，应与 expected 比较。
`volatile` 不替代缓存维护或跨核原子协议。
