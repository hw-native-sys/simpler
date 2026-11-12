# 4. resident 条件分支：native graph 与异步完成

[总索引](README.md) · [上一段：协作与完成](03-cooperative-completion.md)

结构的作用范围、数量、存放位置、读写者和生命周期可对照 [数据结构总览](00-data-structure-scopes.md)；下面在关键结构的字段表前也说明其范围与用途。

这两条分支仍使用前三篇的 bootstrap、ready、own slot、依赖传播和退出协议。普通图、无异步操作的例子可沿对应的空分支直接跳过。

## 4.1 native graph 接入哪些位置

分支判断在 [aicore_execute](../../../src/a5/runtime/host_build_graph/aicore/aicore_executor.cpp#L163)（第 163 行）：
`native_header_offset != 0` 选择 WithNativeGraph 模板。

| 接入位置 | native 行为 |
| -------- | ----------- |
| local config 初始化 | 校验 native image、epoch 和 control inbox |
| bootstrap 扫描 | 区分 TOP_KERNEL/CHILD_KERNEL/控制节点/HOST_DONE；为 child 绑定描述符参数 |
| fanin 读取 | 由普通 task_ref 数组切换到 template 的局部 CSR 和 completion 映射 |
| ready 发布 | 纯 AIC 和 MIX kernel 去共享 AIC 队列，纯 AIV 去共享 AIV 队列；控制节点去独立 control inbox |
| 执行循环 | 在 completion 与 fill 之间服务有界数量的控制工作 |
| 参数 materialize | CHILD_KERNEL 从 instance.args_base 和 child plan 中读取参数 |
| 总完成 | 控制节点也必须完成依赖传播，计入 logical resolved |

[SchedulerNodeKind](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_native_graph.h#L26)（第 26 行）：

- `TOP_KERNEL`：原始顶层 kernel。
- `CHILD_KERNEL`：graph instance 内部 kernel。
- `CONTROL_DUMMY`：只携带依赖语义的控制节点。
- `GRAPH_ENTER`：子图进入控制节点。
- `GRAPH_EXIT`：子图完成汇合节点。
- `HOST_DONE`：输入时已经完成的节点。

子图外部依赖通过 completion_refs 指向相应完成节点，不能把 GRAPH shell 进入当成整个子图完成。

在 [scheduler_service_controls](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_graph_control.h#L23)（第 23 行），本核每片最多开始 8 个控制节点、消耗 64 个 work units。
从 own 或轮转 peer control inbox 领到控制节点后，将其 BLOCKED→DONE，随后沿同一 resolver 的 CSR/remaining 或兼容 wake-list 分支传播完成；解析未结束时保存 `local.control_resolve`，下一轮继续。
只有 resolver 真正结束才增加 pending_completed。

控制节点不领取业务 slot，也不进入 kernel admission。即使当前 payload slot 或 admission candidate 已占用，主循环仍在 fill 之前给控制服务一个有界切片；两个共享业务队列不会替代这条控制进展路径。

## 4.2 native 图的数据结构字段

定义集中在 [scheduler_native_graph.h](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_native_graph.h#L35)（第 35 行）。

**范围与用途：** NodeBinding 为 12B，按展开后的 resident 节点各一份，放在 GM 只读表中，用于把 task_ref 映射回源节点与实例。

| SchedulerNodeBinding 字段 | 作用 |
| ------------------------- | ---- |
| `source_index` | 本节点对应的源任务/定义内节点索引，按 kind 解读 |
| `instance_index` | 所属 graph instance；不属于 instance 时为无效标记 |
| `kind` | 节点类别 |

**范围与用途：** native 图每轮一份 128B、64B 对齐的 GM image 头，version=2，为所有核提供布局入口；它本身不记录每个核的调度进度。

| SchedulerNativeGraphHeader 字段 | 作用 |
| ------------------------------- | ---- |
| `magic / version / header_bytes` | image 身份、版本和头大小 |
| `image_bytes` | image 范围 |
| `node_count` | 展开后的 resident 节点数 |
| `original_task_count` | 原始顶层任务条目数 |
| `definition_count / instance_count` | 子图模板数/实例数 |
| `bindings_offset` | 每节点 binding 数组 |
| `completion_refs_offset` | 原始任务索引到 resident 完成节点的映射 |
| `templates_offset / instances_offset` | 模板与实例数组 |
| `control_inboxes_offset / control_inbox_count` | 控制节点队列区及数量 |
| `readonly_end_offset` | 只读区域边界 |
| `run_epoch` | 本轮图身份 |

**范围与用途：** 每个去重后的子图定义一份 128B、64B 对齐的 GM 模板，多个实例共享它。它描述结构和布局，不保存某次实例的实际边界参数地址。

| SchedulerGraphTemplate 字段 | 作用 |
| --------------------------- | ---- |
| `definition_offset / definition_bytes` | 原始 Definition 的位置/大小 |
| `child_plans_offset` | 每个 child 的执行计划数组 |
| `fanin_offsets_offset` | CSR 每个节点的边区间表 |
| `fanin_indices_offset` | CSR 局部前驱索引表 |
| `sinks_offset` | 子图 sink 节点列表 |
| `task_count / edge_count / sink_count` | 定义内节点数、边数、sink 数 |
| `args_bytes` | 一次实例所需参数描述符存储大小 |
| `execution_count` | 模板所描述的 block/subtask 执行数量 |
| `required_heap` | 实例所需 heap 字节数 |
| `legacy_execution_storage_bytes` | 与输入 Definition 对齐校验的 execution_storage_bytes；保留字段名不表示转入 legacy 流程 |
| `graph_key` | 图定义标识 |

**范围与用途：** 每个模板中的每个 child 一份 64B GM 计划，随模板被多个实例共享；其中参数偏移与具体 instance.args_base 合起来才得到本次 child 的参数地址。

| SchedulerGraphChildPlan 字段 | 作用 |
| ---------------------------- | ---- |
| `metadata` | 本 child 的 SchedulerTaskMetadata |
| `tensors_offset / scalars_offset` | 在实例 args 区中该 child 的 tensor/scalar 区偏移 |
| `predicate_offset` | 实例 args 区的 predicate 位置；无 predicate 时为哨兵 |
| `tensor_count / scalar_count` | 两类参数数量 |
| `execution_prefix` | 本 child 前面累计的执行数量前缀 |

**范围与用途：** 每次子图实例化一份 64B GM 定位记录，指向该实例独立的边界参数、heap 和 args 区。同一个 template 可对应多个 instance。

| SchedulerGraphInstance 字段 | 作用 |
| --------------------------- | ---- |
| `definition_index` | 使用哪个模板 |
| `original_shell_id` | 原始 GRAPH shell 的身份索引 |
| `enter_ref / exit_ref` | 实例进入/退出控制节点 |
| `child_begin / child_count` | 展开的 child 节点区间 |
| `boundary_tensor_count / boundary_scalar_count` | 实例边界参数数量 |
| `boundary_tensors_address / boundary_scalars_address` | 边界参数区域 GM 地址 |
| `heap_base` | 本实例 heap GM 基址 |
| `args_base` | 本实例 materialized 参数区 GM 基址 |

参数绑定读 [scheduler_bind_graph_arguments](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_graph_binding.h#L158)（第 158 行）。
它把 child 需要的描述符和 scalar 绑定到 instance 存储，并在 bootstrap scan barrier 前发布；
后续执行核仍需观察这些参数。绑定描述符不等于复制 tensor buffer 的业务数据。

## 4.3 async 为什么让 kernel 返回与 DONE 分离

常见例子是 kernel 提交某种设备异步操作，函数已经返回，但结果还不能被后继读取。
resident 需要先等已注册 completion 条件，再发布 task DONE。

顺序沿以下函数：

1. [scheduler_async_begin_execution](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_async_control.h#L310)（第 310 行）：把 task/block/subtask 映射到本次 AsyncWriter，写 GlobalContext。
2. kernel 执行：使用 writer 登记 completion 条件。
3. [scheduler_async_seal_execution](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_async_control.h#L330)（第 330 行）：封存条件，发布 group 数据和检查任务；之后释放本次执行资源。
4. [scheduler_async_close_computation](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_async_control.h#L394)（第 394 行）：所有 block/subtask 的计算已完成，给逻辑任务关闭 computation。
5. [scheduler_service_async_slice](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_async_finish.h#L82)（第 82 行）：交替服务 CHECK 与 FINISH，每次有界预算。
6. CHECK 观察 completion 条件、完成所需 acquire，减少该任务 pending group 数。
7. computation 已关闭且 pending=0 时，仅有一个 FINISH owner。
8. [scheduler_async_finish_step](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_async_finish.h#L19)（第 19 行） 写任务 DONE，运行可续行 resolver，传播完所有后继后增加 pending_completed。

参数抓取选中输出时，入口会设置 `async_output_capture_refused`；异步登记入口据此拒绝过早输出快照。该限制由有效字段表达，每次 dispatch 重置。

有 async 布局但此次 kernel 未登记非空条件，仍可同步完成；不能仅凭 async 指针非空就断言任务一定延迟。

[SchedulerAsyncLocalState](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_async_control.h#L19)（第 19 行）：

**范围与用途：** 每个 worker 一份私有 async 服务状态，记录本核扫描到哪里、何时服务和本地错误。待完成条件、共享进度及独占服务权在 GM 的 group/join/lease 中，不在这个局部对象里。

| 字段 | 作用 |
| ---- | ---- |
| `base / bytes` | async 所在 image 的基址和范围 |
| `layout` | 本轮布局指针 |
| `check_cursor / finish_cursor` | 两类服务的轮转扫描游标 |
| `failed_task_ref / error_code` | 本地首次失败任务和错误码 |
| `business_returns` | 自上次 async 服务以来的业务 kernel 返回数 |
| `finish_first` | CHECK/FINISH 轮替选择状态 |
| `work_activated` | 已知本轮存在需要服务的 async 工作 |
| `idle_probe_countdown` | 未激活时降低探测频率的循环倒计数 |

主循环在有业务返回积累时也给 async 服务机会；空闲路径同样服务。倒计数按调度机会计，不是 sleep。

## 4.4 async 共享布局、所有权和条件

[SchedulerAsyncLayout](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_async_layout.h#L18)（第 18 行） 的全部字段：

**范围与用途：** 启用 async 的一轮运行只有一份 128B、64B 对齐的 GM 布局描述，所有服务核读取。它定位整轮 async 数据，具体某次执行的条件由独立 AsyncGroupData 保存。

| 字段 | 作用 |
| ---- | ---- |
| `run_epoch / image_bytes` | 本轮身份和可访问范围 |
| `task_map_offset` | task_ref 到 async task 紧凑索引的映射 |
| `task_data_offset` | async task 描述表 |
| `groups_offset` | 各次物理执行的条件 group 表 |
| `group_leases_offset` | CHECK group 所有权/状态表 |
| `group_progress_offset` | CHECK 的可续行检查进度 |
| `joins_offset` | 每任务计算与异步完成的汇合字 |
| `finish_leases_offset` | FINISH 所有权/状态表 |
| `finish_progress_offset` | FINISH 的 resolver 续行状态 |
| `check_directory_offset / finish_directory_offset` | 两类可服务工作位图 |
| `task_count / async_task_count / group_count` | 总调度节点数、有 async 布局的任务数、group 数 |
| `sdma_guard_count / sdma_guards_offset` | SDMA 提交互斥 guard 数与位置 |
| `activity_offset` | 本轮 async 活动发布标志 |

小结构字段：

| 结构 | 全部字段及作用 |
| ---- | -------------- |
| `SchedulerAsyncTaskData` | `task_ref`：逻辑任务；`group_base`：首 group 索引；`block_count`：逻辑 block 数；`async_mask`：需要 async group 的子任务位图 |
| `SchedulerAsyncGroupProgress` | `pending_mask`：尚未通过的条件位图；`condition_cursor`：下次检查位置；`acquire_cursor`：完成 acquire 的推进位置 |
| `SchedulerAsyncFinishProgress` | `resolver`：可迁移的完整依赖解析游标 |
| `SchedulerAsyncDirectoryWord` | `bits`：64 个工作项的可发现位 |
| `SchedulerAsyncActivity` | `activated`：本轮存在 async 工作，device 从 0 发布成 1 |
| `SchedulerAsyncLease` | `word`：高位 generation、低 3 位 phase；phase 为 DORMANT/RUNNABLE/OWNED/TERMINAL/FAILED |
| `SchedulerAsyncLeaseOutcome` | `result`：操作结果；`word`：成功的新状态或失败观察值 |
| `SchedulerAsyncTaskJoin` | `word`：低 16 位 pending group 数，接下来的 16 位 registered group 数，bit32 为 computation_closed |
| `AsyncSdmaSubmissionGuard` | `word`：同物理 cluster SDMA 提交路径的独占状态 |
| `AsyncReservation` | `run_epoch/group_index/serial`：本次登记预留身份；`capacity`：预留容量 |

`SchedulerAsyncTaskData` 为 16B；CHECK progress、lease、join、directory word、activity 与 SDMA guard 各以 64B 行隔离。FINISH progress 为 192B，内含 160B resolver。

join 的原子操作使“最后一个 group 完成”和“computation 关闭”竞态时，也只能产生一个 FINISH owner。
lease generation 让旧 owner 无法提交到已经换代的服务记录。

[AsyncWriter](../../../src/a5/runtime/host_build_graph/runtime/async_condition_writer.h#L21)（第 21 行） 的 `image/image_bytes/run_epoch/group_offset/group_index`
分别是映像指针、范围、运行身份、group 偏移和 group 编号；与 GlobalContext 的 async 字段一一对应。

[AsyncGroupData](../../../src/a5/runtime/host_build_graph/runtime/async_completion_types.h#L28)（第 28 行）：

**范围与用途：** 每个配置 async 的 task/block/subtask 执行对应一份 GM group，负责执行的核登记条件并 seal，之后持有 CHECK lease 的核处理它。它可以活过业务 kernel 返回，且不随 own DispatchPayload 的下一次复用而丢失。这个 group 与 MIX 的一个 cluster/block group 不是同一含义。

header 总长 128B、64B 对齐。第一行包含登记计数/错误/预留/状态，以及不变的 group_index、run_epoch、task_ref、entries_offset、sdma_guards_offset；
第二行保存 block_index、subtask_slot、entries_bytes、sdma_guard_count、capacity。
封口发布第一行时也写回其中不变的 Host 字段；第二行保持只读。token 数组在 header 外独立按 64B 对齐。

| 字段 | 作用 |
| ---- | ---- |
| `run_epoch / task_ref / group_index` | 本轮、本任务、本 group 身份 |
| `block_index / subtask_slot` | 由哪次 block/subtask 执行登记 |
| `capacity` | 条件 entry 最大数量 |
| `entries_bytes` | entry 区的字节范围 |
| `count` | 已登记的条件数量 |
| `error_code` | 本 group 首次登记错误 |
| `reservation_serial` | 预留操作序号，识别旧 reservation |
| `reservation_limit` | 当前预留允许写入的条目数上限，不是绝对结束索引；0 表示没有活动预留 |
| `reservation_begin` | 当前预留区起点 |
| `reservation_written` | 当前预留已写数量 |
| `state` | OPEN/SEALED/FAILED；seal 后交给 CHECK |
| `sdma_guards_offset / sdma_guard_count` | 可访问提交 guard 区及数量 |
| `entries_offset` | completion entry 区位置 |

[DeferredCompletionEntry](../../../src/a5/runtime/host_build_graph/runtime/aicore_completion_mailbox_types.h#L45)（第 45 行）：
`addr` 为后端解释的 completion 地址；`backend_cookie` 为后端辅助身份；
`expected_value` 为目标完成值；`engine` 标识后端；`completion_type` 标识 counter/event-record 条件；
`_pad` 为填充。字段的具体读取语义由对应后端实现决定，不能把所有 entry 都当成对同一种内存 counter 的比较。

共享类型头有 AICPU mailbox 的说明，但此次 resident 的 CHECK/FINISH 在 AICore 服务，不转入该 mailbox 调度流程。
