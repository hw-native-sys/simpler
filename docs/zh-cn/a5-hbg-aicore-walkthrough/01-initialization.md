# 1. 初始化与启动数据

[回到总索引](README.md) · 下一步：[依赖与执行](02-ready-execution.md)

结构的作用范围、数量、存放位置、读写者和生命周期可对照 [数据结构总览](00-data-structure-scopes.md)；下面在关键结构的字段表前也说明其范围与用途。

## 1.1 起点：device 已有本轮 GM 数据

跳过 host 的分配、建图和上传，只接受这些输入：Runtime 启动描述、只读图/参数、scheduler state，以及每个核的 context 和 payload 存储。

初值契约可核对 [scheduler_types.h](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_types.h#L1711)（第 1711 行）：尚未完成任务的 `state=BLOCKED`，`wake_list_head=-1`，`next_waiter=-1`，`waiting_producer=-1`；领取/完成游标为 0；依赖 remaining 由 Host 填入未完成且去重的前驱数；ready inbox 的 head/tail 均为 -1；payload 的 `args[48]`、`args[49]` 已指向自身两个 context。已经 inline 完成的节点另作 DONE 处理。
静态依赖路径通过 ready-pool header 的 `dependencies_offset` 定位 CSR/remaining；为 0 时保留 wake-list 路由。这只是输入契约，不沿 Host 构造流程展开。

从 [AicpuExecutor::init](../../../src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp#L360)（第 360 行） 开始，leader 先调用 `pre_handshake_init`：

- 校验 worker 数在 1..108。
- 清 `cores_`、物理核列表、握手计时。
- 清每核 `teardown_gates`，发布屏障，避免沿用上一轮退出许可。
- 保存核数、AICPU 线程数、寄存器地址表。
- 清握手失败标志、已退休核计数；按配置初始化诊断采集。

[AicoreLifecycle](../../../src/a5/runtime/host_build_graph/aicpu/aicore_lifecycle.h#L68)（第 68 行） 的成员：

**范围与用途：** `AicoreLifecycle` 是 AICPU executor 内的一份共享生命周期管理对象；其中 `CoreState` 按 worker 各一条，`ThreadHandshakeTiming` 按 AICPU 线程各一条。它保存发现结果和启动/退出进度，不保存每个业务任务的执行参数。

| 字段 | 作用 |
| ---- | ---- |
| `CoreState.reg_addr` | 该核寄存器窗口地址；0 说明尚未成功发现 |
| `CoreState.physical_core_id` | 核报告的硬件编号 |
| `CoreState.core_type` | AIC 或 AIV |
| `cores_[]` | 按 worker 编号记录上述三项 |
| `physical_core_ids_[]` | 物理核编号数组，供 PMU 初始化等使用 |
| `ThreadHandshakeTiming.start_cycles/end_cycles` | 本 AICPU 线程发现其核分区的开始/结束时间 |
| `thread_handshake_timing_[]` | 各线程的上述记录 |
| `handshake_failed_` | 任一发现分区失败的共享标志 |
| `retired_core_count_` | 已证实停止执行或未打开窗口的核数 |
| `lifecycle_traces_` | GM 中生命周期计时数组 |
| `core_count_` | 本轮启动的 worker 总数 |
| `aicpu_thread_num_` | 负责生命周期的 AICPU 线程数 |
| `regs_base_` | 用 physical_core_id 索引的寄存器地址表基址 |

## 1.2 AICore 报到，AICPU 发现拓扑

读 [平台 kernel 入口](../../../src/a5/platform/onboard/aicore/kernel.cpp#L118)（第 118 行）。设启动 cluster 数为 C：

- AIC 的 worker_index 为硬件 block_idx，范围 [0,C)。
- AIV 的 worker_index 为 `C + block_idx * subblockdim + subblockid`；此处一个 cluster 两个 AIV。
- 初始化本核的 profiling 地址和 report_epoch，再调用 `aicore_execute(runtime, worker_index, core_type)`。

入口参数字段含义：`runtime_args` 是 Runtime GM 地址；`enable_profiling_flag` 是诊断位图；`force_simt_anchor` 为编译分类用入口参数，正常为 0；`chip_swimlane_aicore_rotation_table` 是每核采集 head 的槽地址表；`aicore_pmu_ring_addrs` 是每核 PMU ring 地址表；`pmu_reg_addrs` 是按物理核索引的 PMU 寄存器表；`report_epoch` 是本轮报告身份。诊断地址未启用时清空本地缓存，避免继承上次 launch。

[SchedulerBootstrapInputs](../../../src/common/host_build_graph/runtime.h#L219)（第 219 行），独占 64 字节：

**范围与用途：** 每轮启动描述中一份，放在 GM。所有 AICPU/AICore 从这里获知本轮模式和 context 数组入口；执行期间只读。它不是每核一份，也不是完整调度状态。

| 字段 | 作用 |
| ---- | ---- |
| `runtime_mode` | 选择 resident/明确 fallback；这里读 1 或 2 才走主线 |
| `reserved_` | 对齐保留字段 |
| `worker_context_base` | worker 0 的 SchedulerWorkerContext 完整 GM 地址 |

[AICore 报告](../../../src/a5/runtime/host_build_graph/aicore/aicore_executor.cpp#L70)（第 70 行） 写入 [Handshake](../../../src/common/host_build_graph/runtime.h#L119)（第 119 行），每核一条 64B cache line：

**范围与用途：** `Handshake` 按 worker 每核一份，位于 GM；核只向自己的条目报到，负责它的 AICPU 线程应答。它覆盖启动和退休握手，不随每个 task 更新。

| 字段 | 写者 → 读者 | 作用 |
| ---- | ----------- | ---- |
| `aicpu_ready` | AICPU → AICore | context 配置完成后写 RESIDENT_READY；native 报告先置为 PENDING |
| `aicore_done` | AICore → AICPU | 启动报告有效标志，写 worker_index+1；不是逐任务完成计数 |
| `task` | AICPU → AICore | resident 中是 SchedulerWorkerContext 地址 |
| `core_type` | AICore → AICPU | AIC/AIV 类型 |
| `physical_core_id` | AICore → AICPU | 硬件核编号，供寄存器寻址 |
| `report_epoch` | AICore → AICPU | 非零 epoch 的本轮报告提交标记，防止接受旧报告 |

report_epoch 非零时，AICore 先清旧的反向应答字段、写身份和 done，经 store barrier 再写 epoch，最后 DCCI 写回并 DSB。
epoch 为 0 时，沿无 stamp 的报告协议写身份，屏障后写 done。AICPU 接受条件始终要求 done 非零；非零 expected_epoch 还必须匹配 epoch。

[handshake_partition](../../../src/a5/runtime/host_build_graph/aicpu/aicore_lifecycle.cpp#L99)（第 99 行） 中线程 t 负责
`[t*N/T, (t+1)*N/T)`。每轮扫描自己分区，接受报告后执行 `rmb()`，再读报告 payload 并建立 `cores_[i]`。

所有分区完成后，leader 进入 [post_handshake_init](../../../src/a5/runtime/host_build_graph/aicpu/aicore_lifecycle.cpp#L154)（第 154 行）：

1. 分别为 AIC、AIV 编排 `type_rank`。
2. 校验 AIV 数恰好为 AIC 数的两倍，与图中预期 worker/cluster 数一致。
3. 按 compact launch 编号建立 cluster：AIC worker c、AIV worker C+2c、C+2c+1。
4. 每个核都设置 `is_scheduler=1`、`scheduler_index=worker_index`；并非只有 AIC 调度。
5. 校验图需求的 AIC/AIV 数足够；填写 run_control 的实际核数和超时。
6. 发布 `topology_layout=COMPACT_LAUNCH`，表示这套拓扑已被校验。

## 1.3 每核 context 的字段

[SchedulerWorkerContext 定义](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_types.h#L1216)（第 1216 行）。当前结构为 384B、64B 对齐：前五行配置 READY 后不变，最后一行保存本核诊断时间。

**范围与用途：** 这是 GM 中每个 worker 的一份配置入口，按 `worker_index` 寻址。它既包含本核身份和本核 payload 起点，也包含指向本轮共享任务表、队列和 RunControl 的偏移。不同 worker 的 context 可以指向同一个共享对象；并不是每核复制整张图。启动前输入和 AICPU 拓扑发现共同填好配置；READY 后配置只读，末尾统计区由本核更新。

| 字段 | 作用 |
| ---- | ---- |
| `core_type` | 本核 AIC/AIV |
| `physical_core_id` | 本核物理编号 |
| `type_rank` | 本核在同类核中的编号 |
| `run_control_offset` | 全局运行控制对象在 state 中的偏移 |
| `task_controls_offset` | 每任务可变控制表偏移 |
| `graph_storage_address` | ChipTaskStorage 数组完整 GM 地址 |
| `scheduler_state_base_address` | 本轮 scheduler state 完整 GM 基址 |
| `dispatch_payload_offset` | 本 worker 两个 own payload 中第一个的偏移；当前 resident 执行复用这两个 payload |
| `trace_cells_offset` | 每任务执行记录数组偏移 |
| `task_window_last_index` | 图存储容量的末索引，用于范围检查/诊断；不能当 ring mask |
| `graph_task_count` | resident 可寻址的调度节点数 |
| `worker_index` | 本 worker 的启动编号 |
| `aicpu_lifecycle_traces_offset` | AICPU 生命周期记录数组偏移 |
| `task_metadata_offset` | 紧凑任务元数据表偏移 |
| `ready_inboxes_offset` | 按任务类型分组的 inbox 数组偏移 |
| `ready_directory_offset` | inbox 非空目录位图偏移 |
| `activity_buffers_offset` | 本核活动/空闲记录存储偏移 |
| `worker_contexts_offset` | 全部 worker context 数组偏移 |
| `callable_addresses_address` | kernel 入口地址表完整 GM 地址，位于 callable 注册存储中 |
| `runtime_worker_count` | 本轮 worker 总数 |
| `callable_addresses_count` | kernel 地址表长度，用于 kernel_id 越界检查 |
| `pmu_cells_offset` | 每次 subtask 执行的 PMU 记录存储偏移 |
| `args_dump_address` | 本轮参数抓取 image 的完整 GM 地址，0 表示未启用 |
| `graph_summary_offset` | 本轮不可变图/拓扑摘要偏移 |
| `scheduler_count` | scheduler 数；当前等于 worker 数 |
| `cluster_count` | cluster 总数 |
| `cluster_index` | 本核所属 cluster |
| `scheduler_index` | 本核 scheduler 编号；用于 bootstrap 分片 |
| `is_scheduler` | 是否承担调度；当前主线所有核为 1 |
| `cluster_worker_ids[3]` | 本 cluster 的 AIC、AIV0、AIV1 worker 编号 |
| `mix_payloads_offset / mix_traces_offset` | 按每个 AIV 额外预留两个 MIX payload 及对应 trace 的区域；AIC 没有这两个额外 payload。当前 GM MIX 用各核 own slot，不沿 SSBUF 分支消费 |
| `native_header_offset` | 0 为普通图；非零为 resident native graph 头偏移 |
| `original_task_count` | 原始 ChipTaskStorage 条目数；native 展开节点数可不同 |
| `graph_view_version / graph_view_bytes` | 图视图协议版本、结构大小 |
| `scheduler_state_bytes` | 本轮 state 存储字节数，做范围检查 |

因此，按内存布局计数：AIC worker 对应 2 个 own payload；AIV worker 对应 2 个 own payload + 2 个额外预留 MIX payload，共 4 个。按当前 resident 主循环实际执行计数：两类 worker 都只使用自己的 2 个 own payload，GM MIX 也复用它们。

本 worker 的第 s 个 own payload 地址为 `scheduler_state_base_address + dispatch_payload_offset + s * sizeof(DispatchPayload)`，其中 s 为 0 或 1，当前 `sizeof(DispatchPayload)=512`。own 表示本核自己准备并执行，不表示只供非 MIX 任务使用；两个 slot 也不表示一个核同时执行两个 kernel。

WorkerContext 的物理分组与访问范围如下。这里的间距由 `alignas(64)` 和字段顺序约束，不存在供业务读取的 padding 字段。

| 字节范围 | 内容与使用者 |
| -------- | ------------ |
| 0–63 | 核身份、scheduler 标志、RunControl/TaskControl/payload 定位及 peer context 定位 |
| 64–127 | 完整 graph view、格式/范围与 GraphSummary 定位 |
| 128–191 | metadata、ready、callable 表和保留 MIX 存储定位 |
| 192–255 | scheduler/cluster 拓扑与 AICPU lifecycle trace 定位 |
| 256–319 | task trace、activity、PMU、参数抓取的配置入口 |
| 320–383 | 本核写入的启动诊断行 |

保留的诊断字段：

| 字段 | 统计含义 |
| ---- | -------- |
| `bootstrap_task_count` | 本核 bootstrap 扫描到的任务数 |
| `bootstrap_start_cycles` | 本核开始 bootstrap 的时间 |
| `target_bootstrap_end_cycles` | 扫描屏障结束且本核 free slots 初始化后、首批 fill 前的时间 |
| `trace_aicore_entry_cycles` | AICore 进入本轮报告阶段的时间 |
| `trace_handshake_publish_cycles` | 本轮报告发布完成的时间 |
| `trace_register_release_cycles` | 观察到 DMB 开放的时间 |
| `trace_descriptor_cache_observed_cycles` | 完成图描述缓存观察的时间 |

每次 kernel 与 scheduler 阶段的观测来自对应 task/executor/protocol 记录；不能把此 context 当作全套热路径聚合计数器。

## 1.4 READY、bootstrap 与 DMB

[publish_context_partition](../../../src/a5/runtime/host_build_graph/aicpu/aicore_lifecycle.cpp#L300)（第 300 行） 先写 `Handshake.task=context_address`，store barrier 后写 `aicpu_ready=RESIDENT_READY`。
只有已接受本轮报告且配置成功的 worker 会被应答。

AICore 在 READY 前通过 bootstrap 基址定位 pending_context，只用于找到超时配置；不能拿这个旧缓存视图执行。
READY 后重新观察 context 的相关 cache line，再进入 [run_resident_scheduler](../../../src/a5/runtime/host_build_graph/aicore/resident_scheduler_impl.h#L685)（第 685 行），依次建立 GraphView、LocalState、admission 和 async 状态。

[bootstrap_ready_graph](../../../src/a5/runtime/host_build_graph/aicore/resident_scheduler_impl.h#L268)（第 268 行） 按 scheduler_index 均分节点：

1. 处理本分片节点；native child 绑定参数描述符。
2. 有依赖 header 时读取 remaining：0 发布 ready，非零等待前驱通知；没有 header 时才挂 wake list。native 控制节点进入 control inbox，kernel 进入其类型队列。
3. 所有核完成扫描，推进 `bootstrap_scan_arrived_count`；最后一核置 `bootstrap_scan_complete=1`。
4. 初始化本核两个 free slot，尝试准备首批任务；可能只是等待协作准入，并不保证每核都有可执行任务。
5. 推进 `bootstrap_arrived_count`；最后一核置 `bootstrap_complete=1`。

启用参数抓取时，AICPU 在执行和等待阶段还轮询 `ResidentArgsDumpService`，但不接管逐任务调度。AICPU 等到最后一个标记后，才经 [platform_init_aicore_regs](../../../src/a5/platform/shared/aicpu/platform_regs.cpp#L34)（第 34 行） 写 `DATA_MAIN_BASE=AICPU_IDLE_TASK_ID`。
AICore 在 [763 行](../../../src/a5/runtime/host_build_graph/aicore/resident_scheduler_impl.h#L763)（第 763 行） 等这个值或 EXIT；正常开放后写 COND idle，再进入执行循环。

READY 发布的是可读配置；DMB 开放的是 kernel 执行权限。bootstrap 期间允许处理调度控制和准备任务，尚不调用业务 kernel。

## 1.5 RunControl 与 GraphSummary

[SchedulerRunControl](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_types.h#L1152)（第 1152 行） 的字段：

**范围与用途：** 每轮一份，位于 GM，由所有参与核和 AICPU 共享。它用于判断全图 bootstrap、进展、完成与错误。当前为 448B、64B 对齐；动态完成计数从 offset 192 开始，错误载荷从 256 开始，最终错误发布字在独立的 offset 384 行。AICPU 补齐配置并监视，AICore 按协议更新计数和错误；各字段有各自写者，不是任意核都可以覆盖整个结构。

| 字段 | 作用 |
| ---- | ---- |
| `aic_task_count / aiv_task_count` | 图配置提供的类型工作计数，不是实时完成计数 |
| `active_worker_count` | 实际 worker 总数 |
| `expected_task_count` | 本轮应完成的逻辑调度节点总数 |
| `inline_completed_count` | 输入图中已经完成、不再由主执行循环完成的节点数 |
| `aic_active_worker_count / aiv_active_worker_count` | 实际两类核数 |
| `aic_worker_demand / aiv_worker_demand` | 图要求的核资源需求，拓扑发现后校验 |
| `dispatch_payloads_offset` | 所有 own payload 的起点 |
| `task_metadata_offset / ready_inboxes_offset / ready_directory_offset / graph_summary_offset` | 对应 GM 表的偏移 |
| `scheduler_count` | 本轮 scheduler 数 |
| `resolved_task_count` | 依赖完成传播结束的逻辑节点累计数，由各核批量相加 |
| `bootstrap_scan_arrived_count / bootstrap_scan_complete` | 扫图阶段到达数/完成标记 |
| `bootstrap_arrived_count / bootstrap_complete` | 首批准备阶段到达数/完成标记 |
| `scheduler_timeout_cycles` | 调度等待/进展 watchdog 的时间预算 |
| `chip_swimlane_level` | 采集级别 |
| `sampled_task_timing_enabled` | 指定任务计时是否开启 |
| `error_claimed` | 第一条错误记录的写者选举 |
| `error_task_ref / error_core_id / error_core_type` | 出错任务、物理核、核类型 |
| `error_graph_task_count / error_storage_address / error_task_window_last_index` | 出错时图视图信息 |
| `error_site` | 报错点枚举 |
| `error_run_epoch / error_order_key / error_block_idx / error_group_seq` | 出错运行/准入顺序/block/MIX group 身份 |
| `error_worker_index / error_cluster_lane` | 出错 worker 和 cluster lane |
| `error_identity_fields` | 哪些附加错误身份字段有效 |
| `scheduler_error` | 最后发布的错误码；0 表示尚无已发布错误 |

[SchedulerGraphSummary](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_types.h#L1035)（第 1035 行） 的全部字段：

**范围与用途：** 每轮一份 GM 摘要，表示这轮图有哪些任务形状、需要哪些调度区域及其布局。AICPU 确认拓扑并发布 READY 后，它保持不变。与 RunControl 的区别是：Summary 描述静态事实，RunControl 还承载动态执行进度。

| 字段 | 作用 |
| ---- | ---- |
| `mix_task_count` | 多子任务 MIX 节点数 |
| `cooperative_task_count` | 需要 admission 的节点数 |
| `worker_count / cluster_count` | 预期 worker/cluster 数 |
| `admission_member_word_count` | 每个成员位图包含多少个 64 位 word |
| `admission_records_offset / admission_record_stride` | 按 task_ref 寻址的 record 区及每条跨度 |
| `admission_log_offset / admission_control_offset` | 全局 admission 顺序日志和发布控制对象 |
| `group_mailboxes_offset` | 按 cluster 寻址的 GM MIX mailbox 区 |
| `mix_close_seals_offset` | 各 MIX task/cluster 的关闭凭据区 |
| `run_epoch` | 本轮调度协议身份 |
| `protocol_layout_offset` | 可选协议诊断区 |
| `topology_layout` | GENERAL 或已经校验的 COMPACT_LAUNCH |
| `ready_pool_offset` | 非零启用分组 ticket ready pool；0 使用 inbox/directory |
| `async_layout_offset` | 可选异步完成区布局 |

到这里，AICore 已有配置、图视图和私有状态；继续读下一篇的依赖、ready 和 payload。
