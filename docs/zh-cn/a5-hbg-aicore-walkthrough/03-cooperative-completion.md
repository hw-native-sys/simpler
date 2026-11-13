# 3. MIX/SPMD、完成传播与退出

[总索引](README.md) · [上一段：依赖与执行](02-ready-execution.md) · [条件分支：native/async](04-native-async.md)

结构的作用范围、数量、存放位置、读写者和生命周期可对照 [数据结构总览](00-data-structure-scopes.md)；下面在关键结构的字段表前也说明其范围与用途。

## 3.1 什么任务需要有序准入

在 [scheduler_task_requires_admission](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_types.h) 看条件：
任务 executable、非 inline，并满足 logical_block_num>1、MIX、SYNC_START 之一。

若图中此类任务数为 0，`local.admission=nullptr`，走普通路径。
若非 0，`scheduler_initialize_admission` 挂上本地状态，这一轮图中的普通任务也受准入顺序约束。

[fill_ordered_slots](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_dispatch.h) 按下面的顺序推进：

1. 尝试发布自己上轮已认领的协作任务。
2. 在有限候选槽内从本类型共享 ready 队列发现工作；AIC 队列同时包含纯 AIC 和 MIX。
3. 普通任务先保留候选；协作任务独占认领，构造 record，再提交全局顺序日志。
4. 扫描本核尚未处理的 admission 日志。不是成员则跳过，是成员则建立 current obligation。
5. MIX 推进 GM mailbox；非 MIX SPMD 领取 block 并准备 own slot。
6. 本核当前协作义务可关闭后，推进 scan_cursor；普通候选再按许可准备。
7. 真正调用 kernel 前，再由 `scheduler_ordered_can_enter_slot` 检查一次。

“有序”指所有成员遵守同一准入顺序，不意味着整个图每个 kernel 都串行。

成员集合由 [scheduler_admission_expected_members](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_admission.h) 固定：普通非同步 SPMD 若 block 数少于同类型可用核数，
选取从 `member_rank_offset % eligible_count` 开始的循环区间，成员数为 block_num；否则使用完整可用集合。
Host 按 task_ref 顺序分别累计 AIC/AIV 的普通非同步 SPMD block 数，保存本任务之前的累计值。
不同资源类型、普通单 block、MIX 和 sync_start 不消耗该类型的轮转位置；变宽任务按实际 block 数推进。
MIX 和 sync_start 继续采用其完整资源成员集合。成员位图使用 compact worker index，不能套用物理核编号或交错编号假设。
一般拓扑仍经过合法性检查，COMPACT_LAUNCH 只是已验证拓扑的快速构建路径。

[SchedulerAdmissionControl](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_types.h)：

**范围与用途：** 存在协作任务时，本轮 GM 中一份共享控制对象。多个核竞争提交锁以发布同一条全局 admission 日志；每个核自己的阅读位置放在 AdmissionLocalState，不放在这里。
`commit_lock` 串行化日志提交，持锁者只做有界发布；
`commit_tail` 为已经完整提交的日志长度；
两者各占独立128B原子区域，控制对象总长256B。

[SchedulerAdmissionRecord](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_types.h)：

**范围与用途：** 每个协作任务一条有效 GM 记录，header 为 64B，位图跟在独立对齐的 record stride 内，存储按 task_ref 寻址。任务的独占认领者准备并发布它，参与该任务的所有 worker 读取同一条记录；提交后不再改写。

| 字段 | 作用 |
| ---- | ---- |
| `run_epoch` | record 属于哪次运行 |
| `task_ref` | 协作任务 |
| `order_key` | 在全局 admission 日志中的位置 |
| `member_words_offset` | 参与此任务的 worker 位图 |
| `mix_close_seals_offset` | 此任务各 cluster 的 MIX 关闭凭据；非 MIX 不使用 |
| `logical_block_num` | 逻辑 block 总数 |
| `member_count` | 位图中的成员数量 |
| `member_rank_offset` | Host 规划的同资源类型普通 SPMD block 前缀；MIX/sync_start 为 0，header 仍为 64B |
| `active_mask` | 本任务有效 subtask |
| `flags` | 任务属性副本 |

提交次序见 [commit_validated_record](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_admission.h)：
`record.order_key` → 发布 record → `log[tail]=task` → `admission_state=ADMITTED` → `commit_tail=tail+1`。
读取 tail 的核必须能读到前面完整发布的记录。

[SchedulerAdmissionLocalState](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_ready.h)：

**范围与用途：** 每个 worker 各一份本地进度，当前图存在协作任务时启用。它表示“我读到哪条 admission、我现在欠哪个任务的参与义务”，因此各核的 scan_cursor/current_task_ref 可以暂时不同；共享顺序仍来自同一条 GM 日志。

| 字段 | 作用 |
| ---- | ---- |
| `epoch` | 本轮身份 |
| `control_offset / records_offset / record_stride / log_offset` | 共享控制、record 数组、record 跨度、顺序日志 |
| `capacity` | 本轮预计进入日志的协作任务总数 |
| `member_word_count` | 成员位图长度 |
| `pending_task_ref` | 本核已独占认领、尚未完成日志提交的任务 |
| `scan_cursor` | 本核在全局 admission 日志中的处理位置 |
| `current_task_ref` | 当前对本核有资源义务的协作任务 |
| `current_block_num / current_active_mask / current_flags` | 当前义务的 block 数、subtask、属性 |
| `initialized` | 初始化成功标志 |
| `next_candidate` | 普通候选的轮转公平性游标 |
| `sync_claimed` | SYNC_START 任务中本成员是否已经领过 block |
| `ticket_exhausted` | 本成员是否已确认没有可领取票 |
| `candidates[2]` | 本核独占保留的 ready 候选 |
| `gm_mix` | 本 cluster MIX group 的本地推进状态 |
| `topology_cube_words[]` | 启动拓扑中 AIC worker 位图缓存 |
| `topology_vector0_words[]` | 启动拓扑中第一 AIV worker 位图缓存 |
| `topology_worker_count` | 缓存拓扑包含的 worker 数 |

`SchedulerOrderedCandidate` 的 `ready` 保存 ready claim，`slot_index` 指已为它准备的 slot（UINT8_MAX 表示还没有），`predicate` 保存求值结果，`args_observed` 保存参数观察状态。

普通 kernel 入口检查的核心含义是：本核已经处理到当前提交的协作日志尾部，且没有必须优先兑现的当前/待提交义务。许可只覆盖本次进入，不能缓存给下一次调用。

## 3.2 SPMD 如何领取和完成 block

读 [scheduler_claim_ordered_cursor](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_admission.h)。

非 SYNC_START 的多 block 任务走 ticket：
原子增加 `claim_tickets` 得到唯一 ticket，ticket 小于 block_num 才拥有合法 block；
然后增加 `next_block` 确认有效领取。最后一个有效确认者发布 `claim_seal=block_num`。
因此 `claim_tickets` 可能越界少量，`next_block` 仍有严格有效范围。

另一分支用 CAS 推进 `next_block`，SYNC_START 成员成功一次后设置 `sync_claimed`，不继续包办多个 block。
这里的 SYNC_START 资源/领取约束不能直接理解为所有核在同一个时钟周期进入 kernel。

[claim_ordered_block](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_admission.h) 把 task、block 和候选来源写入本核 slot；不再维护独立的 cooperative 字段。
同一逻辑任务可以分多波执行，payload.block_num 始终是总 block 数。

每次子任务返回后，[scheduler_contribute_subtask](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_completion.h) 贡献一次实际执行完成。
累计达到 `logical_block_num × active_mask.count()` 的最后一个贡献者处理逻辑完成；普通、SPMD 和 MIX 使用同一计数。

## 3.3 MIX：一个 cluster 内各核自己准备和执行

主链入口是 [scheduler_service_gm_mix](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_dispatch.h)，实现位于 [scheduler_gm_mix.h](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_gm_mix.h)。

按一个逻辑 block 走一遍：

1. cluster 的 AIC coordinator 领取 block，生成新的 `group_seq`。
2. AIC 填 GM descriptor：task、block、参与 lane 和各 lane 的前序 group。
3. 写回 descriptor 后，发布 `published_seq`，再为参与 AIV 更新 `assigned_seq`。
4. 各参与核观察分配、校验 epoch/order/group；等本核 own slots 都 free 后写 `arrived_seq`。
5. 确认本 group 的所有资源成员都 arrived，才各自准备自己的 own slot。
6. 每个核调用自己的 subtask kernel；没有业务 subtask 的资源成员直接释放。
7. 实际执行 lane 贡献一次子任务完成，再完成 trace 收尾并写 `released_seq`。
8. AIC 确认所有资源成员 release，退休该 group 的资源；不重复贡献完成计数。
9. 最后一个实际子任务触发逻辑完成或异步 FINISH；这可以早于 group 资源退役。

即使是纯双 AIV MIX，AIC 仍作为 coordinator 进入 resource_mask；它可能没有业务 kernel。
当前主线 descriptor 的逻辑 subtask 与物理通信 lane 保持对应。

ready 领取、组协调与 slot 管理是三个不同职责：AIC 从共享队列认领 MIX task 并发布 admission；完成路径也可由 AIV 直接持有 MIX 候选并推进 admission；各 cluster 的 AIC 领取该 task 的 block、协调 mailbox；每个执行成员通过 [scheduler_gm_mix_prepare_lane](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_gm_mix.h) 选择并填写自己的 `local->slots[index]`。
每个 worker 只有两个 own slot，普通任务与 MIX 共用 payload 和本核直接退休流程。coordinator 汇总成员 release，不替其他成员填写或回收 slot。

[SchedulerGroupMailbox](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_types.h)：

**范围与用途：** 每个 cluster 一份 GM mailbox，多个 MIX task/block 按 group_seq 先后复用它。descriptor 由该 cluster 的 AIC 发布；每个 lane 的 arrived/released ACK 由该 lane 发布，assigned_seq 由 AIC 发布。不能把它理解成每个 MIX 任务各有一个 mailbox。
`descriptor` 保存下面的 group 描述；`published_seq` 是其发布标记；
descriptor 占 128B，published_seq 位于单独的 64B 行；`lanes[3]` 各占 64B，mailbox 总长 384B。

| SchedulerGroupDescriptor 字段 | 作用 |
| ----------------------------- | ---- |
| `run_epoch` | 当前运行身份 |
| `task_ref` | MIX 任务 |
| `order_key` | 该任务 admission 顺序位置 |
| `group_seq` | cluster 内本次 group 序号 |
| `block_idx / logical_block_num` | 当前 block 及总 block 数 |
| `cluster_index` | 此 group 的 cluster |
| `active_mask` | 真正有业务 subtask 的 lane |
| `resource_mask` | 必须参与协调的 lane，含 AIC coordinator |
| `lane_subtasks[3]` | 每个物理 lane 执行哪个 subtask；UINT8_MAX 表示无业务 subtask |
| `flags` | 任务属性 |
| `previous_lane_seq[3]` | 每个 lane 上一次参与的 group，防止跨过未处理分配 |

`SchedulerGroupLaneAck` 三个有效字段：
`assigned_seq` 是 coordinator 指派给该 lane 的 group；
`arrived_seq` 是该 lane 已腾出本地执行资源并到场的 group；
`released_seq` 是该 lane 已完成并释放的 group；每个 ACK 对象为 64B，三个字段之外没有可读的 reserved 字段。
检查 ACK 时，低成本 query 仅作预筛，真正判定还要原子确认。

`SchedulerGmMixLocalState` 的全部字段：

| 字段 | 作用 |
| ---- | ---- |
| `mailbox_offset` | 本 cluster mailbox 偏移 |
| `group_seq` | 当前/最近观察到的 group |
| `task_last_seq` | 本任务在此 cluster 最近 group |
| `vector_last_published_seq[2]` | coordinator 保存的两条 AIV 最近分配序号 |
| `block_idx` | 当前逻辑 block |
| `cluster_index` | cluster 身份 |
| `slot_index` | 当前 group 绑定的本核 own slot；与 task/block/subtask、phase 和 group_seq 一起校验 |
| `subtask_slot` | 本核应执行的逻辑子任务 |
| `phase` | IDLE、CLAIMED、PUBLISHED、ARRIVED、RUNNING、RELEASED |

`SchedulerMixCloseSeal` 独占 64B 行，其 `value` 是某任务在某 cluster 不再产生 group 的持久关闭凭据。它与 slot release 不同，用于允许其他成员安全推进 admission 游标。

## 3.4 kernel 返回怎样变成逻辑任务完成

读 [executor 的返回路径](../../../src/a5/runtime/host_build_graph/aicore/resident_scheduler_impl.h)：
dispatch 已验证 task/block/subtask/group 绑定；kernel 只接收 GM 参数和 payload context，不接收本核私有 slot/admission 地址，因此正常返回期间绑定保持不变。
executor 跨调用仅保留已验证的 async group index，从本轮固定布局重建 writer 并 seal 注册状态，完成启用的 DFX 与 AFTER_COMPLETION 抓取，然后进入 EXIT/本worker错误门禁。
门禁允许继续时，同一调用链直接调用 [scheduler_complete_own_slot](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_completion.h)：

1. 保留本次 EXECUTING 身份，释放本地 slot 的业务绑定。
2. 一次原子操作累计实际子任务完成数及 async-check 票数。
3. 最后一个贡献者若看到累计零票，可直接完成；否则关闭 async computation，必要时等待 FINISH。
4. 逻辑完成发布 DONE 并解析后继，可以直接保留一个有权推进的候选。
5. 整理和发布 trace，再释放 MIX lane；所有诊断写入在槽复用、成员关闭和 EXIT ACK 前完成。

重复退休 FREE/READY slot 会报绑定错误。失败沿首错与退出路径处理，不重试已经部分推进的 completion。
predicate 跳过业务调用仍走一次相同退休路径；INLINE/native 控制节点保留各自的直接完成入口。

## 3.5 完成传播：CSR 计数与 wake-list 共用 resolver

[同步 resolver](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_completion_resolver.h) 提供 `scheduler_resolve_csr_completion` 与 `scheduler_resolve_wake_completion` 两种算法。普通 kernel、native control、async FINISH 都由唯一完成者在当前调用中传播到结束；没有跨调用保存的 resolver phase 或 route cursor。

默认Host生成CSR，执行同步计数传播。未附依赖表的有效图执行同步wake-list：原子关闭producer的head，依次读取waiter的`wake_cursor`，把旧next与下一fanin位置保存在本次调用的局部值中。位置已到fanin末尾就发布Ready，否则使用`scheduler_route_detached_waiter`寻找尚未完成的前驱。挂链CAS成功即转移waiter所有权，原完成者只能沿之前保存的旧next前进。Ready前缀发布不会丢弃当前fanin view或要求重读原子游标。

CSR 算法只读取一次后继行范围，对每条去重边原子递减 remaining；旧值为1的唯一通知者使后继Ready。它不领取 wake-list。同核类型的非MIX后继可以直接保留；有ordered admission时，完成核也能持有MIX候选，但执行仍由AIC与各lane许可控制。其他kernel进入AIC（含MIX）/AIV两个共享队列，控制节点进入control inbox。

局部 `SchedulerReadyPublication` 保存最多8个同destination的ref、count及CONTROL目标。普通业务队列在一次调用内完成reserve和publish；control inbox先准备链，再循环尝试CAS接入。内部发布phase仅活在同步调用期间，不写入FINISH或LocalState供其他核接续。

native control每轮最多处理8个节点，每个节点完整解完依赖后才计入 `pending_completed`。async FINISH在join满足后取得唯一lease，完成TaskDone、全部后继传播后将lease置为TERMINAL；条件CHECK仍独立按片检查，未满足的异步条件不会阻塞业务。

完整resolve计时包含摘链、fanins扫描、重挂链和Ready发布；唯一owner在返回前发布起止计时。局部结构保持自然8B对齐，不能把GM结构的过度对齐搬入CCEC栈对象。

同步任务发布 DONE 后，由唯一完成者在解析返回时增加本地 `pending_completed`；解析报错也只计一次。在空闲、收尾，以及 native 的有界批次条件下，
[flush_completions](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_completion.h#L21)（第 21 行） 先屏障，再原子相加到 `run_control.resolved_task_count`。

这解释了为什么全局 resolved 计数可能暂时落后于实际 kernel 返回数：它统计的是逻辑节点，而且是批量发布。

## 3.6 图完成与双向退出

[AicpuExecutor::run](../../../src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp) 的 supervisor 使用：

```cpp
bootstrap_complete &&
resolved_task_count == expected_task_count - inline_completed_count
```

同时检查共享错误和无进展超时。不是等每个 kernel 的 COND 完成 token。

随后各线程 [广播 EXIT](../../../src/a5/runtime/host_build_graph/aicpu/aicore_lifecycle.cpp)。
AICore 观察 EXIT 后，刷新 pending 完成计数，确保已观察本轮 EXIT，再 [drain ordered obligations](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_dispatch.h)。
正常成功路径要求无未提交候选、无仍占用 slot，并让本核 admission 进度收敛。

退出握手数据为 [AicoreTeardownControl](../../../src/common/task_interface/aicore_teardown.h)：

| 字段 | 写者 | 作用 |
| ---- | ---- | ---- |
| `exit_request` | AICPU | 本次退休要求回显的保留 nonce |
| `post_close_release` | AICPU | 已确认 ACK 并完成寄存器窗口收尾，允许 AICore 返回 |

AICore 对这条独立 cache line 只做 bypass load，不写回它。

实际顺序见 [platform_retire_aicore_group](../../../src/a5/platform/shared/aicpu/platform_regs.cpp) 和
[acknowledge_aicore_cond_nonce_and_wait](../../../src/a5/platform/onboard/aicore/inner_kernel.h)：

1. AICPU 读取旧 COND，挑一个与旧值不同的 nonce，写 exit_request；发布后再次广播 EXIT。
2. AICore 完成 DDR 屏障，等合法 request，把该值写入 COND；之后不再写 SPR。
3. AICPU 轮转扫描所有核，只有 COND 等于本次 request 才认可 ACK。
4. 对已确认的核执行窗口收尾，写 DMB=IDLE 并读回，随后 drain。
5. AICPU 写 post_close_release=1。
6. AICore 等到 release 才从平台入口返回。
7. AICPU 清已退休核的 Handshake 报告标记，避免下次误认。

若初始化/执行失败，错误路径仍需要退出握手；“发现错误”并不自动证明所有核已经停止访问 GM。
