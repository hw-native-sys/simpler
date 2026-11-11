# 3. MIX/SPMD、完成传播与退出

[总索引](README.md) · [上一段：依赖与执行](02-ready-execution.md) · [条件分支：native/async](04-native-async.md)

结构的作用范围、数量、存放位置、读写者和生命周期可对照 [数据结构总览](00-data-structure-scopes.md)；下面在关键结构的字段表前也说明其范围与用途。

## 3.1 什么任务需要有序准入

在 [scheduler_task_requires_admission](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_types.h#L916)（第 916 行） 看条件：
任务 executable、非 inline，并满足 logical_block_num>1、MIX、SYNC_START 之一。

若图中此类任务数为 0，`local.admission=nullptr`，走普通路径。
若非 0，`scheduler_initialize_admission` 挂上本地状态，这一轮图中的普通任务也受准入顺序约束。

[fill_ordered_slots](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_dispatch.h#L170)（第 170 行） 按下面的顺序推进：

1. 尝试发布自己上轮已认领的协作任务。
2. 在有限候选槽内从本类型共享 ready 队列发现工作；AIC 队列同时包含纯 AIC 和 MIX。
3. 普通任务先保留候选；协作任务独占认领，构造 record，再提交全局顺序日志。
4. 扫描本核尚未处理的 admission 日志。不是成员则跳过，是成员则建立 current obligation。
5. MIX 推进 GM mailbox；非 MIX SPMD 领取 block 并准备 own slot。
6. 本核当前协作义务可关闭后，推进 scan_cursor；普通候选再按许可准备。
7. 真正调用 kernel 前，再由 `scheduler_ordered_can_enter_slot` 检查一次。

“有序”指所有成员遵守同一准入顺序，不意味着整个图每个 kernel 都串行。

成员集合由 [scheduler_admission_expected_members](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_admission.h#L207) 固定：普通非同步 SPMD 若 block 数少于同类型可用核数，
选取从 `member_rank_offset % eligible_count` 开始的循环区间，成员数为 block_num；否则使用完整可用集合。
Host 按 task_ref 顺序分别累计 AIC/AIV 的普通非同步 SPMD block 数，保存本任务之前的累计值。
不同资源类型、普通单 block、MIX 和 sync_start 不消耗该类型的轮转位置；变宽任务按实际 block 数推进。
MIX 和 sync_start 继续采用其完整资源成员集合。成员位图使用 compact worker index，不能套用物理核编号或交错编号假设。
一般拓扑仍经过合法性检查，COMPACT_LAUNCH 只是已验证拓扑的快速构建路径。

[SchedulerAdmissionControl](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_types.h#L979)（第 979 行）：

**范围与用途：** 存在协作任务时，本轮 GM 中一份共享控制对象。多个核竞争提交锁以发布同一条全局 admission 日志；每个核自己的阅读位置放在 AdmissionLocalState，不放在这里。
`commit_lock` 串行化日志提交，持锁者只做有界发布；
`commit_tail` 为已经完整提交的日志长度；
两者各占独立 64B 行，控制对象总长 128B。

[SchedulerAdmissionRecord](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_types.h#L960)（第 960 行）：

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

提交次序见 [commit_validated_record](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_admission.h#L410)（第 410 行）：
`record.order_key` → 发布 record → `log[tail]=task` → `admission_state=ADMITTED` → `commit_tail=tail+1`。
读取 tail 的核必须能读到前面完整发布的记录。

[SchedulerAdmissionLocalState](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_ready.h#L121)（第 121 行）：

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

读 [scheduler_claim_ordered_cursor](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_admission.h#L734)（第 734 行）。

非 SYNC_START 的多 block 任务走 ticket：
原子增加 `claim_tickets` 得到唯一 ticket，ticket 小于 block_num 才拥有合法 block；
然后增加 `next_block` 确认有效领取。最后一个有效确认者发布 `claim_seal=block_num`。
因此 `claim_tickets` 可能越界少量，`next_block` 仍有严格有效范围。

另一分支用 CAS 推进 `next_block`，SYNC_START 成员成功一次后设置 `sync_claimed`，不继续包办多个 block。
这里的 SYNC_START 资源/领取约束不能直接理解为所有核在同一个时钟周期进入 kernel。

[claim_ordered_block](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_admission.h#L845)（第 845 行） 把 task、block、cooperative 标记放入本核 slot。
同一逻辑任务可以分多波执行，payload.block_num 始终是总 block 数。

每个 block 返回后，[finish_ordered_block](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_admission.h#L910)（第 910 行） 先释放本地 slot，再原子增加 `completed_blocks`。
只有使其达到 logical_block_num 的最后一个 block，才能进入整个任务的完成处理。

## 3.3 MIX：一个 cluster 内各核自己准备和执行

主链入口是 [scheduler_service_gm_mix](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_dispatch.h#L22)（第 22 行），实现位于 [scheduler_gm_mix.h](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_gm_mix.h#L160)（第 160 行）。

按一个逻辑 block 走一遍：

1. cluster 的 AIC coordinator 领取 block，生成新的 `group_seq`。
2. AIC 填 GM descriptor：task、block、参与 lane 和各 lane 的前序 group。
3. 写回 descriptor 后，发布 `published_seq`，再为参与 AIV 更新 `assigned_seq`。
4. 各参与核观察分配、校验 epoch/order/group；等本核 own slots 都 free 后写 `arrived_seq`。
5. 确认本 group 的所有资源成员都 arrived，才各自准备自己的 own slot。
6. 每个核调用自己的 subtask kernel；没有业务 subtask 的资源成员直接释放。
7. 本核完成消费和 trace 收尾后写 `released_seq`。
8. AIC 确认所有资源成员 release，给 completed_blocks 贡献一个逻辑 block。
9. 全部 block 完成后，整个任务才进入 DONE/依赖传播。

即使是纯双 AIV MIX，AIC 仍作为 coordinator 进入 resource_mask；它可能没有业务 kernel。
当前主线 descriptor 的逻辑 subtask 与物理通信 lane 保持对应。

ready 领取、组协调与 slot 管理是三个不同职责：AIC 从共享队列认领 MIX task 并发布 admission；各 cluster 的 AIC 领取该 task 的 block、协调 mailbox；每个执行成员通过 [scheduler_gm_mix_prepare_lane](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_gm_mix.h#L374) 选择并填写自己的 `local->slots[index]`。
每个 worker 只有两个 own slot，普通任务与 MIX 共用 generation、payload 和本地完成消费流程。coordinator 汇总成员 release，不替其他成员填写或回收 slot。

[SchedulerGroupMailbox](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_types.h#L1007)（第 1007 行）：

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
| `slot_generation / slot_index` | 当前 group 绑定的本核 own slot 及发布代号 |
| `subtask_slot` | 本核应执行的逻辑子任务 |
| `phase` | IDLE、CLAIMED、PUBLISHED、ARRIVED、RUNNING、RELEASED |

`SchedulerMixCloseSeal` 独占 64B 行，其 `value` 是某任务在某 cluster 不再产生 group 的持久关闭凭据。它与 slot release 不同，用于允许其他成员安全推进 admission 游标。

## 3.4 kernel 返回怎样变成逻辑任务完成

读 [executor 的返回路径](../../../src/a5/runtime/host_build_graph/aicore/resident_scheduler_impl.h#L628)（第 628 行）：
校验 generation、seal async，然后写
`local_completed_generations[slot]=ready_generation`。
这一步只登记本核执行返回。

下一轮 [scheduler_service_cluster_completions](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_completion.h#L345)（第 345 行）：

1. 对每个 own slot 检查完成 generation 非零且尚未消费。
2. 校验它匹配 slot.task_ref、slot.generation 和 READY 状态。
3. 保存执行记录，标记该 generation 已消费，释放/处理 slot。
4. 普通任务可以完成；SPMD 等最后 block；MIX 发布本 lane release，等待 AIC 汇总 group。
5. 如果有异步条件，可能只关闭 computation，而延迟逻辑 DONE。
6. 真正满足完成条件时写 `TaskControl.state=DONE`，并进入依赖完成解析。

generation 的作用不是给 kernel 编号，而是防止同一个 slot 复用后把旧完成再次消费。

## 3.5 完成传播：CSR 计数与 wake-list 共用 resolver

入口 [scheduler_resolve_completion_step](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_completion_resolver.h#L28) 是 `static noinline` 函数，普通完成、native 控制节点与 async FINISH 都使用它。
外联不改变每片预算和唯一 owner 的规则。

```text
START   确认 producer DONE，关闭 wake_list_head；计数路径要求没有动态 waiter
SELECT  有依赖 header：从 producer 的 CSR 行取后继，原子递减其 remaining
        旧值大于 1 → 继续下一后继；旧值等于 1 → 进入 APPEND
        无 header：取已摘下链的 waiter，并保存其旧 next
ROUTE   仅 wake-list 路径：推进剩余 fanin，必要时把 waiter 迁移到下一 producer
APPEND  保留可直接执行的后继，或收集入按类型分组的 ready 发布批次
PUBLISH 完成批次发布；预算不足则保存全部游标状态
DONE    该 producer 的全部后继已经处理或责任已交接
```

wake-list 路径必须先保存旧 next，因为 route 可能改写 waiter 的 next_waiter。
计数路径按 immutable CSR 前进，visited_count 同时定位当前行中的边，不重复通知已经消费的边。
计数从 1→0 的后继同样可以 direct refill：调用方提供直接交接位置、后继为同核类型的非 MIX kernel 时，resolver 可保留一个 ready claim；实际准备和执行仍受普通/有序派发入口的约束。其他 kernel 按 AIC（含 MIX）/AIV 发布到两个共享队列，控制节点进入 control inbox。

[解析游标](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_resolve_state.h#L31)（第 31 行） 的所有字段：

**范围与用途：** 完整 SchedulerResolveCursor 为 160B、自然 8B 对齐；这些结构表示一个正在推进的解析过程。普通同步解析的 cursor 是调用内局部值；native 控制解析放在本核 LocalState 中跨循环保存；async FINISH 的 cursor 则保存到 GM，持有 lease 的服务核可接续。它们的归属取决于谁拥有当前解析责任，不能仅凭结构名判断都在本核或都在 GM。

| 结构 | 字段及作用 |
| ---- | ---------- |
| `SchedulerRouteCursor` | `task_ref`：正在路由的 waiter；`producer_ref`：待注册前驱；`next_fanin`：当前依赖位置；`fanin_count`：总依赖数；`result`：就绪、等待、已完成、错误或续行 |
| `SchedulerRouteProgress` | `result`：这次推进结果；`work_units`：消耗的预算单位 |
| `SchedulerReadyPublication` | `refs[8]`：待发布任务；`count`：数量；`destination`：AIC/AIV/CONTROL；`inbox_offset/target`：仅 CONTROL 使用，分别定位 inbox 和 worker，业务队列时均为 0；`phase`：COLLECTING/PREPARED/DONE/FAILED |
| `SchedulerResolveCursor` | `producer_ref`：正在解析哪个完成任务；`next_waiter`：下一个 CSR 后继或链节点；`waiter_ref`：当前节点；`saved_next`：下一个 CSR 后继或迁移前保存的 next；`route`：依赖检查子游标；`publication`：ready 发布子状态；`phase`：上述 START..DONE 状态；`visited_count`：已消费的 CSR 边/链节点数，亦用于边界检查 |
| `SchedulerResolveProgress` | `result`：DONE/YIELD/ERROR；`work_units`：本片工作量 |

一个发布批次最多容纳 8 个同 destination 的 ref；AIC 与 MIX 可以共用一个批次。业务队列在一次有界调用内完成预留和原子发布；control inbox 则先准备链，再 CAS 追加，争用时保留 PREPARED 状态续行。续行保存整个 publication，不能只保存下一个后继索引。

解析完成后才增加本地 `pending_completed`。在空闲、收尾，以及 native 的有界批次条件下，
[flush_completions](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_completion.h#L21)（第 21 行） 先屏障，再原子相加到 `run_control.resolved_task_count`。

这解释了为什么全局 resolved 计数可能暂时落后于实际 kernel 返回数：它统计的是逻辑节点，而且是批量发布。

## 3.6 图完成与双向退出

[AicpuExecutor::run](../../../src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp#L644)（第 644 行） 的 supervisor 使用：

```cpp
bootstrap_complete &&
resolved_task_count == expected_task_count - inline_completed_count
```

同时检查共享错误和无进展超时。不是等每个 kernel 的 COND 完成 token。

随后各线程 [广播 EXIT](../../../src/a5/runtime/host_build_graph/aicpu/aicore_lifecycle.cpp#L424)（第 424 行）。
AICore 观察 EXIT 后，刷新 pending 完成计数，确保已观察本轮 EXIT，再 [drain ordered obligations](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_dispatch.h#L359)（第 359 行）。
正常成功路径要求无未提交候选、无仍占用 slot，并让本核 admission 进度收敛。

退出握手数据为 [AicoreTeardownControl](../../../src/common/task_interface/aicore_teardown.h#L25)（第 25 行）：

| 字段 | 写者 | 作用 |
| ---- | ---- | ---- |
| `exit_request` | AICPU | 本次退休要求回显的保留 nonce |
| `post_close_release` | AICPU | 已确认 ACK 并完成寄存器窗口收尾，允许 AICore 返回 |

AICore 对这条独立 cache line 只做 bypass load，不写回它。

实际顺序见 [platform_retire_aicore_group](../../../src/a5/platform/shared/aicpu/platform_regs.cpp#L60)（第 60 行） 和
[acknowledge_aicore_cond_nonce_and_wait](../../../src/a5/platform/onboard/aicore/inner_kernel.h#L151)（第 151 行）：

1. AICPU 读取旧 COND，挑一个与旧值不同的 nonce，写 exit_request；发布后再次广播 EXIT。
2. AICore 完成 DDR 屏障，等合法 request，把该值写入 COND；之后不再写 SPR。
3. AICPU 轮转扫描所有核，只有 COND 等于本次 request 才认可 ACK。
4. 对已确认的核执行窗口收尾，写 DMB=IDLE 并读回，随后 drain。
5. AICPU 写 post_close_release=1。
6. AICore 等到 release 才从平台入口返回。
7. AICPU 清已退休核的 Handshake 报告标记，避免下次误认。

若初始化/执行失败，错误路径仍需要退出握手；“发现错误”并不自动证明所有核已经停止访问 GM。
