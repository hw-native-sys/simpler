# 3. MIX/SPMD、完成传播与退出

[总索引](README.md) · [上一段：依赖与执行](02-ready-execution.md) · [条件分支：native/async](04-native-async.md)

结构的作用范围、数量、存放位置、读写者和生命周期可对照 [数据结构总览](00-data-structure-scopes.md)；下面在关键结构的字段表前也说明其范围与用途。

## 3.1 什么任务需要有序准入

在 [scheduler_task_requires_admission](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_types.h#L921)（第 921 行） 看条件：
任务 executable、非 inline，并满足 logical_block_num>1、MIX、SYNC_START 之一。

若图中此类任务数为 0，`local.admission=nullptr`，走普通路径。
若非 0，`scheduler_initialize_admission` 挂上本地状态，这一轮图中的普通任务也受准入顺序约束。

[fill_ordered_slots](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_dispatch.h#L165)（第 165 行） 按下面的顺序推进：

1. 尝试发布自己上轮已认领的协作任务。
2. 在有限候选槽内从本类型/MIX ready 队列发现工作。
3. 普通任务先保留候选；协作任务独占认领，构造 record，再提交全局顺序日志。
4. 扫描本核尚未处理的 admission 日志。不是成员则跳过，是成员则建立 current obligation。
5. MIX 推进 GM mailbox；非 MIX SPMD 领取 block 并准备 own slot。
6. 本核当前协作义务可关闭后，推进 scan_cursor；普通候选再按许可准备。
7. 真正调用 kernel 前，再由 `scheduler_ordered_can_enter_slot` 检查一次。

“有序”指所有成员遵守同一准入顺序，不意味着整个图每个 kernel 都串行。

[SchedulerAdmissionControl](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_types.h#L996)（第 996 行）：

**范围与用途：** 存在协作任务时，本轮 GM 中一份共享控制对象。多个核竞争提交锁以发布同一条全局 admission 日志；每个核自己的阅读位置放在 AdmissionLocalState，不放在这里。
`commit_lock` 串行化日志提交，持锁者只做有界发布；
`commit_tail` 为已经完整提交的日志长度；
`lock_padding/tail_padding` 隔离两种访问区域。

[SchedulerAdmissionRecord](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_types.h#L978)（第 978 行）：

**范围与用途：** 每个协作任务一条有效 GM 记录，存储按 task_ref 寻址。任务的独占认领者准备并发布它，参与该任务的所有 worker 读取同一条记录；提交后不再改写。

| 字段 | 作用 |
| ---- | ---- |
| `run_epoch` | record 属于哪次运行 |
| `task_ref` | 协作任务 |
| `order_key` | 在全局 admission 日志中的位置 |
| `member_words_offset` | 参与此任务的 worker 位图 |
| `mix_close_seals_offset` | 此任务各 cluster 的 MIX 关闭凭据；非 MIX 不使用 |
| `logical_block_num` | 逻辑 block 总数 |
| `member_count` | 位图中的成员数量 |
| `active_mask` | 本任务有效 subtask |
| `flags` | 任务属性副本 |
| `reserved[14]` | 保留 |

提交次序见 [commit_validated_record](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_admission.h#L381)（第 381 行）：
`record.order_key` → 发布 record → `log[tail]=task` → `admission_state=ADMITTED` → `commit_tail=tail+1`。
读取 tail 的核必须能读到前面完整发布的记录。

[SchedulerAdmissionLocalState](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_ready.h#L144)（第 144 行）：

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

读 [scheduler_claim_ordered_cursor](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_admission.h#L653)（第 653 行）。

非 SYNC_START 的多 block 任务走 ticket：
原子增加 `claim_tickets` 得到唯一 ticket，ticket 小于 block_num 才拥有合法 block；
然后增加 `next_block` 确认有效领取。最后一个有效确认者发布 `claim_seal=block_num`。
因此 `claim_tickets` 可能越界少量，`next_block` 仍有严格有效范围。

另一分支用 CAS 推进 `next_block`，SYNC_START 成员成功一次后设置 `sync_claimed`，不继续包办多个 block。
这里的 SYNC_START 资源/领取约束不能直接理解为所有核在同一个时钟周期进入 kernel。

[claim_ordered_block](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_admission.h#L764)（第 764 行） 把 task、block、cooperative 标记放入本核 slot。
同一逻辑任务可以分多波执行，payload.block_num 始终是总 block 数。

每个 block 返回后，[finish_ordered_block](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_admission.h#L829)（第 829 行） 先释放本地 slot，再原子增加 `completed_blocks`。
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

[SchedulerGroupMailbox](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_types.h#L1028)（第 1028 行）：

**范围与用途：** 每个 cluster 一份 GM mailbox，多个 MIX task/block 按 group_seq 先后复用它。descriptor 由该 cluster 的 AIC 发布；每个 lane 的 arrived/released ACK 由该 lane 发布，assigned_seq 由 AIC 发布。不能把它理解成每个 MIX 任务各有一个 mailbox。
`descriptor` 保存下面的 group 描述；`published_seq` 是其发布标记；
`publication_padding` 隔离发布区；`lanes[3]` 是各成员 ACK。

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
| `reserved[48]` | 保留 |

`SchedulerGroupLaneAck` 三个有效字段：
`assigned_seq` 是 coordinator 指派给该 lane 的 group；
`arrived_seq` 是该 lane 已腾出本地执行资源并到场的 group；
`released_seq` 是该 lane 已完成并释放的 group；`reserved[104]` 为填充。
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

`SchedulerMixCloseSeal.value` 是某任务在某 cluster 不再产生 group 的持久关闭凭据。它与 slot release 不同，用于允许其他成员安全推进 admission 游标。

## 3.4 kernel 返回怎样变成逻辑任务完成

读 [executor 的返回路径](../../../src/a5/runtime/host_build_graph/aicore/aicore_executor.cpp#L555)（第 555 行）：
校验 generation、seal async，然后写
`local_completed_generations[slot]=ready_generation`。
这一步只登记本核执行返回。

下一轮 [scheduler_service_cluster_completions](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_completion.h#L398)（第 398 行）：

1. 对每个 own slot 检查完成 generation 非零且尚未消费。
2. 校验它匹配 slot.task_ref、slot.generation 和 READY 状态。
3. 保存执行记录，标记该 generation 已消费，释放/处理 slot。
4. 普通任务可以完成；SPMD 等最后 block；MIX 发布本 lane release，等待 AIC 汇总 group。
5. 如果有异步条件，可能只关闭 computation，而延迟逻辑 DONE。
6. 真正满足完成条件时写 `TaskControl.state=DONE`，并进入依赖完成解析。

generation 的作用不是给 kernel 编号，而是防止同一个 slot 复用后把旧完成再次消费。

## 3.5 完成传播：关闭 wake list，再逐个迁移 waiter

入口 [scheduler_resolve_completion_step](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_completion_resolver.h#L19)（第 19 行）。

```text
START   确认 producer DONE，exchange(wake_list_head, CLOSED) 一次性摘走链表
SELECT  取一个 waiter，先保存它的旧 next
ROUTE   继续扫描这个 waiter 的剩余依赖
APPEND  依赖已满足：保留为直接后继，或收集入 ready 发布批次
PUBLISH 发布批次；若本片预算不足，保存游标续行
DONE    已经把这条链上的所有后继交接完
```

必须先保存旧 next，因为 route 会把 waiter 迁移到另一 producer 的链表，并改写 next_waiter。
单依赖后继可以走直接就绪优化；有合适的本核后继时可直接 refill，避免入队再出队。
其他后继按类型发布 ready，随后被某个核领取。

[解析游标](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_resolve_state.h#L31)（第 31 行） 的所有字段：

**范围与用途：** 这些结构表示一个正在推进的解析过程。普通同步解析的 cursor 是调用内局部值；native 控制解析放在本核 LocalState 中跨循环保存；async FINISH 的 cursor 则保存到 GM，持有 lease 的服务核可接续。它们的归属取决于谁拥有当前解析责任，不能仅凭结构名判断都在本核或都在 GM。

| 结构 | 字段及作用 |
| ---- | ---------- |
| `SchedulerRouteCursor` | `task_ref`：正在路由的 waiter；`producer_ref`：待注册前驱；`next_fanin`：当前依赖位置；`fanin_count`：总依赖数；`result`：就绪、等待、已完成、错误或续行 |
| `SchedulerRouteProgress` | `result`：这次推进结果；`work_units`：消耗的预算单位 |
| `SchedulerReadyPublication` | `refs[8]`：待发布任务；`inbox_offset`：目标队列定位/准备状态；`count`：数量；`type`：AIC/AIV/MIX/control；`target`：目标队列；`phase`：收集/准备/目录/结束/失败阶段 |
| `SchedulerResolveCursor` | `producer_ref`：正在解析哪个完成任务；`next_waiter`：下一个链节点；`waiter_ref`：当前节点；`saved_next`：当前节点迁移前保存的 next；`route`：依赖检查子游标；`publication`：ready 发布子状态；`phase`：上述 START..DONE 状态；`visited_count`：已访问节点数，亦用于检测异常链 |
| `SchedulerResolveProgress` | `result`：DONE/YIELD/ERROR；`work_units`：本片工作量 |

解析完成后才增加本地 `pending_completed`。在空闲、收尾，以及 native 的有界批次条件下，
[flush_completions](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_completion.h#L21)（第 21 行） 先屏障，再原子相加到 `run_control.resolved_task_count`。

这解释了为什么全局 resolved 计数可能暂时落后于实际 kernel 返回数：它统计的是逻辑节点，而且是批量发布。

## 3.6 图完成与双向退出

[AicpuExecutor::run](../../../src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp#L528)（第 528 行） 的 supervisor 使用：

```cpp
bootstrap_complete &&
resolved_task_count == expected_task_count - inline_completed_count
```

同时检查共享错误和无进展超时。不是等每个 kernel 的 COND 完成 token。

随后各线程 [广播 EXIT](../../../src/a5/runtime/host_build_graph/aicpu/aicore_lifecycle.cpp#L426)（第 426 行）。
AICore 观察 EXIT 后，刷新 pending 完成计数，确保已观察本轮 EXIT，再 [drain ordered obligations](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_dispatch.h#L368)（第 368 行）。
正常成功路径要求无未提交候选、无仍占用 slot，并让本核 admission 进度收敛。

退出握手数据为 [AicoreTeardownControl](../../../src/common/task_interface/aicore_teardown.h#L25)（第 25 行）：

| 字段 | 写者 | 作用 |
| ---- | ---- | ---- |
| `exit_request` | AICPU | 本次退休要求回显的保留 nonce |
| `post_close_release` | AICPU | 已确认 ACK 并完成寄存器窗口收尾，允许 AICore 返回 |

AICore 对这条独立 cache line 只做 bypass load，不写回它。

实际顺序见 [platform_retire_aicore_group](../../../src/a5/platform/shared/aicpu/platform_regs.cpp#L56)（第 56 行） 和
[acknowledge_aicore_cond_nonce_and_wait](../../../src/a5/platform/onboard/aicore/inner_kernel.h#L151)（第 151 行）：

1. AICPU 读取旧 COND，挑一个与旧值不同的 nonce，写 exit_request；发布后再次广播 EXIT。
2. AICore 完成 DDR 屏障，等合法 request，把该值写入 COND；之后不再写 SPR。
3. AICPU 轮转扫描所有核，只有 COND 等于本次 request 才认可 ACK。
4. 对已确认的核执行窗口收尾，写 DMB=IDLE 并读回，随后 drain。
5. AICPU 写 post_close_release=1。
6. AICore 等到 release 才从平台入口返回。
7. AICPU 清已退休核的 Handshake 报告标记，避免下次误认。

若初始化/执行失败，错误路径仍需要退出握手；“发现错误”并不自动证明所有核已经停止访问 GM。
