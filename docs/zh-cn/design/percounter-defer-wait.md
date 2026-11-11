# percounter COUNTER defer_wait 设计

本文定义 a5 `percounter` runtime 的 deferred completion 第一期：业务 kernel 登记
COUNTER 条件后正常返回，lane 立刻去跑别的工作，任务的计数器要等条件满足才发布。

本文规定待实现的行为与接口，不表示功能已交付。基础是现有的 per-lane 自调度、MIX
rendezvous 与 sync-start cohort 协议；A5 HBG 的[异步完成设计](a5-hbg-async-completion.md)
定义了同一套用户接口与条件语义，本文只记录 percounter 与之不同的部分。

## 1. 范围

| 范围 | 第一期 |
| ---- | ------ |
| 条件类型 | COUNTER（int32 单调通知）；一个 group 可含多条 |
| 检查者 | 登记该 group 的 lane 自己（owner-only） |
| 任务形状 | 普通 AIC/AIV、SPMD、MIX 掩码 3/5/6/7、sync-start cohort、Graph body |
| 前端 | 既有 `save_expected_notification_counter` / `register_completion_condition` |
| 错误 | 登记失败、条件非法、溢出、超时、lane 故障 |

**不做**：SDMA post-done 条件与其提交互斥字、跨 lane 的共享检查、kernel 抢占与恢复、
AICPU 侧条件轮询。第 7 节记录这些需要什么才能加上。

### 1.1 现状缺陷

percounter 的 aicore include 路径含 `{HBG}/runtime`，`async_kernel_api.h` 现在就能编译；
而 `host/state.h` 只预填 `local_context.async_ctx.task_token = TaskId::invalid()`，
`global_context.async_*` 全为零，`build_tables` 的 `memset` 覆盖整个 payload 数组，所以
`completion_count` 确实是空指针。于是 `async_ctx_is_deferred()` 为假，而**后果取决于
kernel 怎么写**：

| kernel | 今天在 percounter 上的行为 |
| ------ | -------------------------- |
| PyPTO 生成的 waiter（自带检查，见 `tests/fdwic/.../combine_wait.cpp`） | slab 指针为空 ⇒ `trap()`，核故障，不说明原因 |
| 直接用 runtime 自带的 `save_expected_notification_counter()` | 返回值被 `(void)` 丢弃 ⇒ **任务照常发布，消费者在数据落地前执行** |

两种都不可接受，且都不指向 percounter 或 defer_wait。**运行时侧无法单独修好第二种**：
登记失败时 kernel 什么也没写，设备上不留痕迹。所以"先堵住静默失败"不能作为独立一步，
它与第 9 节的 S1 合并 —— 一旦每个 lane 有了真正的 group 记录，登记就会机械地成功并留下
`count > 0`，percounter 在 kernel 返回后读到它，就能指名道姓地报错。

## 2. 为什么 percounter 的延迟完成比 HBG 小一个量级

percounter 发布完成就是两条原子加（`percounter_publish_completion`）：

```cpp
scheduler_gm_fetch_add(cfg.counters[task.id].value, count);
if (task.sink) scheduler_gm_fetch_add(cfg.counters[task.shell].value, count);
```

消费者 `percounter_fanin_ready` 读的就是这个字。计数器本身已经是一个 N 路 join —— 每个
block / subtask 加自己那份，消费者判的是"达到预置总数"。**所以每个 deferred group 各自
独立结算，不需要 join 对象、pending 计数或 FINISH owner 选举。**

其次，percounter 已经把两件事分在不同的字上：

| 语义 | 载体 | 延迟完成是否改变 |
| ---- | ---- | ---------------- |
| 这个 block 跑完了（进度、退出、看门狗） | `completed_block_count`、`sync_queue->completed`、`mix_tickets[DONE]` | **不变** |
| 这个 task 的消费者可以跑了 | `counters[id].value` | **推迟** |

cohort block 尤其如此：`sync_queue->completed` 必须仍在 kernel 返回处加，否则 cohort
协议停滞，全设备等待一个已经执行完的 block。

因此 HBG 的异步调度层在此处的对应物是：

| HBG | percounter |
| --- | ---------- |
| `async_completion_types.h`、`async_condition_writer.h`、`SchedulerAsyncCompletionObserver` | 原样复用，不复制 |
| `SchedulerAsyncLease` 与 CAS 租约 | 不需要：一个 group 只有 owner 一个检查者 |
| `check_directory` / `finish_directory` 位图、`activity` | 不需要：owner 知道自己的 index |
| `SchedulerAsyncTaskJoin`、`close_computation`、FINISH owner 选举 | 不需要：计数器就是 join |
| `finish_leases`、`finish_progress`、`scheduler_async_copy_resolver` | 不需要：没有 ready queue 要路由 |
| `task_map` / `task_data` | 不需要：group index 即 lane |

## 3. 数据布局

### 3.1 group 按 lane 索引

与 `DispatchPayload` 完全一致：一个 lane 同时只跑一个 kernel，也只拥有一个 payload。

```cpp
// host/region_plan.h，排在 counters 之后，与 mix / sync 的按需保留不同形
!PERCOUNTER_RESERVE(SCHEDULER_WORKER_CAPACITY, pto2::resident::AsyncGroupData, async_groups_offset) ||
!PERCOUNTER_RESERVE(
    SCHEDULER_WORKER_CAPACITY * PERCOUNTER_ASYNC_CAPACITY, DeferredCompletionEntry, async_entries_offset
)
```

**无条件保留，不按图开关。** 一个 kernel 会不会登记条件，bind 时看不出来 —— host 只知道
kernel id，不知道函数体里有没有 `defer_wait`。记录正是"看出来"的手段，所以它必须对每张图
都在。

**排在 counters 之后**，于是整块落在 `upload_size` 之外：bind 不为它付任何 DMA，由每个
lane 在进入自己的循环前写一次自己那条记录（`percounter_open_async_record`）。那次写必须
带写回 —— 记录所在的存储是 pipeline slot 跨 run 复用的、`memset` 不覆盖的区域，而且一个
在入口做整 cache 失效的 kernel 否则会读到上一次 bind 留下的字节。

三条布局约束，由 `pto2::resident::detail::layout_valid` 强制：

- entries 数组排在 groups 数组之后（`entries_offset >= group_offset + sizeof(AsyncGroupData)`）。
- region 基址 128 对齐。`SCHEDULER_STATE_ALIGNMENT == 128 == ASYNC_RECORD_ALIGNMENT`，已成立。
- 每条记录的 entry 跨距必须已经是 128 的倍数，否则按 `lane * 跨距` 寻址会落在非对齐处。
  `PERCOUNTER_ASYNC_CAPACITY = 4` 时 `4 × 32 B = 128 B`，一条 static_assert 顶着。

容量取 4 而**不取 `MAX_COMPLETIONS_PER_TASK = 64`**：记录是按 lane 而非按 task 的，所以这
个上限要乘 108 付一遍，无论有没有 kernel 登记。4 时每 lane 是 128 B 头部 + 128 B entries，
全设备 27 KiB；64 时是 235 KiB，比旁边的 payload 数组还大四倍。

**每 lane 不止一条记录（`PERCOUNTER_ASYNC_SLOTS = 2`），否则这个特性什么也不做。** 一条
记录从开启它的那次 dispatch 一直活到条件满足，而 lane 在这期间照常跑别的任务 —— 那个间
隔正是全部意义所在。那些任务需要自己的记录：拿到一条已封口记录的 kernel 根本登记不了，
而运行时读回一条已结算但未释放的记录，会把一个什么都没登记的任务当成登记了而拒绝。换句
话说，1 深不是"连续两个登记任务要排队"，是**任何下一个任务都撞车**。

槽位满了的 lane 在下一次 dispatch 前等一个结算，这是槽数唯一被感觉到的地方；那个等待是
checkpoint 自旋，不是阻塞 —— lane 在那儿照样服务 MIX 和 cohort、照样看得见 EXIT 和别人
的错误。2 是 COUNTER 用例需要的；加大只多一条记录加它的 entries，不改协议。

### 3.2 寻址不额外碰任何 cache line

`async_image_address` / `async_image_bytes` / `async_run_epoch` / `async_group_offset` /
`async_group_index` / `async_present` 都在 `GlobalContext` 里，而 `GlobalContext` 落在
`DispatchPayload` 的第 7 条 cache line —— 也就是 `sub_block_id` 所在那条，
`percounter_launch_kernel` 每次 dispatch 本就在写、本就在
`scheduler_writeback_dispatch_payload` 里刷。**带 slot 维度、每次 dispatch 重写 group 地址，
不增加任何脏行，也不增加任何 dcci。**

### 3.3 `PercounterControl` 的新字段

只读字段放进现有 `reserved` 区（48 B 可用，需 32 B）：

| 字段 | 含义 |
| ---- | ---- |
| `async_run_epoch` | 非零；`AsyncWriter` 用它拒绝上一次 run 遗留的记录 |
| `async_groups_offset` / `async_entries_offset` | 区域偏移，零表示本图无异步任务 |
| `async_capacity` | 每个 group 的条件上限 |

两个跨核原子字**单独占一条 64 B 线**，不与 `completed_block_count` 共线，也不与只读字段
共线 —— 与该字段注释记录的理由相同（`lane_error` 与它共线时 paged_attention Case1 每任务
从 94 周期涨到 919）：

| 字段 | 写 | 读 |
| ---- | -- | -- |
| `async_outstanding` | seal 时 +1，settle 时 −1 | `percounter_lane_may_exit` 的完成判据 |
| `async_settled` | settle 时 +1，单调 | `percounter_lane_may_exit` 的进展量 |

两者分开的理由见第 6 节，这不是冗余。

## 4. 协议

### 4.1 登记

`percounter_launch_kernel` 增加出参 `bool *deferred`。

**kernel 之前**：该任务带"可能登记"标记时，重置 group 头部（`run_epoch`、
`task_ref = task.id`、`group_index`、`block_index`、`subtask_slot`、`capacity`、
`entries_offset`、`count = 0`、`error_code = 0`、`state = OPEN`、
`sdma_guards_offset = 0`、`sdma_guard_count = 0`）并写回，置 `async_present = 1`；
否则置 `async_present = 0` 且**完全不碰 group**。

标记来自任务描述符中的一个位，lane 本就要读该描述符。因此不登记的任务只多一次
`async_present = 0` 的存储，落在已脏的第 7 行上。

**kernel 返回后**：调用 `seal_conditions()`。

- `count == 0` → `*deferred = false`，走原同步路径。**这条必须保留**：waiter kernel 通常
  按条件登记，零登记是常态而非异常。
- `count > 0` → 由 `scheduler_async_condition_mask(count)` 算出 `pending_mask`，发布 group，
  `async_outstanding += 1`，`*deferred = true`。
- `state == FAILED` 或 `error_code != 0` → `percounter_fail()`，不发布也不延迟。

### 4.2 检查

插入点只有一个 —— `percounter_checkpoint_serve`：

```cpp
if (!percounter_async_tick(cfg, state, served)) return false;   // 新增，且排在最前
if (*served) return true;
if (!percounter_sync_serve_released(cfg, state, timeout_cycles, served)) return false;
if (*served) return true;
return percounter_mix_discover(cfg, state, timeout_cycles, served);
```

排在最前且无条件执行：这一项是本 lane 欠下的、别人替不了的债，而后两项是可以由别的 lane
服务的工作。一个欠着完成的 lane 必须在服务别人之前先看自己那笔。不欠时它只是一次本地
整数判断（`async_outstanding_hint`），不走记录也不读 entries。

**这一行是死锁正确性的要求，不只是延迟优化。** `percounter_wait_for_fanin` 与
`percounter_wait_for_entry` 的自旋体每轮都调 `percounter_checkpoint_serve`。一个 lane 延迟
了任务 X，而它 range 里的下一个任务以 X 为 fanin 时，它会阻塞在只有自己能发布的计数器上；
tick 在 checkpoint 里，这个自环自动解开，三个阻塞点全部继承这一性质。

`percounter_async_tick` 的内容：group 头部 `observe`，按 `pending_mask` 逐条取
`DeferredCompletionEntry`，用 `SchedulerAsyncCompletionObserver` 判定，每次调用有条数预算；
`pending_mask` 清空则结算。COUNTER 的观察是一次对齐的 32-bit device load 加一次有符号比较，
负值（含 `INT32_MIN`）为 PENDING，不清零通知。

### 4.3 结算

结算就是 owner lane 代替当初的发布点做那次 `percounter_publish_completion`，随后
`async_outstanding -= 1`、`async_settled += 1`。没有其他任何动作。

### 4.4 三个发布点

| 位置 | 改动 |
| ---- | ---- |
| `percounter_lane_loop` | `deferred` 为真时跳过 `percounter_publish_completion`；`++state.local_done` 保留 |
| `percounter_sync_serve_block` | 同上；`sync_queue->completed` 的加、`cohort_hold_set(0)` 保留在原处 |
| `percounter_serve_mix` | 见 4.5 |

`local_done`、`sync_queue->completed`、`mix_tickets[DONE]` 三者**全部不动**：它们计的是
"block 执行完了"，而 block 确实执行完了。

cohort 这条路上，`sync_queue->completed` 原本遵守一条顺序规则 —— 它排在任务计数器之后，
因为看到它达到总数的 lane 可以退出，而那时不能还有消费者在等。延迟之后任务计数器不在那里
加了，这条规则改由**退出判据**保证：`async_outstanding` 非零时 lane 不退出，而一个被延迟
的 cohort block 会一直让它非零。

**槽位等待发生在已经承诺执行之后。** `percounter_async_acquire` 在
`percounter_launch_kernel` 内部，所以在那里等的 lane 不服务 MIX 和 cohort，服务 cohort
block 的还握着那个 block。这是停顿不是挂死：等待期间它轮询自己的条件、检查停止信号、到
看门狗为止，而 MIX 的 arming 读到它的 hold 会正确地不点它的名。

### 4.5 MIX：延迟的 subtask 先把自己那份减掉

MIX 由掩码最低位的 lane 在 `percounter_mix_wait(finished)` 之后一次性加
`popcount(mask)`。若某个 subtask 延迟，这次加不能带它那份 —— 而发布者在加之前无从知道
哪几份该去掉。

**`PercounterMixSignal` 恰好是满的 64 B，没有位置放一个 deferred 掩码**，所以不走"告诉
发布者"这条路：延迟的 lane 在 `scheduler_ssbuf_store_release(&signal->finished[lane],
generation)` **之前**自己做一次 `fetch_add(counter, -1)`（sink 任务两个计数器都减），结算
时再加回来。发布者照旧加 `popcount(mask)`，不需要知道任何事。

这不是绕路，而是该计数器本来的用法：host 把它预置成 `1 - total_subtasks`，
`percounter_counter_published` 判的是 `(int64_t)value > 0`，负的中间值是这套机制的常态。

顺序由既有的 release/acquire 对保证，不需要新的 fence：发布者只在 acquire 读到
`finished[lane] >= generation` 之后才加，所以减一定排在加之前；计数器在中间只会偏小，
消费者只会多等，不会早走。

代价是延迟的 MIX subtask 多一次原子（sink 两次）。不延迟的 subtask、不含异步任务的图、
以及所有非 MIX 路径都是零。替代方案是改成每 lane 各加 1，热路径上每个 MIX block 从
1 次原子变 3 次，不取。

### 4.6 与 cohort 协议的关系

延迟完成不改变 cohort 协议的任何一处。队列保持一个 cohort outstanding，
`cohort_hold[lane]` 只被读作"该 lane 是否欠一个 block"；`defer_wait` 不阻塞，kernel 正常
返回，lane 在 `percounter_sync_serve_block` 里清除 hold 的时机不变。

这成立的前提是 **COUNTER 条件不往 kernel 内部引入任何新的阻塞点**。一个 lane 在 kernel
里到不了任何 checkpoint，而 cohort 与 MIX 的 gathering 都可能在等它 —— 这是该协议最敏感
的性质。SDMA 会引入这样的阻塞点，这是第一期不做它的第三个理由（见 7.1）。

## 5. 不变量

1. 一个 group 只有一个写者（其 owner lane）和一个检查者（同一个 lane）。
2. group 身份由 `run_epoch`、`group_index` 和头部的 `task_ref` / `block_index` /
   `subtask_slot` 确定，不依赖可复用的 slot。
3. 每个非零 group 恰好结算一次；零登记任务仍须 seal，并走同步发布路径。
4. 每个逻辑 block 恰好贡献一次执行进度（`local_done` 或 cohort / MIX 的 done 计数），
   与它是否延迟完成无关。
5. 每个 block / subtask 对计数器的净贡献恰好为 1：lane 自己发布时要么在 kernel 返回处加、
   要么在结算处加；由他人代发布时（MIX）先减后加，净值不变。
6. 持有未结算 group 的 lane 不退出。
7. 延迟完成不向 percounter 的 list-order 无环性论证增加任何边：它只延迟沿既有边的发布。

## 6. 退出、看门狗与错误

`percounter_lane_may_exit` 的两处改动，**必须是两处而不是一处**：

- `complete` 的合取加上 `scheduler_gm_query(cfg.control->async_outstanding) == 0`。
- `*progress` 的和加上 `async_settled`。

`async_outstanding` 有升有降，而 `progress` 的整个看门狗论证建立在被加数单调上。把它折进
`progress`，一个 outstanding 震荡的设备会不断刷新看门狗，挂死最后由 STARS 的 45 秒
`HandleTaskTimeout` 杀 `aicpu-sd` 收场 —— 按 `running-onboard.md` 的分诊表那是另一类故障。
分开之后，真死锁表现为 settled 停止前进、outstanding 非零、progress 平坦，
由 `scheduler_watchdog_expired` 报 `SchedulerGraphResult::TIMEOUT`。

**跨卡死锁。** rank0 等 rank1、rank1 等 rank0 的依赖环是用户程序的错误，运行时不可能阻止，
与 MPI 中写错收发顺序同类。但要区分它与**资源死锁**：阻塞式等待下，rank0 的所有核都卡在
kernel 里等通知，而"给 rank1 发通知"的任务没有空闲核去跑 —— 两边依赖图都无环却照样挂死。
percounter 比 HBG 更暴露于此，因为 lane 就是核，没有 dispatch slot 的间接层。消除这一类
正是本特性的目的。

**超时不等于可以回收通信资源。** lane 报错退出时若 `async_outstanding != 0`，对端引擎可能
仍在读本卡内存。核的 ACK 不是外部访问结束的证明，run 不得据此回收通信域或 workspace。
percounter 目前没有任何通信资源，但这条必须与第一个跨卡条件类型在同一改动中落地。

## 7. 后续

### 7.1 SDMA

需要三样 COUNTER 不需要的东西：

- **预留协议**。一次 TGET 的完成记录条数要提交后问 SDK 才知道，所以要提交前按上界预留、
  提交后登记实际条数、归还未用预留。`async_condition_writer.h` 里的
  `reserve_conditions` / `commit_condition` / `finish_reservation` 已经是这个，免费。
  没有它，半途失败的登记会表现为零登记成功，即 DMA 在飞而任务已发布。
- **每 cluster 一个提交互斥字**。SDK 按 `get_block_idx()` 选 channel group，同 cluster
  两个 AIV 共享该值；交权要整 DCache 写回加设备屏障，否则旧 owner 的后续 session 缓存维护
  会覆盖新 owner 已推进的 SQ 状态。`AsyncGroupData` 的 `sdma_guards_offset` /
  `sdma_guard_count` 已为此预留，第一期置零。
- **提交争用的 5 秒有界自旋是一次 kernel 内阻塞**。同 cluster 的 peer lane 若正在一次 MIX
  或 cohort 的 gathering 中等待持有互斥字的 lane，该 gathering 最长等 5 秒。不是死锁，
  但它正是 4.6 所说的"lane 在 kernel 里到不了 checkpoint"，须与 cohort 协议一并论证。

另须注意提交资格与观察资格分开：提交 helper 可能仅在 AIV 可用，观察必须每个参与调度的核
都能做。

### 7.2 共享检查

owner-only 的代价是延迟：owner 在长 kernel 里时，已满足的条件无人观察，消费者多等一个
kernel。若实测需要，加回 HBG 的 `check_directory` 位图与 `SchedulerAsyncLease` 即可让任何
已排空的 lane 检查任何 group —— `percounter_mix_drain` 已经是这些 lane 的驻留点。这是纯粹
的增量，不改变本文其余任何合同。

## 8. 验证

| 用例 | 覆盖 |
| ---- | ---- |
| 单 lane COUNTER | 登记一条、kernel 返回、消费者在信号到达前不执行 |
| 等待者多于核数 | 等待不占核；对照上游 `defer_wait_progress` 的 `c_plus_one` |
| 负初值与边界 | `INT32_MIN`、`-1`、`0`、`1`、`INT32_MAX` 阈值与当前值 |
| 零登记 | 带标记但未登记的任务走同步路径，不创建 CHECK |
| 自环 | 延迟任务的消费者由同一 lane 领取，不死锁 |
| SPMD | 同一任务的多个 block 分别延迟与不延迟，计数器总数正确 |
| MIX 3/5/6/7 | 部分 subtask 延迟；计数器终值与全不延迟时一致，中间值可为负 |
| sync-start cohort | 延迟不影响 `sync_queue->completed` 与 cohort 释放 |
| Graph body | body sink 延迟时 shell 计数器同样推迟 |
| 溢出与非法 | 超过 capacity、非法 expected、非法对齐各自锁存对应错误码 |
| 超时 | 条件永不满足时报 TIMEOUT，而非由 45 秒 op 超时收场 |

容量、区域字节数与同步路径开销分别度量；优化不得引入强制的额外用户声明。

## 9. 落地顺序

每一步各自正确、可单独提交，**不存在"合起来才对"的中间状态**。做到这一点靠一条规则：
host 只给本步已经支持的任务形状打"可能登记"标记，其余形状按既有的
`TaskPlacement::UNSUPPORTED` 路径拒绝，而不是默默走同步发布。

| 步 | 内容 | 完成判据 |
| -- | ---- | -------- |
| S1 ✅ | 布局、寻址与**拒绝**：region 预留 group 与 entries、`PercounterControl` 新字段、host 预填、每次 dispatch 开启与封口；kernel 返回后 `count > 0` 或 `error_code != 0` 即报错 | 登记条件的 kernel 得到指名的运行时错误，而非 trap 或错误答案；不登记的图行为一字不变 |
| S2 ✅ | 把拒绝换成延迟：`percounter_async_tick`、结算、退出判据，只接 `percounter_lane_loop` 一个发布点 | 条件满足的用例通过（S1 下同一 kernel 报 error 13）；永不满足的报 error 10 TIMEOUT 而非挂到 op 超时；回归集 9 passed，两遍一致 |
| S3 ✅ | cohort 路径：`percounter_sync_serve_block` 单 lane 分支 | 单核 `require_sync_start` 多 block 的登记用例通过，同一用例在 S2 构建上报 error 13（0 of 7 blocks resolved）；回归集 11 passed，失败仍只有那两个 swimlane 用例 |
| S4 | MIX 路径：4.5 的先减后加，接 `percounter_serve_mix` | MIX 3/5/6/7 用例；计数器终值与全不延迟时一致 |
| S5 | Graph body、溢出与非法条件、容量与开销度量 | 第 8 节矩阵补齐 |

**S1 既是布局也是 1.1 的修复**，两者不可分：登记失败的 kernel 在设备上不留痕迹，所以只有
先给它一个真正能写进去的 group，percounter 才看得见"这个任务登记了条件"。S1 之后登记机械
地成功、被读到、被拒绝；S2 把拒绝换成延迟，只改"读到之后做什么"。

S2 是第一个有功能的提交，也是第一个不能再拆的：只登记不检查会挂死，检查了不改退出判据
会在消费者未释放时退出。S3 到 S5 在它之后各自独立成立，靠的是 host 只给本步支持的任务
形状打"可能登记"标记，其余形状仍走 S1 的拒绝路径。
