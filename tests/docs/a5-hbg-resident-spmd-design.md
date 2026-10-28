# a5 HBG resident 调度器支持 SPMD 设计（per-core 版）

> 面向：simpler 仓库的 runtime 开发者与本设计的评审者。
> 状态：设计稿，尚未实现。本稿**取代**基于 cluster-scheduler 模型的旧稿。
> 范围：a5 架构 `host_build_graph` 的 resident AICore 调度器。
>
> **基线**：分支 `a5-hbg-per-core-phase-a` @ `872591b7`（per-core 改造阶段 1 + D1b + D3a 已落地）。
> 文内行号一律按此 commit，repo 相对路径。per-core 改造本身见
> [a5-hbg-per-core-scheduler-plan.md](a5-hbg-per-core-scheduler-plan.md) 与
> [a5-hbg-per-core-scheduler-progress.md](a5-hbg-per-core-scheduler-progress.md)。
>
> **全文的 108 / 36 / 72 是容量常量，不是实测核数。** `SCHEDULER_WORKER_CAPACITY = 108`
> （36 cluster × 3），而本机实测 `num_cores` 是 **96**（32 cluster，32 AIC + 64 AIV）。
> progress §A4 专门记过一次把容量当实测值写进基线的错（"21/108"应为"21/96"）。
> 凡本文说"全机""所有核"，按运行时真实的 `cluster_count` 读；写用例阈值时一律从 run
> 自报的 `num_cores` 推导，**不要写死这三个数**。

## 1. 问题与现状

`scheduler_resident_v0_task_shape_supported`
（`src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_graph.h:58-61`）：

```cpp
return active_subtasks >= 1 && active_subtasks <= 3 && logical_block_num == 1 && !sync_start;
```

四个 clause 里只有 `logical_block_num == 1` 实际生效。命中后
`host/runtime_maker.cpp:1259-1266` 对**整张图**调 `select_legacy_scheduler`，没有 per-task 混合模式
——图里只要有一个 SPMD 任务，整个 run 都用不上 resident 调度。

排查关键字：`A5 HBG: retaining AICPU scheduling for task ref=... with SPMD or sync-start shape`。

这不是边缘情况。`examples/a5/host_build_graph/qwen3_14b_decode/kernels/orchestration/decode_fwd_layers.cpp`
一层内就有数十个 SPMD 任务（`set_block_num` 1~50，另有一个 Mix SPMD 在 `:532`）；分支自带的
`examples/a5/host_build_graph/deepseek_v4_pro_csa/` 整图 60 个多 block 任务、最大 `block_num=512`。

## 2. 旧稿为何必须重构

旧稿的实现前提是「每核一个 Scheduler」改造先行，并据此声明"任何空出来的核都能自己到队首认领
block"。改造已落地，但它同时改掉了旧稿赖以成立的另外几件事。

| 旧稿前提 | 现状 | 后果 |
| -------- | ---- | ---- |
| 每 cluster 一个 Scheduler 管 3 lane；Mix 由主管理核放置 | 每核自调度；Mix 由本 cluster **AIC** 协调、写 AIC 独占的 mix 槽（`scheduler_ready.h:236`、`scheduler_mix.h:80-82`） | 认领主体 36 → 108（容量；实测机为 96）。"空闲 lane 数"不再由任何一方集中掌握，`scheduler_claim_spmd_blocks` 里"算本 cluster 该 type 的空闲 lane 数 k"这一步**没有数据来源** |
| ready inbox 任何核都能消费（偷同 shard） | **D1b**：按 core type + `type_rank` 编址（36/72/36，`scheduler_types.h:686-695`）；push 只投**本 cluster** 同 type 核（`scheduler_ready.h:990-999`）；steal 只扫自己那 7 个 owner 的 shard（`scheduler_ready.h:1265-1288`） | **致命**。躺在某个 inbox 里的任务只有 7 个同 type 核能看见。实测单根扇出只摸到 14/96 核（progress §A4）。`block_num=50` 的任务**不可能**靠"待在 ready 队列里等人认领"铺开 |
| 「队首非空 → 全局停取新任务」 | 旧模型 36 个 Scheduler 各停，代价 3 lane | 现在是 108 个核各停。AIC-only 的 SPMD 会让 **72 个 AIV 停工**。旧稿 §3 已证明 non-sync_start 不需要它；本稿 §8.2 进一步发现 **sync_start 也不需要**（co-residency 由 §8.1 的四条保证），所以这条规则整体降级为可选的延迟优化 |
| `next_block`/`completed_subtasks` 放 TaskControl 第一条线，"是否拆到别的线由实测决定" | progress §7.10 已证明：第二条线被整线回写时会**还原该核缓存里的旧副本** | 不是实测取舍而是**硬约束**：跨核 RMW 字段只能放第一条线。该线现为 `state`/`wake_list_head`/`next_waiter` + `padding[40]`（`scheduler_types.h:838-862`），仍够放 |
| Mix 的 SPMD「一个计数器同时覆盖 block 维与 lane 维」 | per-core 的 `mix_trackers` 是**核本地**的（`scheduler_ready.h:231` 的 `has_mix` 同族） | block 内聚合用本地 tracker，block 间聚合必须走 GM 原子：**两级**而非一级 |
| §7.2 host 侧按形状校验 `block_num` 上限 | **已经实现**：`aic_worker_demand = max(block_num)`、`aiv_worker_demand = max(block_num × AIV 子任务数)`（`runtime_maker.cpp:1244,1249` → `:1326-1327`），`aicpu/aicore_lifecycle.cpp:242` 比对真实核数 | 不必新增。但它**管的是所有 SPMD 任务而不只是 sync_start**，见 §9 —— 不放宽就会拒掉 qwen 与 CSA |
| §7.4 gang 预留字段"留给实现阶段决定" | 阶段 B 已全删，`scheduler_task_is_gang` 也没了 | 无事可做。但这意味着放行后**下游没有任何守卫**：形状准入门是唯一事实来源（见 §9 末） |

### 2.1 三个旧稿没有的发现

1. **trace 索引不含 block。** `scheduler_task_trace_index`（`scheduler_types.h:813-815`）只按
   subtask 取偏移，而 `trace_index_base` 按 `block_num × active_subtasks` 步进
   （`runtime_maker.cpp:1253`）——空间已预留，索引却没用上，一个 SPMD 任务的所有 block 会写进
   **同一个 trace cell**。见 §7.3。
2. **SPMD + predicate 被 host 硬拒而非回退**：`runtime_maker.cpp:1185` 的
   `has_predicate() && (active_subtasks != 1 || logical_block_num != 1)` 直接 `return false`
   （`prepare_native_run failed with code -1000`，已由 `tests/st/a5/host_build_graph/mix_predicated_dispatch`
   钉住）。⇒ **SPMD 路径上不存在 predicate 一致性问题**，旧稿关于求值收归一处的顾虑在此不适用。
3. **Mix 的 payload 仍由 AIC 准备**：`scheduler_mix.h:47-62` 的 `prepare` 循环跑在协调者上，
   执行侧 `aicore_executor.cpp:464-466` 是 `observe_dispatch_payload_*`（读别人写的 payload）。
   计划 D3 的"prepare 搬到 AIV 侧"**未实现**。⇒ 对 SPMD 的影响不是"谁认领"（Mix block 的
   认领者就是协调者自己），而是**一次认领要写几份 payload**：非 Mix 的认领核写自己那一份，
   Mix 的协调者要替两个 AIV 各写一份到它们的 mix payload 区，`block_idx` 也由它写三遍。

## 3. 硬约束（逐条复核）

| # | 约束 | 变化 |
| - | ---- | ---- |
| 3.1 | 一个核的视野：本 cluster 三条 lane（`config.worker_ids[0..2]`，lane0=AIC）；跨 cluster 必须经 GM | 不变 |
| 3.2 | ready inbox 的链接字段 `next_waiter` 只有一个，一个任务同时只在一条链上（wake 链或 ready 链） | 不变。**所以这条链无法表达"N 个核各认领一段 block"** |
| 3.3 | 同一核的 2 个 slot 是排队位不是执行位：executor 取 `local_ready_mask` 里最老的那个（`scheduler_ready.h:450-467`），串行执行 | 不变。认领宽度必须按**核**算，不按 slot 算 |
| 3.4 | generation 不可撤销：`stage` 一旦发布，该 slot 的消费者就锁定这个号 | 不变。不存在"先 stage 再放弃" |
| 3.5 | `resolved_task_count` 每 task 只能加 1（`aicpu/aicpu_executor.cpp` 的收尾核对） | 不变 |
| 3.6 | SSBUF 跨 cluster 不可寻址 | 不变。做不到 TMR 式的全局 rendezvous |
| 3.7 | **跨核 RMW 字段只能放 TaskControl 第一条线**，且该线不得被 `publish_cache_line` 整线发布 | **新增**（progress §7.10 的结论） |
| 3.8 | GM 的**读改写**原语只有 64 位：`scheduler_gm_fetch_add` / `_fetch_or` / `_fetch_and` **只有 `uint64_t`** 一个重载，CAS 有 `int64_t`+`uint64_t`（`scheduler_memory.h:195-258`）。非 RMW 的 store/load **有 32 位形式**（`scheduler_gm_store(uint32_t&)`@175、`scheduler_gm_query_u32_pair`@137），所以不能由本条推出"32 位共享字段不存在"。`scheduler_gm_query` 走 `ld_dev` 绕过标量 DCache，读一致、无需 invalidate | **新增**。所有新增**计数器**一律 `uint64_t` |
| 3.9 | **一个核在 `execute_task` 内部时整个调度循环停摆** —— 它不取任务、不服务完成、不读 SPMD 队首。sync_start 的 block 在 barrier 上自旋就处于这个状态 | **新增**。§8.1 的直填旁路与"每核 ≤1 块"都源于此 |
| 3.10 | 本核就绪槽按 **generation 最小**优先执行（`scheduler_local_ready_pop`，`scheduler_ready.h:450-467`），generation 是本核单调递增的 `dispatch_sequence` | **新增**。曾被当作「两个 cohort 不互锁」的依据，**那条链已于 2026-10-27 实测证伪**（§8.1）：自有槽与 mix 槽是两套独立的 generation 序列，执行器又无条件先取自有槽，所以本条只在单一通道内成立。信箱模型落地后次序由 running / 信箱本身表达，generation 只剩完成匹配一个用途 |

## 4. 调度模型：队首即认领对象，任务自身游标作权威

§2 的致命项决定了模型：**SPMD 任务不能靠"躺在 per-core 队列里"被认领**，因为那种队列的可见
半径是 7 个同 type 核。所以 SPMD 任务进一条**全局唯一的 MPMC 队列**，而**队首就是当前正在
认领 block 的那个任务**。

```
         resolve / bootstrap
                 │  (SPMD flag)
                 ▼
        ┌────────────────────────────────────────────┐
        │  SPMD ready 队列：全局唯一 MPMC             │
        │  endpoints = CAS 打包的闭区间 (head, tail)  │
        │                                            │
        │   head ──► 当前认领对象（不被取走）          │
        │   ...  ──► 排队等待的后续 SPMD 任务          │
        └────────────────────────────────────────────┘
                     │ 读 endpoints 取 head（1 次 ld_dev）
     ┌───────────────┼───────────────┐
     ▼               ▼               ▼
 核 A 认领 block i  核 B 认领 block j  核 C 认领 block k
 （CAS 任务自己的 next_block；全部认领完后任何核都可推进 head）
```

- **进入**：SPMD 任务与普通任务一样被 resolve 成 ready，但进这条全局队列，不按 core type 或
  `type_rank` 分片。任何核都能 push。
- **认领**：任何核空出 dispatch 槽时读一次 `endpoints` 取 head；队首任务与自己 core type
  匹配就 CAS 它的 `next_block` 取一个 block。
- **让位**：队首的 block 全部认领完（**不是执行完**）后，任何观察到这一点的核都可以推进 head。

### 4.1 不需要独立的"认领槽"——队首就是它

队首天然具备一个独立槽位要提供的两件事，这是旧稿的论证，本稿照用：

- **认领期间稳定**：队列是 FIFO，入队只改 tail（空队列那一次 CAS 同时设好两端），
  所以新任务永远顶不掉正在认领的队首。若改成 LIFO，入队就是 CAS head，新任务会把正在认领的
  那个顶下去 —— 两个任务同时在认领，直接破坏"一次一个"。
- **"进入认领中"的唯一权**：空队列时入队的那一次 `CAS(endpoints, EMPTY, …)` 本身就是互斥，
  赢家的任务成为队首。不需要第二个 CAS 去抢一个槽。

一个独立槽位唯一能多提供的是**让 core type 不匹配的核少读一次 metadata**（把形状预先摊在槽
的 tag 里）。而 `SchedulerTaskMetadata` 是 16 字节、落在一条 cache line 内
（`scheduler_types.h:805-811`），且认领成功后 prepare 本来就要读它 —— 省下的只是不匹配者的
一次 `ld_dev`，不值得用一个新结构 + 搬运协议去换。

**于是现有 `SchedulerReadyInbox` 原样够用，零新结构。** progress §7.9 已把它做成 CAS 打包的
闭区间 FIFO、§7.10 把链接字段改成 `atomicExch` 并搬到只走原子原语的第一条缓存线 —— 它本来
就是 MPMC 的。SPMD 与普通队列的差别只在**消费方式**：

| | 普通 ready 队列 | SPMD 队列 |
| --- | --- | --- |
| 看有没有工作 | 读 `endpoints` | 同 |
| 取工作 | pop：CAS 推进 head，**任务被取走** | **peek**：只读 head，任务留在原地 |
| 何时推进 head | 取走时立即 | **队首 block 全部认领完之后**（延迟 pop，走现有 `scheduler_ready_pop_from_inbox`） |

### 4.2 同时只有一个 SPMD 任务处于认领中

旧稿的规则，本稿**原样保留**。它的作用不止于 sync_start 互斥（那只是它顺带覆盖的一条）：

- **它让"认领失败"只有一种处置。** Mix block 要落在本 cluster 的 (AIC + AIV) 上，如果只有
  AIC 空、两个 AIV 还在执行，placement 就失败 —— 此时正确的动作是**本轮放弃、等下一轮**，
  而不能转而认领别的 SPMD 任务。队首唯一，所以"不能转而认领别的"是**结构性事实**，不是一条
  要靠代码纪律维持的规则。这对 sync_start 尤其要紧：cohort 正在集结时跑去认领一个 AIC-only
  任务，会把本该留给 cohort 的核占走。
- **它免掉了"认领谁"的选择点。** 多个任务同时可认领时，核每轮都要在它们之间选，而选错
  （持续挑 non-sync_start）会让 cohort 饿死 —— 那需要一整套优先级规则。队首唯一，没有选择。
- **它本身就是 sync_start 的互斥。** 队首唯一那个任务要么是 cohort 要么不是，两个 cohort
  不可能同时在认领，**不需要任何额外的令牌位**。

代价是**旧稿**（那份文档的 §4.5）已经承认过的那笔：**AIC-only 的任务在认领期间，AIV 核
拿不到它的 block**（它们仍会从自己 inbox 取普通任务，只是吃不到这个 SPMD 任务的并行度）。
本稿接受这笔代价，不做任何消除它的设计。

**权威在任务自己的 `next_block`，不在队列。** 认领不动 `endpoints`，只 CAS
`SchedulerTaskControl::next_block`。于是 head 已经推进、而某个核还在用旧 head 的情形是无害的：
它会去 CAS 一个已认领干净的任务，`current >= block_num` 使 CAS 不成立，无副作用 —— 错误路径
在结构上不存在，而不是靠协议防它（与 progress F-1 同一思路）。

### 4.3 为什么 SPMD 队列是全局唯一的 MPMC 队列

**per-core 分片对 SPMD 毫无价值,还要付扩容的代价。** 分片（D1b 按 `type_rank` 编址）的收益是
让同 shard 的 7 个 owner 同类型、让 steal 命中率高；但 SPMD 任务**不是被 pop 去执行的**,它只是
等着被搬进认领槽,进槽之后谁都能认领。所以分片对它只剩下坏处:

| | per-core 36 inbox | **全局唯一（选定）** |
| --- | --- | --- |
| inbox 数组 | 144 → **180** | 144 → **145** |
| directory | `[3][11]` → `[4][11]` | **不要** —— 只有一个 owner,判空直接读它的 `endpoints` |
| steal | 要配第 4 个 victim cursor + 扫 shard | **不需要**,没有第二个 owner 可偷 |
| 搬进槽的角色 | 只有持有该 inbox 的 AIC | **任何核** |
| 可见半径 | 7 个同 type 核 | 全机 |

**现有 `SchedulerReadyInbox` 已经是 MPMC 的**,直接复用,零新结构:progress §7.10 把 `endpoints`
改成 CAS 打包的闭区间两端、把链接字段 `next_waiter` 改成 `atomicExch` 并搬到只走原子原语的第一
条缓存线。全局 SPMD 队列就是这个结构的一个实例,`owner_count = 1`。

落地参数：

- `SCHEDULER_READY_QUEUE_COUNT` 3 → 4。保留 queue 维度是为了 `resolve_completion` 的
  `batches[]` 与 `scheduler_task_ready_queue()` 分类机制原样可用,不必为 SPMD 另开一条发布路径。
- `scheduler_ready_queue_capacity(3) = 1`、`base(3) = 144`,
  `SCHEDULER_READY_INBOX_CAPACITY = 144 + 1 = 145`。
- `scheduler_ready_queue_owner_count(ctx, 3) = 1`、`scheduler_ready_push_target(ctx, 3, ref) = 0`。

**三处必须动的**（其余自动泛化）：

| 函数 | 现在对 `core_type_index == 3` 的行为 | 要怎样 |
| ---- | ------------------------------------ | ------ |
| `scheduler_ready_queue_base`（`scheduler_types.h:691-695`） | `0?0 : 1?36 : 36*(1+2)` ⇒ **108,与 `base(2)` 撞车** | 显式 `base(3) = 144`。**这是会静默共享队列的错误**；`:1203-1207` 的 contiguous 断言只枚举了 0/1/2,补到 3 才抓得住 |
| `scheduler_ready_queue_holder`（`scheduler_ready.h:614-616`） | `(3==1)==(self==1)` ⇒ 只有 AIC 为真 | 全局队列**所有核都是持有者**,要对 queue 3 返回 true |
| `scheduler_refresh_ready_inbox` / `_type`（`scheduler_ready.h:938-978`） | 维护 directory bit、`owner_ready_queue_mask` | queue 3 **跳过 directory 那一整套**,判空直接读 `endpoints` |

`scheduler_ready_queue_capacity` / `_owner_count` / `_push_target` 三个**不用改**:
非 1 分支天然给出 1 / 1 / 0（前两个需把 `clusters` 换成按 type 取值,见上表参数）。

**`ready_victim_cursors` 仍要扩到 4 个元素**（两处花括号现在只写 2 个初值,
`aicore_executor.cpp:245-248`、`:281-284`,第三个靠零初始化兜着）——queue 3 不走 steal,
但数组按 `QUEUE_COUNT` 声明,少写一个初值就又回到"写少了但恰好没错"的状态。

### 4.4 队首占用期很长，但只影响其他 SPMD 任务

队首在它的 block **全部认领完**之前不让位，而"认领完"对大任务是个很长的过程：
`block_num = 512` 在约 96 个核、每核每轮 1 块、每核 2 槽的节奏下，要经历好几波**执行**才认领
得完（秒级），不是几轮 CAS。所以后续 SPMD 任务在队列里等很久是常态而非例外。

**这正是 SPMD 不能混进 AIC/AIV ready 队列的理由。** 混进去，队首那个 SPMD 任务会把排在它
后面的**普通任务**一起堵住（D1b 把 MIX 单独成链是同一考虑）。给 SPMD 一条自己的队列，
等待只发生在 SPMD 之间，而这正是"一次一个"本来就接受的串行。

### 4.5 SPMD 优先于其余一切取任务，但不停取

核每轮的次序是：**先看 SPMD 队首，再填 Mix，最后看自己的 inbox**。SPMD 排在最前而不只是
排在 inbox 之前，因为 **Mix 填槽争的正是 Mix SPMD 认领要用的那批资源** —— 协调者的自有槽
与本 cluster 的 mix 槽。先跑 Mix，普通 Mix 工作会把放置容量吃掉，一个 Mix SPMD 队首就探不到
位置、白等一轮。这同时是 sync_start cohort 能够集结的 liveness 来源（§8.1 第 1 条）。

但这只是**次序**，不是停取：队首与本核 core type 不匹配、或 placement 失败时，本核照常
往下去填 Mix、去 inbox 取普通任务。旧稿那条"队首非空即全局停取新任务"**不采用** —— 在
per-core 下它会让 72 个 AIV 核为一个 AIC-only 任务整体空转（§8.2 另有论证说明它也不是
正确性所需）。

每核每轮至多认领 1 个 block（约束 3.3）。**这条是定下来的，不作为开放项**：一次认领多块虽能
减少 CAS 轮数，但会让先空出来的核把 block 抢走、后空出来的核无事可做，负载均衡直接退化。
CAS 轮数可以忍，不均衡不可以。第二个 dispatch 槽由后续轮次自然补上 —— 别人都忙时槽照样填满，
别人空闲时不抢跑。

## 5. 数据结构变更

### 5.1 没有新的共享结构

SPMD 队列复用 `SchedulerReadyInbox`（§4.1），认领游标与完成计数挂在任务自己的
`SchedulerTaskControl` 上（§5.2）。**本设计不引入任何新的 GM 共享结构** —— 既没有认领槽，
也没有 summary 位图或 entry 数组。

这一节保留，是为了让下一个读者知道"找不到新结构"不是遗漏。

### 5.2 `SchedulerTaskControl` 第一条线（`scheduler_types.h:838-862`）

| 字段 | 读写 | 说明 |
| ---- | ---- | ---- |
| `next_block`（新增 `volatile uint64_t`） | 认领者 CAS | 下一个未认领 block。权威游标（§4 第 2 点） |
| `completed_blocks`（新增 `volatile uint64_t`） | 完成者 `fetch_add` | 已完成 block 数 |
| `block_num`（新增 `volatile uint64_t`） | host 写一次，核经 `gm_query` 读 | 本任务 block 总数 |

**三个字段都必须是 `uint64_t`，不能用 `int64_t`。** `scheduler_gm_fetch_add`
（`scheduler_memory.h:195-203`）**只有 `uint64_t` 一个重载**，`_fetch_or` / `_fetch_and`
同样只有 `uint64_t`；只有 CAS 两种都有。写成 `int64_t`
会在 `fetch_add` 那一行直接编不过。

该线现用 24 字节（`state`@846 / `wake_list_head`@847 / `next_waiter`@852）+
`atomic_line_padding[40]`@853，加三个 64 位字后为 48 + `padding[16]`，
`sizeof == 128`（`:1177`）与两条 offset 断言（`offsetof(next_waiter) < 64`@1182、
`offsetof(next_fanin_index) == 64`@1184）都不变。

**`block_num` 放这条线而不是 `SchedulerTaskMetadata`**：metadata 是 16 字节的热结构
（`scheduler_types.h:805-811`，`static_assert` 在 `:1175`），每个任务每次 dispatch 都要读，
为 SPMD 加字段会让全图 metadata 读放大一倍。而第一条线只经 `ld_dev` 读，不需要 invalidate
（约束 3.8），放一个只读字没有代价。

**反转旧稿一条结论**：旧稿把 `next_block` 与 `completed_subtasks` 的 false sharing 列为
"待实测确认，可能拆到另一条线"。**不能拆** —— 第二条线会被
`scheduler_publish_cache_line(&control->next_fanin_index)`（`scheduler_ready.h:1671`，另一处在 `:706`）整线回写，
把该核缓存里的旧副本拍回 GM。这正是 progress §7.10 那个真机回归的机制。注释里要写明这是
约束而非取舍。

### 5.3 `SchedulerLocalSlotState`（`scheduler_ready.h:144-150`）

需要携带 `block_idx` 到完成路径（计数不需要它，trace 需要，见 §7.3）。当前 15 字节用满 + 1 字节
padding，而 `block_idx` 要 16 位（非 sync_start 的 block_num 不设上限，CSA 已有 512）。

取**加宽到 24 字节**并同步 `sizeof(SchedulerLocalSlotState) == 16` 与
`sizeof(SchedulerLocalState) == 432` 两条断言（`scheduler_ready.h:312-313`）。这两条是防意外
膨胀的 tripwire，不是 host↔device wire 契约（local state 是核本地的），可以改。
8 个 slot state 共 +64 字节。

> `aicore_executor.cpp:119-120` 记着"把时间戳挪出 kernel 调用的 live set，否则 resident 循环
> 的 spill 会在 A5 上 fault"。局部状态变大不是免费的。若实测出现栈压力，退路是把
> `state`（4 个取值）与 `subtask_slot`（0~2 + 无效）压进一个字节腾出空间，保持 16 字节。

### 5.4 其他

| 对象 | 变更 |
| ---- | ---- |
| `SchedulerGraphSummary`（`scheduler_types.h:885-888`） | 加 `spmd_task_count`，派生 `local->has_spmd`。**没有 SPMD 的图一轮都不读 SPMD 队列** —— 每核每轮一次 GM load 不是免费的，这个门必须有，形状照 `has_mix` |
| `SchedulerFreeSlotClaim` | 加 `uint16_t block_idx{0}` 与 `uint16_t block_num{1}`（prepare 要把两者写进 payload，commit/completion 要用 block_idx 取 trace 下标）。**不设哨兵**：非 SPMD 任务就是「第 0 块，共 1 块」，这是它的真实形状而不是"没有值"，所以每个既有派发点一个分支都不用加 |
| `SchedulerLocalSlotState` | 同样带 `block_idx` 与 **`block_num`**。后者是 §7.1 判「要不要跨 block 聚合」的依据（`block_num > 1`），放在槽里使完成路径不必读 metadata，也省掉 §7.1 原本要做的 `gm_query(control->block_num)` |
| `SchedulerReadySource`（`scheduler_types.h:740-744`） | 加 `SPMD_CLAIM = 3`。不加的话这些 block 在 swimlane 里会伪装成 `LOCAL`，而 `ready_source` 正是 §10.2 用来判断 block 从哪来的字段 |
| `SchedulerErrorSite`（`scheduler_types.h:914-947`） | 新增 `SPMD_COMPLETION_OVERRUN`（`completed_blocks > block_num`），沿用 ready 的 80 段。**没有专属 site 的失败在设备日志里无法归因** —— `running-onboard.md` 的 507018 triage 整套流程都是按 site 走的。初版还想加一个 `SPMD_CLAIM_INVALID_BLOCK` 查 block_num 范围，**不要加**：host 侧已经硬拒，设备侧永不可达（见 §10.4 的实做修正 5） |

**不需要** 任何新的 layout offset：没有新结构，所以 `AicoreSchedulerLayout`、
`SchedulerWorkerContext` 的 `topology_reserved`、`SchedulerLocalConfig` 全都不动。
SPMD 队列是 `ready_inboxes` 数组里的第 145 个元素，已有的 `ready_inboxes_offset` 就能寻址。

## 6. 认领与发布

整条路径只有一个新函数 `scheduler_claim_spmd_block`，它同时承担 peek、认领、推进 head 三件事
—— 没有"搬上板"这一步，队首就是认领对象（§4.1）。

### 6.1 调用点：主循环里自己的一次调用，排在所有取任务之前

**不能放进 `scheduler_fill_own_slots`。** 那个函数开头就有
`!scheduler_has_usable_slot(...) || !scheduler_ready_directory_nonempty(...)` 的提前返回
（`scheduler_dispatch.h:45-51`），而 SPMD 队列**不进 directory**（§4.3），所以塞在那道闸门
之后的认领**一次都不会执行**。

门控 `local->has_spmd`，位置是 `scheduler_service_cluster_completions` 之后、
**`scheduler_fill_mix_after_completions` 之前**，`scheduler_fill_own_slots` 更在其后。
放在 Mix 之前是必须的，理由见 §4.5：两者争同一批放置资源。

一个有利的副作用：完成路径攒下的 `direct_mix_ready` 候选是在
`scheduler_fill_mix_after_completions` 里才真正派发的，所以认领排在它前面，等于让队首先看到
那些刚空出来的 mix 槽。

**bootstrap 不特殊处理**：`bootstrap_ready_graph` 照常 batch + push，root SPMD 任务在第一个
稳态轮次被认领。bootstrap 与稳态同一条路径。

### 6.2 认领（任何核）

1. 读 `endpoints`，取 head。`EMPTY` ⇒ 返回（一次 `ld_dev`）。
2. 读队首任务的 metadata，判形状：
   - 与本核 core type 不匹配 ⇒ 返回（去 inbox 取普通任务，§4.4）。Mix 型只有
     `is_mix_coordinator()` 继续。
   - 是 sync_start 时，先扫本核自己的槽（自有 2 个 + 协调者的 mix 3×2）确认没有同一
     `task_ref`（§8.1 第 2 条）。
3. **先探放置，再 CAS 认领。顺序不能反。**
   - 非 Mix：确认本核有 `state == FREE` 的自有槽。
   - Mix：跑一遍 `scheduler_plan_mix_placement`，它可能失败（本 cluster 的 mix 槽或自有槽不空）。
   - 探测**不改任何共享状态**，所以失败就是**本轮放弃、返回**，没有需要回滚的东西，
     也**没有"换一个 SPMD 任务试试"这个选项**（队首唯一，§4.2）。
   反过来写（先 CAS 后放置）会把一个 block 认领下来却无处安放，而 `next_block` **不能回退**
   —— 那个 block 就永久丢失，任务的 `completed_blocks` 永远凑不满。
4. 循环 CAS 推进 `control->next_block`，认领 **1 个**（约束 3.3）：

   ```cpp
   uint64_t current = scheduler_gm_query(control->next_block);
   uint64_t claimed = UINT64_MAX;
   while (current < block_num) {
       const uint64_t observed = scheduler_gm_compare_exchange(control->next_block, current, current + 1);
       if (observed == current) { claimed = current; break; }
       current = observed;
   }
   ```

   `current >= block_num` 退出 ⇒ 队首已认领干净，去推进 head（§6.3）再返回。
   这也是"head 已被别人推进、本核还拿着旧 head"的唯一后果：权威在任务自己的游标上，
   所以那次 CAS 不成立，无副作用（§4.2 末）。
5. 填槽：构造带 `block_idx = claimed` 与 `block_num` 的 `SchedulerFreeSlotClaim`，
   以及 `source = SchedulerReadySource::SPMD_CLAIM`（**新增枚举值**，否则这些 block 在
   swimlane 里会伪装成 `LOCAL`，而 `ready_source` 正是 §10.2 用来判断 block 从哪来的字段）
   的 `SchedulerReadyClaim`，走现成的 `scheduler_fill_dispatch_slot`。

   **Mix 的三个值要填进 placement。** `scheduler_plan_mix_placement` 产出的
   `placement.slots[0..2]` 本身就是三个 `SchedulerFreeSlotClaim`，所以 `block_idx` /
   `block_num` 要在第 3 步探测成功后写进这三个 claim，`scheduler_dispatch_mix_ready` 随后把
   它们原样传给三次 `scheduler_prepare_dispatch_slot` —— 不要在 prepare 里重新读 GM
   （§6.4）。三个 subtask 共享同一个 `block_idx`：它们是同一个 block 的不同部分。
6. 若 `claimed + 1 == block_num` ⇒ 推进 head（§6.3）。

#### 稳态轮询的真实代价是两次 GM 访问，用一个核本地的字消掉第二次

第 1 步是一次 `ld_dev`，但**第 2 步的 metadata 读更贵**：`SchedulerTaskMetadata` 是走
DCache 的 16 字节结构，要 `scheduler_observe_cache_line`。所以"每核每轮一次 GM load"是错的，
实际是一次 `ld_dev` 加一次 cache-line observe。

最糟的场景恰好是 §4.2 已经接受的那个：一个 AIC-only 的 `block_num=512` 任务占着队首数秒，
期间 **72 个 AIV 核每一轮都付这两次而一块也拿不到**。§4.2 把这笔账只记成"吃不到并行度"，
漏了它同时是持续的轮询开销。

**消掉它只要一个核本地的字**：记住上轮看到的 head `task_ref` 与当时判出的结论
（匹配 / 不匹配 / 是 cohort）。head 没动就跳过第 2 步，稳态退回真正的一次 `ld_dev`。
head 变了才重读 metadata。这是纯本地状态，不涉及任何共享结构，也不改变任何不变式 ——
权威仍在任务自己的 `next_block` 上（§4.2 末），memo 过期的唯一后果是多跑一次 CAS 失败。

**§8.1 第 1 条的直填抑制必须复用这个 memo。** 那条规则要求完成路径去读 SPMD 队首，
而 §7.1 为了省掉完成路径上一次 metadata 读专门把判据放进了槽 —— 不复用 memo 就等于把省下
的读原样加回同一条路径。门控顺序：`has_spmd` 为假直接跳过；为真时先比 memo 的 head，
没变就用上轮结论。

### 6.3 推进 head（让位给下一个 SPMD 任务）

**就是现有的 pop**：`scheduler_ready_pop_from_inbox` 对 SPMD 队列调用一次，把 head 推到
`next_waiter`。它的 CAS 以 `endpoints` 当前值为期望，所以两个核同时推进只有一个成功，
另一个重试时发现 head 已变、或队列已空，都是正常路径。

> **以上两句是错的，S1 实测推翻（2026-10-23）。不要照它实现。**
>
> CAS 保证的是**更新原子**，不是**决定正确**。两个核都判定旧队首已认领干净时：第一个 CAS
> 成功把 head 推到后继；第二个 CAS 失败后**重试，读到新队首，把它也弹掉**——而新队首的
> block 一个都还没发完，弹掉之后再没有任何队列能找到它。
>
> 两个核同时到达这一步是**常态而非竞态边角**：抢最后一块的所有核，CAS 循环都以
> `current >= block_num` 结束，于是都会去推进。
>
> 实测签名（`spmd_mixed_waves` 一个 4×8 的波，第一次跑就命中）：
>
> | 任务 | next_block | completed | block_num |
> | ---- | ---------: | --------: | --------: |
> | AIC_SPMD | 7 | 7 | 16 |
> | MIX_SPMD | 1 | 1 | 8 |
> | MIX_COHORT | 1 | 1 | 4 |
>
> `next_block == completed < block_num` 且**队列为空**——认领中途停了，队首却已让位。
> 表现为 45s op-execute 超时、无 allocator fatal，即"假死锁"签名。
>
> **正确做法：推进必须带上"我认为该让位的是谁"，head 不是它就放手。** 这也顺带解决了
> `placement` 那条（见下），因为不再复用 `scheduler_ready_pop_from_inbox`：
>
> ```cpp
> scheduler_advance_spmd_head(..., int64_t exhausted, ...)
>   读 endpoints；空 ⇒ 返回
>   head != exhausted ⇒ 返回（别人已经推过了）
>   取 next_waiter（未发布则重读）
>   CAS(endpoints, packed, next_packed)；失败重试，但每轮都重新校验 head
> ```

**`placement` 必须传 `nullptr`。这一条会静默挂死，不是风格问题。**
`scheduler_ready_pop_from_inbox`（`scheduler_ready.h:1186`）里有：

```cpp
if (placement != nullptr && !scheduler_plan_mix_placement(..., head, placement)) return true;  // 不 pop
```

Mix 队列的调用方靠这一行实现"放不下就别取"。照抄到 SPMD 队列上会在**最坏的时刻**生效：
一个 Mix SPMD 队首的 block 全部认领完时，它的 mix 槽**正被自己的 block 占着**，所以
placement 必然失败 ⇒ head 永不推进 ⇒ 其后每个 SPMD 任务永久饿死。表现是 45s op-execute
timeout 且**无 allocator fatal**，即 `running-onboard.md` 那张表里最难归因的"假死锁"签名。

放置与让位在这里是两件无关的事：让位发生在 block 已经全部认领完之后，此刻没有任何东西
要放。配一条 UT —— 构造一个 mix 槽占满的 Mix SPMD 队首，断言 head 仍然推进。

**任何观察到 `next_block >= block_num` 的核都可以推进**，不必是认领到最后一块的那个 ——
这样即便那个核紧接着进了长 kernel（约束 3.9：此时它的调度循环整个停摆），队首也不会被
一直占着。

**不需要三态占位，因为没有"搬运"这一步。** 任务从入队到让位始终在同一条链上，
一生只入链一次（progress F-1 的性质原样成立）。

### 6.4 把 block 送进 payload

`scheduler_prepare_dispatch_slot`（`scheduler_ready.h:1325`）末尾补两行：

```cpp
payload->local_context.block_idx = slot_claim.block_idx;
payload->local_context.block_num = slot_claim.block_num;
```

**两个值都从 claim 来，不在这里重新读 GM。** 认领那一步为了跑 CAS 循环本来就已经读过
`control->block_num`（§6.2 第 4 步），所以 `SchedulerFreeSlotClaim` 同时带 `block_idx` 与
`block_num` 两个 `uint16_t`。`scheduler_prepare_dispatch_slot` 在 Mix 下会被调三次，让它各自
重读一次 GM 是没必要的。

**不需要分支，因为非 SPMD 不是特例。** claim 的默认值就是 `0 / 1`，而那正是单块任务的真实
形状；写成哨兵（曾考虑的 `BLOCK_IDX_NONE = UINT16_MAX`）会在这里和另外三个取下标的地方各引
入一个 `== NONE ? 0 : x` 三元，去表达一件 `0` 已经表达了的事。

这两个字段现在由 `host/runtime_maker.cpp:1386-1387` 在 per-run 镜像里硬编码成 `0 / 1`，
resident 路径从不更新 —— **SPMD 在数据结构层面无法表达，根源就在这里**。legacy AICPU 路径
写的是同样两个字段（`runtime/scheduler/scheduler_dispatch.cpp:178-179`），可作对照。
非 SPMD 任务仍得到 `0 / 1`，与现状一致，不需要分支。

`local_context` 在 CL0，而 `scheduler_writeback_dispatch_payload` 本来就回写整个 512B
（`scheduler_memory.h:77-85`），**cache 维护代码一行不改**。

## 7. 完成聚合

### 7.1 两级计数

`scheduler_service_cluster_completion_slot` 的 `task_complete` 分支
（`scheduler_completion.h:181-186`）扩展为：

```cpp
bool task_complete = true;
if (mix_tracker != nullptr) {
    ... 现有 Mix 逻辑不变,聚合的是「这个 block 的各子任务」...
}
// 判据来自槽本身:block_num 是派发时从 claim 写进去的,而 host 写一次 control->block_num
// 后再不改变,两者恒等。所以这里既不读 metadata 也不读 GM,而单块任务一条原子都不付。
// completed_blocks 是 uint64_t —— fetch_add 没有 int64 重载(§5.2)。
if (task_complete && completed_block_num > 1) {
    const uint64_t done = scheduler_gm_fetch_add(control->completed_blocks, 1) + 1;
    task_complete = (done == completed_block_num);
}
```

**判据是 `block_num > 1`,不是哨兵。** 本稿初版曾给 `block_idx` 设一个
`SCHEDULER_BLOCK_IDX_NONE = UINT16_MAX` 哨兵来回答"要不要跨 block 聚合",**已撤销**:
非 SPMD 任务就是「第 0 块,共 1 块」,那是它的真实形状而非"没有值"。把 `block_num` 一并
存进 `SchedulerLocalSlotState` 是免费的(槽已因对齐填到 24 字节),于是判据成了它的定义本身,
省掉四处 `== NONE ? 0 : x` 三元,也省掉这里原本要做的 `gm_query(control->block_num)`。

这个函数在**关 timing 的运行里**不读 `SchedulerTaskMetadata`,上面的判据必须保持这个性质
—— 为拿 `flags` 去加载 metadata 会给每个任务的完成路径添一次 GM 读。但不要在注释里写成
"这个函数从不读 metadata":开了 timing 时 `:132` 的 `scheduler_trace_index_at` 内部第一行
就是 `scheduler_task_metadata_at`,那一次读本来就付了,而 §7.3 正要给同一个函数加形参。

- **非 Mix SPMD**：一个 block 就是一个子任务，每块一次 GM 原子。
- **Mix SPMD**：block 内由认领它的那个 AIC 的**本地** tracker 聚合（不变），tracker 凑齐时才
  做一次 GM 原子。所以是 `block_num` 次原子，而不是 `block_num × popcount(active_mask)` 次
  —— 比旧稿"一个计数器同时覆盖 block 维与 lane 维"更省，而且本地 tracker 本来就在那里。

`completed_block_num` 与 `completed_block_idx` 都要和现有的 `completed_subtask_slot` 一样，
在**清槽之前**从 `local_slot` 取出来（`scheduler_completion.h:176` 取 `completed_subtask_slot`，
清槽在 `:187-190`）——前者是上面的判据，后者是 §7.3 的 trace 下标。**S0 已经把 trace 下标
那一处做好了**（它在 `:132`，天然早于清槽）；S1 只需在同一函数里把 `block_num` 也取出来。

> **同一个 Mix SPMD 任务在同一个 AIC 上可以有两个 tracker**（两个 block 各一个，
> `SCHEDULER_MIX_TRACKER_COUNT = 6` 够用）。看着像 bug 其实不是：tracker 按
> `local_slot->mix_tracker` **按下标**查，不按 task_ref 查，所以两个同 task_ref 的 tracker
> 互不干扰。sync_start 则被 §8.1 第 2 条禁止出现这种情况。

`task_complete` 为 true 之后下游**完全不动**：DONE 置位、wake list 解依赖、`++pending_completed`
全部沿用，`resolved_task_count` 仍每 task 加 1，约束 3.5 满足。

### 7.2 非末块不得解依赖

`scheduler_resolve_completion` 只在 `task_complete` 为真时被调用，而 `direct_ready`（完成时把
刚唤醒的后继直填回刚空出的槽）是在它**内部**产生的，所以非末块自动既不解依赖也不直填。
这条要在代码里写成 present-tense 不变式并配一个 UT：解两次依赖的表现是
`site=62 COMPLETION_WAKE_ALREADY_CLOSED`（progress §7.10 见过这个签名）。

### 7.3 trace 必须按 block 展开

`scheduler_task_trace_index` 改为：

```cpp
trace_index_base + block_idx * popcount(active_mask) + popcount(active_mask & ((1u << subtask) - 1))
```

非 SPMD 传 `block_idx = 0`，结果与现状逐位相同。`trace_index_base` 本来就按
`block_num × active_subtasks` 步进，**空间已经预留**（§2.1 第 1 点）。
调用点三处：`scheduler_commit_dispatch_slot`、完成路径的 `scheduler_trace_index_at`
（`scheduler_ready.h:838-841`，签名要加 `block_idx` 形参，值从 `local_slot->block_idx` 来）、
以及 `resolve_completion` 里那段 `ready_transition_cycles` 循环（`scheduler_ready.h:1636-1648`）
—— 最后一处在"还没有任何 block 被认领"的时刻执行，只写 block 0 的 cell。

下游影响待确认：`simpler_setup/tools/swimlane_converter.py` 会收到**同一 task_id 的多条
记录**（每 block 一条，`worker_id` 各不相同）。这正是验证"block 真的铺开了"所需的数据，
但 converter 与 `dfx/chip_swimlane` 用例的断言需要同步核对。

## 8. sync_start

sync_start 的 kernel 内部有跨 block barrier，硬要求是 **co-residency**（所有 block 都拿到核），
不是"同一纳秒启动"。先到的在 barrier 上等后到的，正常。

旧稿 §3 的死锁分析仍然成立，逐条复核后不变：

- non-sync_start 之间并发认领不死锁（已认领的 block 跑完即释放核）；
- non-sync_start 与 cohort 并存不死锁（同理）；
- **只有两个 sync_start cohort 之间会死锁**（各占一半、都等凑齐）。

### 8.1 co-residency 靠哪四条成立

旧稿的"全局停取"**不在**这个集合里（§8.2）。而"一次一个"直接免掉了两条本来会需要的规则：
cohort 之间的互斥令牌、以及"cohort 优先于 non-sync_start"的优先级规则 —— 队首唯一，
两者都成了结构性事实。

| # | 规则 | 没有它会怎样 |
| - | ---- | ------------ |
| 1 | **SPMD 优先于 inbox（§4.4 的次序），且队首是 cohort 时抑制直填** | 直填绕过这个次序（见下方「直填旁路」），核永远没有空槽给 cohort |
| 2 | **每核每 cohort 至多 1 个 block** | 单核内自锁：块 B 在 barrier 等 B'，而 B' 正排在同一个核的另一个槽里等 B 让出执行流（约束 3.3）。判据见下方「谁来判、扫哪些槽」 |
| 3 | **`block_num ≤` 该形状可用核数** | barrier 永远等不齐。由 §9 第 2 条保留下来的 demand 校验提供 |
| 4 | **队首在 block 全部认领完之前不让位**（§6.3） | 若中途让位，cohort 的剩余 block 再也没人认领 |

**两个 cohort 不可能同时在认领**，因为队首唯一：C2 要等 C1 的 block 全部认领完、head 推进
之后才成为队首。所以旧稿需要的"cohort 互斥"与本稿初版设想的 `summary bit[32]` 令牌
**都不需要存在**。

#### 如果将来要去掉 cohort 互斥：ticket 全序（设计备查，**不在当前计划内**）

队首唯一是在代偿 Coffman 四条件里的 **hold-and-wait**：cohort 是**全或无的资源集合，却被
增量获取**——每个核各领一块，谁都不知道全局能否凑齐，这是典型的 gang scheduling 死锁。
"一次只让一个 cohort 处于获取态"回避了它，代价是 **cohort 之间完全串行**：即使 C1 已全部
跑起来、不再需要任何新核，C2 也要等它**完成**。

要去掉互斥，必须换一条 Coffman 条件来断。两条路线：

**路线 A——全或无分配（断 hold-and-wait）。** cohort 在放置任何一块之前，先在一份按核型分的
全局可用位图上原子预留全部 `block_num` 个核，拿不全就一个都不拿，期间不持有任何核。无死锁是
构造性的。**但有两个硬伤**：满宽 cohort 会被源源不断的普通任务饿死，而一旦为此引入"预留但未
运行"的核，**hold-and-wait 就回来了**；另外还要付全局位图 CAS 争用和 allocator 选举的开销。

**路线 B——ticket 全序（断循环等待）。推荐这条。**

给每个 cohort 一个全局单调 ticket（ready 时分配）。规则只有一条：

> **核空闲时，只服务它够格参与的、ticket 最小的那个未满足 cohort。**

无死锁的论证：设 C_min 是 ticket 最小的未满足 cohort。所有够格的核都会选它而不是别人，所以它
**单调累积**、不会有核中途叛逃到更高 ticket ⇒ 必然填满 ⇒ 开跑 ⇒ 完成。更高 ticket 的 cohort
不会抢先占核，因为一个 cohort 只有在它是最小者时才被服务 ⇒ **不存在"低 ticket 还在攒、高
ticket 已开跑"**。于是任一时刻**最多一个 cohort 处于累积态**——这是**推导出来的**，不是强加的。

**它比队首互斥强在哪**：互斥只落在**累积态**上。C1 全部放置完、进入运行后就不再需要新核，
**C2 可以立刻在剩余核上开始累积**；今天必须等 C1 完成。所以路线 B 不是推翻现有设计，而是把
"一次一个"从*完成*粒度放宽到*累积*粒度 —— **队首唯一本身就是 ticket 序的退化形式**（队首即
最小者），A1 则是它的交接正确性。

**两条路线的共同硬约束：不可抢占。** 核一旦**开始执行** cohort 块就卡在用户 barrier 里，
运行时无法回收——用户 kernel 不可中断。所以只能在"尚未开始"的阶段做文章，Coffman 第三条
在这里动不了。

| | 现在（队首互斥） | A 全或无 | B ticket 全序 |
| ---- | ---- | ---- | ---- |
| 无死锁 | ✅ | ✅ 构造性 | ✅ 需全序可见 |
| 饥饿 | 不会（队首霸占） | ⚠️ 需 aging，而 aging 把 hold-and-wait 带回来 | 不会（最小者单调累积） |
| cohort 间并发 | 无（等**完成**） | 高 | **运行态可重叠，累积态互斥** |
| 新增状态 | — | 全局位图 + allocator 选举 | 每 cohort 一个 ticket |
| 与现有结构的距离 | — | 远 | **近，是现有设计的推广** |

**不要先于槽模型改造做。** 信箱模型解决"同一个核上的义务次序"，ticket 解决"多个 cohort 之间"，
两者正交**但有依赖**：ticket 的归纳同样要靠"核内 FIFO、前面的更老"，而那正是信箱模型提供的。
槽模型未统一之前上 ticket，等于在两个不稳的地基上同时施工。

路线 B 的收益是**吞吐**（cohort 运行态可重叠），**需要上板基线才能判断值不值得**。建议次序：
上板基线 → 信箱模型 → 量 cohort 串行的实际代价 → 代价显著再上 ticket。

#### C1 必然完成：这条链是错的（2026-10-27 实测证伪）

> **先读这一段再读下面的链。** 下面那条推理**只在单一执行通道内成立**，而自有槽与 mix 槽
> 是两套独立的 generation 序列，且执行器**无条件先取自有槽**
> （`aicore_executor.cpp:406-461` 取到就 `execute_task` 然后 `continue`，mix 槽扫描在
> `:472`，只有自有槽没活时才到得了）。所以"同一核上 C1 的块必先于 C2 的块执行"在
> **C1 是 Mix cohort、C2 是非 Mix cohort** 时不成立：C1 的块在该 AIV 的 mix 槽，C2 的块在
> 它的自有槽，自有槽先跑。
>
> **已在 a5 硬件上复现**：`spmd_cohort_with_ordinary`。
>
> 真正的性质不是"会不会被挤到后面"，而是**挤多久**：Mix 子任务等的不是正在跑的那个任务，
> 而是该核**还能领到的每一个**普通任务 —— 延迟等于积压深度，而积压无界。实测（最早 end
> 卡在 barrier 上限，是 AIC 子任务等不到而放弃）：
>
> - a5sim（8 AIC / 16 AIV），积压 320 个任务：最晚 start 1,100,044 us，最早 end 500,001 us
> - **a5 硬件（32 AIC / 64 AIV），积压 1280 个任务：最晚 start 1,100,078 us**，最早 end 500,002 us
>
> 两个平台相差 34 us，都约等于 20 × 50 ms 的积压量 —— 这说明它是**结构性的**，不是时序巧合。
>
> **这个构造走了弯路，值得记下来。** 最初每核只放 2 个普通任务，硬件上几微秒就跑完，
> Mix 子任务几乎不用等 —— 于是它在 sim 上红、在硬件上绿，我一度据此说"硬件不复现"。
> 错在**那个构造量的是延迟存不存在，而这个缺陷的本质是延迟有多长**。把占用改成按时间计、
> 再给每核排一队，两个平台的读数立刻一致。
>
> 同样的理由，**最初只把 `xfail` 标在 3 个 cohort 上也是高报**：换成积压驱动之后 **2 个就
> 复现**，所以它根本不依赖 "Mix + AIV-only cohort" 那个特定配对 —— **任何自有槽积压都足以
> 饿死 mix 槽**。
>
> **这是共驻违约，但不等于死锁。** 积压终会排干，Mix 子任务终会执行。所以真实 kernel
> （不会主动放弃）看到的是**无界延迟**，期间它的 AIC 子任务一直占着核空转；要得到永久挂死，
> 需要后继 cohort 也依赖被 C1 占住的核。那个形状**尚未构造**。
>
> 正确的修复不是在现有槽模型上加判据，而是**换掉槽模型**：见下面的「执行位 + 跨核信箱」。

下面是原来的链，**保留作为被证伪的记录**：

head 在"C1 全部认领完"（而非全部完成）时就推进，C2 随即成为队首并开始被认领。此时 C1 的某些
block 可能还在槽里排队未执行，所以"C1 必然凑齐"需要一条链：

- C2 的块一定在 C1 之后被认领 ⇒ 它的 generation 更大（generation 是本核单调递增的
  `dispatch_sequence`）；
- 执行取 generation 最小（约束 3.10）⇒ 同一核上 C1 的块**必先于** C2 的块执行；
- 于是 C1 的块前面只可能排着**一定会完成**的东西（普通任务或 non-sync_start 的块），
  **绝不会排着另一个 cohort 的块** ⇒ C1 凑齐 ⇒ 完成 ⇒ C2 才真正开始跑。

**这条链是约束 3.10 的唯一用户。** 谁把槽选择从"最老优先"改成 LIFO 或轮转，就静默破坏了
sync_start 的正确性：X 先跑 C2 块等 C2 齐、Y 先跑 C1 块等 C1 齐，各自的另一半压在对方的
第二个槽里。

#### 执行位 + 跨核信箱（2026-10-27 决定，取代 §8.1 第 2 条的扫槽判据）

> 本节一度叫「双发射槽模型」。那个名字把它读成一条两级发射队列，而它不是：深度是
> **1 个执行位 + 1 个跨核信箱**，信箱只为 Mix 存在。照「两个对等的槽轮流发」去实现，
> 会在「谁能写第二个槽」上重新撞回单写者问题。

上面那条被证伪的链说明问题不在判据，在**槽模型本身**。现在是按类别分的四个槽
——`slots[2]`（自有）加 `mix_slots[lane][2]`（协调者持有）——一个核最多压四个义务，
而执行器对类别有**无条件偏好**（`aicore_executor.cpp:406-461` 取到自有槽就执行并
`continue`，mix 扫描在 `:472`）。正是"按类别排序"造出了那个洞。

换成：每核一个 `running`（执行位）加一个 `pending`（跨核信箱），不分类别，
**信箱优先于自领**，`running` 空出后由 `pending` 晋升。

**「信箱优先于自领」是这次改造的目的，不是一条防御性判据。** 没有它，Mix 子任务照样排在
无界的自领工作后面 —— 那正是 `spmd_cohort_with_ordinary` 复现的积压饿死。下面「实现清单」
第 3 条是它的落点。

##### 谁能写什么

本仓没有中心调度器，每个核都跑 resident 循环，所以"调度器"要落到具体角色：

| 字 | **唯一**写者 | 条件 |
| ---- | ---- | ---- |
| `running` | **owner 自己**（自领、晋升或直填） | 自领仅当**本地完全空**（running 与信箱都空） |
| `pending.published` | **本 cluster 的协调者** | Mix 放置时要求**三条 lane 的信箱都已被取走**，全或无 |
| `pending.consumed` | **owner 自己** | 晋升之后回写，等于告诉协调者信箱空了 |

**协调者不写 `running`，即使它看到 running 是空的。** 这一条是整个模型无锁的来源：
两个字各有唯一写者，不需要原子操作。让协调者走"running 空就直接填"的捷径，就会和 owner
的自填撞在同一个字上，单写者性质立刻消失。代价只是空 running 的核多等一轮晋升。

**owner 也不写 `pending.published`，清空信箱同样不行。** 晋升之后「信箱空了」这件事由
第三个字表达：owner 把它刚取走的那个 `published` 值**原样回写**进 `consumed`，协调者放置
前判的是 `consumed == published`。少了这个字，最自然的写法 —— owner 顺手把 `published`
清零 —— 会让它有两个写者。这是**回收**，与下面「撕裂」那条讲的**发布**不是一回事，
两边都要有协议。

**回写的是值，不是一个忙闲标志位。** 标志位有**丢发布**的竞态：协调者写 `published = G`，
owner 还没观察到，标志位仍是上一轮留下的「空」，协调者下一轮又判为空，于是把 `G'` 盖在
没被取走的 `G` 上。那个 Mix 任务就此消失 —— fanin 已解、任务永不执行、没有任何核报错，
正是 `.claude/rules/running-onboard.md` 那张表里最难归因的「假死锁」签名。协调者自己记得
发过什么也补不上这个窗口：它缺的恰好是「对方取没取」，而标志位分不清「这一次取的」和
「上一次取的」。要分清就得让 owner 的写带序，哪怕只有 1 bit 翻转 —— 那已经是 generation
的退化形式，不如直接回写 64 位原值。

**`consumed` 放 `SchedulerSsbufLane::reserved`**：与 `mix_completion` 同结构、同方向、
同写者（owner 写，协调者轮询），布局零增长。

> **GM 那条「第二条线被整线回写会还原旧副本」的陷阱（约束 3.7、progress §7.10）在 SSBUF
> 上不存在。** SSBUF 的读写就是一条 volatile 访存加 `dsb` 定序（`scheduler_ssbuf.h` 的
> `scheduler_ssbuf_load_relaxed` / `_store_relaxed`），没有 cache line 发布这回事。
> 写在这里，是因为从 §7.10 过来的读者很容易把那条陷阱外推到 SSBUF，然后为了躲一个不存在
> 的问题去拆结构。

边界两条，都安全：一条 lane 只属于一个 cluster，所以 `published` 只有一个协调者会写；
lane 0 的 owner 就是协调者本人，它自己的信箱由自己写，没有跨核。**lane 0 仍然走信箱**，
不开「协调者直接填自己 running」的捷径 —— 否则三条 lane 的放置条件各不相同，全或无就要
分两种写法。

##### 信箱只为 Mix 存在，而自领不预取并不是代价

`pending` 的存在理由只有一个：**协调者不能写 `running`，需要一个地方把 Mix 子任务放下，
等 owner 自己来取。** 没有 Mix 的图里它一次都不被使用，所以它照 `has_mix` 门控（形状与
`has_spmd` 相同，§5.4），常态下一个字都不读。

于是**普通任务的深度就是 1**。今天不是：`scheduler_fill_own_slots`（`scheduler_dispatch.h`）
是一个遍历 `SCHEDULER_PENDING_SLOT_COUNT` 的循环，一趟把所有 `FREE` 槽都填掉，所以一个核
一轮能从 inbox 领走两个任务。它看起来像流水，但核一进 `execute_task` 整个调度循环就停摆
（约束 3.9），取和构造没有任何东西能与执行重叠：

```text
取A、取B → 构造A → 构造B → 执行A → 执行B
取A → 构造A → 执行A → 取B → 构造B → 执行B
```

两行逐项相等。**预取只是把同样的工作挪早，在本核上没有收益**，而代价是任务压在本核、
刚空出来的别的核领不走。§4.5 对 SPMD 的 block 已经裁过同一个问题（「一次认领多块……
负载均衡直接退化」），普通任务的两槽预取只是历史遗留，没被同一把尺子量过。

**但这条论证不能推广到 Mix 槽，两者不对称。** 自有槽上取、构造、执行三件事都在同一个核
上串行，所以预取挪不动总时长；而 Mix 的 payload 由**协调者**准备、由 **AIV** 执行
（§2.1 第 3 点：计划 D3 的「prepare 搬到 AIV 侧」未实现）。协调者给第二个 Mix 任务做认领
与构造时，那条 lane 的 AIV 正在跑第一个 —— **这是两个核真并行**，第二个位置买到的是
实打实的不空档。所以 `SCHEDULER_MIX_SLOT_COUNT = 2` 有它的理由，不是和自有槽一样的遗留。

信箱模型把这个深度**原样保留**：

```text
今天： mix_slots[lane][0] 在跑     mix_slots[lane][1] 已备好
信箱： running 在跑（Mix 子任务）  pending 信箱里已备好
```

变的不是深度，是**那两个位置不再是 Mix 专属** —— `running` 既可能是 Mix 子任务也可能是
普通任务，于是普通活不能再无限插到 Mix 子任务前面。这正是本节要解的那个饿死。
**读到「预取没有收益」时不要顺手把信箱也砍掉**：那句话只对同核的预取成立。

依赖链不受影响：完成路径解出的后继**直填进 `running`**（刚空出的就是它），这是 owner
写自己的字，不经过队列也不空转一轮。前提是信箱为空 —— 信箱非空时那个位置属于晋升，
后继照常入队。

##### 为什么不死锁

**核心性质：一个核上排在某个义务前面的，一定比它更老。** FIFO 执行使这条自动成立。

由此归纳，按"块被放置的时刻"给 cohort 排序：最老的那个，它的块在每个核上前面排的只可能是
普通任务（更老的 cohort 不存在）⇒ 普通任务必然终止 ⇒ 它的块全部进入 running ⇒ 凑齐 ⇒
完成 ⇒ 释放核 ⇒ 次老的同理。依次解开。

**gap 3 被消除**：AIV 欠着 Mix 子任务时本地不空，没有资格自己领 AIV-only cohort 的块。
**Mix↔Mix 同槽号的隐患也消失**：三条 lane 全空的要求使同一 cluster 同时最多一个 Mix
任务 pending，不存在两条 lane 挑到不同槽号。

##### 这条归纳依赖什么

| 前提 | 由什么保证 |
| ---- | ---- |
| FIFO 执行、无类别偏好 | 本次改造本身。**谁改成 LIFO 或轮转就静默破坏它** |
| 「前面的更老」 | **A1**（已合入 `ec51a1149` 之后的 `d462d5cfe`） |
| 普通任务必然终止 | 用户 kernel 的前提，不是运行时能保证的 |
| `block_num ≤ 该形状核数` | 现有 demand 校验 |

**A1 是前提而不是独立优化。** 没有它，队首在最后一块**发布之前**就推进，于是某个此刻还空的
核会先领走**新** cohort 的块进 running，老 cohort 的块随后才落进它的 pending ——
「前面的更老」当场被破坏，归纳垮掉。

##### 四个并发问题，分别处置

owner 写 `running` 与协调者写 `pending.published` **会同时发生**，但不是同一个字的写-写
冲突。真正要处理的是另外四件：

| 问题 | 是否发生 | 处置 |
| ---- | ---- | ---- |
| 同字写-写 | **否** | 前提是协调者不写 running、owner 不写 `published` |
| **次序倒置**（TOCTOU） | **会** | 见下，不致死锁 |
| **撕裂** | **会** | 必须单字发布 |
| **回收** | **会** | 第三个字 `consumed`，见上方「谁能写什么」 |

**次序倒置**：两边的前置检查都是"读—判断—写"，中间有窗口 —— owner 读到本地空、同时协调者
读到 pending 空，结果是 `(running=自领的 X, pending=C)`，而 C 可能比 X 更早被决定放置。

它**不致死锁，前提是 A1 在**：队首唯一 ⇒ 同一时刻只有一个 cohort 在被领取；A1 又保证队首在
上一个 cohort 最后一块派发完成前不动 ⇒ **两个 cohort 不可能并发获取** ⇒ 被倒置挤到前面的 X
只可能是普通任务 ⇒ 必然终止 ⇒ C 迟早进入 running。代价是集结延迟。共驻本身仍然成立：cohort
块一旦进入 running 就卡在 barrier 不退出，所以只会累积，最终全部在核上。

**撕裂**发生在晋升那一刻：owner 读 `pending` 准备搬运时，协调者可能正写到一半。解法仓里现成
——`SchedulerSsbufDispatchControl` 的 `publication` 是最后写的单个 volatile uint64，消费侧只
认它。合并槽位后 `pending` 沿用同一套协议：**先写内容，最后单字 release 发布**，owner 只在
publication 有效时才晋升。

##### 代价

**本稿初版把「流水变浅」列为这里的主要代价，那是错的** —— 上面「信箱只为 Mix 存在」已经
算过：取与构造无法与执行重叠，预取只是挪早，本核上零收益。所以取消自领预取不花钱，反而把
任务留在共享队列里给真正空闲的核。这一格原本写着「用正确性换、要实测代价」，那笔实测不需要了。

真正的代价有两条，都在 Mix 上：

- **Mix 子任务要多等一轮。** 协调者把它放进信箱，owner 下一轮循环才晋升。这是单写者的
  直接后果，换不掉 —— 让协调者直接写 running 就是上面否掉的那条捷径。
- **协调者自身卡在 barrier 时，该 cluster 的放置与完成服务停摆**（约束 3.9）。按归纳不致
  死锁，但会放大集结延迟。

##### 实现清单（按依赖序）

1. `SchedulerLocalSlotState` 从 `slots[2]` + `mix_slots[lane][2]` 合并为每 lane 一个
   `{running, pending}`，信箱带 `published` / `consumed` 两个字。容量随之重算：
   `SCHEDULER_MIX_SLOT_COUNT` 2 → 1 —— 这个 1 指的是**信箱**，lane 的第二个 Mix 位置
   由 `running` 顶上，所以每条 lane 的 Mix 深度仍是 2，**并发度没有被砍半**（理由见上方
   「不对称」一段）。`SCHEDULER_MIX_TRACKER_COUNT` 相应从 `3 × SCHEDULER_MIX_SLOT_COUNT`
   （6）收到 **2**：它数的是协调者手上同时在飞的 Mix 派发，而一个 cluster 最多一个在
   running、一个在信箱。tracker 耗尽是软失败 —— `scheduler_plan_mix_placement` 返回
   false，任务回 ready 队列下轮再试 —— 所以算紧是安全的
2. SSBUF 布局重排：`mix_dispatch[3][2]` 缩成每 lane 一个信箱加一个 `consumed` 字。
   `scheduler_ssbuf.h` 末尾那几条 offset / size static_assert 连同 `reserved` 的尺寸一起
   重算 —— 那里的注释写明了「sizing one means recomputing this」，不要假设空间就在那儿
3. 执行器去掉「先自有后 mix」的分支，改成取 `running`；两条路径现在有四个不同端点
   （task_ref 来源、payload 位置、trace 去向、generation 发布），**合并时把差异收敛到
   「取 payload」一步**，其余共用。这是改动最大的一条
4. 自领路径加「本地空」前置条件，也就是「信箱优先于自领」。这一条是目的，不是判据
5. `plan_mix_placement` 改成「三条 lane 都 `consumed == published`」的全或无；协调者去掉
   写 running 的捷径，lane 0 也走信箱
6. **完成路径只移交退槽那一半，其余不动。** 今天 AIV 执行完 Mix 子任务只往
   `lanes[lane].mix_completion` 发布一个 generation，协调者在自己的循环里轮询，既聚合
   tracker、解依赖，**也负责退槽**。新模型里 `running` 的单写者是 owner，所以退槽移交
   给它：**owner 退自己的 running、晋升信箱、回写 `consumed`**。发布 generation 与协调者
   轮询聚合那一半**原样保留，不需要新机制**。所以净改动比「把
   `scheduler_service_cluster_completion_slot` 一分为二」小得多——owner 多三件事，
   协调者少一件
7. 信箱整体照 `has_mix` 门控，形状照 `has_spmd`（§5.4）

##### 验收

`spmd_cohort_with_ordinary` 的 `cohort_count` **2 与 3 两档**都是 `xfail(strict)`。
**这个模型落地后它们会因「意外通过」而失败**，那就是要删掉标记的信号 —— 不要先删标记
再改代码。

C++ UT 至少要钉四条，它们都属于「错了不会当场报错」的那一类：

| 落点 | 测什么 |
| ---- | ------ |
| 信箱握手 | `consumed != published` 时放置必须失败——这一条同时钉住「回写值而非标志位」：协调者在上一次发布尚未被取走时再次放置，不得成功 |
| 信箱优先 | 信箱非空时自领必须被拒；`running` 空且信箱非空时，晋升先于一切 |
| 全或无 | 三条 lane 有一条信箱未被取走时，整个 Mix 放置不得发生任何一半 |
| 退槽移交 | owner 退 running 并回写 `consumed`，但**不**解依赖；协调者收到完成才聚合 tracker 并解依赖 |

#### 第 2 条：谁来判、扫哪些槽

> **这一节描述的是今天的四槽模型。信箱模型落地后它整节退役**：核一次只持有一个执行位加
> 一个信箱，而信箱优先于自领，所以「同一个核已经有这个 cohort 的另一块」这件事不再需要
> 扫槽去判 —— 本地非空就不自领，结构上就排除了。下面保留，是因为在那之前它仍然是现行
> 判据，而且「随完成自动失效」那一段对**写用例**仍然成立。

判据不需要新增状态 —— block 一直占着槽直到完成，所以"扫槽找同一个 `task_ref`"就是精确判断，
且随完成自动失效。但**扫哪些槽取决于谁是认领者**，两种情况不同：

> **"随完成自动失效"这句的另一面，构造用例时会直接撞上。** 槽是在 block **完成**那一刻
> 清掉的，所以一个立即返回的 kernel 根本不在扫描视野里停留：它的核转头就能领走同一个
> cohort 的第二块。这不是缺陷——真正 barrier 的 cohort 里，第一块在所有块就位之前不可能
> 完成，槽一直占着，判据就是精确的；一个互相不等的 "cohort" 也确实没有东西可以死锁。
>
> 但它意味着 **"每核恰好一块"这个分布只在 kernel 真的占住核时才是性质，否则是赛跑。**
> 2026-10-23 的满宽 AIV cohort 用例就栽在这里：16 个块落在 **9 个核**上，而同一断言在
> 8 核的 AIC 版上一直是绿的——后者只是块数少、赛跑赢面大。两个用例现在都改用一个先自旋
> 再 stamp 的 kernel（`spmd_cohort_full_width/kernels/aic/kernel_stamp_block_resident.cpp`），
> 把 barrier 的"占住槽"这一面补上；AIC 版此前的绿是**运气**，不是看护。

| cohort 形状 | 认领者 | 扫什么 |
| ----------- | ------ | ------ |
| AIC-only / AIV-only | **执行核自己** | 自己的 `slots[0..1]` |
| Mix | **本 cluster 的协调者 AIC** | 自己的 `slots[0..1]` **加** `mix_slots[1..2][0..1]`（三条 lane 全覆盖） |

**Mix 的判据只能由协调者做，这一点容易写错。** AIV 核执行 mix 槽里的任务时，task_ref 是从
SSBUF 的 `mix_dispatch` 读的（`aicore_executor.cpp:453`），它的 local state 里**根本没有
mix 槽的 slot state** —— 那份状态由协调者独占持有（`scheduler_ready.h:296` 的注释写明了
这件事）。所以 AIV 核无法自查"我是不是已经有这个 cohort 的块"；必须由协调者在认领时
一次性确认整个 cluster 的三条 lane 都没有它。

好在这正是协调者本来就能做到的：`mix_slots` 只有它写，所以它看到的是权威状态而非快照
（这与 §6.2 第 3 步 Mix placement 探测之所以精确是同一个理由）。

#### 直填旁路：为什么第 1 条不只是"检查次序"

执行器主循环的顺序是
`scheduler_service_cluster_completions` → `scheduler_fill_mix_after_completions` →
`scheduler_fill_own_slots`，而**直填发生在第一步内部**：一个核完成任务后，把自己刚解出的
后继直接填回刚空出的槽（commit `344fc19e` 恢复的路径，`scheduler_completion.h` 的
`resolved_ready`）。它绕过 ready 队列 —— **也绕过 SPMD 队首那次检查**。所以把队首检查排在
`fill_own_slots` 之前并不够：只要每次完成都能解出一个后继，这个核在队首检查的那一刻
**永远没有空槽**。

后果在有限 DAG 里不是死锁（链总会耗尽），但严重程度接近：cohort 的集结被拖到"图里其余
可达工作基本跑完"，而**已认领的 block 此时卡在 barrier 里自旋——那个核在 `execute_task`
内部，整个调度循环都停了**（约束 3.9），连它的第二个槽都不会再被填。真实负载上这与挂死无异。

所以第 1 条要补一句：**队首是 cohort 且其 core type 与本核相符时，跳过直填**，让那个槽空出来
给队首。这是旧稿"全局停取"的最小必要版本 —— 只针对直填、只针对 cohort、只针对匹配的
core type，而不是让整类核停止取新任务。

推论：**槽里的普通任务一定会终止** —— 任务只有在 fanin 全 DONE 之后才会被放进槽，所以它不可能
反过来依赖这个 cohort。于是有了 1~4，每个核都会在"一个普通任务的时长"之内空出槽，并在那一刻
优先认领 cohort block。**但请注意这个"一个普通任务的时长"正是第 1 条补丁买来的** ——
没有抑制直填，它会变成"剩余图的时长"。

### 8.2 停取规则是性能项，不是正确性项

「队首是 cohort 期间，该类型的核只认领 cohort block，**不从自己 inbox 取新任务**」——
注意它与 §8.1 第 1 条的补丁不是一回事：补丁只拦**直填**（因为直填连队首都不看），
这里拦的是**从 inbox 正常取任务**（那条路径本来就排在队首检查之后）。由 §8.1 的
推论，没有它 cohort 同样会凑齐，只是每个核可能先跑完手上那个普通任务，barrier 的等待因此变长。
所以它是**可选的延迟优化**：范围按 `tag` 的 core type 收窄（AIC-only 的 cohort 不冻结 AIV 核），
实测 barrier 等待成为瓶颈时再加。

> 本稿初版把它列为"唯一一处错了会静默挂死的机制"。那是错的 —— 把 §8.1 的四条判据写全之后，
> 它落在正确性集合之外。记在此处是因为"停取"听起来像安全底线，容易被下一个人重新当成必需品。

**一个 per-core 新增的延迟项（同样非正确性）**：Mix cohort 的 AIV 子任务落在 mix 槽上，而执行器
先取自有槽、再扫 mix 槽（`aicore_executor.cpp:366-445`）—— 计划 D4 的"mix 优先 + 有界让行"
**未实现**。所以 cohort 的 AIV 子任务可能排在两个普通任务之后。有界，不死锁，但会把 barrier
等待拉长，和上面那条是同一类问题、同一个处置：实测说话。

**不设超时。** cohort 已 ready 说明其 fanin 全 DONE，不依赖任何待执行任务；§8.1 的四条保证
每个核都会在有限时间内空出槽并优先认领 cohort block。cohort 必然凑齐。

TMR 的 stop-the-world / 二叉 ack 树 / coordinator / `force_gate` + doorbell / rendezvous launch
在此**一个都不需要**，理由与旧稿相同（它们服务于"没有 SPMD 互斥"的前提）。

## 9. Host 侧放行

1. **`scheduler_resident_v0_task_shape_supported` 整个删除。** 去掉 `logical_block_num == 1`
   与 `!sync_start` 之后，剩下的 `active_subtasks >= 1 && <= 3` **恒真**（mask 只有 3 bit，
   0 已被 `runtime_maker.cpp` 的乘积上界硬拒），谓词失去意义。

   **连带：`SCHEDULER_RUNTIME_MODE_LEGACY_UNSUPPORTED_SHAPE`（mode=4）一并退役。**
   本稿初版想给它换个判据保住它，那是错的——**SPMD 不存在"回退"这个选项**（用户裁定，
   2026-10-23）：AICore 调度跑不了的，AICPU 调度一样跑不了。于是 mode=4 没有任何产生者，
   留着它就是在暗示一个已不存在的选择。枚举值、`aicore_scheduler_runtime_mode_is_explicit_legacy`
   的分支、`select_legacy_scheduler` 的 `always_assert`、conftest 的 `_MODE_NAMES` /
   `_LEGACY_MODES`、两处 UT 全部删掉；**mode=3（`TaskKind::GRAPH`）是仅剩的回退**，它的选择
   在形状遍历之前（`runtime_maker.cpp` 的 `slot.task_kind == TaskKind::GRAPH` 分支），所以
   三个钉 `legacy` 的 graph 用例不受放行影响（已核实）。
2. **worker demand 收窄到只由 sync_start 贡献。** 这是放行的硬前置：

   原先对**所有**任务取 `aic_worker_demand = max(block_num)`、
   `aiv_worker_demand = max(block_num × AIV 子任务数)`，`aicore_lifecycle.cpp:242` 据此拒绝
   超出真实核数的图。对 non-sync_start 这是错的：block_num 大于核数完全正常，blocks 分波次
   跑完即可。不改就会拒掉：

   | 负载 | 原算法得出的 demand | 真实核数 | 结果 |
   | ---- | ------------------: | -------: | ---- |
   | qwen3 decode 的 `q_proj_spmd`：`rt_submit_aic_task` + `set_block_num(50)`（`decode_fwd_layers.cpp:406,415`） | **AIC 50** | 36 | **拒绝 bind** |
   | CSA，`block_num` 最大 512 | AIV ≥ 512 | 72 | **拒绝 bind** |

   第一行是查实的（不是按 Mix 形状推的）：它是 AIC-only，50 个 block 要 50 个 AIC，而全机只有
   36 个。同一文件 `:532` 还有一个 `rt_submit_task(mixed_11, …)` 的 **Mix + sync_start** 任务
   （`set_require_sync_start(true)`@521、`set_block_num(24)`）。

   改法：`const uint64_t sync_start_blocks = requires_sync_start() ? logical_block_num : 1;`
   两处 demand 都用它。sync_start 仍贡献——它才真需要 co-residency。

   **超限的 cohort 硬失败，不回退。** 本稿初版主张改成整图回退 legacy，理由是"今天所有
   sync_start 图都回退 legacy 并能跑"——对超限的图这句话不成立（用户确认 legacy 同样跑不了
   `block_num > 核数` 的 sync_start）。回退只会把一个 bind 期的清晰报错换成 45s 设备超时，
   诊断性严格变差。

   > **但真正说话的不是 `aicore_lifecycle.cpp:242`（S1 实测，2026-10-23）。**
   > 共享的 HBG orchestrator 里早就有一道**无条件**守卫
   > （`src/common/host_build_graph/host/orchestrator.cpp:815`）：任何 `block_num > 1` 且
   > `require_sync_start` 的任务，在 **submit 时**就按 `total_aiv_count`（AIV 形状）或
   > `total_cluster_count`（MIX/AIC）拒掉，报
   > `REQUIRE_SYNC_START_INVALID / require_sync_start block_num=N > limit=M (deadlock guaranteed)`，
   > 还带 hint 与文档链接。图根本到不了 `create_scheduler_state`。
   >
   > **`:242` 仍然保留，但它不是这条路径的守卫。** 它比的是**握手时实际发现**的核数，而编排
   > 用的是 orchestration 时的拓扑——两者可能不同（有核没握上手）。它是"拓扑中途缩水"的兜底，
   > 与刚退役的 `SPMD_CLAIM_INVALID_BLOCK`（和同一个值比第二次）性质不同，不要一并删掉。
   >
   > 用例 `spmd_cohort_too_wide` **成对**断言这两半：64 块 + sync_start 必须被拒、64 块无
   > sync_start 必须跑通。只断言一半没有意义——"所有任务都算 demand"会两个都拒，
   > "都不算"会两个都收。

   > **qwen 同时含 sync_start，这件事有两个后果。** 其一，它那条 sync_start 任务按本方案
   > **仍然贡献 demand**：AIC 24 ≤ 36、AIV 24×2 = 48 ≤ 72，**恰好过关** —— 它是"碰巧不触发"
   > 而不是"被放宽救下"，别把它算进放宽的收益里。其二，qwen 因此**不是一张 non-sync_start
   > 的图**，所以 §10.2 那条"S1 后对 non-sync_start 图不得出现回退日志"在 qwen 上要连 cohort
   > 路径一起看 —— 反过来说，这让 qwen 成了 S1 里同时覆盖两条路径的现成负载。

   顺带一提，收窄后的 demand 口径与旧稿 §7.2 那张"按形状取上限"的表**自动等价**
   （Mix 的 `block_num × 2 ≤ 72` 即 `block_num ≤ 36 =` cluster 数），所以旧稿要求的那个新增
   校验**不需要写** —— 现成的这条就是它。
3. 既有的两道乘积上界保留：`logical_block_num > UINT16_MAX / active_subtasks`
   （`runtime_maker.cpp:1174`）与 `expected_subtasks > UINT16_MAX`（`:1184`）。
   `total_required_subtasks` 与 trace cell 数都受它们约束。
4. **三个新字段有两处初始化点，必须写明归属。** 设备侧 `scheduler_init_data_from_layout`
   （`scheduler_types.h:1366`）先 `__builtin_memset(base, 0, total_size)`，再对每个 control
   显式写 4 个字段（`state` / `wake_list_head` / `next_waiter` / `waiting_producer`）；而每个
   任务的 `logical_block_num` 只有 host 侧 `create_scheduler_state`（`runtime_maker.cpp:1088`）
   知道。所以：`next_block = 0` / `completed_blocks = 0` 跟着设备侧那个显式循环走（与已有的
   4 个字段并列），`block_num` 由 host 写，非 SPMD 填 1。
   **不写明归属的后果会被零值掩盖** —— memset 恰好给出 0，于是"两边都以为对方负责"在今天
   能跑，直到 layout 变了或 memset 被收窄才爆。
5. `scheduler_task_ready_queue`（`scheduler_types.h:816-818`）先判 SPMD flag 再判 MIX，
   使 Mix+SPMD 也进 SPMD 队列。认领者从 metadata 读形状（§6.2 第 2 步），不需要别处再存一份。

> **准入门是唯一事实来源。** 阶段 B 删掉了 `scheduler_task_is_gang` 及其全部下游守卫
> （progress D-2），当时记下的附带后果正是此刻要处理的第一件事：放行后，一个形状错误的 SPMD
> 任务不会再被下游拒绝，而是**静默走错路径**。所以 §9 的五条里每一条都要有对应的负向用例
> （见 §10.4），不能只靠"跑通了"。

## 10. 落地顺序与验证

### 10.1 阶段划分

| 阶段 | 内容 | 出口条件 |
| ---- | ---- | -------- |
| **S0** | 基础设施：SPMD ready 队列（全局唯一，复用 `SchedulerReadyInbox`）、`inbox_base`/`holder`/`refresh`/`push_target` 四处改动、TaskControl 三个字段、`has_spmd` 门、`block_idx`+`block_num` 贯穿 claim/payload/slot/trace，外加 `TaskType` 枚举（见下）。**不改准入谓词** | **已完成**（分支 `a5-hbg-resident-spmd`，未提交）：C++ UT 317 全绿、a5sim ST 全绿。队列恒空、`has_spmd` 恒假 |

> **S0 实做时比本节原计划多了一笔：`core_type_index` → `enum class TaskType`。**
> 第 4 条队列让这个名字彻底失效 —— 0/1/2/3 是 AIC/AIV/MIX/SPMD **队列**，而
> `self_core_type` / `scheduler_core_type_index()` 仍是真正的 core type，两者共用名字又共用
> 0/1 编码。`scheduler_self_task_type()` 是它们唯一相遇处。换成枚举后编译器逼出约 60 处隐式
> 转换，其中一处是真 bug：`scheduler_resolve_completion` 的 `direct_core_type` 名义是 core
> type，UT 却传 `MIX_QUEUE`(2) 表示"不直填"。顺带把误导人的 `capacity`/`owner_count` 改成
> `inbox_capacity`/`inbox_count` —— 它们是**inbox 个数**，不是队列能排几个任务（一个 inbox
> 是侵入式链表，任务数无上限；SPMD 队列只有 1 个 inbox 正是因此才够用）。
| **S1** | 全部放行：认领、推进 head、两级计数、demand 收窄、§8.1 的四条。non-sync_start / sync_start / Mix 一并 | 新增 ST 全绿；`scheduler_mode` 断言为 resident；分布断言显示 block 真的跨 cluster；CSA 门禁转为通过 |

#### S1 的实际覆盖（2026-10-26）

##### 当前进度

**已提交：`ec51a1149` "Add: run SPMD and sync-start on the a5 HBG resident scheduler"**
（54 files, +4654/-149）。代码、12 个 `spmd_*` 用例目录、本设计文档在同一个 commit 里。

| 口径 | 结果 | 日期 |
| ---- | ---- | ---- |
| C++ UT（`ctest -LE requires_hardware`） | 317 全绿，其中 SPMD 14 条 | **10-26，重编后重跑** |
| a5sim 全量（`--manual include`） | 33 scene + 20 standalone 全绿 | **10-26，最终构建上重跑** |
| **a5 onboard，12 个 `spmd_*` 目录** | **全绿**，含 `manual` 压力档 | **10-26** |
| a5 onboard，其余 20 个 a5 HBG 目录 | **19 绿 / 1 红**：`paged_attention::Case2`（见下） | **10-26** |
| **硬件口径反证** | 3 条全部变红（见下） | **10-26** |

**"全量绿"默认不含 `manual` 档**（§10.5 要求带 `--manual include`，前几轮的全量都漏了这个）。

**H（直填抑制：cohort 真的及时集结）不再追。** 它在 sim 上造不出来——抑制开/关噪声内相同、
断言本身 flaky，见下。硬件时序与 sim 不同，所以"硬件上也造不出来"**没有被验证过**，只是
决定不花这个成本。**这一格长期留 ❌**，下面那段测量数据是将来重新讨论 §8.1 第 1 条时的
起点，不是待办。

> **硬件口径的反证：3 条全部变红，用例在硬件上确实还有鉴别力。** 这件事必须单独做，
> 因为 dcci 那个 bug 已经证明"sim 绿"有系统性盲区，同理"sim 上能反证"也不蕴含
> "硬件上还有鉴别力"——`spmd_last_block_releases` 靠非 0 块自旋 20 万次撑出观测窗口，
> 硬件主频与 sim 不同，窗口完全可能被压没，那它在 onboard 档就是个永远绿的空壳。
>
> | 把实现改成 | 用例 | 硬件上的表现 |
> | ---------- | ---- | ------------ |
> | 两级计数外层 `done == 1`（第一块即完成） | `spmd_last_block_releases` | `-7.0 vs 28.0`：block 0 写了 0.0，其余 7 块仍是哨兵 |
> | `fanin_count >= 1`（每次解依赖都入队） | `spmd_multi_fanin` | 6 个 consumer 槽错，与 sim 同形 |
> | 允许 SPMD 后继直填刚空出的槽 | `spmd_releases_spmd` | 17 个槽错，stage 1–3 几乎全是哨兵 |
>
> 三条的红形态都与 sim 上同一反证一致，说明窗口没有被硬件时序压没。反证后已全部撤回，
> 三个用例在硬件上重新变绿，`src/` 无残留。
>
> **`paged_attention::Case2` 上板红，与 SPMD 无关，但不能当它不存在。** 它报
> `FATAL: Graph Too Large!  Tasks: used=16384/16384` —— 这是容量判决，不是缺陷：
> 同目录的 `Case1` 显式写了 `"runtime_env": {"ring_task_window": 131072}`，**`Case2`
> 一个都没写**，于是吃默认的 16384，装不下它的图。
>
> 它是 `manual: True` 且 `platforms` 只有 `a5`，所以 sim 永远够不着、PR 门禁也不跑
> （`ci.yml` 的 `st-onboard-a5` 不传 `manual_mode`，默认 `exclude`）。**但 `daily.yml`
> 的 `st-onboard-a5` 传了 `manual_mode: include`，每天的全量上板扫会跑到它。**
> 所以"CI 不跑"只对 PR 门禁成立，对 Daily 不成立。
>
> 归因**已实测**，不是推断：`git stash` 掉 `src/` 的全部 S1 改动、重装后跑同一个 case，
> **失败完全一致**（同样的 `prepare_native_run -1000`、同样的 `Tasks: used=16384/16384`）。
> 所以它是既存缺陷，与 S1 无关。修法是给 `Case2` 补上 `ring_task_window`，照 `Case1` 的
> 写法——**不在本次改动范围内，没动。**

#### 测试里的等待预算一律按时间计，不要用圈数

**2026-10-27 一天之内在三处踩到同一个坑**，所以写成通则：凡是用例里"等某件事发生，等不到就
放弃"的预算，必须用**实时上限**表达，不能用轮询/自旋圈数。

- `spmd_cohort_saturation/.../kernel_cohort_barrier.cpp`：2e8 圈 → 1000 ms
- `spmd_cohort_with_ordinary/.../kernel_busy_mark.cpp`：2e6 圈 → 50 ms
- `single_block_mix/kernels/rendezvous.h`（既存用例，本次顺带修）：1e7 圈 → 2000 ms

圈数有两重不可移植，**第二重是真正咬人的那个**：

1. **跨平台**：一圈在 CPU-stub 构建与硅上代价不同。按 sim 标定的 2e8 圈在板上远超设备的
   op-execute 超时，整个 op 被杀 —— 用例死于"被杀的 op"，而不是它本该写出的毒值。
2. **跨负载**：**机器忙时一圈更贵。** 所以一个单独跑时很宽裕的预算，和别的用例同场跑会用尽。
   `single_block_mix` 就是这么挂的：它一行没被改过，只因为同一次全量里多了 1280 个普通任务和
   几个满宽 cohort，256 个任务里就有 2 个报了 rendezvous 超时。

写法仓里现成，照 `kernel_spmd_spin.cpp`：`PLATFORM_PROF_SYS_CNT_FREQ` 给每秒 tick 数，
板上走 `get_sys_cnt()`、CPU-stub 走 `steady_clock`，两边都是纳秒。

**推论：时间阈值也不要用来当判据。** 同一天还栽过一次——用"cohort 有没有在 1 秒内集结"判
共驻，结果同一个 build 在全量下判"通过"、单独跑判"失败"，因为判的是一场**和时钟的赛跑**。
共驻该从 swimlane 的区间相交来判（见 `conftest.py` 的 `assert_cohorts_were_coresident`）：
负载只改变区间的绝对位置，不改变它们相不相交。**时间上限留给"防止永久挂死"，不要留给判定。**

#### 用例清单：每个目录测什么

12 个 `spmd_*` 目录，共 24 个可执行用例（scene case + standalone test 分开计）。

| 目录 | 用例 | 这一条钉住的点 |
| ---- | ---- | -------------- |
| `spmd_multiblock_aic` | `aic_block_num_8` | AIC 多块的基本派发：每块拿到互不相同的 `block_idx`，且都看到同一个 `block_num` |
| | `aiv_block_num_16` | 认领者是 AIV 自己而非 cluster 的 AIC |
| | `aiv_block_num_128_waves` | `block_num > 核数`：demand 收窄后不再拒图，分波次跑完 |
| | `test_spmd_blocks_reach_past_one_shard` | 块真的跨 shard 铺开——全局队列绕开了 7 核可见半径 |
| `spmd_last_block_releases` | `sum_of_8_blocks` | **两级计数的外层**：只有末块解依赖。consumer 读每一块的槽，非 0 块自旋制造窗口 |
| `spmd_mix_block_closes_last` | `mix_blocks_6` | **两级计数的内层**：一个 Mix block 在它**最后一个**子任务上关闭，不是第一个 |
| `spmd_releases_spmd` | `chain_of_4_spmd_stages` | SPMD 后继**必须入队、不可直填**（没有核能独占一个全机认领的任务）；4 级链每级都让队列空一次 |
| `spmd_multi_fanin` | `fanin_4_into_spmd` | fanin=4 的 SPMD 后继**只在最后一次解依赖时入队**；4 个前驱分属 AIC/AIV/AIC-SPMD/AIV-SPMD，覆盖两条不同的解依赖代码路径 |
| `spmd_cohort_full_width` | `test_full_width_cohort_uses_every_core_once` | 满宽 AIC cohort：每核恰好一块。宽度取自拓扑，少一块就有余量、错了看不出来 |
| `spmd_cohort_aiv_full_width` | `test_full_width_aiv_cohort_uses_every_aiv_once` | 形状决定谁能认领：`mask 0b010` 的块**只有 AIV 能领**，AIC 必须拒。宽度是 AIV 数，比 AIC 多 |
| `spmd_sync_start_mix_aiv` | `mix_aiv_cohort_8` | 没有 AIC 子任务的 Mix cohort（`flags` 说 MIX 但只有两个 AIV） |
| | `test_cohort_blocks_land_on_distinct_clusters` | 该形状的放置必须**扫 mix 槽**；只扫自有槽会漏判 |
| `spmd_cohort_too_wide` | `test_cohort_wider_than_the_machine_is_refused` | 超宽 cohort 在**提交时**被拒，报错指名 demand 与核数 |
| | `test_ordinary_task_that_wide_runs_in_waves` | 同样宽度的**非** sync_start 任务照常跑——与上一条成对才有意义 |
| | `test_sync_start_on_a_single_block_is_dropped_not_refused` | 一块的 cohort 静默降级为普通任务，**不是报错** |
| `spmd_predicated` | `test_predicate_on_a_multi_block_task_is_refused` ×2（gate 两种取值） | predicate × 多 block 在 bind 期拒图；gate 取值不影响结论——影响了就说明 gate 被读过 |
| | `test_the_same_predicate_on_one_block_runs` ×2 | 同一个 predicate 在单块上正常工作——否则上一条会被"predicate 整体坏掉"满足 |
| `spmd_mixed_waves` | `waves_4x8` / `waves_4x8_shifted` | 8 种形状（普通/SPMD × AIC/AIV/Mix + 两种 cohort）成波同时 ready，交叉组合**不死锁**。两个 seed 错开形状与波边界的相对位置 |
| | `waves_40x24_stress`（manual） | 同一转换重复数百次、单波宽于整机的规模档 |
| `spmd_paged_attention` | （manual） | 真实负载形状下的 SPMD |

**这张清单读法**：每行"钉住的点"是**该用例唯一能抓到的东西**。能被别的用例抓到的性质
不写在这里——否则一条失败会指向三个地方，而这正是这张表想避免的。

| 场景 | 它唯一能抓到的东西 | 载体 | 状态 |
| ---- | ------------------ | ---- | ---- |
| AIV-only 多块 | 认领者是 AIV 而非 cluster 的 AIC；HBG 此前零覆盖 | `spmd_multiblock_aic::aiv_block_num_16` | ✅ |
| block_num > 核数 | demand 收窄 + 分波次执行 | `spmd_multiblock_aic::aiv_block_num_128_waves` | ✅ |
| block 跨 shard 铺开 | 全局队列绕开 7 核可见半径 | `test_spmd_blocks_reach_past_one_shard` | ✅ |
| AIV-only Mix cohort 放置 | **扫 mix 槽**（扫自有槽会漏判此形状） | `test_cohort_blocks_land_on_distinct_clusters` | ✅ |
| 非末块不解依赖 | 后继被提前放行读到未写数据 | `spmd_last_block_releases` | ✅ **已反证** |
| 多 SPMD 同时 ready / 由 producer 解出 / 交叉与规模 | 队首唯一、推进 head、混合形状并发**不死锁** | `spmd_mixed_waves` | ✅ |
| 两 cohort 背靠背**不互锁** | 队首唯一即互斥 | `spmd_mixed_waves`（调色板同波含两种 cohort） | ◐ §8.1 的 generation 序"C1 必然完成"链未单独断言 |
| **满宽 cohort** | 最紧的 co-residency：少一块就有余量，错了也看不出来 | `spmd_cohort_full_width` | ✅ |
| 超宽 cohort 被拒 / 同宽普通任务跑通 | §9 改过的 demand 判据，**成对**才有意义 | `spmd_cohort_too_wide` | ✅ |
| Mix 子任务完成错开 | **两级计数的内层**——block 内聚合 | `spmd_mix_block_closes_last` | ✅ **已反证** |
| 直填抑制：memo 陈旧时**不丢工作** | 两个陈旧方向各自的安全性 | C++ UT（`AStaleCohortMemo…` / `AMemoThatHasNotSeen…`） | ✅ |
| **直填抑制：cohort 真的及时集结** | §8.1 第 1 条的**目的**——没有它，cohort 要等图里其余可达工作跑完 | — | ❌ **试过，造不出来**（见下） |
| **AIV-only cohort** | `mask 0b010` + sync_start；调色板里只有 AIC 与 Mix cohort | `spmd_cohort_aiv_full_width` | ✅ **已反证** |
| **SPMD → SPMD 依赖** | SPMD 任务的 producer 也是 SPMD；`spmd_mixed_waves` 的每个波都挂在普通 join 上 | `spmd_releases_spmd` | ✅ **已反证** |
| **SPMD 任务多 fanin** | 多个 producer 解出同一个 SPMD 任务 | `spmd_multi_fanin` | ✅ **已反证** |
| **sync_start + block_num=1** | 编排侧 `block_num > 1` 才 `set_sync_start()` | `spmd_cohort_too_wide::…single_block_is_dropped…` | ✅ 但**不是"被拒"**（见下） |
| **predicate × 多 block** | 两个特性从未在一张图里碰过；bind 直接拒图 | `spmd_predicated` | ✅ 钉住现状（**既存限制**，见 `KNOWN_ISSUES.md`） |
| **onboard** | sim 是 8 cluster，硬件 32/36；时序与竞争完全不同 | 全部 12 个 `spmd_*` 目录 | ✅ **2026-10-26 首次上板，并由此揪出一个 bug（见下）** |

> **上板前这张表每一格都是 sim 口径，而 sim 看不见 cache。** 2026-10-26 第一次把
> `spmd_*` 送上 a5 硬件，12 个目录里当场红了 3 个——全是"kernel 读另一个核写的 GM"
> 的那几个（`spmd_last_block_releases`、`spmd_mix_block_closes_last`、
> `spmd_releases_spmd`），只写不读的 `spmd_multiblock_aic` 照常绿。
>
> 根因**不在 runtime，在这批测试 kernel 自己**：`dcci` 在 CCE 与 `pto/common/cpu_stub.hpp`
> 里都是**函数**，从来不是宏，而这些 kernel 写的是 `#ifndef dcci` 就把它 `#define` 成空操作。
> 这个条件**恒真**，于是上板时整套 cache 维护被编译掉：producer 从不回写，consumer 读到初值。
> 仓里另外 60 个 kernel 用的都是 `#ifdef PTO_CPUSTUB_HPP`（只在 CPU stub 下置空）——
> **是 SPMD 这批自己另起了一套写法**，不是仓库没有约定。
>
> 两个"看起来很像"的假设都被**测**掉而不是想掉：给完成路径加
> `block_idx < block_num` 的独立对账（没响）、给 bootstrap 加游标非零检查（没响）。
> 真正定性的是 chip swimlane——8 个 block 最晚结束 `…195492276`，consumer 开始
> `…195502470`，**晚 10 µs**。**解依赖是对的**，最初顺着"没等所有 block"查是查错了方向。
> 这两个诊断用的守卫事后都已撤回，`src/` 没有因此留下改动。
>
> 修法两步，都照仓里已有写法：14 个 kernel 的守卫改成 `#ifdef PTO_CPUSTUB_HPP`；4 个读前驱
> 输出的 kernel 补上读前失效（`dcci(ptr, SINGLE_CACHE_LINE)` + `dsb`），照
> `fanout_imbalance/kernels/fanout_worker.cpp`。注意**不能写成
> `cache_line_t::SINGLE_CACHE_LINE`**：这些文件自己 `#define SINGLE_CACHE_LINE 0`，宏先展开
> 会变成 `cache_line_t::0`，sim 编得过、ccec 编不过。
>
> 留下的规矩：**新 kernel 直接照抄仓里已有的前导，不要自己发明条件编译**；以及
> **每造完一个用例就上板一次**，sim 绿不代表硬件绿，这条表之前整体是 sim 口径就是代价。
>
> **这张表自己的教训：`spmd_mixed_waves` 覆盖的是"不死锁"，不是"不被拖慢"。**
> 它的 golden 顺序无关——这正是它能跑任意形状、任意规模的前提，也正因如此它对**时序**
> 完全不敏感。先前据它把"cohort 与普通任务交错"标成已覆盖是高报：那条规则
> （§8.1 第 1 条）的目的是 cohort 不被直填饿死，而饿死的表现是慢，不是错。
>
> 另外两条（SPMD→SPMD 依赖、SPMD 多 fanin）在开工前的维度分析里出现过，但**没有被带进
> 那张"不冗余场景"表**——筛选时漏掉的，不是后来才想到的。凡是依赖关系落在 SPMD 任务
> **之后**的场景都在这个盲区里：`spmd_mixed_waves` 的每个波都挂在普通 join 上，所以
> "末块解依赖"这条今天改过两次的路径，只有 `spmd_last_block_releases` 一个 AIC-only 的
> consumer 在守。
>
> **直填抑制（§8.1 第 1 条）的收益，在 sim 上量不出来。** 2026-10-23 试过一个专门的用例：
> 满宽 cohort + 每核两条自我补给的链（两个 dispatch 槽都占满）+ 一个 gate 任务把 cohort 的
> 到达推迟到链跑起来之后（否则 cohort 只是**赢在起跑**——认领排在 `fill_own_slots` 之前，
> 和链同时 ready 时它先拿到空槽，根本不存在饥饿）。把抑制编译掉再跑同一个图：
>
> ```text
> 抑制关闭: cohort (665500, 966730)   最后一层 (3054520, 5247200)
> 抑制开启: cohort (634880, 937320)   最后一层 (3095980, 5300250)
> ```
>
> 噪声范围内相同。而且那个用例**本身是 flaky 的**——换一轮跑，cohort 就落到最后一层之后去了，
> 它断言的顺序根本不稳定。已删除：交一个 flaky 用例比不交更坏，噪声会盖住真实失败。
>
> 所以这条留在 ❌，而且**结论比"没测"更强一点**：要么这个形状上直填根本没发生，要么认领
> 那一遍无论如何都能摸到每个刚空出的槽——从外面看两者一样。§8.1 第 1 条是否真的必要是
> **开放的**，上面的数据是重新讨论它时的起点。也可能只是 8 核、空 kernel 的规模下观察不到。
>
> **"已反证"是比"通过"强的状态**：故意把实现改坏，确认该用例变红。本轮对三处做过——
> 两级计数的**外层**旁路后 `spmd_last_block_releases` 报 `-7.0 vs 28.0`（consumer 在第一块
> 完成时被放行，读到 7 个哨兵）；**内层**旁路后 `spmd_mix_block_closes_last` 报
> `site=74 COMPLETION_GENERATION_MISMATCH`——注意这里**不是**数据错误：每个子任务都去关闭
> tracker 并清掉 `task_ref`，同 block 的第二个子任务当场被既有校验拦住，比数据错误早一步；
> 以及用探针确认 `manual` / 多 case 的 case 真的被执行（`waves_40x24_stress` 报 1001 个任务）。
> 对"错了结果可能仍然正确"的性质，不做这一步就只是自我安慰。
>
> 2026-10-23 补的三条也都反证过：把两级计数外层改成"第一块即完成"，`spmd_releases_spmd`
> 报 18 个槽被污染（stage 1–3 的全部 block）；把 `type == direct_type` 放宽到允许 SPMD 直填，
> 同一用例再次变红（SPMD 后继被当成单块塞进刚空出的槽，只跑了 block 0）；把
> `fanin_count == 1` 放宽成 `>= 1`，`spmd_multi_fanin` 报 6 个 consumer 槽错（任务被重复入队）；
> 把 `scheduler_shape_runs_on_core_type` 改成恒真，`spmd_cohort_aiv_full_width` 直接 poison
> （AIC 核认领了 AIV-only 的块）。
>
> **`sync_start + block_num=1` 这条的"被拒"是写错了的预期。** 编排侧
> `src/common/host_build_graph/host/orchestrator.cpp:815` 是
> `if (block_num > 1 && args.launch_spec.require_sync_start())` —— block_num 为 1 时
> `set_sync_start()` **根本不会被调用**，属性在建图之前就没了。所以
> `runtime_maker.cpp:1221` 那条 `sync-start task must have block_num > 1` 守卫，
> **从这套 API 过来的任何图都碰不到**，它是防御性 backstop 而非可观测契约。
>
> 一块的 cohort 一经派发就天然共驻，没有东西要等，静默清掉属性在语义上是对的，不是缺陷。
> 所以这一格落到 `spmd_cohort_too_wide` 里断言的是**它照常跑完**，而不是它被拒绝——
> 断言拒绝等于断言一条没有调用者的路径。

S0 单独合入的价值与阶段 0（改名）相同：它是可验证的零行为变化改动，把 S1 的 diff 压到只剩
真正的逻辑。

**sync_start 不单独成阶段的理由**：它的增量只有"认领时扫自己槽"（§8.1 第 2 条）、直填抑制
（§8.1 第 1 条）、以及可选的停取早返回 —— 与 non-sync_start 共用队列 / 认领 / 计数 /
`block_idx` 的全部代码。互斥令牌**不存在**（队首唯一即互斥，§8.1）。
§8.2 把停取规则定性为性能项之后，S1 里只有两处失败不会当场报错：

| 出错的地方 | 表现 |
| ---------- | ---- |
| 认领 / 两级计数 / 推进 head | `site=44 DISPATCH_INVALID_SHAPE`、`site=62 COMPLETION_WAKE_ALREADY_CLOSED`、或结果比对失败 |
| demand / 准入 | bind 直接拒绝，或 `scheduler_mode` 断言报 legacy |
| §8.1 的第 2 条（每核 ≤1 块） | cohort 死锁 ⇒ 45s op-execute timeout 且**无 allocator fatal**，即 `.claude/rules/running-onboard.md` 的"假死锁"签名 |
| §8.1 第 1 条的直填补丁 | **不是死锁，所以更难发现**：cohort 最终会凑齐，只是要等图里其余可达工作跑完。表现为 cohort 用例"能过但异常慢"，或在大图里撞上 45s 超时。**用例必须量 cohort 的集结时间，不能只断言结果正确** |

后两行是 S1 里仅有的两处需要靠签名/时延识别的失败，所以它们的用例（§10.4 的 cohort 满额档）
要先写。其余的都会当场报错。

### 10.2 S1 的主验收信号

**"测试绿"不能证明改动生效** —— SPMD 用例即使整图回退 legacy 也会结果正确地通过。这是本设计
最容易误判的一点，但阶段 A 已经把工具建好了：

| 信号 | 来源 | 判据 |
| ---- | ---- | ---- |
| 走的是 resident 而非 legacy | `@pytest.mark.scheduler_mode("resident")`（阶段 A2 的 conftest fixture，`tests/st/a5/host_build_graph/conftest.py`） | 每个用例无条件校验，且 `published == 0` 也算失败 |
| block 真的铺开了 | chip_swimlane 的每 block 一条记录（§7.3），distinct `worker_id` | 阈值按 run 自报的 `num_cores` 推导，照 `fanout_imbalance` 的 `test_fanout_spread_is_wide` 写法，**自建 Worker 自开 swimlane**，不能用 `if level == 0: return` 那种会静默跳过的形态 |
| 日志里不再出现回退 | `A5 HBG: retaining AICPU scheduling ... with SPMD or sync-start shape` | S1 后对 non-sync_start 图不得出现 |
| 端到端性能真的改善 | 同一用例改前（legacy）/ 改后（resident）对照 | 口径照 progress §0.5「控制变量」：同一张卡、`--rounds 100`、Trimmed Avg、剔零轮与首轮、2% 以内不算信号 |

**CSA 用例是天然的验收负载**（progress F-6）：它整图因 SPMD 回退 legacy，其门禁
`run.py l4 -d 0 --require-aicore-scheduler` 现在**必须失败**，S1 后应当**转为通过** ——
这是一个已经建好、语义明确、且不依赖新写断言的验收点。唯一阻塞是 399 MiB 冻结数据缺失，
缺数据时 pytest 明确 skip，不会把跳过当成功。

### 10.3 必须新增的 C++ UT

`tests/ut/cpp/a5/runtime/host_build_graph/` 现有 10 个文件 144 个 `TEST`，改 `runtime/` 下的
C++ **必须跑它**（`.claude/skills/testing/SKILL.md:146` 给出命令本身；`:143` 那行把
`src/{arch}/runtime/<rt>/*` 的 scope 定为单 runtime pytest，要靠 `:148`「Mixed → 升到最宽
scope」才推到 C++ UT —— 所以这是**推出来的**而不是直接写明的，但 progress §7.6 记过一次只跑 sim ST 就
提交、结果 UT 编不过的教训）。SPMD 的大部分不变式在 UT 里比在真机上好守得多：

| 落点 | 测什么 |
| ---- | ------ |
| `test_hbg_scheduler_contracts.cpp` | **S0 已做**：`inbox_base(SPMD) == 144` 且不与 MIX 撞车、`inbox_capacity(SPMD) == 1`、`INBOX_CAPACITY == 145` 与补到 SPMD 的 tiling / contiguous static_assert；SPMD 无 directory 行；`scheduler_task_ready_queue` 中 SPMD 压过 MIX；`scheduler_self_task_type` 不映射到 SPMD；trace 下标按 block 步进且 block 0 与改前逐位相同；`SchedulerTaskControl` 三个新字段 `offsetof < 64` 且 `sizeof == 128` 不变；单块任务是「第 0 块共 1 块」 |
| `test_hbg_scheduler_ready.cpp` | SPMD 队列的 push target 恒为 0；`holder(3, *)` 对两种 core type 都为真；`refresh` 对 queue 3 不碰 directory；**peek 不推进 head**（读完 `endpoints` 后队列状态不变） |
| `test_hbg_scheduler_dispatch.cpp` | 认领循环（并发 CAS 下 `block_num` 个 block 恰好各被认领一次、无重无漏）；**先探放置后 CAS**（构造一个放不下的 Mix 场景，断言 `next_block` 没动）；拿旧 head 去认领一个已认领干净的任务应无副作用；**全部认领完才推进 head**，推进后下一个 SPMD 任务成为队首 |
| 同上 | 两级计数：`block_num - 1` 次完成不得置 DONE、不得解依赖、`pending_completed` 不增；第 `block_num` 次三者同时发生一次 |
| 同上 | cohort：同一核第二次认领同 `task_ref` 必须被拒（§8.1 第 2 条）；**队首是 cohort 时完成路径不得直填**（§8.1 的直填旁路——构造"完成后解出一个同类型后继"的场景，断言那个槽留给了队首） |

UT 覆盖不到的只有两类：`aicore_executor.cpp`（**C++ UT 不编译它**，设备侧由 ccec 编）和真实
并发时序。前者靠 `pip install --no-build-isolation -e .` 之后的 sim/onboard 兜，后者靠 §10.4
的 manual 规模档。

### 10.4 ST 的家底与缺口（已调查，2026-10-22）

> **S1 实做修正（2026-10-23）。** 本节下面的调查结论仍然有效，但开工后有四件事与它不同，
> 其中两件改变了后续用例的写法：
>
> 1. **`mix_payloads` 数组的 context 指针从未被初始化** —— 既有缺陷，S1 发现并修复。
>    只有每核自有槽的 payload 预填了 `args[48]/[49]`（Local/GlobalContext），mix 槽那一份没有。
>    于是任何落在 mix 槽上的子任务一调 `get_block_idx()` / `get_block_num()` /
>    `get_sub_block_id()` 就解引用空指针（sim 上表现为 host 段错误，因为 kernel 是 dlopen 在
>    host 跑的）。一直没被发现，是因为现有 HBG Mix kernel 没有一个读 context；**而 Mix SPMD
>    必须读 `get_block_idx`**，所以不修这条路根本走不通。a2a3 的 HBG 没有 mix payload 数组，
>    不受影响。
> 2. **a5sim 是 8 cluster / 24 核**（8 AIC + 16 AIV）。本机 a5 是 32 cluster，别的机器 36。
>    用例规模一律由 `rt_available_cluster_count()` / `rt_available_aiv_count()` 在编排里推，
>    或取与拓扑无关的值；**Python 侧没有核数接口**，要拿只能从 swimlane 的 `metadata` 读。
> 3. **普通 SPMD 任务的 block 分布不确定。** 同一张图量到过 8 个核，也量到过 7 个——一个核
>    会在相邻轮次里拿第二块而另一个核一块没拿到。设计只承诺"每核每轮至多 1 块"，从不承诺
>    均分，所以铺开断言只能是**下限**。而且在 AIC 上测不出来：sim 只有 8 个 AIC，一个
>    directory shard 容 7 个 owner，余量只有 1 个核。用 AIV（sim 16 个）才有余量。
> 4. **cohort 的放置断言反而是确定性的。** 规则 2（每核每 cohort 至多一块）是硬约束而非调度
>    偏好，所以"N 块占据 N 个互不相同的 cluster"要么成立要么不成立，不会抖。这使得 sync_start
>    的放置**不需要**写一个会挂死 45s 的 barrier kernel 来验证——TMR 的 sync_start kernel 里
>    本来也没有 barrier，没有现成的可借。
> 5. **`SPMD_CLAIM_INVALID_BLOCK` 已删除。** 它判 `block_num == 0 || > UINT16_MAX`，而
>    `runtime_maker` 在 host 侧已经硬拒这两种情况，设备侧永远不可能看到——不可达的守卫
>    比没有守卫更坏，它暗示了一种不存在的失败。`SPMD_COMPLETION_OVERRUN` 保留：
>    `completed_blocks > block_num` 是计数逻辑自身出错的签名，不是输入问题。
>
> **S1 至今改出的缺陷（两个，归属不同）：**
>
> | 缺陷 | 谁引入的 | 为什么之前没暴露 |
> | ---- | -------- | ---------------- |
> | mix payload 的 context 指针未初始化 | **per-core 改造**（建 `mix_payloads` 时漏了），不是 SPMD | HBG 现有 Mix kernel 一个都不读 context；而 Mix SPMD kernel 必须读 `get_block_idx` |
> | 推进 head 缺身份校验 | **SPMD 方案本身**，而且 §6.3 明确论证过它安全 | 只有队列里排着第二个 SPMD 任务时才可能发生；前面 5 个用例全是单 SPMD 任务 |
>
> 由此得到一条判读规则：**设计里凡是用"因为是 CAS / 原子"收尾的论证，都按未验证对待。**
> 同型的只有 §6.3 一处（已修）；§4.2 靠的是 CAS **失败**（安全侧），§6.2 的放置探测靠的是
> 协调者独占写 mix 槽，都不是从原子性推出的。

> **用户决定：先把功能做好，暂不移植 TMR 用例。** 本节是调查结论的留存，供移植阶段开工时
> 直接取用，**不是 S1 的前置条件**。但其中「S1 必然变红的 4 个用例」是 S1 自己要处理的。

#### 家底：HBG 侧没有一个能在 resident 上跑多块

`tests/st/a5/host_build_graph/` 下 `block_num > 1` 的用例**总共 4 个，全部钉死
`scheduler_mode("legacy")` 或 manual**：

| 用例 | 形状 | 状态 | S1 之后 |
| ---- | ---- | ---- | ------- |
| `spmd_legacy_fallback::TestSpmdLegacyFallback` | AIC-only `block_num=2` | 非 manual，钉 `legacy` | **变红**（见下） |
| `single_root::TestHbgSingleAicSpmdLegacyFallback` | 同上 | 非 manual，**无 marker** | 行为改变但不报错 |
| `graph_execution::...MixSpmd` | Mix + sync_start，`block_num = cluster 数` | 非 manual，钉 `legacy` | **变红**；但 `TaskKind::GRAPH` 仍回退 legacy（§13），所以它**应该继续绿** —— 要核实 |
| `graph_sync_start_early_dispatch` | Mix sync_start 满 cluster 宽 | 非 manual，钉 `legacy` | 同上 |
| `spmd_paged_attention` | Mix `block_num=24` | **全部 manual**，仅 a5 | 不在默认门禁里 |

**`spmd_legacy_fallback` 连"真的跑了 2 块"都证明不了**：它的 kernel 只写 `out[0]=42`，
两块写同一地址，所以 golden 对块数和块序都不敏感。它唯一守住的是那个 `legacy` marker。

**S1 的处置**：按 §9 第 2 条把它改向到"`block_num` 超过本机该形状核数的 sync_start cohort"
—— 那是 mode=4 唯一剩下的产生者。`conftest.py:48` 的 `_LEGACY_MODES = {3, 4}` 同时接受两种
legacy，改向时顺手收紧成只接受 4。`single_root` 那个继承版按 progress 的记录本来就守不住
（父类先跑时它绑到父类的单块图），一并删掉。

#### 三条核心性质：HBG 没有，TMR 也没有

| 性质 | 为什么现有用例守不住 |
| ---- | -------------------- |
| **非末块不得解依赖**（§7.2） | 所有 golden 只看最终输出。提前解依赖只在"后继读前驱全部块的输出"时才暴露，而现有用例的后继要么不存在、要么不读 |
| **两级计数**（Mix 的 block 内 subtask + block 间 block） | 同上，且需要让同一 block 内 AIC 与 AIV 完成时间错开才测得到 |
| **block 真的跨 cluster 铺开** | `graph_execution` 只断言"每个 `block_idx` 出现一次" —— 所有块挤在一个核上也能过 |

这三条**必须新写**，是移植阶段的最高优先级。前两条是计数逻辑最容易写错的地方，而且错了之后
**结果可能仍然正确**。

#### 可移植清单（TMR 侧，9 个低到中成本）

kernel 基本不动（HBG 已有先例直接引用 TMR 的 kernel），编排各改三处：`Tensor` 类型、
`set_core_num` → `set_block_num`、`layout` 汇报方式。最对口的两个：

| TMR 用例 | 直接对应本设计的什么 |
| -------- | -------------------- |
| `spmd_multiblock_mix` | 5 个无依赖 Mix 任务同时 ready，`block_num ∈ {2,8,12,24,48}` —— 队首唯一 + 超 cluster 数 |
| `spmd_batch_dispatch_oob` | 2 个 `block_num=48` 背靠背 —— 队首唯一的核心场景 |
| `spmd_multiblock_aiv` | AIV-only 多块（HBG 完全没有），最大 96 |
| `spmd_starvation` | 18 个普通 Mix + 2 个 sync_start Mix 交错 —— SPMD 混 sync_start 不饿死 |
| `spmd_sync_start{,_aiv,_edge}` | sync_start 各形状；`_edge` 是满额档候选 |
| `spmd_sync_start_stress` | 54 任务 / 6 轮，kernel 近空 —— **最适合做调度开销基线** |

**不要移植** `spmd_sync_start_mix_spill`（依赖 TMR 独有的 pre-stage / rendezvous 机制，HBG 下
语义不同）与 `spmd_sync_start_early_dispatch`（HBG 已有 Graph 版）。

仍缺、TMR 也没有的档：`block_num = 7`（跨 cluster 非整数倍）、AIV-only 128、Mix 64、
**AIC-only 多块**（TMR 一个都没有）。

#### 性能对照有一个结构性障碍

`conftest.py` 的 `scheduler_mode` **是断言，不是切换开关**。S1 之后 SPMD 不再回退，
**同一个二进制跑不出 legacy 那一侧**。所以对照只能**跨 commit**做（S0 的 HEAD vs S1 的 HEAD），
这与 progress §0.5 要求的"同一张卡、同一时段、自己跑的 before/after"仍能满足，但必须记清
是两个构建而非一个开关。

另外唯一能自动跑的 `spmd_paged_attention` 只有**一个** SPMD 任务，调度开销被计算完全淹没 ——
它测的是单任务多块吞吐，不是本设计关心的东西。调度开销要靠移植后的 `spmd_sync_start_stress`
（54 任务、kernel 近空）。

#### 两个未查实的问题（开工前要确认）

1. **HBG 里多个无依赖多块任务是否真的同时 ready** —— 取决于入队时机。这直接决定"队首唯一"
   用例怎么构造：若它们不是同时 ready，用例就退化成串行提交，守不住队列排队。
2. **大 `block_num` 开 swimlane 会不会超时** —— `fanout_imbalance` 在 1024 任务时撞过
   scheduler timeout（记在该用例注释里），所以 `block_num = 512` + swimlane 的分布断言
   规模要先试探。

### 10.5 回归口径

```bash
# 改了 C++ 之后,顺序固定(progress §0.5)
pip install --no-build-isolation -e .
cmake --build tests/ut/cpp/build -j 24 && ctest --test-dir tests/ut/cpp/build -LE requires_hardware
LD_LIBRARY_PATH=/opt/gcc-15/lib python -m pytest tests/st/a5/host_build_graph --platform a5sim -q
```

**性能/规模档必须带 `--manual include`**：progress §7.10 那个 MPMC 真机回归正是在 manual
规模档上暴露的，pytest 默认口径完全覆盖不到。SPMD 的并发度比它更高，这一条尤其适用。

## 11. 风险

| 风险 | 说明 / 对策 |
| ---- | ---- |
| **队首轮询开销** | 每核每轮一次 `ld_dev` 读 `endpoints`，**外加一次 metadata 的 cache-line observe**（§6.2 末）。`has_spmd` 门使无 SPMD 的图完全不付（§5.4）；有 SPMD 时靠核本地的 head memo 让稳态退回一次 `ld_dev`。最糟的场景是 AIC-only 大任务占队首数秒，72 个 AIV 核持续付而一块也拿不到。若实测仍显形，退路是降频到每 N 轮一次 —— 代价是认领延迟，不是正确性（O-7） |
| **认领 CAS 争用** | `block_num = 512` 时约 96 核在一个 `next_block` 上 CAS。量级与 ready `endpoints` 的 CAS 相同（已在用），但 `ld_dev` 严格串行（见 `.claude/rules/ascend.md`），需实测（O-8）。**不能靠"一次认领多块"缓解** —— 那条已被负载均衡否掉 |
| **非末块误解依赖** | 表现为 `site=62`，不是静默错误。§7.2 的不变式 + UT |
| **第二条线陷阱复发** | 任何新增的跨核 RMW 字段若放错线，失效形态是"旧值被还原成历史上合法的值"，极难定位（progress §7.10 造了七个观测仪器，六个是坏的）。TaskControl 第一条线要写明"不得整线发布"，并配 `offsetof < 64` 的 static_assert |
| **队首长期占用** | 队首在 block 全部认领完前不让位，大任务可达秒级（§4.4）。只影响其他 SPMD 任务 —— 这是"一次一个"本来就接受的串行。多 SPMD 排队用例（§10.4）守住"不丢任务、不乱序" |
| **铺开与收束的不对称** | SPMD 的 block 经全局队首发现，绕开了 per-core 队列 7 核的可见半径，能摊到全机；但它的**后继**走回普通队列 —— 只有最后一块 `resolve_completion`（§7.2），把全部 waiter 推给**那个核自己 cluster** 的同 type peer，再往外只能靠同 shard 的 7 个核偷。于是占用形态是宽 → 窄 → 宽，且"最后完成的是哪个核"随机，后继落到哪个 cluster 不可预测。**后继数量与 `block_num` 无关**（那是图的性质），所以这不是 SPMD 放大了扇出，而是它让本来就存在的收束（`fanout_imbalance` 量到的 14/96）前后反差变显眼。不在本设计范围内修；记在此处是因为 §10.2 的分布断言会先在这里显形，别误判成 block 没铺开 |
| **局部状态膨胀** | `SchedulerLocalSlotState` 16 → 24 字节，8 个 slot 共 +64B。executor 的 spill-to-fault 前科（`aicore_executor.cpp:119-120`）说明这不免费。退路：压 `state` + `subtask_slot` 进一个字节 |
| **swimlane 下游** | 同一 task_id 多条记录（§7.3）。converter 与 `dfx/chip_swimlane` 用例需同步核对，否则 DFX 产物结构变化会被当成失败 |
| **demand 放宽的副作用** | 放宽后 `aic/aiv_worker_demand` 只反映 sync_start。lifecycle 的那条检查注释与日志文本要同步改，否则它说的和它做的不是一回事 |
| **Mix cohort 的 barrier 等待** | §8.2 末尾那条延迟项。有界不死锁，S1 实测决定是否补 D4（与 O-6 同一处置） |

## 12. 开放项（本稿**没有**定下来的事）

以下**两条**是设计选择而非查实结论。写在这里是为了下一个人不要把它们当既定事实接着往下盖。

> **已定（用户决策）**：
> ① sync_start 与 non-sync_start **同期做**，Mix SPMD 与非 Mix **同期做**。
> ② **每核每轮只认领 1 个 block** —— 一次认领多块虽能减少 CAS 轮数，但会让先空出来的核把
> block 抢走、后空出来的核无事可做，**负载均衡直接退化**。CAS 轮数可以忍，不均衡不可以。
> ③ SPMD ready 队列**全局唯一、MPMC**，不按 core type 或 rank 分片（§4.2）。
> ④ **同时只有一个 SPMD 任务处于认领中**，由队首直接承担，**不设独立的认领槽**（§4.1、§4.2）。
> ⑤ **「一次一个」是硬性指标，不是待权衡项**（用户决策，2026-10-23）。它把 SPMD 任务之间
> 压成严格串行，而 qwen 一层内有数十个 SPMD 任务、CSA 整图 60 个 —— 若它们大多相互独立，
> 这就是方案的主要吞吐瓶颈。**仍然这么做**：先保证功能，可并发的 block 认领留作后续的
> 独立设计。所以不要把"队首唯一"当成可以顺手放宽的性能项去改 —— 它同时承担互斥、
> 认领失败的处置、cohort 不饿死三件事（§4.2），放宽等于重做 §6~§8。
> ⑥ **非 SPMD 任务不设哨兵**：它就是「第 0 块，共 1 块」。初版的
> `SCHEDULER_BLOCK_IDX_NONE = UINT16_MAX` 已撤销，判据是 `block_num > 1`（§7.1）。
> 本稿初版曾设想一个 8 entry 的 claim board，已整体撤销 —— 它的全部收益都建立在"多个任务
> 并发认领"之上，而那条放宽未经同意、也没有把"认领失败怎么办""cohort 如何不被饿死"设计完整。

| # | 开放项 | 现状取值与依据强度 | 怎么定 |
| - | ------ | ------------------ | ------ |
| O-6 | **从 inbox 取新任务的停取规则做不做** | §8.2 定性为性能项，**首版不做** | barrier 等待若在 cohort 用例上成为瓶颈，再按队首任务的 core type 收窄着加。同一处置适用于 D4 的 mix 让行。**注意这条不包括 §8.1 第 1 条的直填抑制** —— 那条是必做的 |
| O-7 | **`has_spmd` 为真时的每轮轮询开销** | 门本身是必须的（§5.4）；§6.2 末的 head memo 让稳态只付一次 `ld_dev`，**两者都未实测** | 先确认 memo 真的把 metadata 读压到了 head 变动那几轮。若仍显形，退路是降频到每 N 轮读一次 —— 代价是认领延迟，不是正确性。看小图档：`multi_core_dag` 的 `mixed_chain_1` / `mixed_chain_2`（都标了 `"manual": True`，采集要带 `--manual include`；progress 记的固定开销地板约 124~128 us） |
| O-8 | **`next_block` 的 CAS 争用** | 量级与现用的 ready `endpoints` CAS 相同，**未实测** | `block_num = 512` 时约 96 核抢一个字。用 CSA 量 CAS 重试数。**不能靠"一次认领多块"缓解** —— 那条已被负载均衡否掉（见上方已定 ②） |

## 13. 换窗口接手须读

本文自包含到可以照着开工，但有三件事在它之外：

| 事项 | 在哪 |
| ---- | ---- |
| **环境与复现**（venv、`source set_env.sh`、`LD_LIBRARY_PATH=/opt/gcc-15/lib`、改 C++ 后的固定三步、onboard 跑法、性能采集脚本与两个坑） | [a5-hbg-per-core-scheduler-progress.md](a5-hbg-per-core-scheduler-progress.md) §0.5，**自包含，照着走即可** |
| **per-core 改造的决策与教训**（D-1~D-9、F-1~F-6、§7.9 的 FIFO 返工、§7.10 的 MPMC 回归与"七个坏掉的观测仪器"） | 同上。本文多处引用，不重复 |
| **基线分支** | `a5-hbg-resident-spmd` @ `60b7edd1`（S0），其基线是 `30e5302d`（per-core 改造完成点，**不是 main，也不是 `feat/fully_distributed_within_core`**） |

开工前的状态核对：

```bash
cd /data/disk2/pyptouser/chenpeng/workspace/simpler
git log --oneline -1 a5-hbg-resident-spmd    # 应为 60b7edd1（S0）
git worktree list                             # 本设计的只读 worktree 可能还在 .claude/worktrees/spmd-read
source .venv/bin/activate && pip install --no-build-isolation -e .   # HEAD 动过就必须重装
```

**S0 已落地，S1 尚未写一行代码。** 文档里所有 `file:line` 原按 `872591b7` 核对
（2026-10-21，含 9 处行号修正）；S0 之后 `scheduler_ready.h` / `scheduler_types.h` 行号已明显
漂移，且 `core_type_index` / `scheduler_ready_queue_capacity` 等名字**已不存在**（见 §10.1 的
注），引用前先 grep。

### 已定的决策（不要重开）

见 §12 开头的「已定」列表。其中两条最容易被"优化"掉，所以在此重复：

1. **同时只有一个 SPMD 任务处于认领中**，由队首直接承担，**不设独立认领槽**。
   本稿初版曾设想 8 entry 的 claim board，**已整体撤销** —— 它的收益全建立在"多任务并发
   认领"上，而那条放宽会引出"认领失败换不换任务""cohort 如何不被饿死"两整套设计。
2. **每核每轮只认领 1 个 block**。一次多块能减少 CAS 轮数，但先空出来的核会把 block 抢走，
   **负载均衡直接退化**。

### 还没做的事

| | |
| --- | --- |
| 代码 | **S0 已完成并提交**（`60b7edd1`，分支 `a5-hbg-resident-spmd`，基线 `30e5302d`）；S1 未开始 |
| 用例 | 按用户决定**先做功能、暂不移植**（§10.4 存了调查结论） |
| 开放项 | O-6 / O-7 / O-8 全是实测项，S0 不碰 |

**S1 开工第一件事**：§6.3 的 `placement = nullptr`、§6.2 末的 head memo，这两条都是 S0
之后新加进本稿的，而且都属于"错了不会当场报错"的那一类 —— 前者是假死锁签名，后者是
持续的轮询开销。先把它们和 §8.1 的四条一起落在代码里，再谈认领逻辑本身。

## 14. 不在范围

- **`TaskKind::GRAPH`**（graph execution）仍整图回退 legacy（`runtime_maker.cpp:1105-1111`）。
  task_ref 今天的编码是 GLOBAL 任务的表下标，SUB_TASK 没有表槽位 —— 阶段 0 的三个访问器就是为
  那次扩展留的接口，与本设计无关。
- **SPMD + predicate**：host 硬拒（§2.1 第 2 点），本设计不放开。
- **a2a3 的 HBG resident 路径**：它没有 resident scheduler。
- **"resolve 时直接把任务放到队首"优化**（§4.1 末）、**停取规则**（§8.2 / O-6）与
  **D4 执行流让行**（§8.2 末）：三者都要等 S1 的实测数据说话。
- **D2 的 SSBUF cluster 内快通道**：与本设计正交。注意它会重新定义 SSBUF 里现在空着的那
  576 字节（`scheduler_ssbuf.h:96-97` 的两条 static_assert 守着），**SPMD 不占用那块空间** ——
  SPMD 队列与认领游标都必须在 GM，因为认领者跨 cluster（约束 3.6）。
