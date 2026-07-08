# 原子访问 Minibench —— 跨核共享数据结构的竞争防护

本文档面向 **fully_distributed_within_core** 运行时（SPMD 全分布式 runtime，编排/调度/执行
全部跑在 AICore 上、AICPU 不参与关键路径）在 **A5 平台**上的落地，目标是把"哪些数据结构被
多核并发访问、其访问过程如何、如何保证不产生跨核数据竞争"讲清楚，并为每一类共享结构配一组
**minibench**（小型压力/差分基准），把竞争问题在上真机之前就用可复现的判据钉死。

- 系统级设计与并发纪律的完整论证见 [fully_distributed_within_core.md](fully_distributed_within_core.md)
  （下文以 §号引用其章节）。
- 硬件内存模型与缓存一致性契约见 [hardware/chip-architecture.md](hardware/chip-architecture.md)、
  [hardware/cache-coherency.md](hardware/cache-coherency.md)。
- 引用的实现主体是 `src/common/runtime/fully_distributed_within_core/dist_engine.cpp`
  （private / shared 两种 TensorMap 模式统一在此），以及每平台的一致性/原子 seam
  （a5 落地契约见 fully_distributed_within_core.md §14）。

---

# 第一部分 — NPU SoC 内存模型

## 1. 三层内存域

分布式 runtime 的所有状态最终落在下面三个**可见域**之一。判定一个字段"能不能被别的核看到、
要不要加原子/一致性纪律"，第一步永远是先确定它在哪一层。

| 层级 | 物理位置 | 可见范围 | 谁能寻址 | 本 runtime 中的载体 |
| ---- | -------- | -------- | -------- | ------------------- |
| **GM（全局共享）** | 全局可寻址的共享内存后备——**不限定介质**：A5 上是共享 HBM（经各 core 的 L2 访问），未来芯片可用大容量 SRAM 等替代（见 §1.1） | 一个 `device_id` 内的**所有** AICPU + AICore | AICPU（分配）+ 任意 AICore（经指针） | `DistGlobal` 段（cursor / flags / frontier / shared_map / heap …） |
| **线程组共享（block-shared）** | 仍在 GM，但只被同一物理 block 的核并发访问 | 一个 block（1 AIC + 2 AIV，见 §1.2）内的核 | 该 block 的 AIC/AIV 核 | `DistGlobal::blocks[block_id]`（`BlockWon` 投递表） |
| **线程私有（thread-private）** | 两类：① 核内/核近私有存储（含未来的 thread-private 3D DRAM，见 §1.3）；② GM 上单一 owner 独占的一段 | 仅该 AI core worker | 仅该核 | ① AIC 的 L1/L0A/L0B/L0C、AIV 的 UB（+ 未来的私有 3D DRAM）；② `DistGlobal::cores[core_id]`（`DistCore`） |

> **术语对齐（thread group ↔ block）。** 用户所说的"1 AIC + 1 AIV 组成一个 thread group"，在
> A5 上对应硬件把 **1 个 AIC + 2 个 AIV 固定绑定为一个 block**（§3.1）。MIX 任务的最小共同所有权
> 配对（1C+1V）确实只用到 AIC + AIV0 这**一对**核，AIV1 空闲；但 block-共享内存（`block.won`）是
> **以 block 为粒度**分配的（`blocks[block_id]`），2V 任务时 AIV0/AIV1 都会来抽取。下文统一用
> "线程组 / block"指这一硬件绑定单元，用"配对（1C+1V）"指 MIX 最小协作对。

```text
┌──────────────────────────── Ascend 芯片 (一个 device_id) ───────────────────────────┐
│                                                                                      │
│   ┌── AICore cluster / block 0 ──┐   ┌── block 1 ──┐        ┌── block b ──┐          │
│   │  AIC        AIV0     AIV1    │   │  ...        │  ...   │  ...        │          │
│   │ L1/L0*      UB       UB      │   │             │        │             │          │
│   │ [+私有3DDRAM: Solomon/WSE]   │   │             │        │             │          │
│   │  └────── L2 (cluster 内共享) ─┘   └─────────────┘        └─────────────┘          │
│   │       ▲ 线程私有：L1/L0/UB（+ 未来私有 3D DRAM），核内、不可跨核寻址            │
│   └───────┼──────────────────────────────────────────────────────────────┐         │
│           │  线程组共享：blocks[block_id].won（GM，仅本 block 并发）        │         │
│           ▼                                                                ▼         │
│   ┌── GM（全局可寻址；后备介质 = A5:HBM / Solomon:大容量 SRAM …）──────────────────┐  │
│   │ DistGlobal: cursor[] · flags[] · frontier · vend[] · shared_map · heap · …    │  │
│   │            cores[core_id]（每核私有，但物理在 GM；未来可迁入私有 3D DRAM）     │  │
│   └──────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.1 GM（全局共享）

- GM 是一个**语义概念**——"一个 `device_id` 内所有 AICPU / AICore 都能寻址的共享内存后备"——
  **不绑定具体物理介质**。A5 上它由**共享 HBM** 承载（经各 core 的 L2 访问，
  [chip-architecture.md](hardware/chip-architecture.md)）。
- **前瞻（Solomon 等未来芯片）**：GM 不一定是 HBM。**Solomon 芯片可能去除 HBM**，改由系统提供的
  **大容量 SRAM** 承载 GM 空间。对本 runtime 而言这**不改变任何软件契约**——`DistGlobal` 段仍是
  "AICPU 分配 + 经 base 指针下发 + 各核经 `offsetof` 访问"的同一套模型（§13）。真正会变的只有
  §2 的**一致性/原子 seam**（`pto_gm_alloc`、`dcci` 失效/刷出、硬件 GM 原子）需要针对新介质重新
  落地：SRAM 后备的 GM 可能有不同的缓存层次与一致性行为（例如若 SRAM 直接可寻址、无 L2 缓存旁路，
  则部分 `dcci` 可退化为空操作，但**必须逐芯片验证**，不可假设）。介质替换属"换 seam 实现"，不属
  "改数据结构或访问过程"。
- 全部**跨核共享**的编排状态收拢进**唯一一份** `DistGlobal` 结构，由 AICPU 在 register 阶段分配、
  初始化、刷出，把段基址经 `runtime->dist.global_data_base` / `rt->dist_global` 下发；AICore 侧只能
  经 base 指针 + `offsetof` 访问，**不产生任何进程全局符号**（CCEC 约束，§13）。
- **关键**：GM 是"全局可寻址"，但**不等于"跨核自动可见"**——见 §2。这一点与后备介质是 HBM 还是
  SRAM 无关：只要多核经各自缓存访问共享后备，跨核可见性就仍需显式发布/观察纪律。

### 1.2 线程组共享（block-shared）

- 物理仍在 GM，但语义上只由**同一物理 block（1 AIC + 2 AIV）**内的核并发访问。
- 唯一的 block-共享结构是 **`block.won` 投递表**（`DistGlobal::blocks[block_id]`）：anchor 赢得一个
  多核任务（MIX / 2V）后，把其余激活槽子任务**推送**进本 block 的投递表，follower 核**异步抽取**
  属于自己 lane 的项（§3.1）。以任务 id 为键，`remaining` 计数驱动单一完成标志。
- 因为访问者被限制在一个 block 内、且条目以 lane 定向，竞争面比全局 GM 结构小得多，但**仍是
  真正的跨核并发**（anchor 写、follower 读；`remaining`/`state`/`drained[]` 上有 RMW），需要原子 +
  一致性纪律。

### 1.3 线程私有（thread-private）

"线程私有"是一个**归类**，判据是"只被单个 AI core worker 访问、对其它核不可寻址（或虽可寻址但
按约定不被别核触碰）"。按此判据，下面几类都是 thread-private：

1. **核内片上私有存储**：AIC 的 L1 / L0A / L0B / L0C，AIV 的 UB。kernel 计算的中间数据在这里，
   **对 host / AICPU / 其它核都不可寻址**。它天然无跨核竞争，不在本文 minibench 范围内（但它是
   "为什么产出数据必须经 GM + dcci 才能被下游核读到"的原因）。
2. **前瞻（Solomon / WSE 芯片）的 thread-private 3D DRAM**：随着芯片堆叠工艺演进，未来的 Solomon
   与 WSE（wafer-scale engine）芯片会出现**每核私有的 3D DRAM**（堆叠在核上/核近、只服务于本核的
   大容量私有内存）。尽管它容量远大于 L1/UB、介质是 DRAM 而非 SRAM，但**只要它对其它核不可寻址、
   只服务单核，就同样归入 thread-private**——与 L1/L0/UB 同一类别，天然无跨核竞争。它主要影响的是
   **单核能容纳多大的私有工作集**（例如更深的私有任务环、更大的 outpool、把原本必须外溢到 GM 的私有
   状态就地放下），从而**减少对 GM 的访问压力**；但它**不改变跨核数据结构的竞争模型**——凡是需要被
   别的核看到的东西，仍必须落到 GM（§1.1）并走 §2 的发布/观察纪律。
3. **GM 上单一 owner 独占的一段**：`DistCore`（`DistGlobal::cores[core_id]`）物理在 GM、因而
   "全局可寻址"，但**只由拥有它的那个核读写**（single owner, no lock）：私有 TensorMap 副本、
   私有任务环 `slots[kPrivateSlots]`、输出池 `outpool[]`、`local_index` / `heap_next`。正确性依赖
   "确定性重放使各核副本内容一致、但进度不同"（§4/§9.3），**只要没有别的核去写它，就没有竞争**。
   minibench 需要验证的正是"确实没有别的核写它"这一前提。
   > 在具备 thread-private 3D DRAM 的芯片上，第 3 类里"物理在 GM、逻辑私有"的 `DistCore` 是**把它
   > 搬进私有 3D DRAM 的天然候选**——它本就单 owner 无锁，迁到私有介质后既省 GM 容量/带宽，又进一步
   > 把"没有别的核写它"从"约定"变成"介质保证"。这属于**布局优化**（换存放位置），不改访问过程与
   > 竞争模型。

## 2. 跨核可见性与原子性（A5 契约）

GM 全局可寻址，但 **A5 的 AICore 之间没有硬件缓存一致**：一个核写入的字，停在它自己的数据
缓存里，直到显式 `dcci … CACHELINE_OUT` 刷回 HBM；另一个核在显式失效（`dcci`）本地陈旧副本前
读到的是旧值（[cache-coherency.md](hardware/cache-coherency.md)）。因此 `std::atomic` 在真机上
**必要但不充分**——它只排序本核访问，不跨核发布。落地契约（§11.5 / §14）：

| 机制 | sim（`DIST_SIM_HOST_CLOCK=1`，同进程 host 线程） | A5 onboard（CCEC） |
| ---- | ---- | ---- |
| **RMW 原子**（claim、frontier CAS、`remaining` 递减、投递 state/drained CAS、append 定序器） | `std::atomic` CAS/fetch，单地址空间即真原子 | **A5 硬件 GM 原子** `atomicMax`/`atomicAdd`/`atomicCAS`（`kernel_operator_atomic_impl.h`）——内存级、核间序列化，读真值+发布，无需外围 dcci（§11.1.1） |
| **纯读共享量** | `std::atomic` load | 先 `dcci` 失效再读（否则读本核陈旧缓存），或用幂等原子 `atomicMax(p, INT32_MIN)` 读回真值 |
| **发布数据 + 标志** | release-store 即可见 | producer：写输出→`dcci` 刷出→置标志；consumer：acquire 见标志→`dcci` 失效→读数据 |
| **抽象层** | `Coherent<T>` 的 `dcci` hook 编成空操作，逐位等价于原 `std::atomic` | `Coherent<T>`：读前 `inval`、写后 `flush`，RMW 走硬件原子（§14.1） |

> **表中"标志"指什么、放在哪。** 即 §3 的**完成标志环 `flags[]`**——GM 上 `DistGlobal` 段里一个
> 定长环 `Coherent<int32_t> flags[kFlagCap=65536]`，每任务 id 一个一次性置位布尔（`1=done`），
> `flag(N)` 落在 `flags[N & (kFlagCap−1)]`。producer 执行完把输出刷回 GM 后，用内存级
> `atomicMax(&flags[N],1)`（`dist_set_flag`）置位；consumer（`drain_phase_b`）对自己任务的每个
> fan-in producer id 做 acquire-load，全为 `1` 才就绪、才读那些 producer 的输出。用 `int32_t` 而非
> `uint8_t` 是为了能走内存级原子、并避免稠密字节环的 cacheline 邻居 clobber（§2 开头风险 1、MB-2）。

关键实现（一致读、置标志的一致性替身）：

```1625:1636:src/common/runtime/fully_distributed_within_core/dist_engine.cpp
__aicore__ bool claim(__gm__ Coherent<int32_t> *cursor, int32_t N) {
#if DIST_SIM_HOST_CLOCK
    int32_t c = coherent_load(cursor, std::memory_order_acquire);
    while (true) {
        if (N <= c) return false;
        if (coherent_cas_weak(cursor, c, N, std::memory_order_acq_rel, std::memory_order_acquire)) return true;
    }
#else
    const int32_t old = dist_atomic_max(&cursor->a, N);  // hardware fetch_max
    return old < N;
#endif
}
```

> **两条已知的 A5 遗留风险（minibench 必须覆盖）：**
> 1. **cacheline clobber（C4/C15）**：`flags[]` 是稠密的每任务标志环，多个 flag 共用一条 64B cache
>    line；`dcci` 以整行为单位刷出，会把邻居"未刷出的写"连带写回，跨核并发置位会互相覆盖。实现
>    改用**内存级 `atomicMax(&flags[N], 1)`**（`dist_set_flag`）直接写 HBM 单字，绕开整行回写。
> 2. **跨核 RMW 原子性**：`Coherent<T>` 把"是否有真核间原子"这一决策**集中到一处**，但并未消除
>    它——若某原语在 HW 上非真原子，认领仲裁可能失效（§14.3）。minibench 是验证它"确实是真原子"
>    的手段。

置标志的一致性写法（避免整行 clobber）：

```584:591:src/common/runtime/fully_distributed_within_core/dist_engine.cpp
__aicore__ inline void dist_set_flag(__gm__ Coherent<int32_t> *c) {
#if DIST_SIM_HOST_CLOCK
    c->a.store(1, std::memory_order_release);
    pto_dcci_flush(&c->a, sizeof(c->a));  // no-op on sim; keeps parity with coherent_store
#else
    (void)dist_atomic_max(&c->a, 1);
#endif
}
```

---

# 第二部分 — 关键共享数据结构清单

以下清单以 `dist_engine.cpp` 的实际结构体为准，标注 **可见域 / 访问过程 / 并发纪律 / 竞争风险**。
minibench（第三部分）逐一对应。

## 3. GM 全局共享结构（`DistGlobal`）

| 结构（字段） | 类型 | 作用 | 访问过程 & 并发纪律 | 主要竞争风险 |
| ------------ | ---- | ---- | ------------------- | ------------ |
| `cube_cursor` / `vector_cursor` / `alloc_cursor`（各 `PaddedCursor[kCursorShards=4]`） | `Coherent<int32_t>`（cacheline 对齐分片） | 每类型 claim 高水位；到达 N 且 `old<N` 即胜出、独占任务 N（§2） | `claim()` = 单条 `atomicMax`（HW）/ acq-rel CAS 回路（sim）；shard = `N % 4`，纯 N 函数、各核一致 | 多核抢同一 id → 必须**恰一胜者且无跳过**；分片错误会造成伪共享或漏认领 |
| `flags[kFlagCap=65536]` | `Coherent<int32_t>` 环 | 每任务一次性完成标志（1=done），唯一共享的 per-task 状态 | producer：数据刷出后 `dist_set_flag`（`atomicMax(,1)`）；consumer：`drain_phase_b` acquire-load 轮询 | **cacheline 邻居 clobber**（C4）；标志早于数据可见（缺序）→ 读到未完成数据 |
| `frontier`（F） | `Coherent<int32_t>` | 全局连续完成前沿：所有 id≤F 已完成 | 任一核置 flag 后 `advance_frontier`：`while flag(F+1): CAS(F,F+1)` | 多核并发推进 → CAS 竞争；跳步/回退会破坏回收前沿 `R=F−H` |
| `vend[kFlagCap]` | `Coherent<uint64_t>` | 到任务 N 的累计虚拟堆字节（确定性、各核相同） | 每核对自己走位的 N 做 relaxed store（值相同）；读用于回收窗口 `[vend[R], heap_next)` | 各核写同一 N 应写**相同值**——若不同即确定性被破坏 |
| `heap_base` / `heap_size` + 每核 `heap_next` | 指针/`size_t` + 每核 `uint64_t` | 确定性 GM 输出堆（有界环） | 分配是**每核确定性 bump**（无原子）；物理地址 = 虚拟偏移 mod ring；回收由 F 派生的 R 驱动 + 反压自旋 | 回收过早 → 覆盖仍被读的产出；地址非确定 → 消费者算错 producer 地址 |
| `shared_map`（`SharedTensorMap`） | ring-per-bucket + 每槽 `Coherent<uint64_t> seq` | shared 模式全局唯一 TensorMap（tensor 区域→producer id） | 单一串行追加者（`tm_insert_next` 定序）；lookup 每次**一次** acquire（tail 快照）+ 各槽 relaxed 读 + `seq` ABA 护栏 | 追加乱序 → 与 private 不一致；`seq` 缺失 → 槽复用 ABA；越窗回收 → use-after-recycle |
| `tm_insert_next` | `Coherent<int32_t>` | 追加定序器（下一个应追加的任务 id） | `CAS(N → BUSY)` 抢占 → 追加 → `store(N+1)`，严格 id 序 | 多核抢同一 N 追加 → 必须恰一次、且 0..N−1 已追加 |
| `core_progress[RUNTIME_MAX_WORKER]` | `Coherent<int32_t>` | 各核 replay 进度（发布用于 min） | 每任务 relaxed store 自己进度；`min` 驱动 shared 回收前沿 + run-ahead 反压 | 读到陈旧 min → 回收过早或反压失效 |
| `fatal` | `Coherent<int32_t>` | 全局致命标志（首写者胜） | 任意核 store 1；各核轮询 | 良性竞争（幂等），但需跨核可见 |
| `replay_done` / `started_count` | `Coherent<int32_t>` | 启动栅栏 + 尾部空转计数（§7） | 入口 `fetch_add` + 自旋到 `num_workers`；replay 完成计数 | 栅栏漏计 → 冷启动错位 / 提前终止漏抽取 |
| `dep_sig` / `dep_edges` | `Coherent<uint64_t>` | 依赖图签名（验证用，XOR 累加） | 每解析一条 fan-in 边 `fetch_xor`；顺序无关 | 仅验证用，不影响功能——是 minibench 的**判据来源** |

## 4. 线程组共享结构（`block.won`）

`DistGlobal::blocks[block_id]` —— 每 block 一个 `BlockWon`，含 `kPrivateSlots` 个 `WonSlot`：

```1068:1084:src/common/runtime/fully_distributed_within_core/dist_engine.cpp
struct WonSlot {
    Coherent<int32_t> state;  // 0=free, 1=published, 2=reserving
    int32_t task_id;
    Coherent<int32_t> remaining;                         // co-owners (incl. anchor) left to finish
    Coherent<int32_t> drained[PTO2_SUBTASK_SLOT_COUNT];  // 0/1 per follower lane
    BuiltSubtask lane[PTO2_SUBTASK_SLOT_COUNT];             // deposited follower subtasks
};

struct BlockWon {
    WonSlot slots[kPrivateSlots];
    Coherent<int32_t> any_pub;
};
```

访问过程（anchor → follower）：

1. **anchor 预定槽**：`alloc_won_slot` 用 `CAS(state: 0→2)` 抢一个空 `WonSlot`（2V 下 AIV0/AIV1
   都可能是 anchor，故必须原子）。
2. **anchor 投递**：写 `task_id` / 各 lane 的 `BuiltSubtask` / `remaining = popcount(active_mask)`，
   置 `any_pub`，最后 `state = 1`（published，release）。
3. **follower 抽取**：`drain_block_won` acquire-load `state==1` → 检查 `lane[self].present` →
   `CAS(drained[self]: 0→1)` 认领（防重复抽取）→ 建进自己私有环槽。
4. **完成收尾**：每个 co-owner 执行完自己子任务后 `remaining.fetch_sub(1)`；把 `remaining` 减到 0
   的那个核置**单一**完成标志 `flags[task_id]`，并 `state = 0` 回收槽。

竞争风险：槽预定/回收的 ABA（`state`）、重复抽取（`drained`）、`remaining` 少减/多减导致标志
早置或永不置（→ 消费者挂死）、`any_pub` 快路径短路的可见性。

## 5. 每核私有结构（`DistCore`，单一 owner 无锁）

`DistGlobal::cores[core_id]`：`map`（private `DistTensorMap`）、`slots[kPrivateSlots]`（私有任务环）、
`outpool[kOutPoolSlots]`（物化输出 Tensor 环）、`local_index` / `heap_next` / `occupied_count`。

正确性**不靠锁**，靠两条不变式：(a) 只有 owner 核读写自己的 `DistCore`；(b) 每核走**相同的确定性
submit 序列**、对每个任务无条件维护自己的 map/heap 记账，使各核副本**内容一致、仅进度不同**。
minibench 的职责是验证这两条前提在多核并发下确实成立（尤其 private vs shared **逐位一致**）。

---

# 第三部分 — Minibench 设计

## 6. 通用方法论

单纯"跑通不崩"无法证明无竞争——竞争往往是偶发、与调度相关、在 sim 上被单地址空间掩盖。因此每个
minibench 都由三要素构成：

1. **压力（stress）**：制造最坏并发——所有核抢同一 id / 同一 bucket / 同一 block，高核数（6/24/72），
   并用 `PTO_DIST_FAKE_EXEC_NS` 或 skip-exec 拉大 run-ahead `Δ`，逼出越窗与 ABA。
2. **判据（oracle）**：一个**与调度顺序无关**、竞争一旦发生就必然改变的不变式。首选：
   - **依赖图签名 `PTO_DIST_DEPSIG`**：对每条 fan-in 边 `(consumer, producer)` XOR 累加——免疫浮点
     累加顺序噪声，只取决于边的**集合**（§12.10）。
   - **计数守恒**：胜者计数 == 任务数、每任务恰一次追加、`remaining` 归零次数 == 多核任务数。
   - **差分**：private vs shared **逐位相同**；ring 语义 vs 参考实现（`test_dist_tensormap_ring.cpp`）
     逐次相同。
3. **判定（verdict）**：sim 绿 → 结构与纪律正确；A5 绿 → HW 原子/一致性 seam（`atomicMax`/`dcci`）
   正确。**同一判据在两平台都跑**，sim 定位逻辑错误、onboard 定位一致性错误。

> 复用现成设施：`PTO_DIST_DEPSIG`（签名）、`PTO_DIST_OVERHEAD`（`inserts`/`lookups`/`scans` 计数）、
> `PTO_DIST_TENSORMAP_MODE`（private|shared）、`PTO_DIST_RUNAHEAD`（Δ 上界）、`PTO_DIST_H`（依赖跨度）。
> C++ 侧差分 UT 参考 `tests/ut/cpp/a2a3/test_dist_tensormap_ring.cpp`。

## 7. Minibench 清单

每个 minibench 给出：**目标结构 → 竞争场景 → 压力构造 → 判据 → 通过条件**。

### MB-1 claim cursor —— 恰一胜者 & 无跳过（`atomicMax`）

- **目标 / 访问过程**：`cube_cursor`/`vector_cursor` 的 `claim()`（§3、`dist_engine.cpp:1625`）。
- **竞争场景**：N 个核同时到达任务 id N，必须**恰好一个** `old<N` 返回 true；且 cursor 只在本类型
  子序列上单调跃进，**不跳过任何 id**（§11.1）。
- **压力**：一段无依赖、全同类型（全 AIC-only 或全 AIV-only）的长 submit 流；6/24/72 核；
  `kCursorShards=4` 与设为 1 各跑一遍（验证分片语义等价）。
- **判据**：
  - 全核累加 `owned_total` == 任务数（**每任务恰一 owner**）；
  - 逐 id 校验 `won` 的集合是 `[0, num_tasks)` 的一个**排列**（无重复、无遗漏）——可让每个胜者对
    `dep_sig` 风格的 XOR 累加器异或自己认领的 id，期望 == `XOR(0..num_tasks-1)`。
- **通过条件**：sim + a5 onboard 上判据均成立；分片 1 与 4 结果一致。**这是 A5 硬件 `atomicMax`
  是否真核间原子的直接探针**（§14.3 最大遗留风险）。

### MB-2 完成标志环 `flags[]` —— 置位/轮询 + cacheline 邻居防 clobber

- **目标 / 访问过程**：`dist_set_flag`（`atomicMax(,1)`）+ `drain_phase_b` 的 acquire 轮询
  （`dist_engine.cpp:584` / `:1829`）。
- **竞争场景**：**相邻 task id 的标志共享一条 64B cache line**，多核同时置位，`dcci` 整行回写会把
  邻居未刷出的写覆盖（C4/C15）——被 clobber 的任务，其消费者永远等不到 `flags[id]==1` 而挂死。
- **压力**：制造大量**结构相同、id 连续**的单核任务（如 512 个独立 1V），每个都有一个下游消费者
  等它；高核数使连续 id 落在不同核并发置位。**关键**：必须在 onboard（真有 cache line 概念）上跑，
  sim 掩盖此问题。
- **判据**：全部消费者最终就绪并完成（无挂死）；`frontier` 推进到 `num_tasks-1`；看门狗不触发。
- **通过条件**：onboard 上 512+ 连续标志全部无丢失。**若失败**即坐实需要 flag 按 cacheline 对齐或
  per-core 分片（§14.3 粒度问题）。

### MB-3 完成前沿 `frontier` —— 协作式 CAS 推进

- **目标 / 访问过程**：`advance_frontier`（`dist_engine.cpp:1642`）。
- **竞争场景**：多核在乱序完成后并发 `CAS(F, F+1)` 推进前沿；必须**单调、不回退、不跳步**，且
  只在 `flag(F+1)==1` 时才推进。
- **压力**：任务**乱序完成**（用不同 `PTO_DIST_FAKE_EXEC_NS` 让后置 id 先完成），逼出"前沿卡在空洞
  处、后续标志已置但 F 不该越过空洞"的场景。
- **判据**：任一时刻 `frontier` 满足"所有 id≤F 的 `flags`==1"（可在 core 0 周期性快照断言）；运行
  结束 `frontier == num_tasks-1`；回收前沿 `R=F−H` 单调不减。
- **通过条件**：sim + onboard 均无空洞越过、无回退。

### MB-4 `block.won` 投递表 —— anchor 投递 / follower 抽取 / remaining 归零

- **目标 / 访问过程**：`alloc_won_slot`（`state` CAS）→ 投递（`state=1` release）→ `drain_block_won`
  （`drained` CAS）→ `remaining.fetch_sub`（`dist_engine.cpp:1916/1947/1800`）。
- **竞争场景**（block 内 1 AIC + 2 AIV 并发）：
  - `state` 的预定/回收 ABA（0→2→1→0 再被复用）；
  - 同一 lane 被**重复抽取**（`drained` 未原子认领）；
  - `remaining` **少减/多减**：少减→标志永不置（消费者挂死）；多减→标志早置（读未完成数据）。
- **压力**：一段**全 MIX（1C+2V）**任务流，使每个 block 的 3 个 lane 都要投递+抽取+归零；配合 2V
  任务让 AIV0/AIV1 都可能当 anchor（触发 `alloc_won_slot` 的原子竞争）；`BLOCK_WON_SLOTS` 打满触发
  反压。
- **判据**：
  - 每个多核任务的 `remaining` **恰好**被减到 0 一次 → 其 `flags[task_id]` 恰置一次；
  - 全 block 抽取次数 == 投递的 lane 子任务总数（无重复、无遗漏）；
  - 结束时所有 `WonSlot.state==0`（无泄漏槽）；
  - DEPSIG 与 private 参考一致。
- **通过条件**：sim + onboard 全绿，无挂死、无早置。

### MB-5 shared TensorMap —— 串行追加定序 + 单 acquire lookup + seq 防 ABA

- **目标 / 访问过程**：`tm_shared_claim_append`（`tm_insert_next` 定序，`dist_engine.cpp:2051`）、
  `shared_tm_append` / `shared_tm_lookup`（`:899/:941`，含 `seq`）。
- **竞争场景**：
  - 多核抢同一任务 N 的追加——必须**恰一次**且在 0..N−1 追加之后（§12.4）；
  - reader 扫描时 `head` 并发回收把物理槽复用 → `seq != k` 必须识别并跳过（ABA，§12.7.1）；
  - 快核 run-ahead 使 `Δ+H` 存活窗口超过 `cap` → 必须转反压而非静默丢/覆写（§12.7.2）。
- **压力**：`PTO_DIST_TENSORMAP_MODE=shared`；BGEMM 型工作负载（数百个 disjoint tile 全落**同一
  bucket**，最热）；`PTO_DIST_RUNAHEAD` 调小逼反压、调 0 逼确定性 FATAL（验证不静默丢）；6/24/72 核。
- **判据**：
  - **private vs shared DEPSIG 逐位相同**（首要判据，§12.10 已实测 750/15/… 边一致）；
  - `PTO_DIST_OVERHEAD` 的 `inserts`：shared ≈ D（单环追加），private ≈ C×D（每核复制）——下沉比
    符合预期即证明"仅 winner 追加"生效；
  - ring 溢出走 FATAL/反压、**从不静默覆写**（注入超 `cap` 场景断言）。
- **通过条件**：sim（a2a3sim/a5sim）+ onboard 上 shared==private 签名；反压场景顺利完成、签名不变。

### MB-6 确定性 GM 输出堆 —— 无原子分配 + 前沿回收反压

- **目标 / 访问过程**：`dist_submit_impl` 步骤 (a) 的确定性 bump（`heap_next` / `vend[]`，
  `dist_engine.cpp:2181`）+ 回收反压自旋（`:2403`）。
- **竞争场景**：分配**无原子**（每核复算），要求 `addr(N)` 是 N 的纯函数、各核完全相同；回收由
  `R=F−H` 驱动，**过早回收**会覆盖仍被下游读的产出。
- **压力**：有真实跨任务依赖的图（PagedAttention / BGEMM）；`PTO_DIST_HEAP_MB` 调小逼近工作集，
  触发回收反压自旋；`PTO_DIST_H` 设过小以主动触发"heap span exceeded"确定性 FATAL（验证诊断而非
  静默错读）。
- **判据**：
  - 各核对同一 N 写入的 `vend[N]` **相同**（可采样断言）→ 地址确定性成立；
  - 数值 golden 通过（跨任务依赖数据正确，说明无过早回收）；
  - 堆太小/H 太小时报确定性 FATAL，绝不静默覆写。
- **通过条件**：sim + onboard golden 通过；反压不死锁（最慢核推进 → R 前进 → 释放）。

### MB-7 `core_progress[]` + run-ahead 反压 —— min 一致性与均衡

- **目标 / 访问过程**：`tm_shared_min_progress`（min over cores）+ `dist_runahead_throttle`
  （`dist_engine.cpp:1999/2030`）。
- **竞争场景**：反压读各核进度取 min；若读到陈旧 min → 快核冲太远（`Δ` 爆炸）→ shared 环溢出 /
  负载失衡。反压等待期间必须**协作式 drain**（不 park），否则死锁。
- **压力**：极不均衡的核速（skip-exec + 少数核 `FAKE_EXEC_NS` 很大）；`PTO_DIST_RUNAHEAD` 从 0
  （关）到小值（紧）扫一遍。
- **判据**：任一时刻 `max local_index − min local_index ≤ runahead_max`（周期采样）；无死锁（最慢核
  永不被节流 → min 上升 → 释放）；DEPSIG 不因节流改变（节流只改**谁**赢，不改依赖图）。
- **通过条件**：sim + onboard 上窗口约束成立、无死锁、签名不变。

### MB-8 `Coherent<T>` / dcci 一致性 seam —— HW 发布/观察正确性

- **目标 / 访问过程**：`coherent_load`/`coherent_store`/`coherent_fetch_*`/`coherent_cas_*` 及
  `pto_dcci_inval`/`pto_dcci_flush`（`dist_engine.cpp:344`–`:591`、§14）。
- **竞争场景**：producer 写 GM 数据后未刷出、consumer 读前未失效 → 读到陈旧值。这是**所有其它
  minibench 的底座**：MB-1..7 的判据在 onboard 失败时，根因常在此 seam。
- **压力**：一个最小 producer→consumer 链（core A 写一段 GM 数组 + 置标志，core B 等标志后校验
  数组），跨 block 运行强制走 HBM；反复多轮触发缓存复用。
- **判据**：consumer 每轮读到的数组 == producer 本轮写入值（逐元素）；标志与数据不出现"标志到、
  数据旧"。
- **通过条件**：sim（seam 编空）恒绿作对照；onboard 上 seam 生效后同样恒绿。**这是把 `dcci`/屏障
  三处 HW 实现钉死的最小可复现单测**。

### MB-9 private 复制确定性 —— 每核副本逐位一致

- **目标 / 访问过程**：`DistCore::map`（private `DistTensorMap`）的每核无条件 insert/retire
  （§4/§9.3、`dist_engine.cpp:727` 起）。
- **竞争场景**：private 模式下 `DistCore` 是每核私有，**不应有任何跨核写**；正确性全靠"确定性重放
  使副本一致"。若某处误让别的核写了它，或重放非确定，副本发散。
- **压力**：同一图分别用 1 核与 N 核跑（N=6/24/72）；private 模式；随机化调度（不同 FAKE_EXEC）。
- **判据**：
  - 所有核最终解析出的 DEPSIG **相同**且 == 单核参考；
  - C++ 差分 UT（仿 `test_dist_tensormap_ring.cpp`）：ring-per-bucket 的 lookup 结果对随机 SPMD 序
    op 流与"链表参考语义"**逐次逐位相同**，含"快核超前/滞后回收"越窗条件。
- **通过条件**：差分 UT 绿；多核 DEPSIG == 单核。

## 8. 覆盖矩阵

| Minibench | 目标结构 | 核心竞争 | 主判据 | 必测平台 |
| --------- | -------- | -------- | ------ | -------- |
| MB-1 | claim cursor | 恰一胜者/无跳过 | id 集合为排列 | sim + **onboard** |
| MB-2 | `flags[]` | cacheline clobber | 无消费者挂死 | **onboard**（sim 掩盖） |
| MB-3 | `frontier` | 协作 CAS 单调 | 无空洞越过 | sim + onboard |
| MB-4 | `block.won` | 投递/抽取/remaining | 恰一次归零、无泄漏 | sim + **onboard** |
| MB-5 | `shared_map` | 追加定序/seq/反压 | private==shared DEPSIG | sim + onboard |
| MB-6 | GM 堆 | 确定性分配/回收 | vend 一致 + golden | sim + onboard |
| MB-7 | `core_progress`/反压 | min 一致/均衡 | 窗口约束 + 无死锁 | sim + onboard |
| MB-8 | `Coherent<T>`/dcci | 发布/观察可见性 | 数据逐元素一致 | **onboard**（底座） |
| MB-9 | private map | 复制确定性 | 差分 UT + DEPSIG | sim + onboard |

## 9. 落地建议（目录 / 命令 / CI）

- **C++ 差分/单元级**（MB-1、MB-3、MB-5、MB-9 的逻辑判据）：放
  `tests/ut/cpp/a5/test_dist_atomic_*.cpp`，仿 `tests/ut/cpp/a2a3/test_dist_tensormap_ring.cpp`——
  纯 host 可 `g++ -O2 -std=c++17` 独立编译，把结构体逻辑抽出来在多线程下差分，快、可 CI。
- **端到端 st**（MB-2、MB-4、MB-6、MB-7、MB-8 需真并发/真缓存）：放
  `tests/st/a5/fully_distributed_within_core/atomic_minibench/`，各配最小 orchestration + kernels
  （参考同目录 `mix_coown`/`vector_example`/`benchmark_bgemm`），用 env 切模式/核数/反压。
- **判据自动化**：st 用例统一开 `PTO_DIST_DEPSIG=1`，对比 private/shared 与单核参考签名；
  `PTO_DIST_OVERHEAD=1` 断言 `inserts` 下沉比；看门狗触发即视为挂死失败。
- **平台矩阵**：先 a5sim（`DIST_SIM_HOST_CLOCK=1`，验证结构与纪律，seam 编空）→ 再 a5 onboard
  （seam 生效，验证 `atomicMax`/`dcci`/屏障三处 HW 实现）。onboard 专属项（MB-2/MB-8）不在 sim 判定
  通过即放行。

## 10. 相关文档

- [fully_distributed_within_core.md](fully_distributed_within_core.md) —— 系统设计、并发纪律、
  §11.1（claim 原子性）/§11.5（跨核标志可见性）/§12（private/shared TensorMap）/§14（`Coherent<T>`）。
- [hardware/chip-architecture.md](hardware/chip-architecture.md) —— 三层内存域、执行层级。
- [hardware/cache-coherency.md](hardware/cache-coherency.md) —— GM↔AICore/AICPU 一致性规则、
  `dcci` 整行语义与 cacheline 伪共享。
- `src/common/runtime/fully_distributed_within_core/dist_engine.cpp` —— 全部结构与访问过程的实现。
- `tests/ut/cpp/a2a3/test_dist_tensormap_ring.cpp` —— 差分 minibench 的参考写法。
