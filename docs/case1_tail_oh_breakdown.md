# Case1 Tail OH 完整 Breakdown

> 数据来源：`PA_CASE=Case1 --enable-profiling`，16,704 tasks, 3 scheduler threads × 24 cores/thread

---

## Part 1: 每任务时间分解（Perf 采集数据）

每个任务经历四段时间：

```
dispatch_time ──→ start_time ──→ end_time ──→ finish_time
     │   Head OH    │   Exec     │   Tail OH     │
```

| 分量 | 总时间 (us) | 每任务平均 (us) | 占 Wall-clock |
|------|------------|----------------|---------------|
| Kernel Exec (end − start) | 29,743 | 1.78 | 82.9% |
| Head OH (start − dispatch) | 30,672 | 1.84 | 85.5% |
| **Tail OH (finish − end)** | **793,724** | **47.52** | **2212.7%** |

- Wall-clock 总耗时：**35,872 us**
- Tail OH 总和远超 wall-clock，因为 16,704 个任务的 Tail OH 是**各自独立累加**的（存在大量并行重叠）。

---

## Part 2: AICPU 调度器循环 CPU 时间 Breakdown（Device Log）

### 2.1 三个调度线程概况

| Thread | Loops | 完成任务数 | 总 CPU 时间 (us) |
|--------|-------|-----------|-----------------|
| T0 | 706 | 5,864 | 42,679 |
| T1 | 690 | 5,663 | 42,648 |
| T2 | 591 | 5,177 | 42,653 |
| **SUM** | **1,987** | **16,704** | **127,979** |

### 2.2 调度器循环各阶段 CPU 时间

每次循环按顺序执行：

```
┌─ Phase 1: Complete ─┐  ┌─ Phase 2: Dispatch ─┐  ┌─ Scan ─┐  ┌─ Orch Drain ─┐  ┌─ Yield ─┐
│ 遍历所有 24 个 core  │  │ 为空闲 core 派发任务 │  │ 发现新  │  │ 处理编排器  │  │ 无进展  │
│ 检查 handshake      │  │ pop ready queue     │  │ 根任务  │  │ 就绪队列    │  │ 让出CPU │
│ 记录 finish_ts      │  │ build_payload       │  │         │  │             │  │         │
│ 解析 fanout 依赖    │  │ cache flush (dc+dsb)│  │         │  │             │  │         │
└─────────────────────┘  └─────────────────────┘  └─────────┘  └─────────────┘  └─────────┘
```

| 阶段 | CPU 时间 (us) | 占比 | 每任务 (us) | 主要开销 |
|------|--------------|------|------------|---------|
| **Dispatch** | **79,587** | **62.2%** | **4.76** | cache flush (`dc cvac` + `dsb sy`) |
| Complete | 43,968 | 34.4% | 2.63 | handshake 轮询 + fanout atomic ops |
| Scan | 3,797 | 3.0% | 0.23 | 新任务发现 |
| Orch Drain | 64 | 0.0% | 0.00 | 编排器就绪队列消费 |
| Yield | 563 | 0.4% | 0.03 | thread_yield() |
| **Total** | **127,979** | | **7.66** | |

### 2.3 锁竞争

| 分项 | 等锁 (us) | 持锁 (us) |
|------|----------|----------|
| Dispatch (pop ready_q) | 29,156 | 6,443 |
| Complete (push ready_q) | 3,043 | 1,200 |
| Scan | 394 | 335 |
| **Total** | **32,592 (25.5%)** | **7,978 (6.2%)** |

### 2.4 Fanout 依赖解析

- 总遍历次数：22,088
- 最长 fanout 链：35
- 平均 fanout/任务：1.3
- Fanout 锁竞争：spin=0us, hold=0us（无竞争）

---

## Part 3: Tail OH 分布

| 分位数 | Tail OH (us) |
|--------|-------------|
| P10 | 33.4 |
| P25 | 41.0 |
| **P50** | **48.3** |
| P75 | 54.6 |
| P90 | 59.8 |
| P95 | 62.9 |
| P99 | 68.8 |
| Max | 192.4 |
| **Mean** | **47.5** |

---

## 关键问题解析

### Q1: 为什么 Part 1 的每任务 Tail OH (47.52 us) 和 Part 2 的每任务 CPU 时间 (7.66 us) 对不上？

**核心区别：Part 1 测的是 wall-clock 等待时间，Part 2 测的是 CPU 分摊成本。**

调度器循环结构如下（以一个线程为例）：

```
Loop iteration #N (avg 64.4 us)
├── Phase 1: 遍历 24 cores，检查哪些完成      ← 某个任务的 finish_ts 在这里记录
├── Phase 2: 遍历 24 cores，派发就绪任务
├── Scan: 扫描新提交的任务
└── Yield (如果无进展)

Loop iteration #N+1 ...
├── Phase 1: 再次遍历 24 cores               ← 上一轮没检测到的任务，在这里被发现
...
```

**每次循环迭代平均处理 ~8.4 个任务**（16,704 tasks ÷ 1,987 loops）。

- **Part 2 的 7.66 us/task**：把一次循环 64.4 us 的 CPU 时间平摊到这 8.4 个任务上 → 64.4 ÷ 8.4 ≈ 7.66 us。这是 **AICPU 为每个任务付出的 CPU 成本**。

- **Part 1 的 47.52 us/task**：每个任务从 kernel 执行完 (`end_time`) 到被 Phase 1 检测到 (`finish_time`) 的 **wall-clock 等待**。即使循环只花 7.66 us 的 CPU 在"你的"任务上，你仍需要等整个循环把其他 7-8 个任务的工作也做完。

**类比**：银行柜台有 3 个窗口（3 threads），每个窗口每轮叫 8 个号。柜员处理你的业务只要 1 分钟（CPU cost），但你要等前面 7 个人都处理完才能轮到——排队等待 8 分钟（wall-clock wait）。

数值验证：
```
每线程每循环时间 = 42,660 us ÷ 706 loops ≈ 60.4 us (T0)
任务平均在循环中间某个时刻完成
→ 平均等待 ≈ 0.5 ~ 0.8 × 循环时间 ≈ 30 ~ 50 us
→ 实测 Tail OH 均值 47.5 us ✓
```

### Q2: 为什么 Part 3 的 Tail OH 这么长？为什么 Part 2 没有体现？

**Part 2 的数字已经完整体现了原因，只是需要换一个视角来理解。**

Part 2 告诉我们：**每次循环迭代耗时 64.4 us**。这 64.4 us 就是 Tail OH 的根本上限。

Tail OH 长的原因是调度循环慢。循环慢的原因在 Part 2 中清晰可见：

```
每次循环迭代 64.4 us 的时间花在哪里：

  Dispatch (cache flush):  62.2% → ~40 us    ← 主要瓶颈
  Complete (poll+fanout):  34.4% → ~22 us
  Scan + Yield:             3.4% →  ~2 us
```

**Dispatch 阶段的 cache flush 是根因**。每次派发任务需要：
1. `dc cvac` 逐 cacheline 刷新 PTO2DispatchPayload (多次, ~160 bytes / 64 = 3 lines)
2. `dc civac` 刷新 Handshake (1 次)
3. `dsb sy` 全局屏障：**阻塞 AICPU 流水线直到所有 dc 操作完成**

一个循环中可能派发 8+ 个任务，每个都要经历这套 flush。加上锁竞争（29,156 us 总等锁），Dispatch 消耗了大量时间。

**Part 2 和 Part 3 的联系**：

| Part 2 观察 | → | Part 3 后果 |
|-------------|---|------------|
| 循环迭代 64.4 us | → | P50 Tail OH ≈ 48.3 us（等待约 0.75 个循环） |
| Dispatch 占 62% | → | 即使 kernel 已完成，Phase 1 还没到就被 Dispatch 阻塞 |
| 锁竞争 25.5% | → | 3 线程争抢 ready_q 锁，进一步拉长循环 |
| P99 = 68.8 us ≈ 1 loop | → | 极端情况刚好错过本轮 Phase 1，要等完整下一轮 |
| Max = 192.4 us ≈ 3 loops | → | 偶发竞争或 OS 调度导致多轮延迟 |

### 总结：Tail OH 的因果链

```
                    Root Cause
                        │
           ┌────────────┴────────────┐
           │  每次 Dispatch 需要     │
           │  dc cvac + dsb sy      │
           │  刷新 AICPU cache      │
           └────────────┬────────────┘
                        │
           ┌────────────┴────────────┐
           │  Dispatch 占循环 62%    │
           │  + 锁竞争 25.5%        │
           └────────────┬────────────┘
                        │
           ┌────────────┴────────────┐
           │  循环迭代 ~64 us        │
           │  (Phase1+Phase2+Scan)   │
           └────────────┬────────────┘
                        │
           ┌────────────┴────────────┐
           │  任务完成后平均等       │
           │  ~47.5 us 才被检测到   │
           └────────────┬────────────┘
                        │
              Tail OH ≈ 47.5 us/task
              (占端到端时间的主导部分)
```

### 潜在优化方向

1. **减少 cache flush 次数**：批量派发后统一执行一次 `dsb sy`，而非每个任务一次（见下方风险分析）
2. **减少 flush 范围**：只 flush 真正需要的 cacheline（如 tensor_copies 部分可能不需要每次 flush）
3. **降低锁竞争**：增加 ready_q shard 数量（当前 shard 数可能不足）
4. **缩短 Phase 1 + Phase 2 路径**：减少每轮遍历的 core 数（针对实际使用的 core 数优化）

---

## 优化方案风险分析：批量 `dsb sy`

### 当前实现：每派发一个任务执行一次完整 flush

```
for each idle core with a ready task:
    build_pto2_payload(payload, ...)       // 写 payload 数据
    h->task = payload_addr                 // 写 handshake.task
    h->task_status = 1                     // 写 handshake.task_status = 1 (启动信号)
    dc cvac payload  (×3 cachelines)       // 刷 payload 到 HBM
    dc civac handshake                     // 刷+失效 handshake 到 HBM
    dsb sy                                 // 等待所有 dc 操作完成 ← 阻塞 ~3-5 us
```

### 提议优化：批量 flush

```
// Step 1: 批量写入所有任务
for each idle core with a ready task:
    build_pto2_payload(payload, ...)
    h->task = payload_addr
    h->task_status = 1
    dc cvac payload  (×3 cachelines)
    dc civac handshake
    // 不等待 ←── 省掉 dsb sy

// Step 2: 一次性等待全部完成
dsb sy                                     // 所有 dc 操作在这里统一完成
```

### 风险 1 (致命)：Payload 与 Handshake 的到达顺序不可控

**AICPU 和 AICore 之间通过 HBM 通信，不共享缓存。** 通信协议如下：

```
AICPU 端:                              AICore 端 (轮询循环):
                                         while (true):
  [1] 写 payload 到 AICPU cache             dcci(handshake)        // 失效自身缓存,从HBM读
  [2] 写 handshake.task_status = 1           if task_status == 1:   // 看到启动信号?
  [3] dc cvac payload   → 刷到 HBM              读 payload         // 通过 handshake.task 指针读
  [4] dc civac handshake → 刷到 HBM              execute_task(payload)
  [5] dsb sy            → 保证[3][4]完成         task_status = 0    // 通知完成
```

**关键不变式**：AICore 看到 `task_status=1` 时，`payload` 必须已经在 HBM 中。

没有 `dsb sy` 时，`dc cvac`（payload）和 `dc civac`（handshake）仅仅是**发射**了缓存操作，
ARM 架构**不保证**它们按程序顺序完成到 HBM。可能出现：

```
时间线:
  AICPU cache ops issued:   dc cvac(payload_A)  dc civac(hank_A)  dc cvac(payload_B) ...
  HBM 写入实际顺序:         hank_A arrives ✓    payload_B arrives   payload_A arrives (延迟)
                                  ↑
                            AICore 此时 dcci 看到 task_status=1
                            但 payload_A 还没到 HBM → 读到旧数据 → 跳转到错误地址 → HANG
```

**结论：这是一个硬件级的数据竞争 (data race)，会导致随机 hang 或数据损坏。**

> ARM Architecture Reference Manual (D5.10.2): "A data cache operation is only guaranteed
> to be complete when a DSB is executed after the cache maintenance instruction."

### 风险 2 (中等)：批量延迟导致 AICore 空转时间增加

当前实现中，第一个 task dispatch 后立即 `dsb sy` 完成，AICore 可能在 ~3-5 us 后就开始执行。
批量方案中，所有 task 的 flush 要等到最后一个 task 准备好后才统一 `dsb sy`。
如果一次循环派发 8 个 task，前面几个 task 的 AICore 要多等几个 us：

```
当前:  dispatch_A → dsb(3us) → AICore_A starts │ dispatch_B → dsb(3us) → AICore_B starts
批量:  dispatch_A → dispatch_B → ... → dsb(3us) → AICore_A starts, AICore_B starts (同时)
                                                    ↑ AICore_A 多等了 N×(build_payload) 时间
```

对于执行时间 ~1.78 us 的短 kernel，这个额外等待可能显著。

### 风险 3 (低)：Phase 1 重入 stale 读

Phase 1 用 `dc civac` 在 handshake 上做 clean+invalidate。如果批量 dispatch 改变了
handshake 的 flush 时机，Phase 1 下一次循环读到的可能是 AICPU 自身缓存中的旧值
而非 AICore 写回 HBM 的 `task_status=0`。当前 per-task `dsb sy` 保证了 flush 完成后
才进入下一轮循环；批量化后这个保证变弱。

### 安全的折中方案

如果要优化 `dsb sy` 开销，可以考虑以下方案：

#### 方案 A：两阶段 flush（保持正确性，减少 dsb 次数）

```
// Step 1: 批量发射所有 payload flush
for each task:
    build_pto2_payload(...)
    h->task = payload_addr
    // 先不写 task_status
    dc cvac payload

// Step 2: 确保所有 payload 到达 HBM
dsb sy                                  // ← 第一个 barrier

// Step 3: 现在安全地设置启动信号并 flush handshake
for each task:
    h->task_status = 1
    dc civac handshake

// Step 4: 确保所有 handshake 到达 HBM
dsb sy                                  // ← 第二个 barrier
```

**2 次 `dsb sy` 替代 N 次**，同时保证 payload 一定在 handshake 之前到达 HBM。

> 预期收益：N 个 task 从 N 次 dsb (~N×3us) 降到 2 次 dsb (~6us)。
> 但需要两次遍历 core 列表，增加代码复杂度。

#### 方案 B：仅合并 dsb sy，保持 dc 操作分散

```
for each task:
    build_pto2_payload(...)
    h->task = payload_addr
    h->task_status = 1
    dc cvac payload
    dc civac handshake
    // 不 dsb

dsb sy   // 循环最后统一 barrier
```

**风险：直接触发风险 1（payload/handshake 到达顺序不可控），不安全。**

### 结论

| 方案 | dsb 次数 | Payload→Handshake 顺序保证 | 安全性 |
|------|---------|---------------------------|--------|
| 当前 | N/循环 | ✅ 每个 task 独立保证 | ✅ 安全 |
| 方案 A (两阶段) | 2/循环 | ✅ 全局 barrier 分隔 | ✅ 安全 |
| 方案 B (末尾单 dsb) | 1/循环 | ❌ 无保证 | ❌ 可能 hang |

**推荐方案 A**。主要风险是代码复杂度增加和"前几个 task 的 AICore 需多等几 us"（风险 2），
但不会引入正确性问题。

---

## 优化方案风险分析：减少 flush 范围

### 当前状态：flush 了什么、没 flush 什么

代码注释声称有 **3 个区域**需要 flush，但实际只 flush 了 2 个：

```
注释列出的 3 个区域:                     实际代码:
┌─────────────────────────────────────┐  ┌──────────────────────┐
│ ① tensor_copies[] (~2688B, ~42 CL) │  │ ❌ 没有 flush        │
│   Thread 3 (orch) 写入 buffer.addr │  │                      │
│   AICore 通过 args[i] → Tensor*    │  │                      │
│   间接读取                          │  │                      │
├─────────────────────────────────────┤  ├──────────────────────┤
│ ② PTO2DispatchPayload (~288B, ~5CL)│  │ ✅ dc cvac × ~5      │
│   scheduler 线程 build_pto2_payload │  │                      │
├─────────────────────────────────────┤  ├──────────────────────┤
│ ③ Handshake (~64B, 1 CL)           │  │ ✅ dc civac × 1      │
│   scheduler 线程写 task_status=1    │  │                      │
└─────────────────────────────────────┘  └──────────────────────┘
                                         + dsb sy × 1
```

**关键发现：`tensor_copies[]` 当前没有被 flush，但 Case1 大部分情况下能通过。**

### AICore 读取 tensor_copies 的完整路径

```
AICPU 端 (Thread 3 编排器):
  pto2_submit_task():
    task->tensor_copies[i] = *params[i].tensor;     // [W1] 拷贝 Tensor 元数据
    task->tensor_copies[i].buffer.addr = alloc_addr; // [W2] 填入 heap 分配地址
    task->params[i].tensor = &task->tensor_copies[i]; // 指针重定向

AICPU 端 (Thread 0/1/2 调度器):
  build_pto2_payload():
    out->args[n] = (uint64_t)task->params[i].tensor; // [W3] 把 &tensor_copies[i] 写入 payload
  // dc cvac payload → 刷 args[] 到 HBM (包含指向 tensor_copies 的指针值)
  // dc civac handshake → 刷 task_status=1
  // dsb sy
  // ⚠️ tensor_copies[i] 本身没有 flush！

AICore 端:
  aicore_executor:
    dcci(handshake)                                   // 从 HBM 读 handshake
    if (task_status == 1):
      payload = (PTO2DispatchPayload*)handshake->task  // [R1] 读 payload (已 flush ✓)
      kernel(payload->args)                            // args 包含 Tensor* 指针
  
  qk_matmul kernel:
    Tensor* qi = (Tensor*)args[0];                    // [R2] 拿到指向 tensor_copies[0] 的指针
    bfloat16_t* addr = (bfloat16_t*)qi->buffer.addr;  // [R3] 读 tensor_copies[0].buffer.addr ⚠️
    uint64_t offset = qi->start_offset;                // [R4] 读 tensor_copies[0].start_offset ⚠️
    // 如果 tensor_copies 没被 flush 到 HBM，
    // AICore dcci 读到的是 HBM 中的旧值 → buffer.addr=0 → 访问地址 0 → HANG
```

### 为什么 Case1 没有 flush tensor_copies 但能工作？

**时间窗口效应**：tensor_copies 由 Thread 3（编排器）写入，由 Thread 0/1/2（调度器）dispatch。
中间经历了多个步骤：

```
Thread 3 写 tensor_copies [W1/W2]
    │
    ├── STEP 2: TensorMap lookup (遍历已有 tensor，查 fanin)
    ├── STEP 3: Heap 分配 (可能 stall 等待空间)
    ├── STEP 4: TensorMap insert
    ├── STEP 5: 构建 fanin 链表
    ├── atomic store fanin_count (SEQ_CST)
    │
    │  ···  其他任务也在被编排、提交  ···
    │
    ▼
Thread 0/1/2 发现任务就绪，dispatch [W3]
    │
    ├── build_pto2_payload (读 task->params[i].tensor)
    ├── dc cvac payload
    ├── dc civac handshake
    └── dsb sy
```

在 [W1/W2] 和 [W3] 之间通常有 **数十到数百 us** 的间隔（依赖解析、其他任务编排等）。
AICPU 的 L1/L2 cache 是 write-back 策略，脏 cacheline 会在以下情况被自然逐出到 HBM：

1. **Cache 容量压力**：后续大量内存访问（其他 task 的 tensor_copies、TensorMap 操作等）
   会自然逐出旧的 cacheline
2. **L2 cache 替换策略**：LRU 或 pseudo-LRU，早期写入的 tensor_copies 会被后续访问自然逐出
3. **AICPU 集群内部一致性**：Thread 3 的写和 Thread 0/1/2 的读在同一 AICPU 集群内，
   集群内是 cache-coherent 的，所以 scheduler 线程通过 `task->params[i].tensor` 读到的指针值是正确的

**Case1 能工作的原因**：
- Case1 每 batch 有 `64 × 1 × (2 blocks) = 128` 组 scope，每 scope 提交 5-6 个 task
- 总共 ~16,704 个 task，大量 tensor_copies 写入造成足够的 cache 压力
- 从 submit 到 dispatch 的时间窗口足够长，tensor_copies 已被自然逐出到 HBM

### 什么情况下 tensor_copies 未 flush 会出问题？

| 风险场景 | 说明 | 可能性 |
|---------|------|--------|
| **短依赖链** | 任务 A 的 fanin=0（根任务），submit 后立即可 dispatch，tensor_copies 可能还在 L1 | **高** |
| **大 Tensor 结构体** | head_dim 较大时 Tensor 使用更多 strides/repeats 字段，脏数据量更大 | 中 |
| **低 cache 压力** | 少量任务场景（block_num 较小），cache 不够满不触发自然逐出 | **高** |
| **跨集群调度** | 如果 Thread 3 和 Thread 0 在不同 AICPU 集群（极端配置），无集群内一致性 | 低 |

**特别注意：AIV_HUB 任务是每个 scope 的第一个任务（fanin_count=0），submit 后立即就绪。
如果 Hub 的 tensor_copies（oi, li_update, mi_update 的 buffer.addr=0）还在 cache 中
没有到 HBM，AICore 读到的可能是旧 slot 的残留值。不过 Hub kernel 是空函数，
它的 tensor_copies 只是被下游引用（通过 TensorMap），不被 Hub kernel 自身读取。**

### 优化方案分析

#### 方案 1: 完全不 flush tensor_copies（当前做法）

```
风险:   依赖 AICPU cache 自然逐出，非确定性行为
收益:   节省 ~42 × dc cvac / dispatch = 减少 Dispatch phase ~70% 的 dc 操作
现状:   Case1 (16704 tasks, 长依赖链) 大部分通过
```

#### 方案 2: 每次 dispatch 都 flush 全部 tensor_copies（保守方案）

```
风险:   无正确性风险
代价:   每次 dispatch 额外 ~42 次 dc cvac，Dispatch phase 耗时可能增加 ~5-8 us/task
        循环迭代从 ~64 us 增到 ~100+ us，Tail OH 恶化 ~50%
```

#### 方案 3: 由编排器（Thread 3）在 submit_task 末尾 flush（推荐）

```cpp
// pto_orchestrator.cpp: pto2_submit_task() 末尾
#ifdef __aarch64__
    // Flush tensor_copies to HBM immediately after writing.
    // Scheduler threads on the same AICPU cluster can read via cache coherency,
    // but AICore reads from HBM via dcci — must ensure data is in HBM.
    uintptr_t tc0 = (uintptr_t)task->tensor_copies & ~63ULL;
    uintptr_t tc1 = (uintptr_t)(task->tensor_copies + task->param_count);
    for (uintptr_t a = tc0; a < tc1; a += 64) {
        __asm__ volatile("dc cvac, %0" :: "r"(a) : "memory");
    }
    __asm__ volatile("dsb sy" ::: "memory");
#endif
```

```
优点:   ① tensor_copies 在写入后立即 flush，到 dispatch 时一定在 HBM 中
        ② dsb sy 在编排器线程执行，不阻塞调度器线程 → 不增加 Tail OH
        ③ 编排器的 submit_task 本身就不在关键路径上（它是流水线式提交）
风险:   编排器吞吐量略降（每次 submit 多 ~3-5 us），
        但编排器通常领先调度器很多（orch_drain 只占 0.0%）
```

#### 方案 4: 仅 flush 实际使用的 tensor_copies（精确方案）

```cpp
// 只 flush param_count 个 tensor，而非固定 16 个
for (int i = 0; i < task->param_count; i++) {
    if (task->params[i].tensor == &task->tensor_copies[i]) {
        uintptr_t a = (uintptr_t)&task->tensor_copies[i] & ~63ULL;
        uintptr_t end = (uintptr_t)(&task->tensor_copies[i] + 1);
        for (; a < end; a += 64)
            __asm__ volatile("dc cvac, %0" :: "r"(a) : "memory");
    }
}
```

```
优点:   QK kernel 只有 3 个 tensor param → ~8 CL 而非 42 CL
风险:   代码复杂度增加，需要正确跟踪哪些 param 是 tensor
```

### 总结对比

| 方案 | 正确性 | Tail OH 影响 | 编排器影响 | 复杂度 |
|------|--------|-------------|-----------|--------|
| 1 (不 flush) | ⚠️ 依赖自然逐出，非确定性 | 无 | 无 | 最低 |
| 2 (dispatcher 全 flush) | ✅ | 恶化 ~50% | 无 | 低 |
| **3 (orch flush)** | **✅** | **无** | **轻微 (~3-5 us/submit)** | **低** |
| 4 (精确 flush) | ✅ | 无或极小 | 轻微 | 中 |

**推荐方案 3**：在编排器 submit_task 末尾 flush tensor_copies。
它将 flush 成本从调度器关键路径转移到编排器的非关键路径，
既保证正确性又不增加 Tail OH。

### 附注：tensor_copies 未 flush 的典型表现

当 tensor_copies 未被 flush 到 HBM 时，AICore 通过 dcci 从 HBM 读到的 Tensor.buffer.addr
可能是旧值（0 或上一轮残留地址），导致 kernel 读取到垃圾数据或 NaN，并通过
pipeline (QK → SOFTMAX → PV → UPDATE) 传播到最终输出。

**方案 3（在编排器中 flush tensor_copies）已实现，解决了此类问题。**
