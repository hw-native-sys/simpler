# PR: 异步任务完成与跨卡通知机制 — 实现与设计对齐

## 概述

本 PR 是 [`docs/runtime_async.md`](runtime_async.md) 设计文档的首轮落地实现，在 `tensormap_and_ringbuffer` 运行时中新增：

1. **Deferred Completion（延迟完成）**— 对应设计文档 §2 `complete_in_future` 机制
2. **Notification Counter Gating（通知计数器门控）**— 对应设计文档 §2.4.2 通知计数器协议
3. 两个硬件双卡 demo — 对应设计文档 §3（SDMA 场景）和 §4（AllReduce 通知场景）

## 设计对齐详细对照

### §2.1 `complete_in_future` 属性

| 设计文档 | 本 PR 实现 |
|---|---|
| Task descriptor 新增 `complete_in_future: bool` 字段 | `PTO2TaskPayload::complete_in_future`（`pto_runtime2_types.h`）|
| 默认 `false`，标记为 `true` 的任务延迟完成 | 编排层通过 `PTOParam::complete_in_future = true` 设置，orchestrator 提交任务时写入 payload |

### §2.2 Modified Worker Return Behavior

| 设计文档 | 本 PR 实现 |
|---|---|
| 函数返回时释放 core，但 `complete_in_future` 任务**不调用** `on_task_complete` | `aicpu_executor.cpp` Phase 1：`mixed_complete` 后先调 `async_wait_list.register_deferred()`，若注册成功则跳过 `on_mixed_task_complete`，core 照常释放给下一个 ready task |
| 任务保持 RUNNING 状态 | 已注册到 `PTO2AsyncWaitList` 的任务不进入完成流程，不触发 fanout 传播，不释放 buffer |
| `completed_this_turn` 不计入 | `if (mixed_complete && !slot_state.payload->complete_in_future)` 才递增 `completed_this_turn` |

### §2.3 `waiting_completion_count`

| 设计文档 | 本 PR 实现 |
|---|---|
| 每次注册预期完成事件时递增，每次匹配时递减，归零时调用 `on_task_complete` | `PTO2AsyncWaitEntry::waiting_completion_count`，由 `register_deferred` 从 CQ 读取条件数初始化 |
| 支持多个完成条件组合 | `PTO2AsyncWaitEntry::conditions[PTO2_MAX_COMPLETIONS_PER_TASK]` 数组，每个条件独立轮询、独立递减 |

### §2.4.1 Request/Completion Queue 协议

| 设计文档 API | 本 PR 实现 |
|---|---|
| `pto2_send_request_entry(RQ_TYPE, RQ_ID, *descriptor) → tag` | `pto_rq_kernel_api.h` 中 `pto2_send_request_entry()` — Kernel 侧向硬件引擎提交请求 |
| `pto2_save_expected_completion(CQ_TYPE, CQ_ID, tag, task_id)` | `pto_cq_kernel_api.h` 中 `pto2_save_expected_completion()` — Kernel 侧将 {type, tag, expected_value} 写入 `PTO2CompletionQueue`（设备内存），调度器轮询读取 |

**实现细化**：设计文档中 CQ 条目由调度器直接管理。实现中采用 Kernel→共享内存→调度器 的间接传递：
- Kernel 将完成条件写入 `PTO2CompletionQueue`（`pto_cq_types.h`，在 GM 上）
- 调度器在 `register_deferred` 中读取 CQ 内容，解析为 `PTO2CompletionCondition` 注册到 `PTO2AsyncWaitList`

### §2.4.2 Notification Counter 协议

| 设计文档 API | 本 PR 实现 |
|---|---|
| `pto2_send_notification(REMOTE_ADDR, atomic_op)` | `pto_notify_kernel_api.h` 中 `pto2_send_notification()` — 封装 pto-comm-isa `TNOTIFY` 指令，向远端 RDMA 窗口原子操作 |
| `pto2_save_expected_notification_counter(LOCAL_ADDR, expected, task_id)` | **拆分为两个变体**（见下文） |

**设计细化**：设计文档的 notification counter 是 deferred completion 语义（任务已运行，等计数器到达后完成）。实现中将其拆分为两个独立机制：

| 变体 | 语义 | 实现 |
|---|---|---|
| **Deferred Completion** | 任务已执行，等 CQ 中的 COUNTER 条件满足后完成 | `pto2_save_expected_completion(CQ_TYPE=COUNTER, CQ_ID, counter_addr, expected)` → 通过 `PTO2AsyncWaitList` 轮询 |
| **Pre-launch Gating** | 任务尚未执行，等本地计数器到达后才放行启动 | `pto2_rt_expect_notification_counter(params, counter_addr, expected)` → 通过 `PTO2NotificationWaitList` 轮询 |

Pre-launch gating 是对设计文档的增量扩展：设计文档假设任务先运行再等完成条件，而 `async_notify_demo` 的场景是消费者**在远端通知到达之前不应启动**（避免读到脏数据）。

### §2.5 Scheduler Polling and Completion Resolution

| 设计文档 | 本 PR 实现 |
|---|---|
| 调度器维护两个 watch list，轮询 CQ 和计数器 | 调度循环新增 Phase 0 和 Phase 0b |
| CQ 匹配后 `waiting_completion_count--`，归零调 `on_task_complete` | `PTO2AsyncWaitList::poll_and_complete` — 逐条检查条件，全部满足后调 `on_mixed_task_complete` + deferred release |
| Counter 到达 `expected_value` 后同上 | **Phase 0**: counter 类型条件在 `poll_and_complete` 中一起处理（deferred completion 语义）|
| | **Phase 0b**: `PTO2NotificationWaitList::poll_and_enqueue` — counter 到达后直接入 ReadyQueue（pre-launch gating 语义）|

调度循环整体结构：

```
每次迭代:
  Phase 0:  async_wait_list.poll_and_complete()     ← 设计文档 §2.5 expected_completion_list
  Phase 0b: notification_wait_list.poll_and_enqueue() ← 设计文档 §2.5 expected_notification_counter_list
  Phase 1:  check_running_cores_for_completion()     ← 已有，+register_deferred 判断
  Phase 2:  dispatch_ready_tasks_to_idle_cores()     ← 已有，无修改
```

### §3 Example: SDMA Completion → `async_completion_demo`

| 设计文档 §3 场景 | `async_completion_demo` 实现 |
|---|---|
| Task A: SDMA prefetch，`complete_in_future=True` | Producer (func_id=2): `kernel_producer_async.cpp` — 通过 `TGET_ASYNC` 发起异步远程 RDMA 读取，将完成条件写入 CQ，立即返回 |
| Task B: 消费 tensor_X，等 A 完成后才就绪 | Consumer (func_id=1): `kernel_consumer.cpp` — 依赖 producer 输出，调度器在 SDMA 完成后才放行 |
| Core 释放后可执行其他任务 | Producer 返回即释放 AICore，调度器继续分派其他 ready task |
| 调度器轮询 CQ，匹配后完成 A | Phase 0 `poll_and_complete` 检测 SDMA event 完成 → `on_mixed_task_complete(producer)` → consumer 变 READY |

### §4 Example: Notification Counter → `async_notify_demo`

| 设计文档 §4 场景 | `async_notify_demo` 实现 |
|---|---|
| Task AR: 写本地数据到共享 GM，向所有 peer 发 `ATOMIC_INCREMENT` | Producer (func_id=0): `kernel_producer_notify.cpp` — 计算 `out = in * 2`，然后 `TNOTIFY(AtomicAdd)` 向对端窗口计数器 +1 |
| 等本地计数器达到 expected_value 后完成 | **变体**：本 PR 采用 pre-launch gating 而非 deferred completion — Consumer 提交时声明 `expect_notification_counter(addr, 1)`，调度器在计数器 ≥ 1 前不启动 Consumer |
| 调度器轮询计数器 | Phase 0b `poll_and_enqueue` 每次迭代 `cache_invalidate_range` + 读计数器，到达 1 后将 Consumer 入 ReadyQueue |
| Rank 1 故意延迟 2M 次循环再通知 | `kernel_producer_notify.cpp` line 96-98: `if (my_rank == 1) { for (volatile int i = 0; i < 2000000; ++i) {} }` — 使得时序差异可见 |

## 实现文件清单

### Runtime 核心（对应设计文档 §2）

| 文件 | 设计文档对应 | 职责 |
|---|---|---|
| `runtime/pto_async_wait.h` | §2.3 + §2.5 | `PTO2AsyncWaitList`（wait list + polling）、`PTO2CompletionCondition`、`poll_and_complete`、`register_deferred` |
| `runtime/pto_cq_types.h` | §2.4.1 | `PTO2CompletionQueue`、`PTO2CompletionEntry` — CQ 数据结构 |
| `runtime/pto_cq_kernel_api.h` | §2.4.1 | `pto2_save_expected_completion` — Kernel 侧 CQ 写入 API |
| `runtime/pto_rq_kernel_api.h` | §2.4.1 | `pto2_send_request_entry` — Kernel 侧 RQ 提交 API |
| `runtime/pto_notify_kernel_api.h` | §2.4.2 | `pto2_send_notification`、`pto2_save_expected_notification_counter` — Kernel 侧通知 API |
| `runtime/pto_scheduler.h` | §2.5 | `PTO2NotificationWaitList` — pre-launch gating watch list |
| `runtime/pto_shared_memory.h` | — | `pto2_record_scheduler_error` — 错误报告辅助 |
| `runtime/pto_types.h` | §2.3 | `PTO2AsyncEngine` 枚举、async context 地址 |
| `runtime/pto_runtime2_types.h` | §2.1 | `PTO2TaskPayload::complete_in_future` 字段 |
| `aicpu/aicpu_executor.cpp` | §2.2 + §2.5 | Phase 0/0b 轮询逻辑、`register_deferred` 调用点、core 释放行为修改 |

### 编排层 API（对应设计文档 §2.4.2 扩展）

| 文件 | 新增 API | 说明 |
|---|---|---|
| `orchestration/pto_orchestration_api.h` | `pto2_rt_expect_notification_counter(params, addr, expected)` | 编排侧 pre-launch gating 声明（设计文档的增量扩展）|

### 平台层与 Python 层

| 文件 | 职责 |
|---|---|
| `platform/include/host/comm.h` | 后端无关通信 C API（5 函数）|
| `platform/onboard/host/comm_hccl.cpp` | HCCL 硬件后端（RDMA 窗口分配）|
| `platform/sim/host/comm_sim.cpp` | POSIX 共享内存仿真后端 |
| `platform/include/common/comm_context.h` | `CommDeviceContext` — 设备侧 RDMA 窗口上下文 |
| `examples/scripts/distributed_code_runner.py` | L3 编排器：compile → run → verify |
| `examples/scripts/distributed_worker.py` | per-rank 独立进程 |
| `python/bindings.py` | 分布式 launch 接口 + `RUNTIME_ENV` 注入 |

### 示例

| Demo | 设计文档对应 | 说明 |
|---|---|---|
| `async_completion_demo` | §3 SDMA 场景 | 2 卡，Producer 异步 RDMA 读取 + CQ 延迟完成 |
| `async_notify_demo` | §4 通知计数器场景 | 2 卡，Producer TNOTIFY 跨卡通知 + Consumer pre-launch gating |

## 与设计文档的差异总结

| 设计文档描述 | 实际实现 | 原因 |
|---|---|---|
| Notification counter 统一为 deferred completion（任务已运行，等计数器后完成）| 拆分为 deferred completion + pre-launch gating 两个独立机制 | `async_notify_demo` 场景中消费者应在远端通知到达**之前**就不启动，避免读脏数据。Pre-launch gating 比 deferred completion 更高效（任务无需先运行再等待）|
| CQ 轮询由调度器直接读硬件 CQ | Kernel 将条件写入 GM 上的 `PTO2CompletionQueue`，调度器读 GM | AICPU 调度器无法直接访问 AICore 的硬件 CQ 寄存器，需通过 GM 中转 |
| `pto2_save_expected_notification_counter` 在 Kernel 内调用 | Pre-launch gating 变体 `pto2_rt_expect_notification_counter` 在编排层调用 | 编排层提交任务时即可声明门控条件，无需在 Kernel 内处理 |
| 设计文档 §5.1 列出 4 个新 API | 全部实现，另外新增 `pto2_rt_expect_notification_counter` 编排层 API | 编排层 pre-launch gating 是对设计的补充 |

## 面向后续重构的设计结论：移除 Pre-launch Gating，统一到 Deferred Completion

本节不是对当前 PR 实现的描述，而是针对协作者评审的**后续重构设计**。目标是回答两个问题：

1. 当前 `notification_wait_list` 方案为什么不够理想？
2. 如果要尽量少改 runtime，notification wait 应该建模成什么任务？

### 需要解决的问题

当前实现为了支持 `async_notify_demo`，引入了与 deferred completion 并列的第二套机制：

- 编排层通过 `pto2_rt_expect_notification_counter(params, addr, expected)` 给 task 打上 launch gating 元数据
- orchestrator / scheduler 在 fanin 满足后，不是将 task 放入 ReadyQueue，而是放入 `PTO2NotificationWaitList`
- AICPU 调度循环新增 Phase 0b，轮询计数器并在满足条件后再将 task 入队

这个方案能工作，但有三个问题：

1. **调度语义分裂**
   - 当前 runtime 原本只有一种核心语义：task 是否 READY 由 fanin 决定；task 是否 COMPLETE 由 worker return 或 deferred completion 决定
   - pre-launch gating 额外引入了“fanin 已满足但仍不可调度”的隐藏状态，导致 READY 判定不再统一

2. **重复了一套 polling / wakeup 机制**
   - `PTO2AsyncWaitList` 已经负责“task return 后等待外部条件满足再 complete”
   - `PTO2NotificationWaitList` 又负责“task launch 前等待外部条件满足再 enqueue”
   - 两者都在做“轮询某个外部条件，然后触发 task 状态推进”

3. **侵入 scheduler 热路径**
   - payload / params 里增加了 `has_launch_counter` / `launch_counter_addr` / `launch_counter_expected`
   - orchestrator 和 scheduler 的 ready 判定路径都被改写
   - AICPU 主循环被迫引入 Phase 0b 和单独的 notification polling 锁

从架构分层上看，这不符合 `architecture.md` 中“orchestrator 负责建图、scheduler 负责执行已有图和完成已有 task”的简洁边界。

### 两种候选建模方式

#### 候选 A：保留当前方案，notification wait 作为 launch gating

语义：

- 真正的 consumer task 在编排时就提交
- 但它在 notify 到达前不会被 AICPU dispatch
- notify 到达后，scheduler 再把该 task 放入 ReadyQueue

执行流程：

1. orchestrator 提交 producer 和 consumer
2. consumer 的 fanin 满足后，不进入 ReadyQueue，而进入 `notification_wait_list`
3. AICPU Phase 0b 轮询本地 counter
4. counter 达标后，AICPU 将 consumer 入队
5. consumer 被 dispatch 并运行

优点：

- consumer 自身不会在 notify 前被 dispatch
- 没有额外的 proxy task

缺点：

- scheduler 需要继续维护专用 launch-gating 机制
- READY 判定不再只由 fanin 决定
- notification 相关逻辑散落在 params、payload、orchestrator、scheduler、executor 多处

#### 候选 B：引入显式 notification-wait task，并统一使用 deferred completion

语义：

- orchestrator 显式提交一个 `wait_task`
- `wait_task` 是一个非常短的 task：它只负责注册“本地 counter 达到 expected_value 才 complete”的 deferred completion 条件，然后立即 return
- consumer 不再直接依赖 launch gating，而是依赖 `wait_task` 的真正完成

这里有一个关键约束：**当前 runtime 的依赖关系是通过 TensorMap 上的 tensor producer/consumer 边来发现的**，不是通过显式 `task_id -> task_id` API 建边。

因此，consumer 对 `wait_task` 的依赖，不能只在概念上写成“depends on wait_task”，而要在编排时通过一个**显式 dependency token tensor** 来表达：

- `wait_task` 输出一个 token tensor（可以是 1-element `int32`，仅作为依赖 token 使用）
- `consumer` 把这个 token tensor 作为一个额外 input
- 这样 orchestrator 就会通过 TensorMap 自动建立 `wait_task -> consumer` 的 fanin/fanout 关系

推荐的 DAG 形态：

```text
producer  --------\
                   \
                    -> consumer
                   /
wait_task --------/
```

其中：

- `producer` 负责写本地输出并通过 `TNOTIFY` 更新对端 counter
- `wait_task` 负责“把 notify 条件转换成一个 deferred-completion task”
- `consumer` 同时依赖 `producer` 和 `wait_task`，其中对 `wait_task` 的依赖通过 token tensor 表达

执行流程：

1. orchestrator 预先提交 `producer`、`wait_task`、`consumer`
2. `wait_task` 一旦 READY，就像普通 task 一样被 AICPU dispatch
3. `wait_task` 在 worker 内调用 `pto2_save_expected_notification_counter(...)` 或等价接口，把本地 counter 条件写入 CQ
4. `wait_task` 立即 return，但由于 `complete_in_future=true`，其逻辑状态仍未 COMPLETE
5. AICPU Phase 0 通过现有 `async_wait_list` 轮询该 counter 条件
6. counter 达标后，AICPU 走现有 deferred-completion 路径将 `wait_task` complete
7. `consumer` 只有在 `producer` 和 `wait_task` 都完成后才会 READY

优点：

- 不需要单独的 pre-launch gating 语义
- scheduler 继续只理解两件事：`READY` 和 `deferred completion`
- notification 被建模成 task graph 里的显式节点，依赖关系更清晰
- 更符合“用已有机制表达新语义，而不是额外引入状态机分支”的原则

代价：

- 图里会多一个 proxy task
- 需要定义 notification-wait task 的提交方式和示例写法

### 建议的新 API：用“提交 wait task”替代“给 consumer 打 launch gating 标记”

当前 API：

```cpp
pto2_rt_expect_notification_counter(params_consumer, counter_addr, expected);
pto2_rt_submit_aiv_task(CONSUMER_KERNEL_ID, params_consumer);
```

这个 API 的问题是：它把 notification wait 绑定在 consumer 本身上，因此 runtime 只能在 scheduler 内部增加一个“launch 前门控”的特殊分支。

推荐替换为两层 API：

#### 1. Worker 侧 API：保留现有 counter-based deferred completion

worker 侧不需要引入新的调度语义，只需要继续使用现有 API：

```cpp
pto2_save_expected_notification_counter(cq, local_counter_addr, expected_value);
```

也就是说，worker 的职责只是：

- 接收本地 counter 地址
- 接收 expected value
- 接收 cq 地址
- 将该条件写入 CQ
- 立即 return

#### 2. Orchestration 侧 API：新增“提交 notification-wait task”的 helper

推荐新增一个编排 helper，语义类似：

```cpp
Tensor pto2_rt_submit_notification_wait_task(
    int32_t kernel_id,
    uint64_t local_counter_addr,
    uint32_t expected_value);
```

返回值：

- 一个 token tensor
- 该 token tensor 只用于建依赖边，不承载业务数据

helper 内部做的事情：

1. 创建一个 1-element token tensor
2. 创建 `PTOParam params_wait`
3. `params_wait.add_output(token_tensor)`
4. `params_wait.add_scalar(local_counter_addr)`
5. `params_wait.add_scalar(expected_value)`
6. 分配 `cq_addr = pto2_rt_alloc_cq()`
7. 调用 `pto2_rt_submit_aiv_task_deferred(kernel_id, params_wait, cq_addr)`
8. 返回 token tensor 给 orchestrator

这样，编排侧就从：

```cpp
pto2_rt_expect_notification_counter(params_consumer, counter_addr, expected);
```

变成：

```cpp
Tensor notify_token = pto2_rt_submit_notification_wait_task(
    WAIT_KERNEL_ID, counter_addr, expected);
params_consumer.add_input(notify_token);
```

这比给 consumer 打 launch-gating 标记更符合当前 runtime 的依赖表达方式，因为它显式产出了一个 producer tensor，并让 consumer 像依赖其他中间 tensor 一样依赖它。

### `async_notify_demo` 中推荐的新 API 写法

当前 `async_notify_demo` 的核心逻辑是：

- `producer` 写 `out = in * 2`
- `producer` 对 peer 发 `TNOTIFY(AtomicAdd)`
- `consumer` 读取 `out` 和本地 `notify_counter`
- 现实现通过 `pto2_rt_expect_notification_counter(...)` 对 consumer 做 launch gating

推荐重构后的编排代码形态：

```cpp
uint32_t data_shapes[1] = {128 * 128};
uint32_t token_shapes[1] = {1};

Tensor ext_in = make_tensor_external(in_ptr, data_shapes, 1, DataType::FLOAT32);
Tensor ext_out = make_tensor_external(out_ptr, data_shapes, 1, DataType::FLOAT32);
Tensor ext_result = make_tensor_external(result_ptr, data_shapes, 1, DataType::FLOAT32);

PTOParam params_producer;
params_producer.add_input(ext_in);
params_producer.add_output(ext_out);
params_producer.add_scalar((uint64_t)(uintptr_t)notify_counter_ptr);
params_producer.add_scalar((uint64_t)(uintptr_t)comm_ctx);
pto2_rt_submit_aiv_task(PRODUCER_KERNEL_ID, params_producer);

Tensor notify_token = pto2_rt_submit_notification_wait_task(
    WAIT_KERNEL_ID,
    (uint64_t)(uintptr_t)notify_counter_ptr,
    1);

PTOParam params_consumer;
params_consumer.add_input(ext_out);
params_consumer.add_input(notify_token);
params_consumer.add_output(ext_result);
params_consumer.add_scalar((uint64_t)(uintptr_t)notify_counter_ptr);
pto2_rt_submit_aiv_task(CONSUMER_KERNEL_ID, params_consumer);
```

推荐的 wait kernel 参数形态：

```cpp
args[0] = &Tensor(token_tensor)   // 仅用于依赖建图，可不实际消费
args[1] = local_counter_addr
args[2] = expected_value
args[3] = cq_addr
```

wait kernel 的逻辑：

```cpp
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    __gm__ Tensor* token_tensor = reinterpret_cast<__gm__ Tensor*>(args[0]);
    auto* counter_addr = reinterpret_cast<volatile __gm__ int32_t*>(args[1]);
    uint32_t expected = static_cast<uint32_t>(args[2]);
    uint64_t cq_addr = static_cast<uint64_t>(args[3]);

    volatile __gm__ PTO2CompletionQueue* cq = pto2_cq_get(cq_addr);
    pto2_cq_reset(cq);
    pto2_save_expected_notification_counter(cq, counter_addr, expected);
    pto2_cq_flush(cq);

    (void)token_tensor;
}
```

这里 token tensor 的作用只有一个：让 consumer 能在图上依赖 `wait_task`。它不是业务数据，不要求 consumer kernel 真正读取它。

### AllReduce / Barrier 场景中的 API 形态

在 distributed allreduce/barrier 场景中，这个 API 更容易看清价值。

假设每个 rank 都有：

- `local_reduce_task`：完成本地 partial reduce
- `notify_peers_task`：对所有 peer 执行 `TNOTIFY(AtomicAdd)`
- `barrier_wait_task`：等待本地 counter 达到 `nranks`
- `post_reduce_task`：继续执行后续计算

推荐 DAG：

```text
local_reduce_task --> notify_peers_task --------\
                                                 \
                                                  -> post_reduce_task
                                                 /
barrier_wait_task ------------------------------/
```

其中：

- `notify_peers_task` 负责向所有 peer 发通知
- `barrier_wait_task` 输出一个 barrier token tensor
- `post_reduce_task` 通过 `add_input(barrier_token)` 依赖 barrier 完成

推荐的编排代码形态：

```cpp
Tensor barrier_token = pto2_rt_submit_notification_wait_task(
    BARRIER_WAIT_KERNEL_ID,
    local_barrier_counter_addr,
    nranks);

PTOParam params_post;
params_post.add_input(reduced_tensor);
params_post.add_input(barrier_token);
params_post.add_output(next_tensor);
pto2_rt_submit_aiv_task(POST_KERNEL_ID, params_post);
```

这样 barrier 的语义就很直观：

- `barrier_wait_task` 完成之前，`post_reduce_task` 永远不会 READY
- 但 scheduler 不需要理解“barrier task”这种特殊任务
- 它只需要理解一个普通的 deferred-completion task 和一条普通的 tensor dependency

### 为什么推荐 token tensor，而不是新建 task-to-task 依赖 API

这一节的结论**只针对 `tensormap_and_ringbuffer` runtime**。

理论上也可以设计一个新的 orchestrator API，显式声明：

```cpp
pto2_rt_add_dependency(consumer_task, wait_task);
```

但这会明显更侵入 runtime，因为当前实现：

- 没有稳定暴露“提交后 task handle”的上层 API
- fanin/fanout 的发现依赖 TensorMap 查找 producer tensor
- task ring / slot state / dependency pool 都是围绕 tensor producer lookup 建的

相比之下，token tensor 方案：

- 完全复用现有 TensorMap 依赖构建逻辑
- 不需要引入新的 task handle API
- 不需要引入 scheduler 侧新的依赖边类型

因此，从“最小侵入”原则看，推荐先采用 **wait_task + token tensor**，而不是额外设计 task-to-task dependency 接口。

### `aicpu_build_graph` 的情况（简述）

上面的 token tensor 方案主要是 `tensormap_and_ringbuffer` 的约束结果。`aicpu_build_graph` 已经有显式依赖 API：

- submit 返回 `PTO2TaskId`
- orchestration 可直接调用 `pto2_rt_add_dependency(rt, producer, consumer)`

因此在 `aicpu_build_graph` 中，不需要 token tensor。推荐直接写成：

```cpp
PTO2TaskId t_wait = pto2_rt_submit_notification_wait_task(
    rt, WAIT_KERNEL_ID, local_counter_addr, expected_value);
pto2_rt_add_dependency(rt, t_wait, t_consumer);
```

AllReduce / barrier 场景也同理：`wait_task` 作为普通 deferred-completion task 提交，然后通过 `pto2_rt_add_dependency(...)` 把它连到 post-barrier task 即可。

### 结论：两种 runtime 的推荐表达不同

因此，这一设计结论应该拆成两层：

1. **共性结论**
   - 都推荐把 notification wait 建模成显式 `wait_task + deferred completion`
   - 都不推荐继续依赖 scheduler 专用的 pre-launch gating 语义

2. **runtime-specific 表达方式**
   - 对 `tensormap_and_ringbuffer`：推荐 `wait_task + token tensor`
     - 原因：依赖通过 TensorMap/tensor producer 自动发现
   - 对 `aicpu_build_graph`：推荐 `wait_task + explicit dependency`
     - 原因：submit 返回 `PTO2TaskId`，且已有 `pto2_rt_add_dependency(...)`

也就是说，**wait_task 本身是通用设计，token tensor 不是**。token tensor 只是 `tensormap_and_ringbuffer` 在当前机制下的最小侵入落地方式。

### 为什么推荐候选 B

关键点在于：我们真正需要保证的是**consumer 不得早于 notify 条件满足而运行**，而不是“wait_task 本身绝不能在 notify 前被 dispatch”。

只要 `wait_task`：

- 被标记为 `complete_in_future=true`
- 在 worker 内只做“注册等待条件”而不做实际数据消费
- return 后不触发 `on_task_complete`

那么它即使被提前 dispatch，也不会让 consumer 提前运行。因为 consumer 依赖的是 `wait_task` 的**真正完成**，而不是它的 dispatch 或 return。

换句话说：

- `wait_task` 提前 dispatch：可以
- `wait_task` 提前 return：可以
- `wait_task` 提前 complete：不可以
- `consumer` 提前 READY：不可以

这恰好就是 deferred completion 已经提供的语义。

### AICPU / Scheduler 在推荐方案中的职责

推荐方案下，AICPU **不负责动态创建 task**。它只做两件事：

1. 像平常一样 dispatch 已经由 orchestrator 提交好的 `wait_task`
2. 像平常一样在 async wait path 中轮询它的 completion condition，并在满足时 complete 它

职责分工保持为：

- **Orchestrator**：建图，预先提交 `producer` / `wait_task` / `consumer`
- **AICPU Scheduler**：dispatch ready task；轮询 deferred completions；推动 fanin/fanout
- **AICore Worker**：执行 `wait_task` 的短函数体，只注册 completion condition 后返回

不推荐的做法是：

- AICPU 在 observe 到 notify 后再“新建一个 task”
- 或者 scheduler 维护一条“event → create task → attach dependency → complete task”的新链路

那会引入新的 task creation / dependency hookup / flow-control 复杂度，明显比复用已有 deferred-completion 机制更侵入 runtime。

### 推荐方案对 runtime 的实际改动范围

如果采用候选 B，目标应当是**删除 launch gating 机制，保留并小幅扩展 deferred completion**。

#### 可以删除的内容

- `PTOParam::has_launch_counter`
- `PTOParam::launch_counter_addr`
- `PTOParam::launch_counter_expected`
- `PTO2TaskPayload` 中对应字段
- `pto2_rt_expect_notification_counter(...)`
- `PTO2NotificationWaitList`
- scheduler 中“fanin 满足但不入 ReadyQueue”的分支
- executor 的 Phase 0b notification polling 逻辑

#### 需要保留并扩展的内容

- `complete_in_future`
- per-task CQ
- `PTO2AsyncWaitList`
- `PTO2CompletionType::COUNTER`
- `pto2_save_expected_notification_counter(...)`

#### 需要补强的点

当前 deferred `COUNTER` 路径在语义上已经存在，但要用于 remote notify，还需要补一个关键点：

- **在 deferred-completion 的 counter polling 路径中加入 cache invalidation**

原因：

- `TNOTIFY` / RDMA 对本地 counter 的更新可能绕过 AICPU cache
- 当前 `notification_wait_list` 路径在读 counter 前会显式 `cache_invalidate_range(...)`
- 但 `PTO2AsyncWaitList::test_notification()` 目前只是直接读 `*counter_addr`

因此，如果要把 notify waiting 统一迁移到 deferred completion，必须把 cache visibility 逻辑迁到 `PTO2AsyncWaitList` 的 `COUNTER` polling 路径里。否则 remote notify 可能已经发生，但 AICPU 仍读到 stale counter。

### 推荐方案的示例执行时序

以当前 `async_notify_demo` 为例，推荐的重构后时序应为：

1. Rank N orchestrator 提交：
   - `producer`
   - `wait_task(complete_in_future=true)`
   - `consumer(depends on producer output + wait_task token)`
2. `wait_task` 被 AICPU 像普通 ready task 一样 dispatch
3. `wait_task` worker 将 `{type=COUNTER, addr=local_notify_counter, expected=1}` 写入 CQ
4. `wait_task` return，但保持 incomplete
5. `producer` 正常运行，完成本地输出，并通过 `TNOTIFY` 更新 peer 的 counter
6. Peer 侧 AICPU 在 Phase 0 中轮询 `async_wait_list`
7. 观察到本地 counter 已达标后，complete `wait_task`
8. `consumer` 的 fanin 全部满足，进入 READY
9. `consumer` 被 dispatch，读取的本地数据和 notify 条件都已有效

### 结论

面向后续重构，建议将 notification wait 统一收敛到**显式 wait_task + deferred completion** 模型：

- 不再把 notify 建模成“launch 前的特殊门控状态”
- 而是把 notify 建模成“一个普通 task 的延迟完成条件”

这样做的核心收益是：

- 复用已有 async-completion 机制
- 删除 scheduler 专用分支
- 将复杂度从 scheduler 状态机迁回 task graph 建模
- 更符合本项目现有 runtime 的职责划分

### 供协作者评审的重点问题

在正式改代码前，建议协作者重点确认以下几点：

1. 是否接受 graph 中新增一个显式 `wait_task`，以换取 scheduler 机制简化？
2. `wait_task` 是否应当直接零 fanin 提前提交，还是仍依赖某个本地前置 task？
3. 是否接受使用 token tensor 来表达 `wait_task -> consumer` 的依赖，而不是新增 task-to-task dependency API？
4. 是否同意删除 `notification_wait_list`，将 notify waiting 全部并入 `PTO2AsyncWaitList::COUNTER`？
5. `COUNTER` polling 的 cache invalidation 应放在 `test_notification()` 内部，还是抽成统一 helper？
6. 对 `aicpu_build_graph`，是否应直接采用 `wait_task + pto2_rt_add_dependency(...)`，而不引入 token tensor？
7. `async_notify_demo` 是否应该在重构后作为“deferred counter completion”的主示例，而不是“launch gating”示例？

## 测试

```bash
# 运行两个 async demo（需要 2 张 Ascend 卡）
CANN_ENV_SCRIPT=/path/to/set_env.sh \
  examples/scripts/run_async_tests.sh --devices 6,7
```

## 变更统计

- **41 文件变更**，+4291 / -15 行
- Runtime 核心：~1200 行（async wait list, scheduler, CQ/RQ/notify API, shared memory）
- 平台层：~850 行（comm.h + HCCL/sim 实现 + CommDeviceContext）
- Python 基础设施：~970 行（distributed runner + worker + bindings）
- 示例代码：~1100 行（两个完整 demo）
