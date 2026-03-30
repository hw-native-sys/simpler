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
