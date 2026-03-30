# Runtime Extension for Asynchronous Hardware Engine Requests

## 1. Background: Current Runtime Model

In the current pypto runtime design, each level of the hierarchy has three roles:

| Role | Responsibility |
|---|---|
| **Orchestrator** | Submits tasks to workers at the current level, or to lower-level orchestrators |
| **Scheduler** | Manages task readiness, dependency tracking, and buffer lifecycle |
| **Worker** | Executes worker functions (tasks), typically run-to-completion |

### 1.1 Run-to-Completion Task Lifecycle

A typical task is a **run-to-completion function**. When the worker function returns, the scheduler calls `pto2_scheduler_on_task_complete`, which performs two dependency-tracking operations:

1. **Fanout propagation (consumer readiness)**: Walks the completing task's fanout list and increments each consumer task's `fanin_refcount`. When a consumer's `fanin_refcount` equals its `fanin_count`, that consumer transitions to READY and is placed in the ready queue.

2. **Fanin retirement (producer release)**: Walks the completing task's fanin list and increments each producer's `fanout_refcount`. When a producer's `fanout_refcount` equals its `fanout_count`, the producer transitions to CONSUMED and its output buffers become eligible for release.

Note that **fanin, fanout, and `ref_count` are tracked at the task level** (producers and consumers), not at the individual tensor level.

### 1.2 The Asynchronous Hardware Engine Challenge

This run-to-completion model assumes that **function return = task complete**. This creates a fundamental problem for worker functions that submit requests to asynchronous hardware engines:

| Hardware Engine | Function |
|---|---|
| **SDMA** | System DMA — bulk data movement between memory regions |
| **RoCE** | RDMA over Converged Ethernet — inter-node network data transfer |
| **UMA** | Unified Memory Access — cross-die or cross-chip memory operations |
| **CCU** | Cache Coherence Unit — cache management and coherence operations |

When a worker function submits a request to one of these engines and then returns, the hardware engine may still be:
- **Reading** from the task's IN parameters (the buffer must not be released yet), or
- **Writing** to the task's INOUT/OUT parameters (the data is not yet valid for consumers).

If the scheduler calls `pto2_scheduler_on_task_complete` at function return time, it would prematurely release producer buffers and unblock consumer tasks before the hardware operation finishes — leading to data corruption or races.

**Goal**: Keep the existing runtime mechanisms for task lifecycle management, buffer management, dependency resolution, and task scheduling intact, while adding the ability to defer task completion until asynchronous hardware operations finish.

---

## 2. Design: `pl.complete_in_future` and Deferred Completion

### 2.1 New Function Attribute: `pl.complete_in_future`

A new optional attribute is added to the `pl.function` definition:

```python
@pl.function(complete_in_future=True)
def sdma_prefetch(src_tensor, dst_tensor):
    ...
```

By default, `complete_in_future` is `False` (standard run-to-completion). Functions that submit asynchronous hardware requests or rely on external completion signals should be marked with `complete_in_future=True`.

### 2.2 Modified Worker Return Behavior

When a worker function returns, the runtime performs the following:

```
on worker function return(task_id):
    (a) Release the core / worker thread → available to execute the next ready task

    (b) if task.complete_in_future:
            // Do NOT call pto2_scheduler_on_task_complete.
            // Task remains in RUNNING state (logically incomplete).
        else:
            pto2_scheduler_on_task_complete(sched, task_id)   // standard path
```

A `complete_in_future` task's function return **releases the core** but does **not complete the task**. The scheduler keeps the task in RUNNING state. Dependency propagation and buffer release are deferred.

### 2.3 Task Descriptor Extensions

Two fields are added to `PTO2TaskDescriptor`:

| Field | Type | Default | Description |
|---|---|---|---|
| `complete_in_future` | `bool` | `false` | Whether this task defers completion beyond function return |
| `waiting_completion_count` | `int32_t` | `0` | Number of outstanding completion events before the task is truly complete |

The `waiting_completion_count` is incremented each time the task registers an expected completion event (via the APIs below). When the count reaches zero, the runtime calls `pto2_scheduler_on_task_complete`.

### 2.4 New Runtime APIs

Four new APIs are introduced, called from within the worker function body:

#### 2.4.1 Request/Completion Queue Protocol

```c
tag = pto2_send_request_entry(RQ_TYPE, RQ_ID, *descriptor);
success = pto2_save_expected_completion(CQ_TYPE, CQ_ID, tag, task_id);
```

| Parameter | Description |
|---|---|
| `RQ_TYPE` / `CQ_TYPE` | Engine type: `SDMA`, `RoCE`, `UMA`, `CCU`, etc. |
| `RQ_ID` | Index of the request queue for the given engine type |
| `CQ_ID` | Index of the completion queue for the given engine type |
| `descriptor` | Engine-specific request descriptor (DMA address, length, etc.) |
| `tag` | Unique handle returned by `pto2_send_request_entry`, used to match the completion entry |
| `task_id` | The task that should be completed when this tag appears in the CQ |

**Workflow**: The worker function calls `pto2_send_request_entry` to submit a request to a hardware engine. The returned `tag` uniquely identifies this request. The worker then calls `pto2_save_expected_completion` to register this tag in the scheduler's **expected completion list**. This also increments `waiting_completion_count` for the task.

#### 2.4.2 Notification Counter Protocol

```c
pto2_send_notification(REMOTE_NOTIFICATION_COUNTER_ADDRESS, atomic_op);
pto2_save_expected_notification_counter(LOCAL_NOTIFICATION_COUNTER_ADDRESS, expected_value, task_id);
```

| Parameter | Description |
|---|---|
| `REMOTE_NOTIFICATION_COUNTER_ADDRESS` | Memory address of a counter on a remote node (or local) |
| `atomic_op` | Atomic operation to perform (e.g., `ATOMIC_INCREMENT`) |
| `LOCAL_NOTIFICATION_COUNTER_ADDRESS` | Memory address of a counter on the local node |
| `expected_value` | The value at which the counter triggers completion |
| `task_id` | The task that should be completed when the counter reaches `expected_value` |

**Workflow**: `pto2_send_notification` performs a remote atomic memory operation on the target counter address. `pto2_save_expected_notification_counter` registers the local counter in the scheduler's **expected notification counter list** and increments `waiting_completion_count` for the task.

### 2.5 Scheduler Polling and Completion Resolution

The runtime scheduler maintains two watch lists:

1. **Expected completion list**: Entries of the form `{CQ_TYPE, CQ_ID, tag, task_id}`
2. **Expected notification counter list**: Entries of the form `{counter_address, expected_value, task_id}`

When either list is non-empty, the scheduler **polls** the corresponding completion queues and counter addresses:

```
scheduler_poll_loop:
    for each entry in expected_completion_list:
        if CQ[entry.CQ_TYPE][entry.CQ_ID] contains entry.tag:
            remove entry from list
            task = get_task(entry.task_id)
            task.waiting_completion_count--
            if task.waiting_completion_count == 0:
                pto2_scheduler_on_task_complete(sched, entry.task_id)

    for each entry in expected_notification_counter_list:
        if *entry.counter_address >= entry.expected_value:
            remove entry from list
            task = get_task(entry.task_id)
            task.waiting_completion_count--
            if task.waiting_completion_count == 0:
                pto2_scheduler_on_task_complete(sched, entry.task_id)
```

**Multiple completion conditions**: A single task may register multiple expected completions and/or notification counters. Each registration increments `waiting_completion_count`. Each match decrements it. The task completes only when `waiting_completion_count` reaches zero — i.e., **all** registered conditions are satisfied.

---

## 3. Example: Task Waiting for SDMA Completion

A data prefetch scenario: Task A uses SDMA to move a tensor from host memory to device memory. Task B is a compute task that consumes this tensor.

### 3.1 Task DAG

```
Task A (SDMA prefetch, complete_in_future=True)
  OUT: tensor_X ───fanout───▶ Task B (compute, run-to-completion)
                                IN: tensor_X
```

### 3.2 Timeline

```
Worker Core
──────────────────────────────────────────────────────────────────▶ time
│       Task A                       │     Task C (unrelated)    │
│ 1. Build SDMA descriptor          │     (core reused)         │
│ 2. tag = pto2_send_request_entry(  │                           │
│          SDMA, rq_id, &desc)       │                           │
│ 3. pto2_save_expected_completion(  │                           │
│          SDMA, cq_id, tag, A)      │                           │
│ 4. return → core released          │                           │
├────────────────────────────────────┼───────────────────────────┤
  Task A status: RUNNING (not COMPLETED)

SDMA Engine
──────────────────────────────────────────────────────────────────▶ time
│        DMA transfer: host mem ──────────▶ device mem           │
│                                      completion entry posted ──┤

Scheduler
──────────────────────────────────────────────────────────────────▶ time
│ Watch list: [{SDMA, cq_id, tag, A}]                            │
│ ...poll... CQ match found!                                     │
│ A.waiting_completion_count-- → 0                               │
│ → pto2_scheduler_on_task_complete(A)                           │
│   fanout: B.fanin_refcount++ → B becomes READY                │

Worker Core
──────────────────────────────────────────────────────────────────▶ time
                                              │     Task B       │
                                              │  compute on      │
                                              │  tensor_X (valid)│
```

### 3.3 Key Observations

1. **Core reuse**: Task A returns at step 4 and the core immediately picks up Task C. No core time is wasted waiting for DMA.

2. **Task A stays in RUNNING state**: The scheduler does not call `pto2_scheduler_on_task_complete` at function return. Task A's OUT buffer (`tensor_X`) is not yet marked valid.

3. **Task B is safely blocked**: Task B's `fanin_refcount < fanin_count` because Task A has not completed. Task B remains PENDING — it cannot execute on stale or partially-written data.

4. **Deferred completion**: Only when the scheduler's polling loop detects the SDMA completion entry matching `tag` does it trigger `pto2_scheduler_on_task_complete(A)`, which propagates readiness to Task B through the standard fanout mechanism.

5. **Data integrity**: The DMA transfer is guaranteed complete before Task B reads `tensor_X`. The runtime's existing dependency tracking is fully preserved — only the **timing** of the completion call is changed.

---

## 4. Example: Task Waiting for Notification Counter (AllReduce)

An AllReduce operation across 4 nodes. Each node contributes a partial result and must wait for all peers to finish before the reduced result is valid. Each node sends a notification to all peers upon completing its local contribution, and waits for its local counter to reach the expected value (4 = one increment per node).

### 4.1 Task DAG (Node 0)

```
Task P (local partial reduce, run-to-completion)
  OUT: partial ───fanout───▶ Task AR (allreduce exchange, complete_in_future=True)
                               INOUT: partial
                               OUT: reduced ───fanout───▶ Task C (post-reduce compute)
                                                           IN: reduced
```

### 4.2 Timeline (Node 0)

```
Worker Core
──────────────────────────────────────────────────────────────────▶ time
│   Task P              │  Task AR                               │  Task D (unrelated)
│ local partial reduce  │ 1. Write local partial to shared GM    │  (core reused)
│ → on_task_complete(P) │ 2. For each peer (including self):     │
│   → AR becomes READY  │      pto2_send_notification(           │
│                       │        peer.COUNTER_ADDR,              │
│                       │        ATOMIC_INCREMENT)               │
│                       │ 3. pto2_save_expected_notification_    │
│                       │      counter(MY_COUNTER_ADDR, 4, AR)  │
│                       │ 4. return → core released              │
├───────────────────────┼────────────────────────────────────────┤
  Task AR status: RUNNING (not COMPLETED)

Node 0's Notification Counter
──────────────────────────────────────────────────────────────────▶ time
  value: 0
     +1 (self)  → 1
         +1 (Node 1) → 2
             +1 (Node 2) → 3
                 +1 (Node 3) → 4  ← matches expected_value

Scheduler
──────────────────────────────────────────────────────────────────▶ time
│ Watch list: [{MY_COUNTER_ADDR, expected=4, AR}]                │
│ ...poll... counter == 4, match found!                          │
│ AR.waiting_completion_count-- → 0                              │
│ → pto2_scheduler_on_task_complete(AR)                          │
│   fanout: C.fanin_refcount++ → C becomes READY                │
│   fanin:  P.fanout_refcount++ → P may become CONSUMED         │

Worker Core
──────────────────────────────────────────────────────────────────▶ time
                                                  │    Task C     │
                                                  │ post-reduce   │
                                                  │ compute on    │
                                                  │ reduced (valid│
                                                  │ from all 4    │
                                                  │ nodes)        │
```

### 4.3 Key Observations

1. **Distributed barrier without blocking**: The notification counter acts as a barrier across 4 nodes. Each node atomically increments counters on all peers when its local work is done. No core spins or blocks — the scheduler polls the counter asynchronously.

2. **Core reuse**: Task AR returns immediately after sending notifications and registering the expected counter. The core proceeds to execute Task D.

3. **Task C is safely gated**: `pto2_scheduler_on_task_complete(AR)` is not called until the counter reaches 4. Task C cannot become READY until all nodes have contributed. The `reduced` output is guaranteed to reflect the fully-reduced result.

4. **Producer buffer lifetime**: Task P's `partial` buffer is used as INOUT by Task AR. The runtime does not release this buffer until Task AR completes (via fanin retirement in `pto2_scheduler_on_task_complete`). This keeps the buffer alive while remote nodes may still be reading it via RDMA.

5. **Composable with CQ events**: If Task AR also submits an SDMA request (e.g., to move the reduced result to a different memory region), it can call both `pto2_save_expected_completion` and `pto2_save_expected_notification_counter`. The `waiting_completion_count` starts at 2. Each event independently decrements the count. Task AR completes only when **both** the notification counter reaches 4 **and** the SDMA CQ entry arrives.

---

## 5. Conclusion

### 5.1 Summary of Changes

This extension introduces a minimal set of additions to the pypto system, spanning the frontend language and the runtime layer:

**PyPTO Frontend (`pl.` DSL)**:

| Addition | Description |
|---|---|
| `pl.function(complete_in_future=True)` | New optional attribute on `pl.function`. Marks a worker function whose task completion is deferred beyond function return. Default is `False` (standard run-to-completion). |

**PTO Runtime — New APIs**:

| API | Purpose |
|---|---|
| `pto2_send_request_entry(RQ_TYPE, RQ_ID, *descriptor) → tag` | Submit a request to a hardware engine's request queue. Returns a unique `tag` identifying the request. |
| `pto2_save_expected_completion(CQ_TYPE, CQ_ID, tag, task_id)` | Register an expected completion queue entry. The scheduler polls the CQ for the matching `tag` and defers task completion until it arrives. |
| `pto2_send_notification(REMOTE_COUNTER_ADDR, atomic_op)` | Perform a remote atomic operation on a notification counter (e.g., increment a counter on a peer node). |
| `pto2_save_expected_notification_counter(LOCAL_COUNTER_ADDR, expected_value, task_id)` | Register an expected notification counter value. The scheduler polls the local counter and defers task completion until it reaches the expected value. |

**PTO Runtime — Task Descriptor Extension**:

| Field | Type | Default | Description |
|---|---|---|---|
| `complete_in_future` | `bool` | `false` | If `true`, the task's function return releases the core but does not trigger `pto2_scheduler_on_task_complete`. |
| `waiting_completion_count` | `int32_t` | `0` | Number of outstanding completion conditions. Incremented by each `pto2_save_expected_completion` or `pto2_save_expected_notification_counter` call. Decremented by the scheduler when a match is detected. Task completes when this reaches zero. |

### 5.2 Core Runtime Mechanisms Remain Intact

This enhancement is designed to be **non-invasive** to the existing pypto runtime architecture. The following core mechanisms are completely unchanged:

- **Task ring** — allocation, slot management, and ring pointer advancement
- **Dependency tracking** — fanin/fanout linked lists, `fanin_refcount` / `fanout_refcount` protocol
- **`pto2_scheduler_on_task_complete`** — the three-step completion propagation (fanout propagation → fanin retirement → consumed check) is identical; only the **trigger point** is changed from "function return" to "all completion conditions satisfied"
- **Buffer lifecycle** — heap ring allocation, scope-based lifetime, and fanout-driven release
- **Ready queue** — enqueue/dequeue protocol and worker dispatch
- **Orchestrator/Scheduler/Worker roles** — unchanged at every level

The only behavioral change is: for `complete_in_future` tasks, the call to `pto2_scheduler_on_task_complete` is **deferred** from the worker thread (at function return) to the scheduler thread (when all registered completion conditions are met). The function itself is called with the same arguments and produces the same effects.

### 5.3 Applicability Across All Hierarchy Levels (L2–L7)

This scheme is not limited to L2 (hardware core level). It applies uniformly across the entire hierarchical runtime system:

| Level | Typical Async Use Case |
|---|---|
| **L2** (InCore) | SDMA transfers, CCU cache operations, TPUSH/TPOP hardware flag waits |
| **L3** (Server) | UMA cross-die memory operations, local SDMA between NPU chips |
| **L4** (Pod) | RoCE RDMA transfers between servers within a pod |
| **L5** (Service Pool) | Cross-pod data movement, distributed AllReduce notification barriers |
| **L6** (Cluster) | Inter-cluster data synchronization, federated aggregation barriers |
| **L7** (Global) | Cross-datacenter transfers, global notification barriers |

At every level, the runtime's `LevelRuntime::on_task_complete` (L3–L7) or `pto2_scheduler_on_task_complete` (L2) follows the same deferred-completion protocol: check `complete_in_future`, poll the watch lists, decrement `waiting_completion_count`, and trigger the standard completion path only when all conditions are satisfied. This provides a **single, unified mechanism** for managing asynchronous hardware operations at any scale — from a single DMA transfer on one chip to a global multi-node synchronization barrier.
