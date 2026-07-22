# PTO2 Device Log Profiling Guide

## How to Find Device Logs

AICPU logs (via `LOG_INFO_V9`) are written by CANN's **dlog** subsystem and do **not** appear in the `python test_*.py` / pytest terminal output. They are written to CANN's device log directory:

```text
$HOME/ascend/log/debug/device-<device_id>/device-<pid>_<timestamp>.log
```

Each run produces a new log file (or appends to an existing one). Find the most recent file by modification time:

```bash
ls -lt $HOME/ascend/log/debug/device-<device_id>/ | head -5
```

## Log Structure Overview

A single run produces two profiling blocks in the device log:

| Block | Emitted by | Function | Content |
| ----- | ---------- | -------- | ------- |
| **Orchestrator Profiling** | Thread 3 (orchestrator) | `aicpu_orchestration_entry` | Time breakdown of graph construction on device |
| **PTO2 Scheduler Summary** | Threads 0/1/2 (schedulers) | `SchedulerContext::resolve_and_dispatch` | Per-thread scheduling statistics, phase timing, and lock contention |

All timing values are in microseconds (us), converted from AICPU cycle counters.

---

## Block 1: Orchestrator Profiling

Thread 3 loads the orchestration `.so` via `dlopen`, calls `aicpu_orchestration_entry`, and prints a profiling summary after it returns.

### Example (from a real run: batch=64, 16704 tasks)

```text
Thread 3: Calling aicpu_orchestration_entry from SO
Thread 3: aicpu_orchestration_entry returned, cost 20943.940us
Thread 3: === Orchestrator Profiling: 16704 tasks, total=14601.580us ===
Thread 3:   task+heap_alloc: 1081.900us (7.4%)
Thread 3:   lookup+dep     : 7290.300us (49.9%)
Thread 3:   tensormap_ins  : 1890.380us (12.9%)
Thread 3:   param_copy     : 2147.800us (14.7%)
Thread 3:   fanin+ready    : 1207.400us (8.3%)
Thread 3:   avg/task       : 0.874us
Thread 3: PTO2 total submitted tasks = 16704
```

### Field Reference

| Field | Source (`pto_orchestrator.cpp`) | Description |
| ----- | ------------------------------- | ----------- |
| **cost** | Wall-clock around `orch_func()` call | Total time including orchestration logic + scope overhead |
| **total** | Sum of all sub-steps below | Accumulated time inside `submit_task` across all tasks |
| **task+heap_alloc** | `g_orch_alloc_cycle` | Bump-allocating a task slot plus its packed output buffer |
| **lookup+dep** | `g_orch_lookup_cycle` | TensorMap lookup for inputs/inouts + building fanin/fanout dependency edges |
| **tensormap_ins** | `g_orch_insert_cycle` | Inserting output/inout tensors into the TensorMap |
| **param_copy** | `g_orch_args_cycle` | Copying param descriptors + tensor descriptor copies into task-owned storage |
| **fanin+ready** | `g_orch_fanin_cycle` | Appending dependency edges and finalizing publish-gated fanin state |
| **avg/task** | `total / submit_count` | Average orchestrator time per task submission |

### Interpreting the Numbers

- **cost > total**: The difference is overhead outside `submit_task` (the orchestration user code itself, scope_begin/end, TensorCreateInfo construction, etc.).
- **lookup+dep** is typically the dominant cost (~50%) because it includes TensorMap hash lookups and dependency discovery.
- **param_copy** scales with the number of parameters per task.
- **avg/task < 1us** indicates efficient graph construction.

---

## Block 2: PTO2 Scheduler Summary

Each of the 3 scheduler threads (Thread 0, 1, 2) prints its own summary after completing all tasks. The output has two sub-sections: **summary** and **phase breakdown**.

### Example (Thread 0, from a different run: batch=1, 1044 tasks)

```text
Thread 0: completed=352 tasks in 3477.420us (147 loops, 2.4 tasks/loop)
Thread 0: --- Phase Breakdown ---
Thread 0:   complete:    1485.020us (42.7%)
Thread 0:   scan:        14.400us (0.4%)
Thread 0:   dispatch:    1973.060us (56.7%)
Thread 0:   idle:        4.940us (0.1%)
```

### Summary Line

```text
Thread N: completed=X tasks in Yus (Z loops, W tasks/loop)
```

| Field | Description |
| ----- | ----------- |
| **completed** | Number of tasks this thread processed to completion |
| **Y us** | Total scheduler loop time (sum of all phase cycles) |
| **Z loops** | Number of scheduler loop iterations |
| **W tasks/loop** | Average tasks completed per loop iteration; higher = better throughput |

### Phase Breakdown

The scheduler loop runs four phases each iteration. Each phase's time is accumulated across all loop iterations.

| Phase | What it does | Inline stats |
| ----- | ------------ | ------------ |
| **complete** | Polls handshake on each managed core; when all subtasks complete, `on_task_complete` closes and traverses the task's fanout list | `fanout`: edges traversed for consumer notification |
| **scan** | Updates the perf profiling header with latest scheduler state | — |
| **dispatch** | For each idle core, pops a task from the shape-based ready queue via `get_ready_task(shape)`, builds the dispatch payload, and writes the task to the core's handshake register | `pop`: `hit` = successful pops (task dispatched), `miss` = empty queue pops, `hit_rate` = hit/(hit+miss) |
| **idle** | Scheduler loop iteration where no progress was made (no completions, no dispatches) | — |

**Interpreting phase percentages:**

- **dispatch** is typically the largest (~55-60%) because it includes lock-free ready-queue pops, payload construction, and cache flush (`dc cvac` + `dsb sy`).
- **complete** closes and traverses the fanout list, increments consumer
  `fanin_refcount`, and conditionally pushes newly ready tasks. Producer
  release, consumed transitions, and ring-pointer advancement do not exist in
  the single-shot model.
- **scan** is small (<1%) — only updates the perf header.
- **idle** is negligible when tasks are flowing; high idle% indicates the scheduler is starved.

**Interpreting pop hit_rate:**

- **High hit_rate (>50%)**: Ready queue is well-supplied; dispatch is efficient.
- **Low hit_rate (<10%)**: Ready queue is mostly empty when cores become idle. The bottleneck is upstream (orchestrator submission speed or fanout resolution latency), not dispatch itself.

### Per-Task Averages

Divide each thread's phase times by its `completed` count to get per-task scheduling cost:

| Metric | Formula | Typical value |
| ------ | ------- | ------------- |
| Scheduling overhead per task | total_time / completed | ~5-10 us/task |
| Dispatch per task | dispatch_time / completed | ~3-6 us/task |
| Complete per task | complete_time / completed | ~2-4 us/task |

---

## Cross-Referencing with Host Profiling

When `--enable-l2-swimlane` is used, the host terminal prints a **Task Statistics by Function** table with `Total_Exec` (total AICore kernel execution time). Combined with device log data:

| Metric | Source | Description |
| ------ | ------ | ----------- |
| Avg kernel exec time | `Total_Exec / total_tasks` (host) | Time AICore spends executing each kernel |
| Avg scheduling overhead | `sum(thread_total) / total_tasks` (device log) | Time AICPU spends scheduling each task |
| Sched/Exec ratio | scheduling / execution | Scheduling overhead relative to kernel execution |

A high sched/exec ratio (e.g., >3x) indicates that scheduling overhead dominates, and optimizations should target the scheduler's dispatch hot path (cache flush, payload construction) or upstream task flow.

---

## Quick Reference: Extracting Profiling Data

```bash
# Find the latest device log for device 2
ls -t $HOME/ascend/log/debug/device-2/device-*.log | head -1

# Extract orchestrator profiling (Thread 3)
grep "Thread 3:" <logfile>

# Extract scheduler profiling (Threads 0/1/2)
grep -E "Thread [012]:" <logfile>
```
