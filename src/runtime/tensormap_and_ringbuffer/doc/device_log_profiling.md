# PTO2 Device Log Profiling Guide

## How to Find Device Logs

AICPU logs (via `DEV_ALWAYS`) are written by CANN's **dlog** subsystem and do **not** appear in the `run_example.py` terminal output. They are written to CANN's device log directory:

```
$HOME/ascend/log/debug/device-<device_id>/device-<pid>_<timestamp>.log
```

Each run produces a new log file (or appends to an existing one). Find the most recent file by modification time:

```bash
ls -lt $HOME/ascend/log/debug/device-<device_id>/ | head -5
```

## Log Structure Overview

A single run produces two profiling blocks in the device log:

| Block | Emitted by | Function | Content |
|-------|-----------|----------|---------|
| **Orchestrator Profiling** | Thread 3 (orchestrator) | `aicpu_orchestration_entry` | Time breakdown of graph construction on device |
| **PTO2 Scheduler Summary** | Threads 0/1/2 (schedulers) | `resolve_and_dispatch_pto2` | Per-thread scheduling statistics, phase timing, and lock contention |

All timing values are in microseconds (us), converted from AICPU cycle counters.

---

## Block 1: Orchestrator Profiling

Thread 3 loads the orchestration `.so` via `dlopen`, calls `aicpu_orchestration_entry`, and prints a profiling summary after it returns.

### Example (from a real run: batch=64, 16704 tasks)

```
Thread 3: Calling aicpu_orchestration_entry from SO
aicpu_orchestration_entry ">>>>>> batch = 64"
Thread 3: aicpu_orchestration_entry returned, cost 20943.940us
=== Orchestrator Profiling: 16704 tasks, total=14601.580us ===
  sync_tensormap : 286.300us (2.0%)
  task_ring_alloc: 380.400us (2.6%)
  param_copy     : 2147.800us (14.7%)
  lookup+dep     : 7290.300us (49.9%)
  heap_alloc     : 701.500us (4.8%)
  tensormap_ins  : 1890.380us (12.9%)
  fanin+ready    : 1207.400us (8.3%)
  finalize+SM    : 697.500us (4.8%)
  scope_end      : 364.080us
  avg/task       : 0.874us
PTO2 total submitted tasks = 16704
```

### Field Reference

| Field | Source (`pto_orchestrator.cpp`) | Description |
|-------|-------------------------------|-------------|
| **cost** | Wall-clock around `orch_func()` call | Total time including orchestration logic + scope overhead |
| **total** | Sum of all sub-steps below | Accumulated time inside `pto2_submit_task` across all tasks |
| **sync_tensormap** | `g_orch_sync_cycle` | TensorMap validity sync and optional cleanup before each submission |
| **task_ring_alloc** | `g_orch_alloc_cycle` | Allocating a task slot from the task ring buffer |
| **param_copy** | `g_orch_params_cycle` | Copying param descriptors + tensor descriptor copies into task-owned storage |
| **lookup+dep** | `g_orch_lookup_cycle` | TensorMap lookup for inputs/inouts + building fanin/fanout dependency edges |
| **heap_alloc** | `g_orch_heap_cycle` | Allocating packed output buffers from the heap ring |
| **tensormap_ins** | `g_orch_insert_cycle` | Inserting output/inout tensors into the TensorMap |
| **fanin+ready** | `g_orch_fanin_cycle` | Building the fanin list + checking if task is already ready (Step 5/5b) |
| **finalize+SM** | `g_orch_finalize_cycle` | Initializing task in scheduler + updating shared memory `current_task_index` |
| **scope_end** | `g_orch_scope_end_cycle` | `pto2_scope_end` overhead (notifying scheduler of scope completion) |
| **avg/task** | `total / submit_count` | Average orchestrator time per task submission |

### Interpreting the Numbers

- **cost > total**: The difference is overhead outside `pto2_submit_task` (the orchestration user code itself, scope_begin/end, make_tensor calls, etc.).
- **lookup+dep** is typically the dominant cost (~50%) because it involves TensorMap hash lookups and building dependency edges with spinlock-protected fanout list insertions.
- **param_copy** scales with the number of parameters per task.
- **avg/task < 1us** indicates efficient graph construction.

---

## Block 2: PTO2 Scheduler Summary

Each of the 3 scheduler threads (Thread 0, 1, 2) prints its own summary after completing all tasks. The output has three sub-sections: **summary**, **phase breakdown**, and **lock contention**.

### Example (Thread 0, from the same run)

```
Thread 0: === PTO2 Scheduler Summary ===
Thread 0: completed=6068 tasks in 31398us (977 loops, 6.2 tasks/loop)
Thread 0: --- Phase Breakdown (execution order) ---
Thread 0:   scan:            2295us ( 7.3%)
Thread 0:   early_ready:       77us ( 0.2%)  (deps already met at submit time)
Thread 0:   complete:       11374us (36.2%)  [fanout: edges=7578, max_degree=20, avg=1.2]
Thread 0:   dispatch:       17651us (56.2%)  [steal: own=4443, steal=1625, pct=26.8%]
Thread 0: --- Lock Contention (ready_q) ---
Thread 0:   total:         wait= 8366us hold= 4144us
Thread 0:   scan:          wait=  318us hold=  704us
Thread 0:   early_ready:   wait=    0us hold=    0us
Thread 0:   complete:      wait= 1374us hold=  781us
Thread 0:   dispatch:      wait= 6674us hold= 2659us
Thread 0:     hit:         wait= 1361us hold=  551us (dequeued task)
Thread 0:     miss:        wait= 5313us hold= 2108us (empty queue)
```

### Summary Line

```
Thread N: completed=X tasks in Yus (Z loops, W tasks/loop)
```

| Field | Description |
|-------|-------------|
| **completed** | Number of tasks this thread processed to completion |
| **Y us** | Total scheduler loop time (sum of all phase cycles) |
| **Z loops** | Number of scheduler loop iterations |
| **W tasks/loop** | Average tasks completed per loop iteration; higher = better throughput |

### Phase Breakdown

The scheduler loop runs four phases in order each iteration. Each phase's time is accumulated across all loop iterations.

| Phase | What it does | Inline stats |
|-------|-------------|-------------|
| **scan** | Scans newly submitted tasks in shared memory; enqueues root tasks (those with `fanin_count == 0`) into the ready queue | — |
| **early_ready** | Drains the orchestrator's ready queue (tasks whose dependencies were all satisfied at submit time, detected via Step 5b in `pto2_submit_task`) | — |
| **complete** | Polls register `COND` on each managed core; when a core becomes `IDLE`, traverses the completed task's fanout list, increments consumer refcounts, and enqueues newly ready consumers | `edges`: total fanout edges traversed; `max_degree`: largest fanout list; `avg`: average fanout per completed task |
| **dispatch** | For each idle core, pops a task from the ready queue (own shard first, then work-stealing from other shards), builds the dispatch payload, and writes the task to the core's register | `own`: tasks dequeued from own shard; `steal`: tasks stolen from other shards; `pct`: steal percentage |

**Interpreting phase percentages:**

- **dispatch** is typically the largest (~55%) because it includes both ready-queue pops (with lock contention) and the actual register writes + payload construction.
- **complete** is the second largest (~36%) because fanout traversal involves atomic operations (`SEQ_CST` fetch_add on refcounts) and conditional ready-queue pushes.
- **scan** is small (~7%) — it only runs until all submitted tasks have been scanned.
- **early_ready** is negligible in most cases.

### Lock Contention (ready_q)

Ready queues are sharded (one shard per scheduler thread). Access is protected by per-shard spinlocks. This section reports cumulative lock **wait** (time spinning to acquire) and **hold** (time from acquire to release) for each phase.

```
Thread N:   total:         wait=Xus hold=Yus        # sum across all phases
Thread N:   scan:          wait=Xus hold=Yus        # lock during root-task enqueue
Thread N:   early_ready:   wait=Xus hold=Yus        # lock during orch-ready drain
Thread N:   complete:      wait=Xus hold=Yus        # lock during fanout push
Thread N:   dispatch:      wait=Xus hold=Yus        # lock during ready-queue pop (sum of hit+miss)
Thread N:     hit:         wait=Xus hold=Yus        # pop attempts that dequeued a task
Thread N:     miss:        wait=Xus hold=Yus        # pop attempts on empty queue
```

**Key observations from the example:**

| Metric | Thread 0 | Thread 1 | Thread 2 |
|--------|----------|----------|----------|
| completed | 6068 | 5997 | 4639 |
| total time (us) | 31398 | 31410 | 31406 |
| dispatch % | 56.2% | 56.0% | 60.5% |
| complete % | 36.2% | 36.7% | 35.4% |
| steal % | 26.8% | 25.8% | 41.4% |
| lock wait (us) | 8366 | 9176 | 8807 |
| lock hold (us) | 4144 | 4688 | 3971 |

- **Lock contention is moderate**: total wait ~8-9ms out of ~31ms total time (~27-30%).
- **dispatch.miss dominates wait time**: most lock wait comes from polling empty queues, not actual contention. Thread 0's dispatch miss wait = 5313us vs hit wait = 1361us.
- **Work stealing is active**: Thread 2 steals 41.4% of its tasks, indicating it finishes its own shard's tasks faster and helps drain other shards.
- **Threads are well-balanced**: all three complete within ~31ms, despite different task counts (Thread 2 has fewer tasks but higher steal rate).

### Per-Task Averages

Divide each thread's phase times by its `completed` count to get per-task scheduling cost:

| Metric | Formula | Typical value |
|--------|---------|---------------|
| Scheduling overhead per task | total_time / completed | ~5-7 us/task |
| Lock overhead per task | (total_wait + total_hold) / completed | ~1.5-2.5 us/task |
| Dispatch per task | dispatch_time / completed | ~3-4 us/task |
| Complete per task | complete_time / completed | ~2-3 us/task |

---

## Cross-Referencing with Host Profiling

When `--enable-profiling` is used, the host terminal prints a **Task Statistics by Function** table with `Total_Exec` (total AICore kernel execution time). Combined with device log data:

| Metric | Source | Description |
|--------|--------|-------------|
| Avg kernel exec time | `Total_Exec / total_tasks` (host) | Time AICore spends executing each kernel |
| Avg scheduling overhead | `sum(thread_total) / total_tasks` (device log) | Time AICPU spends scheduling each task |
| Sched/Exec ratio | scheduling / execution | Scheduling overhead relative to kernel execution |

A high sched/exec ratio (e.g., >3x) indicates that scheduling overhead dominates, and optimizations should target the scheduler's non-lock paths (dispatch polling, fanout traversal) before reducing lock contention.

---

## Quick Reference: Extracting Profiling Data

```bash
# Find the latest device log for device 2
ls -t $HOME/ascend/log/debug/device-2/device-*.log | head -1

# Extract orchestrator profiling
grep -E "Orchestrator Profiling|sync_tensormap|task_ring_alloc|param_copy|lookup\+dep|heap_alloc|tensormap_ins|fanin\+ready|finalize\+SM|scope_end|avg/task" <logfile>

# Extract scheduler summary
grep -E "Scheduler Summary|completed=|Phase Breakdown|scan:|early_ready:|complete:|dispatch:" <logfile>

# Extract lock contention
grep -E "Lock Contention|total:|scan:|early_ready:|complete:|dispatch:|hit:|miss:" <logfile>
```
