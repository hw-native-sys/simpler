# Device Error Codes

What a failing run's error code means, and what is known about chasing each one
down. The host log now names the code it prints, so this page is the long form of
what the `error detail:` / `ACL error detail:` lines already tell you.

## How an error reaches you

The `tensormap_and_ringbuffer` runtime runs orchestration and scheduling on the
AICPU. On a fatal condition the runtime **latches** a code into the shared-memory
header; the host reads it back in `validate_runtime_impl` and prints:

```text
[ERROR] PTO2 runtime failed: orch_error_code=1 sched_error_code=0 runtime_status=-1
[ERROR] error detail: orch_error_code=1 SCOPE_DEADLOCK - tasks submitted in one scope ...
[ERROR] error hint: raise ring_task_window (PTO2_RING_TASK_WINDOW) or split the scope ...
```

At most one of `orch_error_code` (1-11) and `sched_error_code` (100+) is ever
non-zero. `runtime_status` is just the latched code negated.

The exception you see (`RuntimeError: run failed with code <rc>`) is **not** always
the same number:

| Environment | `<rc>` |
| ----------- | ------ |
| **sim** (`a5sim` / `a2a3sim`) | `-N`, where `N` is the latched code. `-1` is SCOPE_DEADLOCK, `-100` is SCHEDULER_TIMEOUT. |
| **onboard** | Usually a CANN watchdog fires first and masks the code as a generic `507018`. **Do not read `<rc>` as the error code here.** |

So on hardware the exception code is the *least* informative thing on screen. The
failure lines above are printed either way — start there.

**First move, always:**

```bash
grep -E "orch_error_code=|sched_error_code=|sub_class=|error detail:" <run log>
```

## 1. Code reference

### Orchestration errors (1-11)

| Code | Name | Meaning | Whose bug, usually |
| ---- | ---- | ------- | ------------------ |
| 1 | SCOPE_DEADLOCK | Tasks in one scope hit the ring task-window cap; slots are not reclaimed until `scope_end` | orchestration |
| 2 | HEAP_RING_DEADLOCK | Ring task slots *and* heap bytes both exhausted | orchestration / config |
| 3 | FLOW_CONTROL_DEADLOCK | Task window blocked while heap is not full (classically: nesting on one ring) | orchestration |
| 4 | DEP_POOL_OVERFLOW | A task's fanin edges overflowed the ring's dependency spill pool | orchestration / config |
| 5 | INVALID_ARGS | An orchestration API rejected its arguments | orchestration |
| 6 | *(retired)* | Was per-task fanin overflow; now reported as 4 | — |
| 7 | REQUIRE_SYNC_START_INVALID | `require_sync_start()` asked for more blocks than the core type physically has | orchestration |
| 8 | TENSOR_WAIT_TIMEOUT | Waiting for tensor data timed out | kernel / orchestration |
| 9 | EXPLICIT_ORCH_FATAL | Orchestration called `rt_report_fatal()` itself | orchestration (deliberate) |
| 10 | SCOPE_TASKS_OVERFLOW | Scope task-record buffer saturated | runtime-internal |
| 11 | TENSORMAP_OVERFLOW | Tensormap entry pool wedged | runtime-internal / extreme scale |

### Scheduler errors (100+)

| Code | Name | Meaning | Whose bug, usually |
| ---- | ---- | ------- | ------------------ |
| 100 | SCHEDULER_TIMEOUT | No forward progress within the timeout. See the sub-class below | depends on sub-class |
| 101 | ASYNC_COMPLETION_INVALID | Malformed async completion condition | kernel (async) |
| 102 | ASYNC_WAIT_OVERFLOW | Async wait list full (in-flight completions over the 64 cap) | kernel (async) |
| 103 | ASYNC_REGISTRATION_FAILED | Illegal async completion message kind | runtime-internal |

### SCHEDULER_TIMEOUT sub-classes

Code 100 is a funnel. The runtime sub-classifies it on device and prints the
verdict plus locators, so you rarely need the device log:

```text
[ERROR] PTO2 scheduler timeout sub_class=S1:running-stalled (detail=1) completed=0/1 \
        running=1 ready=0 waiting=0 orch_done=1 stuck_task_id=42 stuck_core=5
```

| Sub-class | Meaning | Cause |
| --------- | ------- | ----- |
| **S1** running-stalled | A task is on a core and never completes | AICore kernel hang, or a kernel that is merely very slow |
| **S3** ready-but-all-idle | Cores idle, a fanin-satisfied task exists, nothing dispatched | dispatch loop / sync-start gate |
| **S4** dependency-deadlock | Only WAIT tasks remain, fanin never resolves | dependency wiring |
| **S5** orchestrator-starvation | Submitted tasks all done, orchestration not finished | orchestration stuck upstream |
| **unknown** | Bookkeeping invariant violated | runtime bug / memory corruption |

The counters *are* the decision: priority is `running > ready > waiting > orch not
done`. `running=1` is always S1; `running=0, ready>0` is S3.

**Only S1 and S3 are reproducible in practice.** S4, S5 and unknown are defensive
labels — see [4. Codes with no end-to-end test](#4-codes-with-no-end-to-end-test).

### Host-side CANN codes

These come from the driver, not from the runtime. Names are CANN's own
(`acl/error_codes/rt_error_codes.h`).

| Code | Name | Meaning |
| ---- | ---- | ------- |
| 107022 | ACL_ERROR_RT_DEVICE_TASK_ABORT | Task aborted, usually collateral damage from an earlier fault |
| 207001 | ACL_ERROR_RT_MEMORY_ALLOCATION | Device out of memory |
| 507000 | ACL_ERROR_RT_INTERNAL_ERROR | Runtime internal error; on a5 this is how an AICPU op timeout surfaces |
| 507014 | ACL_ERROR_RT_AICORE_TIMEOUT | AICore task exceeded its execution timeout |
| 507015 | ACL_ERROR_RT_AICORE_EXCEPTION | AICore task faulted |
| 507017 | ACL_ERROR_RT_AICPU_TIMEOUT | AICPU task exceeded its execution timeout |
| 507018 | ACL_ERROR_RT_AICPU_EXCEPTION | AICPU task faulted — **the generic one** |
| 507046 | ACL_ERROR_RT_STREAM_SYNC_TIMEOUT | Host timed out on a stream sync |
| 507899 | ACL_ERROR_RT_DRV_INTERNAL_ERROR | Driver internal error; typically a *sticky* error on an already-poisoned context |
| -1 | SIMPLER_DEVICE_UNUSABLE | Our own sentinel: the DeviceRunner had already marked the device unusable |

`507018` is a **generic host-side code**: several unrelated device mechanisms all
surface as it. Never conclude "deadlock" or "OOM" from it alone. `507899` and
`107022` are almost always *fallout* — scroll up for the first failure on that
device. For classifying `507018` against the device log, see the triage table in
[`.claude/rules/running-onboard.md`](../../.claude/rules/running-onboard.md).

## 2. Chasing down a capacity code (1, 2, 4)

These three all say the same thing — "some resource ran out" — without saying
*which* resource or *which* scope. Do not guess at the ring sizes. Turn on
`scope_stats`, which records the high-water mark of all four resources (task-window
slots, heap bytes, dep-pool entries, tensormap entries) per `PTO2_SCOPE`:

```python
cfg = CallConfig()
cfg.enable_scope_stats = True
cfg.output_prefix = "outputs/my_run"
worker.run(callable, args, cfg)
```

It works on a **failing** run: the metadata line is marked `"fatal": true` and
everything written before the fatal is kept. So point it straight at the workload
that trips the code.

| Bottleneck resource | Code | Fix |
| ------------------- | ---- | --- |
| task window | 1 | raise `ring_task_window` (`PTO2_RING_TASK_WINDOW`), or split the scope |
| heap | 2 | raise `ring_heap` (`PTO2_RING_HEAP`), or shrink per-task args / intermediate tensors |
| dep pool | 4 | raise `ring_dep_pool` (`PTO2_RING_DEP_POOL`), or cut the fanin count |

Report fields, the `Top Peaks` table and the plotting tool are documented in
[`../dfx/scope-stats.md`](../dfx/scope-stats.md).

### Minimal reproductions

Each code has a live, minimal trigger in `tests/st/runtime_fatal_codes/`. These are
the fastest way to see what a code looks like, and the shape to copy when you
suspect one:

| Code | How the ST provokes it | Fixture |
| ---- | ---------------------- | ------- |
| 1 | 8 tasks in one scope with `ring_task_window=4` | `kernels/orchestration/scope_deadlock_orch.cpp` |
| 2 | One task with a 32 KiB output against `ring_heap=1024` | `heap_ring_deadlock_orch.cpp` |
| 3 | 8 nested scopes, 3 tasks each — depth ≥3 all land on ring 3 and fill it | `flow_control_deadlock_orch.cpp` |
| 4 | 128 producers into one consumer against `ring_dep_pool=4` | `dep_pool_overflow_orch.cpp` |
| 5 | `set_dependencies(nullptr, 4)` — null pointer, non-zero count | `invalid_args_orch.cpp` |
| 7 | SPMD AIV task asking for 1000 blocks | `require_sync_start_orch.cpp` |
| 8 | `get_tensor_data()` on the output of a `while(true)` kernel | `tensor_wait_timeout_orch.cpp` |
| 9 | `rt_report_fatal(...)` plus post-fatal API calls, to prove they no-op | `explicit_fatal_orch.cpp` |
| 100/S1 | An AIC kernel that spins forever | `aicore_hang_orch.cpp` + `kernels/aic/kernel_hang.cpp` |
| 101 | Kernel defers `PTO2_ERROR_ASYNC_COMPLETION_INVALID` into the slab directly | `kernels/aiv/kernel_async_completion_invalid.cpp` |
| 102 | Kernel registers `MAX_COMPLETIONS_PER_TASK + 1` conditions | `kernels/aiv/kernel_async_wait_overflow.cpp` |

## 3. Chasing down a stall (100, and code 8)

### The timeout race decides what you see

Three watchdogs compete, and whichever fires first determines whether you get a
clean `-100` with a `sub_class`, or a masked `507018`:

- the scheduler no-progress timeout (`PTO2_SCHEDULER_TIMEOUT_MS`),
- the STARS op-execute timeout (`PTO2_OP_EXECUTE_TIMEOUT_US`, ~45 s, kills `aicpu-sd`),
- the host stream-sync timeout (`PTO2_STREAM_SYNC_TIMEOUT_MS`).

**A 45 s op-execute kill is not proof of a deadlock** — the kernel may simply be
slow. To find out which, order the race deliberately. The STs do exactly this, and
the technique transfers:

- To make the **scheduler** win (get a `sub_class`), squeeze it below the others:
  `SCHEDULER=2000ms < OP_EXECUTE=3s < STREAM_SYNC=4000ms` (`aicore_hang` case).
- To let a **slow-but-alive** path finish, push the others out of the way:
  `SCHEDULER=30000ms`, `OP_EXECUTE=30s`, `STREAM_SYNC=40000ms` — this is how the
  `tensor_wait_timeout` case lets the 15 s tensor-data wait land code 8 instead of
  being reaped first.

**These three are read once, at `Worker.init()`.** Changing them between `run()`
calls on the same Worker does nothing — you must rebuild the Worker (or use a
separate process) per value. Defaults and the rationale are in
[`local-timeout-defaults.md`](local-timeout-defaults.md).

### S1: find the stuck kernel

The `sub_class=` line gives you `stuck_task_id` and `stuck_core`. Map the task id
back to your orchestration and look at that kernel for an infinite loop, a wait on
a signal that never arrives, or simply too much work.

When the task id is not enough, raise the log level to **V0** and the device log
prints a task snapshot at the moment of the stall:

```text
[STALL thread=0 idle_iterations=...] TASK ring=1 task_id=42 state=RUNNING \
    fanin_refcount=0/2 kernels=[aic:3 aiv0:7 aiv1:-1] \
    running_on=[owner_thread=0 cores=[core=5(AIC) core=6(AIV0)]]
```

`kernels=[...]` are the kernel ids in the task's three sub-core slots and
`cores=[...]` the physical cores running it — that maps "stuck task" to "which
kernel, on which core".

Setting the level:

- **Worker directly**: `logging.getLogger("simpler").setLevel("V0")` **before**
  `worker.init()` — the level is snapshotted at init and pushed to the device, so
  setting it between `run()` calls has no effect.
- **pytest / scene test**: `--log-level v0`.

V0 is the most verbose level (V0 verbose … V9 terse, default V5). Device logs land
in the shared `~/ascend/log/debug/device-<id>/` by default, where several processes
interleave; redirect them per-run with `ASCEND_PROCESS_LOG_PATH` (the directory must
exist) before reading. See the "Device logs" section of
[`.claude/rules/running-onboard.md`](../../.claude/rules/running-onboard.md).

### Code 8, specifically

The tensor-data wait defaults to 15 s (`PTO2_TENSOR_DATA_TIMEOUT_MS`,
frequency-scaled). It means either the producer never completed, or a consumer never
released its fanout reference. Check for a hung producer first (that is S1 above),
then verify the consumer really declares the dependency and exits. If the kernel is
merely slow, raising the timeout will prove it.

## 4. Codes with no end-to-end test

Three codes and three stall sub-classes have no ST — not from neglect, but because
they cannot be provoked. Recording why, so the next person does not re-derive it
(the argument lives in `tests/st/runtime_fatal_codes/test_runtime_fatal_codes.py`):

- **103 ASYNC_REGISTRATION_FAILED** is structurally pre-empted. The AICore side caps
  at `MAX_COMPLETIONS_PER_TASK` (64) and latches **102** on the 65th condition, so
  the scheduler-side "more than 64" check that would raise 103 never sees enough
  messages. 103 is reachable only by corrupting the slab (UB) or a malformed message.
- **10 SCOPE_TASKS_OVERFLOW** cannot be reached either. `scope_tasks_cap` is the sum
  of the per-ring windows, but each ring physically holds only `window - 1` tasks, so
  all rings together top out at `cap - PTO2_MAX_RING_DEPTH` — strictly below the cap.
  The rings always fill first and latch 1 or 3. Shrinking the window shrinks both, so
  the gap holds.
- **11 TENSORMAP_OVERFLOW** needs the 65536-entry pool (`PTO2_TENSORMAP_POOL_SIZE`,
  compile-time, not tunable via `runtime_env`) flooded *and* a stalled producer
  holding entries.
- **S4 / S5 / unknown**: not expressible through the public API. `set_dependencies()`
  can only name already-submitted tasks, so a dependency cycle cannot be written; a
  stuck producer is RUNNING (→ S1), never pure WAIT (S4). `total_tasks_` counts
  *submitted* tasks, so once they retire `completed == total` and the S5 watchdog
  never fires. They are kept as defensive labels: if a future bug produces the state,
  it gets named instead of mislabelled.

The name tables are held complete for these anyway
(`tests/ut/cpp/common/test_error_code_names.cpp`) — an unreachable code is exactly
the one that would otherwise print as a bare number on the day it finally fires.

If you do hit 10, 11, 103, S4, S5 or unknown, it is a runtime bug. Keep the device
log and report it.

## References

- Code definitions: `src/{arch}/runtime/tensormap_and_ringbuffer/common/pto_runtime_status.h`
- Name/hint tables: `src/common/runtime_status/error_names.h`, `src/common/platform/include/host/acl_error_names.h`
- Host print site: `.../host/runtime_maker.cpp` (`validate_runtime_impl`)
- Sub-class logic: `.../runtime/scheduler/scheduler_cold_path.cpp` (`classify_stall_reason`)
- End-to-end negative tests: `tests/st/runtime_fatal_codes/`
- Onboard `507018` mechanism triage + device logs: [`.claude/rules/running-onboard.md`](../../.claude/rules/running-onboard.md)
- Timeout defaults: [`local-timeout-defaults.md`](local-timeout-defaults.md)
- Capacity DFX: [`../dfx/scope-stats.md`](../dfx/scope-stats.md)
