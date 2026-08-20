# Device Error Codes

The runtime names, describes and hints every code it latches, at the moment it
fails. This page is what those lines cannot carry: which field to read, who the
code usually blames, and how to chase each family down.

## Start by reading the two lines you already have

The `error hint:` line that sent you here is the second of a pair:

```text
[ERROR] PTO2 runtime failed: orch_error_code=1 sched_error_code=0 runtime_status=-1
[ERROR] error detail: orch_error_code=1 SCOPE_DEADLOCK - tasks submitted in one scope ...
[ERROR] error hint: raise ring_task_window (PTO2_RING_TASK_WINDOW) or split the scope ...
```

`error detail:` gives the **name** and a full description of the mechanism.
`error hint:` gives the **next action**, including the knob to turn. Host-side
CANN failures print the same pair as `ACL error detail:` / `ACL error hint:`.

So the code's meaning and its first remedy are already on your screen. Come here
for the parts that are not: the triage columns below, and the chase procedures.

**First move, always:**

```bash
grep -E "orch_error_code=|sched_error_code=|sub_class=|error detail:" <run log>
```

## How an error reaches you

The `host_build_graph` runtime runs orchestration on the host, transfers the
prepared image to the AICPU, and runs scheduling there. The
`tensormap_and_ringbuffer` runtime runs both orchestration and scheduling on the
AICPU. On a fatal condition the runtime **latches** a code into the shared-memory
header; the host reads it back in `validate_runtime_impl` and prints the lines
above.

At most one of `orch_error_code` (1-11) and `sched_error_code` (100+) is ever
non-zero. `runtime_status` is just the latched code negated.

The native-run exception normally has the form
`RuntimeError: <phase>_native_run failed with code <rc>`, where `<phase>` is
`prepare`, `launch`, `poll`, or `finalize`. When that failure leaves the run lane
terminally poisoned, the same error may carry a
`chip run lane is poisoned:` prefix. The reported `<rc>` is **not** always the
same number:

| Environment | `<rc>` |
| ----------- | ------ |
| **sim** (`a5sim` / `a2a3sim`) | `-N`, where `N` is the latched code. `-1` is SCOPE_DEADLOCK, `-100` is SCHEDULER_TIMEOUT. |
| **onboard** | Usually a CANN watchdog fires first and masks the code as a generic `507018`. **Do not read `<rc>` as the error code here.** |

So on hardware the exception code is the *least* informative thing on screen. The
`error detail:` / `error hint:` pair is printed either way — start there.

## Triage: who the code usually blames

The description and remedy come from the log. What it cannot tell you is which
layer to go looking in, which is what these columns are for.

### Orchestration errors (1-11)

| Code | Name | Whose bug, usually |
| ---- | ---- | ------------------ |
| 1 | SCOPE_DEADLOCK | orchestration |
| 2 | HEAP_RING_DEADLOCK | orchestration / config |
| 3 | FLOW_CONTROL_DEADLOCK | orchestration |
| 4 | DEP_POOL_OVERFLOW | orchestration / config |
| 5 | INVALID_ARGS | orchestration |
| 6 | *(retired — reported as 4)* | — |
| 7 | REQUIRE_SYNC_START_INVALID | orchestration |
| 8 | TENSOR_WAIT_TIMEOUT | kernel / orchestration |
| 9 | EXPLICIT_ORCH_FATAL | orchestration (deliberate) |
| 10 | SCOPE_TASKS_OVERFLOW | runtime-internal |
| 11 | TENSORMAP_OVERFLOW | runtime-internal / extreme scale |

### Scheduler errors (100+)

| Code | Name | Whose bug, usually |
| ---- | ---- | ------------------ |
| 100 | SCHEDULER_TIMEOUT | depends on the sub-class below |
| 101 | ASYNC_COMPLETION_INVALID | kernel (async) |
| 102 | ASYNC_WAIT_OVERFLOW | kernel (async) |
| 103 | ASYNC_REGISTRATION_FAILED | runtime-internal |
| 104 | READY_QUEUE_OVERFLOW | runtime-internal / config |

### SCHEDULER_TIMEOUT sub-classes

Code 100 is a funnel. The runtime sub-classifies it on device and prints the
verdict plus locators, so you rarely need the device log:

```text
[ERROR] PTO2 scheduler timeout sub_class=S1:running-stalled (detail=1) completed=0/1 \
        running=1 ready=0 waiting=0 orch_done=1 stuck_task_id=42 stuck_core=5
```

| Sub-class | Cause |
| --------- | ----- |
| **S1** running-stalled | AICore kernel hang, or a kernel that is merely very slow |
| **S3** ready-but-all-idle | dispatch loop / sync-start gate |
| **S4** dependency-deadlock | dependency wiring |
| **S5** orchestrator-starvation | orchestration stuck upstream |
| **unknown** | runtime bug / memory corruption |

The counters *are* the decision: priority is `running > ready > waiting > orch not
done`. `running=1` is always S1; `running=0, ready>0` is S3.

**Only S1 and S3 are reproducible in practice.** S4, S5 and unknown are defensive
labels — see [Codes with no end-to-end test](device-error-codes/untested.md).

### Host-side CANN codes

These come from the driver, not from the runtime. Unlike the codes above, a CANN
number can reach you from output we do not wrap (a driver log, a raw ACL return),
so the names are worth keeping here rather than only in
`acl/error_codes/rt_error_codes.h`.

| Code | Name |
| ---- | ---- |
| 107022 | ACL_ERROR_RT_DEVICE_TASK_ABORT |
| 207001 | ACL_ERROR_RT_MEMORY_ALLOCATION |
| 507000 | ACL_ERROR_RT_INTERNAL_ERROR |
| 507014 | ACL_ERROR_RT_AICORE_TIMEOUT |
| 507015 | ACL_ERROR_RT_AICORE_EXCEPTION |
| 507017 | ACL_ERROR_RT_AICPU_TIMEOUT |
| 507018 | ACL_ERROR_RT_AICPU_EXCEPTION — **the generic one** |
| 507046 | ACL_ERROR_RT_STREAM_SYNC_TIMEOUT |
| 507899 | ACL_ERROR_RT_DRV_INTERNAL_ERROR |
| -1 | SIMPLER_DEVICE_UNUSABLE (our own sentinel: the DeviceRunner had already marked the device unusable) |

`507018` is **generic**: several unrelated device mechanisms all surface as it.
Never conclude "deadlock" or "OOM" from it alone. `507899` and `107022` are almost
always *fallout* — scroll up for the first failure on that device. For classifying
`507018` against the device log, see the triage table in
[`running-onboard.md`](../../.claude/rules/running-onboard.md).

## Chasing it down

| You have | Go to |
| -------- | ----- |
| a capacity code — 1, 2, 3, 4, 11 | [Capacity](device-error-codes/capacity.md) |
| a stall — 100, or code 8 | [Stalls](device-error-codes/stall.md) |
| a `VEC`/`CUBE` instruction error or UB out of bounds in the device log | [AICore faults](device-error-codes/aicore-fault.md) |
| 10, 11, 103, S4, S5 or unknown | [Codes with no end-to-end test](device-error-codes/untested.md) — and it is a runtime bug |

## Minimal reproductions

Each publicly triggerable code listed below has a live, minimal trigger in
`tests/st/runtime_fatal_codes/`. These are the fastest way to see what a code
looks like, and the shape to copy when you suspect one:

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

## Adding or changing a code

The names, descriptions and hints live in switch statements, and the compiler
enforces coverage. Edit those and the log carries the new code correctly:

| What | Where |
| ---- | ----- |
| runtime code names / descriptions / hints | `src/common/runtime_status/error_names.h` |
| host-side CANN names / descriptions / hints | `src/common/platform/include/host/acl_error_names.h` |
| `SCHEDULER_TIMEOUT` sub-class labels | `src/{arch}/runtime/{host_build_graph,tensormap_and_ringbuffer}/common/pto_runtime_status.h` |
| completeness test | `tests/ut/cpp/common/test_error_code_names.cpp` |

**This page does not need updating for a new code** — deliberately. The tables
above carry only the triage column and the CANN names, neither of which the log
can supply. Everything a reader would otherwise look up here is printed at failure
time, so there is nothing to drift out of sync.

## References

- Code definitions: `src/{arch}/runtime/{host_build_graph,tensormap_and_ringbuffer}/common/pto_runtime_status.h`
- Host print site: `.../host/runtime_maker.cpp` (`validate_runtime_impl`)
- Sub-class logic: `.../runtime/scheduler/scheduler_cold_path.cpp` (`classify_stall_reason`)
- End-to-end negative tests: `tests/st/runtime_fatal_codes/`
- Onboard `507018` mechanism triage + device logs: [`running-onboard.md`](../../.claude/rules/running-onboard.md)
- Timeout defaults: [`local-timeout-defaults.md`](local-timeout-defaults.md)
- Capacity DFX: [`../dfx/scope-stats.md`](../dfx/scope-stats.md)
