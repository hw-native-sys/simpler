# a5 AICore op-timeout poisons the shared L2 worker for the rest of an xdist session

**Date**: 2026-06-08
**Verdict**: contained (two independent guards landed — driver fail-fast +
fixture rebuild-then-skip). The poison itself is unrecoverable in-process on
a5; containment stops it from cascading.

## Question

CI run 27090242388, job `st-onboard-a5` (branch `swimlane-queue-depth-viz`,
PR #1000 — PR only touches a2a3 + tooling, so the failure is **not** PR
code). On the gw0 xdist worker:

```text
10:47:04  standalone test_aicore_op_timeout ... PASS (separate process, dev=[5])
10:48:17  TestPagedAttentionUnrollManualScope  rtKernelLaunchWithHandleV2 failed: 207001  <- trigger
10:48:22..58  7 subsequent gw0 tests all fail   rtMalloc failed: 507899 (ChipCallable buffer)
```

gw1 (other device) ran clean throughout. One device-side error took down
every remaining test on the worker. The hypothesis to check: an AICore
op-timeout (or a launch failure) leaves the ACL/device context in a
sticky-error state, and because the L2 `st_worker` pool reuses **one**
`ChipWorker` per (runtime, device) for the life of an xdist worker process
(`conftest.py::_l2_worker_pool`), the poison is never cleared — every later
test inherits it.

## What was tried

All runs on a dedicated, exclusively-locked a5 device
(`Ascend950PR_9599`) via `task-submit --device auto` (see
`.claude/rules/running-onboard.md`). Three experiments:

1. **Cascade variant** — `aicore_op_timeout` (standalone subprocess) then
   `paged_attention_unroll_manual_scope` (L2 subprocess), 10×, exactly the
   CI ordering.
2. **Control** — `paged_attention_unroll_manual_scope` alone, 10×.
3. **Within-process probe** (`/tmp/cascade_probe/probe.py`) — one L2
   `Worker`: `run(HANG)` (forever-spinning AIC task → op-timeout) then
   `run(NOOP)` (trivial AIC task) **on the same worker**, 10×. This mirrors
   gw0's persistent process reusing one pooled `ChipWorker`.

## Result

| Experiment | Cascades / failures |
| ---------- | ------------------- |
| Cascade variant (separate processes) | **0 / 10** |
| Control (PA-unroll alone) | **0 / 10** |
| Within-process probe (hang → noop, same worker) | **10 / 10** |

The cascade is **not** about the standalone→xdist ordering: with the device
held exclusively, `aicore_op_timeout` resets the device on its own
`worker.close()` / process exit, and the next process starts clean (0/10).

The cascade **is** a within-process, same-worker effect, and it is
deterministic (10/10). The exact a5 trace:

- STEP1 `run(HANG)`: `aclrtSynchronizeStreamWithTimeout (AICPU) failed:
  507000/507046` — STARS reaps the op, surfaced at AICPU stream sync.
- STEP2 `run(NOOP)` on the same worker: fails **early**, at
  `init_aicore_register_addresses → get_aicore_reg_info → halResMap failed
  for core 0 (rc=62)` (`device_runner.cpp:160`). a5 hits the poisoned
  context at register mapping; a2a3/CI hit it slightly later at `rtMalloc`
  (507899). Same root cause, different surfacing call.

**Which side leaks:** the **driver/device context**. The op-timeout leaves
the device sticky-errored for the rest of the process. It does *not* survive
a process exit + device reset (cascade variant 0/10). The amplifier is the
`_l2_worker_pool` reuse: nothing rebuilds the `Worker` after a failed run,
so the poison spreads to every later test in the worker's process — exactly
the CI cascade.

**Recovery is not possible in-process.** Two reset paths were measured:

- `aclrtSynchronizeDeviceWithTimeout` (the bounded device drain in fix #1)
  returns the *same* 507046 — it does not clear the sticky error.
- `Worker.close()` → `DeviceRunner::finalize` → `rtDeviceReset` followed by a
  fresh `Worker.init()` on the same device **in the same process** fails at
  `aclrtSetOpExecuteTimeOutV2 (507000)` / `rtStreamCreate (AICPU) 507899`
  (`simpler_init failed with code 507899`). The op-timeout poison survives a
  device reset within the process.

The only reset that recovers is **process exit** — which is exactly why the
separate-process cascade variant is 0/10. `aclrtResetDeviceForce` *would*
reset the physical device, but this box is shared (see
`.claude/rules/running-onboard.md`) and force-resetting kills other users'
work on the same die, so it is not an option. Net: after an AICore
op-timeout an a5 worker process cannot get a usable device back; only a new
process can.

## Fix

Two independent guards, both landed:

1. **Driver fail-fast (root-cause containment)** —
   `src/a5/platform/onboard/host/device_runner.{h,cpp}`. On any
   `launch_aicore_kernel` error **or** `sync_run_streams` error,
   `recover_device_or_mark_unusable()` best-effort drains via the bounded
   `aclrtSynchronizeDeviceWithTimeout`; if that also errors, it sets
   `device_unusable_`. `run()` checks the flag on entry and fails fast with
   an actionable message instead of cascading into the confusing
   `halResMap rc=62` / `rtMalloc 507899`. Verified: after the fix STEP2
   fails immediately at the run() guard (`code -1`, "Rebuild the Worker"),
   not at `halResMap`.

2. **Fixture rebuild-then-skip (CI containment)** — `conftest.py`. A
   `pytest_runtest_makereport` hook stashes the call-phase exception; the
   `st_worker` L2 path heals **only** on a device-runtime `RuntimeError`
   (`run_prepared failed …` / `prepare_callable failed …` / `DeviceRunner
   marked unusable` / `simpler_init failed …`), never on golden mismatches.
   The heal `close()`s the pooled `Worker` and drops it so the next test
   **rebuilds**. Because the a5 rebuild's `Worker.init()` then fails (device
   still poisoned — see above), the create path catches that device error,
   marks the runtime poisoned (`_l2_poisoned`), and `pytest.skip`s — that
   test and every later one for the runtime — with a clear reason. On an
   arch where in-process re-init *does* work, the rebuild succeeds and tests
   continue; on a5 they skip cleanly. Either way the worker-wide 507899
   failure storm is gone.

Verification (pooled L2 cascade through the real `st_worker` fixture: a hang
class poisons, a noop class follows on the same pool key; plus a normal L2
class for no-regression):

| Phase | TestBNoopAfter outcome |
| ----- | ---------------------- |
| before (fix #2 off, HEAD conftest) | FAILED 3/3 (cascade) |
| after fix #2 = rebuild-then-skip | SKIPPED 10/10 — 0/10 cascades |
| normal L2 (PA-unroll) regression check | PASSED 2/2 |

Each "after" run is "1 failed, 1 skipped": the one real failure
(TestAHangPoison) stands, every later test is a clean skip.

The skip carries its reason, e.g.: *"L2 Worker.init for runtime
'tensormap_and_ringbuffer' failed with a device-runtime error (simpler_init
failed with code 507899); the device context is not recoverable in-process
after an earlier AICore error — skipping remaining 'tensormap_and_ringbuffer'
L2 tests (a fresh worker process recovers)."*

## Why this shape

The cascade cannot be *recovered* on a5 — the device is dead in-process and
force-reset is unsafe on shared hardware — so the goal is **containment**,
not recovery:

- fix #1 (driver) turns the confusing `halResMap rc=62` / `rtMalloc 507899`
  cascade into an immediate, self-explanatory "DeviceRunner marked unusable;
  rebuild the Worker" — for *any* caller, not just pytest.
- fix #2 (fixture) turns the rest of an xdist worker's L2 tests from a storm
  of red 507899 errors into **one real failure + clean skips** with a
  reason. A fresh worker process (the next CI shard, or the standalone phase)
  starts clean.

Both are cheap, independent, and additive.

## When to reconsider

- If a future CANN release makes `aclrtSynchronizeDeviceWithTimeout` (or a
  scoped reset API) actually clear an op-timeout sticky error in-place,
  fix #1 could be upgraded from fail-fast to true in-place recovery, and
  the worker would not need rebuilding at all.
- a2a3 shows the same `finalize_common` cascade note (507018/507899/507901)
  but the trigger there has historically been blamed on runner contention;
  the a2a3 side was intentionally **not** touched (could not reproduce the
  cascade deterministically on a2a3 hardware here). If an a2a3 repro lands,
  port the same two guards to `src/a2a3/`.

## References

- CI run 27090242388, job `st-onboard-a5` (`gh run view --job 79952403308`).
- Fix: `src/a5/platform/onboard/host/device_runner.{h,cpp}`
  (`device_unusable_`, `recover_device_or_mark_unusable`), `conftest.py`
  (`pytest_runtest_makereport`, `_register_l2_pool_heal`).
- `DeviceRunnerBase::finalize_common` — prior note on why finalize skips a
  pre-destroy stream sync on the error-state stream.
- `.claude/rules/running-onboard.md` — `task-submit` device isolation.
