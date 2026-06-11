# a5 AICore op-timeout poisons the shared L2 worker for the rest of an xdist session

**Date**: 2026-06-08 (force-reset recovery added 2026-06-09)
**Verdict**: recovered. #1005 *contained* the cascade (driver fail-fast +
fixture rebuild-then-skip). A follow-up found the poison **is** clearable
in-process with a *force* device reset (`aclrtResetDeviceForce`), so the
worker now recovers and the would-be-skipped tests actually run — see
"Force-reset recovery" below.

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

**Soft-reset recovery is not possible in-process** (a *force* reset is — see
"Force-reset recovery" below). Two soft-reset paths were measured:

- `aclrtSynchronizeDeviceWithTimeout` (the bounded device drain in fix #1)
  returns the *same* 507046 — it does not clear the sticky error.
- `Worker.close()` → `DeviceRunner::finalize` → `rtDeviceReset` followed by a
  fresh `Worker.init()` on the same device **in the same process** fails at
  `aclrtSetOpExecuteTimeOutV2 (507000)` / `rtStreamCreate (AICPU) 507899`
  (`simpler_init failed with code 507899`). The op-timeout poison survives a
  device reset within the process.

A *soft* reset does not recover, but a **force** reset does. Follow-up
measurement (2026-06-09): after poisoning a card, `aclrtResetDeviceForce`
(rc=0) lets a fresh `Worker.init()` on the **same card in the same process**
succeed — a trivial noop then runs clean. The earlier belief that "only a new
process recovers" was specific to the *soft* `rtDeviceReset`; the *force*
reset clears the op-timeout sticky-error in place. Two safety facts were also
verified:

- **Per-card scope.** With process A force-resetting device 2 while process B
  ran a noop loop on device 3, B's 14/14 iterations passed straight through
  A's reset — `aclrtResetDeviceForce(2)` does not disturb device 3. So under
  xdist (gw0→dev4, gw1→dev5) one worker force-resetting its card cannot break
  another's.
- **Exclusive ownership.** Force reset is destructive to anything else on the
  *physical* card, but onboard work always holds an exclusive `task-submit`
  lock (`.claude/rules/running-onboard.md`), so the only thing on the card is
  this job.

This is what the **force-reset recovery** fix below builds on.

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

## Force-reset recovery (2026-06-09)

The original containment left two gaps: the triggering test stays red, and the
"skipped" tests are never validated (real CI run 27182822054 went from 8 red to
`1 failed, 13 skipped` — contained, but the job is still red and 13 cases are
unverified). The follow-up measurement above showed `aclrtResetDeviceForce`
clears the poison in-process and is per-card safe under an exclusive lock, so
recovery is possible after all:

- `DeviceRunner::finalize()` calls `force_reset_device()`
  (`aclInit` + `aclrtSetDevice` + `aclrtResetDeviceForce`) whenever
  `device_unusable_` is set. The card is reset before the runner is torn down.
- The conftest heal then closes + drops the poisoned `Worker` (unchanged from
  #1005); the **next** test's `Worker.init()` lands on the now-clean card and
  **succeeds**, so the previously-skipped tests actually run. No CLI/env flag —
  it fires only on the rare device-poison path and onboard always holds an
  exclusive `task-submit` lock (see `.claude/rules/env-macro-gating.md` for why
  this is unconditional rather than gated).

Verified on a5: a pooled `hang → noop` pair, hang FAILED (trigger, not retried),
noop **PASSED** 3/3 (force-reset recovered the card and the victim ran). The
`fixture rebuild-then-skip` path remains as the fallback when a force reset ever
fails (and on a2a3, which has no force-reset).

This is the recovery layer; the triggering test itself is still red (retrying it
on a fresh card is a separate, later step).

## Why this shape

- fix #1 (driver) turns the confusing `halResMap rc=62` / `rtMalloc 507899`
  cascade into an immediate, self-explanatory fail-fast — for *any* caller,
  not just pytest.
- force-reset recovery clears the card in-process so the worker re-inits and
  the rest of the L2 tests **run** instead of cascading or being skipped.
- fixture rebuild-then-skip is the fallback when a force reset itself fails.

The same three guards are mirrored into `src/a2a3/` (the a2a3 device_runner
had none of them; #1005 was a5-only). a2a3 has the identical
`device_unusable_` / `recover_device_or_mark_unusable` / `force_reset_device`
chain wired into its `run()` / `finalize()`. The mechanism is CANN-level
(`aclrtResetDeviceForce`), so it is expected to behave as on a5, but a2a3
**was not hardware-verified locally** — this dev box is Ascend950PR (a5) only;
the `st-onboard-a2a3` CI job is the a2a3 verification channel.

## When to reconsider

- The triggering test is still reported failed (its `207001` is likely
  transient). Adding a bounded retry (≤3 attempts) of just that test on a
  freshly-reset card would turn the last red into green — a follow-up step.
- a2a3 now carries the mirrored guards (see "Why this shape") but they are
  CI-verified only — no a2a3 silicon on this box. If the `st-onboard-a2a3`
  job ever shows the force reset failing or not recovering on real a2a3
  hardware, revisit `force_reset_device()` there (the CANN force-reset
  semantics may differ from a5).

## References

- CI run 27090242388, job `st-onboard-a5` (`gh run view --job 79952403308`) —
  original 8-failure cascade. CI run 27182822054 — post-#1005 contained run
  (`1 failed, 13 skipped`).
- Containment (#1005): `src/a5/platform/onboard/host/device_runner.{h,cpp}`
  (`device_unusable_`, `recover_device_or_mark_unusable`), `conftest.py`
  (`pytest_runtest_makereport`, `_register_l2_pool_heal`).
- Force-reset recovery: `src/a5/platform/onboard/host/device_runner.{h,cpp}`
  (`force_reset_device`, called from `finalize()` on the `device_unusable_`
  path).
- a2a3 mirror (CI-verified only): `src/a2a3/platform/onboard/host/`
  `device_runner.{h,cpp}` — identical `device_unusable_` /
  `recover_device_or_mark_unusable` / `force_reset_device` chain.
- `DeviceRunnerBase::finalize_common` — prior note on why finalize skips a
  pre-destroy stream sync on the error-state stream.
- `.claude/rules/running-onboard.md` — `task-submit` device isolation.
- `.claude/rules/env-macro-gating.md` — why force reset is unconditional
  (no opt-in flag) rather than gated.
