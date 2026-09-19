# How-to: debug a failed run

Work outward from the symptom. Most confusion here comes from treating a
host-side error code as if it named the cause — it usually does not.

## First, classify the symptom

| Symptom | Most likely cause | Go to |
| ------- | ----------------- | ----- |
| `507018`, `507014`, `507899`, `507046` at `run()` | a generic host-side code masking a device-side event | [Reading a 5xxxxx code](#reading-a-5xxxxx-code) below |
| Error at `init()`, not `run()` | wrong platform, missing runtime binaries, or a busy device | [Failures at init()](#failures-at-init) below |
| Golden mismatch, run itself succeeded | kernel logic, tile geometry, or arg order | [Wrong results](#wrong-results) below |
| Hangs forever in simulation | more simulated cores than the host can schedule | [sim oversubscription hang](../../troubleshooting/sim-oversubscription-hang.md) |
| Build or import failure on macOS | toolchain / OpenMP collision | [macos-build](../../troubleshooting/macos-build.md), [macos-libomp-collision](../../troubleshooting/macos-libomp-collision.md) |

## Reading a 5xxxxx code

A `507018` means "the stream did not finish in time", **not** "deadlock". Several
distinct device-side mechanisms surface as the same host code, so name the
mechanism before you theorize:

1. **Read the host log first.** It names the code and the device-side error class
   it masks — look for `error detail:`, `orch_error_code=`, `sched_error_code=`,
   `sub_class=`.
2. **Then look up the code** in
   [device-error-codes](../../troubleshooting/device-error-codes.md), the repo's
   own table, with per-mechanism notes in
   [`device-error-codes/`](../../troubleshooting/device-error-codes/README.md):
   [aicore-fault](../../troubleshooting/device-error-codes/aicore-fault.md),
   [capacity](../../troubleshooting/device-error-codes/capacity.md),
   [stall](../../troubleshooting/device-error-codes/stall.md).
3. **Only then open the device log**, which is the ground truth for what the
   AICPU was doing.

The decisive distinction: if no capacity or deadlock detector fired and only an
op-execute timeout did, the op was **slow or stalled, not deadlocked**. A real
capacity exhaustion trips its own detector; silence means look for a race or for
"just too slow" instead of enlarging a ring.

### Get your own device log

The device log defaults to a directory shared with every other user and process
on the machine, so finding your run's file means guessing by pid and timestamp.
Redirect it into your run's own output directory instead — the directory must
exist first, because the driver opens the file without creating the path:

```bash
LOGDIR="$PWD/outputs/my_run/ascend"
mkdir -p "$LOGDIR"
export ASCEND_PROCESS_LOG_PATH="$LOGDIR"
# run; the log is then at $LOGDIR/device-<id>/device-*.log
```

### If it is merely slow, not stuck

The timeouts are compile-time defaults — op-execute 45 s, stream-sync 50 s,
scheduler 20 s — and all three are environment-overridable, with the ordering
between them validated at startup. Increase them together when checking whether
a workload can complete with a larger budget. For a deliberately short onboard
failure window instead, use this valid ordering:

```bash
export SIMPLER_SCHEDULER_TIMEOUT_MS=2000
export SIMPLER_OP_EXECUTE_TIMEOUT_US=3000000
export SIMPLER_STREAM_SYNC_TIMEOUT_MS=4000
```

For scalar tensor waits (code 8), `tensormap_and_ringbuffer` has a separate
`SIMPLER_TENSOR_DATA_TIMEOUT_MS` budget: 30 s in sim, 15 s onboard by default.
Set it before `Worker.init()`. Raising it does not extend the outer watchdogs.

See [local-timeout-defaults](../../troubleshooting/local-timeout-defaults.md).
Note CI runs deliberately short timeouts, so a CI failure should be read against
the CI values rather than the header defaults.

## Failures at init()

`init()` is where runtime binaries are resolved and the device is opened, so it
is the first place setup problems appear:

- **Wrong `--platform` for this machine's silicon.** An `a2a3` invocation on a5
  hardware (or the reverse) produces failures that look like genuine runtime
  bugs. Confirm which platform this box can actually run before blaming the code.
- **Missing runtime binaries.** They are loaded from `build/lib/<platform>/<runtime>/`,
  not rebuilt on import. After changing runtime or platform C++, re-run
  `pip install --no-build-isolation -e .`; see
  [Developer Guide](../../developer-guide.md#when-to-rebuild).
- **A stale install after a struct-layout change.** If a whole suite fails with
  something like a zeroed `launch_aicpu_num`, the installed `_task_interface`
  predates a `CallConfig` layout change — reinstall rather than bisect.
- **Device busy.** A previous run that skipped `close()` still holds it.

## Wrong results

The run completed, so this is your kernel or your arg wiring:

- **Arg order is positional** and must match the callable `signature`. A swapped
  input pair produces plausible-looking wrong numbers, not an error.
- **`func_id` must match** the id the orchestration submits, and
  `function_name` the exported orchestration symbol — a mismatch can run the
  wrong kernel body.
- **Tile geometry.** Example kernels frequently hard-code a shape with no
  boundary handling; feeding a different shape silently reads out of range.
- Capture what a task actually received with `--dump-args` and inspect it —
  see [Args Dump](../../dfx/args-dump.md) and
  [How-to: profile a kernel](profile-a-kernel.md).

## Before concluding "simpler is broken"

If the failing workload lives in another repo (pypto, pypto-lib,
pypto-serving), first confirm which simpler is actually loaded and whether the
workload is covered by that repo's CI — a script that no workflow runs may be an
experimental draft that was never expected to pass. The repo map and the
import chain are in the [top-level README](../../../README.md#where-simpler-sits-in-the-stack).

If the mechanism you depend on may simply not be wired up on your arch, check
the [Capability Survey](../../capability-survey.md) — several communication
engines are declared but unreachable, and it records which.
