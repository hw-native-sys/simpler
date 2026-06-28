# NPU Hardware Isolation via `task-submit`

## Why

This dev box is **shared** across many users running NPU work concurrently
(`pypto-serving`, model decode benchmarks, PTOAS validation, other `simpler`
hacking, etc.). `task-submit` is the queue / per-device lock used on this
box and on CI runners — running hardware work through it keeps two users
from grabbing the same device, keeps the `--list` queue accurate, and keeps
local runs comparable to CI (CI always wraps pytest in `task-submit`, see
`.github/workflows/ci.yml`).

## Autonomous invocation — detect capability, then run without asking

When a task needs the NPU, do **not** ask the user for permission to run it.
Detect-then-run:

1. **Detect silicon capability** for the arch the work targets (`a2a3` /
   `a5`) via the `onboard-arch-precheck` skill:

   ```bash
   .claude/skills/onboard-arch-precheck/check.sh a2a3   # exit 0 ⇒ this box can run a2a3 onboard
   ```

   Sim variants (`a2a3sim` / `a5sim`) are silicon-agnostic — always runnable.

2. **Check `task-submit`** is on `PATH` (`command -v task-submit`).

3. **If both hold, invoke directly** through `task-submit` (per the Rule
   below) — no "shall I run it?" prompt. Run it and report the results.

Only fall back to asking the user when the precheck fails (the box's silicon
does not match the requested arch) or `task-submit` is absent — i.e. when the
box genuinely cannot run the work as requested.

## Rule

**Any hardware (onboard) work on this box — stress repro, perf benchmarks,
flaky-test investigation, ChipWorker scripts that take NPU exclusively —
must be wrapped in `task-submit --device <list> --run "..."` when the
command is available on `PATH`.**

```bash
# Check once at the top of any onboard investigation:
if command -v task-submit >/dev/null 2>&1; then
    HAS_TASK_SUBMIT=1
else
    HAS_TASK_SUBMIT=0
fi
```

### When `task-submit` IS available (this dev box, CI runners)

Use it for **every** invocation that touches an NPU. Bare `pytest --device 8`
in a shell loop is forbidden.

```bash
# Single-shot
task-submit --timeout 1800 --max-time 1800 --device 8,9 \
    --run "python -m pytest tests/st/... --platform a2a3 --device 8-9 ..."

# Long-running stress harness — let task-submit own the whole loop so the
# lock is held for the whole duration, not re-acquired per iter:
task-submit --timeout 7200 --max-time 7200 --device 10,11 \
    --run "/tmp/my_stress.sh 50 10 11"
```

`--device auto` picks free devices automatically — preferable for one-off
runs that don't care which die:

```bash
task-submit --timeout 1800 --max-time 1800 --device auto --device-num 2 \
    --run "python -m pytest ..."
```

Before submitting, sanity-check what's currently held to know whether the
task will queue or run immediately:

```bash
task-submit --list                          # see Pending / Running / Done
# Replace 4 with the chip ID you intend to use (npu-smi groups by chip).
npu-smi info | grep -A1 -B1 '^| 4 '         # raw per-chip view
```

### When `task-submit` is NOT available (laptop, fresh dev container, …)

Fall back to plain commands, but **document in the output** that the run is
unisolated:

```bash
echo "[WARN] task-submit not found; running unlocked — results may be noisy if any other process is on this NPU"
python -m pytest tests/st/... --platform a2a3 --device 8-9 ...
```

If unlocked results contradict CI or prior runs, **first** rule out the
usual environment causes (binary out of sync with source, wrong arch,
stale CMake cache) before treating the discrepancy as a real signal.

## Pre-flight: arch precheck

Before `task-submit … --run "… pytest … --platform a2a3|a5 …"`, gate the
invocation through
[`onboard-arch-precheck`](../skills/onboard-arch-precheck/SKILL.md). The
precheck takes ~600 ms cold (cached afterwards at ~5 ms) and refuses a
wrong-arch invocation BEFORE any device lock is acquired. Running the
wrong arch produces 507018 / 507899 cascades that look like genuine bugs
and routinely waste hours of debugging — see the skill for the failure
signatures.

```bash
.claude/skills/onboard-arch-precheck/check.sh a2a3 || exit 1
task-submit --device auto --device-num 1 \
    --run "python -m pytest ... --platform a2a3 --device \$TASK_DEVICE"
```

Sim variants (`a2a3sim`, `a5sim`) pass the precheck unconditionally — they
are silicon-agnostic. The precheck is purely about onboard invocations.

## Device logs: redirect them out of the shared default

The AICPU/CCECPU device log (where `report_deadlock`, `HandleTaskTimeout`,
stall diagnostics, AICore faults, and the `PTO2_PROFILING` Total/Orch/Sched
markers land — the ground truth behind a host-side `507018`) defaults to:

```text
~/ascend/log/debug/device-<id>/device-<pid>_<timestamp>.log
```

That directory is **shared across every user/process on the box**, so finding
*your* run's log means guessing by pid/timestamp and racing other writers.
`ASCEND_PROCESS_LOG_PATH` redirects it to a directory you control.

**`export` and `--env` are equivalent** — `task-submit` inherits the caller's
environment, so you do *not* need `--env`:

```bash
# these two are the same; the dir MUST exist first (the driver fopens, no mkdir)
export ASCEND_PROCESS_LOG_PATH="$LOGDIR"   # then: task-submit --run "..."
task-submit --env ASCEND_PROCESS_LOG_PATH="$LOGDIR" --run "..."
```

**Recommended: put the device log under the run's own output dir** so it is
isolated from other users and co-located with that run's artifacts. For a
simpler scene test the per-case root is `outputs/<case>_<ts>/`; for a JIT
example it is `build_output/_jit_*/dfx_outputs/`. Make an `ascend/` subdir
under it and point the var there:

```bash
LOGDIR="$PWD/outputs/<case>/ascend"   # or .../dfx_outputs/ascend
mkdir -p "$LOGDIR"
export ASCEND_PROCESS_LOG_PATH="$LOGDIR"
task-submit --device auto --device-num 1 --run "python ... -d \$TASK_DEVICE"
# this run's device log is now at $LOGDIR/device-<id>/device-*.log
```

Then read it directly (`$LOGDIR/device-*/device-*.log`) — no more `grep`-ing
`~/ascend/log/debug/` and guessing which file is yours.

## `507018` triage: classify the mechanism before concluding

`507018` (and `507014` / `507899`) is a **generic host-side code** —
`run_prepared failed` / `aclrtSynchronizeStreamWithTimeout failed`. Several
*distinct* on-device mechanisms all surface as the same code. Do NOT call it a
"deadlock" or "OOM" from the host error alone — **read the device log and grep
for the signature that actually fired:**

| device-log signature | mechanism | note |
| -------------------- | --------- | ---- |
| `FATAL: Task Allocator Deadlock` / `Provable head-of-line` | ring/heap or dep-pool **deadlock** (alloc can't reclaim) | AICPU detector: 500ms backstop (`PTO2_ALLOC_DEADLOCK_TIMEOUT_CYCLES`) or immediate structural `head_blocked_on_scope_end`. Real capacity/scope deadlock. |
| `Timeout (N cycles): producer/consumers ...` | **SPIN** wait on a specific producer/consumer | `pto_runtime2.cpp`. |
| `HandleTaskTimeout` / `kill aicpu-sd` | **OS op-execute timeout** | STARS/tsdaemon, default 3s (`PLATFORM_OP_EXECUTE_TIMEOUT_US`). **A 3s kill ≠ deadlock** — the op was merely long or stalled. Raise this constant to measure true on-device duration. |
| `log_stall_diagnostics` (cores idle + tasks `state=WAIT fanin 0/N` + `completed` frozen) | **forward-progress stall** | No dedicated detector — intermittent races (often contention-triggered) land here and are reaped only by the op-timeout above. |

Decisive rule: **if a capacity/deadlock detector did NOT fire (counts of
`Task Allocator Deadlock` / `Timeout (cycles)` are 0) and only
`HandleTaskTimeout` did, it is NOT a proven deadlock or capacity bug** — it is a
long/stalled op. A capacity exhaustion would trip its own detector; silence ⇒
look for a race or just-too-slow, not "the ring is too small". For the
host/device timing breakdown of a (completed) run, parse its `[STRACE]` markers
with `python -m simpler_setup.tools.strace_timing <log> --rounds-table` (see
`simpler_setup/tools/README.md`); for the per-thread `loops`/`tasks_scheduled`
deep-dive, rebuild with `PTO2_SCHED_PROFILING=1` and read the device log
directly.

## Anti-patterns

- ❌ Bash `for i in $(seq 1 50); do pytest ... --device 8 & pytest ... --device 9 & wait; done`
  with no lock — you'll race other users for the same device and break
  the shared queue's accounting.
- ❌ Reading `gh pr checks <PR>` "ci passed" as proof a fix worked while
  your own local repro (unlocked) shows the bug — your local environment is
  the outlier, not CI.
- ❌ Claiming "X% reproduction rate" from unlocked runs without listing
  `task-submit --list` at the time of the run.
- ❌ Bypassing `onboard-arch-precheck` — the `--platform` mismatch failure
  modes are silent (look like real bugs) and burn hours of investigation
  time. Always run the gate.
- ❌ Fishing your run's device log out of the shared `~/ascend/log/debug/`
  by pid/timestamp guesswork. Set `ASCEND_PROCESS_LOG_PATH` to a per-run
  dir up front (see "Device logs" above) so the log is isolated and known.

## Quick reference

- **Run pytest on locked NPUs** —
  `task-submit --device N,M --run "python -m pytest ..."`
- **Auto-pick free NPUs** —
  `task-submit --device auto --device-num 2 --run "..."`
- **See queue + running** — `task-submit --list`
- **Wait for a submitted task** — `task-submit --wait <task-id>`
- **Cancel pending** — `task-submit --cancel <task-id>`
- **Per-die utilization + process table** — `npu-smi info`
- **Redirect device log to the run's output** —
  `mkdir -p <outdir>/ascend && export ASCEND_PROCESS_LOG_PATH=<outdir>/ascend`
  (== `task-submit --env ASCEND_PROCESS_LOG_PATH=...`; dir must pre-exist)

`task-submit --help` for the full surface.
