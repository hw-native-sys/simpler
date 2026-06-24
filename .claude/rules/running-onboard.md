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

## Quick reference

- **Run pytest on locked NPUs** —
  `task-submit --device N,M --run "python -m pytest ..."`
- **Auto-pick free NPUs** —
  `task-submit --device auto --device-num 2 --run "..."`
- **See queue + running** — `task-submit --list`
- **Wait for a submitted task** — `task-submit --wait <task-id>`
- **Cancel pending** — `task-submit --cancel <task-id>`
- **Per-die utilization + process table** — `npu-smi info`

`task-submit --help` for the full surface.
