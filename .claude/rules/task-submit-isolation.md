# NPU Hardware Isolation via `task-submit`

## Why

This dev box is **shared** across many users running NPU work concurrently
(`pypto-serving`, model decode benchmarks, PTOAS validation, other `simpler`
hacking, etc.). Without explicit per-device locking, two users hitting the
same physical chip will contend on chip-shared resources (FFTS control,
shared L2, MMU TLB, register MMIO); the symptom on AICore is a dispatched
task that ACKs but never writes FIN — scheduler stalls 800K idle iterations,
CANN reports `errorCode=0x2a` / `507018` "aicpu execute failed" with
`errcode:0`. **It looks exactly like a code bug.** It is not — it is cross-user
contention.

Concrete example: an investigation chased a flaky `st-onboard-a2a3`
`simpler_aicpu_exec` 0x2a for hours assuming a software root cause, and
"reproduced" 100% in iter 1 on devices 8/9 — only to find another user had
been holding `npu-lock 8` for 5 hours with a heavy serving workload. With
`task-submit` exclusive lock on a free pair, all 50 iterations passed clean.
The "reproduction" was contention noise.

CI runs on dedicated runners and **always uses `task-submit`** (see
`.github/workflows/ci.yml`), so CI never sees this class of noise. Local
hardware work that bypasses `task-submit` is comparing against a stricter
environment.

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

If unlocked results contradict CI or prior runs, **first** check whether
another process is on the same chip (`npu-smi info` + the process table at
the bottom) before treating the discrepancy as a real signal.

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
  with no lock — your iter results are entangled with whoever else is on
  those devices.
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
- **Per-die contention map** — `npu-smi info` + bottom process table

`task-submit --help` for the full surface.
