---
name: benchmark
description: Benchmark runtime performance on hardware. If the current branch has commits ahead of upstream/main or uncommitted changes, compares against the fork point (merge-base). Otherwise benchmarks current state only. Use when the user asks to benchmark, measure performance, or compare latency.
---

# Benchmark Workflow

Benchmark runtime performance on Ascend hardware. Automatically detects whether to run a single benchmark or a comparison.

## Modes

| Condition | Mode | What happens |
| --------- | ---- | ------------ |
| 0 commits ahead AND no uncommitted changes | **Single** | Benchmark current state and report its runtime-specific timing columns |
| >= 1 commits ahead OR uncommitted changes | **Compare** | Benchmark merge-base (worktree) AND current workspace, show comparison table |

## Input

Optional benchmark arguments forwarded to `tools/benchmark_rounds.sh`:

```text
/benchmark
/benchmark -d 4 -n 50
/benchmark -d 4 -d 6
/benchmark --serial-orch-sched
```

Extra arguments (`-n`, `-r`, `--serial-orch-sched`, etc.) are forwarded to
`tools/benchmark_rounds.sh`.

### Device arguments (`-d`)

The `-d` flag specifies NPU device IDs.

**Hard rule: one benchmark process per device at any time.** Never run two benchmark processes on the same `-d` device concurrently — not two runtimes, not baseline + current, nothing. This prevents resource contention and ensures stable measurements.

On a shared hardware host, submit every onboard case through the host's
`task-submit` tool when available. The simpler benchmark harness does not
depend on task-submit; never wrap its whole multi-case run in one allocation.

| `-d` count | Compare mode behavior |
| ---------- | --------------------- |
| One device (`-d 4`) | **Sequential**: every case is its own task; baseline completes before current starts on the same device. |
| Two devices (`-d 4 -d 6`) | Not supported by this harness; benchmark cases request one device each. Compare sequentially on one selected device. |
| Zero (not specified) | Let each case task use `task-submit --device auto` (see Step 2) |

**Defaults** (when not specified): use `benchmark_rounds.sh` defaults (device 0, 100 rounds, a2a3, tensormap_and_ringbuffer).

## Runtime Selection

`tools/benchmark_rounds.sh` supports `-r <runtime>`:

- `tensormap_and_ringbuffer` (default)
- `host_build_graph`

Each architecture/runtime quadrant has its own list at the top of the script.
TMR reports Host / Device / Effective / Orch / Sched. HBG reports Host / Device
because its orchestration runs on the host and has no device-side Orch/Sched
windows. `--serial-orch-sched` is TMR-only and must be rejected for HBG.

## Step 1: Detect Mode

```bash
git fetch upstream main --quiet
COMMITS_AHEAD=$(git rev-list HEAD --not upstream/main --count 2>/dev/null || echo "0")
HAS_CHANGES=$(git status --porcelain)

if [ "$COMMITS_AHEAD" -eq 0 ] && [ -z "$HAS_CHANGES" ]; then
  MODE="single"
else
  MODE="compare"
  MERGE_BASE=$(git merge-base upstream/main HEAD)
fi
```

## Step 2: Device Isolation

If the host provides `task-submit`, check the queue and run the architecture
precheck before submitting onboard cases. Submit the scene-test command for
one selected case at a time from the host shell. Do not invoke the benchmark
harness itself inside task-submit; it launches multiple cases and parses their
results.

```bash
task-submit --list
.claude/skills/onboard-arch-precheck/check.sh "$PLATFORM"
```

## Step 3: Confirm PTO-ISA Pin

PTO-ISA is selected by the repo-root `pto_isa.pin`. Record the pin in the
benchmark notes so baseline and current runs can be compared against the same
source revision:

```bash
PTO_ISA_PIN=$(tr -d '[:space:]' < pto_isa.pin)
```

## Step 4: Prepare — Compute Absolute Paths

The Bash tool resets its working directory to the project root on every call. Relative paths like `cd worktree && ...` are fragile and easy to forget. **Compute absolute paths once, then use them everywhere.**

```bash
PROJECT_ROOT="$(pwd)"                    # e.g. /home/user/simpler
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
WORKTREE_ABS="${PROJECT_ROOT}/tmp/worktree_baseline_${TIMESTAMP}"
mkdir -p "${PROJECT_ROOT}/tmp"
```

Store `PROJECT_ROOT` and `WORKTREE_ABS` as shell variables in every Bash call that needs them (the Bash tool does not persist variables across calls). Use this pattern:

```bash
# Correct — self-contained, uses absolute path
WORKTREE_ABS="/home/user/simpler/tmp/worktree_baseline_20260331_102302"
"${WORKTREE_ABS}/tools/benchmark_rounds.sh" -d 2 ...
```

**Do NOT use `cd` + relative `./tools/...`** — this is the #1 source of silent errors (running the wrong workspace).

## Step 5: Run Benchmarks

Compilation artifacts are content-addressed and stored in `build/cache/kernels`.
Prepare the selected callables on CPU first; this does not create a Worker or
touch an NPU. Then submit each scene-test case command separately through the
host's task-submit tool. `--rounds` repetitions for that one case remain in its
task. Parse and aggregate timing after the task exits.

### Single Mode

```bash
./tools/benchmark_rounds.sh -p "$PLATFORM" -r "$RUNTIME" --prepare-only

CASE_FILE="$PROJECT_ROOT/tests/st/a2a3/tensormap_and_ringbuffer/alternating_matmul_add/test_alternating_matmul_add.py"
CASE_SELECTOR="TestAlternatingMatmulAdd::Case1"
task-submit --timeout 7200 --max-time 7200 --device auto \
  --run "python3 '$CASE_FILE' --platform '$PLATFORM' --case '$CASE_SELECTOR' --manual include --rounds 100 --skip-golden --single-case" \
  > "tmp/benchmark_${TIMESTAMP}.txt" 2>&1
python3 -m simpler_setup.tools.strace_timing "tmp/benchmark_${TIMESTAMP}.txt" --rounds-table
```

Repeat the task-submit command once per entry in the selected workload list.
Use `--device N` instead of `--device auto` to pin a device. Submit the serial
orch/sched measurement as another task with its mode environment set.

### Compare Mode

Prepare the worktree, virtual environment, runtime binaries, and compiled
callable cache **before** running onboard tasks. Use one device and run baseline
then current serially so both measurements use the same card. For each selected
case, submit one standalone scene-test invocation and process its timing log
after task-submit returns:

```bash
(
  cd "$WORKTREE_ABS"
  source .venv/bin/activate
  .claude/skills/onboard-arch-precheck/check.sh "$PLATFORM"
  ./tools/benchmark_rounds.sh -p "$PLATFORM" -r "$RUNTIME" --prepare-only
  task-submit --timeout 7200 --max-time 7200 --device "$DEVICE" \
    --run "python3 '$WORKTREE_ABS/tests/st/a2a3/tensormap_and_ringbuffer/alternating_matmul_add/test_alternating_matmul_add.py' --platform '$PLATFORM' --device '$DEVICE' --case TestAlternatingMatmulAdd::Case1 --manual include --rounds 100 --skip-golden --single-case" \
    > "${PROJECT_ROOT}/tmp/benchmark_baseline_${TIMESTAMP}_${RUNTIME}.txt" 2>&1
  python3 -m simpler_setup.tools.strace_timing "${PROJECT_ROOT}/tmp/benchmark_baseline_${TIMESTAMP}_${RUNTIME}.txt" --rounds-table
)
cd "$PROJECT_ROOT"
.claude/skills/onboard-arch-precheck/check.sh "$PLATFORM"
./tools/benchmark_rounds.sh -p "$PLATFORM" -r "$RUNTIME" --prepare-only
task-submit --timeout 7200 --max-time 7200 --device "$DEVICE" \
  --run "python3 '$PROJECT_ROOT/tests/st/a2a3/tensormap_and_ringbuffer/alternating_matmul_add/test_alternating_matmul_add.py' --platform '$PLATFORM' --device '$DEVICE' --case TestAlternatingMatmulAdd::Case1 --manual include --rounds 100 --skip-golden --single-case" \
  > "tmp/benchmark_current_${TIMESTAMP}_${RUNTIME}.txt" 2>&1
python3 -m simpler_setup.tools.strace_timing "tmp/benchmark_current_${TIMESTAMP}_${RUNTIME}.txt" --rounds-table
```

The per-case `task-submit` tasks use the explicit `-d` device above. Do not
wrap either harness invocation in another task-submit command.

## Step 6: Report Results

Parse every `Avg <Metric>:` field present in the runtime's output. Missing
runtime-inapplicable columns are not zero measurements and must not be printed.

| Metric | Source | What it captures |
| ------ | ------ | ---------------- |
| Host | `[STRACE]` `chip.run` span | steady_clock around dispatch (Python overhead included); rendered from markers by `strace_timing --rounds-table` |
| Device | `[STRACE]` `chip.run.runner_run.device_wall` span | full on-NPU AICPU run wall (`AicpuPhase::RunWall`, `max(end) − min(start)` across threads); on TMR this whole run + teardown is strictly larger than the windows below |
| Effective | TMR orch/sched markers' device-domain `ts`+`dur` | TMR only: `max(orch_end,sched_end) − min(orch_start,sched_start)` — the orch∪sched merged window |
| Orch | `[STRACE]` `…device_wall.orch` span (`--rounds-table`) | TMR only: device orchestrator (graph-build) window |
| Sched | `[STRACE]` `…device_wall.sched` span (`--rounds-table`) | TMR only: scheduler dispatch/execution window |

The scene test only *emits* `[STRACE]` markers to stderr; `benchmark_rounds.sh`
tees the run and renders the Host/Device/Effective/Orch/Sched table with
`python -m simpler_setup.tools.strace_timing <log> --rounds-table`. All columns
come from the markers (onboard and sim) — no CANN device log is read.

Use Effective as the TMR headline metric. HBG has no equivalent phase window:
report Host and Device independently and do not synthesize an overall score
from one of them.

For a per-stage breakdown of `Host`/`Device` (host `bind`/`runner_run`/`validate`
plus TMR's AICPU `preamble`/`so_load`/`graph_build`
(`config_validate`/`arena_wire`/`sm_reset` prep sub-phases)/`post_orch`
subdivision), parse the `[STRACE]` markers with
`simpler_setup/tools/strace_timing.py` (add `--tree` for the nested view) — see
[docs/dfx/host-trace.md](../../../docs/dfx/host-trace.md). Same `SIMPLER_HOST_STRACE`
gate, no extra flag (set `SIMPLER_DEVICE_STRACE_ENABLE=0` to drop only the device
`clk=dev` markers).

### Single Mode

Use the runtime's actual column set; the five-column example below is TMR. An
HBG table contains only Host and Device.

```text
Benchmark at: <short SHA>
Args: -d 4 -n 100

Example                          Host (us)   Device (us)   Effective (us)    Orch (us)   Sched (us)
-------------------------------  ---------   -----------   --------------   ----------   ----------
alternating_matmul_add           480000.0        9050.0         1235.5          820.3       1235.4
benchmark_bgemm                  370000.0        7100.0          892.1          650.2        892.0
...
```

### Compare Mode

Show a comparison table per metric (one row per metric per example), **grouped
by runtime**. For TMR, `Effective` is the headline metric used in the overall
summary and the other four are context. HBG has no combined headline metric:
show Host and Device as independent rows and summarize regressions separately.

```text
Merge-base: <short SHA>  →  HEAD: <short SHA> (+ uncommitted)
Args: -d 4 -n 100
Device: baseline=4, current=4  (or baseline=4, current=6)

### tensormap_and_ringbuffer

Example                      Base (us)   HEAD (us)   Delta (us)   Change (%)
---------------------------  ---------   ---------   ----------   ----------
alternating_matmul_add         1240.1      1235.5        -4.6       -0.37%
  (host)                     480000.0    470000.0    -10000.0       -2.08%
  (device)                     9000.0      8800.0       -200.0       -2.22%
  (orch)                        830.0       820.3        -9.7       -1.17%
  (sched)                      1240.0      1235.4        -4.6       -0.37%
benchmark_bgemm                 890.3       892.1        +1.8       +0.20%
  (host)                     370000.0    370500.0      +500.0       +0.14%
  (device)                     7100.0      7080.0       -20.0       -0.28%
  (orch)                        650.0       650.2        +0.2       +0.03%
  (sched)                       890.2       892.0        +1.8       +0.20%
...

Overall: X of Y examples improved, Z regressed   (based on Effective)
```

If baseline and current ran on **different devices**, add a note:

> Note: Baseline and current ran on different NPU devices (4 vs 6). Results within ±2% may reflect device-to-device variance rather than code changes. For definitive comparison, re-run on the same device with `/benchmark -d <single_device>`.

**Interpretation:**

| Change (%) | Assessment |
| ---------- | ---------- |
| < -2% | Notable improvement |
| -2% to +2% | Within noise margin |
| > +2% | Potential regression — flag for review |

If any example shows > 5% regression, highlight it explicitly.

## Error Handling

| Error | Action |
| ----- | ------ |
| `task-submit` cannot allocate the requested device count | Report the queue/allocation error; do not run outside the lock |
| Benchmark script fails | Report which examples failed; continue with remaining |
| No timing data | Warn: "No timing markers — ensure `SIMPLER_HOST_STRACE` is enabled" |
| All examples fail | Check: did you run `pip install -e .` in the worktree venv? |
| Worktree creation fails | Fall back to stash/checkout approach or report error |
| `Pre-built runtime binaries not found` | The venv `pip install -e .` should have built these; re-run it |
| `ModuleNotFoundError: _task_interface` | Venv not activated; add `source .venv/bin/activate &&` before the command |

## Checklist

- [ ] Mode detected (single vs compare)
- [ ] Runtime binaries and selected callable artifacts prepared before device allocation
- [ ] Architecture precheck passed; each case ran in its own `task-submit` task
- [ ] Benchmark harness ran in preparation-only mode, outside device allocation
- [ ] PTO-ISA pinned to CI commit
- [ ] `PROJECT_ROOT` and `WORKTREE_ABS` absolute paths computed
- [ ] (Compare mode) Worktree created, venv built with `pip install -e .`
- [ ] (Compare mode) Baseline completed — venv activated, `pwd` confirmed worktree path before running
- [ ] Current completed in main workspace
- [ ] Worktree cleaned up (compare mode)
- [ ] Results table uses TMR's five columns or HBG's Host / Device columns
- [ ] (Compare mode) Device difference noted if applicable
- [ ] (Compare mode) Regressions > 2% flagged
