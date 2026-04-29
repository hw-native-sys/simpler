---
name: benchmark
description: Benchmark runtime performance on hardware. If the current branch has commits ahead of upstream/main or uncommitted changes, compares against the fork point (merge-base). Otherwise benchmarks current state only. Use when the user asks to benchmark, measure performance, or compare latency.
---

# Benchmark Workflow

Benchmark runtime performance on Ascend hardware. Automatically detects whether to run a single benchmark or a comparison.

## Modes

| Condition | Mode | What happens |
| --------- | ---- | ------------ |
| 0 commits ahead AND no uncommitted changes | **Single** | Benchmark current state, report Elapsed + Orch times |
| >= 1 commits ahead OR uncommitted changes | **Compare** | Benchmark merge-base (worktree) AND current workspace, show comparison table |

## Input

Optional benchmark arguments forwarded to `tools/benchmark_rounds.sh`:

```text
/benchmark
/benchmark -d 4 -n 50
/benchmark -d 4 -d 6
```

Extra arguments (`-n`, `-r`, etc.) are forwarded to `tools/benchmark_rounds.sh`.

### Device arguments (`-d`)

The `-d` flag specifies NPU device IDs.

**Hard rule: one benchmark process per device at any time.** Never run two benchmark processes on the same `-d` device concurrently — not two runtimes, not baseline + current, nothing. This prevents resource contention and ensures stable measurements.

| `-d` count | Compare mode behavior |
| ---------- | --------------------- |
| One device (`-d 4`) | **Sequential**: baseline first, then current, both on the same device. Multiple runtimes also run serially on that device. |
| Two devices (`-d 4 -d 6`) | **Parallel per-runtime**: for each runtime, baseline on first device and current on second device can run in parallel (different devices). Multiple runtimes still run serially — finish one runtime on both devices before starting the next. |
| Zero (not specified) | Auto-detect idle devices (see Step 2) |

**Defaults** (when not specified): use `benchmark_rounds.sh` defaults (device 0, 100 rounds, a2a3, tensormap_and_ringbuffer).

## Runtime Selection

`tools/benchmark_rounds.sh` supports `-r <runtime>`:

- `tensormap_and_ringbuffer` (default)

The example list is defined at the top of the script (`TMR_EXAMPLE_CASES`).

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

## Step 2: Device Detection

If `-d` was provided in args, skip detection and use the user-specified device(s).

Otherwise, detect idle NPU devices (HBM-Usage = 0):

```bash
npu-smi info
```

Pick devices with **HBM-Usage = 0**. Find the longest consecutive sub-range (at most 4). If no idle device is found, prompt user to specify a device ID.

## Step 3: Pin PTO-ISA

Extract pinned commit from `.github/workflows/ci.yml`:

```bash
PTO_ISA_COMMIT=$(grep -oP '(?<=--pto-isa-commit )\w+' .github/workflows/ci.yml | head -1)
```

Append `-c $PTO_ISA_COMMIT` to benchmark args so the underlying `python test_*.py` invocation picks it up.

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

### Single Mode

```bash
./tools/benchmark_rounds.sh $BENCH_ARGS -r "$RUNTIME" 2>&1 | tee "tmp/benchmark_${TIMESTAMP}.txt"
```

### Compare Mode

Use a **git worktree** for the baseline so the current workspace is never disturbed.

#### CRITICAL: Worktree needs its own build environment

The worktree is a fresh checkout at merge-base — it has **no pre-built runtime binaries** and no compiled nanobind extension. Two things must be built:

1. **Runtime `.so` binaries** (`build/lib/`) — loaded via ctypes by `bindings.py`
2. **Nanobind `_task_interface` extension** — compiled C++ Python bindings

Pure Python files under `simpler_setup/` (e.g. `scene_test.py`, `kernel_compiler.py`) are resolved via `sys.path` from the worktree when an editable install is active there, so they correctly come from the worktree. But `_task_interface.*.so` is installed into site-packages by `pip install -e .` and is **shared system-wide**. Without isolation, the worktree would use the main workspace's nanobind extension — which may have incompatible API changes.

**Solution: always create a venv in the worktree** (~26s overhead). This builds both the nanobind extension AND runtime binaries, fully isolating the baseline.

#### 5a. Create worktree, venv, and build

Inline the **absolute** worktree path (copy-paste the value, do not rely on shell variables persisting):

```bash
# Create worktree
git worktree add "$WORKTREE_ABS" "$MERGE_BASE" --quiet

# Create venv with system site-packages (for torch, numpy, etc.)
python3 -m venv "${WORKTREE_ABS}/.venv" --system-site-packages

# Install into venv — builds nanobind extension + runtime binaries
"${WORKTREE_ABS}/.venv/bin/pip" install -e "${WORKTREE_ABS}" -q 2>&1 | tail -3
```

This gives the worktree its own `_task_interface.*.so` in `.venv/lib/python3.*/site-packages/`, completely independent from the main workspace.

#### 5b. Run baseline

Activate the venv so `benchmark_rounds.sh` (which calls `python3`) picks up the worktree's nanobind extension and Python bindings:

```bash
# WORKTREE_ABS must be the literal absolute path (e.g. /home/user/simpler/tmp/worktree_baseline_20260331)
cd "$WORKTREE_ABS" && source .venv/bin/activate && pwd && ./tools/benchmark_rounds.sh -d $BASELINE_DEVICE -c $PTO_ISA_COMMIT -r "$RUNTIME" \
  2>&1 | tee "${PROJECT_ROOT}/tmp/benchmark_baseline_${TIMESTAMP}_${RUNTIME}.txt"
```

**Always include `pwd &&` after `cd` to verify you are in the correct directory.** If `pwd` does not print the worktree path, something went wrong — do not proceed.

#### 5c. Run current

```bash
# Runs from the main workspace (Bash tool default cwd)
./tools/benchmark_rounds.sh -d $CURRENT_DEVICE -c $PTO_ISA_COMMIT -r "$RUNTIME" \
  2>&1 | tee "tmp/benchmark_current_${TIMESTAMP}_${RUNTIME}.txt"
```

#### 5d. Cleanup

```bash
git worktree remove "$WORKTREE_ABS" --force
```

If `git worktree remove` fails (e.g., cwd was inside the deleted worktree), use:

```bash
git -C "$PROJECT_ROOT" worktree remove "$WORKTREE_ABS" --force
```

#### Parallel execution (two devices)

When two devices are available, run baseline and current **for the same runtime** in parallel on separate devices. The venv ensures the worktree has its own nanobind extension, so both workspaces are fully independent.

```bash
# For each runtime (serially):
for RUNTIME in "${RUNTIMES_TO_BENCH[@]}"; do
  # Baseline on device A (from worktree with venv), current on device B (from main) — parallel
  (cd "$WORKTREE_ABS" && source .venv/bin/activate && pwd && ./tools/benchmark_rounds.sh -d $DEVICE_BASELINE -r "$RUNTIME" ...) &
  ./tools/benchmark_rounds.sh -d $DEVICE_CURRENT -r "$RUNTIME" ... &
  wait  # Both finish before starting next runtime
done
```

**Never launch the next runtime until the current one finishes on all devices.**

#### Sequential execution (one device)

```bash
# 1. Worktree + venv already created in step 5a

# 2. For each runtime (serially — one device, one process at a time):
#    Baseline first (from worktree with venv activated)
cd "$WORKTREE_ABS" && source .venv/bin/activate && pwd && ./tools/benchmark_rounds.sh -d $DEVICE -c $PTO_ISA_COMMIT -r "$RUNTIME" \
  2>&1 | tee "${PROJECT_ROOT}/tmp/benchmark_baseline_${TIMESTAMP}_${RUNTIME}.txt"

#    Then current (from main workspace — default cwd, no venv)
./tools/benchmark_rounds.sh -d $DEVICE -c $PTO_ISA_COMMIT -r "$RUNTIME" \
  2>&1 | tee "tmp/benchmark_current_${TIMESTAMP}_${RUNTIME}.txt"

# 3. Cleanup
git -C "$PROJECT_ROOT" worktree remove "$WORKTREE_ABS" --force
```

## Step 6: Report Results

Parse `Trimmed Avg:` for elapsed and `Orch Trimmed Avg:` for orchestration time from benchmark output.

### Single Mode

```text
Benchmark at: <short SHA>
Args: -d 4 -n 100

Example                          Elapsed (us)   Orch (us)
-------------------------------  ------------   ---------
alternating_matmul_add               1235.5       820.3
benchmark_bgemm                       892.1       650.2
...
```

### Compare Mode

Show comparison table with both Elapsed and Orch deltas, **grouped by runtime**:

```text
Merge-base: <short SHA>  →  HEAD: <short SHA> (+ uncommitted)
Args: -d 4 -n 100
Device: baseline=4, current=4  (or baseline=4, current=6)

### tensormap_and_ringbuffer

Example                      Base (us)   HEAD (us)   Delta (us)   Change (%)
---------------------------  ---------   ---------   ----------   ----------
alternating_matmul_add        1240.1      1235.5        -4.6       -0.37%
  (orch)                       830.0       820.3        -9.7       -1.17%
benchmark_bgemm                890.3       892.1        +1.8       +0.20%
  (orch)                       650.0       650.2        +0.2       +0.03%
...

Overall: X of Y examples improved, Z regressed
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
| No idle device and no `-d` specified | Prompt user to specify device ID |
| Benchmark script fails | Report which examples failed; continue with remaining |
| No timing data | Warn: "No timing markers — ensure `PTO2_PROFILING` is enabled" |
| All examples fail | Check: did you run `pip install -e .` in the worktree venv? |
| Worktree creation fails | Fall back to stash/checkout approach or report error |
| `Pre-built runtime binaries not found` | The venv `pip install -e .` should have built these; re-run it |
| `ModuleNotFoundError: _task_interface` | Venv not activated; add `source .venv/bin/activate &&` before the command |

## Checklist

- [ ] Mode detected (single vs compare)
- [ ] Idle device found or user-specified
- [ ] PTO-ISA pinned to CI commit
- [ ] `PROJECT_ROOT` and `WORKTREE_ABS` absolute paths computed
- [ ] (Compare mode) Worktree created, venv built with `pip install -e .`
- [ ] (Compare mode) Baseline completed — venv activated, `pwd` confirmed worktree path before running
- [ ] Current completed in main workspace
- [ ] Worktree cleaned up (compare mode)
- [ ] Results table presented with Elapsed + Orch times
- [ ] (Compare mode) Device difference noted if applicable
- [ ] (Compare mode) Regressions > 2% flagged
