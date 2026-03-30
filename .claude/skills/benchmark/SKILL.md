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
- `aicpu_build_graph`

Each runtime has its own example list defined at the top of the script (`TMR_EXAMPLE_CASES` / `ABG_EXAMPLE_CASES`).

**Auto-detection (compare mode only):** Always benchmark TMR. Also benchmark `aicpu_build_graph` if the diff touches its files:

```bash
RUNTIMES_TO_BENCH=(tensormap_and_ringbuffer)
if git diff --name-only "$MERGE_BASE"...HEAD | grep -q 'aicpu_build_graph'; then
  RUNTIMES_TO_BENCH+=(aicpu_build_graph)
fi
```

Run `benchmark_rounds.sh` once per runtime, with `-r <runtime>` appended. **Runtimes are always benchmarked serially** — finish all baseline+current runs for one runtime before starting the next. This ensures no device ever runs two benchmark processes concurrently.

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
PTO_ISA_COMMIT=$(grep -oP '(?<=-c )\w+' .github/workflows/ci.yml | head -1)
```

Append `-c $PTO_ISA_COMMIT` to benchmark args so `run_example.py` picks it up.

## Step 4: Prepare

```bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p tmp
```

## Step 5: Run Benchmarks

### Single Mode

```bash
./tools/benchmark_rounds.sh $BENCH_ARGS -r "$RUNTIME" 2>&1 | tee "tmp/benchmark_${TIMESTAMP}.txt"
```

### Compare Mode

Use a **git worktree** for the baseline so the current workspace is never disturbed.

#### CRITICAL: Worktree needs runtime binaries built

The worktree is a fresh checkout at merge-base — it has **no pre-built runtime binaries** (`build/lib/` is in `.gitignore` and not checked in). Running `benchmark_rounds.sh` directly in the worktree will fail with:

```text
Pre-built runtime binaries not found for '...' (platform=a2a3)
```

**You MUST run `python examples/scripts/build_runtimes.py` inside the worktree** before benchmarking. This builds the runtime `.so` files into the worktree's own `build/lib/` directory. Unlike `pip install -e .`, it does NOT modify the shared Python environment, so baseline and current workspaces remain fully independent.

#### 5a. Create worktree and build binaries

```bash
WORKTREE_DIR="tmp/worktree_baseline_${TIMESTAMP}"
git worktree add "$WORKTREE_DIR" "$MERGE_BASE" --quiet

# CRITICAL: build runtime binaries in worktree
# Use cd inside the command — the Bash tool cwd does not persist across calls
cd "$WORKTREE_DIR" && python examples/scripts/build_runtimes.py 2>&1 | tail -5
```

**IMPORTANT — Bash tool cwd**: The Bash tool resets to the primary working directory for each call. To run commands inside the worktree, you MUST use `cd "$WORKTREE_DIR" && <command>` in a single Bash call. Do NOT rely on a previous `cd` persisting.

#### 5b. Run baseline

```bash
cd "$WORKTREE_DIR" && ./tools/benchmark_rounds.sh -d $BASELINE_DEVICE -c $PTO_ISA_COMMIT -r "$RUNTIME" \
  2>&1 | tee "$PROJECT_ROOT/tmp/benchmark_baseline_${TIMESTAMP}_${RUNTIME}.txt"
```

#### 5c. Run current

```bash
# Run current benchmark (from main workspace cwd, which is automatic)
./tools/benchmark_rounds.sh -d $CURRENT_DEVICE -c $PTO_ISA_COMMIT -r "$RUNTIME" \
  2>&1 | tee "tmp/benchmark_current_${TIMESTAMP}_${RUNTIME}.txt"
```

#### 5d. Cleanup

```bash
git worktree remove "$WORKTREE_DIR" --force
```

#### Parallel execution (two devices)

When two devices are available, run baseline and current **for the same runtime** in parallel on separate devices. Since `build_runtimes.py` only builds into the local `build/lib/` without modifying the shared Python environment, both workspaces are fully independent.

```bash
# For each runtime (serially):
for RUNTIME in "${RUNTIMES_TO_BENCH[@]}"; do
  # Baseline on device A, current on device B — parallel (different devices)
  cd "$WORKTREE_DIR" && ./tools/benchmark_rounds.sh -d $DEVICE_BASELINE -r "$RUNTIME" ... &
  ./tools/benchmark_rounds.sh -d $DEVICE_CURRENT -r "$RUNTIME" ... &
  wait  # Both finish before starting next runtime
done
```

**Never launch the next runtime until the current one finishes on all devices.**

#### Sequential execution (one device)

```bash
# 1. Build worktree binaries
cd "$WORKTREE_DIR" && python examples/scripts/build_runtimes.py

# 2. For each runtime (serially — one device, one process at a time):
for RUNTIME in "${RUNTIMES_TO_BENCH[@]}"; do
  # Baseline first
  cd "$WORKTREE_DIR" && ./tools/benchmark_rounds.sh -d $DEVICE -c $PTO_ISA_COMMIT -r "$RUNTIME" \
    2>&1 | tee "$PROJECT_ROOT/tmp/benchmark_baseline_${TIMESTAMP}_${RUNTIME}.txt"

  # Then current
  ./tools/benchmark_rounds.sh -d $DEVICE -c $PTO_ISA_COMMIT -r "$RUNTIME" \
    2>&1 | tee "tmp/benchmark_current_${TIMESTAMP}_${RUNTIME}.txt"
done

# 3. Cleanup
git worktree remove "$WORKTREE_DIR" --force
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
| All examples fail | Check: did you run `python examples/scripts/build_runtimes.py` in the worktree? |
| Worktree creation fails | Fall back to stash/checkout approach or report error |
| `Pre-built runtime binaries not found` | Run `python examples/scripts/build_runtimes.py` in that directory first |

## Checklist

- [ ] Mode detected (single vs compare)
- [ ] Idle device found or user-specified
- [ ] PTO-ISA pinned to CI commit
- [ ] (Compare mode) Worktree created and `python examples/scripts/build_runtimes.py` run inside it
- [ ] (Compare mode) Baseline completed from worktree (using `cd $WORKTREE_DIR && ./tools/benchmark_rounds.sh`)
- [ ] Current completed in workspace
- [ ] Worktree cleaned up (compare mode)
- [ ] Results table presented with Elapsed + Orch times
- [ ] (Compare mode) Device difference noted if applicable
- [ ] (Compare mode) Regressions > 2% flagged
