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

```
/benchmark
/benchmark -d 4 -n 50
```

Extra arguments (`-d`, `-n`, `-r`, etc.) are forwarded to `tools/benchmark_rounds.sh`.

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

Run `benchmark_rounds.sh` once per runtime, with `-r <runtime>` appended. Report results in separate tables per runtime.

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

Detect idle NPU devices (HBM-Usage = 0):

```bash
npu-smi info
```

Pick devices with **HBM-Usage = 0**. Find the longest consecutive sub-range (at most 4). If no idle device is found, prompt user to specify a device ID. If `-d` was provided in args, skip detection.

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

Use a **git worktree** for the baseline so the current workspace (with uncommitted changes) is never disturbed. Each run uses its own `./tools/benchmark_rounds.sh` so the script and `run_example.py` are always version-consistent.

**5a. Baseline (worktree at merge-base):**

```bash
WORKTREE_DIR="tmp/worktree_baseline_${TIMESTAMP}"
git worktree add "$WORKTREE_DIR" "$MERGE_BASE" --quiet

for RUNTIME in "${RUNTIMES_TO_BENCH[@]}"; do
  (cd "$WORKTREE_DIR" && ./tools/benchmark_rounds.sh $BENCH_ARGS -r "$RUNTIME") \
    2>&1 | tee "tmp/benchmark_baseline_${TIMESTAMP}_${RUNTIME}.txt"
done

git worktree remove "$WORKTREE_DIR" --force
```

**5b. Current (workspace with changes):**

Run directly in the current workspace — uncommitted changes are compiled as-is:

```bash
for RUNTIME in "${RUNTIMES_TO_BENCH[@]}"; do
  ./tools/benchmark_rounds.sh $BENCH_ARGS -r "$RUNTIME" 2>&1 | tee "tmp/benchmark_current_${TIMESTAMP}_${RUNTIME}.txt"
done
```

No stash, no checkout, no restore needed.

## Step 6: Report Results

Parse `Trimmed Avg:` for elapsed and `Orch Trimmed Avg:` for orchestration time from benchmark output.

### Single Mode

```
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

```
Merge-base: <short SHA>  →  HEAD: <short SHA> (+ uncommitted)
Args: -d 4 -n 100

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
| All examples fail | Suggest specifying device with `-d`. Run `npu-smi info` to find idle devices |
| Worktree creation fails | Fall back to stash/checkout approach or report error |

## Checklist

- [ ] Mode detected (single vs compare)
- [ ] Idle device found or user-specified
- [ ] PTO-ISA pinned to CI commit
- [ ] Baseline completed in worktree (compare mode)
- [ ] Current completed in workspace
- [ ] Worktree cleaned up (compare mode)
- [ ] Results table presented with Elapsed + Orch times
- [ ] (Compare mode) Regressions > 2% flagged
