---
name: benchmark-pr
description: Benchmark a GitHub PR's performance impact by comparing benchmark results between the PR head and its fork point (merge-base). Use when the user asks to benchmark a PR, measure performance changes, or compare latency before/after a PR.
---

# Benchmark PR Workflow

Compare benchmark performance between a PR's changes and its merge-base to quantify performance impact.

## Task Tracking

Create tasks to track progress through this workflow:

1. Match input to PR and fetch metadata
2. Determine merge-base (fork point)
3. Pin PTO-ISA environment
4. Run baseline benchmark (merge-base)
5. Run PR benchmark (HEAD)
6. Compare and report results

## Input

Accept PR number (`123`, `#123`) with optional benchmark arguments:

```
/benchmark-pr #123
/benchmark-pr 123 -d 4 -n 20
```

Extra arguments (`-d`, `-n`, etc.) are forwarded to `tools/benchmark_rounds.sh`.

**Defaults** (when not specified): `-d 0 -n 10 -p a2a3`

## Step 1: Setup and PR Resolution

1. [Setup](../../lib/github/setup.md) — authenticate and detect context (role, remotes, state)

2. Use [lookup-pr](../../lib/github/lookup-pr.md) to find the PR:

```bash
PR_DATA=$(gh pr view $PR_NUMBER --repo "$PR_REPO_OWNER/$PR_REPO_NAME" \
  --json number,title,headRefName,baseRefName,state,headRefOid)
```

Validate PR state: OPEN or MERGED (continue), CLOSED (warn and confirm).

Extract fields:

```bash
HEAD_BRANCH=$(echo "$PR_DATA" | jq -r '.headRefName')
BASE_BRANCH=$(echo "$PR_DATA" | jq -r '.baseRefName')
HEAD_SHA=$(echo "$PR_DATA" | jq -r '.headRefOid')
PR_TITLE=$(echo "$PR_DATA" | jq -r '.title')
```

## Step 2: Determine Merge-Base (Fork Point)

The baseline is the merge-base between the PR and its target branch — the point where the PR diverged.

```bash
# Ensure upstream base branch is fresh
git fetch upstream "$BASE_BRANCH" --quiet

# Fetch the PR head
git fetch upstream "pull/$PR_NUMBER/head:pr-$PR_NUMBER" --quiet

# Compute fork point
MERGE_BASE=$(git merge-base "upstream/$BASE_BRANCH" "pr-$PR_NUMBER")
echo "Merge base: $MERGE_BASE"
echo "PR head:    $(git rev-parse pr-$PR_NUMBER)"
```

**Validation:** Confirm the merge-base is not the PR head itself:

```bash
if [ "$MERGE_BASE" = "$(git rev-parse pr-$PR_NUMBER)" ]; then
  echo "ERROR: merge-base equals PR head — PR has no commits"
  exit 1
fi
```

## Step 3: Stash and Prepare

Save current work before switching branches:

```bash
ORIGINAL_BRANCH=$(git branch --show-current)
STASHED=false
if [ -n "$(git status --porcelain)" ]; then
  git stash push -m "benchmark-pr: auto-stash before benchmarking"
  STASHED=true
fi
```

## Step 4: Pin PTO-ISA Environment

`tools/benchmark_rounds.sh` calls `run_example.py`, which auto-clones PTO-ISA to `examples/scripts/_deps/pto-isa/`. To ensure reproducible builds, pin PTO-ISA to the same commit used by CI.

Extract the pinned commit from `.github/workflows/ci.yml`:

```bash
PTO_ISA_COMMIT=$(grep -oP '(?<=-c )\w+' .github/workflows/ci.yml | head -1)
if [ -z "$PTO_ISA_COMMIT" ]; then
  echo "Could not parse PTO-ISA commit from ci.yml, using latest"
  PTO_ISA_COMMIT=""  # empty means run_example.py will use the latest PTO-ISA commit
fi
echo "PTO-ISA commit: ${PTO_ISA_COMMIT:-latest}"
```

If PTO-ISA is already cloned, pre-checkout the pinned commit so both baseline and PR use the same version. Otherwise, `run_example.py` will clone it automatically when invoked with the `-c` flag.

```bash
PTO_ISA_DIR="examples/scripts/_deps/pto-isa"
if [ -d "$PTO_ISA_DIR/.git" ]; then
  git -C "$PTO_ISA_DIR" fetch origin --quiet
  git -C "$PTO_ISA_DIR" checkout "$PTO_ISA_COMMIT" --quiet
fi
```

Append the commit flag to benchmark args so `run_example.py` picks it up:

```bash
if [ -n "$PTO_ISA_COMMIT" ]; then
  BENCH_ARGS="$BENCH_ARGS -c $PTO_ISA_COMMIT"
fi
```

Create timestamped temp files under the project directory to avoid conflicts with concurrent runs:

```bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p tmp
BENCH_BASELINE="tmp/benchmark_baseline_${TIMESTAMP}.txt"
BENCH_PR="tmp/benchmark_pr_${TIMESTAMP}.txt"
```

**Why this matters:** Without pinning, each checkout (merge-base vs PR HEAD) might resolve a different PTO-ISA version, introducing noise unrelated to the PR's actual changes.

## Step 5: Run Baseline Benchmark (Merge-Base)

```bash
git checkout "$MERGE_BASE" --quiet

echo "Running baseline benchmark at merge-base ($MERGE_BASE)..."
./tools/benchmark_rounds.sh $BENCH_ARGS 2>&1 | tee "$BENCH_BASELINE"
```

Capture the output. Parse trimmed averages per example from the output for comparison.

## Step 6: Run PR Benchmark (HEAD)

```bash
git checkout "pr-$PR_NUMBER" --quiet

echo "Running PR benchmark at HEAD ($HEAD_SHA)..."
./tools/benchmark_rounds.sh $BENCH_ARGS 2>&1 | tee "$BENCH_PR"
```

## Step 7: Restore Working State

```bash
git checkout "$ORIGINAL_BRANCH" --quiet
if [ "$STASHED" = true ]; then
  git stash pop
fi

# Clean up the temporary PR branch
git branch -D "pr-$PR_NUMBER" 2>/dev/null || true
```

## Step 8: Compare and Report

Parse both output files to extract per-example trimmed averages, then compute the delta.

Present results as a table:

```
PR #123: <PR title>
Merge-base: <short SHA>  →  PR HEAD: <short SHA>
Benchmark args: -d 4 -n 20

Example                     Baseline (us)   PR (us)   Delta (us)   Change (%)
--------------------------  -------------   -------   ----------   ----------
alternating_matmul_add         1240.1        1235.5       -4.6       -0.37%
benchmark_bgemm                 890.3         892.1       +1.8       +0.20%
paged_attention_unroll/Case1   2100.0        2050.3      -49.7       -2.37%
...

Overall: X of Y examples improved, Z regressed
```

**Interpretation guidelines:**

| Change (%) | Assessment |
| ---------- | ---------- |
| < -2% | Notable improvement |
| -2% to +2% | Within noise margin |
| > +2% | Potential regression — flag for review |

If any example shows > 5% regression, highlight it explicitly and recommend investigation.

## Error Handling

| Error | Action |
| ----- | ------ |
| PR not found | `gh pr list`; ask user |
| Benchmark script fails on baseline | Report which examples failed; continue with remaining |
| Benchmark script fails on PR | Same as above; compare only examples that succeeded on both |
| No timing data (profiling not enabled) | Warn user: "No timing markers found — ensure `PTO2_PROFILING` is enabled" |
| Device not available | Suggest simulation platform or different device ID |
| All examples fail and no `-d` was specified | Default device 0 is often occupied or unstable. Prompt user to specify a device ID explicitly (e.g. `/benchmark-pr #123 -d 4`). Suggest running `npu-smi info` to find idle devices (HBM-Usage = 0) |

## Checklist

- [ ] PR resolved and metadata fetched
- [ ] Merge-base computed correctly (fork point, not stale local branch)
- [ ] PTO-ISA pinned to CI commit
- [ ] Working state saved (stash if needed)
- [ ] Baseline benchmark completed at merge-base
- [ ] PR benchmark completed at HEAD
- [ ] Working state restored (branch + stash pop)
- [ ] Comparison table presented with per-example deltas
- [ ] Regressions > 2% flagged for attention
