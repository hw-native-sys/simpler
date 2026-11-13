# Simpler's in-repo FDWIC suite

Use [the suite README](../../../../tests/fdwic/dsv41_scheduler/README.md) and [results](../../../../tests/fdwic/dsv41_scheduler/RESULTS.md) from the same checkout. The production ChipWorker runs captured graph slices with vendored generated code and frozen zero-weight inputs. Local cases materialize their prefix inputs, with prefix time outside the case measurement. The suite runs with the checked-in fixtures and the A5 onboard HBG runtime.

The routine baseline selects 20 configurations in [baseline_matrix.json](../../../../tests/fdwic/dsv41_scheduler/baseline_matrix.json). The source corpus still provides nine cuts in each original flavor, `pypto_lib` and `pypto_lib_dsl`.

| Cards | Retained configurations |
| ----- | ----------------------- |
| 1 | SWA: DSL original, bounded-sync, Flat x8, Graph x8; C2A full: original/sync; C1A reuse: Flat/Graph x8; C1A reindex: original/sync; Engram: original/sync |
| 2 | MoE and layers_19_21: PyPTO-lib, DSL original/Flat x1, bounded-sync, Graph x1 |

The sync variant adds sync_start only to compute tasks with 2–64 blocks. Do not count the two EP2 Flat x1 rows twice: they are the ordinary DSL controls. The current baseline uses same-run L1 metrics by explicit measurement contract; other runtime comparisons may still use the normal L0 performance workflow.

## Build and prepare

From the Simpler root, activate the worktree environment and source the matching CANN environment. Build the suite and the A5 onboard runtime as described in the suite README. Use the repository rebuild workflow for runtime edits. Supply the runtime directory expected by the checked-out runner, with its HBG libraries and the dispatcher at the location required by the selected revision. The verified `prepare.py` checks `../dispatcher/libsimpler_aicpu_dispatcher.so` relative to the supplied runtime directory. Verify that the capture/replay adapters, prepared plans, and each runtime arm are ABI-compatible.

```bash
export DSV41="$PWD/tests/fdwic/dsv41_scheduler"
export DSV41_RUNTIME="$PWD/build/lib/a5/onboard"
export PREPARED="$DSV41/build/prepared"
bash "$DSV41/build.sh" --pto-isa "$PWD/build/pto-isa"
python3 "$DSV41/production/prepare.py" "$DSV41_RUNTIME" "$PREPARED"
```

Prepare into a fresh path. Preparation captures the two-rank source forward and runs single-rank materialization. Let the wrapper acquire the right card count for each invocation. For faster iteration, pass `--flavor pypto_lib` or `--flavor pypto_lib_dsl` to both commands. Reuse a verified prepared bundle across comparable arms when fixture, generated kernels, adapters, and runtime ABI match. Regenerate the bundle when those inputs change.

## Run and collect

`run.sh` takes `BUNDLE RUNTIME OUTPUT LEVEL WARMUP ROUNDS`. The wrapper reserves the number of cards required by the bundle when `TASK_DEVICE` is unset. Allocate through `task-submit --device auto --device-num N`, then run matched configurations inside that same reservation using the assigned `TASK_DEVICE`. Do not set a fixed device sequence. Give each arm a fresh output directory.

```bash
export RUN_DIR="$DSV41/outputs/repro"
flavor=pypto_lib
case_name=swa
bash "$DSV41/run.sh" "$PREPARED/cases/$flavor/$case_name" \
  "$DSV41_RUNTIME" "$RUN_DIR/l1/$flavor/$case_name" 1 1 10
bash "$DSV41/run.sh" "$PREPARED/cases/$flavor/$case_name" \
  "$DSV41_RUNTIME" "$RUN_DIR/trace/$flavor/$case_name" all 1 3
```

Level `0` is the unprofiled performance path, `1` records kernel lanes, and `3` records scheduling phases. The retained baseline deliberately uses L1 for both reported metrics, including L1 overhead. `all` collects a short L0 series plus L1 and L3 traces. The analyzer reads results for every bundle in its cases directory. For a single-case run, give it a temporary directory containing a symlink to that bundle. For a complete flavor run, pass the full prepared flavor directory.

```bash
(
  selected_cases=$(mktemp -d) || exit 1
  trap 'rm -f -- "$selected_cases/$case_name"; rmdir -- "$selected_cases"' EXIT
  ln -s "$PREPARED/cases/$flavor/$case_name" "$selected_cases/$case_name" || exit 1
  PYTHONPATH=.:python python3 "$DSV41/analyze.py" \
    "$RUN_DIR/trace/$flavor" "$selected_cases" \
    "$PREPARED/capture/$flavor" \
    "$DSV41/build/portable/forward/$flavor/name_map.json" \
    "$RUN_DIR/report-$flavor-$case_name" --convert
)
```

Check rank results and replay validation. For the retained 20-configuration baseline, use one L1 warmup and ten measured L1 rounds. Report the arithmetic mean of device wall and the AICore kernel envelope, both from each same invocation. The envelope is max(kernel_end) minus min(kernel_start), including gaps between kernels. For two-card cases calculate each metric within each rank, take its maximum rank value per round, then average; never subtract absolute timestamps across devices. These means are not directly comparable to an L0 baseline.

Use `production/baseline_report.py` and the full retained matrix for this report. For the single-device Graphs, collect L4 separately to export the actual native kernel DAG with `production/graph_topology.py`; only its structure is used. Model weights remain the measured L1 kernel durations. The model removes scheduler costs and applies dependency, per-task capacity and total AIC/AIV work lower bounds. It does not construct an attainable schedule. Communication/defer_wait cases remain N/A when their logical completion events are unavailable; do not equate kernel return with completion or silently assume zero communication latency.

See the suite [RESULTS.md](../../../../tests/fdwic/dsv41_scheduler/RESULTS.md) for exact formulas, bundle identity checks and commands. Keep one current mean table, with raw samples and diagnostic captures on the experiment host.
