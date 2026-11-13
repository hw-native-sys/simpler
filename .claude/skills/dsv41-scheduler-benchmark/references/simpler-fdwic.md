# Simpler's in-repo FDWIC suite

Use [the suite README](../../../../tests/fdwic/dsv41_scheduler/README.md) and [results](../../../../tests/fdwic/dsv41_scheduler/RESULTS.md) from the same checkout. The production ChipWorker runs captured graph slices with vendored generated code and frozen zero-weight inputs. Local cases materialize their prefix inputs, with prefix time outside the case measurement. The suite runs with the checked-in fixtures and the A5 onboard HBG or percounter runtime. Percounter multi-device communication is not supported; mark its MoE and layers_19_21 entries unavailable.

The default matrix has nine cases per flavor, `pypto_lib` and `pypto_lib_dsl`.

| Cards | Cases |
| ----- | ----- |
| 1 | `swa`, `c2a_full`, `c2a_reuse`, `c1a_full`, `c1a_reindex`, `c1a_reuse`, `engram` |
| 2 | `moe`, `layers_19_21` |

The default DSL cases preserve original sync-start flags and add flags until approximately half of each case's SPMD tasks require sync-start. AIC/MIX additions are capped at 32 blocks and AIV at 64.
This is one 18-configuration matrix, without a third all-sync DSL group. Regenerate bundles with the current exporter; identify the policy from `case.json`.

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

`run.sh` takes `BUNDLE RUNTIME OUTPUT LEVEL WARMUP ROUNDS`. The wrapper reserves the number of cards required by the bundle when `TASK_DEVICE` is unset. Use `task-submit --device auto --device-num N` and run matched arms inside that allocation, using `TASK_DEVICE` as returned. Give each arm a fresh output directory. Append `--runtime=percounter` for percounter; HBG is the default.

```bash
export RUN_DIR="$DSV41/outputs/repro"
flavor=pypto_lib
case_name=swa
bash "$DSV41/run.sh" "$PREPARED/cases/$flavor/$case_name" \
  "$DSV41_RUNTIME" "$RUN_DIR/l0/$flavor/$case_name" 0 1 10
bash "$DSV41/run.sh" "$PREPARED/cases/$flavor/$case_name" \
  "$DSV41_RUNTIME" "$RUN_DIR/trace/$flavor/$case_name" all 1 3
```

Level `0` is for performance, `1` for kernel lanes, and `3` for scheduling phases. `all` collects a short L0 series plus L1 and L3 traces. The analyzer reads results for every bundle in its cases directory. For a single-case run, give it a temporary directory containing a symlink to that bundle. For a complete flavor run, pass the full prepared flavor directory.

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

Check rank results and replay validation. For the repository results protocol, use L0 device wall with one warmup and ten measured rounds, reporting the mean. For two-card cases use the maximum rank time in each round before taking the mean. If reproducing a separate fastest-rank experiment, report that statistic explicitly alongside the repository convention.
