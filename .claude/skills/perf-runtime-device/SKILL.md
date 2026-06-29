---
name: perf-runtime-device
description: Benchmark the hardware performance of all scene tests under a runtime (tests/st/<arch>/<runtime>/, $ARGUMENTS). Use when the user asks to benchmark all examples of a runtime on hardware.
---

# Benchmark the hardware performance of all scene tests under `tests/st/<arch>/<runtime>/`

If `$ARGUMENTS` is provided, use it as the runtime name. Otherwise default to
`tensormap_and_ringbuffer`. Reference `tools/benchmark_rounds.sh` for the full
implementation pattern (device log resolution, timing parsing, reporting
format). Detection / isolation procedures live in
[`../../lib/onboard-detection.md`](../../lib/onboard-detection.md).

1. Validate the runtime is one of: `host_build_graph`,
   `tensormap_and_ringbuffer`. If not, list valid runtimes and stop.
2. If `command -v npu-smi` is not found, tell the user this requires hardware
   and stop.
3. **Precheck + detect platform** (§A) — gate on real silicon, then read the
   detected arch into `PLATFORM`.
4. **Select a single idle device** (§C) — or let `task-submit` pick via
   `--device auto --device-num 1`. Hold one lock for the whole sweep (let
   `task-submit` own the loop) rather than re-locking per example.
5. Enumerate all subdirectories under `tests/st/<platform>/$ARGUMENTS/` that
   contain a `test_*.py` directly beneath them.
6. **Run through `task-submit`** (§E). For each example, run the same
   `run_bench()` pattern from `tools/benchmark_rounds.sh`: snapshot logs, run
   `python <example>/test_<name>.py -p <platform> -d $TASK_DEVICE --rounds 10 --skip-golden`,
   find the new log, parse timing, report results.
7. Print a final summary table with example name, average latency, trimmed
   average, and pass/fail.
