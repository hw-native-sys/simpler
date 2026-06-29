---
name: perf-example-device
description: Benchmark the hardware performance of a single example ($ARGUMENTS). Use when the user asks to benchmark or measure the latency of one example on hardware.
---

# Benchmark the hardware performance of a single example at $ARGUMENTS

Reference `tools/benchmark_rounds.sh` for the full implementation pattern
(device log resolution, timing parsing, reporting format) — this skill runs the
same logic for a single example. Detection / isolation procedures live in
[`../../lib/onboard-detection.md`](../../lib/onboard-detection.md).

1. Locate the test file under `$ARGUMENTS/`: pick the single `test_*.py` that
   lives directly in that directory. If none exists, tell the user the
   directory is not a scene test and stop.
2. If `command -v npu-smi` is not found, tell the user this requires hardware
   and stop.
3. **Precheck + detect platform** (§A) — derive the arch from `$ARGUMENTS`,
   gate on real silicon, then read the detected arch into `PLATFORM`.
4. **Select a single idle device** (§C) — or let `task-submit` pick via
   `--device auto --device-num 1`.
5. **Run through `task-submit`** (§E, `--device-num 1`) following the same
   pattern as `run_bench()` in `tools/benchmark_rounds.sh`: snapshot logs, run
   `python $ARGUMENTS/test_<name>.py -p <platform> -d $TASK_DEVICE --rounds 10 --skip-golden`,
   find the new log, parse timing, report results.
