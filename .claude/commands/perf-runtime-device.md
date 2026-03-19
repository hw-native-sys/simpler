Benchmark the hardware performance of all examples under `tests/device_tests/a2a3/<runtime>/`.

If `$ARGUMENTS` is provided, use it as the runtime name. Otherwise, default to `tensormap_and_ringbuffer`.

Reference `tools/benchmark_rounds.sh` for the full implementation pattern (device log resolution, timing parsing, reporting format).

1. Validate the runtime is one of: `host_build_graph`, `aicpu_build_graph`, `tensormap_and_ringbuffer`. If not, list valid runtimes and stop
2. Check `command -v npu-smi` — if not found, tell the user this requires hardware and stop
3. Run `npu-smi info`, find the lowest-ID idle device (HBM-Usage = 0). If none, stop
4. Enumerate all subdirectories under `tests/device_tests/a2a3/$ARGUMENTS/` that contain both `kernels/kernel_config.py` and `golden.py`
5. For each example, run the same `run_bench()` pattern from `tools/benchmark_rounds.sh`:
   - Snapshot logs, run `run_example.py` with `-n 10`, find new log, parse timing, report results
6. Print a final summary table with example name, average latency, trimmed average, and pass/fail
