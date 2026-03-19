Benchmark the hardware performance of a single example at $ARGUMENTS under `tests/device_tests/a2a3/tensormap_and_ringbuffer/`.

Reference `tools/benchmark_rounds.sh` for the full implementation pattern (device log resolution, timing parsing, reporting format). This skill runs the same logic but for a single example only.

1. Verify `$ARGUMENTS` exists and contains `kernels/kernel_config.py` and `golden.py`
2. Check `command -v npu-smi` — if not found, tell the user this requires hardware and stop
3. Run `npu-smi info`, find the lowest-ID idle device (HBM-Usage = 0). If none, stop
4. Run the example following the same pattern as `run_bench()` in `tools/benchmark_rounds.sh`:
   - Snapshot logs, run `run_example.py` with `-n 10`, find new log, parse timing, report results
