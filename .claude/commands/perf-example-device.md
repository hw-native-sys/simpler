Benchmark the hardware performance of a single example at $ARGUMENTS.

Reference `tools/benchmark_rounds.sh` for the full implementation pattern (device log resolution, timing parsing, reporting format). This skill runs the same logic but for a single example only.

1. Verify `$ARGUMENTS` exists and contains `kernels/kernel_config.py` and `golden.py`
2. Check `command -v npu-smi` — if not found, tell the user this requires hardware and stop
3. **Detect platform**: Run `npu-smi info` and parse the chip name. Map `910B`/`910C` → `a2a3`, `950` → `a5`. If unrecognized, warn and default to `a2a3`
4. Find the lowest-ID idle device (HBM-Usage = 0) from the `npu-smi info` output. If none, stop
5. Run the example following the same pattern as `run_bench()` in `tools/benchmark_rounds.sh`:
   - Snapshot logs, run `run_example.py` with `-n 10`, find new log, parse timing, report results
