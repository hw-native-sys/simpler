# Benchmark the hardware performance of a single example at $ARGUMENTS

Reference `tools/benchmark_rounds.sh` for the full implementation pattern (device log resolution, timing parsing, reporting format). This skill runs the same logic but for a single example only.

1. Locate the test file under `$ARGUMENTS/`: pick the single `test_*.py` that lives directly in that directory. If none exists, tell the user the directory is not a scene test and stop.
2. Check `command -v npu-smi` — if not found, tell the user this requires hardware and stop.
3. **Detect platform**: Run `npu-smi info` and parse the chip name. Map `910B`/`910C` → `a2a3`, `950` → `a5`. If unrecognized, warn and default to `a2a3`.
4. Find the lowest-ID idle device (HBM-Usage = 0) from the `npu-smi info` output. If none, stop.
5. Run the example following the same pattern as `run_bench()` in `tools/benchmark_rounds.sh`:
   - Snapshot logs, run `python $ARGUMENTS/test_<name>.py -p <platform> -d <device_id> --rounds 10 --skip-golden`, find the new log, parse timing, report results.
