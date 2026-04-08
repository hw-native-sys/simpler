Benchmark the hardware performance of a single example at $ARGUMENTS.

Reference `tools/benchmark_rounds.sh` for the full implementation pattern (device log resolution, timing parsing, reporting format). This skill runs the same logic but for a single example only.

1. Verify `$ARGUMENTS` exists and contains `kernels/kernel_config.py` and `golden.py`
2. Require the example path to live under `examples/a2a3/` or `examples/a5/`. If it does not, stop and report that root-level `examples/{runtime}/...` paths are invalid.
3. Check `command -v npu-smi` — if not found, tell the user this requires hardware and stop
4. **Detect platform**: Infer the architecture from the example path (`examples/a2a3/...` → `a2a3`, `examples/a5/...` → `a5`). Use `npu-smi info` only as a sanity check; if the detected chip family conflicts with the path, report the mismatch and stop instead of silently switching platforms.
5. Find the lowest-ID idle device (HBM-Usage = 0) from the `npu-smi info` output. If none, stop
6. Run the example following the same pattern as `run_bench()` in `tools/benchmark_rounds.sh`:
   - Snapshot logs, run `run_example.py` with `-n 10`, find new log, parse timing, report results
