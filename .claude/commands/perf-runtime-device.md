# Benchmark the hardware performance of all scene tests under `tests/st/<arch>/<runtime>/`

If `$ARGUMENTS` is provided, use it as the runtime name. Otherwise, default to `tensormap_and_ringbuffer`.

Reference `tools/benchmark_rounds.sh` for the full implementation pattern (device log resolution, timing parsing, reporting format).

1. Validate the runtime is one of: `host_build_graph`, `tensormap_and_ringbuffer`. If not, list valid runtimes and stop.
2. Check `command -v npu-smi` — if not found, tell the user this requires hardware and stop.
3. **Detect platform**: Run `npu-smi info` and parse the chip name. Map `910B`/`910C` → `a2a3`, `950` → `a5`. If unrecognized, warn and default to `a2a3`.
4. Find the lowest-ID idle device (HBM-Usage = 0) from the `npu-smi info` output. If none, stop.
5. Enumerate all subdirectories under `tests/st/<arch>/$ARGUMENTS/` that contain a `test_*.py` directly beneath them.
6. For each example, run the same `run_bench()` pattern from `tools/benchmark_rounds.sh`:
   - Snapshot logs, run `python <example>/test_<name>.py -p <platform> -d <device_id> --rounds 10 --skip-golden`, find the new log, parse timing, report results.
7. Print a final summary table with example name, average latency, trimmed average, and pass/fail.
