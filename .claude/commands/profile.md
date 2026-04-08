Run the example at $ARGUMENTS with profiling enabled on hardware.

1. Verify the directory exists and contains `kernels/kernel_config.py` and `golden.py`
2. Require the example path to live under `examples/a2a3/` or `examples/a5/`. If it does not, stop and report that root-level `examples/{runtime}/...` paths are invalid.
3. Infer the platform from the example path (`examples/a2a3/...` → `a2a3`, `examples/a5/...` → `a5`).
4. Run: `python examples/scripts/run_example.py -k $ARGUMENTS/kernels -g $ARGUMENTS/golden.py -p <platform> --enable-profiling`
5. If the test passes, report the swimlane output file location in `outputs/`
6. Summarize the task statistics from the console output (per-function timing breakdown)
