Run the example at $ARGUMENTS with profiling enabled on hardware.

1. Verify the directory exists and contains `kernels/kernel_config.py` and `golden.py`
2. Run: `python examples/scripts/run_example.py -k $ARGUMENTS/kernels -g $ARGUMENTS/golden.py -p a2a3 --enable-profiling`
3. If the test passes, report the swimlane output file location in `outputs/`
4. Summarize the task statistics from the console output (per-function timing breakdown)
