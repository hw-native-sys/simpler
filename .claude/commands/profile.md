# Run the example at $ARGUMENTS with profiling enabled on hardware

1. Locate the test file under `$ARGUMENTS/`: pick the single `test_*.py` that lives directly in that directory. If none exists, tell the user the directory is not a scene test and stop.
2. Pick an idle device (HBM-Usage = 0) from `npu-smi info`. If none, stop.
3. Run: `python $ARGUMENTS/test_<name>.py -p a2a3 -d <device_id> --enable-profiling`
4. If the test passes, report the swimlane output file location in `outputs/`.
5. Summarize the task statistics from the console output (per-function timing breakdown).
