---
name: profile
description: Run an example ($ARGUMENTS) with profiling enabled on hardware and report the swimlane/timing breakdown. Use when the user asks to profile an example on hardware.
---

# Run the example at $ARGUMENTS with profiling enabled on hardware

Detection / isolation procedures referenced below live in
[`../../lib/onboard-detection.md`](../../lib/onboard-detection.md).

1. Locate the test file under `$ARGUMENTS/`: pick the single `test_*.py` that
   lives directly in that directory. If none exists, tell the user the
   directory is not a scene test and stop.
2. **Precheck + detect platform** (§A) — derive the arch from `$ARGUMENTS`,
   gate on real silicon, then read the detected arch into `PLATFORM`.
3. **Select a single idle device** (§C) — or let `task-submit` pick via
   `--device auto --device-num 1`.
4. **Run through `task-submit`** (§E, `--device-num 1`):

   ```bash
   python $ARGUMENTS/test_<name>.py -p <platform> -d $TASK_DEVICE --enable-profiling
   ```

5. If the test passes, report the swimlane output file location in `outputs/`.
6. Summarize the task statistics from the console output (per-function timing
   breakdown).
