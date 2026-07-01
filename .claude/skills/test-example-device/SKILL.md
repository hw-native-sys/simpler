---
name: test-example-device
description: Run the hardware (onboard) device test for a single example/scene-test directory ($ARGUMENTS). Use when the user asks to run or onboard one example on real hardware.
---

# Run the hardware device test for the example at $ARGUMENTS

Detection / isolation procedures referenced below live in
[`../../lib/onboard-detection.md`](../../lib/onboard-detection.md).

1. Locate the test file under `$ARGUMENTS/`: pick the single `test_*.py` that
   lives directly in that directory (not in a subdirectory). If none exists,
   tell the user the directory is not a scene test and stop.
2. If `command -v npu-smi` is not found, tell the user to use
   `/test-example-sim` instead and stop.
3. **Precheck + detect platform** (§A) — derive the arch from `$ARGUMENTS`,
   gate on real silicon, then read the detected arch into `PLATFORM`.
4. **Select a single idle device** (§C) — or let `task-submit` pick via
   `--device auto --device-num 1`.
5. **Run through `task-submit`** (§E, `--device-num 1`). The underlying
   command:

   ```bash
   python $ARGUMENTS/test_<name>.py -p <platform> -d $TASK_DEVICE
   ```

6. Report pass/fail status with any error output.
