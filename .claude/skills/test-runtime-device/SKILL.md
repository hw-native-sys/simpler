---
name: test-runtime-device
description: Run hardware (onboard) device tests for one runtime ($ARGUMENTS — host_build_graph or tensormap_and_ringbuffer). Use when the user asks to onboard or test a single runtime on real hardware.
---

# Run hardware device tests for a single runtime specified by $ARGUMENTS

Detection / isolation procedures referenced below live in
[`../../lib/onboard-detection.md`](../../lib/onboard-detection.md).

1. Validate that `$ARGUMENTS` is one of: `host_build_graph`,
   `tensormap_and_ringbuffer`. If not, list the valid runtimes and stop.
2. If `command -v npu-smi` is not found, tell the user to use
   `/test-runtime-sim` instead and stop.
3. **Precheck + detect platform** (§A) — gate on real silicon, then read the
   detected arch into `PLATFORM`.
4. **Extract CI timeout** (§D, `st-onboard-<platform>` job):
   `--pto-session-timeout`.
5. **Select a device range** (§C, range ≤4) — or, when wrapping in
   `task-submit`, let it pick via `--device auto --device-num <range size>`.
6. **Run through `task-submit`** (§E). The underlying command:

   ```bash
   pytest examples tests/st --platform <platform> --runtime $ARGUMENTS \
     --device <range-or-$TASK_DEVICE> \
     --pto-session-timeout <timeout> -v
   ```

   Hardware parallelism is auto-driven by `--device` (one subprocess per
   device); no extra flag needed.
7. Report the results summary (pass/fail counts per task).
8. If any tests fail, show the relevant error output and which device failed.
