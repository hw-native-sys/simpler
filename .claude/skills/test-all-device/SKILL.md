---
name: test-all-device
description: Run the full hardware (onboard) CI pipeline (examples + tests/st) with automatic platform and device detection. Use when the user asks to run all device/hardware tests or the full onboard CI locally.
---

# Run the full hardware CI pipeline with automatic device detection

Detection / isolation procedures referenced below live in
[`../../lib/onboard-detection.md`](../../lib/onboard-detection.md).

1. If `command -v npu-smi` is not found, tell the user to use `/test-all-sim`
   instead and stop.
2. **Precheck + detect platform** (§A) — gate on real silicon, then read the
   detected arch into `PLATFORM`.
3. **Extract CI timeout** (§D, `st-onboard-<platform>` job):
   `--pto-session-timeout`.
4. **Select a device range** (§C, range ≤4) — or, when wrapping in
   `task-submit`, let it pick via `--device auto --device-num <range size>`.
5. **Run through `task-submit`** (§E). The underlying command:

   ```bash
   pytest examples tests/st --platform <platform> --device <range-or-$TASK_DEVICE> \
     --pto-session-timeout <timeout> -v
   ```

   Parallelism is auto-driven by `--device`: on hardware, one in-flight
   subprocess per device (`--max-parallel auto` = `len(--device)`); see
   `docs/testing.md` for the full reuse hierarchy.
6. Report the results summary (pass/fail counts per task).
7. If any tests fail, show the relevant error output and which device failed.
