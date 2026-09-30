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
6. **Read the marker selector out of the same job** you took the timeout from.
   `st-onboard-a2a3` is three pytest passes: the sweep deselects
   `-m "not sdma and not sdma_fault"`, a later step runs
   `-m "sdma and not sdma_fault"`, and a one-device step runs `-m sdma_fault`.
   `st-onboard-a5` uses `--exclude-level 4` and
   does not exclude `sdma`. Both quarantined tests are
   `tensormap_and_ringbuffer`, so the second pass is only needed when
   `$ARGUMENTS` names that runtime.
7. **Run through `task-submit`** (§E). On a5, the underlying command excludes
   only level-4 tests, so SDMA remains in the sweep:

   ```bash
   pytest examples tests/st --exclude-level 4 --platform a5 --runtime $ARGUMENTS \
     --device <range-or-$TASK_DEVICE> \
     --pto-session-timeout <timeout> -v
   ```

   On a platform whose job separates SDMA tests, the underlying sweep command
   is:

   ```bash
   pytest examples tests/st -m "not sdma and not sdma_fault" --exclude-level 4 --platform <platform> --runtime $ARGUMENTS \
     --device <range-or-$TASK_DEVICE> \
     --pto-session-timeout <timeout> -v
   ```

   Then, for `tensormap_and_ringbuffer` on a platform whose job carries the
   filter, the quarantined pass afterwards — never concurrently, since the
   split exists so that no fault-injection case meets a device that has already
   provisioned SDMA ([`../testing/SKILL.md`](../testing/SKILL.md), issue #1425):

   ```bash
   pytest examples tests/st -m sdma --platform <platform> --runtime $ARGUMENTS \
     --device <2 devs or $TASK_DEVICE> --pto-session-timeout <timeout> -v
   ```

   Hardware parallelism is auto-driven by `--device` (one subprocess per
   device); no extra flag needed.
8. Report the results summary (pass/fail counts per task) across all applicable
   passes.
9. If any tests fail, show the relevant error output and which device failed.
