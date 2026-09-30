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
5. **Read the marker selector out of the same job** you took the timeout from.
   `st-onboard-a2a3` is three pytest passes, not one: the sweep deselects
   `-m "not sdma and not sdma_fault"`, a later step runs
   `-m "sdma and not sdma_fault"`, and a one-device step runs `-m sdma_fault`.
   `st-onboard-a5` uses
   `--exclude-level 4` and does not exclude `sdma`. Take the expression from
   `ci.yml` rather than assuming, so this skill cannot drift from the job it
   reproduces.
6. **Run through `task-submit`** (§E). On a5, the underlying command excludes
   only level-4 tests, so SDMA remains in the sweep:

   ```bash
   pytest examples tests/st --exclude-level 4 --platform a5 \
     --device <range-or-$TASK_DEVICE> --pto-session-timeout <timeout> -v
   ```

   On a platform whose job separates SDMA tests, the underlying sweep command
   is:

   ```bash
   pytest examples tests/st -m "not sdma and not sdma_fault" --exclude-level 4 --platform <platform> \
     --device <range-or-$TASK_DEVICE> --pto-session-timeout <timeout> -v
   ```

   Then the quarantined pass, after it — never at the same time, since the
   point of the split is that no fault-injection case meets a device that has
   already provisioned SDMA:

   ```bash
   pytest examples tests/st -m sdma --platform <platform> \
     --device <2 devs or $TASK_DEVICE> --pto-session-timeout <timeout> -v
   ```

   Dropping either pass changes what you tested: a single flat sweep is not
   what CI runs and reports failures CI never sees
   ([`../testing/SKILL.md`](../testing/SKILL.md), issue #1425).

   Parallelism is auto-driven by `--device`: on hardware, one in-flight
   subprocess per device (`--max-parallel auto` = `len(--device)`); see
   `docs/testing.md` for the full reuse hierarchy.
7. Report the results summary (pass/fail counts per task) across all applicable
   passes.
8. If any tests fail, show the relevant error output and which device failed.
