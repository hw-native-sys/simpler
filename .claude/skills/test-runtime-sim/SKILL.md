---
name: test-runtime-sim
description: Run simulation tests for one runtime ($ARGUMENTS — host_build_graph or tensormap_and_ringbuffer). Use when the user asks to test a single runtime in simulation.
---

# Run simulation tests for a single runtime specified by $ARGUMENTS

Detection procedures referenced below live in
[`../../lib/onboard-detection.md`](../../lib/onboard-detection.md).

1. Validate that `$ARGUMENTS` is one of: `host_build_graph`,
   `tensormap_and_ringbuffer`. If not, list the valid runtimes and stop.
2. **Extract CI timeout** (§D, sim jobs): `--pto-session-timeout`.
3. **Detect platform** (sim, §B).
4. Run:

   ```bash
   pytest examples tests/st --platform <platform> --runtime $ARGUMENTS \
     --pto-session-timeout <timeout> -v
   ```

   xdist parallelism is auto-selected via `--max-parallel auto` (min of nproc
   and device pool size on sim); see `docs/testing.md`.
5. Report the results summary (pass/fail counts).
6. If any tests fail, show the relevant error output.
