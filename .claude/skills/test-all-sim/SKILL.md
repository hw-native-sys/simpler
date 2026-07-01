---
name: test-all-sim
description: Run the full simulation CI pipeline (examples + tests/st) on a2a3sim/a5sim. Use when the user asks to run all sim tests or the full simulation CI locally.
---

# Run the full simulation CI pipeline

Detection procedures referenced below live in
[`../../lib/onboard-detection.md`](../../lib/onboard-detection.md).

1. **Extract CI timeout** (§D, sim jobs): `--pto-session-timeout`.
2. **Detect platform** (sim, §B).
3. Run:

   ```bash
   pytest examples tests/st --platform <platform> \
     --pto-session-timeout <timeout> -v
   ```

   xdist parallelism is auto-enabled via `--max-parallel`; see `docs/testing.md`.
4. Report the results summary (pass/fail counts).
5. If any tests fail, show the relevant error output.
