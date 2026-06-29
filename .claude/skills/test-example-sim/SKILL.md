---
name: test-example-sim
description: Run the simulation test for a single example/scene-test directory ($ARGUMENTS). Use when the user asks to run one example in simulation.
---

# Run the simulation test for the example at $ARGUMENTS

Detection procedures referenced below live in
[`../../lib/onboard-detection.md`](../../lib/onboard-detection.md).

1. Locate the test file under `$ARGUMENTS/`: pick the single `test_*.py` that
   lives directly in that directory (not in a subdirectory). If none exists,
   tell the user the directory is not a scene test and stop.
2. **Extract CI pins** (§D, sim jobs): `--pto-isa-commit`.
3. **Detect platform** (sim, §B — prefer the "from a path" rule using
   `$ARGUMENTS`).
4. Run standalone:

   ```bash
   python $ARGUMENTS/test_<name>.py -p <platform> \
     --clone-protocol https --pto-isa-commit <commit>
   ```

5. Report pass/fail status with any error output.
