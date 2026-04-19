# Run the hardware device test for the example at $ARGUMENTS

1. Locate the test file under `$ARGUMENTS/`: pick the single `test_*.py` that lives directly in that directory (not in a subdirectory). If none exists, tell the user the directory is not a scene test and stop.
2. Check `command -v npu-smi` — if not found, tell the user to use `/test-example-sim` instead and stop.
3. **Detect platform**: Run `npu-smi info` and parse the chip name. Map `910B`/`910C` → `a2a3`, `950` → `a5`. If unrecognized, warn and default to `a2a3`.
4. Read `.github/workflows/ci.yml` to extract the current `--pto-isa-commit` value from the `st-onboard-<platform>` job's `pytest` invocation.
5. Pick an idle device: from `npu-smi info`, find one whose HBM-Usage is 0. If none is free, report and stop.
6. Run standalone:

   ```bash
   python $ARGUMENTS/test_<name>.py -p <platform> -d <device_id> \
     --clone-protocol https --pto-isa-commit <commit>
   ```

7. Report pass/fail status with any error output.
