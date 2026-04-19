# Run the simulation test for the example at $ARGUMENTS

1. Locate the test file under `$ARGUMENTS/`: pick the single `test_*.py` that lives directly in that directory (not in a subdirectory). If none exists, tell the user the directory is not a scene test and stop.
2. Read `.github/workflows/ci.yml` to extract the current `--pto-isa-commit` value from the `st-sim-*` jobs' `pytest` invocations.
3. **Detect platform**: Infer the architecture from the path (e.g., `examples/a2a3/...` or `tests/st/a2a3/...` → `a2a3sim`; `examples/a5/...` or `tests/st/a5/...` → `a5sim`). If the path doesn't contain an arch prefix, default to `a2a3sim`.
4. Run standalone:

   ```bash
   python $ARGUMENTS/test_<name>.py -p <platform> \
     --clone-protocol https --pto-isa-commit <commit>
   ```

5. Report pass/fail status with any error output.
