# Run the full simulation CI pipeline

1. Read `.github/workflows/ci.yml` to extract the current `--pto-isa-commit` and `--pto-session-timeout` values from the `st-sim-*` jobs' `pytest` invocations
2. **Detect platform**: If `npu-smi` is available, parse the chip name from `npu-smi info`. Map `910B`/`910C` → `a2a3sim`, `950` → `a5sim`. If `npu-smi` is not found, default to `a2a3sim`
3. Build the command:

   ```bash
   pytest examples tests/st --platform <platform> \
     --pto-session-timeout <timeout> --clone-protocol https \
     --pto-isa-commit <commit> -v
   ```

4. Run the command (xdist parallelism is auto-enabled via `--max-parallel`, see `docs/testing.md`)
5. Report the results summary (pass/fail counts)
6. If any tests fail, show the relevant error output
