# Run simulation tests for a single runtime specified by $ARGUMENTS

1. Validate that `$ARGUMENTS` is one of: `host_build_graph`, `aicpu_build_graph`, `tensormap_and_ringbuffer`. If not, list the valid runtimes and stop.
2. Read `.github/workflows/ci.yml` to extract the current `--pto-isa-commit` and `--pto-session-timeout` values from the `st-sim-*` jobs' `pytest` invocations.
3. **Detect platform**: If `npu-smi` is available, parse the chip name from `npu-smi info`. Map `910B`/`910C` → `a2a3sim`, `950` → `a5sim`. If `npu-smi` is not found, default to `a2a3sim`.
4. Run:

   ```bash
   pytest examples tests/st --platform <platform> --runtime $ARGUMENTS \
     --pto-session-timeout <timeout> --clone-protocol https \
     --pto-isa-commit <commit> -v
   ```

   xdist parallelism is auto-selected via `--max-parallel auto` (min of nproc and device pool size on sim); see `docs/testing.md`.
5. Report the results summary (pass/fail counts).
6. If any tests fail, show the relevant error output.
