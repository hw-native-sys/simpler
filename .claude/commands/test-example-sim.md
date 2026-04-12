Run the simulation test for the example at $ARGUMENTS.

1. Verify the directory exists and contains `kernels/kernel_config.py` and `golden.py`
2. Read `.github/workflows/ci.yml` to extract the current `-c` (pto-isa commit) flag from the `st-sim-*` jobs' `./ci.sh` invocations
3. **Detect platform**: Infer the architecture from the example path (e.g., `examples/a2a3/...` → `a2a3sim`, `examples/a5/...` → `a5sim`). If the path doesn't contain an arch prefix, default to `a2a3sim`
4. Run: `python examples/scripts/run_example.py -k $ARGUMENTS/kernels -g $ARGUMENTS/golden.py -p <platform> -c <commit>`
5. Report pass/fail status with any error output
