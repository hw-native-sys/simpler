Run the simulation test for the example at $ARGUMENTS.

1. Verify the directory exists and contains `kernels/kernel_config.py` and `golden.py`
2. Require the example path to live under `examples/a2a3/` or `examples/a5/`. If it does not, stop and report that root-level `examples/{runtime}/...` paths are invalid.
3. Read `.github/workflows/ci.yml` to extract the current `-c` (pto-isa commit) flag from the `st-sim-*` jobs' `./ci.sh` invocations
4. **Detect platform**: Infer the architecture from the example path (`examples/a2a3/...` → `a2a3sim`, `examples/a5/...` → `a5sim`).
5. Run: `python examples/scripts/run_example.py -k $ARGUMENTS/kernels -g $ARGUMENTS/golden.py -p <platform> -c <commit>`
6. Report pass/fail status with any error output
