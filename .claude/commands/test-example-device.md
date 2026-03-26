Run the hardware device test for the example at $ARGUMENTS.

1. Verify the directory exists and contains `kernels/kernel_config.py` and `golden.py`
2. Check `command -v npu-smi` — if not found, tell the user to use `/test-example-sim` instead and stop
3. **Detect platform**: Run `npu-smi info` and parse the chip name. Map `910B`/`910C` → `a2a3`, `950` → `a5`. If unrecognized, warn and default to `a2a3`
4. Read `.github/workflows/ci.yml` to extract the current `-c` (pto-isa commit) flag from the `st-onboard-<platform>` job's `./ci.sh` invocation
5. Run: `python examples/scripts/run_example.py -k $ARGUMENTS/kernels -g $ARGUMENTS/golden.py -p <platform> -c <commit>`
6. Report pass/fail status with any error output
