Run the simulation test for the example at $ARGUMENTS.

1. Verify the directory exists and contains `kernels/kernel_config.py` and `golden.py`
2. Read `.github/workflows/ci.yml` to extract the current `-c` (pto-isa commit) flag from the `run-example-on-sim` job's `./ci.sh` invocation
3. Run: `python examples/scripts/run_example.py -k $ARGUMENTS/kernels -g $ARGUMENTS/golden.py -p a2a3sim -c <commit>`
4. Report pass/fail status with any error output
