Run the hardware device test for the example at $ARGUMENTS.

1. Verify the directory exists and contains `kernels/kernel_config.py` and `golden.py`
2. Require the example path to live under `examples/a2a3/` or `examples/a5/`. If it does not, stop and report that root-level `examples/{runtime}/...` paths are invalid.
3. Check `command -v npu-smi` — if not found, tell the user to use `/test-example-sim` instead and stop
4. **Detect platform**: Infer the architecture from the example path (`examples/a2a3/...` → `a2a3`, `examples/a5/...` → `a5`). Use `npu-smi info` only as a sanity check; if the detected chip family conflicts with the path, report the mismatch and stop instead of silently switching platforms.
5. Read `.github/workflows/ci.yml` to extract the current `-c` (pto-isa commit) flag from the `st-onboard-<platform>` job's `./ci.sh` invocation
6. Run: `python examples/scripts/run_example.py -k $ARGUMENTS/kernels -g $ARGUMENTS/golden.py -p <platform> -c <commit>`
7. Report pass/fail status with any error output
