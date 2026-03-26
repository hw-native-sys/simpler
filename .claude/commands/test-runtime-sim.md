Run simulation tests for a single runtime specified by $ARGUMENTS.

1. Validate that `$ARGUMENTS` is one of: `host_build_graph`, `aicpu_build_graph`, `tensormap_and_ringbuffer`. If not, list the valid runtimes and stop
2. Read `.github/workflows/ci.yml` to extract the current `-c` (pto-isa commit) and `-t` (timeout) flags from the `st-sim-*` jobs' `./ci.sh` invocations
3. Detect CPU core count: `CORES=$(nproc)`
4. **Detect platform**: If `npu-smi` is available, parse the chip name from `npu-smi info`. Map `910B`/`910C` → `a2a3sim`, `950` → `a5sim`. If `npu-smi` is not found, default to `a2a3sim`
5. Build the command: `./ci.sh -p <platform> -r $ARGUMENTS -c <commit> -t <timeout>` and append `--parallel` if `CORES >= 16`
6. Run the command
7. Report the results summary (pass/fail counts)
8. If any tests fail, show the relevant error output
