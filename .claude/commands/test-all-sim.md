Run the full simulation CI pipeline.

1. Read `.github/workflows/ci.yml` to extract the current `-c` (pto-isa commit) and `-t` (timeout) flags from the `st-sim-*` jobs' `./ci.sh` invocations
2. Detect CPU core count: `CORES=$(nproc)`
3. **Detect platform**: If `npu-smi` is available, parse the chip name from `npu-smi info`. Map `910B`/`910C` → `a2a3sim`, `950` → `a5sim`. If `npu-smi` is not found, default to `a2a3sim`
4. Build the command: `./ci.sh -p <platform> -c <commit> -t <timeout>` and append `--parallel` if `CORES >= 16`
5. Run the command
6. Report the results summary (pass/fail counts)
7. If any tests fail, show the relevant error output
