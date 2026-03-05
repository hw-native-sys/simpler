Run the full simulation CI pipeline.

1. Read `.github/workflows/ci.yml` to extract the current `-c` (pto-isa commit) and `-t` (timeout) flags from the `run-example-on-sim` job's `./ci.sh` invocation
2. Detect CPU core count: `CORES=$(nproc)`
3. Build the command: `./ci.sh -p a2a3sim -c <commit> -t <timeout>` and append `--parallel` if `CORES >= 16`
4. Run the command
5. Report the results summary (pass/fail counts)
6. If any tests fail, show the relevant error output
