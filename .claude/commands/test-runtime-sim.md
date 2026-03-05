Run simulation tests for a single runtime specified by $ARGUMENTS.

1. Validate that `$ARGUMENTS` is one of: `host_build_graph`, `aicpu_build_graph`, `tensormap_and_ringbuffer`. If not, list the valid runtimes and stop
2. Read `.github/workflows/ci.yml` to extract the current `-c` (pto-isa commit) and `-t` (timeout) flags from the `run-example-on-sim` job's `./ci.sh` invocation
3. Detect CPU core count: `CORES=$(nproc)`
4. Build the command: `./ci.sh -p a2a3sim -r $ARGUMENTS -c <commit> -t <timeout>` and append `--parallel` if `CORES >= 16`
5. Run the command
6. Report the results summary (pass/fail counts)
7. If any tests fail, show the relevant error output
