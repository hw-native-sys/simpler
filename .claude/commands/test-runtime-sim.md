Run simulation tests for a single runtime specified by $ARGUMENTS.

1. Validate that `$ARGUMENTS` is one of: `host_build_graph`, `aicpu_build_graph`, `tensormap_and_ringbuffer`. If not, list the valid runtimes and stop
2. Detect CPU core count: `CORES=$(nproc)`
3. If `CORES >= 16`, run: `./ci.sh -p a2a3sim -r $ARGUMENTS --parallel`
4. Otherwise, run: `./ci.sh -p a2a3sim -r $ARGUMENTS`
5. Report the results summary (pass/fail counts)
6. If any tests fail, show the relevant error output
