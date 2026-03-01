Run the full simulation CI pipeline.

1. Detect CPU core count: `CORES=$(nproc)`
2. If `CORES >= 16`, run: `./ci.sh -p a2a3sim --parallel`
3. Otherwise, run: `./ci.sh -p a2a3sim`
4. Report the results summary (pass/fail counts)
5. If any tests fail, show the relevant error output
