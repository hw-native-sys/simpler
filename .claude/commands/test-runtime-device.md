Run hardware device tests for a single runtime specified by $ARGUMENTS.

1. Validate that `$ARGUMENTS` is one of: `host_build_graph`, `aicpu_build_graph`, `tensormap_and_ringbuffer`. If not, list the valid runtimes and stop
2. Check `command -v npu-smi` — if not found, tell the user to use `/test-runtime-sim` instead and stop
3. **Detect platform**: Run `npu-smi info` and parse the chip name. Map `910B`/`910C` → `a2a3`, `950` → `a5`. If unrecognized, warn and default to `a2a3`
4. Read `.github/workflows/ci.yml` to extract the current `-c` (pto-isa commit) and `-t` (timeout) flags from the `st-onboard-<platform>` job's `./ci.sh` invocation
5. From the `npu-smi info` output, find devices whose **HBM-Usage is 0** (idle)
6. From the idle devices, take **at most 4**. If no idle device is found, report the situation and stop
7. Build the device range flag: from the idle devices, find the **longest consecutive sub-range** (at most 4). Pass as `-d <start>-<end>`. If no consecutive pair exists, use the lowest-ID idle device as `-d <id>`
8. If **2 or more** idle devices selected, run: `./ci.sh -p <platform> -r $ARGUMENTS -d <range> -c <commit> -t <timeout> --parallel`
9. If only **1** idle device, run: `./ci.sh -p <platform> -r $ARGUMENTS -d <id> -c <commit> -t <timeout>`
10. Report the results summary (pass/fail counts per task)
11. If any tests fail, show the relevant error output and which device/round failed
