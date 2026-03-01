Run hardware device tests for a single runtime specified by $ARGUMENTS.

1. Validate that `$ARGUMENTS` is one of: `host_build_graph`, `aicpu_build_graph`, `tensormap_and_ringbuffer`. If not, list the valid runtimes and stop
2. Check `command -v npu-smi` — if not found, tell the user to use `/test-runtime-sim` instead and stop
3. Run `npu-smi info` and parse the output to find devices whose **HBM-Usage is 0** (idle)
4. From the idle devices, take **at most 4**. If no idle device is found, report the situation and stop
5. Build the device range flag: from the idle devices, find the **longest consecutive sub-range** (at most 4). Pass as `-d <start>-<end>`. If no consecutive pair exists, use the lowest-ID idle device as `-d <id>`
6. If **2 or more** idle devices selected, run: `./ci.sh -p a2a3 -r $ARGUMENTS -d <range> --parallel`
7. If only **1** idle device, run: `./ci.sh -p a2a3 -r $ARGUMENTS -d <id>`
8. Report the results summary (pass/fail counts per task)
9. If any tests fail, show the relevant error output and which device/round failed
