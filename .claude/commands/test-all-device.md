Run the full hardware CI pipeline with automatic device detection.

1. Check `command -v npu-smi` — if not found, tell the user to use `/test-all-sim` instead and stop
2. Run `npu-smi info` and parse the output to find devices whose **HBM-Usage is 0** (idle)
3. From the idle devices, take **at most 4**. If no idle device is found, report the situation and stop
4. Build the device range flag: from the idle devices, find the **longest consecutive sub-range** (at most 4). Pass as `-d <start>-<end>`. If no consecutive pair exists, use the lowest-ID idle device as `-d <id>`
5. If **2 or more** idle devices selected, run: `./ci.sh -p a2a3 -d <range> --parallel`
6. If only **1** idle device, run: `./ci.sh -p a2a3 -d <id>`
7. Report the results summary (pass/fail counts per task)
8. If any tests fail, show the relevant error output and which device/round failed
