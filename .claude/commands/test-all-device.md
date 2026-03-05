Run the full hardware CI pipeline with automatic device detection.

1. Check `command -v npu-smi` — if not found, tell the user to use `/test-all-sim` instead and stop
2. Read `.github/workflows/ci.yml` to extract the current `-c` (pto-isa commit) and `-t` (timeout) flags from the `run-example-on-device` job's `./ci.sh` invocation
3. Run `npu-smi info` and parse the output to find devices whose **HBM-Usage is 0** (idle)
4. From the idle devices, take **at most 4**. If no idle device is found, report the situation and stop
5. Build the device range flag: from the idle devices, find the **longest consecutive sub-range** (at most 4). Pass as `-d <start>-<end>`. If no consecutive pair exists, use the lowest-ID idle device as `-d <id>`
6. If **2 or more** idle devices selected, run: `./ci.sh -p a2a3 -d <range> -c <commit> -t <timeout> --parallel`
7. If only **1** idle device, run: `./ci.sh -p a2a3 -d <id> -c <commit> -t <timeout>`
8. Report the results summary (pass/fail counts per task)
9. If any tests fail, show the relevant error output and which device/round failed
