Run the full hardware CI pipeline with automatic device detection.

1. Check `command -v npu-smi` — if not found, tell the user to use `/test-all-sim` instead and stop
2. **Detect platform**: Run `npu-smi info` and parse the chip name. Map `910B`/`910C` → `a2a3`, `950` → `a5`. If unrecognized, warn and default to `a2a3`
3. Read `.github/workflows/ci.yml` to extract the current `-c` (pto-isa commit) and `-t` (timeout) flags from the `st-onboard-<platform>` job's `./ci.sh` invocation
4. From the `npu-smi info` output, find devices whose **HBM-Usage is 0** (idle)
5. From the idle devices, take **at most 4**. If no idle device is found, report the situation and stop
6. Build the device range flag: from the idle devices, find the **longest consecutive sub-range** (at most 4). Pass as `-d <start>-<end>`. If no consecutive pair exists, use the lowest-ID idle device as `-d <id>`
7. If **2 or more** idle devices selected, run: `./ci.sh -p <platform> -d <range> -c <commit> -t <timeout> --parallel`
8. If only **1** idle device, run: `./ci.sh -p <platform> -d <id> -c <commit> -t <timeout>`
9. Report the results summary (pass/fail counts per task)
10. If any tests fail, show the relevant error output and which device/round failed
