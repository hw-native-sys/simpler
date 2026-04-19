# Run the full hardware CI pipeline with automatic device detection

1. Check `command -v npu-smi` — if not found, tell the user to use `/test-all-sim` instead and stop
2. **Detect platform**: Run `npu-smi info` and parse the chip name. Map `910B`/`910C` → `a2a3`, `950` → `a5`. If unrecognized, warn and default to `a2a3`
3. Read `.github/workflows/ci.yml` to extract the current `--pto-isa-commit` and `--pto-session-timeout` values from the `st-onboard-<platform>` job's `pytest` invocation
4. From the `npu-smi info` output, find devices whose **HBM-Usage is 0** (idle)
5. From the idle devices, take **at most 4**. If no idle device is found, report the situation and stop
6. Build the device range flag: from the idle devices, find the **longest consecutive sub-range** (at most 4). Pass as `--device <start>-<end>`. If no consecutive pair exists, use the lowest-ID idle device as `--device <id>`
7. Run:

   ```bash
   pytest examples tests/st --platform <platform> --device <range-or-id> \
     --pto-session-timeout <timeout> --clone-protocol https \
     --pto-isa-commit <commit> -v
   ```

   Parallelism is auto-driven by `--device`: on hardware, one in-flight subprocess per device (`--max-parallel auto` = `len(--device)`); see `docs/testing.md` for the full reuse hierarchy.
8. Report the results summary (pass/fail counts per task)
9. If any tests fail, show the relevant error output and which device failed
