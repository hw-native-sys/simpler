Run the full hardware CI pipeline across multiple NPU devices.

1. Run: `./ci.sh -p a2a3 -d 4-7 --parallel`
2. Report the results summary (pass/fail counts per task)
3. If any tests fail, show the relevant error output and which device/round failed
