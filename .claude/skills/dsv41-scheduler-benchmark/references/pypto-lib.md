# Live pypto-lib decode and daily CI

## Select and freeze the workload

Use the checked-out pypto-lib branch's `.github/scripts/perf_cases.json`, `.github/scripts/ci_perf.py`, and `.github/workflows/daily_ci.yml` as the source for the case IDs, entrypoint flags, and CI environment. Set up PyPTO and its dependencies with the checkout's runtime pin and the Simpler [multi-repo setup](../../multi-repo-setup/SKILL.md). For a runtime A/B, apply each runtime override through the mechanism supported by the pinned PyPTO checkout and verify the actual binaries loaded by the worker.

The A5 decode set in the verified dev manifest contains six 1-card attention cases using 12 rows at context length 8192 and one 2-card balanced-routing MoE case.

| Case | Daily ID |
| ---- | -------- |
| SWA | `dsv4.1-flash-decode-swa-b12-8k` |
| C1A full | `dsv4.1-flash-decode-c1a-full-b12-8k` |
| C1A reindex | `dsv4.1-flash-decode-c1a-reindex-b12-8k` |
| C1A reuse | `dsv4.1-flash-decode-c1a-reuse-b12-8k` |
| C2A full | `dsv4.1-flash-decode-c2a-full-b12-8k` |
| C2A reuse | `dsv4.1-flash-decode-c2a-reuse-b12-8k` |
| MoE EP2 | `dsv4.1-flash-decode-moe-ep2-b12-eplb` |

For HBG no-graph, verify per-task submission in the generated orchestration, such as `rt_submit_task`, and compare the same submission mode on both arms. A configuration selecting HBG alone leaves graph submission mode ambiguous. If a dev-branch harness selects no-graph with a PyPTO pass context, record the harness and keep it identical across arms.

## Run

From the pypto-lib root in the prepared environment, use the daily entry point for a CI reproduction. It manages reservations and uses the manifest sampling.

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python .github/scripts/ci_perf.py ci --platform a5 --output-dir /path/to/run/daily
```

For a targeted experiment with this runner, execute `case` inside a matching reservation. The example is a short sample using daily workload arguments. The parent directory for the capture should exist. Use the manifest round counts for a full CI reproduction.

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYPTO_BENCH=1 PYPTO_BENCH_WARMUP=3 PYPTO_BENCH_ROUNDS=10
export CASE_ID=dsv4.1-flash-decode-swa-b12-8k
export CAPTURE=/path/to/run/swa/raw-result.json
mkdir -p "$(dirname "$CAPTURE")"
task-submit --device auto --device-num 1 --max-time 1800 \
  --run 'python .github/scripts/ci_perf.py case --case "$CASE_ID" --output "$CAPTURE"'
```

Use a two-device reservation for the EP2 case. Use the same explicit card or pair for matched arms. When running several cases within a reservation, set each case's output file and use the correct allocation size for that case.

## Read the timing

- Daily `effective_us` sums dispatch values within each rank and round, takes the median across rounds for each rank, and reports the fastest rank's median for MoE. Save rank results so the selected rank is auditable. The daily reducer validates sample counts against the manifest. For shorter experiments, report the actual sampling and reduce the per-round data with the same aggregation.
- For device wall time, use the pinned checkout's raw benchmark or strace capture and verify it contains `chip.run.runner_run.device_wall` per measured dispatch. The daily `case` JSON in the verified dev checkout serializes effective samples. Where supported, enable `PYPTO_BENCH_RAW=1` and `SIMPLER_DEVICE_STRACE_ENABLE=1` for a separate wall-time capture and extract the field from the raw per-dispatch records. Keep those settings identical across arms. When wall time is required, confirm the raw capture fields on a smoke run. For the fastest-rank MoE convention, take the minimum of the fixed-rank medians across rounds and label it in the table.

For a live lib percounter L1 capture with Perfetto export, see
[Capture percounter L1](percounter-l1.md).
