---
name: dsv41-scheduler-benchmark
description: Run and compare DeepSeek V4.1 Flash decode scheduler benchmarks on Ascend A5. Use for pypto-lib dev branches or daily CI performance cases, Simpler tests/fdwic/dsv41_scheduler, A5 HBG no-graph baselines, runtime A/B or stacked-PR comparisons, and L1/L3 traces for these cases.
---

# DeepSeek V4.1 scheduler benchmarks on A5

## Choose the workload

- Use [pypto-lib decode](references/pypto-lib.md) for the live dev branch, pypto-lib daily CI cases, or a no-graph A/B with live model kernels. Read the case manifest in the chosen lib checkout.
- Use [Simpler FDWIC](references/simpler-fdwic.md) for vendored scheduler cases, the 18-case matrix, or reproducible scheduler traces with captured inputs. Use the suite README in the chosen Simpler checkout.
- Use both when checking whether a runtime change generalizes. Report each suite with its own case inputs, timed region, and rank aggregation.

## Run a matched comparison

1. Pin the workload and each runtime arm. Record the resolved commits and any local diff for the relevant repos, CANN and toolchain versions, the loaded runtime build and scheduler variant, device IDs, graph mode, input shape and routing, warmup, rounds, and the metric and its aggregation. Verify the loaded artifacts for each arm. For a runtime-only comparison, hold the generated business kernels, inputs, graph mode, and build flags fixed.
2. Follow the [A5 precheck](../onboard-arch-precheck/SKILL.md) and the [onboard reservation rules](../../rules/running-onboard.md). Build outside the reservation when possible. Use one representative case for smoke and iteration, then run the requested matrix. Group jobs by topology and reuse an existing reservation when it fits the workload. Keep the card budget and round count appropriate for the requested comparison.
3. Start with profiling disabled for performance samples. Validate golden or replay checks on every arm. Run baseline and candidate on the same physical device or pair under matching settings, alternating arm order where practical. Capture L1 for kernel lanes and L3 for scheduler stages in separate runs when investigating gaps. For a PR staircase, build each step from the same pinned base and report each absolute time and its change from the preceding step.
4. Save the report with the run configuration and references to raw results. Keep traces, captures, bundles, and binaries on the experiment host. Label missing or failed cases explicitly.

Keep `effective_us` and `device_wall_us` as distinct metrics. For a zero-scheduler-overhead critical path, use a separate trace-derived model with actual dependency edges and kernel durations. Include communication and asynchronous completion dependencies where observable, document any resource constraints used by the model, and leave the estimate unavailable when the trace cannot establish the needed events.
