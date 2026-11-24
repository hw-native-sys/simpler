---
name: dsv41-scheduler-benchmark
description: Benchmark DeepSeek V4.1 Flash decode on Ascend A5 and export Percounter swimlanes to Perfetto. Use for pypto-lib main, daily CI or dev cases, DSpark 8K EP2/EP8, Simpler FDWIC, runtime A/B and stacked-PR comparisons.
---

# DeepSeek V4.1 scheduler benchmarks on A5

## Choose the workload

- Use [pypto-lib decode](references/pypto-lib.md) for Lib main, a dev branch, daily CI cases, or a no-graph A/B with live model kernels. Read the case manifest in the chosen Lib checkout.
- Use [DSpark full decode at 8K](references/dspark-8k.md) for EP2/TP2 or EP8/TP8 target-plus-drafter benchmarks and separate capture and Perfetto export commands.
- Use [Simpler FDWIC](references/simpler-fdwic.md) for vendored scheduler cases, the 18-case matrix, or reproducible scheduler traces with captured inputs. Use the suite README in the chosen Simpler checkout.
- Use both when checking whether a runtime change generalizes. Report each suite with its own case inputs, timed region, and rank aggregation.
- For [operator Percounter swimlanes](references/percounter-l1.md), capture the matching dependency graph and check that the timing records cover the executable tasks.

## Run a matched comparison

1. Pin the workload and each runtime arm. Record the resolved commits and any local diff for the relevant repos, CANN and toolchain versions, the loaded runtime build and scheduler variant, device IDs, graph mode, input shape and routing, warmup, rounds, and the metric and its aggregation. Verify the loaded artifacts for each arm. For a runtime-only comparison, hold the generated business kernels, inputs, graph mode, and build flags fixed.
2. Follow the [A5 precheck](../onboard-arch-precheck/SKILL.md) and the [onboard reservation rules](../../rules/running-onboard.md). Compile, prime caches and prepare inputs before reserving cards. Check existing allocations and queued work against the authorized card budget. Reserve cards for replay and trace collection. Use one representative case for smoke and iteration, then run the requested matrix.
3. Start with profiling disabled for performance samples. Validate golden or replay checks on every arm. Run baseline and candidate on the same physical devices under matching settings, alternating arm order where practical. For calendar scheduler traces, use L2 for scheduling transitions and L3 or L4 for finer diagnosis in a separate capture. Export after releasing devices. For a PR staircase, build each step from the same pinned base and report each absolute time and its change from the preceding step.
4. Save the report with the run configuration and references to raw results. Keep traces, captures, bundles, and binaries on the experiment host. Label missing or failed cases explicitly.

When a request covers both Lib operators and DSpark, present one row per case in a combined report. Include the topology, effective median, device-wall median when collected, and run status. Link the underlying samples for each group.

Keep `effective_us` and `device_wall_us` as distinct metrics. For a zero-scheduler-overhead critical path, use a separate trace-derived model with actual dependency edges and kernel durations. Include communication and asynchronous completion dependencies where observable, document any resource constraints used by the model, and leave the estimate unavailable when the trace cannot establish the needed events.

Report the rank selected for performance separately from the source of the Perfetto trace. To provide the fastest observed trace, select the shortest physical Worker View across the available captures at the requested profiling level. Report its rank, dispatch, duration and absolute path on the experiment host.
