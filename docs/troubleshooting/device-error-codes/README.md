# Device Error Code Diagnostics

Focused procedures for chasing runtime and device failures live in this
directory. Start with [Device Error Codes](../device-error-codes.md) to identify
the reported code or watchdog signature, then use the matching guide below.

## Capacity and progress failures

| Document | What it covers |
| -------- | -------------- |
| [Capacity Codes](capacity.md) | Diagnosing codes 1, 2, 3, 4, 11, and 12 across HBG fixed capacity and TRB reclaim |
| [Stalls](stall.md) | Diagnosing scheduler timeout code 100 and tensor-wait code 8 by ordering watchdogs and locating the stuck task |

## Core faults and defensive codes

| Document | What it covers |
| -------- | -------------- |
| [AICore Faults](aicore-fault.md) | Separating kernel addressing faults from control-flow corruption using device logs and static inspection |
| [Codes Without End-to-End Tests](untested.md) | Why codes 10, 11, 103 and stall classes S4, S5, and unknown cannot be triggered through the public API |

## Related, outside this directory

| Document | What it covers |
| -------- | -------------- |
| [Local Runtime Timeouts](../local-timeout-defaults.md) | Default watchdog ordering and local overrides used to expose the most useful failure code |
| [Debug a Failed Run](../../user/how-to/debug-a-failed-run.md) | User-facing first-response workflow for collecting and classifying failure evidence |
| [Core Swimlane Profiling](../../dfx/core-swimlane-profiling.md) | Intra-core task timing for kernels that run but are unexpectedly slow |
