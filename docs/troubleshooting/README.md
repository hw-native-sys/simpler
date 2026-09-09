# Troubleshooting

Every troubleshooting reference lives in this directory. Start with the device
error-code reference when a run fails, or jump to the environment-specific
guide when the failure happens during build, test setup, or simulation.

## Runtime and device failures

| Document | What it covers |
| -------- | -------------- |
| [Device Error Codes](device-error-codes.md) | Runtime and CANN error classification, first-response triage, minimal reproductions, and links to focused diagnostic guides |
| [Local Runtime Timeouts](local-timeout-defaults.md) | Local and CI timeout defaults, override variables, and the required ordering for onboard watchdogs |
| [A2/A3 Worker Retirement](a2a3-worker-retirement.md) | Why EXITED is not permission to return, and the group-wide CLOSE / GM-release handoff |
| [A2/A3 Scheduling Tails](a2a3-scheduling-tails.md) | Periodic real-time CPU throttling, normal AICPU thread scheduling, and worker-policy lifetime |
| [a2a3 AICPU Shared-SO Device Fault](a2a3-507899-aicpu-shared-so-fault.md) | Diagnosing mass `507899`/`507018` cascades caused by an AICPU shared-library device fault |
| [Sim CPU Oversubscription](sim-oversubscription-hang.md) | Simulation hangs and false timeouts when AICPU and AICore host threads are CPU-starved |

## Build and test environments

| Document | What it covers |
| -------- | -------------- |
| [macOS Build and Lint Toolchain](macos-build.md) | Compiler, SDK, sysroot, and `clang-tidy` behavior in the macOS build pipeline |
| [macOS libomp Collision](macos-libomp-collision.md) | Duplicate OpenMP runtime crashes in single-process simulation CI and their mitigation |
| [C++ Unit-Test gtest ABI Mismatch](ut-cpp-gtest-abi.md) | Link failures caused by a system GoogleTest library built with the old libstdc++ ABI |

## Related, outside this directory

| Document | What it covers |
| -------- | -------------- |
| [DFX and Profiling](../dfx/README.md) | Timing, traces, dependencies, and resource-usage evidence for runs that complete or make partial progress |
| [Testing](../testing.md) | Test types, commands, platform selection, and failure-reproduction workflows |
| [Debug a Failed Run](../user/how-to/debug-a-failed-run.md) | User-facing first-response path from the surfaced error to the focused guides in this directory |
