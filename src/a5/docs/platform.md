# Platform Backends (a5)

Two platform backends under `src/a5/platform/`, providing different execution environments for the same runtime code. For the underlying chip hardware layout (die / device-id mapping, AICPU and AICore counts, host bus), see [hardware.md](hardware.md).

## Comparison

| Feature | onboard | sim |
| ------- | ------- | --- |
| Execution | Real Ascend hardware | Thread-based host simulation |
| Requirements | CANN toolkit, `ccec`, aarch64 cross-compiler | gcc/g++ only |
| Use case | Production, hardware validation | Development, debugging, CI |

## onboard

Real hardware backend. Requires `ASCEND_HOME_PATH` environment variable.

Key directories:

- `src/a5/platform/onboard/host/`
- `src/a5/platform/onboard/aicpu/`
- `src/a5/platform/onboard/aicore/`

## sim

Thread-based simulation. No hardware or SDK required.

The simulator allocates one aligned 3 KiB SSBUF region per cluster, shared by
its AIC and two AIV threads and released after those threads exit. The
[HBG reservation](runtimes.md#host_build_graph) matches onboard execution.

Key directories:

- `src/a5/platform/sim/host/`
- `src/a5/platform/sim/aicpu/`
- `src/a5/platform/sim/aicore/`

## Shared Interface

Platform-agnostic headers in `src/a5/platform/include/`, shared source in `src/a5/platform/shared/`.
