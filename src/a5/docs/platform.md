# Platform Backends (a5)

Two platform backends under `src/a5/platform/`, providing different execution environments for the same runtime code.

## Comparison

| Feature | onboard | sim |
|---------|---------|-----|
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

Key directories:
- `src/a5/platform/sim/host/`
- `src/a5/platform/sim/aicpu/`
- `src/a5/platform/sim/aicore/`

## Shared Interface

Platform-agnostic headers in `src/a5/platform/include/`, shared source in `src/a5/platform/src/`.
