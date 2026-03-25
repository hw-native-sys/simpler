# Platform Backends (a2a3)

Two platform backends under `src/a2a3/platform/`, providing different execution environments for the same runtime code.

## Comparison

| Feature | onboard | sim |
|---------|---------|-----|
| Execution | Real Ascend hardware | Thread-based host simulation |
| Requirements | CANN toolkit, `ccec`, aarch64 cross-compiler | gcc/g++ only |
| AICore compilation | `ccec` (Bisheng CCE compiler) | g++ with `-D__CPU_SIM` |
| AICPU compilation | aarch64-target-linux-gnu-g++ | Host g++ |
| Host compilation | Host g++ | Host g++ |
| Device memory | Real GM/L1/L2 via Ascend driver | `malloc`-backed simulation |
| Use case | Production, hardware validation | Development, debugging, CI |

## onboard

Real hardware backend. Requires:
- `ASCEND_HOME_PATH` environment variable pointing to the Ascend toolkit
- `ccec` compiler for AICore kernels
- aarch64 cross-compiler for AICPU code

Key directories:
- `src/a2a3/platform/onboard/host/` — Host runtime library (device_runner, memory_allocator)
- `src/a2a3/platform/onboard/aicpu/` — AICPU kernel entry and platform registers
- `src/a2a3/platform/onboard/aicore/` — AICore kernel build (ccec + ld.lld)

## sim

Thread-based simulation. No hardware or SDK required. Each AICore/AICPU "device" runs as a host thread.

Key directories:
- `src/a2a3/platform/sim/host/` — Simulated device runner and memory
- `src/a2a3/platform/sim/aicpu/` — Simulated AICPU executor
- `src/a2a3/platform/sim/aicore/` — Simulated AICore executor

## Shared Interface

Platform-agnostic headers live in `src/a2a3/platform/include/`, split by target:
- `host/` — Host-side platform API
- `aicpu/` — AICPU platform API (registers, timing)
- `aicore/` — AICore platform API
- `common/` — Shared types and utilities (unified_log, tensor, common.h)

Shared source implementations in `src/a2a3/platform/src/`.
