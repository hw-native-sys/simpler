# Platform Layer

Two backends provide the same interface for different execution targets:

- **`a2a3/`** — Real Ascend hardware. Uses `ccec` for AICore, aarch64 cross-compiler for AICPU, host `g++` for Host.
- **`a2a3sim/`** — Thread-based simulation. Uses `g++` only. Runs on Linux and macOS without hardware.

## Key Directories

- `include/` — Shared interface headers split by component (`host/`, `aicpu/`, `aicore/`, `common/`)
- `src/` — Shared source (currently `aicpu/` and `host/`)
- `a2a3/` and `a2a3sim/` — Platform-specific implementations mirroring the same component split

## Rules

1. New platform APIs must be declared in `include/` and implemented in **both** `a2a3/` and `a2a3sim/`
2. Simulation must remain behaviorally equivalent to hardware — same handshake protocol, same memory layout
3. Use `volatile` on shared-memory struct members (see `.ai-instructions/coding/codestyle.md`)
4. Format with `clang-format -i <file>` before committing
