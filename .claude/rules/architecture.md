# Architecture Quick Reference

## Three-Program Model

PTO Runtime compiles three separate programs per invocation:
- **Host** (`.so`): Runs on host CPU. Manages device memory, builds the task graph, launches execution.
- **AICPU** (`.so`): Runs on AICPU processor. Schedules tasks to AICore via handshake buffers.
- **AICore** (`.o`): Runs on AICore compute cores. Executes computation kernels (add, mul, matmul, etc.).

Each program is compiled independently with its own toolchain and linked at runtime.

## Runtime Variants

Three runtimes under `src/runtime/`, each providing a different graph-building strategy:
- **`host_build_graph`** -- Host CPU builds the full task dependency graph before launching
- **`aicpu_build_graph`** -- AICPU builds and manages the graph on-device
- **`tensormap_and_ringbuffer`** -- Advanced runtime with tensor maps, ring buffers, shared memory, and multi-core orchestration

Each runtime has a `build_config.py` declaring its include/source directories for the three components (host, aicpu, aicore). The `RUNTIME_CONFIG.runtime` field in `kernel_config.py` selects which runtime to use.

## Platform Backends

Two backends under `src/platform/`:
- **`a2a3`** -- Real Ascend hardware (requires CANN toolkit, `ccec` compiler for AICore, aarch64 cross-compiler for AICPU)
- **`a2a3sim`** -- Thread-based simulation (g++ only, no device required, runs on Linux and macOS)

Shared interfaces live in `src/platform/include/` (split into `host/`, `aicpu/`, `aicore/`, `common/`).
Platform-specific implementations live in `src/platform/src/` and `src/platform/a2a3/` or `src/platform/a2a3sim/`.

## Compilation Pipeline

Python modules under `python/` drive the build:
1. `kernel_compiler.py` -- compiles user-written kernel `.cpp` files (one per `func_id`)
2. `runtime_compiler.py` -- compiles runtime sources for each component (host, aicpu, aicore)
3. `runtime_builder.py` -- orchestrates the full build pipeline (compile + link)
4. `bindings.py` -- provides ctypes wrappers for calling the host `.so` from Python

## Example / Test Directory Layout

Every example and device test follows this structure:
```
my_example/
  golden.py              # generate_inputs() + compute_golden()
  kernels/
    kernel_config.py     # KERNELS list + ORCHESTRATION dict + RUNTIME_CONFIG
    aic/                 # AICore kernel sources (optional)
    aiv/                 # AIV kernel sources (optional)
    orchestration/       # Orchestration C++ source
```

Run with: `python examples/scripts/run_example.py -k <kernels_dir> -g <golden.py> -p <platform>`
