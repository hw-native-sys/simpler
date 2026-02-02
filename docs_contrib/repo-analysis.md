# PTO Runtime — Repository Analysis (Contributor Guide)

This document explains how the repository fits together (Python → C API → host runtime → AICPU scheduler → AICore workers), where to change code, and what invariants to keep when adding new runtimes/examples.

## 1) What this repo builds (the “3 binaries” model)

Conceptually, PTO Runtime is **three independently built artifacts** that cooperate at runtime:

1. **Host runtime** (shared library) — manages device/sim execution and exposes a C API for Python.
2. **AICPU scheduler** (shared library) — runs the task scheduler (`aicpu_execute`) on device or in simulation.
3. **AICore program** (kernel object / shared library) — runs worker loops that execute task kernels by function pointer.

In Python, `RuntimeBuilder` compiles these three pieces for a selected platform (`a2a3` or `a2a3sim`). See `python/runtime_builder.py:9` and `python/runtime_builder.py:87`.

## 2) Repository layout (what lives where)

- **Platform code**: `src/platform/`
  - Hardware platform: `src/platform/a2a3/` (CANN/Ascend runtime integration).
  - Simulation platform: `src/platform/a2a3sim/` (host threads + dlopen + mmap).
  - Shared headers: `src/platform/include/common/` and `src/platform/include/host/`.
- **Runtime implementations**: `src/runtime/<runtime_name>/`
  - Each runtime provides per-target code (typically `host/`, `aicpu/`, `aicore/`, `runtime/`) plus `build_config.py`.
- **Python toolchain**: `python/`
  - `binary_compiler.py` builds the three binaries via CMake (`python/binary_compiler.py:9`).
  - `pto_compiler.py` builds orchestration `.so` and kernel objects, and uses CCEC for `a2a3` incore compilation (`python/pto_compiler.py:9`).
  - `bindings.py` is the ctypes binding over the shared C API (`python/bindings.py:52`).
- **Examples + runner**: `examples/`
  - `examples/scripts/run_example.py` is the CLI entry.
  - `examples/scripts/code_runner.py` orchestrates end-to-end compilation + execution.
- **Tests**: `tests/` (pytest).

## 3) How CI runs (what must stay working)

- GitHub CI runs **simulation** with Python 3.10 and only installs `numpy` (`.github/workflows/ci.yml:12`, `.github/workflows/ci.yml:57`).
- `./ci.sh` runs `pytest -v`, then runs `run_example.py` on macOS (sim only) and Linux (hardware + sim) (`ci.sh:7`, `ci.sh:13`).

If you change core scheduling or the runner defaults, keep the simulation path green first.

## 4) End-to-end flow (Python → C API → execution)

### 4.1 Python: compile + run

The default example flow is implemented in `examples/scripts/code_runner.py`:

1. **Build runtime binaries** (host/aicpu/aicore) via `RuntimeBuilder` and `BinaryCompiler` (`examples/scripts/code_runner.py:456`, `python/runtime_builder.py:57`).
2. **Compile orchestration** (host `.so`) via `PTOCompiler.compile_orchestration()` (`python/pto_compiler.py:231`).
3. **Compile and register kernels**:
   - Compile kernel `.o` (or sim `.so`) via `PTOCompiler.compile_incore()` (`python/pto_compiler.py:73`).
   - Extract `.text` and register with the host runtime (`python/elf_parser.py:26`, `python/bindings.py:252`).
4. **Initialize runtime** by calling the orchestration function via the host runtime C API (`python/bindings.py:150`).
5. **Launch** the runtime (`python/bindings.py:287`) with `aicpu_thread_num` and `block_dim`.

Note the runner defaults: `CodeRunner` sets `aicpu_thread_num = 3` and `block_dim = 3` (`examples/scripts/code_runner.py:314`).

### 4.2 C API: stable ABI boundary

The shared header `src/platform/include/host/pto_runtime_c_api.h` defines:
- `init_runtime()` for runtime graph construction (`src/platform/include/host/pto_runtime_c_api.h:63`)
- `launch_runtime()` for execution (`src/platform/include/host/pto_runtime_c_api.h:127`)
- `register_kernel()` to provide function binaries (`src/platform/include/host/pto_runtime_c_api.h:177`)

Python uses `ctypes` to bind these functions (`python/bindings.py:77`).

### 4.3 Runtime initialization (“graph build”)

For `host_build_graph`, `init_runtime()` ultimately loads an orchestration `.so`, resolves the orchestration function name, and calls it to populate the `Runtime` object (`src/runtime/host_build_graph/host/runtime_maker.cpp:61`, `src/runtime/host_build_graph/host/runtime_maker.cpp:103`).

Important: orchestration runs on the **host** process and uses `Runtime::host_api` function pointers for device memory operations (`src/platform/a2a3/host/pto_runtime_c_api.cpp:64`).

### 4.4 Launch: setting scheduling parameters and bootstrapping workers

On hardware `a2a3`, the host `DeviceRunner::run()`:

- Validates constraints:
  - `launch_aicpu_num` in `[1, PLATFORM_MAX_AICPU_THREADS]` (`src/platform/a2a3/host/device_runner.cpp:248`)
  - `block_dim % launch_aicpu_num == 0` (strict even distribution) (`src/platform/a2a3/host/device_runner.cpp:262`)
- Writes execution params into the runtime:
  - `runtime.worker_count` and `runtime.sche_cpu_num` (`src/platform/a2a3/host/device_runner.cpp:286`)
- Initializes handshake buffers and core typing (`src/platform/a2a3/host/device_runner.cpp:293`)
- **Sets `Task::function_bin_addr` for each task** from the kernel registration table (`src/platform/a2a3/host/device_runner.cpp:303`)
- Launches:
  - AICPU init kernel (`src/platform/a2a3/host/device_runner.cpp:324`)
  - AICPU main kernel with `launch_aicpu_num` instances (`src/platform/a2a3/host/device_runner.cpp:332`)
  - AICore kernel (`src/platform/a2a3/host/device_runner.cpp:340`)

The simulation platform follows the same contract but executes with host threads and dlopen’d executor symbols (`src/platform/a2a3sim/host/device_runner.cpp:157`, `src/platform/a2a3sim/host/device_runner.cpp:242`).

## 5) The core data model: `Runtime`, `Task`, `Handshake`

`src/runtime/host_build_graph/runtime/runtime.h` defines the core structures used by both platforms:

- `Handshake` is the AICPU↔AICore mailbox (`src/runtime/host_build_graph/runtime/runtime.h:92`).
- `Task` includes:
  - `func_id` (kernel ID)
  - `function_bin_addr` (runtime-dispatched function pointer target)
  - `fanin`/`fanout` dependency data (`src/runtime/host_build_graph/runtime/runtime.h:128`).
- `Runtime` embeds:
  - `workers[]` handshake array at offset 0 (important for kernel ABI) (`src/runtime/host_build_graph/runtime/runtime.h:169`)
  - `sche_cpu_num` used by AICPU to size scheduling threads (`src/runtime/host_build_graph/runtime/runtime.h:173`).

The ABI between host/AICPU/AICore depends on `KernelArgs` layout (`src/platform/include/common/kernel_args.h:57`).

## 6) Scheduler implementation details (AICPU)

The AICPU scheduler entrypoint is `aicpu_execute(Runtime*)` (`src/runtime/host_build_graph/aicpu/aicpu_executor.cpp:633`), called by the exported kernel wrapper (`src/platform/a2a3/aicpu/kernel.cpp:55`).

Key behaviors:

- One-time initialization per launch via `initialized_` + `init_done_` (`src/runtime/host_build_graph/aicpu/aicpu_executor.cpp:69`).
- “Core discovery” handshake:
  - AICPU sets `aicpu_ready` for all workers and waits `aicore_done` (`src/runtime/host_build_graph/aicpu/aicpu_executor.cpp:156`).
  - AICore reports `core_type` during that handshake (`src/runtime/host_build_graph/aicore/aicore_executor.cpp:57`).
- Strict distribution constraints:
  - AIC and AIV counts must be divisible by thread count (`src/runtime/host_build_graph/aicpu/aicpu_executor.cpp:229`).
- Execution loop:
  - Each AICPU thread gets a `thread_idx` and a fixed set of worker cores (`src/runtime/host_build_graph/aicpu/aicpu_executor.cpp:498`).
  - Threads poll worker handshakes, dispatch ready tasks, update fanin, and do timeout diagnostics (`src/runtime/host_build_graph/aicpu/aicpu_executor.cpp:312`).

Platform capacity constants are centralized in `src/platform/include/common/platform_config.h`, notably `PLATFORM_MAX_AICPU_THREADS = 4` (`src/platform/include/common/platform_config.h:41`).

## 7) Worker implementation details (AICore)

On AICore, tasks are executed by casting `Task::function_bin_addr` to a unified kernel function pointer and calling it with the task args (`src/runtime/host_build_graph/aicore/aicore_executor.cpp:10`, `src/runtime/host_build_graph/aicore/aicore_executor.cpp:43`).

This is why host-side kernel registration + `function_bin_addr` assignment is a correctness invariant: if `function_bin_addr == 0`, the task is skipped (`src/runtime/host_build_graph/aicore/aicore_executor.cpp:37`).

## 8) Adding a new runtime implementation (recommended pattern)

To add a runtime named `my_runtime`:

1. Create `src/runtime/my_runtime/` with `build_config.py` and subfolders (match `host_build_graph`):
   - `host/` (host-only runtime helpers like runtime maker/validators)
   - `aicpu/` (scheduler-side runtime logic)
   - `aicore/` (worker-side logic)
   - `runtime/` (shared data structures, if runtime-specific)
2. In `build_config.py`, define `BUILD_CONFIG` include/source dirs for `aicore`, `aicpu`, and `host` (see `python/runtime_builder.py:79` for how this is resolved).
3. Make it runnable via the example runner:
   - Ensure orchestration and kernels can include your runtime headers by adding your runtime’s `runtime/` include dir (see how `code_runner.py` constructs include dirs around `examples/scripts/code_runner.py:476`).

Tip: keep cross-platform headers in `src/platform/include/` and runtime-specific structures in `src/runtime/<name>/runtime/` when possible.

## 9) Adding a new example (how `run_example.py` discovers things)

An example is a folder containing:
- `kernels/kernel_config.py` describing kernels + orchestration sources.
- `golden.py` with:
  - `generate_inputs(params) -> dict`
  - `compute_golden(tensors, params) -> None`

The CLI requires `--kernels` and `--golden` (`examples/scripts/run_example.py:34`). `CodeRunner` validates this contract (`examples/scripts/code_runner.py:334`).

## 10) Style, safety, and “don’t break the ABI”

- Format C/C++ with `.clang-format` (Google base, 4 spaces, 120 columns).
- Prefer `snake_case` identifiers (repository history indicates a rename to snake_case).
- Treat these as ABI/stability constraints unless you are intentionally breaking compatibility:
  - `KernelArgs` layout (`src/platform/include/common/kernel_args.h:57`)
  - `Runtime` layout (workers at offset 0; used by both AICPU and AICore) (`src/runtime/host_build_graph/runtime/runtime.h:169`)
  - `Handshake` fields and semantics (`src/runtime/host_build_graph/runtime/runtime.h:92`)

## 11) Common failure modes (debug checklist)

- **Launch rejects parameters**: check `block_dim % aicpu_thread_num == 0` (`src/platform/a2a3/host/device_runner.cpp:262`) and thread/core divisibility in AICPU executor (`src/runtime/host_build_graph/aicpu/aicpu_executor.cpp:229`).
- **No kernels executed**: ensure kernels were registered and `function_bin_addr` was set (`src/platform/a2a3/host/device_runner.cpp:303`).
- **Hangs**: start with simulation to reproduce quickly; AICPU has timeout diagnostics (`src/runtime/host_build_graph/aicpu/aicpu_executor.cpp:475`).
- **PTO ISA headers missing**: set `PTO_ISA_ROOT`, or rely on the runner’s auto-clone logic in `code_runner.py`.

