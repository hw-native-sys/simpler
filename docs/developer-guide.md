# Developer Guide

## Directory Structure

```text
pto-runtime/
├── src/
│   ├── common/task_interface/            # Cross-architecture shared headers (data_type.h, tensor_arg.h, task_args.h)
│   └── {arch}/                         # Architecture-specific code (a2a3, a5)
│       ├── platform/                   # Platform-specific implementations
│       │   ├── include/                # Shared headers (host/, aicpu/, aicore/, common/)
│       │   ├── src/                    # Shared source (compiled into both backends)
│       │   ├── onboard/               # Real hardware backend
│       │   │   ├── host/              # Host runtime (.so)
│       │   │   ├── aicpu/             # AICPU kernel (.so)
│       │   │   └── aicore/            # AICore kernel (.o)
│       │   └── sim/                   # Thread-based simulation backend
│       │       ├── host/
│       │       ├── aicpu/
│       │       └── aicore/
│       │
│       └── runtime/                   # Runtime implementations
│           ├── common/                # Shared components across runtimes
│           ├── host_build_graph/      # Host-built graph runtime
│           ├── aicpu_build_graph/     # AICPU-built graph runtime
│           └── tensormap_and_ringbuffer/  # Advanced production runtime
│
├── python/                            # Language bindings
│   ├── bindings.py                    # ctypes wrapper (C -> Python)
│   ├── runtime_compiler.py            # Multi-platform runtime compiler
│   ├── kernel_compiler.py             # Kernel compiler
│   ├── elf_parser.py                  # ELF binary parser
│   └── toolchain.py                   # Toolchain configuration
│
├── examples/                          # Working examples
│   ├── scripts/                       # Build and test framework
│   │   ├── run_example.py             # Run a single example
│   │   ├── code_runner.py             # Example execution engine
│   │   ├── runtime_builder.py         # Runtime binary builder (pre-built lookup or compile)
│   │   ├── build_runtimes.py          # Pre-build all runtime variants
│   │   └── platform_info.py           # Platform/runtime discovery utilities
│   └── {arch}/                        # Architecture-specific examples
│       ├── host_build_graph/
│       ├── aicpu_build_graph/
│       └── tensormap_and_ringbuffer/
│
├── tests/                             # Test suite
│   ├── ut/                           # Python unit tests
│   ├── st/                           # Device scene tests (hardware-only)
│   └── cpp/                          # C++ unit tests (GoogleTest)
│
└── docs/                              # Documentation
```

## Role-Based Directory Ownership

| Role | Directory | Responsibility |
| ---- | --------- | -------------- |
| **Platform Developer** | `src/{arch}/platform/` | Platform-specific logic and abstractions |
| **Runtime Developer** | `src/{arch}/runtime/` | Runtime logic (host, aicpu, aicore, common) |
| **Codegen Developer** | `examples/` | Code generation examples and kernel implementations |

**Rules:**

- Stay within your assigned directory unless explicitly requested otherwise
- Create new subdirectories under your assigned directory as needed
- When in doubt, ask before making changes to other areas

## Compilation Pipeline

The build has two layers: **runtime binaries** (platform-dependent, user-code-independent) and **user code** (orchestration + kernels, compiled per-example).

### Runtime binaries

Runtime binaries (host `.so`, aicpu `.so`, aicore `.o`) are pre-built during `pip install .` and cached in `build/lib/{arch}/{variant}/{runtime}/`. The pipeline:

1. `examples/scripts/build_runtimes.py` — detects available toolchains, iterates all (platform, runtime) combinations
2. `examples/scripts/runtime_builder.py` — orchestrates per-runtime build (lookup pre-built or compile)
3. `python/runtime_compiler.py` — invokes cmake for each target (host, aicpu, aicore)

Persistent cmake build directories under `build/cache/` enable incremental compilation — only changed files are recompiled.

**Architecture note:** a2a3 and a5 differ only at runtime (device selection, block dimensions, etc.). The compiled binaries are architecture-independent — the same toolchain and flags produce artifacts that work on both chips. Therefore `pip install .` should build **all** architectures (both a2a3 and a5, both onboard and sim) whenever the corresponding toolchain is available. Toolchain detection (`build_runtimes.py`):

- **sim** (a2a3sim, a5sim): requires `gcc` + `g++` in `PATH`
- **onboard** (a2a3, a5): requires `ccec` in `PATH` + cross-compiler under `ASCEND_HOME_PATH`

### User code (per-example)

1. `python/kernel_compiler.py` — compiles user-written kernel `.cpp` files (one per `func_id`)
2. `python/bindings.py` — provides ctypes wrappers for calling the host `.so` from Python

## Cross-Platform Preprocessor Convention

When preprocessor guards are used to isolate platform code paths, the `__aarch64__` block must be placed first:

```cpp
#if defined(__aarch64__)
// aarch64 path (must be first)
#elif defined(__x86_64__)
// x86_64 host simulation path
#else
// other platforms
#endif
```

## Example / Test Layout

Examples must live under `examples/{arch}/{runtime}/{name}/`, and device scenes must
live under `tests/st/{arch}/{runtime}/{name}/`. Every example and device test follows
this structure:

```text
my_example/
  golden.py              # generate_inputs() + compute_golden()
  kernels/
    kernel_config.py     # KERNELS list + ORCHESTRATION dict + RUNTIME_CONFIG
    aic/                 # AICore kernel sources (optional)
    aiv/                 # AIV kernel sources (optional)
    orchestration/       # Orchestration C++ source
```

Run with: `python examples/scripts/run_example.py -k <kernels_dir> -g <golden.py> -p <platform>`

## Build Workflow

### Initial setup

```bash
pip install -e .
```

This builds the nanobind `_task_interface` extension **and** pre-builds all runtime binaries for available toolchains into `build/lib/`. Sim platforms (a2a3sim, a5sim) are built when `gcc`/`g++` are available; onboard platforms (a2a3, a5) are built when `ccec` and the cross-compiler under `ASCEND_HOME_PATH` are available. Since a2a3 and a5 share the same compilation — differing only at runtime — both architectures are always built together when their toolchain is present.

### When to rebuild

| What changed | Action |
| ------------ | ------ |
| First time / clean checkout | `pip install -e .` |
| Runtime C++ source (`src/{arch}/runtime/`, `src/{arch}/platform/`) | Pass `--build` to `run_example.py` (incremental, ~1-2s) |
| Nanobind bindings (`python/bindings/`) | Re-run `pip install -e .` |
| Python-only code (`python/*.py`, `examples/scripts/*.py`) | No rebuild needed (editable install) |
| Examples / kernels (`examples/{arch}/`, `tests/st/`) | No rebuild needed, just re-run |

### The `--build` flag

By default, `run_example.py` loads pre-built runtime binaries from `build/lib/`. When runtime C++ source has changed, pass `--build` to recompile incrementally:

```bash
python examples/scripts/run_example.py --build \
    -k examples/a2a3/host_build_graph/vector_example/kernels \
    -g examples/a2a3/host_build_graph/vector_example/golden.py \
    -p a2a3sim
```

This uses the persistent cmake cache in `build/cache/`, recompiling only what changed. In CI, `pip install .` pre-builds all runtimes before `ci.sh` runs, so examples use pre-built binaries.

### Disk layout

```text
build/
  cache/{arch}/{variant}/{runtime}/   # cmake intermediate files (persistent)
    host/                             # cmake build dir for host target
    aicpu/                            # cmake build dir for aicpu target
    aicore/                           # cmake build dir for aicore target
  lib/{arch}/{variant}/{runtime}/     # final binaries (stable lookup paths)
    libhost_runtime.so
    libaicpu_kernel.so
    aicore_kernel.o                   # or .so for sim
```

## Dynamic Kernel Compilation

Compile and load kernels at runtime without rebuilding:

```cpp
// In host code
runner.CompileAndLoadKernel(func_id, "path/to/kernel.cpp", core_type);
```

This compiles the kernel source using `ccec`, loads the binary to device memory, and registers it for task dispatch.

## Features

- **Three programs compile independently** with clear API boundaries
- **Full Python API** with ctypes and NumPy integration
- **Modular design** enables parallel component development
- **Runtime linking** via binary loading
