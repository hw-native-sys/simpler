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
│           └── tensormap_and_ringbuffer/  # Advanced production runtime
│
├── python/                            # Language bindings
│   ├── bindings/                      # nanobind extension module (_task_interface)
│   │   ├── CMakeLists.txt
│   │   ├── task_interface.cpp
│   │   └── worker_bind.h
│   └── simpler/                       # Stable user-facing API (packaged in wheel)
│       ├── worker.py                  # Unified Worker (L2 single-chip, L3 distributed)
│       ├── task_interface.py          # Python re-exports of nanobind types + helpers
│       ├── env_manager.py             # Environment variable management
│       ├── kernel_compiler.py         # [transitional, NOT in wheel — use simpler_setup version]
│       ├── runtime_compiler.py        # [transitional, NOT in wheel — use simpler_setup version]
│       ├── elf_parser.py              # [transitional, NOT in wheel — use simpler_setup version]
│       └── toolchain.py               # [transitional, NOT in wheel — use simpler_setup version]
│
├── simpler_setup/                     # Test framework + authoritative compilers (packaged in wheel)
│   ├── scene_test.py                  # SceneTestCase + @scene_test decorator
│   ├── runtime_builder.py             # RuntimeBuilder (pre-built lookup or compile)
│   ├── runtime_compiler.py            # Authoritative copy of runtime cmake driver
│   ├── kernel_compiler.py             # Authoritative copy of kernel compiler
│   ├── toolchain.py                   # Authoritative copy
│   ├── elf_parser.py                  # Authoritative copy
│   ├── platform_info.py               # Platform/runtime discovery
│   ├── environment.py                 # PROJECT_ROOT resolver (wheel vs source tree)
│   ├── pto_isa.py                     # PTO_ISA_ROOT discovery
│   ├── build_runtimes.py              # Pre-build all runtime variants (invoked by pip install)
│   └── _assets/                       # (wheel only) src/ + build/lib/ shipped with wheel
│
├── examples/                          # Working examples
│   └── {arch}/                        # Architecture-specific examples
│       ├── host_build_graph/
│       └── tensormap_and_ringbuffer/
│
├── tests/                             # Test suite
│   ├── ut/                           # Unit tests
│   │   ├── py/                       # Python unit tests (pytest)
│   │   └── cpp/                      # C++ unit tests (GoogleTest)
│   └── st/                           # Device scene tests (hardware-only)
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

Runtime binaries (host `.so`, aicpu `.so`, aicore `.o`) are pre-built during `pip install .` and cached in `build/lib/{arch}/{variant}/{runtime}/`. After wheel install they are shipped under `simpler_setup/_assets/build/lib/...`; `simpler_setup/environment.py::PROJECT_ROOT` resolves the right location automatically (see [Path resolution](#path-resolution)). The pipeline:

1. `simpler_setup/build_runtimes.py` — detects available toolchains, iterates all (platform, runtime) combinations
2. `simpler_setup/runtime_builder.py` — orchestrates per-runtime build (lookup pre-built or compile)
3. `simpler_setup/runtime_compiler.py` — invokes cmake for each target (host, aicpu, aicore)

Persistent cmake build directories under `build/cache/` enable incremental compilation — only changed files are recompiled.

**Architecture note:** a2a3 and a5 differ only at runtime (device selection, block dimensions, etc.). The compiled binaries are architecture-independent — the same toolchain and flags produce artifacts that work on both chips. Therefore `pip install .` should build **all** architectures (both a2a3 and a5, both onboard and sim) whenever the corresponding toolchain is available. Toolchain detection (`build_runtimes.py`):

- **sim** (a2a3sim, a5sim): requires `gcc` + `g++` in `PATH`
- **onboard** (a2a3, a5): requires `ccec` in `PATH` + cross-compiler under `ASCEND_HOME_PATH`

### User code (per-example)

1. `simpler_setup/kernel_compiler.py` — compiles user-written kernel `.cpp` files (one per `func_id`)
2. `python/bindings/` — nanobind extension providing ChipWorker, task types, and distributed types to Python

### Path resolution

`simpler_setup/environment.py::PROJECT_ROOT` returns one of two layouts depending on how the package was installed:

| Install mode | `PROJECT_ROOT` | Where `src/` lives | Where `build/lib/` lives |
| ------------ | -------------- | ------------------ | ------------------------ |
| `pip install .` (wheel) | `<site-packages>/simpler_setup/_assets/` | `_assets/src/` | `_assets/build/lib/` |
| `pip install -e .` (editable) | repo root | `<repo>/src/` | `<repo>/build/lib/` |
| Source tree without install | repo root | `<repo>/src/` | `<repo>/build/lib/` |

The resolver uses `importlib.resources.files("simpler_setup") / "_assets"`. If `_assets/src/` exists there, it picks that; otherwise it falls back to `Path(__file__).parent.parent` (repo root). All compilers (`runtime_compiler`, `kernel_compiler`, `runtime_builder`) consume `PROJECT_ROOT` and stay agnostic to install mode.

### Python package layout

| Package | Where on disk | What it ships in wheel |
| ------- | ------------- | ---------------------- |
| `simpler` | `python/simpler/` | Stable user API: `task_interface`, `worker`, `env_manager`, `__init__` |
| `simpler` (excluded) | same dir | `kernel_compiler`, `runtime_compiler`, `toolchain`, `elf_parser` are **transitional** — present in source tree but excluded from wheel via `pyproject.toml::wheel.exclude` |
| `simpler_setup` | `simpler_setup/` | Test framework + authoritative copies of the four transitional files |
| `_task_interface` | built from `python/bindings/` | Top-level nanobind extension (.so) |

**Migration direction:** existing `from simpler.{kernel_compiler,runtime_compiler,toolchain,elf_parser} import ...` should move to `from simpler_setup.{kernel_compiler,...} import ...`. Once all callers migrate, the four transitional files in `python/simpler/` can be deleted and `wheel.exclude` cleaned up.

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

Every example and device test follows this structure:

```text
my_example/
  test_my_example.py     # @scene_test class (CALLABLE + CASES + generate_args + compute_golden)
  kernels/
    aic/                 # AICore kernel sources (optional)
    aiv/                 # AIV kernel sources (optional)
    orchestration/       # Orchestration C++ source
```

Run via pytest (`pytest examples tests/st --platform <platform>`) or standalone
(`python test_my_example.py -p <platform>`).

## Build Workflow

### Initial setup

```bash
pip install --no-build-isolation -e .
```

`--no-build-isolation` is required: scikit-build-core needs the system `nanobind` and `cmake` (and any other build dep already in the venv); build isolation would hide them. The flag is also faster — no temporary build venv per install.

This builds the nanobind `_task_interface` extension **and** pre-builds all runtime binaries for available toolchains into `build/lib/`. Sim platforms (a2a3sim, a5sim) are built when `gcc`/`g++` are available; onboard platforms (a2a3, a5) are built when `ccec` and the cross-compiler under `ASCEND_HOME_PATH` are available. Since a2a3 and a5 share the same compilation — differing only at runtime — both architectures are always built together when their toolchain is present.

### Editable rebuild on import

`pyproject.toml` enables `editable.rebuild = true`. Every fresh Python process that imports `simpler_setup` or `_task_interface` triggers `cmake --build` on the top-level CMake tree before the import returns. This auto-rebuilds the nanobind module when `python/bindings/` C++ changes — no manual `pip install -e .` needed. Also re-invokes `build_runtimes.py` (an `ALL` target), so the per-runtime cmake caches under `build/cache/` get checked too. Costs:

- Nothing changed: a few hundred ms per inner cmake check (10-30+ inner cmakes total depending on installed toolchains)
- Real C++ change: full incremental rebuild blocks the import until done

If startup latency becomes painful, run `unset SKBUILD_EDITABLE_REBUILD` before pytest, or run `pip install --no-build-isolation -e .` once and revert. CI should never use editable installs.

### When to rebuild

| What changed | Action |
| ------------ | ------ |
| First time / clean checkout | `pip install --no-build-isolation -e .` |
| Runtime C++ source (`src/{arch}/runtime/`, `src/{arch}/platform/`) | Pass `--build` to a standalone `python test_*.py` invocation (incremental, ~1-2s); editable rebuild does **not** cover this (runtime cmake is decoupled from the top-level cmake target). For pytest batch runs, pass `--build` at the pytest CLI. |
| Nanobind bindings (`python/bindings/`) | Auto-rebuilt on next import (`editable.rebuild = true`) |
| Python-only code (`python/*.py`, `simpler_setup/*.py`) | No rebuild needed (editable install) |
| Examples / kernels (`examples/{arch}/`, `tests/st/`) | No rebuild needed, just re-run |

### The `--build` flag

By default, scene tests load pre-built runtime binaries from `build/lib/`. When runtime C++ source has changed, pass `--build` to recompile incrementally:

```bash
# Standalone: single test file
python examples/a2a3/host_build_graph/vector_example/test_vector_example.py --build -p a2a3sim

# Pytest: applies to every test in the session
pytest examples tests/st --platform a2a3sim --build
```

This uses the persistent cmake cache in `build/cache/`, recompiling only what changed. In CI, `pip install .` pre-builds all runtimes before pytest runs, so examples use pre-built binaries.

### Disk layout

```text
build/
  cache/{arch}/{variant}/{runtime}/   # runtime cmake intermediate files (persistent)
    host/                             # cmake build dir for host target
    aicpu/                            # cmake build dir for aicpu target
    aicore/                           # cmake build dir for aicore target
  lib/{arch}/{variant}/{runtime}/     # runtime final binaries (stable lookup paths)
    libhost_runtime.so
    libaicpu_kernel.so
    aicore_kernel.o                   # or .so for sim
  {wheel_tag}/                        # scikit-build-core's top-level cmake (e.g. cp39-cp39-linux-x86_64)
                                      #   builds the _task_interface nanobind module
```

All three subdirs are siblings under `build/` and ignored by `.gitignore`. `rm -rf build/` clears everything.

## Dynamic Kernel Compilation

Kernels are compiled externally by `KernelCompiler` and uploaded to the device at runtime:

```python
from simpler_setup.kernel_compiler import KernelCompiler

compiler = KernelCompiler(platform="a2a3sim")
kernel_binary = compiler.compile_incore("path/to/kernel.cpp", core_type="aiv")
```

The compiled binary is then uploaded via `DeviceRunner::upload_kernel_binary(func_id, bin_data, bin_size)`, which loads it into device memory and returns the function address for task dispatch.

## Features

- **Three programs compile independently** with clear API boundaries
- **Full Python API** via nanobind with torch integration
- **Modular design** enables parallel component development
- **Runtime linking** via binary loading
