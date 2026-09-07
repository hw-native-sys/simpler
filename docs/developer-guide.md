# Developer Guide

## Directory Structure

```text
pto-runtime/
├── src/
│   ├── common/task_interface/            # Cross-architecture shared headers (data_type.h, tensor.h, task_args.h)
│   └── {arch}/                         # Architecture-specific code (a2a3, a5)
│       ├── platform/                   # Platform-specific implementations
│       │   ├── include/                # Shared headers (host/, aicpu/, aicore/, common/)
│       │   ├── shared/                 # Sources shared between onboard and sim backends (compiled into both)
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
│       ├── kernel_compiler.py         # Transitional, ships in wheel; use simpler_setup version
│       ├── runtime_compiler.py        # Transitional, ships in wheel; use simpler_setup version
│       ├── elf_parser.py              # Transitional, ships in wheel; use simpler_setup version
│       └── toolchain.py               # Transitional, ships in wheel; use simpler_setup version
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
│   ├── pto_isa.py                     # pinned PTO-ISA checkout management
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

**Architecture note:** binaries are built separately for each architecture and
backend. The onboard AICore targets are `dav-c220-cube/vec` for a2a3 and
`dav-c310-cube/vec` for a5, so their artifacts are not interchangeable.
`pip install .` builds both architectures when their toolchain is available;
build-tool availability does not imply that the host has both kinds of NPU.
Toolchain detection (`build_runtimes.py`):

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
| `simpler` | `python/simpler/` | Every module, including the runtime API and the four transitional compiler/toolchain modules |
| `simpler_setup` | `simpler_setup/` | Test framework + authoritative copies of the four transitional files |
| `_task_interface` | built from `python/bindings/` | Top-level nanobind extension (.so) |

**Migration direction:** existing `from simpler.{kernel_compiler,runtime_compiler,toolchain,elf_parser} import ...` should move to `from simpler_setup.{kernel_compiler,...} import ...`. The `simpler_setup` copies are authoritative. Both copies ship because `wheel.packages` includes both packages and there is no `wheel.exclude`; deletion of the transitional modules requires migrating their callers.

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

For the kernel-author contract (SPMD execution context, accessor functions,
and the gotcha around CCE topology intrinsics that breaks ports from native
CANN code), see [aicore-kernel-programming.md](aicore-kernel-programming.md).

## Build Workflow

### Initial setup

```bash
pip install --no-build-isolation -e .
```

`--no-build-isolation` is required: scikit-build-core needs the system `nanobind` and `cmake` (and any other build dep already in the venv); build isolation would hide them. The flag is also faster — no temporary build venv per install.

This builds the nanobind `_task_interface` extension **and** pre-builds runtime binaries for available toolchains into `build/lib/`. Sim platforms (a2a3sim, a5sim) are built when `gcc`/`g++` are available; onboard platforms (a2a3, a5) are built when `ccec` and the cross-compiler under `ASCEND_HOME_PATH` are available. Each architecture has its own sources, compiler target, and output directory.

### No rebuild on import

`pyproject.toml` sets `editable.rebuild = false`, so importing the package does **not** trigger a `cmake --build`. (Enabling it breaks under build isolation: the cmake path baked into `build.ninja` points into pip's deleted ephemeral env, so the next import fails with `cmake: No such file or directory` — see [Python packaging](python-packaging.md).) Therefore **any** C++ change — nanobind bindings or runtime/platform sources — only takes effect after re-running `pip install --no-build-isolation -e .`. The rebuild is incremental via the persistent cmake caches under `build/cache/` (~1-2s for a one-file change). CI should never use editable installs.

### When to rebuild

| What changed | Action |
| ------------ | ------ |
| First time / clean checkout | `pip install --no-build-isolation -e .` |
| Runtime C++ source (`src/{arch}/runtime/`, `src/{arch}/platform/`) | Re-run `pip install --no-build-isolation -e .` (or `.` for a non-editable install). Incremental via the cmake caches under `build/cache/` (~1-2s). |
| Nanobind bindings (`python/bindings/`) | Re-run `pip install --no-build-isolation -e .` (no rebuild-on-import; `editable.rebuild = false`) |
| Nothing — but `HEAD` moved (branch switch, rebase, pull, **or your own commit**) | Re-run `pip install ...`. You may have changed no compiled file, yet the tree moved out from under a binary frozen at install time. `simpler.task_interface` refuses to import on this skew (see below) rather than let struct fields read as 0. |
| Python-only code (`python/*.py`, `simpler_setup/*.py`) | Nothing to recompile (editable install) — but see the row above: once you *commit* it, the import guard keys on `HEAD`, so reinstall before the next import. |
| Examples / kernels (`examples/{arch}/`, `tests/st/`) | Nothing to recompile, just re-run — same commit caveat as the row above |
| `pto_isa.pin` changed | Re-run `pip install`. The cmake cache stamp and the `host_runtime` ccache key include the pinned PTO-ISA commit for a2a3 onboard (and a5 onboard when the SDMA overlay is enabled), so a pin bump invalidates stale runtime objects automatically. |

`_task_interface` records the commit it was compiled from, and
`simpler.task_interface` compares it against the working tree at import,
raising when they differ. It is git-based for the same reason
`RuntimeBuilder._build_cache_stamp` is — mtimes do not survive a branch switch —
and deliberately keyed on the whole HEAD rather than on the binding sources
alone: a root `CMakeLists.txt`, `pyproject.toml` or nanobind change alters the
extension without touching `python/bindings/`, and under-detecting here means
silently wrong values. The cost of over-detecting is one ~20 s reinstall, and
120 of the last 200 commits touched the ABI surface anyway — so most of those
reinstalls are ones you owed regardless, merely made visible.

Verifying that `import simpler` resolves into your worktree does **not** cover
this: under an editable install that only proves the *Python* is live, while the
compiled extension stays at whatever `pip install` produced.

The check is inert outside a source tree — a wheel has no `.git` to compare
against, and a build made without git carries an empty stamp.

A `pto_isa.pin` bump changes the SDMA headers embedded by
`host_runtime.so`. Install-time runtime builds and run-time kernel compilation
both read `pto_isa.pin`; use a different ISA revision by updating that file.

### Runtime binary lookup

Scene tests load pre-built runtime binaries from `build/lib/`. These are produced
by `build_runtimes.py` during `pip install`, using the persistent cmake cache in
`build/cache/` to recompile only what changed. If the binaries are missing, the
runtime loader raises with a hint to run `pip install --no-build-isolation .`.

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

The compiled binaries are packed into a `ChipCallable` (orch SO + each
child `CoreCallable`) and uploaded as a single blob via
`DeviceRunner::upload_chip_callable_buffer(callable)`, which fixes up each
child's `resolved_addr_`, H2Ds once, and returns the device address of the
ChipCallable header. The caller then derives each child's device address
from that header plus the child's recorded offset and writes it into
`Runtime::func_id_to_addr_[]` for AICPU dispatch.

## Features

- **Three programs compile independently** with clear API boundaries
- **Full Python API** via nanobind with torch integration
- **Modular design** enables parallel component development
- **Runtime linking** via binary loading
