# Getting Started

## Cloning the Repository

```bash
git clone <repo-url>
cd simpler
```

The pto-isa dependency will be automatically cloned when you first run an example that needs it.

## PTO ISA Headers

The pto-isa repository provides header files needed for kernel compilation on the `a2a3` (hardware) platform.

The test framework automatically handles PTO_ISA_ROOT setup:

1. Checks if `PTO_ISA_ROOT` is already set
2. If not, clones pto-isa to `examples/scripts/_deps/pto-isa` on first run
3. Passes the resolved path to the kernel compiler

**Automatic Setup (Recommended):**
Just run your example - pto-isa will be cloned automatically on first run:

```bash
python examples/scripts/run_example.py -k examples/a2a3/host_build_graph/vector_example/kernels \
                                       -g examples/a2a3/host_build_graph/vector_example/golden.py \
                                       -p a2a3sim
```

By default, the auto-clone uses SSH (`git@github.com:...`). In CI or environments without SSH keys, use `--clone-protocol https`:

```bash
python examples/scripts/run_example.py -k examples/a2a3/host_build_graph/vector_example/kernels \
                                       -g examples/a2a3/host_build_graph/vector_example/golden.py \
                                       -p a2a3sim --clone-protocol https
```

**Manual Setup** (if auto-setup fails or you prefer manual control):

```bash
mkdir -p examples/scripts/_deps
git clone --branch main git@github.com:PTO-ISA/pto-isa.git examples/scripts/_deps/pto-isa

# Or use HTTPS
git clone --branch main https://github.com/PTO-ISA/pto-isa.git examples/scripts/_deps/pto-isa

# Set environment variable (optional - auto-detected if in standard location)
export PTO_ISA_ROOT=$(pwd)/examples/scripts/_deps/pto-isa
```

**Using a Different Location:**

```bash
export PTO_ISA_ROOT=/path/to/your/pto-isa
```

**Troubleshooting:**

- If git is not available: Clone pto-isa manually and set `PTO_ISA_ROOT`
- If clone fails due to network: Try again or clone manually
- If SSH clone fails (e.g., in CI): Use `--clone-protocol https` or clone manually with HTTPS

Note: For the simulation platform (`a2a3sim`), PTO ISA headers are optional and only needed if your kernels use PTO ISA intrinsics.

## Prerequisites

- CMake 3.15+
- CANN toolkit with:
  - `ccec` compiler (AICore Bisheng CCE)
  - Cross-compiler for AICPU (aarch64-target-linux-gnu-gcc/g++)
- Standard C/C++ compiler (gcc/g++) for host
- Python 3 with development headers

## Environment Setup

```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

## Install / Develop Workflows

Three ways to get the project running, depending on your role. All three assume an activated project-local venv (see [`.claude/rules/venv-isolation.md`](../.claude/rules/venv-isolation.md)).

### At a glance

| Concern | `pip install .` | `pip install -e .` | `cmake + PYTHONPATH` |
| ------- | --------------- | ------------------ | -------------------- |
| **Who it's for** | Users / CI | Python + C++ developers | C++-only developers |
| **`simpler_setup` resolves to** | site-packages | source tree (via `.pth`) | source tree (via `PYTHONPATH`) |
| **`simpler` resolves to** | site-packages (4 files) | source tree `python/simpler/` (all 8 files) | source tree (all 8) |
| **`_task_interface.*.so` lives at** | site-packages root | `build/{wheel_tag}/` (finder-dispatched) | `python/_task_interface.*.so` |
| **`PROJECT_ROOT`** | `<site-packages>/simpler_setup/_assets/` | repo root | repo root |
| **`src/` found under** | `_assets/src/` | `<repo>/src/` | `<repo>/src/` |
| **`build/lib/` found under** | `_assets/build/lib/` | `<repo>/build/lib/` | `<repo>/build/lib/` |
| **Edit `.py` → effect** | reinstall required | immediate | immediate |
| **Edit nanobind `.cpp` → rebuild** | reinstall required | auto on next import (`editable.rebuild`) | manual `cmake --build build/` |
| **Edit runtime `src/` → rebuild** | reinstall or manual | manual (`--build` flag or explicit script) | manual |
| **`from simpler.kernel_compiler import`** | fails (excluded from wheel) | works (transitional copy — use `simpler_setup` instead) | works (transitional copy — use `simpler_setup` instead) |
| **`--build` path writable** | no (site-packages read-only) | yes | yes |

### 1. `pip install .` — user / CI install

```bash
pip install --no-build-isolation .
```

`--no-build-isolation` is required: scikit-build-core consumes the venv's already-installed `scikit-build-core`, `nanobind`, `cmake` directly; isolation would hide them.

**What lands in site-packages:**

```text
site-packages/
├── _task_interface.cpython-*.so      # nanobind extension
├── simpler/                          # stable 4 files only
│   ├── __init__.py
│   ├── env_manager.py
│   ├── task_interface.py
│   └── worker.py
└── simpler_setup/
    ├── *.py                          # test framework + authoritative compilers
    └── _assets/
        ├── src/                      # headers + orchestration sources
        └── build/lib/                # pre-built runtime binaries
```

**Limitations:**

- Python edits require `pip install .` again
- `from simpler.{kernel_compiler,runtime_compiler,toolchain,elf_parser} import ...` does **not** work — use `simpler_setup.*` for those
- `--build` (rebuild runtime from source) won't work (site-packages is read-only)

Best for: `ci.sh` jobs and downstream consumers who only need to run examples.

### 2. `pip install -e .` — editable developer install

```bash
pip install --no-build-isolation -e .
```

The build is invoked once during install; `pyproject.toml` sets `editable.rebuild = true`, so subsequent C++ changes are picked up automatically.

**What happens at install time:**

- `simpler_setup/` and `simpler/` get `.pth` redirects pointing at the source tree
- `_task_interface.*.so` is built into `build/{wheel_tag}/` and dispatched by scikit-build-core's import finder
- `build_runtimes.py` pre-builds runtime binaries into `<repo>/build/lib/`
- `install()` rules also populate `<site-packages>/simpler_setup/_assets/`, but those are shadowed by the source-tree redirect

**Rebuild behavior on import:**

Every fresh Python process that imports `simpler_setup` or `_task_interface` triggers `cmake --build` + `cmake --install` against the top-level CMakeLists before the import returns. This covers:

- nanobind module (`python/bindings/*.cpp`) — real incremental rebuild when source changed
- `build_runtimes` ALL target — re-invokes `build_runtimes.py`, which fans out to per-runtime inner cmakes (each fast no-op when nothing changed)

**Startup cost per fresh process:**

- Nothing changed: ~6-15 seconds, depending on how many toolchains are installed (one inner cmake per runtime × platform combination, each a few hundred ms)
- Real C++ change: full incremental rebuild blocks import until done

**What's still manual:**

- Runtime `src/{arch}/...` edits for `--build` code paths: pass `--build` to `run_example.py` (or re-run `build_runtimes.py`). `editable.rebuild` will also try, but the inner no-op walk is the same — running `--build` explicitly on the affected example is faster.
- Transitional `from simpler.{kernel_compiler,...} import ...` still works in editable mode (source tree has the files); migrate to `simpler_setup.*` when convenient.

Best for: daily development. Python edits are instant, C++ rebuilds without thinking about `pip install`.

**Turning off rebuild temporarily** (e.g. for faster pytest iteration when nothing C++ changed):

```bash
# One-off: skip rebuild for this invocation
SKBUILD_EDITABLE_REBUILD=0 pytest ...

# Or edit pyproject.toml to set editable.rebuild = false, then re-install
```

### 3. `cmake + PYTHONPATH` — manual C++ workflow

This path bypasses pip entirely. Useful if you want `compile_commands.json`, IDE integration, or are debugging a CMake-only concern.

```bash
# Dependencies (one-time, install into the venv)
pip install --no-build-isolation nanobind cmake scikit-build-core torch pytest

# Build
cmake -B build -S .
cmake --build build --parallel

# Make Python find the project
export PYTHONPATH="$(pwd):$(pwd)/python"

# Now run anything
python examples/scripts/run_example.py -k ... -g ... -p a2a3sim
```

**Why `PYTHONPATH="$(pwd):$(pwd)/python"`:**

- `$(pwd)` makes `simpler_setup` importable (it lives at repo root)
- `$(pwd)/python` makes `simpler.*` importable (lives under `python/simpler/`) and also finds `python/_task_interface.*.so`

In this mode `SKBUILD_MODE=OFF`, so CMakeLists.txt takes the non-install branch: the nanobind module's `LIBRARY_OUTPUT_DIRECTORY` is set to `<repo>/python/`, and no `install()` runs. `_assets/` is **not** created — `PROJECT_ROOT` falls back to the repo root.

**What's still needed from pip:**

- `find_package(nanobind CONFIG REQUIRED)` in `python/bindings/CMakeLists.txt` requires `nanobind` to be discoverable via its Python-installed CMake config. Even without `pip install .`, you need `pip install nanobind` in the active venv.

**Rebuild:**

- C++ (nanobind or runtime): manual `cmake --build build/`
- nanobind alone: `cmake --build build --target _task_interface`
- Runtime alone: `cmake --build build --target build_runtimes` (or just `run_example.py --build`)

**Limitations:**

- `editable.rebuild` and everything else in `[tool.scikit-build]` are **not consulted** — this path doesn't go through scikit-build-core
- You manage all dependencies manually
- Good for CMake-centric debugging; not the recommended daily loop

Best for: C++-only iteration, IDE integration, `tests/ut/cpp/` development.

## Build Process

The **RuntimeCompiler** class handles compilation of all three components separately:

```python
from simpler_setup.runtime_compiler import RuntimeCompiler

# For real Ascend hardware (requires CANN toolkit)
compiler = RuntimeCompiler(platform="a2a3")

# For simulation (no Ascend SDK needed)
compiler = RuntimeCompiler(platform="a2a3sim")

# Compile each component to independent binaries
aicore_binary = compiler.compile("aicore", include_dirs, source_dirs)    # → .o file
aicpu_binary = compiler.compile("aicpu", include_dirs, source_dirs)      # → .so file
host_binary = compiler.compile("host", include_dirs, source_dirs)        # → .so file
```

**Toolchains used:**

- **AICore**: Bisheng CCE (`ccec` compiler) → `.o` object file (a2a3 only)
- **AICPU**: aarch64 cross-compiler → `.so` shared object (a2a3 only)
- **Host**: Standard gcc/g++ → `.so` shared library
- **HostSim**: Standard gcc/g++ for all targets (a2a3sim)

## Quick Start

### Running an Example

```bash
# Simulation platform (no hardware required)
python examples/scripts/run_example.py \
  -k examples/a2a3/host_build_graph/vector_example/kernels \
  -g examples/a2a3/host_build_graph/vector_example/golden.py \
  -p a2a3sim

# Hardware platform (requires Ascend device)
python examples/scripts/run_example.py \
  -k examples/a2a3/host_build_graph/vector_example/kernels \
  -g examples/a2a3/host_build_graph/vector_example/golden.py \
  -p a2a3
```

Expected output:

```text
=== Building Runtime: host_build_graph (platform: a2a3sim) ===
...
=== Comparing Results ===
Comparing f: shape=(16384,), dtype=float32
  f: PASS (16384/16384 elements matched)

============================================================
TEST PASSED
============================================================
```

### Python API Example

```python
from simpler.task_interface import ChipWorker
from simpler_setup.runtime_builder import RuntimeBuilder

# Build or locate pre-built runtime binaries
builder = RuntimeBuilder(platform="a2a3sim")
binaries = builder.get_binaries("tensormap_and_ringbuffer")

# Create worker and initialize with platform binaries
worker = ChipWorker()
worker.init(host_path=str(binaries.host_path),
            aicpu_path=str(binaries.aicpu_path),
            aicore_path=str(binaries.aicore_path))
worker.set_device(device_id=0)

# Execute callable on device
worker.run(chip_callable, orch_args, block_dim=24)

# Cleanup
worker.reset_device()
worker.finalize()
```

## Configuration

### Compile-time Configuration (Runtime Limits)

In `src/{arch}/runtime/host_build_graph/runtime/runtime.h`:

```cpp
#define RUNTIME_MAX_TASKS 131072   // Maximum number of tasks
#define RUNTIME_MAX_ARGS 16        // Maximum arguments per task
#define RUNTIME_MAX_FANOUT 512     // Maximum successors per task
```

### Runtime Configuration

Runtime behavior is configured via `kernel_config.py` in each example:

```python
RUNTIME_CONFIG = {
    "runtime": "host_build_graph",    # Runtime to use
    "aicpu_thread_num": 3,            # Number of AICPU scheduler threads
    "block_dim": 3,                   # Number of AICore blocks (1 block = 1 AIC + 2 AIV)
}
```

Device selection is done via CLI flag:

```bash
python examples/scripts/run_example.py -k <kernels> -g <golden.py> -p a2a3 --device 0
```

## Notes

- **Device IDs**: 0-15 (typically device 9 used for examples)
- **Handshake cores**: Usually 3 (1c2v configuration: 1 core, 2 vector units)
- **Kernel compilation**: Requires `ASCEND_HOME_PATH` environment variable
- **Memory management**: MemoryAllocator automatically tracks allocations
- **Python requirement**: NumPy for efficient array operations

## Logging

Device logs written to `~/ascend/log/debug/device-<id>/`

Kernel uses macros:

- `DEV_INFO`: Informational messages
- `DEV_DEBUG`: Debug messages
- `DEV_WARN`: Warnings
- `DEV_ERROR`: Error messages
