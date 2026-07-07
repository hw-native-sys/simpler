# Getting Started

## Cloning the Repository

```bash
git clone <repo-url>
cd simpler
```

The pto-isa dependency will be automatically cloned when you first run an example that needs it.

## PTO ISA Headers

The pto-isa repository provides header files needed for kernel compilation on
the `a2a3` hardware platform and for examples that use PTO ISA intrinsics.
The selected PTO-ISA revision is controlled by the repo-root `pto_isa.pin`
file.

The test framework automatically handles PTO-ISA setup:

1. Reads the required commit from `pto_isa.pin`.
2. Reuses `build/pto-isa` as-is when it already sits at exactly the pinned
   commit.
3. Otherwise (missing, wrong revision, or a dirty working tree) re-clones
   `build/pto-isa` fresh over HTTPS directly at the pinned commit, rather than
   `git checkout`-ing over the existing checkout — a checkout aborts on local
   modifications and would strand the clone at the wrong revision.
4. Passes that managed checkout to the kernel/runtime compilers.

**Automatic Setup (Recommended):**

Just run your example. PTO-ISA will be cloned automatically on first run:

```bash
python examples/a2a3/host_build_graph/vector_example/test_vector_example.py \
  -p a2a3sim
```

**Manual Setup** (if auto-setup fails or you prefer manual control):

```bash
mkdir -p build
git clone --branch main https://github.com/hw-native-sys/pto-isa.git build/pto-isa
```

Manual setup still uses the standard managed location. Before it builds
runtimes or compiles kernels, `simpler` verifies that checkout is at the commit
in `pto_isa.pin`; if it isn't (or the working tree is dirty), it re-clones the
repository fresh at the pinned commit.

**Revision selection and compatibility checks:**

To use a different PTO-ISA revision, update `pto_isa.pin` to the desired
40-character commit SHA. This makes the selected revision visible in the repo
diff and applies the same revision to install-time runtime builds and run-time
kernel compilation.

For a2a3 onboard runtimes, builds record the actual PTO-ISA git HEAD used for
each runtime in `build/lib/pto_isa_build.json`. This JSON is artifact
provenance, not a second configuration source. If the metadata says a
pre-built runtime was built for an older pin, or a partial rebuild did not
record that runtime, lookup fails with a stale-binary diagnostic and asks you
to reinstall or rebuild:

```bash
cat build/lib/pto_isa_build.json
```

Existing installs that do not have `build/lib/pto_isa_build.json` continue to
work; they simply skip this stale-artifact check.

**Troubleshooting:**

- If git is not available: install git, or clone PTO-ISA manually into
  `build/pto-isa` on a machine that can access GitHub.
- If clone fails due to network: try again or manually clone with HTTPS into
  `build/pto-isa`.
- If runtime lookup reports stale PTO-ISA binaries: rerun `pip install` so
  `build/lib` is rebuilt for the current `pto_isa.pin`.

Note: for simulation platforms, PTO ISA headers are optional and only needed if
your kernels use PTO ISA intrinsics.

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

## Install

All workflows assume an activated project-local venv (see [`.claude/rules/venv-isolation.md`](../.claude/rules/venv-isolation.md) for why `--no-build-isolation` is required).

**Recommended daily-dev setup:**

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --no-build-isolation scikit-build-core nanobind cmake pytest torch
pip install --no-build-isolation -e .
```

Editing Python is instant (editable install). Editing C++ requires re-running `pip install --no-build-isolation -e .` (scikit-build-core's auto-rebuild is disabled because it interacts badly with pip's ephemeral build env — see `docs/python-packaging.md`).

**Other supported paths:** `pip install .` (non-editable), `pip install --no-build-isolation .`, `pip install -e .`, and `cmake + PYTHONPATH` (no pip). Full comparison of all 5 paths — what lands where, which entry points work under each, trade-offs — lives in [`docs/python-packaging.md`](python-packaging.md).

**Verifying an install:** the single source of truth is `tools/verify_packaging.sh`, which exercises all 5 install paths × 4 entry points from a fully clean state. CI runs the same script on macOS + Ubuntu (see the `packaging-matrix` job in `.github/workflows/ci.yml`).

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
python examples/a2a3/host_build_graph/vector_example/test_vector_example.py -p a2a3sim

# Hardware platform (requires Ascend device)
python examples/a2a3/host_build_graph/vector_example/test_vector_example.py -p a2a3 -d 0

# Or as a pytest batch:
pytest examples/a2a3/host_build_graph/vector_example --platform a2a3sim
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
from simpler.worker import Worker

# Create a level-2 Worker for device 0.
worker = Worker(level=2, device_id=0, platform="a2a3sim", runtime="tensormap_and_ringbuffer")
worker.init()

# Register the ChipCallable to obtain an opaque callable handle.
handle = worker.register(chip_callable)

# Execute the registered callable on device. Omitting block_dim uses the
# default 0 = auto, which DeviceRunner resolves to the max the AICore
# stream allows. Pass block_dim=<n> to pin a smaller value.
worker.run(handle, orch_args)

# Cleanup
worker.close()
```

Reach for the high-level `Worker` first. Public registration returns a
`CallableHandle`; integer callable slots are backend internals.

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
python examples/a2a3/host_build_graph/vector_example/test_vector_example.py -p a2a3 --device 0
```

## Notes

- **Device IDs**: 0-15 (typically device 9 used for examples)
- **Handshake cores**: Usually 3 (1c2v configuration: 1 core, 2 vector units)
- **Kernel compilation**: Requires `ASCEND_HOME_PATH` environment variable
- **Memory management**: MemoryAllocator automatically tracks allocations
- **Python requirement**: PyTorch for tensor operations in golden scripts

## Logging

Device logs written to `~/ascend/log/debug/device-<id>/`

Both host and AICPU kernel code use the unified `LOG_*` macros from
`common/unified_log.h`:

- `LOG_INFO_V0` .. `LOG_INFO_V9`: INFO with verbosity tier (V0 most verbose,
  V9 most must-see, V5 default)
- `LOG_DEBUG`: Debug messages
- `LOG_WARN`: Warnings
- `LOG_ERROR`: Error messages

Threshold is configured from Python via the `simpler` logger:
`logging.getLogger("simpler").setLevel(simpler.V3)`.
