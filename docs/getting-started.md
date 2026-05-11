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
2. If not, clones pto-isa to `build/pto-isa` on first run
3. Passes the resolved path to the kernel compiler

**Automatic Setup (Recommended):**
Just run your example - pto-isa will be cloned automatically on first run:

```bash
python examples/a2a3/host_build_graph/vector_example/test_vector_example.py -p a2a3sim
```

By default, the auto-clone uses SSH (`git@github.com:...`). In CI or environments without SSH keys, use `--clone-protocol https`:

```bash
pytest examples --platform a2a3sim --clone-protocol https
```

**Manual Setup** (if auto-setup fails or you prefer manual control):

```bash
mkdir -p build
git clone --branch main git@github.com:PTO-ISA/pto-isa.git build/pto-isa

# Or use HTTPS
git clone --branch main https://github.com/PTO-ISA/pto-isa.git build/pto-isa

# Set environment variable (optional - auto-detected if in standard location)
export PTO_ISA_ROOT=$(pwd)/build/pto-isa
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
from simpler.task_interface import ChipWorker
from simpler_setup.runtime_builder import RuntimeBuilder

# Build or locate pre-built runtime binaries
builder = RuntimeBuilder(platform="a2a3sim")
binaries = builder.get_binaries("tensormap_and_ringbuffer")

# Create worker and initialize with platform binaries (attaches the calling
# thread to device 0 internally — no separate set_device step required)
worker = ChipWorker()
worker.init(device_id=0, bins=binaries)

# Register the ChipCallable to obtain a callable_id
cid = worker.register(chip_callable)

# Execute the registered callable on device
worker.run(cid, orch_args, block_dim=24)

# Cleanup
worker.finalize()
```

`ChipWorker` follows the same `register → run(cid)` contract as
`Worker(level=2)`; reach for the high-level `Worker` first and use
`ChipWorker` only when a low-level handle is required.

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
