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
python tests/st/a2a3/host_build_graph/vector_example/test_vector_example.py \
  -p a2a3sim --manual include
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

For platforms that embed PTO-ISA headers into onboard host runtimes (a2a3
always; a5 when an async workspace overlay is ON), builds record the actual
PTO-ISA git HEAD used for each runtime in `build/lib/pto_isa_build.json`.
This JSON is artifact provenance, not a second configuration source. Lookup
of those runtimes **requires** the metadata file: if it is missing, or if it
says a pre-built runtime was built for an older pin / omitted that runtime,
lookup fails with a diagnostic and asks you to reinstall or rebuild:

```bash
cat build/lib/pto_isa_build.json
```

**Troubleshooting:**

- If git is not available: install git, or clone PTO-ISA manually into
  `build/pto-isa` on a machine that can access GitHub.
- If clone fails due to network: try again or manually clone with HTTPS into
  `build/pto-isa`.
- If runtime lookup reports missing or stale PTO-ISA metadata/binaries: rerun
  `pip install` so `build/lib` is rebuilt for the current `pto_isa.pin`.
- Host compile receives the pin-resolved checkout as `-DPTO_ISA_ROOT=` (not
  via ambient `PTO_ISA_ROOT`); kernel compile uses the same
  `ensure_pto_isa_root()` path.

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
pip install --no-build-isolation \
  'scikit-build-core>=0.10.0' 'nanobind>=2.0.0,<3' 'cmake>=3.15' 'ninja>=1.11' 'pytest>=6.0' 'torch>=2.3'
pip install --no-build-isolation -e .
```

Editing Python is instant (editable install). Editing C++ requires re-running `pip install --no-build-isolation -e .` (scikit-build-core's auto-rebuild is disabled because it interacts badly with pip's ephemeral build env — see `docs/python-packaging.md`).

**Other supported paths:** `pip install .` (non-editable), `pip install --no-build-isolation .`, `pip install -e .`, and `cmake + PYTHONPATH` (no pip). Full comparison of all 5 paths — what lands where, which entry points work under each, trade-offs — lives in [`docs/python-packaging.md`](python-packaging.md).

**Verifying an install:** the single source of truth is `tools/verify_packaging.sh`, which exercises all 5 install paths × 2 entry points from a fully clean state. CI runs the same script on macOS + Ubuntu (see the `packaging-matrix` job in `.github/workflows/ci.yml`).

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
python tests/st/a2a3/host_build_graph/vector_example/test_vector_example.py -p a2a3sim --manual include

# Hardware platform (requires Ascend device)
python tests/st/a2a3/host_build_graph/vector_example/test_vector_example.py -p a2a3 -d 0

# Or as a pytest batch:
pytest tests/st/a2a3/host_build_graph/vector_example --platform a2a3sim --manual include
```

The simulator case is marked manual, so include `--manual include` to run it.
It computes `f = (a + b + 1) * (a + b + 2)` for 16,384 elements with `a=2`
and `b=3`, then compares every output against the golden value `42`. A
successful run exits with status 0; the pytest form reports a passed test.

### Python API Example

```python
from simpler.worker import Worker

# Create a level-2 Worker for device 0.
worker = Worker(level=2, device_id=0, platform="a2a3sim", runtime="tensormap_and_ringbuffer")
worker.init()

# Register the ChipCallable to obtain an opaque callable handle.
handle = worker.register(chip_callable)

# Execute the registered callable on device. A run always takes the whole
# device; orchestration reads the resulting width via rt_available_cluster_count().
worker.run(handle, orch_args)

# Cleanup
worker.close()
```

Reach for the high-level `Worker` first. Public registration returns a
`CallableHandle`; integer callable slots are backend internals.

## Configuration

### Graph capacity and runtime limits

For `host_build_graph`, the graph task table defaults to
`CHIP_DEFAULT_GRAPH_TASKS=16384` in `src/common/host_build_graph/runtime_types.h`.
Override it per run through `CallConfig.runtime_env.ring_task_window[0]`.
In Python, assign the whole property because its getter returns a list copy:

```python
cfg = CallConfig()
cfg.runtime_env.ring_task_window = [32768, 0, 0, 0]
worker.run(handle, orch_args, cfg)
```

HBG sizes and commits its graph heap after orchestration; `ring_heap` and
`ring_dep_pool` are TRB settings, not HBG capacity controls.

The HBG orchestration-entry limit is `RUNTIME_MAX_ARGS=128` in
`src/common/host_build_graph/runtime.h`. Per-task fanin is capped at
`CHIP_MAX_FANIN=128`; there is no `RUNTIME_MAX_FANOUT` knob. See
[capacity errors](troubleshooting/device-error-codes/capacity.md) for the
resource-specific limits and diagnostics.

### Runtime Configuration

Scene tests select their runtime with `@scene_test` and keep optional per-case
overrides under `CASES[*]["config"]`:

```python
@scene_test(level=2, runtime="host_build_graph")
class TestVectorExample(SceneTestCase):
    CASES = [
        {"name": "default", "platforms": ["a2a3sim", "a2a3"], "params": {}},
    ]
```

Omitting `aicpu_thread_num` selects the architecture default. Set it only when
the workload intentionally requires a particular AICPU thread topology.
Programmatic users select the runtime when constructing `Worker` and may pass a
`CallConfig` to `worker.run()` for the same per-run overrides.

Device selection is done via CLI flag:

```bash
python tests/st/a2a3/host_build_graph/vector_example/test_vector_example.py -p a2a3 --device 0
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

- `LOG_DEBUG`: Debug messages
- `LOG_INFO`: Lifecycle and summary messages
- `LOG_TIMING`: Stable timing markers such as `[STRACE]`
- `LOG_WARN`: Warnings
- `LOG_ERROR`: Error messages

Threshold is configured from Python via the `simpler` logger:
`logging.getLogger("simpler").setLevel(simpler.TIMING)`.
