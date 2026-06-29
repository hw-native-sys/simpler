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
git clone --branch main git@github.com:hw-native-sys/pto-isa.git build/pto-isa

# Or use HTTPS
git clone --branch main https://github.com/hw-native-sys/pto-isa.git build/pto-isa

# Set environment variable (optional - auto-detected if in standard location)
export PTO_ISA_ROOT=$(pwd)/build/pto-isa
```

**Using a Different Location:**

```bash
export PTO_ISA_ROOT=/path/to/your/pto-isa
```

`PTO_ISA_ROOT` should point to a full pto-isa git checkout, not a headers-only copy.
Install-time a2a3 onboard runtime builds record the PTO-ISA git commit so later
runtime compatibility checks can detect mismatched ISA revisions.

**Revision selection and compatibility checks:**

PTO-ISA is resolved in two separate phases:

- **Install/build time:** `pip install` may build the a2a3 onboard host runtime.
  That runtime embeds PTO-ISA SDMA headers, so the PTO-ISA checkout used during
  installation becomes part of the installed runtime binary.
- **Test/run time:** pytest and the scene-test runner resolve PTO-ISA again for
  kernel compilation. A user may choose a different PTO-ISA checkout or commit
  for a particular test run.

Because those two phases happen independently, the selected PTO-ISA revisions
can drift. `simpler` records the actual build-time git commit for a2a3 onboard
runtimes and checks it later against the run-time commit. The check does not
change how PTO-ISA is selected; it only compares the concrete revisions that
the existing resolution logic selected.

The checkout selection step decides which PTO-ISA directory `simpler` will use.
It follows this order:

1. If `PTO_ISA_ROOT` points to an existing directory, that directory wins.
   `simpler` treats it as user-managed and uses it as-is.
2. If the managed checkout is being used and `--pto-isa-commit` or the CMake
   `SIMPLER_PTO_ISA_COMMIT` define supplies a value, that value wins.
3. If no CLI/CMake value is supplied and `SIMPLER_PTO_ISA_COMMIT` is set in
   the environment, that environment value wins.
4. If no explicit value is requested, `simpler` checks out the commit recorded
   in `pto_isa.pin`.
5. If `pto_isa.pin` is missing, `simpler` warns and falls back to the latest
   `origin/HEAD` behavior for downstream checkout compatibility.

For explicit values in steps 2 and 3, a concrete SHA/tag/ref checks out that
revision. The values `latest`, `head`, and `none` are explicit opt-outs from
the pin and use `origin/HEAD`. For the CMake cache variable, the empty string is
the default value and means "use `pto_isa.pin`"; use `latest`, `head`, or `none`
for an explicit CMake opt-out.

When the managed checkout is selected, `simpler` uses `build/pto-isa`, cloning
it on first use if necessary.

This is only the checkout selection order. The compatibility check later uses a
separate commit lookup order so it can compare the installed runtime's recorded
build commit with the concrete commit selected for the current run.

One important consequence: `PTO_ISA_ROOT` is a path, not a commit selector. If
you export `PTO_ISA_ROOT=/path/to/pto-isa`, `simpler` will not automatically
checkout `--pto-isa-commit "$PTO_ISA_COMMIT"` inside that directory. To run
with a specific commit while using a custom `PTO_ISA_ROOT`, checkout that
commit in the PTO-ISA repository yourself before running `simpler`.

At install time, pass a concrete PTO-ISA commit through CMake when building
a2a3 onboard runtimes:

```bash
export PTO_ISA_COMMIT=0123456789abcdef0123456789abcdef01234567
pip install --no-build-isolation \
  --config-settings=cmake.define.SIMPLER_PTO_ISA_COMMIT="$PTO_ISA_COMMIT" \
  -e .
```

At test/run time, pass the matching commit through pytest or the scene-test
runner:

```bash
export PTO_ISA_COMMIT=0123456789abcdef0123456789abcdef01234567
pytest examples --platform a2a3 --pto-isa-commit "$PTO_ISA_COMMIT"
python path/to/scene_test.py --platform a2a3 --pto-isa-commit "$PTO_ISA_COMMIT"
```

`SIMPLER_PTO_ISA_COMMIT="$PTO_ISA_COMMIT"` can also provide the concrete commit
when the CLI option is omitted. Leaving both unset uses `pto_isa.pin`, so a
plain local install and test run match the repository pin. To explicitly track
the remote default branch instead, pass `--pto-isa-commit latest` or set
`SIMPLER_PTO_ISA_COMMIT=latest`.

For a2a3 onboard runtimes, `pip install` records the actual PTO-ISA git HEAD
used to build `host_runtime.so` in `build/lib/pto_isa_build.json`. The recorded
value comes from `git rev-parse HEAD` in the resolved PTO-ISA checkout; it is
not inferred from the requested option. If the install-time checkout is not a
real git checkout, `simpler` cannot record a trustworthy build revision for
this compatibility check.

Later, when an a2a3 onboard runtime binary is looked up, compatibility
validation uses a different order to identify the run-time PTO-ISA commit for
comparison:

1. `SIMPLER_RUN_PTO_ISA_COMMIT`, which `simpler` sets internally after the
   test/run-time checkout selection step records the selected checkout's git
   HEAD.
2. `PTO_ISA_ROOT`'s current git HEAD, when `PTO_ISA_ROOT` points to a git
   checkout and the recorded run-time commit is unavailable.
3. The resolved concrete request from `SIMPLER_PTO_ISA_COMMIT` or `pto_isa.pin`,
   as a fallback for cases where no git checkout commit can be read. Explicit
   `latest`/`head`/`none` without a readable git checkout remains unverifiable.

If the build commit and run-time commit differ, `simpler` fails early on the
host side before loading the runtime binary. The error reports both commits and
points to `build/lib/pto_isa_build.json`, which you can inspect to see the
PTO-ISA commit recorded at install time:

```bash
cat build/lib/pto_isa_build.json
```

This makes the configuration problem visible before kernel/runtime ABI
mismatches can turn into less direct failures.

The compatibility check is scoped to a2a3 onboard runtime lookup. a5, a2a3sim,
and a5sim are not blocked by this a2a3 onboard metadata. Existing installs that
do not have `build/lib/pto_isa_build.json` also continue to work; they simply
skip this compatibility check.

**Troubleshooting:**

- If git is not available: Clone pto-isa manually and set `PTO_ISA_ROOT`
- If clone fails due to network: Try again or clone manually
- If SSH clone fails (e.g., in CI): Use `--clone-protocol https` or clone manually with HTTPS
- If a2a3 onboard runtime lookup reports a PTO-ISA mismatch: Reinstall
  `simpler` with the same PTO-ISA commit used for the run, or rerun with the
  commit recorded in `build/lib/pto_isa_build.json`

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
