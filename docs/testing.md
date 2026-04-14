# Testing

## Design Principles

Three axioms govern hardware classification across all test categories:

1. **sim = no hardware**: `a2a3sim`, `a5sim`, and no `--platform` are all equivalent — no hardware required.
2. **a2a3 / a5 are distinct platforms**: Tests may support one, both, or neither.
3. **`requires_hardware` has two levels**:
   - `requires_hardware` (no argument) — needs *any* hardware (a2a3 or a5)
   - `requires_hardware("a2a3")` — needs *specifically* a2a3

These principles apply uniformly to ut-py (pytest markers), ut-cpp (ctest labels), and st (`@scene_test(platforms=[...])`).

## Quick Reference

```bash
# Python unit tests, no hardware (sim or github-hosted)
pytest tests/ut

# Python unit tests, a2a3 hardware
pytest tests/ut --platform a2a3

# C++ unit tests, no hardware
cmake -B tests/ut/cpp/build -S tests/ut/cpp && cmake --build tests/ut/cpp/build
ctest --test-dir tests/ut/cpp/build -LE requires_hardware --output-on-failure

# C++ unit tests, a2a3 hardware (only hw + a2a3-specific tests)
ctest --test-dir tests/ut/cpp/build -L "^requires_hardware(_a2a3)?$" --output-on-failure

# Scene tests (pytest, @scene_test classes)
pytest examples tests/st                          # all sim platforms (auto-parametrized)
pytest examples tests/st --platform a2a3sim       # specific sim
pytest examples tests/st --platform a2a3          # hardware

# Scene tests (legacy ci.py, golden.py directory scanning)
python ci.py -p a2a3sim
python ci.py -p a2a3 -d 4-7

# Single scene test (standalone)
python examples/a2a3/tensormap_and_ringbuffer/vector_example/test_vector_example.py -p a2a3sim

# Standalone with build-from-source
python examples/a2a3/tensormap_and_ringbuffer/vector_example/test_vector_example.py -p a2a3sim --build

# Benchmark mode (100 rounds, skip golden comparison)
python examples/a2a3/tensormap_and_ringbuffer/vector_example/test_vector_example.py \
    -p a2a3 -d 0 -n 100 --skip-golden

# Profiling (first round only)
python examples/a2a3/tensormap_and_ringbuffer/vector_example/test_vector_example.py \
    -p a2a3 --enable-profiling

# Tensor dump
python tests/st/a2a3/tensormap_and_ringbuffer/alternating_matmul_add/test_alternating_matmul_add.py \
    -p a2a3 -d 11 --dump-tensor

# Single example via run_example.py (deprecated — prefer test_*.py standalone)
python examples/scripts/run_example.py \
    -k examples/a2a3/host_build_graph/vector_example/kernels \
    -g examples/a2a3/host_build_graph/vector_example/golden.py \
    -p a2a3sim
```

## Test Organization

Three test categories:

| Category | Abbrev | Location | Runner | Description |
| -------- | ------ | -------- | ------ | ----------- |
| System tests | st | `examples/`, `tests/st/` | pytest + `ci.py` (legacy) | Full end-to-end cases (compile + run + validate) |
| Python unit tests | ut-py | `tests/ut/` | pytest | Unit tests for nanobind-exposed and Python modules |
| C++ unit tests | ut-cpp | `tests/ut/cpp/` | ctest (GoogleTest) | Unit tests for pure C++ modules |

**ST migration**: Scene tests are migrating from `ci.py` (golden.py directory scanning) to pytest (`@scene_test` class decorator). New tests should use `@scene_test`. Existing golden.py-based tests continue to work via `ci.py` during the transition.

### Choosing ut-py vs ut-cpp

If a module is exposed via nanobind (used by both C++ and Python), test in **ut-py** (`tests/ut/`).
If a module is pure C++ with no Python binding, test in **ut-cpp** (`tests/ut/cpp/`).

## Scene Test CLI Options

Scene tests support advanced CLI options for benchmarking, profiling, and runtime control. These work identically in both pytest and standalone mode.

### pytest

```bash
pytest --platform a2a3sim                                        # default: 1 round + golden
pytest --platform a2a3 --rounds 100 --skip-golden                # benchmark mode
pytest --platform a2a3 --enable-profiling                        # profiling (first round)
pytest --platform a2a3sim --build                                # compile runtime from source
pytest --platform a2a3sim --log-level debug                        # verbose C++ logging
```

### Standalone (test_*.py)

```bash
python test_xxx.py -p a2a3sim                                    # default: 1 round + golden
python test_xxx.py -p a2a3 -d 0 -n 100 --skip-golden            # benchmark mode
python test_xxx.py -p a2a3 --enable-profiling                    # profiling (first round)
python test_xxx.py -p a2a3 --dump-tensor                         # dump per-task tensor I/O
python test_xxx.py -p a2a3sim --build                            # compile runtime from source
python test_xxx.py -p a2a3sim --log-level debug                  # verbose C++ logging
```

### Option Reference

| Option | Short | Default | Description |
| ------ | ----- | ------- | ----------- |
| `--rounds N` | `-n` | 1 | Run each case N times |
| `--skip-golden` | | false | Skip golden comparison (for benchmarking) |
| `--enable-profiling` | | false | Enable profiling on first round only |
| `--dump-tensor` | | false | Dump per-task tensor I/O during runtime execution |
| `--build` | | false | Compile runtime from source (not pre-built) |
| `--log-level LEVEL` | | (none) | Set `PTO_LOG_LEVEL` env var (`error`/`warn`/`info`/`debug`) |

Profiling is enabled only on the first round to avoid overhead on subsequent iterations. Output tensors are reset to their initial values between rounds.

## Hardware Classification

The `--platform` flag is the single source of truth for hardware availability. No separate `-m` flags are needed.

### Helper function

```python
def is_device(platform: str | None) -> bool:
    """sim and None are both no-hardware."""
    return platform is not None and not platform.endswith("sim")
```

### ut-py: `@pytest.mark.requires_hardware[(platform)]`

| Declaration | no-hw runner | a2a3 runner | a5 runner |
| ----------- | ------------ | ----------- | --------- |
| *(no marker)* | run | skip | skip |
| `@pytest.mark.requires_hardware` | skip | run | run |
| `@pytest.mark.requires_hardware("a2a3")` | skip | run | skip |
| `@pytest.mark.requires_hardware("a5")` | skip | skip | run |

Skip logic (conftest.py):

```python
marker = item.get_closest_marker("requires_hardware")
on_device = is_device(platform)
if marker is None:
    if on_device:
        skip("no-hardware test, runs in no-hw job")
elif marker.args:
    if platform != marker.args[0]:
        skip(f"requires --platform {marker.args[0]}")
else:
    if not on_device:
        skip("requires hardware")
```

### ut-cpp: ctest labels

| Declaration (CMakeLists.txt) | no-hw runner | a2a3 runner | a5 runner |
| ---------------------------- | ------------ | ----------- | --------- |
| *(no label)* | run | skip | skip |
| `LABELS "requires_hardware"` | skip | run | run |
| `LABELS "requires_hardware_a2a3"` | skip | run | skip |
| `LABELS "requires_hardware_a5"` | skip | skip | run |

Selection:

| Runner | Command |
| ------ | ------- |
| No hardware | `ctest -LE requires_hardware` |
| a2a3 | `ctest -L "^requires_hardware(_a2a3)?$"` |
| a5 | `ctest -L "^requires_hardware(_a5)?$"` |

`-LE` (exclude regex) on no-hw runner: `requires_hardware` matches all three label variants, so only unlabeled tests run.
`-L` (include regex) on device runners: only labeled tests run, unlabeled ones are excluded.

### st: `@scene_test(platforms=[...])`

`platforms` lists all supported platform names (both sim and device).

| `--platform` | Behavior |
| ------------ | -------- |
| `a2a3sim` | Run if `"a2a3sim"` in `platforms`, else skip |
| `a2a3` | Run if `"a2a3"` in `platforms`, else skip |
| *(none)* | Auto-parametrize over all `*sim` entries in `platforms`. Skip if no sim platform declared |

Auto-parametrization logic (conftest.py `pytest_generate_tests`):

```python
def pytest_generate_tests(metafunc):
    cls = metafunc.cls
    if not (cls and hasattr(cls, "_scene_platforms")):
        return
    platform = metafunc.config.getoption("--platform")
    if platform is None:
        sims = [p for p in cls._scene_platforms if p.endswith("sim")]
        if sims:
            metafunc.parametrize("st_platform", sims, indirect=True)
```

Examples:

```python
@scene_test(level=2, platforms=["a2a3sim", "a2a3", "a5sim", "a5"], ...)
class TestFoo(SceneTestCase): ...
# --platform a2a3sim  → TestFoo[a2a3sim]
# --platform a2a3     → TestFoo[a2a3]
# (none)              → TestFoo[a2a3sim] + TestFoo[a5sim]

@scene_test(level=2, platforms=["a2a3"], ...)
class TestHwOnly(SceneTestCase): ...
# --platform a2a3     → TestHwOnly[a2a3]
# (none)              → skip (no sim in platforms)
```

No separate `st` or `requires_hardware` marker — `platforms` is the sole declaration.

## Test Directory Structure

```text
tests/
  conftest.py          # pytest configuration (markers, fixtures, parametrization)
  ut/                  # Python unit tests (ut-py)
    test_task_interface.py
    test_runtime_builder.py
    cpp/               # C++ unit tests (ut-cpp, GoogleTest)
  st/                  # Scene tests
    setup/             # Compilation toolchain (KernelCompiler, RuntimeBuilder, etc.)
    conftest.py        # ST-specific sys.path setup
    test_worker_api.py # L3 distributed worker tests
    a2a3/              # Legacy golden.py-based tests (ci.py)
    a5/                # Legacy golden.py-based tests (ci.py)

examples/              # Small examples (sim + onboard)
  a2a3/
    tensormap_and_ringbuffer/
      vector_example/
        test_vector_example.py   # @scene_test — new style
        golden.py                # legacy (ci.py)
        kernels/kernel_config.py # legacy (ci.py)
  a5/...

conftest.py            # Root: --platform/--device options, ST fixtures
```

## Test Types

### C++ Unit Tests (`tests/ut/cpp/`)

GoogleTest-based tests for shared components (`src/common/task_interface/` and `src/{arch}/runtime/common/`):

- `test_data_type.cpp` — DataType enum, get_element_size(), get_dtype_name()

```bash
cmake -B tests/ut/cpp/build -S tests/ut/cpp
cmake --build tests/ut/cpp/build
ctest --test-dir tests/ut/cpp/build --output-on-failure
```

### Python Unit Tests (`tests/ut/`)

Tests for the nanobind extension and the Python build pipeline:

- `test_task_interface.py` — DataType, ContinuousTensor, ChipStorageTaskArgs, torch integration
- `test_runtime_builder.py` — RuntimeBuilder discovery, error handling, build logic (mocked), and real compilation integration tests

```bash
# No-hardware runner (hw tests auto-skip, no-hw tests run)
pytest tests/ut

# a2a3 hardware runner (no-hw tests skip, hw + a2a3-specific tests run)
pytest tests/ut --platform a2a3
```

### Examples (`examples/{arch}/`)

Small, fast examples that run on both simulation and real hardware. Organized by runtime:

- `host_build_graph/` — HBG examples
- `aicpu_build_graph/` — ABG examples
- `tensormap_and_ringbuffer/` — TMR examples

Each example has a `golden.py` with `generate_inputs()` and `compute_golden()` for result validation.

### Device Scene Tests (`tests/st/{arch}/`)

Hardware-only scene tests for large-scale and feature-rich scenarios that are too slow or unsupported on simulation. Organized by runtime. Same structure as examples but focused on testing specific runtime behaviors and edge cases.

| Attribute | `examples/` | `tests/st/` |
| --------- | ----------- | ----------- |
| Runs on sim | Yes | No |
| Runs on device | Yes | Yes |
| Scale | Small, fast | Large, thorough |
| Purpose | Examples + basic regression | Deep functionality/performance |

## Writing New Tests

### New C++ Unit Test

Add a new test file to `tests/ut/cpp/` and register it in `tests/ut/cpp/CMakeLists.txt`:

```cmake
add_executable(test_my_component
    test_my_component.cpp
    test_stubs.cpp
)
target_include_directories(test_my_component PRIVATE ${COMMON_DIR} ${TMR_RUNTIME_DIR} ${PLATFORM_INCLUDE_DIR})
target_link_libraries(test_my_component gtest_main)
add_test(NAME test_my_component COMMAND test_my_component)

# If hardware required:
# set_tests_properties(test_my_component PROPERTIES LABELS "requires_hardware")
# If specific platform required:
# set_tests_properties(test_my_component PROPERTIES LABELS "requires_hardware_a2a3")
```

### New Scene Test

Create a `test_*.py` file using the `@scene_test` decorator:

```python
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestMyKernel(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/my_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.OUT],
        },
        "incores": [
            {"func_id": 0, "source": "kernels/aiv/my_kernel.cpp", "core_type": "aiv"},
        ],
    }

    CASES = [
        {
            "name": "default",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 3},
            "params": {},
        },
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            Tensor("x", torch.ones(1024, dtype=torch.float32)),
            Tensor("y", torch.zeros(1024, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        args.y[:] = args.x + 1

if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
```

Run it:

```bash
# Via pytest (batch, ChipWorker reuse across tests)
pytest examples tests/st --platform a2a3sim

# Standalone (single case)
python test_my_kernel.py -p a2a3sim

# On hardware
pytest examples tests/st --platform a2a3
```

Key fields:

- `level`: 2 = single ChipWorker, 3 = distributed Worker (future)
- `CASES[].platforms`: which platforms each case supports (sim names end in "sim")
- `runtime`: which runtime to use
- `CALLABLE.orchestration.source` / `CALLABLE.incores[].source`: paths relative to the test file

### New Scene Test (Legacy)

The golden.py + kernel_config.py directory format is still supported via `ci.py`:

Create a directory under `tests/st/{arch}/{runtime}/my_test/` with:

- `golden.py` — Input generation and golden output computation
- `kernels/kernel_config.py` — Kernel and runtime configuration

The test will be automatically picked up by `ci.py`. New tests should prefer the `@scene_test` format above.

### Migrating Legacy Tests to @scene_test

#### Field Mapping

| Legacy file | Legacy field | SceneTestCase equivalent |
| ----------- | ------------ | ------------------------ |
| `kernel_config.py` | `ORCHESTRATION` | `CALLABLE["orchestration"]` (source becomes relative path) |
| `kernel_config.py` | `KERNELS` | `CALLABLE["incores"]` |
| `kernel_config.py` | `RUNTIME_CONFIG` | `CASES[*]["config"]` (`aicpu_thread_num`, `block_dim`) |
| `golden.py` | `generate_inputs()` | `generate_args(self, params)` returning `TaskArgsBuilder(...)` |
| `golden.py` | `compute_golden()` | `compute_golden(self, args, params)` using `args.name[:]` |
| `golden.py` | `__outputs__` | `D.OUT` / `D.INOUT` in `CALLABLE["orchestration"]["signature"]` |
| `golden.py` | `ALL_CASES` | `CASES` list with per-case `platforms`, `config`, `params` |
| `golden.py` | `RTOL` / `ATOL` | Class attributes `RTOL` / `ATOL` |

#### Key Conversions

**generate_inputs → generate_args:**

```python
# Legacy (golden.py)
def generate_inputs(params):
    a = torch.full((SIZE,), 2.0, dtype=torch.float32)
    return [("a", a), ("b", b), ("f", f)]

# New (test_*.py)
def generate_args(self, params):
    a = torch.full((SIZE,), 2.0, dtype=torch.float32)
    return TaskArgsBuilder(Tensor("a", a), Tensor("b", b), Tensor("f", f))
```

**compute_golden:**

```python
# Legacy
def compute_golden(tensors, params):
    tensors["f"][:] = torch.as_tensor(tensors["a"]) + 1

# New
def compute_golden(self, args, params):
    args.f[:] = args.a + 1
```

**Source paths:** Legacy uses absolute paths (`str(_KERNELS_ROOT / "aiv" / "kernel.cpp")`). SceneTestCase uses paths relative to the test file (`"kernels/aiv/kernel.cpp"`).

**Import:** `from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test` (not `from setup`). For scalar arguments add `Scalar` and `import ctypes`. Tensors must appear before Scalars in `TaskArgsBuilder`.

#### Running: Before and After

| Action | Legacy | SceneTestCase |
| ------ | ------ | ------------- |
| Single run (sim) | `python examples/scripts/run_example.py -k kernels/ -g golden.py -p a2a3sim` | `python test_*.py -p a2a3sim` |
| Single run (hardware) | `python examples/scripts/run_example.py -k kernels/ -g golden.py -p a2a3 -d 0` | `python test_*.py -p a2a3 -d 0` |
| Batch (pytest) | `python examples/scripts/ci.py` | `pytest examples tests/st --platform a2a3sim` |
| Multi-round | Not supported | `python test_*.py -p a2a3sim -n 3` |
| Benchmark | Manual | `python test_*.py -p a2a3 -d 0 -n 100 --skip-golden --case CaseName` |
| Profiling | `--enable-profiling` flag on run_example.py | `python test_*.py -p a2a3 -d 0 --enable-profiling` |

#### After Migration

Delete `golden.py` and `kernels/kernel_config.py`. Keep `kernels/` C++ sources — `CALLABLE` references them. Once legacy files are removed, the test is no longer discovered by `ci.py` and runs exclusively via pytest or standalone.

If the same example exists in both `examples/` and `tests/st/`, merge into one `test_*.py`: small cases get `platforms: ["a2a3sim", "a2a3"]`, large benchmark cases get `platforms: ["a2a3"], "manual": True`.

## CI Pipeline

See [ci.md](ci.md) for the full CI pipeline documentation, including the job matrix, runner constraints, marker scheme, and `ci.sh` internals.

## Runtime Isolation Constraint (Onboard)

**One device can only run one runtime per process.** Switching runtimes on the same device within a single process causes AICPU kernel hangs.

### Root Cause

CANN's AICPU dispatch uses a framework SO (`libaicpu_extend_kernels.so`) with a global singleton `BackendServerHandleManager` that:

1. **`SaveSoFile`**: Writes the user AICPU .so to disk on first call, then sets `firstCreatSo_ = true` to skip all subsequent writes.
2. **`SetTileFwkKernelMap`**: `dlopen`s the .so and caches function pointers on first call, then sets `firstLoadSo_ = true` to skip all subsequent loads.

When a second runtime launches on the same device (same CANN process context), the Init kernel call hits the cached flags — the new AICPU .so is never written or loaded. The Exec kernel then calls function pointers from the first runtime's .so, which operates on incompatible data structures and hangs.

### Impact

| Scenario | Result |
| -------- | ------ |
| Same runtime, same device, sequential | Works (same .so, cached pointers valid) |
| Different runtime, same device, sequential | **Hangs** (stale .so, wrong function pointers) |
| Different runtime, different device | Works (separate CANN context per device) |
| Different runtime, different process, same device | Works (`rtDeviceReset` between processes clears context) |

### Mitigation in pytest

The `conftest.py` device allocator groups tests by runtime and assigns each runtime group to exclusive devices. See "Device Allocation Algorithm" below.

## Device Allocation Algorithm (Onboard pytest)

When running `pytest --platform a2a3 --device 8-11`, the fixture must allocate devices to tests such that:

1. **Runtime isolation**: A device used by runtime A must not be reused by runtime B in the same process.
2. **L3 multi-device**: L3 tests may need 2+ contiguous devices.
3. **Efficiency**: Devices freed by one test of the same runtime can be reused by the next.

### Algorithm

```text
Phase 1: Group tests by runtime
  tensormap_and_ringbuffer: [TestVectorExample, TestScalarData, TestL3Dependency, ...]
  aicpu_build_graph:        [TestPagedAttentionAicpuBuildGraph]
  host_build_graph:         [TestPagedAttentionHostBuildGraph]

Phase 2: Partition devices across runtime groups
  Available: [8, 9, 10, 11]
  tensormap_and_ringbuffer (6 tests, needs max 2 for L3 group): devices [8, 9]
  aicpu_build_graph (1 test, needs 1):                          devices [10]
  host_build_graph (1 test, needs 1):                           devices [11]

Phase 3: Within each group, allocate from group's device pool
  TestVectorExample:       dev 8 → run → release → dev 8 available again
  TestScalarData:          dev 8 → run → release → OK (same runtime)
  TestL3Dependency:        dev 8 → run → release
  TestL3Group:             dev [8, 9] → run → release
  TestPagedAttentionAicpuBuildGraph: dev 10 → run → release
  TestPagedAttentionHostBuildGraph:  dev 11 → run → release
```

### Implementation

The `DevicePool` in `conftest.py` is extended with runtime-aware partitioning. The `st_worker` fixture checks the test class's `_st_runtime` and allocates from the corresponding partition.

### Sim platforms

On sim (`a2a3sim`, `a5sim`), device IDs are virtual — no hardware state, no isolation constraint. All tests share a single virtual pool with auto-incrementing IDs.

## Per-Case Device Filtering

The `@scene_test(platforms=[...])` decorator provides per-case platform filtering. A single test class declares which platforms it supports:

```python
@scene_test(level=2, platforms=["a2a3sim", "a2a3"], runtime="tensormap_and_ringbuffer")
class TestSmallCase(SceneTestCase):
    ...  # runs on sim and a2a3 hardware

@scene_test(level=2, platforms=["a2a3"], runtime="tensormap_and_ringbuffer")
class TestLargeCase(SceneTestCase):
    ...  # hardware only (too slow for sim)
```

This eliminates the need for separate `examples/` (sim) and `tests/st/` (device) directories when only scale differs. Both cases can live in the same file.

### When separate directories are still needed

When kernels themselves differ (e.g., templated tile sizes tuned for device), separate test files remain the correct approach.
