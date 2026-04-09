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

# Single example (pre-built runtime binaries)
python examples/scripts/run_example.py \
    -k examples/a2a3/host_build_graph/vector_example/kernels \
    -g examples/a2a3/host_build_graph/vector_example/golden.py \
    -p a2a3sim

# Single example (recompile runtime from source)
python examples/scripts/run_example.py --build \
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
from setup import SceneTestCase, scene_test
from task_interface import ArgDirection as D

@scene_test(level=2, platforms=["a2a3sim", "a2a3"], runtime="tensormap_and_ringbuffer")
class TestMyKernel(SceneTestCase):
    ORCHESTRATION = {
        "source": "kernels/orchestration/my_orch.cpp",
        "function_name": "aicpu_orchestration_entry",
        "signature": [D.IN, D.OUT],
    }
    KERNELS = [{"func_id": 0, "source": "kernels/aiv/my_kernel.cpp", "core_type": "aiv"}]
    RUNTIME_CONFIG = {"aicpu_thread_num": 4, "block_dim": 3}
    __outputs__ = ["y"]

    def generate_inputs(self, params):
        return [("x", torch.ones(1024)), ("y", torch.zeros(1024))]

    def compute_golden(self, tensors, params):
        tensors["y"][:] = tensors["x"] + 1

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
- `platforms`: which platforms this test supports (sim names end in "sim")
- `runtime`: which runtime to use
- `ORCHESTRATION.source` / `KERNELS[].source`: paths relative to the test file

### New Scene Test (Legacy)

The golden.py + kernel_config.py directory format is still supported via `ci.py`:

Create a directory under `tests/st/{arch}/{runtime}/my_test/` with:

- `golden.py` — Input generation and golden output computation
- `kernels/kernel_config.py` — Kernel and runtime configuration

The test will be automatically picked up by `ci.py`. New tests should prefer the `@scene_test` format above.

## CI Pipeline

See [ci.md](ci.md) for the full CI pipeline documentation, including the job matrix, runner constraints, marker scheme, and `ci.sh` internals.

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
