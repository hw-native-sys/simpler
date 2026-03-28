# Testing

## Quick Reference

```bash
# Unit tests (no hardware required)
pytest tests -m "not requires_hardware" -v

# Simulation scene tests (no hardware required)
./ci.sh -p a2a3sim

# Hardware scene tests (requires Ascend device)
./ci.sh -p a2a3 -d 4-7 --parallel

# C++ unit tests
cmake -B tests/cpp/build -S tests/cpp && cmake --build tests/cpp/build && ctest --test-dir tests/cpp/build --output-on-failure

# Run a single example (uses pre-built runtime binaries)
python examples/scripts/run_example.py \
    -k examples/a2a3/host_build_graph/vector_example/kernels \
    -g examples/a2a3/host_build_graph/vector_example/golden.py \
    -p a2a3sim

# Run a single example with runtime recompilation (after changing runtime C++ source)
python examples/scripts/run_example.py --build \
    -k examples/a2a3/host_build_graph/vector_example/kernels \
    -g examples/a2a3/host_build_graph/vector_example/golden.py \
    -p a2a3sim
```

> **`--build` flag**: By default, `run_example.py` loads pre-built binaries from `build/lib/`. Pass `--build` to recompile runtimes incrementally from source (uses persistent cmake cache in `build/cache/`). Use this when you've modified runtime C++ source under `src/`. See [developer-guide.md](developer-guide.md#build-workflow) for the full rebuild decision table.

## Test Organization

Three test categories:

| Category | Abbrev | Location | Runner | Description |
| -------- | ------ | -------- | ------ | ----------- |
| System tests | st | `examples/`, `tests/st/` | `ci.sh` | Full end-to-end cases (compile + run + validate) |
| Python unit tests | ut-py | `tests/ut/` | pytest | Unit tests for nanobind-exposed and Python modules |
| C++ unit tests | ut-cpp | `tests/cpp/` | ctest (GoogleTest) | Unit tests for pure C++ modules |

### Choosing ut-py vs ut-cpp

If a module is exposed via nanobind (used by both C++ and Python), test in **ut-py** (`tests/ut/`).
If a module is pure C++ with no Python binding, test in **ut-cpp** (`tests/cpp/`).

## Hardware Classification

All three categories (st, ut-py, ut-cpp) need a way to specify hardware requirements. Three tiers:

| Tier | ut-py (pytest marker) | ut-cpp (ctest label) | st (current mechanism) |
| ---- | --------------------- | -------------------- | ---------------------- |
| No hardware | `-m "not requires_hardware"` | `-L no_hardware` | `examples/` on sim platform |
| Any hardware | `-m requires_hardware` (no `--platform`) | `-L requires_hardware` | — |
| Platform-specific | `-m requires_hardware --platform a2a3` | `-L requires_a2a3` | `tests/st/{arch}/` on device platform |

For st, hardware classification is currently folder-based: `examples/` runs on both sim and device, `tests/st/` runs on device only. This works but may be unified with per-case metadata in the future (see [Future: Per-Case Device Filtering](#future-per-case-device-filtering)).

## Test Directory Structure

```text
tests/
  conftest.py          # pytest configuration (markers, fixtures, parametrization)
  ut/                  # Python unit tests (ut-py)
    test_task_interface.py
    test_runtime_builder.py
  st/                  # Hardware-only scene tests (large-scale, feature-rich)
    a2a3/
      host_build_graph/...
      aicpu_build_graph/...
      tensormap_and_ringbuffer/...
    a5/...
  cpp/                 # C++ unit tests (ut-cpp, GoogleTest)

examples/              # Small examples (sim + onboard)
  a2a3/...
  a5/...
```

## Test Types

### C++ Unit Tests (`tests/cpp/`)

GoogleTest-based tests for shared components (`src/common/task_interface/` and `src/{arch}/runtime/common/`):

- `test_data_type.cpp` — DataType enum, get_element_size(), get_dtype_name()

```bash
cmake -B tests/cpp/build -S tests/cpp
cmake --build tests/cpp/build
ctest --test-dir tests/cpp/build --output-on-failure
```

### Python Unit Tests (`tests/ut/`)

Tests for the nanobind extension and the Python build pipeline:

- `test_task_interface.py` — DataType, ContinuousTensor, ChipStorageTaskArgs, torch integration
- `test_runtime_builder.py` — RuntimeBuilder discovery, error handling, build logic (mocked), and real compilation integration tests

```bash
# All non-hardware unit tests
pytest tests -m "not requires_hardware" -v

# Hardware integration tests only (requires Ascend toolchain)
pytest tests -m requires_hardware --platform a2a3 -v
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

Add a new test file to `tests/cpp/` and register it in `tests/cpp/CMakeLists.txt`:

```cmake
add_executable(test_my_component
    test_my_component.cpp
    test_stubs.cpp
)
target_include_directories(test_my_component PRIVATE ${COMMON_DIR} ${TMR_RUNTIME_DIR} ${PLATFORM_INCLUDE_DIR})
target_link_libraries(test_my_component gtest_main)
add_test(NAME test_my_component COMMAND test_my_component)
```

### New Scene Test

Create a directory under `tests/st/{arch}/{runtime}/my_test/` with:

- `golden.py` — Input generation and golden output computation
- `kernels/kernel_config.py` — Kernel and runtime configuration

The test will be automatically picked up by `ci.sh`.

## CI Pipeline

See [ci.md](ci.md) for the full CI pipeline documentation, including the job matrix, runner constraints, marker scheme, and `ci.sh` internals.

## Future: Per-Case Device Filtering

### Problem

`examples/` and `tests/st/` can duplicate entire kernel directories when only input scale differs — `examples/` carries a small sim-friendly case, and `tests/st/` carries a large device-scale case with the same kernel code.

### Proposed approach

Each `golden.py` can optionally mark specific cases as device-only via a `device_only` flag in `ALL_CASES`:

```python
ALL_CASES = {
    "SmallCase": {"batch": 1, "head_dim": 16, "dtype": "float16"},
    "LargeCase": {"batch": 256, "head_dim": 128, "dtype": "bfloat16", "device_only": True},
}
```

`code_runner.py` would filter out `device_only` cases on sim platforms. This allows a single directory to serve both sim smoke tests and device-scale tests without kernel duplication.

### When separate directories are still needed

When kernels themselves differ (e.g., templated tile sizes tuned for device), separate directories under `examples/` and `tests/st/` remain the correct approach.
