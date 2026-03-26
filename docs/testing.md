# Testing

## Quick Reference

```bash
# Simulation tests (no hardware required)
./ci.sh -p a2a3sim

# Hardware tests (requires Ascend device)
./ci.sh -p a2a3 -d 4-7 --parallel

# Python unit tests
pytest tests -v

# C++ unit tests
cmake -B tests/cpp/build -S tests/cpp && cmake --build tests/cpp/build && ctest --test-dir tests/cpp/build --output-on-failure

# Run a single example
python examples/scripts/run_example.py \
    -k examples/a2a3/host_build_graph/vector_example/kernels \
    -g examples/a2a3/host_build_graph/vector_example/golden.py \
    -p a2a3sim
```

## CI Pipeline (`ci.sh`)

The `ci.sh` script runs the full test suite:

1. **C++ unit tests** — GoogleTest-based tests for shared runtime components (`tests/cpp/`)
2. **Python unit tests** — `pytest tests -v`
3. **Simulation/device tests** — Runs all examples and device tests for the specified platform

### Simulation Mode

```bash
./ci.sh -p a2a3sim
```

Runs all tests using thread-based simulation. No Ascend hardware needed.

### Hardware Mode

```bash
./ci.sh -p a2a3 -d 4-7 --parallel
```

- `-d 4-7`: Use devices 4 through 7
- `--parallel`: Run tests in parallel across devices

## Test Types

### C++ Unit Tests (`tests/cpp/`)

GoogleTest-based tests for shared components (`src/common/task_interface/` and `src/{arch}/runtime/common/`):

- `test_data_type.cpp` — DataType enum, get_element_size(), get_dtype_name()
- `test_task_arg.cpp` — TaskArg packing/unpacking, byte alignment, nbytes(), DMA copy semantics

Build and run independently:
```bash
cmake -B tests/cpp/build -S tests/cpp
cmake --build tests/cpp/build
ctest --test-dir tests/cpp/build --output-on-failure
```

### Python Unit Tests (`tests/`)

Tests for the Python build pipeline:
- `test_runtime_builder.py` — RuntimeBuilder configuration and build flow

```bash
pytest tests -v
```

### Examples (`examples/{arch}/`)

Working examples organized by runtime:
- `host_build_graph/` — HBG examples
- `aicpu_build_graph/` — ABG examples
- `tensormap_and_ringbuffer/` — TMR examples

Each example has a `golden.py` with `generate_inputs()` and `compute_golden()` for result validation.

### Device Tests (`tests/device_tests/{arch}/`)

Integration tests that run on simulation or hardware, organized by runtime. Same structure as examples but focused on testing specific runtime behaviors and edge cases.

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

### New Device Test

Create a directory under `tests/device_tests/{arch}/{runtime}/my_test/` with:
- `golden.py` — Input generation and golden output computation
- `kernels/kernel_config.py` — Kernel and runtime configuration

The test will be automatically picked up by `ci.sh`.
