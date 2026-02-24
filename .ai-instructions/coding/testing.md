# Testing Guide

## Test Types

1. **Python unit tests** (`tests/test_runtime_builder.py`): Standard pytest tests for the Python compilation pipeline. Run with `pytest tests -v`.
2. **Simulation examples** (`examples/*/`): Full end-to-end tests running on `a2a3sim`. No hardware required, works on Linux and macOS.
3. **Device tests** (`tests/device_tests/*/`): Hardware-only tests running on real Ascend devices via `a2a3`. Requires CANN toolkit.

## Running Tests

```bash
# Python unit tests
pytest tests -v

# All simulation tests
./ci.sh -p a2a3sim

# All hardware tests (specify device range)
./ci.sh -p a2a3 -d 4-7 --parallel

# Single example
python examples/scripts/run_example.py \
    -k examples/host_build_graph/vector_example/kernels \
    -g examples/host_build_graph/vector_example/golden.py \
    -p a2a3sim
```

## Adding a New Example or Device Test

1. Create a directory under the appropriate runtime:
   - Examples: `examples/<runtime>/<name>/`
   - Device tests: `tests/device_tests/<runtime>/<name>/`
2. Add `golden.py` implementing `generate_inputs(params)` and `compute_golden(tensors, params)`
3. Add `kernels/kernel_config.py` with `KERNELS` list, `ORCHESTRATION` dict, and `RUNTIME_CONFIG`
4. Add kernel source files under `kernels/aic/`, `kernels/aiv/`, and/or `kernels/orchestration/`
5. The CI script (`ci.sh`) auto-discovers examples and device tests -- no registration needed

## Golden Test Pattern

### `golden.py` required functions

- **`generate_inputs(params)`** -- Returns a dict of torch tensors (inputs + zero-initialized outputs)
- **`compute_golden(tensors, params)`** -- Computes expected outputs in-place by writing to output tensors

### Declaring outputs

Output tensors are identified by one of:
- `__outputs__` list: e.g., `__outputs__ = ["f"]`
- `out_` prefix convention: any tensor named `out_*` is treated as output

### Optional configuration

- `TENSOR_ORDER`: List of tensor names defining the argument order for the orchestration function
- `RTOL` / `ATOL`: Comparison tolerances (default: `1e-5`)
- `PARAMS_LIST`: List of parameter dicts for parameterized tests

### `kernel_config.py` structure

```python
ORCHESTRATION = {
    "source": "path/to/orchestration.cpp",
    "function_name": "build_example_graph",
}

KERNELS = [
    {"func_id": 0, "source": "path/to/kernel.cpp", "core_type": "aiv"},
    {"func_id": 1, "source": "path/to/kernel2.cpp", "core_type": "aic"},
]

RUNTIME_CONFIG = {
    "runtime": "host_build_graph",  # or "aicpu_build_graph", "tensormap_and_ringbuffer"
    "aicpu_thread_num": 3,
    "block_dim": 3,
}
```
