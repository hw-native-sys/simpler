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

- **`generate_inputs(params)`** -- Returns a flat argument list (see below) or a dict of torch tensors (legacy)
- **`compute_golden(tensors, params)`** -- Computes expected outputs in-place by writing to output tensors

### `generate_inputs` return format

Returns a `list` of `(name, value)` pairs where value is either:
- **`torch.Tensor` / numpy array**: A tensor argument. The framework handles `device_malloc`, `copy_to_device`, and copy-back based on `__outputs__`.
- **ctypes scalar** (`ctypes.c_int64`, `ctypes.c_float`, etc.): A scalar value passed directly to the orchestration function. Integer types are zero-extended to uint64; `c_float` is bit-cast to uint32 then zero-extended; `c_double` is bit-cast to uint64.

The list order defines the argument order in the orchestration's `uint64_t* args` array. All named items (tensors and scalars) are collected into the dict passed to `compute_golden`, so `compute_golden` can reference any argument by name.

Example:
```python
import ctypes
import torch

def generate_inputs(params: dict) -> list:
    a = torch.full((16384,), 2.0, dtype=torch.float32)
    b = torch.full((16384,), 3.0, dtype=torch.float32)
    f = torch.zeros(16384, dtype=torch.float32)

    return [
        ("a",      a),                           # args[0]: tensor pointer
        ("b",      b),                           # args[1]: tensor pointer
        ("f",      f),                           # args[2]: tensor pointer (output)
        ("size_a", ctypes.c_int64(a.nbytes)),    # args[3]: scalar
        ("size_b", ctypes.c_int64(b.nbytes)),    # args[4]: scalar
        ("size_f", ctypes.c_int64(f.nbytes)),    # args[5]: scalar
        ("SIZE",   ctypes.c_int64(a.numel())),   # args[6]: scalar
    ]
```

### Declaring outputs

Output tensors are identified by one of:
- `__outputs__` list: e.g., `__outputs__ = ["f"]`
- `out_` prefix convention: any tensor named `out_*` is treated as output

### Optional configuration

- `RTOL` / `ATOL`: Comparison tolerances (default: `1e-5`)
- `ALL_CASES`: Dict of named parameter sets for parameterized tests
- `DEFAULT_CASE`: Name of the default case to run

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
