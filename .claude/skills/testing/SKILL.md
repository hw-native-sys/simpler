---
name: testing
description: Testing guide and pre-commit testing strategy for PTO Runtime. Use when running tests, adding tests, or deciding what to test before committing.
---

# Testing

## Test Types

1. **Python unit tests (ut-py)** (`tests/ut/`): Standard pytest tests for the Python compilation pipeline and nanobind bindings. Run with `pytest tests/ut`. Tests declaring `@pytest.mark.requires_hardware[("<platform>")]` auto-skip unless `--platform` points to a matching device.
2. **C++ unit tests (ut-cpp)** (`tests/ut/cpp/`): GoogleTest-based tests for pure C++ modules. Run with `cmake -B tests/ut/cpp/build -S tests/ut/cpp && cmake --build tests/ut/cpp/build && ctest --test-dir tests/ut/cpp/build -LE requires_hardware --output-on-failure`. Hardware-required tests carry a `requires_hardware` or `requires_hardware_<platform>` ctest label and are filtered via `-LE`.
3. **Simulation examples** (`examples/{arch}/*/`): Small end-to-end examples running on both sim and hardware. No hardware required for sim mode, works on Linux and macOS.
4. **Device scene tests** (`tests/st/{arch}/*/`): Scene tests for large-scale and feature-rich scenarios. Each `@scene_test` class declares its `platforms=[...]` list. Running on real Ascend devices requires the CANN toolkit.

## Running Tests

**Important**: Always read `.github/workflows/ci.yml` first to extract the current `-c` (pto-isa commit) and `-t` (timeout) flags. These ensure reproducible builds by pinning the PTO-ISA dependency to a known-good commit.

**IMPORTANT**: Never pipe `ci.sh --parallel` output through buffering commands like `| tail`, `| grep`, or `| head`. Parallel mode spawns background subprocesses whose stdout is interleaved; pipe filters buffer until the pipe closes (all children exit), producing zero visible output and appearing hung. Always capture full output directly (`2>&1` without pipes).

### Runtime rebuild decision

Before running tests, determine whether runtime binaries need recompilation:

| What changed | Rebuild needed? | How |
| ------------ | --------------- | --- |
| Runtime/platform C++ (`src/{arch}/runtime/`, `src/{arch}/platform/`) | Yes | Pass `--build` to `run_example.py`; `ci.sh` always rebuilds |
| Nanobind bindings (`python/bindings/`) | Yes | Re-run `pip install -e .` |
| Python-only code, examples, kernels | No | Just re-run the test |

`ci.sh` relies on pre-built binaries from `pip install .`, so no `--build` flag is needed for CI runs. For single `run_example.py` invocations during development, pass `--build` when runtime C++ source has changed.

```bash
# Python unit tests (no hardware)
pytest tests/ut

# Python unit tests (a2a3 hardware)
pytest tests/ut --platform a2a3

# C++ unit tests (no hardware)
cmake -B tests/ut/cpp/build -S tests/ut/cpp && cmake --build tests/ut/cpp/build
ctest --test-dir tests/ut/cpp/build -LE requires_hardware --output-on-failure

# C++ unit tests (a2a3 hardware)
ctest --test-dir tests/ut/cpp/build -L "^requires_hardware(_a2a3)?$" --output-on-failure

# All simulation tests (extract -c and -t from ci.yml)
./ci.sh -p a2a3sim -c <commit> -t <timeout>

# All hardware tests (extract -c and -t from ci.yml, auto-detect idle devices)
./ci.sh -p a2a3 -d <range> -c <commit> -t <timeout>

# Single runtime
./ci.sh -p a2a3sim -r host_build_graph -c <commit> -t <timeout>

# Single example (uses pre-built binaries)
python examples/scripts/run_example.py \
    -k examples/a2a3/host_build_graph/vector_example/kernels \
    -g examples/a2a3/host_build_graph/vector_example/golden.py \
    -p a2a3sim -c <commit>

# Single example (recompile runtime from source, use after changing runtime C++)
python examples/scripts/run_example.py --build \
    -k examples/a2a3/host_build_graph/vector_example/kernels \
    -g examples/a2a3/host_build_graph/vector_example/golden.py \
    -p a2a3sim -c <commit>
```

## Pre-Commit Testing Strategy

When changed files require testing (C++, Python, or CMake), follow these steps to decide **what** to test and **how**.

### Step 1 — Platform Availability and Detection

```bash
command -v npu-smi &>/dev/null
```

| Result | Platforms to test |
| ------ | ----------------- |
| Found | `<arch>sim` (simulation) **and** `<arch>` (hardware) |
| Not found | Simulation only (default `a2a3sim`) |

**When `npu-smi` is found**, detect the platform by parsing chip name from `npu-smi info` output:

| Chip name contains | Platform |
| ------------------ | -------- |
| `910B` or `910C` | `a2a3` (sim: `a2a3sim`) |
| `950` | `a5` (sim: `a5sim`) |

Use the detected platform for all subsequent `-p` flags. If the chip name is unrecognized, warn and default to `a2a3`.

### Step 2 — Test Scope

Run `git diff --name-only` (or `git diff --cached --name-only` for staged changes) and match the **first** applicable rule:

| Changed paths | Scope | Command pattern |
| ------------- | ----- | --------------- |
| `src/{arch}/platform/*` | Full (all runtimes) | `./ci.sh -p <platform>` |
| `src/{arch}/runtime/<rt>/*` | Single runtime | `./ci.sh -p <platform> -r <rt>` |
| `examples/{arch}/<rt>/<ex>/*` | Single example | `python examples/scripts/run_example.py --build -k <ex>/kernels -g <ex>/golden.py -p <platform>` |
| `tests/ut/*` (Python) | Python UT only | `pytest tests/ut` (add `--platform <platform>` on a device runner) |
| `tests/ut/cpp/*` | C++ UT only | `cmake -B tests/ut/cpp/build -S tests/ut/cpp && cmake --build tests/ut/cpp/build && ctest --test-dir tests/ut/cpp/build -LE requires_hardware` |
| Mixed (spans multiple categories) | Escalate to the **widest** matching scope | — |

> **Note on `--build`**: When changed paths include `src/{arch}/runtime/` or `src/{arch}/platform/`, always use `--build` for single-example `run_example.py` commands. `ci.sh` uses pre-built binaries from `pip install .`, so runtime C++ changes require re-running `pip install -e .` before `ci.sh`.

### Step 3 — Parallel Strategy

**Simulation (`a2a3sim`)**: based on CPU core count.

```bash
CORES=$(nproc)
```

| Condition | Action |
| --------- | ------ |
| `CORES >= 16` | Append `--parallel` |
| `CORES < 16` | Run sequentially (omit `--parallel`) |

**Hardware (`a2a3`)**: based on idle NPU device count (see Step 4).

| Condition | Action |
| --------- | ------ |
| 2+ idle devices | Append `--parallel` |
| 1 idle device | Run sequentially |

### Step 4 — Device Detection (hardware only)

When testing on `a2a3`, detect idle devices:

```bash
npu-smi info
```

Pick devices whose **HBM-Usage is 0** and find the **longest consecutive sub-range** (at most 4). Pass as `-d <start>-<end>` (or `-d <id>` if only one idle device). If no idle device is found, skip hardware testing and warn.

### Decision Tree

```text
git diff --name-only
  │
  ├─ Only docs/config? ──→ SKIP tests
  │
  └─ Code changed?
       │
       ├─ Determine SCOPE (Step 2)
       │    ├─ platform   → full
       │    ├─ runtime    → single runtime
       │    └─ example    → single example
       │
       ├─ Runtime C++ changed (src/{arch}/)?
       │    ├─ Yes → use --build for run_example.py (ci.sh auto-rebuilds)
       │    └─ No  → omit --build (uses pre-built binaries)
       │
       ├─ Sim:  nproc >= 16?        ──→ --parallel
       ├─ Dev:  idle devices >= 2?  ──→ --parallel
       │
       └─ npu-smi found?
            ├─ Yes → sim + device (idle devs, max 4)
            └─ No  → sim only
```

## Adding a New Example or Device Test

1. Create a directory under the appropriate arch and runtime:
   - Examples: `examples/{arch}/<runtime>/<name>/`
   - Device scene tests: `tests/st/{arch}/<runtime>/<name>/`
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

## Related Skills

- **`git-commit`** — Complete commit workflow (runs testing as a prerequisite)
