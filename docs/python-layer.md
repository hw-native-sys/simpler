# Python Layer

## Purpose

The Python layer (`python/` + `examples/scripts/`) drives the full lifecycle: **compile → load → run → verify**. It provides the build system, kernel compilation, ctypes bindings to the C++ runtime, and a golden-test framework. No compiled Python extensions are used — everything works through `ctypes` and subprocess calls to compilers.

## Module Architecture

```
examples/scripts/
├── run_example.py          CLI entry point (arg parsing, logging setup)
└── code_runner.py          Test orchestrator (CodeRunner class)
        │
        ├── python/runtime_builder.py    Build orchestrator
        │       ├── runtime_compiler.py  CMake-based multi-target compilation
        │       │       ├── toolchain.py         Compiler identity (flags, paths)
        │       │       └── env_manager.py       ASCEND_HOME_PATH management
        │       └── kernel_compiler.py   Single-file kernel/orchestration compilation
        │               └── toolchain.py
        │
        ├── python/bindings.py           ctypes FFI to host .so
        │       └── elf_parser.py        .text section extraction from .o files
        │
        └── golden.py (user-provided)    Input generation + golden computation
```

## Build System

### Toolchain Classes (`python/toolchain.py`)

Each toolchain wraps a compiler identity with its flags and paths:

| Class | Compiler | Platform | Component |
|-------|----------|----------|-----------|
| `CCECToolchain` | `ccec` (Bisheng CCE) | a2a3 | AICore kernels |
| `Gxx15Toolchain` | `g++-15` | a2a3sim | AICore kernels (simulation) |
| `GxxToolchain` | `g++` | Both | Host, orchestration |
| `Aarch64GxxToolchain` | `aarch64-linux-gnu-g++` | a2a3 | AICPU (cross-compile) |

`ToolchainType` enum mirrors the C++ `compile_strategy.h`: `CCEC=0`, `HOST_GXX_15=1`, `HOST_GXX=2`, `AARCH64_GXX=3`.

Toolchain selection is driven by the C++ library — `get_incore_compiler()` and `get_orchestration_compiler()` are queried via the C API, so the C++ runtime decides which compiler to use, not Python.

### RuntimeBuilder (`python/runtime_builder.py`)

Discovers runtime variants by scanning `src/runtime/` for directories containing `build_config.py`. Calling `build(name)` loads the variant's `build_config.py`, resolves include/source paths, and compiles all three targets (host, aicpu, aicore) in parallel via `ThreadPoolExecutor`.

Returns `(host_binary, aicpu_binary, aicore_binary)` as raw bytes.

### RuntimeCompiler (`python/runtime_compiler.py`)

Cached singleton per platform (`get_instance(platform)`). Creates three `BuildTarget` objects, each pairing a toolchain with a CMake source directory and output binary name.

`compile()` runs CMake configure + make in a temporary directory and returns the compiled binary as bytes.

### KernelCompiler (`python/kernel_compiler.py`)

Two entry points:

- **`compile_incore(source, core_type, ...)`** — Compiles a single kernel `.cpp` file.
  - a2a3: uses `ccec`, produces `.o`; caller extracts `.text` section via `elf_parser`
  - a2a3sim: uses `g++-15` or `g++`, produces `.so` (directly loadable via `dlopen`)

- **`compile_orchestration(runtime_name, source, ...)`** — Compiles orchestration `.cpp` to `.so`.
  - Toolchain selected per-runtime (host builds use `HOST_GXX`, device builds use `AARCH64_GXX` on a2a3)

### ELF Parser (`python/elf_parser.py`)

Pure Python, no external dependencies. `extract_text_section()` auto-detects ELF64 or Mach-O format and extracts the `.text` / `__text` section. Used on the a2a3 platform where `ccec` produces `.o` object files — the `.text` section contains the kernel binary that gets uploaded to device GM memory.

## Bindings (`python/bindings.py`)

Pure `ctypes` FFI with no compiled Python extensions.

### Key API

| Function | Role |
|----------|------|
| `bind_host_binary(path_or_bytes)` | Load host `.so` via `CDLL`, return `Runtime` class |
| `set_device(device_id)` | Initialize device and create streams |
| `launch_runtime(runtime, ...)` | Execute runtime with AICPU + AICore binaries |
| `device_malloc(size)` | Allocate device memory |
| `copy_to_device(dev, host, size)` | Host → device transfer |
| `copy_from_device(host, dev, size)` | Device → host transfer |

### Argument Types

Constants matching `pto_runtime_c_api.h`:

| Constant | Value | Meaning |
|----------|-------|---------|
| `ARG_SCALAR` | 0 | Scalar value, passed directly |
| `ARG_INPUT_PTR` | 1 | Input pointer: `device_malloc` + `copy_to_device` |
| `ARG_OUTPUT_PTR` | 2 | Output pointer: `device_malloc` + record for copy-back |
| `ARG_INOUT_PTR` | 3 | Input/output: both copy-to and copy-back |

### Runtime Class

Manages an opaque buffer allocated via `ctypes.create_string_buffer(get_runtime_size())`. Calls `init_runtime()` (placement-new in C++) during `initialize()` and `finalize_runtime()` (destructor + copy-back) during `finalize()`.

## Test Framework (`examples/scripts/code_runner.py`)

### CodeRunner Flow

1. **Load config** — Dynamically import `kernel_config.py` (KERNELS, ORCHESTRATION, RUNTIME_CONFIG) and `golden.py`
2. **Parallel build** — `ThreadPoolExecutor` compiles runtime (3 targets), orchestration, and all kernels simultaneously
3. **Bind** — `bind_host_binary(host_binary)` → `Runtime` class; `set_device(device_id)`
4. **For each parameter set** in `PARAMS_LIST`:
   - `golden.generate_inputs(params)` → dict of torch tensors
   - `_build_func_args()` → flatten tensors to `[ptr_0..ptr_n, nbytes_0..nbytes_n, element_count]`
   - `runtime.initialize(orch_so, func_name, func_args, kernel_binaries)`
   - `golden.compute_golden(tensors, params)` → compute expected output in-place
   - `launch_runtime(runtime, aicpu_binary, aicore_binary)`
   - `runtime.finalize()` → copy results back from device
   - Compare output tensors vs golden via `torch.allclose(rtol, atol)`

### func_args Layout Convention

Tensors are flattened into a `uint64_t` array following this layout:

```
[ptr_0, ptr_1, ..., ptr_n, nbytes_0, nbytes_1, ..., nbytes_n, element_count]
```

Where:
- `ptr_i` = `tensor.data_ptr()` (host address; the C API handles device allocation)
- `nbytes_i` = `tensor.element_size() * tensor.numel()`
- `element_count` = number of elements in the first tensor

`arg_types` follows the same ordering, specifying `ARG_INPUT_PTR`, `ARG_OUTPUT_PTR`, or `ARG_SCALAR` for each position.

## Example Structure

```
examples/<runtime_variant>/<example_name>/
├── golden.py                    # generate_inputs() + compute_golden()
└── kernels/
    ├── kernel_config.py         # KERNELS, ORCHESTRATION, RUNTIME_CONFIG
    ├── aiv/                     # AIV kernel sources (vector)
    │   └── kernel_add.cpp
    ├── aic/                     # AIC kernel sources (compute)
    │   └── kernel_matmul.cpp
    └── orchestration/
        └── example_orch.cpp     # Graph construction function
```

### golden.py Interface

| Symbol | Required | Type | Description |
|--------|----------|------|-------------|
| `generate_inputs(params)` | Yes | `dict → dict` | Returns dict of named torch tensors |
| `compute_golden(tensors, params)` | Yes | `dict, dict → None` | Computes expected output in-place |
| `TENSOR_ORDER` | Yes | `list[str]` | Explicit ordering of tensors in func_args |
| `__outputs__` | No | `list[str]` | Output tensor names (or use `out_` prefix convention) |
| `PARAMS_LIST` | No | `list[dict]` | Multiple parameter sets for repeated testing |
| `RTOL`, `ATOL` | No | `float` | Comparison tolerances (default: `1e-5`) |

### kernel_config.py Interface

```python
KERNELS = [
    {"func_id": 0, "source": "aiv/kernel_add.cpp", "core_type": "aiv"},
    {"func_id": 1, "source": "aiv/kernel_mul.cpp", "core_type": "aiv"},
]

ORCHESTRATION = {
    "source": "orchestration/example_orch.cpp",
    "function_name": "build_example_graph",
}

RUNTIME_CONFIG = {
    "runtime": "host_build_graph",    # Which runtime variant
    "aicpu_thread_num": 3,            # AICPU scheduler threads
    "block_dim": 24,                  # Blocks (1 block = 1 AIC + 2 AIV)
}
```

## Design Principles

1. **Bytes-in, bytes-out** — All compilation produces raw bytes. Binaries are loaded via `ctypes.CDLL` (host `.so`) or passed as byte arrays (AICPU `.so`, AICore `.o`/`.so`). No intermediate files persist.

2. **Platform-runtime orthogonality** — The same runtime variant (e.g., `host_build_graph`) compiles for any platform (`a2a3` or `a2a3sim`). Platform and runtime are independent axes of configuration.

3. **Toolchain selection from C++** — `runtime_compile_info.cpp` in each runtime decides which compiler to use. Python queries this via `get_incore_compiler()` / `get_orchestration_compiler()` through the C API, keeping the decision authoritative.

4. **Parallel compilation** — `RuntimeBuilder` uses `ThreadPoolExecutor` for host/aicpu/aicore. `CodeRunner` adds kernel and orchestration compilation in parallel, maximizing build throughput.

5. **Auto-discovery** — `RuntimeBuilder` discovers runtimes by scanning for `build_config.py`. The CI script discovers examples by scanning for `kernel_config.py`. No manual registration required.

6. **Minimal dependencies** — `bindings.py` uses only `ctypes` (stdlib). `elf_parser.py` uses only `struct` (stdlib). The only external dependency is `torch` for tensor operations in the test framework.

## See Also

- [README.md](../README.md) — API layers (C++ → C → Python) and quick-start usage
- [docs/CONTRIBUTING.md](CONTRIBUTING.md) — Adding a new example
- [.ai-instructions/coding/testing.md](../.ai-instructions/coding/testing.md) — Golden test pattern reference
