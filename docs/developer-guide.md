# Developer Guide

## Directory Structure

```
pto-runtime/
├── src/
│   ├── common/task_interface/            # Cross-architecture shared headers (data_type.h, task_arg.h)
│   └── {arch}/                         # Architecture-specific code (a2a3, a5)
│       ├── platform/                   # Platform-specific implementations
│       │   ├── include/                # Shared headers (host/, aicpu/, aicore/, common/)
│       │   ├── src/                    # Shared source (compiled into both backends)
│       │   ├── onboard/               # Real hardware backend
│       │   │   ├── host/              # Host runtime (.so)
│       │   │   ├── aicpu/             # AICPU kernel (.so)
│       │   │   └── aicore/            # AICore kernel (.o)
│       │   └── sim/                   # Thread-based simulation backend
│       │       ├── host/
│       │       ├── aicpu/
│       │       └── aicore/
│       │
│       └── runtime/                   # Runtime implementations
│           ├── common/                # Shared components across runtimes
│           ├── host_build_graph/      # Host-built graph runtime
│           ├── aicpu_build_graph/     # AICPU-built graph runtime
│           └── tensormap_and_ringbuffer/  # Advanced production runtime
│
├── python/                            # Language bindings
│   ├── bindings.py                    # ctypes wrapper (C -> Python)
│   ├── runtime_builder.py             # Python runtime builder
│   ├── runtime_compiler.py            # Multi-platform runtime compiler
│   ├── kernel_compiler.py             # Kernel compiler
│   ├── elf_parser.py                  # ELF binary parser
│   └── toolchain.py                   # Toolchain configuration
│
├── examples/                          # Working examples
│   ├── scripts/                       # Test framework scripts
│   └── {arch}/                        # Architecture-specific examples
│       ├── host_build_graph/
│       ├── aicpu_build_graph/
│       └── tensormap_and_ringbuffer/
│
├── tests/                             # Test suite
│   ├── cpp/                           # C++ unit tests (GoogleTest)
│   ├── device_tests/                  # Device integration tests
│   └── test_runtime_builder.py        # Python unit tests
│
└── docs/                              # Documentation
```

## Role-Based Directory Ownership

| Role | Directory | Responsibility |
|------|-----------|----------------|
| **Platform Developer** | `src/{arch}/platform/` | Platform-specific logic and abstractions |
| **Runtime Developer** | `src/{arch}/runtime/` | Runtime logic (host, aicpu, aicore, common) |
| **Codegen Developer** | `examples/` | Code generation examples and kernel implementations |

**Rules:**
- Stay within your assigned directory unless explicitly requested otherwise
- Create new subdirectories under your assigned directory as needed
- When in doubt, ask before making changes to other areas

## Compilation Pipeline

Python modules under `python/` drive the build:
1. `kernel_compiler.py` — compiles user-written kernel `.cpp` files (one per `func_id`)
2. `runtime_compiler.py` — compiles runtime sources for each component (host, aicpu, aicore)
3. `runtime_builder.py` — orchestrates the full build pipeline (compile + link)
4. `bindings.py` — provides ctypes wrappers for calling the host `.so` from Python

## Cross-Platform Preprocessor Convention

When preprocessor guards are used to isolate platform code paths, the `__aarch64__` block must be placed first:

```cpp
#if defined(__aarch64__)
// aarch64 path (must be first)
#elif defined(__x86_64__)
// x86_64 host simulation path
#else
// other platforms
#endif
```

## Example / Test Layout

Every example and device test follows this structure:
```
my_example/
  golden.py              # generate_inputs() + compute_golden()
  kernels/
    kernel_config.py     # KERNELS list + ORCHESTRATION dict + RUNTIME_CONFIG
    aic/                 # AICore kernel sources (optional)
    aiv/                 # AIV kernel sources (optional)
    orchestration/       # Orchestration C++ source
```

Run with: `python examples/scripts/run_example.py -k <kernels_dir> -g <golden.py> -p <platform>`

## Dynamic Kernel Compilation

Compile and load kernels at runtime without rebuilding:

```cpp
// In host code
runner.CompileAndLoadKernel(func_id, "path/to/kernel.cpp", core_type);
```

This compiles the kernel source using `ccec`, loads the binary to device memory, and registers it for task dispatch.

## Features

- **Three programs compile independently** with clear API boundaries
- **Full Python API** with ctypes and NumPy integration
- **Modular design** enables parallel component development
- **Runtime linking** via binary loading
