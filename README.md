# PTO Runtime - Task Runtime Execution Framework

Modular runtime for building and executing task dependency runtimes on Ascend devices with coordinated AICPU and AICore execution. Three independently compiled programs work together through clearly defined APIs.

## Architecture Overview

The PTO Runtime consists of **three separate programs** that communicate through well-defined APIs:

```
┌─────────────────────────────────────────────────────────────┐
│                    Python Application                        │
│              (examples/scripts/run_example.py)                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
    Python Bindings   (ctypes)      Device I/O
    bindings.py
         │                │                │
         ▼                ▼                ▼
┌──────────────────┐  ┌──────────────────┐
│   Host Runtime   │  │   Binary Data    │
│ (src/{arch}/     │  │  (AICPU + AICore)│
│  platform/)      │  └──────────────────┘
├──────────────────┤         │
│ DeviceRunner     │         │
│ Runtime          │         │
│ MemoryAllocator  │    Loaded at runtime
│ C API            │         │
└────────┬─────────┘         │
         │                   │
         └───────────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │  Ascend Device (Hardware)   │
    ├────────────────────────────┤
    │ AICPU: Task Scheduler       │
    │ AICore: Compute Kernels     │
    └────────────────────────────┘
```

## Setup

### Cloning the Repository

Simply clone the repository:

```bash
git clone <repo-url>
cd simpler
```

The pto-isa dependency will be automatically cloned when you first run an example that needs it.

### PTO ISA Headers

The pto-isa repository provides header files needed for kernel compilation on the `a2a3` (hardware) platform.

The test framework automatically handles PTO_ISA_ROOT setup:
1. Checks if `PTO_ISA_ROOT` is already set
2. If not, clones pto-isa to `examples/scripts/_deps/pto-isa` on first run
3. Passes the resolved path to the kernel compiler

**Automatic Setup (Recommended):**
Just run your example - pto-isa will be cloned automatically on first run:
```bash
python examples/scripts/run_example.py -k examples/a2a3/host_build_graph/vector_example/kernels \
                                       -g examples/a2a3/host_build_graph/vector_example/golden.py \
                                       -p a2a3sim
```

By default, the auto-clone uses SSH (`git@github.com:...`). In CI or environments without SSH keys, use `--clone-protocol https`:
```bash
python examples/scripts/run_example.py -k examples/a2a3/host_build_graph/vector_example/kernels \
                                       -g examples/a2a3/host_build_graph/vector_example/golden.py \
                                       -p a2a3sim --clone-protocol https
```

**Manual Setup** (if auto-setup fails or you prefer manual control):
```bash
# Clone pto-isa manually (SSH)
mkdir -p examples/scripts/_deps
git clone --branch main git@github.com:PTO-ISA/pto-isa.git examples/scripts/_deps/pto-isa

# Or use HTTPS
git clone --branch main https://github.com/PTO-ISA/pto-isa.git examples/scripts/_deps/pto-isa

# Set environment variable (optional - auto-detected if in standard location)
export PTO_ISA_ROOT=$(pwd)/examples/scripts/_deps/pto-isa
```

**Using a Different Location:**
If you already have pto-isa elsewhere, just set the environment variable:
```bash
export PTO_ISA_ROOT=/path/to/your/pto-isa
```

**Troubleshooting:**
- If git is not available: Clone pto-isa manually and set `PTO_ISA_ROOT`
- If clone fails due to network: Try again or clone manually
- If SSH clone fails (e.g., in CI): Use `--clone-protocol https` or clone manually with HTTPS

Note: For the simulation platform (`a2a3sim`), PTO ISA headers are optional and only needed if your kernels use PTO ISA intrinsics.

## Platforms

PTO Runtime supports multiple target platforms:

| Platform | Description | Requirements |
|----------|-------------|--------------|
| `a2a3` | Real Ascend hardware | CANN toolkit (ccec, aarch64 cross-compiler) |
| `a2a3sim` | Thread-based host simulation | gcc/g++ only (no Ascend SDK needed) |

```python
builder = RuntimeBuilder(platform="a2a3")      # Real hardware
builder = RuntimeBuilder(platform="a2a3sim")   # Simulation
```

The simulation platform (`a2a3sim`) uses host threads to emulate AICPU/AICore execution, enabling development and testing without Ascend hardware. Kernel `.text` sections are loaded into mmap'd executable memory for direct invocation.

## Three Components

### 1. Host Runtime (`src/{arch}/platform/*/host/`)
**C++ library** - Device orchestration and management
- `DeviceRunner`: Singleton managing device operations
- `MemoryAllocator`: Device tensor memory management
- `pto_runtime_c_api.h`: Pure C API for Python bindings
- Compiled to shared library (.so) at runtime

**Key Responsibilities:**
- Allocate/free device memory
- Host <-> Device data transfer
- AICPU kernel launching and configuration
- AICore kernel registration and loading
- Runtime execution workflow coordination

### 2. AICPU Kernel (`src/{arch}/platform/*/aicpu/`)
**Device program** - Task scheduler running on AICPU processor
- `kernel.cpp`: Kernel entry points and handshake protocol
- Runtime-specific executor in `src/{arch}/runtime/*/aicpu/`
- Compiled to device binary at build time

**Key Responsibilities:**
- Initialize handshake protocol with AICore cores
- Identify initially ready tasks (fanin=0)
- Dispatch ready tasks to idle AICore cores
- Track task completion and update dependencies
- Continue until all tasks complete

### 3. AICore Kernel (`src/{arch}/platform/*/aicore/`)
**Device program** - Computation kernels executing on AICore processors
- `kernel.cpp`: Task execution kernels (add, mul, etc.)
- Runtime-specific executor in `src/{arch}/runtime/*/aicore/`
- Compiled to object file (.o) at build time

**Key Responsibilities:**
- Wait for task assignment via handshake buffer
- Read task arguments and kernel address
- Execute kernel using PTO ISA
- Signal task completion
- Poll for next task or quit signal

## Runtime Variants

Three runtime implementations live under `src/{arch}/runtime/`, each providing a different graph-building strategy:

### host_build_graph
The Host CPU builds the full task dependency graph before launching execution. The graph is serialized to device memory and the AICPU scheduler dispatches tasks based on the pre-built dependency structure. Simplest runtime, good for static workloads.

### aicpu_build_graph
The AICPU builds and manages the task graph on-device. The host sends a compact description and the AICPU constructs the full graph locally, reducing host-device data transfer. Includes shared memory management, ring buffers, and a more capable scheduler.

### tensormap_and_ringbuffer
Advanced runtime with tensor maps, ring buffers, shared memory, and multi-core orchestration. Supports dynamic tensor management, pipelined execution via ring buffers, and orchestrator-driven multi-core dispatch. Designed for complex, high-throughput workloads.

Each runtime has a `build_config.py` declaring its include/source directories for the three components (host, aicpu, aicore). The `RUNTIME_CONFIG.runtime` field in `kernel_config.py` selects which runtime to use.

## API Layers

Three layers of APIs enable the separation:

### Layer 1: C++ API (`src/{arch}/platform/*/host/device_runner.h`)
```cpp
DeviceRunner& runner = DeviceRunner::Get();
runner.Init(device_id, num_cores, aicpu_bin, aicore_bin, pto_isa_root);
runner.AllocateTensor(bytes);
runner.CopyToDevice(device_ptr, host_ptr, bytes);
runner.Run(runtime);
runner.Finalize();
```

### Layer 2: C API (`src/{arch}/platform/include/host/pto_runtime_c_api.h`)
```c
int DeviceRunner_Init(device_id, num_cores, aicpu_binary, aicpu_size,
                      aicore_binary, aicore_size, pto_isa_root);
int DeviceRunner_Run(runtime_handle, launch_aicpu_num);
int InitRuntime(runtime_handle);
int FinalizeRuntime(runtime_handle);
int DeviceRunner_Finalize();
```

### Layer 3: Python API (`python/bindings.py`)
```python
Runtime = bind_host_binary(host_binary)
runtime = Runtime()
runtime.initialize()
launch_runtime(runtime, aicpu_thread_num=1, block_dim=1,
               device_id=device_id, aicpu_binary=aicpu_bytes,
               aicore_binary=aicore_bytes)
runtime.finalize()
```

## Directory Structure

```
pto-runtime/
├── src/
│   └── {arch}/                            # Architecture (e.g., a2a3)
│       ├── platform/                      # Platform-specific implementations
│       │   ├── include/                   # Shared platform interfaces
│       │   │   ├── host/                  # Host API headers
│       │   │   │   └── pto_runtime_c_api.h
│       │   │   ├── aicpu/                 # AICPU headers
│       │   │   ├── aicore/                # AICore headers
│       │   │   └── common/                # Shared structures (kernel_args.h)
│       │   ├── src/                       # Shared platform source
│       │   ├── onboard/                   # Real hardware backend (a2a3)
│       │   │   ├── host/                  # Hardware host runtime
│       │   │   ├── aicpu/                 # Hardware AICPU kernel
│       │   │   └── aicore/                # Hardware AICore kernel
│       │   └── sim/                       # Simulation backend (a2a3sim)
│       │       ├── host/                  # Simulation host runtime
│       │       ├── aicpu/                 # Simulation AICPU
│       │       └── aicore/                # Simulation AICore
│       │
│       └── runtime/                       # Runtime implementations
│           ├── host_build_graph/          # Host-built graph runtime
│           │   ├── build_config.py
│           │   ├── host/
│           │   ├── aicpu/
│           │   ├── aicore/
│           │   └── runtime/
│           ├── aicpu_build_graph/         # AICPU-built graph runtime
│           │   ├── build_config.py
│           │   ├── host/
│           │   ├── aicpu/
│           │   ├── aicore/
│           │   └── runtime/
│           └── tensormap_and_ringbuffer/  # Advanced runtime
│               ├── build_config.py
│               ├── host/
│               ├── aicpu/
│               ├── aicore/
│               └── runtime/
│
├── python/                                # Language bindings
│   ├── bindings.py                        # ctypes wrapper (C -> Python)
│   ├── runtime_builder.py                 # Python runtime builder
│   ├── runtime_compiler.py                # Multi-platform runtime compiler
│   ├── kernel_compiler.py                 # Kernel compiler
│   ├── elf_parser.py                      # ELF binary parser
│   └── toolchain.py                       # Toolchain configuration
│
├── examples/                              # Working examples
│   ├── scripts/                           # Test framework scripts
│   │   ├── run_example.py                 # Main test runner
│   │   ├── code_runner.py                 # Test execution engine
│   │   └── README.md                      # Test framework documentation
│   │
│   └── a2a3/                              # Examples for a2a3 architecture
│       ├── host_build_graph/              # Host-built graph examples
│       │   └── vector_example/
│       │       ├── golden.py
│       │       └── kernels/
│       ├── aicpu_build_graph/             # AICPU-built graph examples
│       └── tensormap_and_ringbuffer/      # Advanced runtime examples
│
└── tests/                                 # Test suite
    └── test_runtime_builder.py            # Runtime builder tests
```

## Developer Guidelines

Each developer role has a designated working directory:

| Role | Directory | Responsibility |
|------|-----------|----------------|
| **Platform Developer** | `src/{arch}/platform/` | Platform-specific logic and abstractions |
| **Runtime Developer** | `src/{arch}/runtime/` | Runtime logic (host, aicpu, aicore, common) |
| **Codegen Developer** | `examples/` | Code generation examples and kernel implementations |

**Rules:**
- Stay within your assigned directory unless explicitly requested otherwise
- Create new subdirectories under your assigned directory as needed
- When in doubt, ask before making changes to other areas

## Building

### Prerequisites
- CMake 3.15+
- CANN toolkit with:
  - `ccec` compiler (AICore Bisheng CCE)
  - Cross-compiler for AICPU (aarch64-target-linux-gnu-gcc/g++)
- Standard C/C++ compiler (gcc/g++) for host
- Python 3 with development headers

### Environment Setup
```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

### Build Process

The **RuntimeCompiler** class handles compilation of all three components separately. Use the `platform` parameter to select the target platform:

```python
from runtime_compiler import RuntimeCompiler

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

Each component is compiled independently with its own toolchain, allowing modular development.

### Cross-platform Platform-Isolation Requirement

When preprocessor guards are used to isolate platform code paths, the `__aarch64__`
block must be placed at the very beginning of the conditional chain.

```cpp
#if defined(__aarch64__)
// aarch64 path (must be first)
#elif defined(__x86_64__)
// x86_64 host simulation path
#else
// other platforms
#endif
```

## Usage

### Quick Start - Python Example

```python
from bindings import bind_host_binary
from runtime_compiler import RuntimeCompiler

# Compile all binaries
compiler = RuntimeCompiler()
aicore_bin = compiler.compile("aicore", [...include_dirs...], [...source_dirs...])
aicpu_bin = compiler.compile("aicpu", [...include_dirs...], [...source_dirs...])
host_bin = compiler.compile("host", [...include_dirs...], [...source_dirs...])

# Load and initialize runtime
Runtime = bind_host_binary(host_bin)
runtime = Runtime()
runtime.initialize()  # C++ builds runtime and allocates tensors

# Execute runtime on device
launch_runtime(runtime,
               aicpu_thread_num=1,
               block_dim=1,
               device_id=9,
               aicpu_binary=aicpu_bin,
               aicore_binary=aicore_bin)

runtime.finalize()  # Verify and cleanup
```

### Running the Example

Use the test framework to run examples:

```bash
# Hardware platform (requires Ascend device)
python examples/scripts/run_example.py \
  -k examples/a2a3/host_build_graph/vector_example/kernels \
  -g examples/a2a3/host_build_graph/vector_example/golden.py \
  -p a2a3

# Simulation platform (no hardware required)
python examples/scripts/run_example.py \
  -k examples/a2a3/host_build_graph/vector_example/kernels \
  -g examples/a2a3/host_build_graph/vector_example/golden.py \
  -p a2a3sim
```

This example:
1. Compiles AICPU, AICore, and Host binaries using RuntimeCompiler
2. Loads the host runtime library
3. Initializes DeviceRunner with compiled binaries
4. Creates a task runtime: `f = (a + b + 1)(a + b + 2)` with 4 tasks and dependencies
5. Executes on device (AICPU scheduling, AICore computing)
6. Validates results against golden output

Expected output:
```
=== Building Runtime: host_build_graph (platform: a2a3sim) ===
...
=== Comparing Results ===
Comparing f: shape=(16384,), dtype=float32
  f: PASS (16384/16384 elements matched)

============================================================
TEST PASSED
============================================================
```

## Execution Flow

### 1. Python Setup Phase
```
Python run_example.py
  │
  ├─→ RuntimeCompiler.compile("host", ...) → host_binary (.so)
  ├─→ RuntimeCompiler.compile("aicpu", ...) → aicpu_binary (.so)
  ├─→ RuntimeCompiler.compile("aicore", ...) → aicore_binary (.o)
  │
  └─→ bind_host_binary(host_binary)
       └─→ RuntimeLibraryLoader(host_binary)
            └─→ CDLL(host_binary) ← Loads .so into memory
```

### 2. Initialization Phase
```
runner.init(device_id, num_cores, aicpu_binary, aicore_binary, pto_isa_root)
  │
  ├─→ DeviceRunner_Init (C API)
  │    ├─→ Initialize CANN device
  │    ├─→ Allocate device streams
  │    ├─→ Load AICPU binary to device
  │    ├─→ Register AICore kernel binary
  │    └─→ Create handshake buffers (one per core)
  │
  └─→ DeviceRunner singleton ready
```

### 3. Runtime Building Phase
```
runtime.initialize()
  │
  └─→ InitRuntime (C API)
       └─→ InitRuntimeImpl (C++)
            ├─→ Compile kernels at runtime (CompileAndLoadKernel)
            │    ├─→ KernelCompiler calls ccec
            │    ├─→ Load .o to device GM memory
            │    └─→ Update kernel function address table
            │
            ├─→ Allocate device tensors via MemoryAllocator
            ├─→ Copy input data to device
            ├─→ Build task runtime with dependencies
            └─→ Return Runtime pointer
```

### 4. Execution Phase
```
launch_runtime(runtime, aicpu_thread_num=1, block_dim=1, device_id=device_id,
               aicpu_binary=aicpu_bytes, aicore_binary=aicore_bytes)
  │
  └─→ launch_runtime (C API)
       │
       ├─→ Copy Runtime to device memory
       │
       ├─→ LaunchAiCpuKernel (init kernel)
       │    └─→ Execute on AICPU: Initialize handshake
       │
       ├─→ LaunchAiCpuKernel (main scheduler kernel)
       │    └─→ Execute on AICPU: Task scheduler loop
       │         ├─→ Find initially ready tasks
       │         ├─→ Loop: dispatch tasks, wait for completion
       │         └─→ Continue until all tasks done
       │
       ├─→ LaunchAicoreKernel
       │    └─→ Execute on AICore cores: Task workers
       │         ├─→ Wait for task assignment
       │         ├─→ Execute kernel
       │         └─→ Signal completion, repeat
       │
       └─→ rtStreamSynchronize (wait for completion)
```

### 5. Validation Phase
```
runtime.finalize()
  │
  └─→ FinalizeRuntime (C API)
       └─→ FinalizeRuntimeImpl (C++)
            ├─→ Copy results from device to host
            ├─→ Verify correctness (compare with expected values)
            ├─→ Free all device tensors
            ├─→ Delete runtime
            └─→ Return success/failure
```

## Handshake Protocol

AICPU and AICore cores coordinate via **handshake buffers** (one per core):

```c
struct Handshake {
    volatile uint32_t aicpu_ready;   // AICPU→AICore: scheduler ready
    volatile uint32_t aicore_done;   // AICore→AICPU: core ready
    volatile uint64_t task;          // AICPU→AICore: task pointer
    volatile int32_t task_status;    // Task state: 1=busy, 0=done
    volatile int32_t control;        // AICPU→AICore: 1=quit
};
```

**Flow:**
1. AICPU finds a ready task
2. AICPU writes task pointer to handshake buffer and sets `aicpu_ready`
3. AICore polls buffer, sees task, reads from device memory
4. AICore sets `task_status = 1` (busy) and executes
5. AICore sets `task_status = 0` (done) and `aicore_done`
6. AICPU reads result and continues

## Features

### Dynamic Kernel Compilation
Compile and load kernels at runtime without rebuilding:

```cpp
// In host code
runner.CompileAndLoadKernel(func_id, "path/to/kernel.cpp", core_type);
```

This compiles the kernel source using `ccec`, loads the binary to device memory, and registers it for task dispatch.

### Python Bindings
Full Python API with ctypes:
- No C++ knowledge required
- NumPy integration for arrays
- Easy data transfer between host and device

### Modular Design
- Three programs compile independently
- Clear API boundaries
- Develop components in parallel
- Runtime linking via binary loading

## Configuration

### Compile-time Configuration (Runtime Limits)
In [src/a2a3/runtime/host_build_graph/runtime/runtime.h](src/a2a3/runtime/host_build_graph/runtime/runtime.h):
```cpp
#define RUNTIME_MAX_TASKS 131072   // Maximum number of tasks
#define RUNTIME_MAX_ARGS 16        // Maximum arguments per task
#define RUNTIME_MAX_FANOUT 512     // Maximum successors per task
```

### Runtime Configuration
```python
runner.init(
    device_id=0,              # Device ID (0-15)
    num_cores=3,              # Number of cores for handshake
    aicpu_binary=...,         # AICPU .so binary
    aicore_binary=...,        # AICore .o binary
    pto_isa_root="/path/to/pto-isa"  # PTO-ISA headers location
)
```

## Notes

- **Device IDs**: 0-15 (typically device 9 used for examples)
- **Handshake cores**: Usually 3 (1c2v configuration: 1 core, 2 vector units)
- **Kernel compilation**: Requires `ASCEND_HOME_PATH` environment variable
- **Memory management**: MemoryAllocator automatically tracks allocations
- **Python requirement**: NumPy for efficient array operations

## Logging

Device logs written to `~/ascend/log/debug/device-<id>/`

Kernel uses macros:
- `DEV_INFO`: Informational messages
- `DEV_DEBUG`: Debug messages
- `DEV_WARN`: Warnings
- `DEV_ERROR`: Error messages

## Testing

```bash
./ci.sh
```

## License

This project is licensed under the **CANN Open Software License Agreement Version 2.0**.

See the [LICENSE](LICENSE) file for the full license text.

## References

- [src/a2a3/platform/](src/a2a3/platform/) - Platform implementations (includes onboard and sim backends)
- [src/a2a3/runtime/host_build_graph/](src/a2a3/runtime/host_build_graph/) - Host-built graph runtime
- [src/a2a3/runtime/aicpu_build_graph/](src/a2a3/runtime/aicpu_build_graph/) - AICPU-built graph runtime
- [src/a2a3/runtime/tensormap_and_ringbuffer/](src/a2a3/runtime/tensormap_and_ringbuffer/) - Advanced runtime
- [examples/a2a3/](examples/a2a3/) - Examples for a2a3 architecture
- [python/](python/) - Python bindings and compiler
