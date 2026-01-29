# PTO Runtime - Task Runtime Execution Framework

Modular runtime for building and executing task dependency runtimes on Ascend devices with coordinated AICPU and AICore execution. Three independently compiled programs work together through clearly defined APIs.

## Architecture Overview

The PTO Runtime consists of **three separate programs** that communicate through well-defined APIs:

```
┌─────────────────────────────────────────────────────────────┐
│                    Python Application                        │
│         (examples/host_build_graph_example/main.py)         │
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
│ (src/platform/   │  │  (AICPU + AICore)│
│  a2a3/host/)     │  └──────────────────┘
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
    │ AICPU: Task Scheduler       │  (src/platform/a2a3/aicpu/)
    │ AICore: Compute Kernels     │  (src/platform/a2a3/aicore/)
    └────────────────────────────┘
```

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

### 1. Host Runtime (`src/platform/a2a3/host/`)
**C++ library** - Device orchestration and management
- `DeviceRunner`: Singleton managing device operations
- `Runtime`: Task dependency runtime data structure
- `MemoryAllocator`: Device tensor memory management
- `pto_runtime_c_api.h`: Pure C API for Python bindings
- Compiled to shared library (.so) at runtime

**Key Responsibilities:**
- Allocate/free device memory
- Host ↔ Device data transfer
- AICPU kernel launching and configuration
- AICore kernel registration and loading
- Runtime execution workflow coordination

### 2. AICPU Kernel (`src/platform/a2a3/aicpu/`)
**Device program** - Task scheduler running on AICPU processor
- `kernel.cpp`: Kernel entry points and handshake protocol
- Runtime-specific executor in `src/runtime/host_build_graph/aicpu/`
- Compiled to device binary at build time

**Key Responsibilities:**
- Initialize handshake protocol with AICore cores
- Identify initially ready tasks (fanin=0)
- Dispatch ready tasks to idle AICore cores
- Track task completion and update dependencies
- Continue until all tasks complete

### 3. AICore Kernel (`src/platform/a2a3/aicore/`)
**Device program** - Computation kernels executing on AICore processors
- `kernel.cpp`: Task execution kernels (add, mul, etc.)
- Runtime-specific executor in `src/runtime/host_build_graph/aicore/`
- Compiled to object file (.o) at build time

**Key Responsibilities:**
- Wait for task assignment via handshake buffer
- Read task arguments and kernel address
- Execute kernel using PTO ISA
- Signal task completion
- Poll for next task or quit signal

## API Layers

Three layers of APIs enable the separation:

### Layer 1: C++ API (`src/platform/a2a3/host/device_runner.h`)
```cpp
DeviceRunner& runner = DeviceRunner::Get();
runner.Init(device_id, num_cores, aicpu_bin, aicore_bin, pto_isa_root);
runner.AllocateTensor(bytes);
runner.CopyToDevice(device_ptr, host_ptr, bytes);
runner.Run(runtime);
runner.Finalize();
```

### Layer 2: C API (`src/platform/a2a3/host/pto_runtime_c_api.h`)
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
│   ├── platform/                       # Platform-specific implementations
│   │   ├── a2a3/                       # Ascend A2/A3 platform
│   │   │   ├── host/                   # Host runtime program
│   │   │   │   ├── device_runner.h/cpp  # Device management
│   │   │   │   ├── memory_allocator.h/cpp # Memory allocation
│   │   │   │   ├── function_cache.h    # Kernel binary cache
│   │   │   │   └── pto_runtime_c_api.h/cpp # C API for bindings
│   │   │   ├── aicpu/                  # AICPU kernel (device program)
│   │   │   │   ├── kernel.cpp          # Entry points & handshake
│   │   │   │   └── device_log.h/cpp    # Device logging
│   │   │   ├── aicore/                 # AICore kernel (device program)
│   │   │   │   ├── kernel.cpp          # Task execution kernels
│   │   │   │   └── aicore.h            # AICore header
│   │   │   └── common/                 # Shared structures
│   │   │       └── kernel_args.h       # Kernel argument structures
│   │   │
│   │   └── a2a3sim/                    # Thread-based simulation platform
│   │       ├── host/                   # Simulation host runtime
│   │       │   ├── device_runner.h/cpp  # Thread-based device emulation
│   │       │   ├── memory_allocator.h/cpp # Host memory allocation
│   │       │   └── pto_runtime_c_api.h/cpp # Same C API as a2a3
│   │       ├── aicpu/                  # Simulation AICPU
│   │       ├── aicore/                 # Simulation AICore
│   │       └── common/                 # Shared structures
│   │
│   └── runtime/                        # Runtime implementations
│       └── host_build_graph/           # Host-built graph runtime
│           ├── build_config.py         # Build configuration
│           ├── host/
│           │   └── runtime_maker.cpp    # C++ runtime builder & validator
│           ├── aicpu/
│           │   └── aicpu_executor.cpp # Task scheduler implementation
│           ├── aicore/
│           │   └── aicore_executor.cpp # AICore task executor
│           └── runtime/
│               └── runtime.h/cpp       # Task runtime and handshake structures
│
├── python/                             # Language bindings
│   ├── bindings.py                      # ctypes wrapper (C → Python)
│   ├── runtime_builder.py              # Python runtime builder
│   ├── binary_compiler.py              # Multi-platform compiler
│   ├── pto_compiler.py                 # PTO kernel compiler
│   ├── elf_parser.py                   # ELF binary parser
│   └── toolchain.py                    # Toolchain configuration
│
├── examples/                           # Working examples
│   ├── host_build_graph_example/       # Host-built graph example (a2a3)
│   │   ├── main.py                     # Python orchestration
│   │   ├── README.md                   # Example documentation
│   │   └── kernels/
│   │       ├── kernel_config.py        # Kernel configuration
│   │       ├── aiv/                    # AIV kernels
│   │       │   ├── kernel_add.cpp
│   │       │   ├── kernel_add_scalar.cpp
│   │       │   └── kernel_mul.cpp
│   │       └── orchestration/
│   │           └── example_orch.cpp    # Orchestration kernel
│   │
│   └── host_build_graph_sim_example/   # Simulation example (a2a3sim)
│       ├── main.py                     # Python orchestration
│       └── kernels/                    # Simulation kernels (plain C++)
│
└── tests/                              # Test suite
    └── test_runtime_builder.py         # Runtime builder tests
```

## Developer Guidelines

Each developer role has a designated working directory:

| Role | Directory | Responsibility |
|------|-----------|----------------|
| **Platform Developer** | `src/platform/` | Platform-specific logic and abstractions |
| **Runtime Developer** | `src/runtime/` | Runtime logic (host, aicpu, aicore, common) |
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

The **BinaryCompiler** class handles compilation of all three components separately. Use the `platform` parameter to select the target platform:

```python
from binary_compiler import BinaryCompiler

# For real Ascend hardware (requires CANN toolkit)
compiler = BinaryCompiler(platform="a2a3")

# For simulation (no Ascend SDK needed)
compiler = BinaryCompiler(platform="a2a3sim")

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

## Usage

### Quick Start - Python Example

```python
from bindings import bind_host_binary
from binary_compiler import BinaryCompiler

# Compile all binaries
compiler = BinaryCompiler()
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

```bash
cd examples/host_build_graph_example
python3 main.py [device_id]
```

This example:
1. Compiles AICPU, AICore, and Host binaries using BinaryCompiler
2. Loads the host runtime library
3. Initializes DeviceRunner with compiled binaries
4. Creates a task runtime: `f = (a + b + 1)(a + b + 2)` with 4 tasks and dependencies
5. Executes on device (AICPU scheduling, AICore computing)
6. Validates results and cleans up

Expected output:
```
=== Creating and Initializing Runtime ===
Formula: (a + b + 1)(a + b + 2)

=== Executing Runtime on Device ===

=== Validating Results and Cleaning Up ===
✓ SUCCESS: All 16384 elements are correct (42.0)
Formula verified: (a + b + 1)(a + b + 2) = (2+3+1)*(2+3+2) = 42
```

## Execution Flow

### 1. Python Setup Phase
```
Python main.py
  │
  ├─→ BinaryCompiler.compile("host", ...) → host_binary (.so)
  ├─→ BinaryCompiler.compile("aicpu", ...) → aicpu_binary (.so)
  ├─→ BinaryCompiler.compile("aicore", ...) → aicore_binary (.o)
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

## Components in Detail

### Host Runtime (`src/platform/a2a3/host/`)

**DeviceRunner**: Singleton managing device operations
- Allocate/free device tensor memory
- Copy data between host and device
- Launch AICPU and AICore kernels
- Manage handshake buffers
- Coordinate runtime execution

**Runtime**: Task dependency runtime
- Add tasks with arguments and function IDs
- Add dependencies between tasks (fanin/fanout)
- Query task information and dependency structure
- Calculate topologically ready tasks

**MemoryAllocator**: Device memory management
- Allocate blocks from device GM memory
- Track allocations automatically
- Free with automatic cleanup on finalization

**pto_runtime_c_api**: Pure C interface
- Enables Python ctypes bindings
- Wraps C++ classes as opaque pointers
- Error codes: 0=success, negative=failure
- All memory management in C++

### AICPU Kernel (`src/platform/a2a3/aicpu/`)

**kernel.cpp**: Kernel entry points
- Initialization kernel: Sets up handshake protocol
- Main scheduler kernel: Task scheduling loop
- Handshake initialization and management

**execute.cpp**: Task scheduler
- Ready task identification
- Task dispatch to cores
- Dependency tracking and updates
- Loop until completion

### AICore Kernel (`src/platform/a2a3/aicore/`)

**kernel.cpp**: Computation kernels
- Task execution implementations
- Kernel function pointers indexed by func_id
- Memory access and PTO ISA operations
- Handshake buffer polling

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
In [src/runtime/host_build_graph/runtime/runtime.h](src/runtime/host_build_graph/runtime/runtime.h):
```cpp
#define RUNTIME_MAX_TASKS 1024     // Maximum number of tasks
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

## References

- [src/platform/a2a3/host/](src/platform/a2a3/host/) - Host runtime implementation (real hardware)
- [src/platform/a2a3/aicpu/](src/platform/a2a3/aicpu/) - AICPU scheduler implementation
- [src/platform/a2a3/aicore/](src/platform/a2a3/aicore/) - AICore kernel implementation
- [src/platform/a2a3sim/](src/platform/a2a3sim/) - Thread-based simulation platform
- [src/runtime/host_build_graph/](src/runtime/host_build_graph/) - Host-built graph runtime
- [examples/host_build_graph_example/](examples/host_build_graph_example/) - Hardware example (a2a3)
- [examples/host_build_graph_sim_example/](examples/host_build_graph_sim_example/) - Simulation example (a2a3sim)
- [python/](python/) - Python bindings and compiler
