# Architecture

## Three-Program Model

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
│ Runtime          │    Loaded at runtime
│ MemoryAllocator  │         │
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

## Components

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

## API Layers

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

## Platform Backends

Two backends under `src/{arch}/platform/`: `onboard/` (real Ascend hardware) and `sim/` (thread-based host simulation, no SDK required).

See per-arch platform docs: [a2a3](../src/a2a3/docs/platform.md), [a5](../src/a5/docs/platform.md).
