# Chip-Level Architecture (L2)

This document describes the **single-chip (L2) architecture** — how a host
program, AICPU kernel, and AICore kernel cooperate on one Ascend NPU chip. For
the multi-chip hierarchy (L3+: Orchestrator / Scheduler / Worker composition)
see [hierarchical_level_runtime.md](hierarchical_level_runtime.md). For how task
data (Callable / TaskArgs / CallConfig) flows through all levels, see
[task-flow.md](task-flow.md).

## Three-Program Model

The PTO Runtime consists of **three separate programs** that communicate through well-defined APIs:

```text
┌─────────────────────────────────────────────────────────────┐
│                    Python Application                        │
│   (pytest @scene_test classes, or `python test_*.py`)        │
└─────────────────────────┬───────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
    nanobind          ChipWorker       RuntimeBuilder
    (task_interface)  (dlopen host.so)  (compile binaries)
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

- `DeviceRunner`: Handle-based device context manager (one per `ChipWorker`)
- `MemoryAllocator`: Device tensor memory management
- `pto_runtime_c_api.h`: Pure C API for `ChipWorker` bindings (`src/common/worker/pto_runtime_c_api.h`)
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
- Wire fanout dependency edges from orchestrator's wiring queue (scheduler thread 0)
- Identify ready tasks (fanin satisfied) and enqueue to ready queues
- Dispatch ready tasks to idle AICore cores
- Track task completion and notify downstream consumers
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
DeviceRunner runner;
void *ptr = runner.allocate_tensor(bytes);
runner.copy_to_device(dev_ptr, host_ptr, bytes);
runner.set_executors(aicpu_binary, aicore_binary);   // once, at init time
runner.run(runtime, block_dim, launch_aicpu_num);
runner.finalize();
```

### Layer 2: C API (`src/common/worker/pto_runtime_c_api.h`)

```c
// libsimpler_log.so (RTLD_GLOBAL, loaded first by the Python wrapper):
simpler_log_init(log_level, log_info_v);              // seed HostLogger once

// host_runtime.so (RTLD_LOCAL, loaded after):
DeviceContextHandle ctx = create_device_context();
simpler_init(ctx, device_id,                          // attach + binary takeover
             aicpu_binary, aicpu_size,
             aicore_binary, aicore_size);
size_t size = get_runtime_size();
prepare_callable(ctx, cid, callable);                 // one-time per callable
run_prepared(ctx, runtime, cid, args, block_dim,      // per-launch — no binaries
             aicpu_thread_num,
             enable_l2_swimlane, enable_dump_tensor, enable_pmu, output_prefix);
unregister_callable(ctx, cid);
finalize_device(ctx);
destroy_device_context(ctx);
```

### Layer 3: Python API (`python/bindings/task_interface.cpp` via nanobind)

```python
from simpler.task_interface import ChipWorker, ChipCallable, ChipStorageTaskArgs, CallConfig

worker = ChipWorker()
worker.init(device_id=0, bins=bins)   # bins = RuntimeBuilder(platform).get_binaries(...)

config = CallConfig()
config.block_dim = 24
config.aicpu_thread_num = 3
config.enable_pmu = 0
worker.run(callable, args, config)
worker.finalize()
```

### Python Type Naming Convention

Layer 3 Python types use a **level-prefixed naming convention** that mirrors the
level model (see [hierarchical_level_runtime.md](hierarchical_level_runtime.md)):

| Concept | L2 (Chip) type | L3+ (Distributed) type | Unified factory |
| ------- | -------------- | ---------------------- | --------------- |
| Worker | `ChipWorker` | `Worker` | `Worker(level=N)` |
| Callable | `ChipCallable` | *(planned)* | — |
| TaskArgs | `ChipStorageTaskArgs` | *(planned)* | — |
| Config | `CallConfig` | `CallConfig` | — |

`CallConfig` is the exception — same type used at every level, with no
`Chip*` / unprefixed split (see [task-flow.md](task-flow.md) for details).
The unified `Worker(level=N)` factory already routes to the correct backend.
When new level-specific types are added (e.g. `ChipCallable`), each concept
should follow the same pattern: a `Chip*` concrete type for L2, a prefix-less
concrete type for L3+, and optionally a factory function that routes by level.

## Execution Flow

### 1. Python Setup Phase

```text
Python test_*.py (SceneTestCase)
  │
  ├─→ RuntimeBuilder(platform).get_binaries(runtime_name) → host.so, aicpu.so, aicore.o
  ├─→ KernelCompiler(platform).compile_incore(source, core_type) → kernel .o/.so
  ├─→ KernelCompiler(platform).compile_orchestration(runtime, source) → orch .so
  │
  └─→ ChipWorker()
       └─→ init(device_id, bins)                          # Python wrapper
            ├─→ ctypes.CDLL(libsimpler_log.so, RTLD_GLOBAL)   # once per process
            ├─→ simpler_log_init(log_level, log_info_v) → HostLogger seeded
            ├─→ ctypes.CDLL(libcpu_sim_context.so, RTLD_GLOBAL)  # sim only, once
            └─→ _ChipWorker.init(host_path, aicpu_path, aicore_path, device_id)  # C++
                 ├─→ dlopen(host.so, RTLD_LOCAL) → resolve C API symbols via dlsym
                 ├─→ create_device_context() → DeviceContextHandle
                 └─→ simpler_init(ctx, device_id, aicpu*, aicpu_size, aicore*, aicore_size)
                      ├─→ (onboard) dlog_setlevel(HostLogger.level())   # before context open
                      ├─→ DeviceRunner::attach_current_thread(device_id)
                      │    ├─→ rtSetDevice(device_id) on onboard
                      │    └─→ pto_cpu_sim_bind+acquire on sim
                      └─→ DeviceRunner::set_executors(aicpu, aicore)
```

### 2. Initialization Phase

The thread that called `init()` is now attached to `device_id`. Streams are
created lazily on the first `run()` call (`prepare_run_context`). Subsequent
device-ops (`malloc`, `copy_to`, `copy_from`, `free`) reuse that per-thread
binding — they must be called from the same thread that called `init()`.

### 3. Execution Phase

```text
worker.run(callable, args, CallConfig(block_dim, aicpu_thread_num))
  │
  └─→ run_runtime(ctx, runtime, callable, args, ...)
       │
       ├─→ Upload kernel binaries (upload_kernel_binary per func_id)
       ├─→ Allocate device tensors via MemoryAllocator
       ├─→ Copy input data to device
       ├─→ Build task graph with dependencies
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
       ├─→ rtStreamSynchronize (wait for completion)
       │
       ├─→ Copy results from device to host
       └─→ Clean up device tensors and runtime
```

### 4. Finalization Phase

```text
worker.finalize()
  │
  └─→ finalize_device(ctx)
       ├─→ Release device resources
       └─→ destroy_device_context(ctx)
```

## Handshake Protocol

AICPU and AICore cores coordinate via **handshake buffers** (one per core):

```c
struct Handshake {
    volatile uint32_t aicpu_ready;   // AICPU→AICore: scheduler ready
    volatile uint32_t aicore_done;   // AICore→AICPU: core ready
    volatile uint64_t task;          // AICPU→AICore: task pointer (init only; runtime uses DATA_MAIN_BASE)
};
```

**Flow:**

1. AICPU finds a ready task
2. AICPU writes task pointer to handshake buffer and signals via DATA_MAIN_BASE register
3. AICore polls DATA_MAIN_BASE, reads the task, executes
4. AICore writes FIN to COND; AICPU observes completion
5. AICPU reads result and continues

## Platform Backends

Two backends under `src/{arch}/platform/`: `onboard/` (real Ascend hardware) and `sim/` (thread-based host simulation, no SDK required).

See per-arch platform docs: [a2a3](../src/a2a3/docs/platform.md), [a5](../src/a5/docs/platform.md).
