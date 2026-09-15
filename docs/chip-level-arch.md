# Chip-Level Architecture (L2)

This document describes the **single-chip (L2) architecture** — how a host
program, AICPU kernel, and AICore kernel cooperate on one Ascend NPU chip. For
the multi-chip hierarchy (L3+: Orchestrator / Scheduler / Worker composition)
see [hierarchical-level-runtime.md](hierarchical-level-runtime.md). For how task
data (Callable / TaskArgs / CallConfig) flows through all levels, see
[task-flow.md](task-flow.md).

## Three-Program Model

The simpler runtime consists of **three separate programs** that communicate through well-defined APIs:

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
- `runtime_c_api.h`: Pure C API for `ChipWorker` bindings (`src/common/worker/runtime_c_api.h`)
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
std::unique_ptr<DeviceRunnerBase::PreparedExecution> prepared;
runner.prepare_execution(runtime, config, pipeline_slot, identity, &prepared);
auto launched = runner.launch_execution(std::move(prepared), std::move(permit));
runner.drain_execution(*launched.active);            // child progress path owns progress
runner.finalize();
```

### Layer 2: C API (`src/common/worker/runtime_c_api.h`)

```c
// _task_interface owns this process's SimplerHostLogState. Python seeds its
// threshold before entering the C++ worker.

// host_runtime.so is self-contained and loaded RTLD_LOCAL:
SimplerHostLogBindStateFn bind_host_log_state =
    (SimplerHostLogBindStateFn)dlsym(host_runtime,
                                    "simpler_host_log_bind_state");
bind_host_log_state(host_log_state);
DeviceContextHandle ctx = create_device_context();
simpler_init(ctx, device_id,                          // attach + binary takeover
             aicpu_binary, aicpu_size,
             aicore_binary, aicore_size,
             dispatcher_binary, dispatcher_size,
             prewarm_config);
size_t size = get_runtime_size();
size_t alignment = get_runtime_alignment();
void *runtime = allocate_zeroed_aligned(size, alignment); // stable until finalize
simpler_register_callable(ctx, cid, callable);        // one-time per callable

// Progressable form; simpler_run(...) composes these phases synchronously.
// `descriptor` carries this run's slot/bank/identity and the optional
// launch-acceptance target published at the kernel-launch marker.
simpler_prepare_run(ctx, runtime, cid, args, config, &descriptor); // bind, no device launch
simpler_launch_run(ctx, runtime);  // publishes descriptor acceptance and returns after launch fence
simpler_wait_run(ctx, runtime);                                    // or poll until complete
simpler_finalize_run(ctx, runtime);                                // validate, copy back, destroy

simpler_unregister_callable(ctx, cid);
finalize_device(ctx);
destroy_device_context(ctx);
free(runtime);
```

### Layer 3: Python API (`python/bindings/task_interface.cpp` via nanobind)

```python
from simpler.task_interface import ChipWorker, ChipCallable, ChipStorageTaskArgs, CallConfig

worker = ChipWorker()
worker.init(device_id=0, bins=bins)   # bins = RuntimeBuilder(platform).get_binaries(...)

config = CallConfig()
# A run always takes the whole device; there is no per-call width knob.
config.aicpu_thread_num = 0  # auto
config.enable_pmu = 0
worker.run(callable, args, config)
worker.finalize()
```

### Python Type Naming Convention

Layer 3 Python types use a **level-prefixed naming convention** that mirrors the
level model (see [hierarchical-level-runtime.md](hierarchical-level-runtime.md)):

| Concept | L2 (Chip) type | L3+ (Distributed) type | Unified factory |
| ------- | -------------- | ---------------------- | --------------- |
| Worker | `ChipWorker` | `Worker` | `Worker(level=N)` |
| Callable | `ChipCallable` | *(planned)* | — |
| TaskArgs | `ChipStorageTaskArgs` | *(planned)* | — |
| Config | `CallConfig` | `CallConfig` | — |

`CallConfig` is the exception — same type used at every level, with no
`Chip*` / unprefixed split (see [task-flow.md](task-flow.md) for details).
The unified `Worker(level=N)` factory already routes to the correct backend.
At `level=2` the default `execution_mode="program"` drives `ChipWorker.init` /
`run`. `Worker(level=2, execution_mode="kernel")` routes to the kernel entries
instead: `init(config=...)` to `ChipWorker.kernel_init`,
`kernel_prepare_callable` / `kernel_launch` to `ChipWorker.kernel_*`, and
`close()` to `ChipWorker.finalize`. Its `kernel_mode_supported` property
answers through `ChipWorker.probe_kernel_mode_supported`, before init and without
touching a device. Kernel mode has no L3+ route.
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
            ├─→ _initialize_host_log(log_level) → seed extension-owned state
            └─→ _ChipWorker.init(host_path, aicpu_path, aicore_path,
                                    dispatcher_path, device_id, ...,
                                    sim_context_path)  # C++
                 ├─→ process-registry sim_context.so + bind state     # sim PTO hooks
                 ├─→ dlopen(host.so, RTLD_LOCAL) → resolve C API symbols via dlsym
                 ├─→ simpler_host_log_bind_state(host_log_state)
                 ├─→ create_device_context() → DeviceContextHandle
                 └─→ simpler_init(ctx, device_id, binaries...)
                      ├─→ (onboard) dlog_setlevel(HostLogger.cann_level())   # before context open
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
worker.run(callable, args, CallConfig(aicpu_thread_num))
  │
  └─→ run_runtime(ctx, runtime, callable, args, ...)
       │
       ├─→ Upload the entire ChipCallable buffer (upload_chip_callable_buffer)
       │      then fill func_id_to_addr_[fid] = chip_dev + storage_offset + child_offset(i)
       ├─→ Allocate device tensors via MemoryAllocator
       ├─→ Copy input data to device
       ├─→ Build task graph with dependencies
       │
       ├─→ Copy Runtime to device memory
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
