# NVIDIA Backend Flow Details

This document expands the CUDA backend design with operational flow details:
how `simpler` compiles and launches today, where CUDA differs, which CUDA API
surface should own each step, and how lifecycle, memory, and callable concepts
map onto CUDA.

## Scope

This is not a low-level compiler comparison. The important boundary is how a
`simpler` user gets from Python test/example code to prepared device work:

1. prebuilt runtime lookup or rebuild;
2. user callable compilation;
3. worker initialization;
4. callable preparation;
5. task launch;
6. tensor copy-back and teardown.

## User-Facing Compile and Launch Flow

### Current a2a3/a5 Flow

```text
Python scene/example
  |
  | RuntimeBuilder(platform).get_binaries(runtime, build?)
  v
build/lib/{arch}/{variant}/{runtime}/
  |  libhost_runtime.so
  |  libaicpu_kernel.so
  |  aicore_kernel.o or libaicore_kernel.so for sim
  |
  | KernelCompiler(platform)
  |  - compile_orchestration(runtime, orch.cpp) -> orchestration .so bytes
  |  - compile_incore(kernel.cpp, core_type) -> AICore/AIV bytes
  v
ChipCallable bytes
  |
  | Worker(level=2).register(chip_callable) -> callable_id
  | Worker.init()
  v
ChipWorker.init(device_id, RuntimeBinaries)
  |  dlopen(libhost_runtime.so)
  |  load aicpu/aicore binary files into host memory
  |  create DeviceRunner context
  |  simpler_init(ctx, device_id, log config)
  |
  | Worker.run(callable_id, TaskArgs, CallConfig)
  v
ChipWorker.run_prepared()
  |  view_to_chip_storage(args)
  |  run_prepared C API
  v
host runtime DeviceRunner
  |  prepare runtime args
  |  upload/reuse user kernel and orchestration payloads
  |  launch AICPU scheduler + AICore executor
  |  synchronize stream
  v
results copied back by runtime or explicit worker.copy_from()
```

The important property is that runtime binaries and user callables are
different artifacts:

- runtime binaries are platform/runtime infrastructure built by
  `RuntimeBuilder`;
- `ChipCallable` is user work compiled by `KernelCompiler`;
- `ChipWorker` binds them at runtime through the stable host-runtime C API.

`host_build_graph` and `tensormap_and_ringbuffer` differ after `run_prepared`
enters the host runtime. The outer Python flow stays the same. CUDA should use
runtime names that describe its own execution model: `host_schedule` and
`persistent_device`.

### Proposed CUDA Flow

```text
Python scene/example
  |
  | RuntimeBuilder(platform="cuda").get_binaries(runtime, build?)
  v
build/lib/cuda/onboard/{runtime}/
  |  libhost_runtime.so
  |  optional executor fatbin/cubin/PTX
  |
  | CudaKernelCompiler or extended KernelCompiler
  |  - compile host_schedule callable -> fatbin/cubin/PTX + manifest
  |  - optional compile orchestration metadata for host scheduler
  v
ChipCallable-compatible CUDA payload
  |
  | Worker(level=2, platform="cuda").register(payload) -> callable_id
  | Worker.init()
  v
ChipWorker.init(device_id, RuntimeBinaries)
  |  dlopen(CUDA libhost_runtime.so)
  |  create CUDA DeviceRunner context
  |  retain or create CUDA context for device_id
  |  create stream(s)
  |
  | Worker.run(callable_id, TaskArgs, CallConfig)
  v
ChipWorker.run_prepared()
  |  same C API call shape as a2a3/a5
  v
CUDA host runtime
  |  prepare_callable: load module/library and cache function handles
  |  run_prepared: build kernel params and launch CUDA kernel(s)
  |  synchronize stream or event at the existing run boundary
  v
results copied back by runtime or explicit worker.copy_from()
```

The CUDA target should preserve the `simpler` usage model: users still compile
a callable, register it, initialize a `Worker`, and call `run`. The backend
changes what the callable payload contains and how the host runtime launches
it.

## Runtime Flow Comparison

### a2a3/a5 Hardware Flow

```text
install/build time:
  pip install . or --build
    -> build_runtimes.py
    -> RuntimeBuilder
    -> RuntimeCompiler
    -> host.so + aicpu.so + aicore.o

per callable:
  KernelCompiler.compile_orchestration()
    -> orchestration .so bytes
  KernelCompiler.compile_incore()
    -> kernel object/text bytes per func_id
  ChipCallable.build()
    -> orch bytes + child CoreCallable bytes

Worker.init:
  Python ChipWorker wrapper
    -> C++ ChipWorker::init
    -> load libsimpler_log.so globally
    -> dlopen host runtime locally
    -> dlsym full pto_runtime_c_api.h surface
    -> create_device_context()
    -> read aicpu/aicore binaries into host buffers
    -> simpler_init(ctx, device_id, log level)
    -> DeviceRunner attaches current thread to Ascend device

Worker.prepare/register:
  L2 direct path:
    -> Worker.register returns cid
    -> first run prepares lazily, or explicit prepare_callable
  L3 child process path:
    -> parent prewarms child via mailbox _CTRL_PREPARE
    -> child calls ChipWorker.prepare_callable(cid, callable)

prepare_callable in host runtime:
  -> upload child kernel binaries into device memory or sim cache
  -> upload orchestration SO payload
  -> cache prepared state keyed by callable_id/build-id
  -> AICPU or host side can dlopen orchestration once per cid

run_prepared in host runtime:
  -> copy TaskArgs into runtime ABI storage
  -> allocate/copy tensors as needed
  -> build or bind runtime graph
  -> copy Runtime descriptor to device
  -> launch AICPU init/main scheduler
  -> launch AICore executor
  -> scheduler dispatches tasks to AICore workers
  -> synchronize stream
  -> copy outputs and diagnostic data back

finalize:
  -> unregister callable state when requested
  -> free device allocations and streams
  -> destroy context
  -> dlclose host runtime
```

### CUDA Host-Schedule Flow

```text
install/build time:
  pip install . or --build
    -> build CUDA host runtime
    -> optionally build host_schedule support module/fatbin

per callable:
  CUDA callable compiler
    -> CUDA source/device IR -> PTX, cubin, or fatbin
    -> manifest: entry names, argument layout, launch policy
  ChipCallable-compatible payload
    -> module image bytes + metadata

Worker.init:
  Python ChipWorker wrapper
    -> C++ ChipWorker::init
    -> load libsimpler_log.so globally
    -> dlopen CUDA host runtime locally
    -> dlsym same pto_runtime_c_api.h surface
    -> create_device_context()
    -> simpler_init(ctx, device_id, log level)
    -> CUDA DeviceRunner retains/sets current context
    -> CUDA DeviceRunner creates streams/events

prepare_callable in CUDA host runtime:
  -> load module/library from payload bytes
  -> get function/kernel handles by entry name
  -> cache handles and manifest by callable_id
  -> optionally set per-kernel attributes

run_prepared in CUDA host runtime:
  -> translate ChipStorageTaskArgs into CUDA kernel parameter arrays
  -> apply launch policy from manifest and CallConfig
  -> enqueue kernel(s) on runner stream
  -> enqueue copies or let explicit worker.copy_from handle output
  -> synchronize stream/event before returning to preserve current semantics

finalize:
  -> unload modules/libraries
  -> destroy streams/events
  -> free allocations
  -> release CUDA context ownership
  -> dlclose host runtime
```

### CUDA Persistent-Device Flow

```text
install/build time:
  -> build host runtime
  -> build persistent executor object archive

per callable:
  -> compile/link user task bodies into executor-compatible module
  -> generate dispatch table and metadata

prepare_callable:
  -> load linked executor module
  -> copy dispatch table and callable metadata to device

run_prepared:
  -> copy Runtime descriptor, TensorMap, ring, and args to global memory
  -> launch one persistent scheduler/worker kernel
  -> scheduler blocks/warps consume ready queues
  -> worker blocks/warps call device functions through dispatch table
  -> host synchronizes and copies outputs/diagnostics
```

The persistent runtime is the closer match for `tensormap_and_ringbuffer`, but
it requires a different callable contract. CUDA cannot load an arbitrary
global kernel from device code the way the Ascend scheduler hands work to
AICore executors. The CUDA executor needs linked device functions or a generated
dispatch layer inside the module launched by the host. See
[persistent-device.md](persistent-device.md) for the detailed analysis.

## CUDA Runtime API vs Driver API

NVIDIA exposes two host APIs that can be mixed carefully.

| Concern | CUDA Runtime API | CUDA Driver API |
| ------- | ---------------- | --------------- |
| Context | implicit primary context management | explicit current context model |
| Memory | `cudaMalloc`, `cudaFree`, `cudaMemcpyAsync` | `cuMemAlloc`, `cuMemFree`, `cuMemcpy*Async` |
| Static kernel launch | natural `<<<...>>>` or `cudaLaunchKernel` | `cuLaunchKernel` / `cuLaunchKernelEx` |
| Dynamic module loading | newer `cudaLibraryLoadData` and kernel handles | mature `cuModuleLoadDataEx`, `cuModuleGetFunction` |
| Control level | simpler host code | finer control over module/context lifetime |
| Fit for plugin-style runtime | acceptable with primary context discipline | better ownership and loading semantics |

Relevant NVIDIA constraints:

- Runtime API code is simpler because it manages the primary context and
  modules implicitly.
- Driver API gives more direct control over module loading and lets the runtime
  keep only the modules it needs loaded.
- A Driver API context owns modules, memory, and actions; a host thread needs
  the right current context for most operations.
- Runtime and Driver API can interoperate through the primary context.

Recommended PTO CUDA split:

- Use Driver API for context retention/currentness and module/library loading.
  This matches `ChipWorker` as a plugin-style host runtime loaded by `dlopen`.
- Use Runtime API only where it materially simplifies ordinary memory/stream
  operations and does not hide module ownership.
- Avoid `cudaDeviceReset` inside the backend. It can invalidate shared primary
  context state used by other libraries in the process.
- Treat one `ChipWorker` as owning one CUDA context attachment, stream set, and
  callable-module cache.

The key design choice is not "Runtime API or Driver API everywhere". It is:
make context/module ownership explicit enough for a `dlopen`-loaded runtime,
while keeping simple memory and stream code readable.

## TileLang JIT Flow

TileLang is useful prior art because it has both ordinary TVM-backed JIT and
an NVRTC backend for fast CUDA JIT execution.

Observed TileLang NVRTC flow:

```text
TileLang Python/TIR function
  |
  | TVM/TileLang lowering
  v
CUDA source string
  |
  | tilelang.contrib.nvrtc.compile_cuda(...)
  v
PTX or cubin bytes
  |
  | NVRTCLibraryGenerator.compile_lib()
  v
temp files:
  kernel.cu
  kernel.cubin
  launcher.py
  |
  | NVRTCLibraryGenerator.load_lib()
  v
CUDA Driver API library/module handle + generated Python launcher
  |
  | generated call(...)
  v
cuda.bindings.driver:
  cuKernelSetAttribute
  cuLaunchKernelEx
  optional TMA descriptors and L2 cache setup
```

TileLang's NVRTC path compiles CUDA source at runtime to PTX or cubin, writes
a generated Python launcher, loads the cubin with CUDA Driver API bindings,
and launches through `cuLaunchKernelEx`. The wrapper also generates setup code
for TMA descriptors, L2 persistence, and launch attributes.

What PTO should copy from TileLang:

- specialize and compile user code close to runtime when launch metadata
  depends on shapes or tuning decisions;
- keep a manifest/launcher layer separate from raw code bytes;
- use Driver API module/kernel handles for dynamically loaded CUDA code;
- cache compiled artifacts and function handles by target architecture and
  source/config digest.

What PTO should not copy directly:

- TileLang generates Python launchers; PTO should keep launch execution inside
  `libhost_runtime.so` because `ChipWorker` already centralizes device
  lifecycle in C++.
- TileLang is operator-kernel oriented; PTO also needs DAG/task runtime
  semantics, callable IDs, `TaskArgs`, and a stable C API boundary.

## Lifecycle, Memory, and Callable Mapping

### Lifecycle

| PTO C API concept | a2a3/a5 implementation | CUDA implementation |
| ----------------- | ---------------------- | ------------------- |
| `create_device_context` | heap-allocate `DeviceRunner` | heap-allocate CUDA `DeviceRunner` |
| `simpler_init` | `rtSetDevice`, set log state | retain/set CUDA context, create streams |
| `finalize_device` | destroy streams, free device resources | unload modules, destroy streams, free memory |
| `destroy_device_context` | delete runner | delete runner and release context ref |
| host runtime `dlopen` | `libhost_runtime.so` per runtime | same ABI, CUDA-specific implementation |

CUDA detail: `simpler_init` should make the CUDA context current for the
calling thread or retain the primary context and make it current before each
operation. This mirrors the existing rule that device operations happen on the
thread that initialized the `ChipWorker`, while still allowing future
thread-local context attachment if the Python worker model needs it.

### Memory

| PTO operation | a2a3/a5 | CUDA |
| ------------- | ------- | ---- |
| `device_malloc_ctx` | `DeviceRunner::allocate_tensor` -> Ascend allocator | `cuMemAlloc` or `cudaMalloc` |
| `device_free_ctx` | Ascend allocator free | `cuMemFree` or `cudaFree` |
| `copy_to_device_ctx` | H2D copy through CANN runtime | `cuMemcpyHtoDAsync` or `cudaMemcpyAsync` |
| `copy_from_device_ctx` | D2H copy through CANN runtime | `cuMemcpyDtoHAsync` or `cudaMemcpyAsync` |
| `ContinuousTensor.data` | device pointer integer | `CUdeviceptr` / device pointer integer |
| `child_memory` | skip host copy for child-owned device ptr | same semantic: pointer already device-owned |

PTO should preserve explicit copy semantics at the `ChipWorker` layer. Unified
memory may be useful later, but it should not be the first backend contract
because current tests/examples reason about explicit allocation and copy.

### Callable

| PTO callable concept | a2a3/a5 | CUDA `host_schedule` | CUDA `persistent_device` |
| -------------------- | ------- | -------------------- | ------------------------ |
| `ChipCallable.binary` | orchestration shared object bytes | module image or host-scheduler metadata | linked executor module image |
| `CoreCallable.binary` | AICore text/object or sim SO | CUDA cubin/PTX/fatbin bytes | task body object or IR |
| `func_id` | runtime dispatch id for AICore kernels | entry-name or kernel-handle id | dispatch-table index |
| `prepare_callable` | upload kernels + orch SO; cache by cid | load module; cache kernel handles | load linked executor and table |
| `run_prepared` | launch scheduler/executor pair | launch one or more CUDA kernels | launch persistent executor |
| `unregister_callable` | drop prepared state/refcounts | unload module when no cid references it | unload module and free table state |

The CUDA callable payload should include enough metadata to avoid guessing at
launch time:

- target architecture and image kind (`ptx`, `cubin`, `fatbin`);
- entry names and stable IDs;
- argument ABI for each entry;
- launch policy defaults;
- optional dynamic shared-memory and cluster attributes;
- source/config digest for cache lookup.

### `CallConfig`

Current `CallConfig` is Ascend-shaped but can remain the wire format during
bring-up:

- `block_dim`: use as a launch-policy hint in `host_schedule`, and as the
  persistent executor block count in the future runtime.
- `aicpu_thread_num`: treat as ignored or as a scheduler-role hint for CUDA;
  document exact behavior per runtime.
- diagnostic flags: map to CUDA-side profiler/tensor dump support only when
  those features exist; otherwise fail with a clear not-supported error rather
  than silently producing incomplete artifacts.

Longer term, add neutral aliases while keeping the binary layout compatible:
for example `worker_blocks` for `block_dim` and `scheduler_lanes` for
`aicpu_thread_num`.

## Design Consequences

- CUDA support should start with `host_schedule`; it matches CUDA's natural
  host-launched model and proves the stable C API.
- `tensormap_and_ringbuffer` should not be ported by mechanically renaming
  AICPU/AICore roles. It needs a persistent-kernel design and a linked device
  dispatch table.
- `RuntimeBinaries` is now target-role based. Ascend still exposes its legacy
  `aicpu`/`aicore` roles, while CUDA exposes `host`, `device`, and optional
  `scheduler` without populating AICPU/AICore aliases. The Python and nanobind
  `ChipWorker` boundary accepts role-keyed maps directly, and the CUDA
  `persistent_device` build publishes a `scheduler` role so scheduler/runtime
  artifacts do not have to pass through AICPU naming. The C++ `ChipWorker` now
  probes the optional `simpler_init_roles` host-runtime ABI and passes CUDA
  `device`/`scheduler` role binaries directly when that entry exists; runtimes
  without it keep the legacy two-binary `simpler_init` path.
- The backend should use Driver API for module/context ownership because PTO
  is a dynamic plugin-like runtime loaded through `dlopen`.
- TileLang validates the NVRTC + Driver API model for dynamic CUDA code, but
  PTO should keep C++ host-runtime ownership rather than generating Python
  launch wrappers.

## Sources

- `simpler_setup/scene_test.py`
- `simpler_setup/kernel_compiler.py`
- `simpler_setup/runtime_builder.py`
- `python/simpler/worker.py`
- `python/simpler/task_interface.py`
- `src/common/worker/chip_worker.cpp`
- `src/common/worker/pto_runtime_c_api.h`
- NVIDIA CUDA Runtime API, Runtime vs Driver API:
  <https://docs.nvidia.com/cuda/cuda-runtime-api/driver-vs-runtime-api.html>
- NVIDIA CUDA Programming Guide, Driver API:
  <https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html>
- NVIDIA CUDA Runtime API, execution control:
  <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html>
- NVIDIA CUDA Runtime API, library management:
  <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__LIBRARY.html>
- NVIDIA CUDA Driver API, module management:
  <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html>
- TileLang NVRTC wrapper docs:
  <https://tilelang.com/autoapi/tilelang/jit/adapter/nvrtc/wrapper/index.html>
- TileLang NVRTC library generator docs:
  <https://tilelang.com/autoapi/tilelang/jit/adapter/nvrtc/libgen/index.html>
- TileLang NVRTC compile helper docs:
  <https://tilelang.com/autoapi/tilelang/contrib/nvrtc/index.html>
