# CUDA Persistent Device Runtime Analysis

This document analyzes the largest CUDA design gap: NVIDIA GPUs do not expose
an AICPU-like device scheduler that can launch independent worker kernels.
That changes both runtime architecture and build architecture.

## Core Constraint

In the a2a3/a5 runtime, the AICPU is a device-side control processor. It can
run scheduler code, observe task readiness, and hand work to AICore workers.
CUDA has no equivalent control processor. A CUDA GPU can run kernels launched
from the host, and CUDA Dynamic Parallelism can launch child grids from device
code, but using child kernel launches as the normal task dispatch mechanism is
not the right default for PTO:

- child launches add CUDA device-runtime state and launch-management overhead;
- parent/child completion is nested, which fights a long-lived scheduler loop;
- stream/event objects created on device have grid scope;
- the device runtime has resource limits that become scheduler limits;
- it still launches grids, not lightweight task functions onto reserved worker
  warps.

The CUDA `persistent_device` runtime should therefore launch one persistent
executor kernel from the host. Scheduler warps/blocks inside that executor
manage ready queues and dispatch tasks by calling linked task functions on
worker warps/blocks.

```text
host
  |
  | cuLaunchKernel(pto_persistent_executor<<<grid, block, smem, stream>>>)
  v
CUDA persistent executor grid
  |
  | scheduler warps/blocks
  |  - build or consume TensorMap/ring metadata
  |  - update fanin/fanout counters
  |  - push ready task descriptors
  v
ready queues in global memory
  |
  | worker warps/blocks
  |  - pop descriptor
  |  - decode func_id + args
  |  - call linked task function through generated dispatch
  |  - publish completion
  v
completion queues/counters in global memory
```

This is not a direct AICPU replacement. The scheduler and worker roles share
the same CUDA grid and compete for SM resources. The runtime must decide how
many blocks/warps are scheduler roles and how many are workers.

## Host-Schedule Runtime

The first CUDA runtime should be named `host_schedule`, not `host_build_graph`.
The current Ascend `host_build_graph` name describes graph construction. For
CUDA the essential behavior is broader: the host owns scheduling and uses CUDA
streams to enqueue task kernels.

### Stream Semantics

A CUDA stream is an ordered queue. Work issued to the same stream executes in
issue order. Work issued to different streams may overlap only when:

- the device supports concurrent kernel execution;
- the kernels do not consume all execution resources;
- there is no dependency from events, synchronization, default-stream barriers,
  blocking allocations, or other implicit synchronization;
- the host enqueues independent work before waiting on it.

Therefore `host_schedule` needs multiple non-blocking streams if it wants
concurrent kernel execution. A single stream is correct but serializes all
task kernels.

Recommended `host_schedule` design:

```text
host scheduler
  |
  | ready task queue
  v
stream pool
  |  stream[0]  task A kernel -> event A
  |  stream[1]  task B kernel -> event B
  |  stream[2]  task C kernel -> event C
  |
  | dependencies represented by cudaEventRecord/cudaStreamWaitEvent
  v
worker.run returns after synchronizing all streams touched by this run
```

Rules:

- Create streams with `cudaStreamNonBlocking` or equivalent Driver API flags.
- Do not use the legacy default stream for task kernels.
- Carry the selected stream in the prepared callable manifest. The current
  bring-up ABI uses `PtoCudaHostCallable.version == 2` with `stream_id`, while
  version 1 callables remain mapped to stream 0.
- Track one event per task completion or per stream tail.
- Use `cudaStreamWaitEvent` to express dependencies between streams.
- Delay host synchronization until the existing `run_prepared` boundary unless
  a user-facing API requires earlier completion.
- Size the stream pool separately from `block_dim`; `block_dim` is a kernel
  launch-policy hint, not a stream count.

`host_schedule` can provide concurrency for independent ready tasks, but it is
still host-dispatched. It pays host launch overhead per task and cannot match
the device-side dispatch latency target of `persistent_device`.

## Persistent Device Runtime

The `persistent_device` runtime is the CUDA analogue of
`tensormap_and_ringbuffer`, not a port of the AICPU implementation.

The first implementation slice in this branch is intentionally a tracer
bullet: `persistent_device` is registered as a CUDA runtime and can launch a
single executor kernel that consumes a device array of vector-add task
descriptors. It proves the build/discovery path, module loading, descriptor
memory layout, and "one host launch handles multiple device tasks" shape before
adding TensorMap/ring queues and generated dispatch tables.

The next implementation slice adds a bounded ready ring for the same vector-add
task descriptor. One scheduler block publishes task IDs into global memory and
worker blocks pop those IDs through atomics inside the same executor launch.
The queue uses per-slot sequence flags so a capacity smaller than the task
count can wrap without a later ticket consuming an earlier slot value. This
still is not the final TensorMap/ring runtime, but it exercises the
scheduler/worker split and back-pressure shape that CUDA needs because there
is no AICPU.

The following slice layers a small DAG on top of that bounded ring. Task
descriptors carry a `func_id`, dependent ranges, an initial fan-in count, and
optional tensor shape/stride metadata for tiled callables. The persistent
executor seeds zero-fan-in tasks, dispatches task bodies through a generated
`func_id` switch, decrements dependent fan-in counters when tasks complete,
pushes newly ready dependents back into the ring, and keeps one scheduler
block alive to report malformed graphs that exhaust ready work before every
task completes. The smoke path now uses
`KernelCompiler(platform="cuda").compile_cuda_persistent_device(...)` to
generate the shared task-body wrappers and dispatch switch before compiling
the executor source with `nvcc`. The compiler writes the generated source, PTX,
and JSON manifest under
`build/cache/cuda/onboard/persistent_device/callables/`, matching the intended
per-callable artifact layout.

### Runtime Roles

```text
Persistent executor grid

block 0..S-1:
  scheduler role
  - drain submit/wiring queues
  - compute ready tasks
  - push task descriptors into device ready queues

block S..N-1:
  worker role
  - pop descriptors
  - call generated dispatch(func_id, args)
  - publish completion records
```

The roles can also be assigned by warp instead of block. Warp roles are more
flexible but harder to reason about because scheduler and worker warps share a
CTA's synchronization and shared memory. Block roles are the safer first
implementation.

### Dispatch Without Child Kernels

The executor should not launch `__global__` task kernels from device code.
Instead it calls task functions compiled into the same device image:

```cuda
using PtoTaskFn = void (*)(PtoTaskContext *);

__device__ void vector_add_task(PtoTaskContext *ctx) {
    // user task body
}

__device__ void pto_dispatch(uint32_t func_id, PtoTaskContext *ctx) {
    switch (func_id) {
    case 0:
        vector_add_task(ctx);
        return;
    default:
        pto_fail_unknown_func(func_id);
    }
}

__global__ void pto_persistent_executor(PtoRuntimeState *state) {
    if (is_scheduler_block()) {
        pto_scheduler_loop(state);
    } else {
        pto_worker_loop(state, pto_dispatch);
    }
}
```

A generated `switch` is the recommended first dispatch mechanism. A table of
`__device__` function pointers is possible on CUDA, but it is harder to debug,
less friendly to optimization, and more sensitive to relocation/linking
details. A generated switch also lets `nvcc` inline simple task bodies when
whole-program or LTO compilation can see them.

## One Kernel Body Across Runtimes

The user concern is correct: asking users to write one `__global__` kernel for
`host_schedule` and another `__device__` function for `persistent_device` would
split the programming model.

The proposed contract is: users write a PTO CUDA task body, and the build
system generates runtime-specific wrappers.

```cuda
PTO_DEVICE_TASK(vector_add, PtoTaskContext *ctx) {
    float *a = pto_arg<float *>(ctx, 0);
    float *b = pto_arg<float *>(ctx, 1);
    float *out = pto_arg<float *>(ctx, 2);
    int i = pto_linear_tid();
    if (i < pto_numel(ctx)) {
        out[i] = a[i] + b[i];
    }
}
```

Generated for `host_schedule`:

```cuda
extern "C" __global__ void pto_kernel_vector_add(PtoTaskContext ctx) {
    vector_add(&ctx);
}
```

Generated for `persistent_device`:

```cuda
__device__ void pto_task_vector_add(PtoTaskContext *ctx) {
    vector_add(ctx);
}
```

The exact macro names can change, but the boundary should hold: task body once,
wrappers generated per runtime. `PtoTaskContext` is the common ABI that hides
whether arguments came from `ChipStorageTaskArgs`, a host-scheduled launch
manifest, or a persistent device descriptor.

The current branch has the first codegen slices for that boundary:
`simpler_setup.cuda_callable_compiler.render_cuda_task_wrappers()` renders a
source fragment with one `pto_task_body_<name>` function, one
`pto_kernel_<name>` `__global__` wrapper for `host_schedule`, and one
`pto_task_<name>` `__device__` wrapper for `persistent_device`.
`KernelCompiler(platform="cuda").compile_cuda_host_schedule()` now uses this
renderer for the host-schedule side and writes a cached PTX artifact plus
manifest. It can also generate a host wrapper whose parameters match the
current host-schedule vector-add launch ABI, while the shared task body still
receives `PtoTaskContext *`.

`KernelCompiler(platform="cuda").compile_cuda_persistent_device()` now exposes
the persistent-device generated-dispatch compiler through the same public
compiler object. It accepts task source files plus `func_id` metadata, can
lower those files through the same `CudaTaskBody` wrapper contract as
`host_schedule`, writes the generated dispatch source, PTX, and manifest under
the persistent-device callable cache, and is used by the DAG smoke/evaluation
path. `prepare_cuda_persistent_device_callable()` turns the compiled artifact
into the ctypes manifest consumed by the current host runtime
`prepare_callable` C API, including the callable `stream_id`. L2
`Worker.register(...)` can stage that raw manifest blob, and L2
`Worker.run(...)` can launch backend-specific raw argument structs on the
selected CUDA runtime stream. The normal scene-test compiler and run path can
now consume
host-schedule CUDA callable specs and persistent-device generated-dispatch
DAG specs. The current adapters construct `vector_add_f32`,
`elementwise_binary_f32`, `elementwise_unary_f32`,
`elementwise_scale_f32`, `elementwise_axpy_f32`,
`persistent_dag_fork_join_f32`, `persistent_dag_chain_f32`,
`persistent_dag_reuse_f32`, `persistent_dag_scalar_axpy_f32`,
`persistent_dag_tensor_tile_f32`, `persistent_dag_triad_f32`,
`persistent_dag_quad_f32`, `persistent_dag_generic_args_f32`,
`persistent_dag_graph_f32`, `persistent_dag_unary_square_f32`, and
`persistent_dag_tensor_core_tile_f32` raw argument/state structs from
`TaskArgsBuilder` CPU tensors and scalars. This covers the current
tracer-bullet argument families: unary, binary, triad, quad, fixed scalar
fields, bounded generic tensor/scalar slots, graph descriptors, scalar tensor
tiles, and a WMMA tensor-core tile. The remaining work is general lowering
from normal PTO task graphs into those CUDA descriptor families, plus broader
tuned tensor kernels beyond the current smoke and microbenchmark bodies.

The descriptor now also carries bounded generic argument slots:
`tensor_args[4]`, `scalar_args[4]`, `tensor_arg_count`, and
`scalar_arg_count`. The current `generic_args` smoke and
`persistent_dag_generic_args_f32` SceneTestCase adapter use those slots to pass
two auxiliary tensors and two scalars to a generated-dispatch task body
without adding more fixed `c`/`d`-style fields. This is still an explicit
descriptor adapter rather than full PTO graph lowering, but it sets the ABI
direction for persistent tasks whose arity differs from the early hand-coded
vector cases. The `persistent_dag_graph_f32` adapter can now lower an
explicit scene-test graph descriptor with per-task dependencies, fan-in,
temporary buffers, and the same generic tensor/scalar slots. Graph task
outputs that do not name existing input/output tensors are allocated as
default-sized temporary buffers automatically, so explicit `temporaries`
metadata is only needed for non-default sizes. When graph tasks omit
`dependents`, the adapter infers edges from tensor flow by tracking which
earlier task produced a tensor or temporary read later through
`a`/`b`/`c`/`d` or `tensor_args`. This inference is per task: explicit
`dependents` remain authoritative for tasks that provide them, while omitted
task edges are inferred from tensor flow. It is a descriptor-level stepping
stone toward PTO graph lowering, not yet automatic construction from normal
PTO task graphs.

## Static NVCC Linking Feasibility

The stable path is feasible with ordinary `nvcc`, but it changes what "runtime
binary" means for `persistent_device`.

CUDA supports separate compilation of device code. Device functions can call
device functions and access device variables from other compilation units when
all CUDA translation units are compiled with relocatable device code enabled
(`-dc` or `-rdc=true`) and then device-linked. This gives PTO a stable offline
path:

```text
build reusable runtime object/archive:
  pto_persistent_executor.cu
  pto_scheduler.cu
  pto_runtime_state.cu
    -> nvcc -dc ... -> libpto_cuda_persistent_device.a

per callable:
  user_tasks.cu
  user_orchestrator_device.cu
  generated_dispatch.cu       # rendered from task func_id/name/body metadata
  generated_manifest.cu
    -> nvcc -dc ...
    -> nvcc device link with libpto_cuda_persistent_device.a
    -> cubin or fatbin

prepare_callable:
  -> host runtime loads final cubin/fatbin
  -> copies manifest/runtime metadata to device

run_prepared:
  -> launches pto_persistent_executor from that final module
```

The cost is that `persistent_device` needs a per-callable device link step.
Unlike Ascend, the fully linked scheduler/worker executor cannot be entirely
prebuilt independently of user task code if worker code calls user functions
directly.

That is acceptable if the build system makes the split explicit:

- `RuntimeBuilder` builds reusable host runtime and reusable CUDA device object
  archives.
- `KernelCompiler` or a CUDA-specific callable compiler compiles user tasks,
  generated dispatch, and optional device orchestration.
- The callable compiler links those objects with the runtime archive into the
  final module image stored in the callable payload.

## Orchestrator Separation

The hardest part is the orchestrator. In today's Ascend runtime, orchestration
can be a separately compiled payload. For CUDA `persistent_device`, anything
that runs on the GPU must be compiled into the final device module before the
persistent executor is launched.

There are three viable models.

### Model A: Host Orchestrator, Device Scheduler

The host builds task descriptors and dependencies, then the persistent executor
only schedules ready work on device.

Pros:

- keeps existing orchestration `.so` style closer to today;
- easier first persistent executor;
- avoids device compiling user orchestration code.

Cons:

- not equivalent to `tensormap_and_ringbuffer` if the goal is device-side graph
  construction;
- host still pays graph-build cost;
- dynamic dependencies discovered on device are not supported.

### Model B: Device Orchestrator Linked Into Executor

The user orchestration code is CUDA device code. The callable compiler links
it into the same final module as the scheduler, worker runner, task bodies,
and generated dispatch.

Pros:

- closest CUDA analogue to device-side `tensormap_and_ringbuffer`;
- all graph build and scheduling state can live in device global memory;
- no device-side dynamic linking needed after launch.

Cons:

- orchestration source must obey CUDA device-code restrictions;
- per-callable device link is mandatory;
- Python/C++ codegen must generate a device entry and manifest;
- debugging is harder than host orchestration.

### Model C: Host Submits Descriptors Incrementally

The host runs orchestration and pushes task descriptors into a device queue
while a persistent scheduler is already running.

Pros:

- keeps user orchestration on host;
- scheduler/worker latency after submit is device-side;
- supports streaming workloads.

Cons:

- host/device queue protocol is more complex;
- correctness depends on host/device memory ordering and queue back-pressure;
- still not full device-side graph construction.

Recommended order:

1. `host_schedule` for correct CUDA execution and stream concurrency.
2. `persistent_device` Model A or C as a tracer bullet for device worker
   dispatch.
3. `persistent_device` Model B only after the task ABI and scheduler queues are
   stable.

## Build and Architecture Changes

The current architecture assumes three target roles: `host`, `aicpu`,
`aicore`. CUDA needs runtime-specific target roles:

```python
BUILD_CONFIG = {
    "host": {...},
    "device_archive": {...},      # reusable persistent runtime objects
    "host_schedule": {...},       # optional host-schedule support module
}
```

Recommended build artifacts:

```text
build/lib/cuda/onboard/host_schedule/
  libhost_runtime.so
  libsimpler_log.so

build/lib/cuda/onboard/persistent_device/
  libhost_runtime.so
  libpto_cuda_persistent_device.a
  pto_persistent_manifest_schema.json
```

Per-callable build cache:

```text
build/cache/cuda/onboard/persistent_device/callables/{hash}/
  user_tasks.o
  generated_dispatch.o
  generated_manifest.o
  pto_callable.fatbin
  pto_callable.json
```

Required Python/build changes:

- continue replacing the global `TARGETS = ("host", "aicpu", "aicore")`
  assumption with per-runtime target discovery;
- finish generalizing `RuntimeBinaries`; the current transitional
  `role_paths` map exposes Ascend's legacy roles and CUDA's native `host` /
  `device` roles while preserving legacy path attributes;
- add a CUDA callable compiler that owns wrapper generation and final device
  link;
- add manifest fields to `ChipCallable` or introduce a CUDA callable payload
  format understood by the CUDA host runtime;
- keep `ChipWorker` C API stable unless the binary path generalization forces a
  wrapper-level compatibility shim.

## NVRTC and nvJitLink Position

NVRTC plus nvJitLink can support runtime specialization, but they should not be
the first persistent-device foundation. The offline `nvcc` path is easier to
make reproducible because it uses the same compiler/link flow developers run
locally and in CI.

Use NVRTC/nvJitLink later only if there is a clear need:

- shape-specialized kernels generated at runtime;
- autotuned schedules selected after install;
- avoiding full `nvcc` invocation latency for small generated tasks.

Until then, use:

- `nvcc -dc` / `-rdc=true` for separable device objects;
- normal device link to produce cubin/fatbin;
- optional device LTO once correctness and link structure are stable.

## Sources

- NVIDIA CUDA Programming Guide, streams and overlapping behavior:
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>
- NVIDIA CUDA Dynamic Parallelism:
  <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html>
- NVIDIA CUDA Programming Guide, NVCC and separate compilation:
  <https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html>
- NVIDIA CUDA Programming Guide, language execution specifiers:
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>
