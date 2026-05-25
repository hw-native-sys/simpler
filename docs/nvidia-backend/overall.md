# NVIDIA CUDA Backend Design

This document proposes a CUDA backend for the PTO runtime. It is a design
target, not an implementation plan. The goal is to fit NVIDIA GPUs into the
existing `simpler` / PTO runtime architecture without leaking CUDA-specific
details into the Python user API.

For detailed compile/launch flow comparisons, runtime flow diagrams, CUDA
runtime-vs-driver notes, TileLang JIT notes, and lifecycle/memory/callable
mapping, see [flows.md](flows.md). For the AICPU gap, CUDA stream semantics,
and the persistent device scheduler design, see
[persistent-device.md](persistent-device.md). For current real-device smoke
and microbenchmark results, see [evaluation.md](evaluation.md).

## Current Runtime Shape

The current chip backend has three separately built programs:

- `libhost_runtime.so`: loaded by `ChipWorker` through `dlopen`; owns device
  context, device memory, copies, profiling hooks, and launch orchestration.
- `libaicpu_kernel.so`: device-side scheduler running on Ascend AICPU.
- `aicore_kernel.o` or sim `.so`: device worker executor that receives task
  dispatches and invokes user kernels.

The build and lookup path is:

1. `simpler_setup.platform_info` maps a platform name to `(arch, variant)`.
2. `RuntimeBuilder` discovers `src/{arch}/runtime/*/build_config.py`.
3. `RuntimeCompiler` builds `host`, `aicpu`, and `aicore` targets.
4. `ChipWorker` loads the host runtime and passes the device binaries through
   the stable C API in `src/common/worker/pto_runtime_c_api.h`.

Runtime implementations are independent of platform implementations. Platform
code lives under `src/{arch}/platform/`, while runtime graph/scheduler code
lives under `src/{arch}/runtime/`. The two current runtime variants are:

- `host_build_graph`: host builds the task graph; useful for bring-up.
- `tensormap_and_ringbuffer`: production runtime where AICPU-side
  orchestration derives dependencies and dispatches tasks to AICore workers.

L3+ hierarchical scheduling is separate from the L2 device backend. It sees
`ChipWorker` as an `IWorker` leaf and passes `Callable`, `TaskArgsView`, and
`CallConfig` through the same interface regardless of device vendor.

## CUDA Constraints That Matter

CUDA has a different control model from Ascend:

- CUDA systems are heterogeneous: host CPU code uses CUDA APIs to allocate
  device memory, copy data, launch GPU kernels, and synchronize.
- A GPU consists of Streaming Multiprocessors (SMs). Kernels launch many
  threads organized into thread blocks and grids. Blocks are scheduled onto
  SMs, and independent blocks cannot rely on a global execution order.
- Threads execute in warps of 32 lanes under SIMT. Blocks share fast on-chip
  shared memory; global memory is visible across the device.
- CUDA launches and copies are asynchronous. Streams express ordered queues of
  operations, and synchronization is explicit.
- Offline compilation uses `nvcc`, which separates host and device code,
  emits PTX, assembles target-specific cubins with `ptxas`, and can package
  multiple PTX/cubin images into a fatbin.
- The Driver API can load PTX, cubin, or fatbin modules and resolve kernel
  handles. PTX is the forward-compatible form; cubin is tied to an SM target.
- NVRTC and nvJitLink enable runtime compilation and linking. They are useful
  for user kernels, but they add toolkit/runtime compatibility constraints.

The biggest architectural mismatch is the missing AICPU equivalent. NVIDIA
does not expose a separate device control CPU that can `dlopen` orchestration
code, write hardware dispatch registers for another core type, and invoke
arbitrary separately compiled global kernels. The CUDA backend should treat
that as a hard boundary and avoid copying the AICPU/AICore split literally.

## Naming and Scope

Use `cuda` as the backend name in code and platform strings:

- `src/cuda/platform/`
- `src/cuda/runtime/`
- platform: `cuda`
- optional future host-only simulator: `cudasim`

Use "NVIDIA backend" in prose when referring to hardware/vendor support, and
`cuda` for concrete paths, build targets, and CLI platform names. The Python
package name remains `simpler`.

Phase 1 should target a single NVIDIA GPU on one host. Multi-GPU and
multi-node communication should be a later phase using NCCL or CUDA IPC behind
the existing `comm_*` C API surface.

## Proposed Backend Shape

### Phase 1: Host-Scheduled CUDA Backend

Bring up a CUDA runtime named `host_schedule` first. It is analogous to the
current `host_build_graph` runtime, but the name should describe the CUDA
execution model directly: the host owns graph scheduling and enqueues CUDA
kernel work.

The CUDA host runtime exports the same C API that `ChipWorker` already loads:

- lifecycle: `create_device_context`, `simpler_init`, `finalize_device`
- memory: `device_malloc_ctx`, `device_free_ctx`, `copy_to_device_ctx`,
  `copy_from_device_ctx`
- callable lifecycle: `prepare_callable`, `run_prepared`,
  `unregister_callable`
- diagnostics and comm symbols, initially as no-op or not-supported stubs

The CUDA `DeviceRunner` maps these onto CUDA runtime or driver APIs:

- `cudaSetDevice` or Driver API primary-context retention during
  `simpler_init`
- `cudaMalloc` / `cudaFree` or `cuMemAlloc` / `cuMemFree`
- `cudaMemcpyAsync` / `cuMemcpy*Async` on a runner-owned stream
- `cudaStreamSynchronize` or event synchronization at `run_prepared` boundary
- module loading through Driver API `cuModuleLoadDataEx` or CUDA runtime
  dynamic loading when the minimum toolkit supports it

`prepare_callable` compiles or loads user CUDA device code and caches module
state by `callable_id`. `run_prepared` launches one or more CUDA kernels from
host according to the prepared host graph.

This phase intentionally avoids device-side graph construction. It gives the
project a small compatibility slice:

- Python `Worker` / `ChipWorker` remains unchanged.
- `TaskArgs` and `CallConfig` stay the ABI boundary.
- Tensor memory allocation/copy paths become testable on a real CUDA device.
- User kernels can be expressed as CUDA global kernels before optimizing for
  persistent device-side scheduling.

### Phase 2: Persistent Device Runtime

Porting `tensormap_and_ringbuffer` requires a CUDA-native scheduling model.
The CUDA runtime should be named `persistent_device`. The recommended shape is
a persistent scheduler/worker kernel:

1. Host builds or uploads a runtime descriptor in global memory.
2. Host launches a persistent CUDA kernel with a configured grid.
3. Blocks or warps act as scheduler and worker roles inside that one kernel.
4. Workers call linked task functions through a generated dispatch table
   instead of launching arbitrary global kernels from device code.
5. Completion, dependency counters, and ready queues live in global memory.

This is closer to the existing AICPU/AICore design than host scheduling, but
the dispatch target changes. The scheduler must be part of the CUDA device
binary, and user compute must be compiled into task functions callable by the
persistent executor. See [persistent-device.md](persistent-device.md) for the
static `nvcc` linking design and alternatives.

The first implementation should prefer normal offline `nvcc` compilation and
device linking. NVRTC plus nvJitLink can remain a later optimization path for
autotuning or shape-specialized runtime compilation, but they should not be the
foundation for the initial persistent runtime.

## Directory Design

Add CUDA as a new architecture root rather than a variant under `a5`:

```text
src/cuda/
  docs/
    platform.md
    runtimes.md
  platform/
    include/
      common/
      host/
      cuda/
    onboard/
      host/
      device/
    sim/
      host/
      device/
    src/
      host/
  runtime/
    host_schedule/
      build_config.py
      host/
      runtime/
      orchestration/
    persistent_device/
      build_config.py
      host/
      device/
      runtime/
      orchestration/
```

The `onboard` variant means real CUDA hardware. The `sim` variant should only
be added if the project needs CPU-only CI for CUDA-specific code paths. It is
not required for phase 1 because CUDA compilation and module loading are the
main risk.

## Build System Changes

Platform discovery needs a third arch:

```python
PLATFORM_MAP = {
    "a2a3": ("a2a3", "onboard"),
    "a2a3sim": ("a2a3", "sim"),
    "a5": ("a5", "onboard"),
    "a5sim": ("a5", "sim"),
    "cuda": ("cuda", "onboard"),
}
```

The current `TARGETS = ("host", "aicpu", "aicore")` is Ascend-specific. CUDA
should introduce target roles that describe responsibilities rather than
hardware names:

- `host`: `libhost_runtime.so`, always present.
- `device`: CUDA cubin/fatbin/PTX for the executor or persistent runtime.
- `scheduler`: optional alias for the `persistent_device` scheduler module.

For a low-risk transition, `RuntimeBinaries` can keep `aicpu_path` and
`aicore_path` temporarily while CUDA ignores or repurposes the `aicpu` slot.
The cleaner end state is a `TargetBinaries` map keyed by target role, with
Ascend build configs declaring `host`, `aicpu`, `aicore` and CUDA declaring
`host`, `device`.

Add a `CudaNvccToolchain`:

- discovers `nvcc` from `CUDA_HOME`, `CUDA_PATH`, or `PATH`;
- accepts explicit SM targets, for example `sm_80`, `sm_90`, `sm_100`;
- can emit `--fatbin`, `--cubin`, or `--ptx`;
- forwards host compiler choice similarly to existing GCC handling.

Add an optional `CudaNvrtcToolchain` only when runtime compilation is needed.
Phase 1 can start with offline `nvcc` for deterministic builds.

`build_runtimes.py` should detect `cuda` when:

- `nvcc` is available for offline builds; and
- either a CUDA-capable device is visible or the requested phase only compiles
  fatbins without loading them.

## Kernel Compilation Contract

Today, `KernelCompiler.compile_incore()` assumes AIC/AIV source and returns a
binary uploaded to the device runtime. CUDA needs a separate path:

- `host_schedule`: compile one task body plus a generated `__global__`
  wrapper; the host runtime launches that wrapper by name.
- `persistent_device`: compile the same task body plus a generated device
  dispatch entry; the persistent executor calls it from worker warps/blocks.

Recommended public additions:

- `compile_cuda_host_schedule(source_path, entry_name, sm_targets)`
- `compile_cuda_persistent_device(sources, dispatch_manifest, sm_target)`

The existing `ChipCallable` can remain the language-level handle, but its
CUDA payload should include:

- module image bytes;
- entry names or lowered names;
- per-function argument layout;
- optional SM/compute target metadata;
- cache key based on source digest, compiler flags, and target SMs.

## Runtime Semantics Mapping

| Existing concept | CUDA `host_schedule` mapping | CUDA `persistent_device` mapping |
| ---------------- | ---------------------------- | -------------------------------- |
| AICPU scheduler | Host code in `libhost_runtime.so` | Scheduler warps/blocks in executor |
| AICore worker | CUDA global kernel wrapper | Worker warps/blocks in executor |
| `block_dim` | grid/block policy hint | persistent block count |
| `aicpu_thread_num` | ignored or scheduler stream count | scheduler-role count |
| kernel upload | `cuModuleLoadDataEx` / runtime library load | same, plus dispatch table |
| task completion | stream/event synchronization | global-memory completion queue |
| device graph build | not in phase 1 | CUDA global-memory TensorMap/ring |

`CallConfig` should gain CUDA-neutral names before the backend matures. The
current `aicpu_thread_num` name can be accepted for compatibility, but CUDA
documentation should call it a scheduler-worker hint and define exact behavior
per runtime.

## Testing Strategy

Phase 1 tests:

- Python unit tests for platform discovery and `cuda` build-target selection.
- C++ unit tests for CUDA host runtime stubs where CUDA headers are available.
- A smoke scene test that allocates two tensors, launches vector add, copies
  results back, and validates output on one CUDA device.
- A no-device test path that verifies `cuda` is skipped with a clear message
  when `nvcc` or the driver is missing.

Phase 2 tests:

- persistent executor unit test with a synthetic ready queue;
- dispatch-table test that invokes two different user functions by id;
- stress test for fanin/fanout counters and queue wraparound;
- comparison against `host_schedule` for the same DAG.

CI should initially treat CUDA as optional and report skipped tests clearly.

## Open Decisions

- Whether phase 1 should use CUDA Runtime API, Driver API, or a thin mix. The
  Driver API is better aligned with dynamic module loading; Runtime API is
  simpler for memory and streams.
- Whether CUDA user kernels should be `.cu` files beside existing examples or
  generated from a portable PTO kernel DSL.
- Whether `RuntimeBinaries` should be generalized before CUDA lands, or
  whether CUDA should temporarily fit into the existing three-path ABI.
- Which SM targets are required for the first deployment environment.
- Whether CUDA Graphs should be used for repeated `host_schedule` launches once
  correctness is established.

## Recommended First Implementation Slice

1. Add platform discovery for `cuda` behind toolchain detection.
2. Add `CudaNvccToolchain` and host/device target compilation.
3. Add `src/cuda/platform/onboard/host` exporting the existing C API.
4. Implement memory allocation, copies, init/finalize, and no-op comm stubs.
5. Port `host_schedule` only.
6. Add one vector-add scene test and skip it when CUDA is unavailable.
7. Revisit `RuntimeBinaries` and device-object packaging before starting
   `persistent_device`.

This keeps the first branch focused on proving the ABI and build integration.
It postpones the hardest CUDA-specific scheduling work until the project has a
real CUDA `ChipWorker` path running end to end.

## Sources

- NVIDIA CUDA Programming Guide v13.2:
  <https://docs.nvidia.com/cuda/cuda-programming-guide/>
- CUDA programming model, hardware model, threads, memory:
  <https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html>
- CUDA asynchronous execution, streams, and synchronization:
  <https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html>
- NVCC workflow and PTX/cubin/fatbin model:
  <https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html>
- NVIDIA CUDA Compiler Driver documentation:
  <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html>
- CUDA Driver API module loading:
  <https://docs.nvidia.com/cuda/archive/13.1.0/cuda-driver-api/group__CUDA__MODULE.html>
- NVRTC runtime compilation:
  <https://docs.nvidia.com/cuda/archive/11.1.0/nvrtc/index.html>
- nvJitLink runtime device-code linking:
  <https://docs.nvidia.com/cuda/archive/13.0.2/nvjitlink/index.html>
- CUDA runtime dynamic loading APIs:
  <https://developer.nvidia.com/blog/dynamic-loading-in-the-cuda-runtime/>
- NVIDIA NCCL documentation for future multi-GPU communication:
  <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/>
