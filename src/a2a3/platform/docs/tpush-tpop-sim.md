# TPUSH/TPOP Support in a2a3 Simulation

This document summarizes the CPU_SIM context plumbing used by TPUSH/TPOP in
`a2a3sim`, including the host-side follow-up that replaced `thread_local`
storage with a pthread-backed per-thread context.

## Problem

`TPUSH/TPOP` depends on CPU_SIM being able to identify:

- the current execution context
- the logical task that owns the pipe state
- the shared storage used by the producer and consumer

The original sim path did not provide a stable task identity. Using reused
payload addresses as the key was not reliable, so mixed AIC/AIV workloads could
not rendezvous correctly in simulation.

## Solution

The simulation solution has three parts:

1. Add a host-side CPU_SIM context module in `src/a2a3/platform/sim/host/cpu_sim_context.cpp`
2. Publish a stable per-dispatch `task_cookie` through platform hooks keyed by `(core_id, reg_task_id)`
3. Inject execution context and `task_cookie` into CPU_SIM before AICore kernel execution

`task_cookie` is taken from `slot_state.task->task_id.raw`.

The host-side CPU_SIM context still has per-thread semantics, but it no longer
uses C++ `thread_local`. The current implementation stores
`CpuSimExecutionContext` behind a `pthread_key_t`, so host runtime reloads do
not depend on ELF TLS descriptors.

## Implementation

The runtime flow is:

1. AICPU builds the dispatch payload
2. AICore sim reads the payload before kernel execution
3. AICore sim calls `CPU_SIM_SET_EXECUTION_CONTEXT(...)` and `CPU_SIM_SET_TASK_COOKIE(...)`
4. The values are exposed through host-side hooks for `pto-isa` CPU_SIM to read

On the host side, `pto_cpu_sim_set_*` and `pto_cpu_sim_get_*` resolve the
calling thread's `CpuSimExecutionContext` through `pthread_getspecific()`.

`CPU_SIM_SET_EXECUTION_CONTEXT(...)` sets exactly three values:

- logical block index
- AIV lane id inside the cluster
- subblock dimension

`task_cookie` is not part of `CPU_SIM_SET_EXECUTION_CONTEXT(...)`. It is set
separately by `CPU_SIM_SET_TASK_COOKIE(...)`.

## Runtime Changes

- AICPU writes the logical block index/count into `LocalContext`
- AICPU publishes `task_cookie` through `platform_set_cpu_sim_task_cookie(core_id, reg_task_id, ...)`
- AICPU initializes per-core `GlobalContext.sub_block_id` once during executor init
- AICore sim reads the payload, resolves `task_cookie` through `platform_get_cpu_sim_task_cookie(core_id, reg_task_id)`,
  and then calls the CPU_SIM hooks before running the kernel
- Sim `DeviceRunner` clears CPU_SIM shared state at run start and finalize to avoid cross-run leakage
- The host CPU_SIM module resets its pthread-backed execution-context key at the same run boundaries
- Onboard builds keep the same interface as no-ops, so hardware behavior does not change

More specifically:

- logical block index
  - prepared in `build_payload()`
  - source is `slot_state.next_block_idx`
  - stored in `LocalContext`
  - field name is `block_idx`
- logical block count
  - prepared in `build_payload()`
  - source is `slot_state.block_num`
  - stored in `LocalContext`
  - exposed to kernels by `get_block_num(args)`
- `sub_block_id`
  - prepared once in `init()` of `AicpuExecutor`
  - stored in `PTO2DispatchPayload::global_context.sub_block_id`
  - `aiv0 = 0`, `aiv1 = 1`
- `subblock_dim`
  - not stored in the payload
  - computed in `aicore_executor.cpp` when dispatching to CPU_SIM
  - value is `PLATFORM_AIV_CORES_PER_BLOCKDIM` for AIV and `1` for AIC
- `task_cookie`
  - prepared in `dispatch_subtask_to_core()`
  - source is `slot_state.task->task_id.raw`
  - published through platform hooks keyed by `(core_id, reg_task_id)`
  - passed separately through `CPU_SIM_SET_TASK_COOKIE(...)`

## Host Context Storage

The host CPU_SIM module keeps the following fields together in one
`CpuSimExecutionContext` object:

- `block_idx`
- `subblock_id`
- `subblock_dim`
- `task_cookie`

The storage model is:

1. Lazily create one `pthread_key_t` for the host runtime process
2. Allocate one `CpuSimExecutionContext` per worker thread on first use
3. Read and write that context only through `pto_cpu_sim_set_*` and `pto_cpu_sim_get_*`
4. Reset the key during `DeviceRunner::run()` start and `DeviceRunner::finalize()`

This keeps the original per-thread behavior required by mixed AIC/AIV
execution, but avoids relying on C++ `thread_local` in a `.so` that is loaded
and unloaded across runs.

## File Structure

The implementation is split by responsibility:

- Host CPU_SIM context
  - `src/a2a3/platform/sim/host/cpu_sim_context.cpp`
  - exports `pto_cpu_sim_set_*`, `pto_cpu_sim_get_*`, and shared-storage hooks
  - stores per-thread context with `pthread_key_t`
- Runtime dispatch metadata
  - `src/a2a3/runtime/tensormap_and_ringbuffer/common/intrinsic.h`
- AICPU payload construction
  - `src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`
- AICore sim context injection
  - `src/a2a3/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp`
- Platform hook glue
  - `src/a2a3/platform/include/aicpu/cpu_sim_task_cookie.h`
  - `src/a2a3/platform/sim/aicore/inner_kernel.h`
  - `src/a2a3/platform/onboard/aicore/inner_kernel.h`
- Run-boundary cleanup
  - `src/a2a3/platform/sim/host/device_runner.cpp`

## Validation

The host-side `cpu_sim_context.cpp` implementation was independently checked
after the pthread conversion with direct C++ compilation.

## pto-isa Dependency

The validated `pto-isa` baseline is:

- `882c4db95570dfeaf04e0ee2c0ab32477ed372fc`

This design assumes CPU_SIM provides:

- execution-context hooks
- task-cookie hooks
- shared-storage hooks

So the actual set/get chain is:

`AICPU build_payload/init_global_context` -> `AICore executor` -> `CPU_SIM_SET_EXECUTION_CONTEXT/CPU_SIM_SET_TASK_COOKIE` -> `pto_cpu_sim_set_*` in `cpu_sim_context.cpp`

`pto-isa` CPU_SIM reads the context through these exported hooks.

Newer `pto-isa` versions must be revalidated independently. Runtime-side
context plumbing can be correct while later CPU_SIM behavior changes still
break the example.

## Scope Note

This document is about the host-side CPU_SIM context used by TPUSH/TPOP in
`a2a3sim`. It does not describe every remaining TLS-like variable under
`src/a2a3/platform/sim/`.
