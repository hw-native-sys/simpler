# Dynamic Linking and Thread-Local Storage

This document describes how shared libraries are loaded, symbols are resolved,
and per-thread state is managed across simulation and onboard platforms.

## SO Loading Hierarchy

### Simulation

```text
Python process (ChipWorker)
  |
  dlopen(host_runtime.so, RTLD_GLOBAL)        ← host SO
    |
    +-- DeviceRunner::ensure_binaries_loaded()
    |     |
    |     +-- dlopen(aicpu_sim_XXXXXX, RTLD_NOW | RTLD_LOCAL)    ← AICPU SO (temp file)
    |     |     |
    |     |     +-- dlopen(libdevice_orch_<PID>.so, RTLD_LAZY | RTLD_LOCAL)  ← orch SO (temp file)
    |     |
    |     +-- dlopen(aicore_sim_XXXXXX, RTLD_NOW | RTLD_LOCAL)   ← AICore SO (temp file)
    |
    +-- DeviceRunner::upload_kernel_binary()
          |
          +-- dlopen(kernel_<func_id>_XXXXXX, RTLD_NOW | RTLD_LOCAL)  ← kernel SOs (temp file, per func_id)
```

### Onboard

```text
Python process (ChipWorker)
  |
  dlopen(host_runtime.so, RTLD_GLOBAL)        ← host SO
    |
    +-- DeviceRunner (handle-based, one per ChipWorker)
    |     |
    |     +-- rtMemcpy(aicpu_binary → device HBM)    ← NOT dlopen, binary blob upload
    |     +-- rtRegisterAllKernel(aicore_binary)      ← CANN kernel registration
    |     +-- rtAicpuKernelLaunchExWithArgs(...)       ← device-side execution
    |
    +-- dlopen("libascend_hal.so", RTLD_NOW | RTLD_LOCAL)  ← CANN HAL (profiling only)
```

Key difference: onboard does **not** dlopen AICPU/AICore as host-side SOs.
They are binary blobs uploaded to device memory and executed by CANN runtime.

## RTLD Flags and Rationale

### Host Runtime SO: `RTLD_NOW | RTLD_GLOBAL`

`RTLD_GLOBAL` is **required**. PTO ISA's TPUSH/TPOP instructions (AIC-AIV
data transfer for mix-type kernels) use `dlsym(RTLD_DEFAULT, ...)` internally
to locate shared storage hooks defined in the host SO:

```cpp
// PTO ISA: pto/common/cpu_stub.hpp
inline GetSharedStorageHookFn ResolveSharedStorageHook() {
    static auto hook = reinterpret_cast<...>(
        dlsym(RTLD_DEFAULT, "pto_cpu_sim_get_shared_storage"));
    return hook;
}
```

With `RTLD_LOCAL`, this symbol is not in the global scope. The hook returns
`nullptr`, and TPUSH/TPOP fall back to a `static` local variable per SO.
Since AIC and AIV kernel threads run in different contexts, they get separate
storage instances and deadlock — the producer (TPUSH) writes to one storage,
the consumer (TPOP) waits on another.

**Cross-runtime isolation** (running different runtime SOs sequentially) relies
on `-fno-gnu-unique` to ensure `dlclose` actually unloads the SO. The next
`dlopen` with `RTLD_GLOBAL` then replaces the global symbol scope with the
new runtime's symbols.

### Inner SOs: `RTLD_LOCAL`

All SOs loaded by DeviceRunner (AICPU, AICore, kernel, orchestration) use
`RTLD_LOCAL` to prevent symbol pollution between them. Functions that inner
SOs need from the host SO are passed via explicit function pointer injection
(see "Function Pointer Injection" below).

### Orchestration SO: `RTLD_LAZY | RTLD_LOCAL`

Loaded by the AICPU executor at runtime from a temp file. Uses `RTLD_LAZY`
because not all symbols may be referenced. Communicates with the runtime
through a function pointer table (`PTO2RuntimeOps`), not direct symbol
linkage.

**File path collision**: all runtimes write the orch SO to
`/var/tmp/libdevice_orch_<PID>.so`. Safe in serial execution (each task
dlcloses before the next writes), but would conflict in parallel in-process
execution.

### CANN HAL: `RTLD_NOW | RTLD_LOCAL`

`libascend_hal.so` is loaded only for performance profiling (SVM memory
mapping via `halHostRegister`/`halHostUnregister`). The handle is cached
in a file-scope `g_hal_handle` and never explicitly dlclosed.

## All dlsym(RTLD_DEFAULT) Calls

| Symbol | File | Used by | How it works |
| ------ | ---- | ------- | ------------ |
| `pto_cpu_sim_set_execution_context` | PTO ISA `cpu_stub.hpp` | Kernel `set_execution_context()` | Sim: injected via `set_sim_context_helpers` (bypasses dlsym) |
| `pto_cpu_sim_get_execution_context` | PTO ISA `cpu_stub.hpp` | Kernel `get_block_idx()` etc. | Sim: same injection mechanism |
| `pto_cpu_sim_get_shared_storage` | PTO ISA `cpu_stub.hpp` | TPUSH/TPOP shared state | Requires `RTLD_GLOBAL` on host SO |
| `pto_cpu_sim_get_task_cookie` | PTO ISA `cpu_stub.hpp` | Kernel `get_task_cookie()` | Requires `RTLD_GLOBAL` on host SO |
| `halMemAlloc` / `halMemFree` | Onboard `device_malloc.cpp` | AICPU device memory | Resolved once, cached in statics |
| `halGetDeviceInfoByBuff` | Onboard `host_regs.cpp` | Core validity query | a2a3 only |
| `halMemCtl` | Onboard `host_regs.cpp` | Register address mapping | a2a3 only |
| `halResMap` | Onboard `host_regs.cpp` | Per-core register mapping | a5 only |

The first two are called from AICore SO code (via `inner_kernel.h` macros).
They were converted from `dlsym(RTLD_DEFAULT)` to function pointer injection
through `set_sim_context_helpers()`, so they work under both `RTLD_GLOBAL`
and `RTLD_LOCAL`.

The next two (`get_shared_storage`, `get_task_cookie`) are called from PTO ISA
template code instantiated **inside kernel SOs** — not the AICore SO. Function
pointer injection into the AICore SO cannot reach them. They require the host
SO to be loaded with `RTLD_GLOBAL`.

The HAL symbols are onboard-only. CANN's scheduler process pre-loads
`libascend_hal.so` into the global scope before launching AICPU kernels.

## Function Pointer Injection

To avoid `dlsym(RTLD_DEFAULT)` in inner SOs loaded with `RTLD_LOCAL`,
DeviceRunner passes function pointers after dlopen:

**AICore SO** (`set_sim_context_helpers`):

```text
DeviceRunner → dlsym(aicore_handle, "set_sim_context_helpers")
             → set_helpers(pto_cpu_sim_set_execution_context,
                           pto_cpu_sim_set_task_cookie,
                           platform_get_cpu_sim_task_cookie)
```

**AICPU SO** (`set_aicpu_sim_context_helpers`):

```text
DeviceRunner → dlsym(aicpu_handle, "set_aicpu_sim_context_helpers")
             → set_helpers(platform_set_cpu_sim_task_cookie)
```

These injected function pointers are stored as globals in the respective SOs
and called instead of `dlsym(RTLD_DEFAULT)`.

## Thread-Local Storage

### Design Principle

**No C++ `thread_local` in any SO that gets dlclosed and re-dlopen'd.**
C++ `thread_local` uses ELF TLSDESC on aarch64, which has known issues
with dlclose/re-dlopen cycles in older glibc versions. The sim platform
uses `pthread_key_t` (POSIX TLS) for per-thread state in framework SOs.

### All TLS Variables

| Variable | Storage | SO | Purpose |
| -------- | ------- | -- | ------- |
| `g_reg_base_key` | `pthread_key_t` | AICore SO | Per-core simulated register base address |
| `g_core_id_key` | `pthread_key_t` | AICore SO | Per-core physical core ID |
| `g_device_id_key` | `pthread_key_t` | Sim Context SO (`libcpu_sim_context.so`) | Per-thread device binding (device_id) |
| `g_subblock_id_key` | `pthread_key_t` | Sim Context SO (`libcpu_sim_context.so`) | Per-thread subblock identity (for TPUSH/TPOP) |
| `g_cluster_id_key` | `pthread_key_t` | Sim Context SO (`libcpu_sim_context.so`) | Per-thread cluster identity (for TPUSH/TPOP) |
| `s_orch_thread_idx` | `__thread int` | AICPU SO | Profiling thread index (profiling off by default) |
| `execution_context` | `thread_local` | Kernel SO (PTO ISA) | Per-thread execution context (fallback, cached values only) |
| `NPUMemoryModel::instance` | `thread_local` | Kernel SO (PTO ISA) | Per-core memory model simulation |

### Known Risks

1. **`s_orch_thread_idx`** uses `__thread` (ELF TLS) in the AICPU SO. Could
   cause issues on aarch64 glibc <2.39 if the AICPU SO is dlclosed and
   re-dlopen'd while profiling is enabled. Currently safe because profiling
   is off by default and the variable is only accessed during profiling.

2. **PTO ISA `thread_local`** variables (`execution_context`,
   `NPUMemoryModel::instance`) are in kernel SOs. Kernel SOs are short-lived
   (loaded per task, dlclosed after validation), and each kernel thread is
   freshly created, so stale TLS is not a concern in practice.

## `-fno-gnu-unique`

GCC emits `STB_GNU_UNIQUE` binding for `static` locals in inline/template
functions. glibc marks such SOs as `NODELETE`, making `dlclose` a no-op.
When multiple runtime SOs are loaded sequentially with `RTLD_GLOBAL`, the
first SO's symbols persist and pollute the second.

Applied to all sim compilation paths:

- 6 CMakeLists (host/aicpu/aicore for a2a3 and a5): `$<$<CXX_COMPILER_ID:GNU>:-fno-gnu-unique>`
- `toolchain.py` (GxxToolchain, Aarch64GxxToolchain): appended to compile flags

Additionally, `data_type.h::get_element_size()` uses `constexpr static`
instead of `static` to avoid generating UNIQUE symbols at the source level.

## AicpuExecutor::deinit() and SchedulerContext::deinit()

The AICPU SO contains a file-scope static `AicpuExecutor g_aicpu_executor`,
which holds a `SchedulerContext sched_ctx_` member owning all scheduler
state (core trackers, dispatch payloads, drain state, task counters,
core-transition flags, one-time init coordination, etc.).

When the AICPU SO is dlclosed and re-dlopen'd between tasks, the static is
reconstructed. But when the AICPU SO is **reused** (same runtime, consecutive
tasks), `deinit()` must reset all fields. Responsibilities are split so that
SchedulerContext owns its own teardown:

- `SchedulerContext::deinit()` resets every scheduler-owned field —
  per-core states, payloads, sync-start drain coordination
  (`sync_start_pending` / `drain_worker_elected` / `drain_ack_mask` /
  `pending_task`), task counters, transition flags, worker-id lists,
  core trackers, `cores_total_num_` / `aic_count_` / `aiv_count_`,
  `regs_`, `sched_`, `func_id_to_addr_`, and the `pto2_init_*` flags.
- `AicpuExecutor::deinit()` calls `sched_ctx_.deinit()` first, then resets
  only its own fields: `thread_num_`, `sched_thread_num_`, `orch_to_sched_`,
  `orch_func_`, `orch_args_cached_`, `orch_so_handle_`, `orch_so_path_`,
  `runtime_init_ready_`, and the lifecycle atomics
  (`initialized_`, `init_done_`, `init_failed_`, `finished_`, `thread_idx_`,
  `finished_count_`).

Applies to all 4 runtime executors: a2a3 (hbg, tmr), a5 (hbg, tmr).

## SO Handle Caching and Reuse

### Simulation

| SO | Caching | Lifecycle |
| -- | ------- | --------- |
| Host runtime | `ChipWorker::lib_handle_` | Per-init: dlopen in `init()`, dlclose in `finalize()` |
| AICPU | `DeviceRunner::aicpu_so_handle_` | Per-run: loaded in `ensure_binaries_loaded()`, closed in `unload_executor_binaries()` at end of `run()` |
| AICore | `DeviceRunner::aicore_so_handle_` | Same as AICPU |
| Kernel | `DeviceRunner::func_id_to_addr_` (map by func_id) | Per-task: uploaded in `init_runtime_impl()`, removed in `validate_runtime_impl()` |
| Orchestration | `AicpuExecutor::orch_so_handle_` | Per-run: loaded by orchestrator thread, closed by last thread in `deinit()` |

### Onboard

| Resource | Caching | Lifecycle |
| -------- | ------- | --------- |
| Host runtime | `ChipWorker::lib_handle_` | Per-runtime-group: shared across tasks in same group |
| AICPU binary | `AicpuSoInfo` in DeviceRunner | Per-runtime-group: uploaded to device HBM once, reused |
| AICore binary | `rtRegisterAllKernel` handle | Per-run: registered each `launch_aicore_kernel()` call |
| Kernel binaries | `func_id_to_addr_` (device GM addresses) | Per-task: uploaded to device GM, cached by func_id |
| CANN HAL | `g_hal_handle` (file-scope static) | Process lifetime: loaded once for profiling, never closed |

### Key difference

Onboard caches more aggressively — the DeviceRunner persists
across tasks within a runtime group, and the AICPU binary stays in device memory. Simulation
re-loads AICPU/AICore SOs every `run()` call because the SO's internal
static state (`g_aicpu_executor`) must be fresh for each task when
different tasks have different configurations.

## Execution Lifecycle

### Simulation (in-process, per-task)

```text
ChipWorker.init(host_path, aicpu_path, aicore_path, simpler_log_path, device_id)
  dlopen(libsimpler_log.so, RTLD_GLOBAL)
  simpler_log_init(log_level, log_info_v)    seeds HostLogger before host_runtime
  dlopen(host_runtime.so, RTLD_LOCAL)
  dlsym: create_device_context, destroy_device_context, simpler_init,
         get_runtime_size, prepare_callable, run_prepared, unregister_callable,
         finalize_device
  create_device_context() → DeviceContextHandle
  simpler_init(ctx, device_id, aicpu*, aicpu_size, aicore*, aicore_size)
    DeviceRunner::attach_current_thread(device_id)
      pto_cpu_sim_bind_device(device_id)
      pto_cpu_sim_acquire_device(device_id)
    DeviceRunner::set_executors(aicpu, aicore)         binaries owned by runner

ChipWorker.run(cid, args, config)
  run_prepared(ctx, buf, cid, args, block_dim, aicpu_thread_num, …)
    new (buf) Runtime()
    bind_prepared_callable_to_runtime(r, cid)   restore kernel/orch addrs
    bind_prepared_to_runtime_impl(r, args)
    DeviceRunner::run(r, block_dim, aicpu_thread_num)
      clear_cpu_sim_shared_storage()
      ensure_binaries_loaded()               dlopen aicpu/aicore SOs once
      launch AICPU + AICore threads
      join all threads
      unload_executor_binaries()             dlclose aicpu/aicore SOs
    validate_runtime_impl(r)                 copy results, remove kernels
    r->~Runtime()

ChipWorker.finalize()
  finalize_device(ctx)
  destroy_device_context(ctx)
  dlclose(host_runtime.so)                   -fno-gnu-unique ensures real unload
```

### Onboard (subprocess per device, ChipWorker reused per runtime group)

```text
device_worker_main(device_id)
  for each runtime_group:
    ChipWorker.init(host_path, aicpu_path, aicore_path, simpler_log_path, device_id)
      dlopen(libsimpler_log.so, RTLD_GLOBAL)
      simpler_log_init(log_level, log_info_v)
      dlopen(host_runtime.so, RTLD_LOCAL)
      create_device_context()
      simpler_init(ctx, device_id, aicpu*, aicpu_size, aicore*, aicore_size)
        DeviceRunner::attach_current_thread(device_id)  rtSetDevice()
        DeviceRunner::set_executors(aicpu, aicore)
        dlog_setlevel(HostLogger.level())               sync CANN dlog

    for each callable:
      ChipWorker.prepare_callable(cid, callable)
        prepare_callable(ctx, cid, callable)
          upload child kernels, copy orch SO to device buffer
      for each launch with that cid:
        ChipWorker.run(cid, args, config)
          run_prepared(ctx, buf, cid, args, block_dim, aicpu_thread_num, …)
            new (buf) Runtime()
            bind_prepared_callable_to_runtime()
            bind_prepared_to_runtime_impl()  rtMalloc, rtMemcpy to device
            DeviceRunner::run()
              ensure_binaries_loaded()       rtMemcpy AICPU SO to HBM (once)
              rtAicpuKernelLaunchExWithArgs() launch on device
              rtStreamSynchronize()          wait for completion
              launch_aicore_kernel()         rtRegisterAllKernel + rtKernelLaunch
            validate_runtime_impl()          rtMemcpy results back to host

    ChipWorker.finalize()
      finalize_device(ctx)                     rtDeviceReset()
      destroy_device_context(ctx)
      dlclose(host_runtime.so)
```
