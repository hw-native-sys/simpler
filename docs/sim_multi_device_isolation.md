# Sim Multi-Device Isolation

## Problem

When 2+ ChipWorkers run concurrently on the sim platform, `host_runtime.so` global state causes crashes. Root causes identified:

1. **Task cookie core-ID collision**: `platform_set/get_cpu_sim_task_cookie()` keys by `(core_id, reg_task_id)`. Both instances allocate cores 0-8, so one instance consumes the other's cookies.
2. **Affinity gate counter collision**: `platform_aicpu_affinity_gate()` uses process-global atomic counters shared across all instances.
3. **`clear_cpu_sim_shared_storage()`** destroys the process-global `pthread_key`, breaking other running instances.

## Solution: Fork+SHM Process Isolation

Each ChipWorker runs in its own forked Python child process. The OS provides full address-space isolation — all global state in `host_runtime.so` is automatically separate per process.

```text
Main process (Scheduler)
  ├── WorkerThread[0] → shm mailbox → Child process 0 (ChipWorker[0])
  └── WorkerThread[1] → shm mailbox → Child process 1 (ChipWorker[1])
                                            └── dlopen(host_runtime.so)
                                                ← isolated globals
```

Communication uses a 4096-byte shared-memory mailbox per chip — the same layout used for SUB-type workers. The parent copies `ChipStorageTaskArgs` into the mailbox; the child copies it to heap before calling `run_runtime` (sim runtime requires heap-backed args, not mmap-backed).

## Why Not Fix the Globals

The global state in `host_runtime.so` spans multiple files (`cpu_sim_context.cpp`, `platform_aicpu_affinity.cpp`, `l2_perf_collector_aicpu.cpp`, `device_log.cpp`) and is deeply embedded in the AICPU/AICore thread model. Fixing each one individually is fragile. Process isolation solves all of them at once with zero platform code changes.

## Files

| File | Role |
| ---- | ---- |
| `src/common/hierarchical/worker_manager.{h,cpp}` | C++ `WorkerThread`: mailbox-IPC dispatcher for the forked chip child |
| `python/simpler/worker.py` (`_chip_process_loop`) | Python child: init ChipWorker, poll mailbox, run tasks |
