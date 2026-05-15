# Runtime Logic: host_build_graph

## Overview

The host_build_graph runtime builds a static task graph on the host, copies the Runtime object to device memory, and lets AICPU scheduler threads dispatch tasks to AICore via a per-core handshake. Dependencies are explicit edges created by orchestration code, so scheduling is a standard fanin/fanout ready-queue model.

## Core Data Structures

- `Runtime` owns the task table, handshake buffers, and host-side device APIs. See `src/runtime/host_build_graph/runtime/runtime.h`.
- `Task` is a fixed-size record that stores `func_id`, argument array, `fanin`, `fanout`, `core_type`, and `function_bin_addr`.
- `Handshake` is the shared per-core control block used by AICPU and AICore for dispatch and completion.
- `HostApi` provides device memory ops used by host orchestration (`device_malloc`, `copy_to_device`, `upload_chip_callable_buffer`, etc.).

## Build And Init Flow

1. Python tooling compiles kernels and orchestration into shared objects.
2. `prepare_callable_impl` uploads the entire ChipCallable buffer (orch SO + all child kernel binaries) in one shot via `host_api.upload_chip_callable_buffer`, then dlopens the orchestration SO and resolves the entry symbol. For each child, host computes `chip_dev + offsetof(ChipCallable, storage_) + child_offset(i)` and stores it in `Runtime::func_id_to_addr_[child_func_id(i)]` via `Runtime::set_function_bin_addr`. See `src/runtime/host_build_graph/host/runtime_maker.cpp`.
3. The orchestration function runs on the host and builds the graph. Because it runs on host, it can (and sometimes must) dereference entry-tensor host pointers — e.g. to read a control tensor that drives per-block dispatch. So the orch owns its own H2D: it allocates device buffers, copies inputs to device, and registers outputs for copy-back via `record_tensor_pair(runtime, ...)`. It adds tasks via `add_task(runtime, ...)` and adds dependency edges via `add_successor(runtime, ...)`. (Contrast with `tensormap_and_ringbuffer`, where the orch runs on device AICPU and `runtime_maker.cpp` centralizes H2D using the chip-level `ArgDirection` signature.)
4. The populated `Runtime` is copied to device memory by the platform layer. AICPU then runs the executor with this Runtime snapshot.

## Execution Flow (Device)

1. `aicpu_executor.cpp` performs core discovery, handshake initialization, and ready-queue seeding using `Runtime::get_initial_ready_tasks`.
2. Scheduler threads maintain per-core and global ready queues. When a task is ready, the scheduler publishes the task pointer and signals the core via `DATA_MAIN_BASE`.
3. AICore reads the task_id from `DATA_MAIN_BASE`, executes the kernel at `Task::function_bin_addr`, and writes FIN to `COND` on completion.
4. AICPU observes completion, resolves dependencies by decrementing fanin, and enqueues newly-ready tasks.
5. The executor shuts down cores by setting `Handshake::control=1` after all tasks complete.

## Finalize And Cleanup

`validate_runtime_impl` copies all recorded output tensors back to the host and frees device allocations recorded in tensor pairs. See `src/runtime/host_build_graph/host/runtime_maker.cpp`.

## Key Files

- `src/runtime/host_build_graph/runtime/runtime.h`
- `src/runtime/host_build_graph/runtime/runtime.cpp`
- `src/runtime/host_build_graph/host/runtime_maker.cpp`
- `src/runtime/host_build_graph/aicpu/aicpu_executor.cpp`
