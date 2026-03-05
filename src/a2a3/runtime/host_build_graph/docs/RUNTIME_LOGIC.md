# Runtime Logic: host_build_graph

## Overview
The host_build_graph runtime builds a static task graph on the host, copies the Runtime object to device memory, and lets AICPU scheduler threads dispatch tasks to AICore via a per-core handshake. Dependencies are explicit edges created by orchestration code, so scheduling is a standard fanin/fanout ready-queue model.

## Core Data Structures
- `Runtime` owns the task table, handshake buffers, and host-side device APIs. See `src/runtime/host_build_graph/runtime/runtime.h`.
- `Task` is a fixed-size record that stores `func_id`, argument array, `fanin`, `fanout`, `core_type`, and `function_bin_addr`.
- `Handshake` is the shared per-core control block used by AICPU and AICore for dispatch and completion.
- `HostApi` provides device memory ops used by host orchestration (`device_malloc`, `copy_to_device`, `upload_kernel_binary`, etc.).

## Build And Init Flow
1. Python tooling compiles kernels and orchestration into shared objects.
2. `init_runtime_impl` loads the orchestration SO from bytes, resolves the entry symbol, and registers kernel binaries with the platform uploader. The resulting GM addresses are stored by `Runtime::set_function_bin_addr`. See `src/runtime/host_build_graph/host/runtime_maker.cpp`.
3. The orchestration function runs on the host and builds the graph. It allocates device buffers, copies input data to device, records output buffers with `runtime->record_tensor_pair`, adds tasks via `runtime->add_task`, and adds dependency edges via `runtime->add_successor`.
4. The populated `Runtime` is copied to device memory by the platform layer. AICPU then runs the executor with this Runtime snapshot.

## Execution Flow (Device)
1. `aicpu_executor.cpp` performs core discovery, handshake initialization, and ready-queue seeding using `Runtime::get_initial_ready_tasks`.
2. Scheduler threads maintain per-core and global ready queues. When a task is ready, the scheduler writes its pointer to the core's `Handshake` and sets `task_status=1`.
3. AICore reads the handshake, executes the kernel at `Task::function_bin_addr`, and writes `task_status=0` on completion.
4. AICPU observes completion, resolves dependencies by decrementing fanin, and enqueues newly-ready tasks.
5. The executor shuts down cores by setting `Handshake::control=1` after all tasks complete.

## Finalize And Cleanup
`validate_runtime_impl` copies all recorded output tensors back to the host and frees device allocations recorded in tensor pairs. See `src/runtime/host_build_graph/host/runtime_maker.cpp`.

## Key Files
- `src/runtime/host_build_graph/runtime/runtime.h`
- `src/runtime/host_build_graph/runtime/runtime.cpp`
- `src/runtime/host_build_graph/host/runtime_maker.cpp`
- `src/runtime/host_build_graph/aicpu/aicpu_executor.cpp`
