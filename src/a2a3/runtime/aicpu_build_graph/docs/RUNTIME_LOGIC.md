# Runtime Logic: aicpu_build_graph

## Overview
The aicpu_build_graph runtime builds the task graph on AICPU using a small orchestration plugin. A dedicated builder thread runs the plugin and emits tasks into the shared Runtime object, while scheduler threads dispatch published tasks to AICore. This enables concurrent build and schedule on device.

## Core Data Structures
- `Runtime` stores task state, orchestration arguments, kernel address table, and the embedded orchestration plugin. See `src/runtime/aicpu_build_graph/runtime/runtime.h`.
- `Task` adds two concurrency flags, `published` and `completed`, so tasks can be made visible to schedulers only when fully defined.
- `AicpuBuildApi` is a device-side function table used by orchestration plugins to add tasks, add edges, and publish tasks without linking against runtime symbols.
- `HostApi` provides device memory ops used during host-side initialization.

## Host Init Flow
1. `init_runtime_impl` registers kernel binaries and fills `Runtime::kernel_addrs[]` so AICPU-side builders can resolve `func_id` to `function_bin_addr`. See `src/runtime/aicpu_build_graph/host/runtime_maker.cpp`.
2. The host marshals orchestration arguments. Pointer args are allocated on device and copied; scalars are passed directly. Output and inout buffers are recorded with `runtime->record_tensor_pair`.
3. The orchestration plugin SO is embedded into `Runtime` (`try_set_aicpu_orch_so`), and the entry symbol name is stored in `Runtime::aicpu_orch_func_name`.
4. The build mode is set from `PTO_AICPU_BUILD_GRAPH_BUILD_MODE` (0 = sequential build then schedule, 1 = concurrent build and schedule).

## Device Build And Schedule Flow
1. AICPU thread 0 loads the embedded orchestration plugin via `dlopen` and calls its entry function. See `src/runtime/aicpu_build_graph/aicpu/aicpu_executor.cpp`.
2. The plugin uses `Runtime::aicpu_build_api` to build the graph. Typical sequence per task is `add_task`, `add_successor_conditional`, then `publish_task`.
3. In concurrent mode, scheduler threads start immediately and only see tasks that have been published. In sequential mode, schedulers wait for the builder to finish.
4. When a task completes, the scheduler decrements fanin counters and pushes newly-ready tasks to the ready queues.
5. Tasks are dispatched to AICore using the same per-core handshake protocol as host_build_graph.

## Finalize And Cleanup
`validate_runtime_impl` copies recorded output tensors back to the host and frees any recorded device allocations. It also clears `tensor_pairs` and `device_allocs` for reuse. See `src/runtime/aicpu_build_graph/host/runtime_maker.cpp`.

## Key Files
- `src/runtime/aicpu_build_graph/runtime/runtime.h`
- `src/runtime/aicpu_build_graph/host/runtime_maker.cpp`
- `src/runtime/aicpu_build_graph/aicpu/aicpu_executor.cpp`
