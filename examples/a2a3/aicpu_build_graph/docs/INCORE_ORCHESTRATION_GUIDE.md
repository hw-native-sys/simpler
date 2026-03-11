# InCore Orchestration Guide: aicpu_build_graph

## Goal
In aicpu_build_graph, the orchestration function runs on AICPU. It reads device pointers from `runtime->orch_args`, allocates intermediate buffers with `device_malloc`, builds the task dependency graph through the `AicpuBuildApi` function-pointer table, and publishes tasks for scheduling.

## Where To Put Orchestration Code
- Each example keeps orchestration sources under `examples/aicpu_build_graph/<example>/kernels/orchestration/`.
- `examples/aicpu_build_graph/<example>/kernels/kernel_config.py` defines the orchestration entry point. Example: `ORCHESTRATION = {"source": ".../orchestration.cpp", "function_name": "orchestration"}`.

## Function Signature
Your orchestration entry must be `extern "C"` and match:

```cpp
int orchestration(Runtime* runtime);
```

`Runtime` is defined in `src/runtime/aicpu_build_graph/runtime/runtime.h`. The plugin `.so` is embedded by the host and loaded via `dlopen` on AICPU.

## Argument Layout
When you use the default `golden.py` tensor argument order (`TENSOR_ORDER`), the host populates `runtime->orch_args[]` with:

```
[dev_ptr_0, dev_ptr_1, ..., dev_ptr_n, scalar_0, scalar_1, ...]
```

- Device pointers are HBM addresses for I/O tensors (the framework allocates and copies them).
- Scalar values (e.g., element count) follow the pointers.
- `runtime->orch_argc` gives the total number of valid entries.

Validate `orch_argc` defensively before accessing elements.

## Building The Graph
A typical AICPU orchestration sequence is:

1. Read device pointers and scalars from `runtime->orch_args[]`.
2. Allocate intermediate device buffers with `api.device_malloc(bytes)`.
3. Create tasks with `api.add_task(runtime, args, num_args, func_id, core_type, 0)`.
4. Add dependency edges with `api.add_successor_conditional(runtime, producer, consumer)`.
5. Publish each task with `api.publish_task(runtime, task_id)` to make it visible to the scheduler.

Where `api` is `runtime->aicpu_build_api`.

## AicpuBuildApi Reference

| Function | Signature | Description |
|----------|-----------|-------------|
| `add_task` | `int (Runtime*, uint64_t* args, int num_args, int func_id, CoreType core_type, uint64_t bin_addr)` | Creates a task. Pass `bin_addr=0` to auto-bind from `kernel_addrs[func_id]`. Returns task ID (>=0) or -1 on error. |
| `add_successor_conditional` | `void (Runtime*, int from, int to)` | Adds a dependency: `from` must complete before `to` runs. Safe to call concurrently with the scheduler. |
| `publish_task` | `void (Runtime*, int task_id)` | Makes a task visible to the scheduler. Tasks with zero pending dependencies enter the ready queue immediately. |
| `device_malloc` | `void* (size_t)` | Allocates device (HBM) memory for intermediate tensors. |
| `device_free` | `void (void*)` | Frees device memory from `device_malloc`. |

## Kernel Mapping
- `func_id` and `core_type` are defined in `kernels/kernel_config.py` under `KERNELS`.
- Kernel binaries are loaded by the host and their addresses stored in `runtime->kernel_addrs[]`. `add_task` with `bin_addr=0` resolves automatically.

## Build Mode
`kernel_config.py` can set `RUNTIME_ENV["PTO_AICPU_BUILD_GRAPH_BUILD_MODE"]`:
- `"1"` (default): Concurrent build and schedule -- schedulers dispatch tasks as the builder publishes them.
- `"0"`: Sequential -- schedulers wait until the builder finishes all tasks.

## Examples
- `examples/aicpu_build_graph/vector_example/kernels/orchestration/orchestration.cpp`
- `examples/aicpu_build_graph/bgemm/kernels/orchestration/bgemm_orch.cpp`
