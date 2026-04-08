# InCore Orchestration Guide: host_build_graph

## Goal

In host_build_graph, the orchestration function runs on the host. It allocates device buffers, builds the task graph by calling `add_task(runtime, ...)`, and wires dependencies with `add_successor(runtime, ...)`.

## Where To Put Orchestration Code
- Each example keeps orchestration sources under `examples/a2a3/host_build_graph/<example>/kernels/orchestration/`.
- `examples/a2a3/host_build_graph/<example>/kernels/kernel_config.py` defines the orchestration entry point. Example: `ORCHESTRATION = {"source": ".../example_orch.cpp", "function_name": "build_example_graph"}`.

## Function Signature

Your orchestration entry must be `extern "C"` and match:

```cpp
int build_graph(OrchestrationRuntime* runtime, const ChipStorageTaskArgs &orch_args);
```

Include `orchestration_api.h`. Do not include `runtime.h` in orchestration sources.

## Argument Layout

`orch_args` contains separated tensor and scalar arguments through `ChipStorageTaskArgs`.

- Use `orch_args.tensor(i)` to read tensor metadata and host pointers
- Use `orch_args.scalar(i)` to read scalar values
- Validate `tensor_count()` / `scalar_count()` defensively in orchestration code

## Building The Graph

A typical host orchestration sequence is:

1. Allocate device buffers with `device_malloc(runtime, size)`.
2. Copy inputs to device with `copy_to_device(runtime, dev_ptr, host_ptr, size)`.
3. Record output buffers with `record_tensor_pair(runtime, host_ptr, dev_ptr, size)` so finalize can copy them back.
4. Create tasks with `add_task(runtime, args, num_args, func_id, core_type)`.
5. Add dependency edges with `add_successor(runtime, producer, consumer)`.

Example: see `examples/a2a3/host_build_graph/vector_example/kernels/orchestration/example_orch.cpp`.

## Kernel Mapping

- `func_id` and `core_type` are defined in `kernels/kernel_config.py` under `KERNELS`.
- The host uploads kernel binaries via `upload_kernel_binary` and stores addresses in `Runtime::func_id_to_addr_[]`. The platform layer resolves per-task `Task::function_bin_addr` from this map before copying to device.

## Debugging Tips

- Use `print_runtime(runtime)` to dump the task graph.
- Fail fast on arg count or allocation errors to avoid undefined behavior.
