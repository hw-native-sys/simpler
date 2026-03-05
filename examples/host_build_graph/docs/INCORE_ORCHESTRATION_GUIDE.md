# InCore Orchestration Guide: host_build_graph

## Goal
In host_build_graph, the orchestration function runs on the host. It allocates device buffers, builds the task graph by calling `Runtime::add_task`, and wires dependencies with `Runtime::add_successor`.

## Where To Put Orchestration Code
- Each example keeps orchestration sources under `examples/host_build_graph/<example>/kernels/orchestration/`.
- `examples/host_build_graph/<example>/kernels/kernel_config.py` defines the orchestration entry point. Example: `ORCHESTRATION = {"source": ".../example_orch.cpp", "function_name": "build_example_graph"}`.

## Function Signature
Your orchestration entry must be `extern "C"` and match:

```cpp
int build_graph(Runtime* runtime, uint64_t* args, int arg_count);
```

`Runtime` is defined in `src/runtime/host_build_graph/runtime/runtime.h`.

## Argument Layout
When you use the default `golden.py` tensor argument order (`TENSOR_ORDER`), the argument layout built by `examples/scripts/code_runner.py` is:

```
[ptr_0, ptr_1, ..., ptr_n, nbytes_0, nbytes_1, ..., nbytes_n, element_count]
```

- Pointers are host pointers to CPU tensors.
- Sizes are byte sizes for each tensor in `TENSOR_ORDER`.
- `element_count` is the element count of the first tensor.

If `golden.py` returns an explicit argument list, that list becomes `args` directly. Validate `arg_count` defensively in your orchestration.

## Building The Graph
A typical host orchestration sequence is:

1. Allocate device buffers with `runtime->host_api.device_malloc`.
2. Copy inputs to device with `runtime->host_api.copy_to_device`.
3. Record output buffers with `runtime->record_tensor_pair(host_ptr, dev_ptr, size)` so finalize can copy them back.
4. Create tasks with `runtime->add_task(args, num_args, func_id, core_type)`.
5. Add dependency edges with `runtime->add_successor(producer, consumer)`.

Example: see `examples/host_build_graph/vector_example/kernels/orchestration/example_orch.cpp`.

## Kernel Mapping
- `func_id` and `core_type` are defined in `kernels/kernel_config.py` under `KERNELS`.
- The host uploads kernel binaries via `upload_kernel_binary` and stores addresses in `Runtime::func_id_to_addr_[]`. The platform layer resolves per-task `Task::function_bin_addr` from this map before copying to device.

## Debugging Tips
- Use `runtime->print_runtime()` to dump the task graph.
- Fail fast on arg count or allocation errors to avoid undefined behavior.
