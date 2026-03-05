# InCore Orchestration Guide: tensormap_and_ringbuffer

## Goal
In tensormap_and_ringbuffer, the orchestration function runs on AICPU and builds the graph directly on device. Dependencies are discovered automatically by TensorMap based on tensor overlap, and task memory is allocated from ring buffers.

## Where To Put Orchestration Code
- Each example keeps orchestration sources under `examples/tensormap_and_ringbuffer/<example>/kernels/orchestration/`.
- `examples/tensormap_and_ringbuffer/<example>/kernels/kernel_config.py` selects the orchestration source and the runtime `tensormap_and_ringbuffer`.

## Required Exports
Your orchestration shared object must export:

```cpp
extern "C" PTO2OrchestrationConfig aicpu_orchestration_config(uint64_t* args, int arg_count);
extern "C" void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count);
```

Both symbols are loaded by AICPU via `dlopen` in `src/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`.

## Argument Layout
Arguments are constructed by `examples/scripts/code_runner.py` and passed through host init into `Runtime::orch_args` as device pointers or scalars. For the default `TENSOR_ORDER` flow, the layout is:

```
[ptr_0, ptr_1, ..., ptr_n, nbytes_0, nbytes_1, ..., nbytes_n, element_count]
```

Validate `arg_count` in `aicpu_orchestration_config` and interpret pointers as device addresses.

## Building The Graph
1. Call `pto2_rt_init_tensor_pool(rt)` at the start of `aicpu_orchestration_entry`.
2. Wrap orchestration in scopes with `PTO2_SCOPE(rt)` to control tensor lifetimes.
3. Use `make_tensor_external` for input/output buffers and `make_tensor` for intermediates.
4. Build `PTOParam` arrays with `make_input_param`, `make_output_param`, `make_inout_param`, and `make_scalar_param`.
5. Submit tasks with `pto2_rt_submit_task(rt, func_id, worker_type, params, num_params)`.

Dependencies are inferred by TensorMap from input/inout/output tensors, so you do not add explicit edges.

## Worker Types And Kernel IDs
- Worker types come from `pto_orchestration_api.h` (`PTO2_WORKER_CUBE`, `PTO2_WORKER_VECTOR`, etc.).
- Kernel `func_id` values are defined in `kernels/kernel_config.py` under `KERNELS`.

## Completion Semantics
Do not call `pto2_rt_orchestration_done` yourself in device mode. The executor wraps the entry call in an outer scope and signals completion after `aicpu_orchestration_entry` returns.

## Examples
- `examples/tensormap_and_ringbuffer/vector_example/kernels/orchestration/example_orchestration.cpp`
- `examples/tensormap_and_ringbuffer/bgemm/kernels/orchestration/bgemm_orch.cpp`
