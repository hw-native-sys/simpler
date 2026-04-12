# PTO Runtime Example - AICPU Builds Graph (aicpu_build_graph)

This example runs the same computation as `host_build_graph_example`, but the task graph is built on **AICPU** (1 builder thread) while scheduling/execution runs on **AICPU** (3 scheduler threads), for a total of **4** AICPU threads.

## Run (simulation)

```bash
python examples/scripts/run_example.py \
  -k examples/aicpu_build_graph/vector_example/kernels \
  -g examples/aicpu_build_graph/vector_example/golden.py \
  -p a2a3sim
```

## Key difference vs host_build_graph/vector_example

- The framework (`init_runtime_impl`) automatically manages I/O tensor device memory
  using `arg_types`/`arg_sizes` and populates `runtime->orch_args[]`.
- `kernels/aicpu/orchestration.cpp` is compiled into a small AICPU-side plugin `.so`.
  - The framework embeds the plugin bytes into `Runtime`.
  - The AICPU runtime `dlopen()`s the embedded plugin and calls `orchestration(Runtime*)` on device.
  - The orchestration allocates intermediate tensors via `api.device_malloc()` (HBM) and builds the task graph.
