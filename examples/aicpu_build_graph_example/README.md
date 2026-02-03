# PTO Runtime Example - AICPU Builds Graph (aicpu_build_graph)

This example runs the same computation as `host_build_graph_example`, but the task graph is built on **AICPU** (1 builder thread) while scheduling/execution runs on **AICPU** (3 scheduler threads), for a total of **4** AICPU threads.

## Run (simulation)

```bash
python examples/scripts/run_example.py \
  -k examples/aicpu_build_graph_example/kernels \
  -g examples/aicpu_build_graph_example/golden.py \
  -p a2a3sim \
  -r aicpu_build_graph
```

## Key difference vs host_build_graph_example

- `kernels/orchestration/example_orch.cpp` does **prepare + marshal only**:
  - allocates/copies tensors,
  - writes `runtime->orch_argc` / `runtime->orch_args[]`,
  - does **not** call `runtime->add_task()` / `runtime->add_successor()`.
- The AICPU binary contains `build_graph_aicpu(Runtime*)`, which reads `orch_args[]` and builds/publishes tasks on device.

