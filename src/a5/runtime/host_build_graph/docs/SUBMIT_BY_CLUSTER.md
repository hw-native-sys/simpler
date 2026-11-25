# Cluster-Aware Submission in `host_build_graph`

## Contract

One call to `rt_submit_task` creates one mixed graph task. The task may use any
nonempty combination of the physical cluster's `AIC`, `AIV0`, and `AIV1`
lanes; all active lanes share one argument payload, dependency set, logical
block count, and completion record.

```cpp
MixedKernels kernels;
kernels.aic_kernel_id = aic_func_id;
kernels.aiv0_kernel_id = aiv0_func_id;
kernels.aiv1_kernel_id = aiv1_func_id;

CoreTaskArgs args;
args.add_inout(output);
args.launch_spec.set_block_num(block_num);
TaskOutputTensors result = rt_submit_task(kernels, args);
```

`rt_submit_aic_task` and `rt_submit_aiv_task` are convenience wrappers over the
same mixed-task contract.

## Task Representation

Host orchestration writes the task, common argument pools and explicit fanin
IDs. Bind generates calendar rows for the task's blocks. Each row names the
kernel role, a bound argument vector, its waits and its completion target.
Graph child kernel bindings are filled by AICPU while device execution is in
progress.

## Resource Shapes

`LaneMask::to_shape()` maps a task to one of four shapes:

| Shape | Meaning |
| ----- | ------- |
| `AIC` | One AIC lane per logical block |
| `AIV` | One AIV lane per logical block |
| `MIX` | Two or three active lanes in one physical cluster |
| `DUMMY` | Dependency-only task with no AICore dispatch |

A standalone Vector kernel can use either physical Vector lane. MIX roles
remain fixed to their requested cluster lanes.

## Logical Blocks and Validation

`block_num` is the number of logical SPMD blocks, not the number of physical
clusters. Non-sync tasks may be wider than the device and dispatch in waves.

Submission validates before allocating or publishing a slot:

```text
block_num >= 1
block_num * popcount(active_mask) <= INT16_MAX
```

The product is the number stored in the 16-bit completion counter. Invalid
values latch `SIMPLER_ERROR_INVALID_ARGS`; they are not capped at the device's
physical cluster count.

`require_sync_start` adds a separate residency constraint because every block
must launch as one cohort:

- AIV-only: `block_num <= rt_available_aiv_count()`
- AIC or MIX: `block_num <= rt_available_cluster_count()`

The cohort is that one task's blocks. Two `sync_start` tasks never launch as
one unit, and the constraint above is per task regardless of where it was
submitted: a task inside a Graph body is bounded by the same device width as a
top-level one. GRAPH_EXECUTION.md, under `src/common/host_build_graph/docs`,
describes the rendezvous itself and what the cohort is scoped to.

## Dependency and Readiness Flow

TensorMap and explicit dependencies determine the DAG. The Host places every
logical block and attaches the dependencies that still need a runtime check.
MIX places all participating roles on one cluster and uses the SSBUF
rendezvous. The Host places a synchronized SPMD cohort on distinct compatible
lanes. Each block enters when its lane and dependencies are ready. The kernel
handles synchronization inside the cohort.

Each lane retires a row after kernel return and the output fence. With no
defer registration it publishes its DONE credit. A deferred registration
keeps logical completion pending until the polling lane observes its signals
and publishes the credit. See the [calendar execution contract](../README.md).

## Capacity

`host_build_graph` is whole-graph-resident. Task slots, heap bytes, fanin IDs,
and TensorMap entries are not reclaimed while the graph is executing. The graph
must fit the configured task count (`runtime_env.ring_task_window`) and the
TensorMap pool before launch; the heap is not a third capacity, since its device
region is committed after orchestration at the size the graph turned out to need.

## Validation

Run the HBG runtime simulation sweep after runtime changes:

```bash
pytest examples tests/st --platform a2a3sim --runtime host_build_graph
pytest examples tests/st --platform a5sim --runtime host_build_graph
```

The A5 HBG SPMD, sync-start capacity, MIX and empty-lifecycle scenes cover
calendar shape selection and retirement. The `spmd_shape_selection` scene checks
that both ordinary and sync-start multi-block AIC tasks publish a fresh calendar
mode and produce the expected output.
