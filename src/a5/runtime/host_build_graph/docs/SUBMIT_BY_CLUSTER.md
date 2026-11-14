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

Host orchestration initializes common `ChipTaskStorage` scratch records. Default
A5 HBG publishes an 80-byte `TaskImageStorage` per top-level task: its descriptor
contains the task ID and kernel IDs, and its payload contains argument counts,
field-relative pool deltas and a predicate. It publishes no `ChipTaskSlotState`
or task-state array. Resident metadata holds shape and block counts; resident
control records hold progress and completion. A separate 64-byte image header
holds the first-error atomic.

Host planning constructs Calls and Graph child arguments before launch. Device
execution tokens select prepared Calls and block/lane geometry. The argument
pools remain available to dependency/predicate access and output diagnostics.
Percounter selects its own current image representation for AICPU Graph
materialization; the default HBG reader only accepts its compact format.

## Resource Shapes

`LaneMask::to_shape()` maps a task to one of four shapes:

| Shape | Meaning |
| ----- | ------- |
| `AIC` | One AIC lane per logical block |
| `AIV` | One AIV lane per logical block |
| `MIX` | Two or three active lanes in one physical cluster |
| `DUMMY` | Dependency-only task with no AICore dispatch |

A single-AIV task is normalized to the AIV0 submit slot. MIX dispatch still
uses the full active mask so unused lanes do not block placement.

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

1. Host orchestration allocates a task slot and builds its payload.
2. TensorMap and explicit dependencies append producer local IDs to the payload's
   fanin region.
3. Submit publishes only the finished graph data; it does not push ready tasks.
4. Host prepares immutable AIC/MIX and AIV catalogs, Calls, execution controls
   and canonical completion gates. The unified graph view maps top-level fanins
   through completion references and Graph child fanins through Definition-local
   indices; roots depend on ENTER and EXIT depends on every sink.
5. After bootstrap, workers scan their engine catalogs in bounded slices to
   claim execution tokens. A token may wait for dependencies before occupying a
   dispatch slot; it advances its fanin prefix only after observing each
   producer's canonical DONE gate. Readiness requires both remaining computation
   and async debt to be zero for every producer.
6. Workers scan ENTER, EXIT and inline control nodes separately. After all fanins
   are DONE, a unique completion CAS settles the node. Consumers observe this
   gate directly; dependency resolution uses no dynamic ready inbox, fanout
   decrement or per-successor notification.

## Dispatch and Completion

Resident workers prepare their own two slots and participate in cooperative
admission when the task shape requires it. Pure AIC and MIX tasks share the
immutable AIC catalog; pure AIV tasks use the AIV catalog. These catalogs have no
enqueue operation or shared queue head. Ready execution tokens transfer into a
worker's private slots; its local ready mask/head orders those slots. The
[resident walkthrough](../../../../../docs/zh-cn/a5-hbg-aicore-walkthrough/README.md)
describes membership, physical placement and the completion resolver.

Every executed block/lane contributes once to the packed completion gate,
decrementing remaining computation and compute debt together. Async groups keep
additional debt until their unique FINISH. A valid gate with zero computation
and debt proves DONE to dependent tokens and control scans. The unique last
return or async join records logical completion. MIX cluster release retires
resources; it is not another completion contribution.

## Executor Model

The host executes orchestration synchronously and publishes the complete graph.
AICore owns dependency resolution, ready scheduling and kernel execution. AICPU
owns discovery, topology/context publication, supervision and partitioned EXIT/ACK
retirement. Invalid startup descriptors enter failure retirement without
executing tasks or reading an unpublished scheduler backing.

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
resident shape selection and retirement. The `spmd_shape_selection` scene checks
that both ordinary and sync-start multi-block AIC tasks publish a fresh resident
mode and produce the expected output.
