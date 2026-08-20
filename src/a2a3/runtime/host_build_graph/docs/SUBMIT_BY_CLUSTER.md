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

The shared graph image separates stable task identity from scheduling state:

- `PTO2TaskDescriptor` contains the task ID, kernel IDs, and packed-buffer
  addresses.
- `PTO2TaskPayload` contains tensors, scalars, predicates, and position-
  independent `fanin_local_ids[]`.
- `PTO2TaskSlotState` contains the active mask, task attributes, logical block
  count, subtask counters, completion state, and descriptor/payload bindings.

This image is built on the host and copied to the device verbatim. It contains
no fanout adjacency or dependency-pool pointers, and its descriptor/payload
bindings are deltas from the slot state's own address rather than pointers.

## Resource Shapes

`ActiveMask::to_shape()` maps a task to one of four shapes:

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
values latch `PTO2_ERROR_INVALID_ARGS`; they are not capped at the device's
physical cluster count.

`require_sync_start` adds a separate residency constraint because every block
must launch as one cohort:

- AIV-only: `block_num <= rt_available_aiv_count()`
- AIC or MIX: `block_num <= rt_available_cluster_count()`

## Dependency and Readiness Flow

1. Host orchestration allocates a task slot and builds its payload.
2. TensorMap and explicit dependencies append producer local IDs to
   `fanin_local_ids[]`.
3. Submit publishes only the finished graph data; it does not push ready tasks.
4. After H2D, device boot scans every submitted task exactly once.
5. A task with every fanin complete is routed to its ready queue; otherwise it
   registers on its latest-submitted unmet producer's wake list, minimising
   transfers between wake lists and their CAS contention.
6. Producer completion reclassifies wake-list consumers until they become ready.

Completion flags are monotonic, so a task never needs periodic fanin polling.

## Dispatch and Completion

- AIC and AIV tasks claim as many free cores as the current scheduler owns and
  requeue remaining logical blocks for later waves.
- MIX placement selects whole clusters whose used lanes can all accept the
  task. High cluster offsets are represented by the runtime's 128-bit bitset.
- `require_sync_start` uses local staging when possible and a generation-tagged
  global drain when the cohort spans scheduler ownership domains.
- Each completed lane increments `completed_subtasks`; the mixed task completes
  exactly once when it reaches `block_num * popcount(active_mask)`.

## Executor Model

The host loads and executes the orchestration shared object synchronously. The
device has no orchestration thread: every launched AICPU thread participates in
scheduling its assigned cores after the boot thread attaches the prebuilt graph.
Cluster ownership is assigned during the AICore handshake and remains stable for
the run.

## Capacity

`host_build_graph` is whole-graph-resident. Task slots, heap bytes, fanin IDs,
and TensorMap entries are not reclaimed while the graph is executing. The graph
must fit the configured task capacity, heap, and TensorMap pool before launch.

## Validation

Run the HBG runtime simulation sweep after runtime changes:

```bash
pytest examples tests/st --platform a2a3sim --runtime host_build_graph
pytest examples tests/st --platform a5sim --runtime host_build_graph
```

The `host_build_graph_wide_dispatch` scene specifically covers wide AIV work,
sync-start MIX placement, normal MIX placement, and bit offsets above 63 with
`aicpu_thread_num=2`.
