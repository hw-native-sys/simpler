# WSE Block deployment

The device configuration supplies a fixed Core count `C` and four PEs per Core.
Every run uses that device. The Host supplies the logical Block count `B`
and one execution Core ID per Block. For example, this four-Block task uses
Cores 6, 2, 6 and 1 in logical Block order.

```cpp
CoreTaskArgs task;
task.add_input(input_view);
task.add_output(output_view);  // An existing view of fixed SRAM.
task.launch_spec.set_block_num(4);
WseLaunchSpec launch;
launch.placement = {{6}, {2}, {6}, {1}};
rt_submit_wse_task(kernel_id, task, launch);
```

This needs no Python binding or `CallConfig` extension. Python registers the
callable and submits ordinary `TaskArgs` through `Worker`.

## Placement and Threads

A single SS task supplies one placement. An SPMD task supplies exactly `B`
placements, indexed by logical Block ID. Core IDs may repeat. Each ID must be
within the configured device. Missing, extra or invalid placements fail during
Host submission. Graph enter and exit nodes carry dependencies and execute no
Blocks.

An orchestration can submit a synchronously starting SPMD task directly on
the outer task list. Set `task.launch_spec.set_require_sync_start(true)`
and provide at least two Blocks with distinct Core IDs. The Host places all
members at the same planned tick. At execution, each member's Core must have
finished its previous Block and reached the task at its queue head. Only after
all members are ready does the runtime release the task's PEs to enter the
kernel. Completion still waits for every Block. For ordinary SPMD, the default
`require_sync_start=false` lets each Core begin independently. Native Graph
recording rejects synchronized SPMD in this version.

In `wse_events.jsonl`, each synchronized member records `BLOCK_READY`
(kind 18) once. The last arrival records one task-wide `SYNC_RELEASE`
(kind 19). Every `BLOCK_BEGIN` for that task follows the release. Compare
event sequence numbers to verify this ordering across Cores.

A placement such as `{6}` runs on Core 6 with one Thread on PE 0 by default.
Setting `launch.thread_num` selects the first one to four PEs on each Core.
An explicit local PE list, such as `{6, {2, 0}}`, selects the Threads for that
Block. Duplicate or out-of-range PEs and unsupported synchronization requirements
reject before launch.

If `B < C`, some Cores have no task Blocks. If `B > C`, each Core executes
its assigned Blocks in Host calendar order. Device initialization and run
completion include all Cores and PEs. One Core runs at most one active Block.
All participating Threads must finish before that Core advances. The task
reports completion exactly once, after all its Blocks finish.

## Fixed data ownership

SRAM placement is part of the fixed data layout, independent of execution
placement. A kernel running on Core 0 may read a Tensor in Core 3's application
window. Remapping a Block never moves or reallocates the Tensor.

Workloads use explicit views for inputs, outputs and intermediates. Different
Blocks may have different Tensor views and scalars, provided their signatures
match. WSE records the complete access set across all Blocks, so a dependency on
a later Block's Tensor is not lost by looking only at the first Block's arguments.

## Graph recording and replay

Native Graph definitions preserve Block placement, per-Block Tensor bindings,
dynamic scalar origins and predicates. Rebinding changes addresses and values
through the Graph boundary; it does not change the fixed device model. Cached
Graph bindings must remain valid for the selected task layout. Predicate-false
tasks retire without executing their kernel body and still satisfy dependents.

SharedTile slots belong to the active Block. The last participating Thread
invalidates the Block scope before its storage is released. An old scope cannot
read or publish into a later Block, and finishing a Block on one Core does not
clear another Core's live slots.

## Validation examples

The WSE scene tests cover:

- Block counts below, equal to and above the configured Core count;
- explicit Core/PE placement and one to four Threads;
- independent per-Block Tensor arguments, cross-Core access and fixed data owners;
- Graph recording, rebinding, predicates and reverse-order waits;
- an `SS → SPMD(4) → SS` Graph on Cores `7 → [6, 2, 6, 1] → 3`, with actual
  dispatch checked against the submitted placements and per-Core calendar on
  each replay, and the final SS consuming all four results;
- counter and SharedTile collaboration, FFN numerics and capacity refusal before GO;
- startup failure rollback, timed waits and owner shutdown.

Run the same fixed-SRAM FFN tests after rebuilding the configuration with 1 MiB
per Core to qualify that device model. Passing with the default 100 MiB model is
not evidence for the smaller capacity.

Run the two Graph tests to verify calendar scheduling with the ordinary Worker.

```sh
python -m pytest tests/st/wse/host_build_graph/calendar --platform wsesim --require-pto-isa
```
