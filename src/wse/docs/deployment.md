# WSE Block deployment

The device configuration supplies a fixed Core count `C` and four PEs per Core.
Every run uses that device. The task supplies a logical Block count `B` through
the existing `CoreTaskArgs.launch_spec`:

```cpp
CoreTaskArgs task;
task.add_input(input_view);
task.add_output(output_view);  // An existing view of fixed SRAM.
task.launch_spec.set_block_num(blocks);
rt_submit_wse_task(kernel_id, task);
```

This needs no Python binding or `CallConfig` extension. Python registers the
callable and submits ordinary `TaskArgs` through `Worker`.

## Placement and Threads

The default assignment is `block_id % C`, with one logical Thread on PE 0.
`WseLaunchSpec` can select one to four logical Threads and an explicit placement
for each Block. A placement names one Core and distinct PEs on that Core. Invalid
Core IDs, duplicate PEs, mismatched table lengths and unsupported synchronization
requirements reject before launch.

If `B < C`, some Cores have no task Blocks. If `B > C`, each Core drains its
queue in order. Device initialization and run completion still include all Cores
and PEs. One Core runs at most one active Block; all participating Threads must
finish before that Core advances. The task reports completion exactly once,
after all its Blocks finish.

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
- counter and SharedTile collaboration, FFN numerics and capacity refusal before GO;
- startup failure rollback, timed waits and owner shutdown.

Run the same fixed-SRAM FFN tests after rebuilding the configuration with 1 MiB
per Core to qualify that device model. Passing with the default 100 MiB model is
not evidence for the smaller capacity.
