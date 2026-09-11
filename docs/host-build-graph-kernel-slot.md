# HBG execution-slot registration and admission

These internal interfaces implement the execution-slot trust root described in
[v9 and the H3 pipeline card](https://icc.gt.tc/vllm-pto#pipeline). Public kernel
init/prepare/launch and the registered CANN kernel entry points are not enabled
by these interfaces. The owner connects the control callbacks and event ordering;
H4 consumes the admission result to validate and restore image contents.

## Ownership and prepare

`KernelResourcePlan` reserves five logical regions within two physical allocations:
heap, runtime/SM, Definitions, A5 scheduler, and a 192-byte registry. The last
four occupy disjoint aligned slices of the packed runtime arena. The registry
adds its bytes and preceding alignment padding to the common resource contract,
but never to a graph's four mutable destination capacities or its payload.
`GraphResourceRequirements` remains a per-graph size snapshot; the plan accounts
for the additional per-context control storage.

The resource schema is version 2. Prepare and close retain existing allocator
accounting and rollback behavior. No third allocation or launch-time allocation
is introduced. Different graphs use one frozen set of destination addresses and
capacities. Registry storage has the same context lifetime as those destinations.

The control sequence outside capture is:

1. Build known graphs, query requirements, and aggregate capacity.
2. `plan.prepare(context, allocator_ops)` allocates the heap and packed arena.
3. `context.freeze_resources()` freezes both addresses and capacities.
4. `seal_graph_execution_slot(context, device_id, generation, runtime_binary_id,
   out)` reads the frozen resource views and produces an independent value record.
   It requires no ReadyEnqueued transition and performs no device I/O.
5. `prepare_graph_execution_slot(..., prepare_ops)` seals and calls the supplied
   registration enqueue operation on the dedicated non-hidden AICPU stream.
   The operation deep-copies the Host record before returning and must enqueue
   device initialization, registration and binding in that order.
6. The enclosing owner orders PrepareTail and marks ready after all preparation
   tasks have been enqueued. Caller, AICPU and hidden AICore remain distinct streams.

`inspect_frozen_resources` only reads frozen context views in Collecting or
ReadyEnqueued state, checking device, generation and resource schema. It does not
freeze, allocate, mark ready or relax the ready-state requirement of launch.
The owner serializes sealing/enqueue with close. A callback failure is returned
unchanged; it is not evidence that any partially enqueued work has completed.
The enclosing owner supplies partial-enqueue handling and readiness ordering.

## Wire and publication

`GraphSlotRegistration` is a 128-byte POD with its own version and checksum:

| Field | Meaning |
| ----- | ------- |
| Header and flags | Frozen-capacity, serialized-execution contract |
| Device and slot generation | Context/slot affinity; generation is nonzero |
| Runtime binary identity | Immutable runtime code/ABI identity supplied by the owner |
| Maximum packet bytes | Exact bound derived from the frozen payload capacities |
| Four destination base/capacity pairs | Heap, runtime/SM, Definitions, scheduler |
| Registry base/capacity | Independent context-owned control storage |
| Checksum | Accidental-corruption detection, not authorization |

The record is derived from the context, never reconstructed from a launch packet.
All nonempty bases are aligned, lengths are overflow-checked, and all five regions
are disjoint. The runtime binary identity must be stable for the registered
runtime/ABI; it is not a callable ID, argument hash or context generation.

`GraphSlotRegistry` is a 192-byte POD aligned to a cache line. Its first line
holds the context identity and publication state; the next two lines hold the
registration. Device control code initializes only newly allocated, exclusively
owned storage, then publishes `Empty -> Publishing -> Ready`. The complete
candidate is checked before claiming Publishing. Ready is release-published
only after copying and flushing the record. Acquisition checks state, invalidates
and revalidates the record before publishing an output value.

Ready records are immutable. Identical registration is idempotent; conflicting
registration, malformed records and mismatched registry identity fail without
replacing any bytes. Initialization refuses an actively bound registry. The
control owner must never use initialization to reset a previously live slot.

The resident AICPU DSO stores only one atomic registry pointer. Binding requires
a valid Ready record and refuses a different live pointer. It stores no slot
contents, generation history, callable table or cross-context conflict state.
After graph destruction and external quiescence, device control must detach that
pointer before `context.close()` releases memory. H3 exposes detachment; the
public close/registration owner still needs to connect this ordered operation.

## Invocation admission

HBG packet version 2 uses the former reserved tail of its 192-byte header for
device ID and runtime binary identity. The outer 64-byte K1 header is unchanged.
Version 1 packets fail closed. Callable generation, callable/argument/function
hashes remain per-invocation fields, so different callables and arguments can
share a slot. This gate does not authenticate their function tables or signatures.

The device and runtime binary arguments come from trusted AICPU initialization,
not from the packet being checked.

`admit_graph_packet_for_restore(packet, bytes, device_id, runtime_binary_id, out)`:

1. Acquires the independently latched registry, checking device, runtime binary
   identity, publication state, registration checksum and context/slot generation.
2. Bounds packet size before parsing its payload and rejects source overlap with
   any working or registry region.
3. Runs the full HBG framing, placeholder-address, region and checksum validation
   in DeviceCopy mode.
4. Compares packet device, slot generation, runtime binary identity and every
   destination base/capacity pair with the sealed record.
5. Publishes a read-only `GraphRestoreView` containing the trusted slot, validated
   header and payload view only after every check succeeds.

Failure leaves the output, registry, working memory and generation unchanged.
A packet with a recomputed checksum still cannot authorize a changed binding.
Admission neither copies images nor releases AICore/scheduler work. H4 must
validate internal image semantics, restore full mutable capacity and publish a
successful restore verdict; launch integration handles cancellation on failure.
The task-owned source and context must remain alive and immutable while the
admission result is consumed.
