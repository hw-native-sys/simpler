# Graph Execution

Graph Execution is available only in the `host_build_graph` runtime. A Graph is
a composite incore task: it is submitted and completed once like an AIC, AIV,
MIX, or SPMD task, but contains a recorded task DAG.

The first invocation executes normally and records the DAG. A later invocation
places one `GRAPH` task in the host task window. The device Scheduler expands
the saved topology and dispatches its internal nodes; the Host Orchestrator does
not submit those nodes again.

## API

A Graph uses `CoreTaskArgs`, the existing incore argument type:

```cpp
void graph_function(const CoreTaskArgs &args, int variant) {
    const ChipTensor &input = args.tensor(0).ref();
    const ChipTensor &weight = args.tensor(1).ref();
    const ChipTensor &output = args.tensor(2).ref();

    const std::array<uint32_t, 1> shape{input.shapes[0]};
    TensorCreateInfo intermediate(
        shape.data(), static_cast<uint32_t>(shape.size()), input.dtype
    );

    CoreTaskArgs matmul_args;
    matmul_args.add_input(input, weight);
    matmul_args.add_output(intermediate);
    matmul_args.copy_scalars_from(args, 0, 1);  // current invocation's value
    TaskOutputTensors matmul = rt_submit_aic_task(
        variant == 0 ? FUNC_MATMUL : FUNC_MATMUL_TRANSPOSED,
        matmul_args
    );

    CoreTaskArgs activation_args;
    activation_args.add_input(matmul.get_ref(0));
    activation_args.add_output(output);
    rt_submit_aiv_task(FUNC_ACTIVATION, activation_args);
}

void submit_layer(const CoreTaskArgs &args) {
    rt_submit_graph(&graph_function, args, /*variant=*/0);
}
```

The function pointer is the default Graph identity. Trailing integral,
`float`, `double`, and `bool` construction parameters are forwarded to the
Graph function and hashed by value into the cache key. They are separate from
execution scalars in `CoreTaskArgs`: changing a construction parameter selects a
different Definition rather than patching an existing one.

An explicit identity is available for call sites that need a stable name:

```cpp
rt_submit_graph(
    GRAPH_KEY("qwen_decoder_layer_v1"),
    &graph_function,
    args,
    /*variant=*/0
);
```

An explicit `GRAPH_KEY` must be unique for every distinct Graph function in an
orchestration callable. The explicit-key overload deliberately excludes the
Graph function pointer from the cache identity so the key remains stable; using
the same key for different functions can select the wrong recorded topology.

There are no public `GraphArgs`, `GraphBindings`, `Patch`, or `ScalarRef`
types. The boundary is represented by `CoreTaskArgs`.

Boundary scalars are pass-through bindings. Forward them directly with
`node_args.add_scalar(args.scalar(i))` or `copy_scalars_from(args, i, count)`
so recording can retain their source indices. A value computed from a boundary
scalar in Graph host code is no longer a direct binding; perform that
transformation in a kernel, or make it a construction parameter when it changes
the Graph structure. Access through a non-const `scalar()` invalidates inherited
boundary provenance conservatively, because returning a mutable reference cannot
distinguish a read from a later write. A Graph containing such an invalidated
binding is not cached, which prevents replay from silently replacing the
transformed value with the unmodified boundary value.

## Supported dynamic and static data

- Boundary ChipTensor addresses may change for every invocation.
- Boundary scalar values may change for every invocation. Their count is fixed
  by the recorded boundary contract, and every boundary scalar must bind at
  least one internal task scalar.
- A Graph boundary contains at least one ChipTensor.
- Construction parameters are part of Graph identity and may control the
  function's task count, kernel selection, or other structural choices.
- Boundary ChipTensor shape, stride, dtype, size, direction, contiguity, and alias
  partition must match the first invocation.
- Internal task scalars with no boundary source are fixed Definition data.
- Boundary storage is caller-owned. `INPUT`, `INOUT`, `OUTPUT_EXISTING`, and
  `NO_DEP` are supported. A boundary `TensorCreateInfo` tagged `OUTPUT` is not.
- Early-resolve hints apply while recording the first invocation. Replayed
  internal nodes use the saved completion topology without the hint.
- A recorded task may depend on a Graph-external producer when that producer
  is the creator of a boundary ChipTensor. The outer Graph owns that dependency on
  replay; arbitrary cross-boundary explicit dependencies remain unsupported.

Structural or alias mismatch logs a warning and executes the Graph function
normally for that invocation. It never reuses heap offsets recorded for a
different shape. Debug builds also assert at these unsupported boundaries so
development catches a violated fixed-shape contract immediately; the ordinary
path remains the defensive release-build behavior.

## Qwen decoder-layer example

The upper layer packages all ChipTensor I/O in `CoreTaskArgs`; the wrapper has no
separate `hidden`, `weight`, or `output` parameters:

```cpp
void qwen_decoder_layer(const CoreTaskArgs &args) {
    const ChipTensor &hidden = args.tensor(0).ref();
    const ChipTensor &attention_weight = args.tensor(1).ref();
    const ChipTensor &mlp_weight = args.tensor(2).ref();
    const ChipTensor &output = args.tensor(3).ref();

    const std::array<uint32_t, 1> hidden_shape{hidden.shapes[0]};
    TensorCreateInfo attention_out(
        hidden_shape.data(), static_cast<uint32_t>(hidden_shape.size()), hidden.dtype
    );

    CoreTaskArgs attention_args;
    attention_args.add_input(hidden, attention_weight);
    attention_args.add_output(attention_out);
    attention_args.copy_scalars_from(args, 0, 1);  // dynamic token position
    TaskOutputTensors attention =
        rt_submit_aic_task(FUNC_ATTENTION, attention_args);

    MixedKernels mlp;
    mlp.aic_kernel_id = FUNC_MLP_AIC;
    mlp.aiv0_kernel_id = FUNC_MLP_AIV;

    CoreTaskArgs mlp_args;
    mlp_args.add_input(attention.get_ref(0), mlp_weight);
    mlp_args.add_output(output);
    rt_submit_task(mlp, mlp_args);
}

void submit_qwen_decoder_layer(const CoreTaskArgs &args) {
    rt_submit_graph(&qwen_decoder_layer, args);
}

void decode_three_layers(
    const std::array<ChipTensor, 3> &hidden,
    const std::array<ChipTensor, 3> &attention_weight,
    const std::array<ChipTensor, 3> &mlp_weight,
    const std::array<ChipTensor, 3> &output,
    const std::array<uint32_t, 3> &token_position
) {
    for (std::size_t layer = 0; layer < hidden.size(); ++layer) {
        CoreTaskArgs args;
        args.add_input(
            hidden[layer],
            attention_weight[layer],
            mlp_weight[layer]
        );
        args.add_output(output[layer]);
        args.add_scalar(token_position[layer]);
        submit_qwen_decoder_layer(args);
    }
}
```

The first layer records ordinary task submissions. Layers two and three submit
one Graph task each when their ChipTensor metadata and boundary scalar count
match. Each replay patches the current layer's `token_position`; its value is
not part of the Graph key.

## Definition

Recording uses host-only C++ state:

- `std::vector` for nodes, tensors, scalars, fanins, and pending uploads;
- `std::unordered_map` for the per-run Definition cache;
- `std::unique_ptr` for the active recording.

The cache stores at most 16 Definitions and allocates each entry to its actual
serialized size. No fixed maximum-size recording array is copied on a cache
hit.

At `graph_end`, recording is compacted into one contiguous, pointer-free POD
Definition. It contains:

- node order and AIC/AIV/MIX/SPMD kernel metadata;
- `root_indices` plus both directions of the immutable topology:
  fanin CSR and fanout CSR;
- one packed-heap offset per node;
- each node's ChipTensor source:
  `BOUNDARY_EXACT`, `BOUNDARY_VIEW`, `INTERNAL`, or `OWN_OUTPUT`;
- fixed scalar values plus boundary-scalar source indices;
- fixed boundary signatures and alias representatives.

The header also carries a content hash of the complete Definition image. The
device execution pool requires this hash, the Graph key, and the node count to
all match before reusing a resident Definition. A new run may record different
metadata under the same function identity, so key-only reuse is not safe.

All references are 32-bit offsets from the Definition base. Cross-boundary
Tensors use the fixed-width `GraphTensor` wire POD rather than the
64-byte-aligned C++ `ChipTensor` object. The upload is therefore one contiguous
copy with no raw Host pointers and no relocation pass.

Before materialization, the Scheduler recomputes the Definition content hash
and validates section ranges, topology indices, node heap offsets, the outer
heap extent, ChipTensor metadata, and ChipTensor-source bounds. Invalid wire data is
rejected before an offset participates in pointer arithmetic.

There is no cache schema version. The cache is per run and starts empty, so a
persistent-format version would currently have no effect.

## Cache hit and memory

For a cache hit, the Host Orchestrator:

1. validates the fixed boundary contract;
2. reserves one task-window slot;
3. reserves one heap block large enough for every internal intermediate;
4. computes only external fanin and boundary tensormap effects;
5. emits one outer `GRAPH` task;
6. stages the exact-size POD submission image for upload after orchestration;
7. asks the host runtime for an aligned execution block sized from the recorded
   node count, Tensor-address and scalar patch capacities, and Definition
   bytes, then writes that device address into the submission wire image.

Internal nodes consume no ring task-window slots. Their descriptor, payload,
and slot state are built in host-owned GM. The runtime retains one grow-only
block per `(pipeline slot, Graph key, occurrence index)`: repeated runs on the
same slot reuse the allocation, repeated uses of one key within a run receive
distinct blocks, and the two pipeline slots never share an active block. Every
allocation goes through the Worker's tracked `MemoryAllocator`, contributes to
`committed_device_memory()`, and is released when the Worker is finalized.

The `GraphSubmission` wire POD carries the aligned device address and usable
byte capacity explicitly. The Scheduler validates both before placement-
constructing `GraphExecution`; it never allocates execution storage from the
AICPU process heap. A block whose prior Definition key and content hash match
retains the local Definition, static node fields, and the Tensor-address and
scalar patch tables generated during its first materialization. That
graph-affine replay skips
topology binding, per-node count/offset validation, tensor-source
classification, tensor wire validation, static field stores, and static scalar
copies. It refreshes only task IDs, packed-buffer bases, boundary/internal
tensor addresses, boundary scalar bindings, scheduling state, dispatch
atomics, and wake registrations.

The retained blocks are addressed directly by `(pipeline slot, Graph key,
occurrence index)`. Occurrence numbering restarts deterministically for every
run, so repeated layers map back to the same block in their pipeline slot;
affinity does not depend on a recycler selecting a recently freed block.

## Scheduler flow

Host orchestration builds the complete task image before device execution. At
the end of orchestration, the Host uploads every exact-size Graph POD image,
relocates the task and payload pointers in the shared-memory image, copies the
complete shared-memory/runtime-arena image to the device, and then launches the
resident Scheduler.

All AICPU threads classify disjoint slices of the completed task window behind
one startup barrier. A Graph task enters preparation and external-fanin
classification during that scan, so Graph execution is interleaved with other
ready tasks at the same scheduling level once the Scheduler starts.

This design does not overlap orchestration and scheduling within one run.
Prepared-successor pipelining can overlap preparation of run N+1 with device
execution of run N, while Graph cache hits reduce repeated orchestration work
inside a run.

A Graph is placed in two independent control flows:

- `graph_prepare_queue`: materialize the saved nodes even while external fanin
  is still pending;
- `graph_ready_queue`: signal that the outer Graph's external fanin is ready.

Core-owning Scheduler threads pop at most one item from each queue per loop. A
prepare call expands at most four nodes and requeues unfinished work,
interleaving Graph expansion with normal scheduling.

Preparation and external readiness set two bits in one atomic activation gate.
Whichever operation sets the second bit activates the saved root nodes exactly
once.

Internal dependency readiness borrows the completion-state polling idea, but
dependency wiring remains an Orchestrator responsibility:

- recording constructs both fanin and fanout CSR in the immutable Definition;
- first materialization builds static runnable node state plus compact Tensor
  address and scalar patch tables; affine replay applies those tables and
  resets only dynamic runnable state;
- materialization registers each non-root on one producer selected from its
  saved fanin CSR;
- a node's release/acquire `task_state` is its Graph-local completion flag, so
  internal nodes need neither ring completion flags nor task-window slots;
- producer completion closes and drains only its current wake-list rather than
  traversing the saved fanout CSR;
- a woken consumer scans its saved fanin CSR and either enters its shape queue
  or registers on the next incomplete producer;
- `WAKE_LIST_SENTINEL` closes the completion/registration race: a failed
  registration observes completion and immediately rescans.

The runtime wake-list registration is a transient polling subscription, not
dependency discovery or Graph rewiring. Fanout CSR remains in the Definition
as part of the complete recorded topology and for DFX, but readiness does not
walk it.

```text
outer GRAPH
  -> activate root_indices[]
  -> producer completion drains its current wake-list
  -> each waiter polls saved fanin completion state
  -> ready waiter enters its ordinary shape queue
     or registers on another incomplete producer
  -> final internal completion completes the outer GRAPH
```

Internal nodes count as zero outer ring tasks. The final node completes
the one outer Graph task, publishes the outer ring completion flag, wakes
external consumers, and contributes one to the host-visible completion count.

Localization or materialization failure is fail-fast: the Scheduler latches an
error instead of leaving an already-submitted outer Graph unable to complete.

## Current unsupported cases

These cases assert in debug builds and execute through the ordinary path in a
release build:

- an empty Graph boundary;
- variable ChipTensor shape or metadata;
- changed boundary aliasing;
- runtime-allocated boundary outputs;
- nested Graph recording;
- dispatch predicates;
- cross-boundary explicit dependencies that are not represented by a boundary
  ChipTensor's creator;
- an unclassifiable internal ChipTensor source;
- a boundary-derived scalar accessed through mutable `scalar()`;
- more than 16 Definitions, 1024 internal nodes, or 32 boundary Tensors;
- insufficient task-window or heap capacity detected before outer submission.

An AICPU execution-pool or materialization failure happens after the outer
Graph has already been submitted. It therefore latches a Scheduler fatal error
instead of falling back; leaving the outer task pending would otherwise wedge
completion.

Explicit dependencies between recorded internal nodes are preserved when they
are otherwise supported; ordinary ChipTensor dependencies are always preserved.

## DFX

With L2 swimlane level 4:

- `Graph Execution` spans an outer Graph execution;
- `AICPU Scheduler` shows bounded `graph_prepare` slices separately from normal
  dispatch;
- existing Scheduler and Worker lanes show the expanded internal tasks.

The scene coverage under `tests/st/a2a3/host_build_graph/graph_execution`
includes an AIV fanin/fanout DAG, a Qwen-style AIV/AIC decoder-layer DAG, a
three-slot multi-block MIX/SPMD Graph, and a manual Qwen3-14B three-layer
decode. Every scene invokes the same fixed Graph three times: one recording
execution followed by two outer-Graph submissions.
