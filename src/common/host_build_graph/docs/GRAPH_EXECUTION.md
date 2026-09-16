# Graph Execution

Graph Execution is available only in the `host_build_graph` runtime. A Graph is
a composite incore task: it occupies one task window slot and completes once,
like any other task of the run, but contains a recorded DAG of in-graph tasks. It
is a container in the same sense an SPMD task is — SPMD expands one slot into
`logical_block_num` blocks, a Graph expands one slot into its recorded in-graph
tasks — where an AIC, AIV or MIX task is a leaf that dispatches straight to cores.

Every invocation places exactly one `GRAPH` task in the host task window. On a
first miss, the caller immediately submits an outer task shell keyed by Graph
identity while a recording thread records the DAG off the ordinary submit path. Internal
submissions build host-only in-graph task metadata and assign output addresses from a
private bit-63 virtual range instead of consuming task-window slots or heap.
Later calls for the same in-flight identity submit more shells without waiting
for recording, and a call for a *different* identity opens its own recording on
its own thread rather than waiting. At orchestration completion, the caller joins
every recording and fills each shell's heap range and Definition content hash.
Cached invocations submit the same one `GRAPH` task directly — a cache hit never
waits on a recording. In both cases the device Scheduler expands the saved
topology and dispatches the in-graph tasks; the Host Orchestrator never submits
them as tasks of the run itself.

Boundary contracts are checked before an in-flight shell is accepted. Once a
shell has entered the task/dependency sequence, an unsupported construct found
by asynchronous recording is terminal for that orchestration; it cannot be
replayed as ordinary tasks without rolling back already assigned task IDs and
TensorMap producers.

## API

A Graph boundary uses `GraphTaskArgs`; an in-graph task's arguments use
`CoreTaskArgs`, the existing incore argument type:

```cpp
void graph_function(const GraphTaskArgs &args, int variant) {
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
    matmul_args.add_scalar(args.scalar(0));  // forwarded boundary parameter
    TaskOutputTensors matmul = rt_submit_aic_task(
        variant == 0 ? FUNC_MATMUL : FUNC_MATMUL_TRANSPOSED,
        matmul_args
    );

    CoreTaskArgs activation_args;
    activation_args.add_input(matmul.get_ref(0));
    activation_args.add_output(output);
    rt_submit_aiv_task(FUNC_ACTIVATION, activation_args);
}

void submit_layer(const GraphTaskArgs &args) {
    rt_submit_graph(&graph_function, args, /*variant=*/0);
}
```

The function pointer is the default Graph identity. Trailing integral,
`float`, `double`, and `bool` construction parameters are forwarded to the
Graph function and hashed by value into the cache key. They are separate from
execution scalars in `GraphTaskArgs`: changing a construction parameter selects a
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
types. The boundary is represented by `GraphTaskArgs`, which a Graph function
receives as `const GraphTaskArgs &`. It is sized independently of
`CoreTaskArgs`, and forwarding a boundary scalar into a task's `CoreTaskArgs`
crosses those two capacities without either naming the other: what
`args.scalar(i)` hands out identifies a parameter, not the `Arg` holding it.

Boundary scalars are formal parameters. `args.scalar(i)` answers parameter `i`
itself rather than its value, so forwarding it —
`task_args.add_scalar(args.scalar(i))` — makes the destination slot follow that
parameter on every replay. A slot names the parameter it came from, not the
`Arg` it was copied through, so provenance survives any number of intermediate
copies.

Reading a parameter as a value freezes it, and the type system says so: an
`InheritableScalar` has no conversion to a number, so `uint64_t v =
args.scalar(i)` and `static_cast<int32_t>(args.scalar(i))` do not compile. A
value read has to name the type it is reading — `args.scalar<T>(i)` — and what
it produces is a plain `T`: the destination slot becomes static Definition data
holding the recording invocation's number, and later cache hits replay that
number. Forwarding never converts, which is what keeps a correct pass-through
silent.

Because the diagnostic is the absence of a conversion rather than a
deprecation, it has no blind spot. A value read inside third-party template
code — `EXPECT_EQ(args.scalar(i), v)` is the case that motivated this — fails
there too, where a `[[deprecated]]` attribute would have been suppressed for
being instantiated inside a system header.

When a value read is what you meant, say so with `args.scalar<T>(i)`. It
applies `to_u64`'s actual inverse, which `static_cast` is not — a float slot
holds a bit pattern, so `static_cast<float>` of `1.0f`'s pattern yields
`1065353216.0`. An enum has no other spelling at all:
`static_cast<DataType>(args.scalar(i))` does not compile, because the handle
converts to nothing and `static_cast` has no conversion to apply.
`InheritableScalar::to<T>()` is the same read on a handle already in hand —
reach for it when the parameter arrived as a function argument and the `Arg` it
came from is no longer reachable.

Freezing on purpose has a second spelling, and the two do different things.
`args.scalar<T>(i)` hands the body a `T` to compute with, and whatever the
body does with it afterwards is ordinary host code.
`task_args.add_static_scalar(args.scalar(i))` instead forwards the parameter
into a slot and drops its origin: the slot carries the same bit pattern a
forward would have, but is recorded as static Definition data rather than
following the parameter. Reach for the first when the body needs the number, the
second when a destination — typically a nested Graph's boundary — should hold
the value the enclosing parameter had at record time.

A derived value freezes the same way, and needs the same explicit read:
`args.scalar(i) + 1` does not compile, `args.scalar<uint64_t>(i) + 1` does and
is frozen. Compute it before constructing the boundary and pass it as its own
parameter, perform the transformation in a kernel, or use a construction
parameter when the value changes the Graph's structure.

Boundary scalar slots are read-only: `scalar()` hands out the parameter, not a
mutable reference, so a binding cannot be overwritten after it is forwarded.

**Only a parameter of the Graph's own boundary is refreshed on replay.** The
Definition's scalar source refs index that boundary and nothing else, so a slot
that inherits anything else — a slot of some other `GraphTaskArgs`, or one built
inside the body — is static Definition data holding the value it resolved to at
record time. The runtime does not reject that; which slot a body inherits from
is the author's declaration, and this is the declared consequence. Two notes on
why it cannot be diagnosed instead:

- An address cannot tell "created inside this body" from "created outside it".
  The body's `Arg`s are stack locals while the boundary is pool storage, so
  their relative addresses are a platform accident, not a guarantee.
- Whether the outside slot's value changes between invocations is invisible
  here. If it does, the Definition keeps replaying the recorded one.

## Supported dynamic and static data

- Boundary ChipTensor addresses may change for every invocation.
- Boundary scalar values may change for every invocation. Their count is fixed
  by the recorded boundary contract. Unused boundary scalars are allowed and do
  not create internal scalar patches.
- A Graph boundary contains at least one ChipTensor.
- Construction parameters are part of Graph identity and may control the
  function's task count, kernel selection, or other structural choices.
- Boundary ChipTensor shape, stride, dtype, size, direction, contiguity, and alias
  partition must match the first invocation.
- Internal task scalars with no boundary source are fixed Definition data.
- Boundary storage is caller-owned. `INPUT`, `INOUT`, `OUTPUT_EXISTING`, and
  `NO_DEP` are supported. A boundary `TensorCreateInfo` tagged `OUTPUT` is not.
- Early-resolve hints apply while recording the first invocation. Replayed
  in-graph tasks use the saved completion topology without the hint.
- A recorded task may depend on a Graph-external producer when that producer
  is the creator of a boundary ChipTensor. The outer Graph owns that dependency on
  replay; arbitrary cross-boundary explicit dependencies remain unsupported.
- A recorded task may carry a dispatch predicate. Submit resolves a predicate
  into an absolute GM address, which no Definition can hold, so the Definition
  stores the operand tensor's classified source plus the element index within
  it and materialize resolves the pair per execution. The operand may be a
  boundary ChipTensor or another in-graph task's output; the predicate itself creates no
  dependency, exactly as on the ordinary path, so the caller still declares one
  on the operand's producer.

Structural or alias mismatch logs a warning and executes the Graph function
normally for that invocation. It never reuses heap offsets recorded for a
different shape. Debug builds also assert at these unsupported boundaries so
development catches a violated fixed-shape contract immediately; the ordinary
path remains the defensive release-build behavior.

## Qwen decoder-layer example

The upper layer packages all ChipTensor I/O in `GraphTaskArgs`; the wrapper has no
separate `hidden`, `weight`, or `output` parameters:

```cpp
void qwen_decoder_layer(const GraphTaskArgs &args) {
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
    attention_args.add_scalar(args.scalar(0));  // dynamic token position
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

void submit_qwen_decoder_layer(const GraphTaskArgs &args) {
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
        GraphTaskArgs args;
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

All three layers submit one Graph task each. The first starts background
recording, while layers two and three immediately submit outer task shells for
the same in-flight identity. The first following non-Graph operation (or
orchestration completion) joins recording and finalizes all three shells with
the new Definition. Each invocation patches the current layer's
`token_position`; it is a dynamic boundary scalar refreshed on every submission
and is not part of the Graph key.

## Definition

Recording uses host-only C++ state:

- `std::vector` for in-graph tasks, tensors, scalars, fanins, and pending uploads;
- `std::unordered_map` for the per-run Definition cache;
- `std::unordered_map` for the recordings in flight, keyed by Graph identity and
  holding each entry by `std::unique_ptr`, guarded by a mutex and completion
  condition while the recording threads publish their Definitions.

The cache stores at most 16 Definitions and allocates each entry to its actual
serialized size. Published and in-flight entries count against the same limit,
since an in-flight one has already claimed its identity. No fixed maximum-size
recording array is copied on a cache hit.

Recorded output addresses start at `GRAPH_RECORD_VIRTUAL_BASE = 1ULL << 63`.
They exist only to classify `OWN_OUTPUT` and `INTERNAL` Tensor sources and are
converted to offsets in the Definition; they are never dereferenced or placed
on the wire. Classification is by address-range containment alone, so it is only
sound while no real heap address can fall in that range:
`TaskAllocator::init()` asserts the whole configured heap lies below
`GRAPH_RECORD_VIRTUAL_BASE`, which makes an overlapping device GM heap a loud
failure at setup instead of a silently misclassified Tensor source. Recording
therefore leaves the shared task allocator unchanged.

### First-miss host threading

`graph_begin` computes the Graph identity before the body is recorded. On a
cache miss, the calling thread allocates a zero-heap outer task shell, records
its boundary dependency edges, and returns. A recording thread receives a deep
copy of the boundary arguments, records the in-graph tasks in the private virtual
address range, and builds and hashes the Definition. The first call waits only
until that private job has been installed in the recorder queue; it does not wait
for the operating system to schedule the thread or for `graph_prepare` to bind
the private recording state. The keyed in-flight entry and zero-heap outer shell
already exist before the job is enqueued, so later same-identity submissions can
safely proceed immediately. Threads remain parked on a condition variable for
the lifetime of the loaded orchestration SO and are reused by later runs. Eight
workers are created when the callable's orchestration SO is loaded, before any
`host_orch` run; unloading the SO stops and joins them before their code is
unmapped.

**Distinct identities record concurrently.** The recorder owns a fixed 16-slot
job queue and 16 reusable boundary snapshots. The eight prewarmed workers cover
a workload that cuts a forward pass into up to eight Definitions without creating
threads or allocating boundary storage between shell submissions. A ninth or later
concurrent miss grows one worker per additional job, up to the 16-Definition
limit, so the prewarm does not turn into an eight-recording concurrency cap.
Growth happens inside the submission that needs it, so it lands on the submitting
thread: a workload whose Definition count exceeds the prewarmed count pays a
`pthread_create` (measured 32-74 us each) in the middle of its submission burst.
Recording touches no shared allocator state and each recording classifies Tensor
sources only against its own in-graph tasks and its own boundary, so two recordings
sharing the `GRAPH_RECORD_VIRTUAL_BASE` range cannot see each other's addresses.
What serializes is only the per-identity rule: at most one recording per Graph
key, which the keyed in-flight map enforces.

`graph_begin` answers a **cache hit before consulting anything in flight**. A
published Definition is immutable, so replaying it depends on no recording; the
lookup order is what keeps an already-built Graph from waiting on an unrelated
Definition. An identity that is neither published nor in flight opens its own
recording rather than falling back to the ordinary path.

The queue handoff hands `graph_prepare` the in-flight entry's own address,
carried through `GraphScopeResult::recording_handle`. Prepare therefore neither
searches for its recording nor reacquires the Definition-state mutex: until its
thread ends or aborts, later same-identity submissions only read the immutable
boundary signature under that mutex. Avoiding the redundant acquire prevents the
short main-thread submit loop from starving a recording thread before it can
enter its private state, and the handle makes recording into another identity's
state unrepresentable rather than merely unlikely.

Calls for the same identity while recording is in flight follow the same shell
submission path on the calling thread. Their task IDs and TensorMap producers
therefore enter the ordinary program-order sequence while its recording thread is
executing `record_in_graph_task` and `build_definition`.

**Ordinary submissions do not join the recorders either.** `rt_submit_task`,
`rt_submit_dummy_task` and `alloc_tensors` proceed while any number of Definitions
are recording, because an ordinary task depends on nothing a recording produces:
the outer shell entered the task sequence and registered its TensorMap producers
at `graph_begin`, so fanin against it is already correct, and the deferred heap
block the shell still needs is an independent bump reservation. The consequence is
that heap-address order stops matching task-id order — an ordinary task submitted
during a recording takes its block first — and nothing depends on that
correspondence: reservations are independent bumps, relocation is
address-window-based rather than order-based, and `host_build_graph` retires
nothing during a run.

`rt_graph_commit` is therefore a barrier at exactly one point, orchestration
completion. It waits for **every** recording in flight, then walks deferred
shells in original submission order, reserves each shell's real heap block using
its Definition's `required_heap`, patches the task descriptor and Definition
content hash, and lets the image be uploaded. A scope transition is deliberately
not a barrier either: the main thread has already submitted the outer Graph shell
into that scope, while scopes executed by a recording thread are no-ops on the
real scope stack.

Making a barrier out of every ordinary submission is what a single-slot recorder
needed and what this design does not. What it costs depends on how an orchestration
interleaves: a loop whose body is nothing but Graph submissions drains only once,
after the loop, by which time the recording has had every later submission to
overlap with. A loop that allocates a cross-block tensor per iteration drains on
its *second* iteration instead, with the recordings just started. In a decode
measured in that second shape — four Definitions, one `alloc_tensors` per iteration
— the drain cost a third of the orchestration window, with the submitting thread
stopped and four recording threads running. With the barrier only at completion,
14% of that pass's submissions land inside the recording span instead of 0.2%, and
the recorders rather than the submitter become the tail.

Host phase records therefore show `graph_submit`, `submit_task` and
`alloc_tensors` on the main lane overlapping `record_in_graph_task` and
`build_definition` on the recording lanes.

At `graph_end`, recording is compacted into one contiguous, pointer-free POD
Definition. It contains:

- in-graph task order and AIC/AIV/MIX/SPMD kernel metadata;
- `root_indices` plus both directions of the immutable topology:
  fanin CSR and fanout CSR;
- each in-graph task's early-dispatch verdicts (`ED_FLAG_CANDIDATE` when every
  producer allows early resolve and the task itself carries no predicate, a
  dispatchable shape and at least one internal producer; `ED_FLAG_TRACKED` when
  some candidate names it as a producer). A candidate's fanin CSR row is stored
  sorted by producer index, so its tail names its deepest producer; every other
  row keeps record order. `bind_graph_topology` validates these flags and
  materialization replays them onto each task's slot, where the publish chain
  reads them: a candidate registers on its producers' chains and pre-stages
  once they have all published. The verdict covers non-root tasks only —
  qualification needs a producer to bet on, and a body root has none inside the
  body — so a root's verdict is not recorded here but decided at
  materialization;
- one packed-heap offset per in-graph task;
- each in-graph task's ChipTensor source:
  `BOUNDARY_EXACT`, `BOUNDARY_VIEW`, `INTERNAL`, or `OWN_OUTPUT`;
- fixed scalar values plus boundary-scalar source indices;
- fixed boundary signatures and alias representatives.

The header also carries a content hash of the complete Definition image. The
device execution pool requires this hash, the Graph key, and the in-graph task count to
all match before reusing a resident Definition. A new run may record different
metadata under the same function identity, so key-only reuse is not safe.

All references are 32-bit offsets from the Definition base. Cross-boundary
Tensors use the fixed-width `GraphTensor` wire POD rather than the
64-byte-aligned C++ `ChipTensor` object. The upload is therefore one contiguous
copy with no raw Host pointers and no relocation pass.

Before materialization, the Scheduler recomputes the Definition content hash
and validates section ranges, topology indices, in-graph task heap offsets, the outer
heap extent, ChipTensor metadata, and ChipTensor-source bounds. Invalid wire data is
rejected before an offset participates in pointer arithmetic.

There is no cache schema version. The cache is per run and starts empty, so a
persistent-format version would currently have no effect.

## Cache hit and memory

For a cache hit, the Host Orchestrator:

1. validates the fixed boundary contract;
2. reserves one task-window slot;
3. reserves one heap block large enough for every internal intermediate, plus
   the Definition's `execution_storage_bytes` for in-graph task storage and its
   argument pools;
4. computes only external fanin and boundary tensormap effects;
5. emits one outer `GRAPH` task;
6. stores boundary values in the outer task's ordinary compact argument pools.

The outer Graph's tensor region is counted in `ChipTensor` pool slots but holds
densely packed `GraphTensor` wire values. Graph scheduling never dispatches the
outer payload as a kernel payload; device localization reads the compact values
directly, so boundary metadata does not need to expand to full `ChipTensor`
records on either side of H2D.

In-graph tasks consume no task-table slots. Their descriptor, payload, slot
state, argument pools, and completion states live in the tail of the outer `GRAPH` task's
own heap block, past `required_heap`:
`[GraphExecution][ChipTaskStorage...][tensor pool][scalar pool][task_states]`. The state
array is last because a byte needs no alignment, so appending it moves no other region. One
`TaskAllocator::alloc` covers both the packed outputs and this execution storage,
so they are reclaimed together without a separate device allocation or release path.

An in-graph task's payload holds no argument array of its own — it names each region by
a delta, like any other payload. Its pools are the last two regions of the execution
storage, sized by the Definition's `tensor_arg_count` / `scalar_arg_count` and indexed by
the task's own `tensor_offset` / `scalar_offset`, so its arguments occupy the same
span in the pool as in the Definition's arg table. There is no fanin region: in-graph
task dependencies come from the Definition's fanin CSR, so such a task's `fanin_count`
stays 0 and its fanin delta unbound.

The Host computes the execution-storage size before allocating the outer task's
heap. It points the outer slot's existing `graph_context` at the shared device
Definition and compacts the outer payload's tensor/scalar regions with every
other task's argument pools. The copied arena zone and compact shared-memory
image travel in one H2D. During the parallel initial classify, the Scheduler
constructs `GraphExecution` in the outer heap tail, binds it to that Definition
and the outer payload, and replaces `graph_context` with the execution pointer.
In-graph task storage remains untouched until bounded materialization begins.

## Scheduler flow

Host orchestration builds the complete task image before device execution. At
the end of orchestration, the Host copies one bind image containing the
compacted shared-memory task window and argument pools, then launches the
Scheduler. Slot task and payload references remain self-relative;
`graph_context` is the absolute address of the retained Definition object until
initial classification localizes the execution in the outer heap.

All AICPU threads classify disjoint slices of the completed task window behind
one startup barrier. A Graph task enters preparation and external-fanin
classification during that scan, so Graph execution is interleaved with other
ready tasks at the same scheduling level once the Scheduler starts.

This design does not overlap orchestration and scheduling within one run.
Prepared-successor pipelining can overlap preparation of run N+1 with device
execution of run N, while Graph cache hits reduce repeated orchestration work
inside a run.

A Graph is placed in two independent control flows:

- `graph_prepare_queue`: materialize the saved in-graph tasks even while external
  fanin is still pending;
- `graph_ready_queue`: signal that the outer Graph's external fanin is ready.

Core-owning Scheduler threads pop at most one item from each queue per loop. A
prepare call expands at most four in-graph tasks and requeues unfinished work,
interleaving Graph expansion with normal scheduling.

Preparation and external readiness set two bits in one atomic activation gate.
Whichever operation sets the second bit activates the saved root in-graph tasks
exactly once.

Internal dependency readiness borrows the completion-state polling idea, but
dependency wiring remains an Orchestrator responsibility:

- recording constructs both fanin and fanout CSR in the immutable Definition;
- materialization builds each in-graph task's runnable state from the Definition,
  resolving its Tensor addresses against the boundary image and its producers'
  packed windows;
- materialization registers each non-root on one producer selected from its
  saved fanin CSR, scanning the row from its tail so the bet lands on the
  producer likeliest to complete last. A row holds the consumer's deduplicated
  operand order, so the tail is exactly the deepest producer only on an
  early-dispatch candidate's row, which recording sorts by producer index;
  elsewhere the direction is a heuristic;
- an in-graph task's completion truth is its execution's own `task_states` byte,
  the same shape the task header gives a GLOBAL task, so such tasks need neither
  a task-table slot nor a byte in the shared-memory array;
- producer completion closes and drains only its current wake-list rather than
  traversing the saved fanout CSR;
- a woken consumer with a single producer enters its shape queue directly; any
  other rescans its saved fanin CSR from its wake-scan cursor and either enters
  that queue or registers on the deepest incomplete producer. Completion is
  monotonic, so rows above the cursor stay complete and are never re-walked;
- `WAKE_LIST_SENTINEL` closes the completion/registration race: a failed
  registration observes completion and immediately rescans.

Early dispatch enters a body from two directions, and each is decided by a
different party:

- **Into the body.** The outer shell qualifies at submit, by the top-level rule
  minus the terms that describe dispatching to cores: a shell carries no
  predicate, has no resource shape and occupies no core, so its producers alone
  decide it. A qualified shell that its producers release early does not stage
  itself — it has nothing of its own to place — but stages the body's roots,
  each an ordinary AICore task. They ring on the ordinary route, when the
  shell's producers complete and `activate_graph_task` opens the external gate.
  A Graph as a *producer* is the direction not supported: a shell publishes no
  placement of its own for a consumer to bet on.
- **A root's own verdict.** Materialization, not recording, decides it, and the
  decision is three terms rather than the recorded conjunction: the shell must
  itself be a candidate, since staging a root can only ever happen on a shell
  release; and the root must be neither `DUMMY` (no dispatchable shape to index
  a per-shape early-dispatch queue with) nor predicated (an early release
  returns before the predicate test). Deciding it at materialization is what
  keeps the flag off a slot a reader can already see.

Which early-dispatch queue a released candidate enters is chosen by the task's
own `sync_start` attribute, never by its cohort: a `sync_start` candidate needs
an all-or-nothing stage and parks in the single shape-agnostic queue, every
other candidate in its per-shape one. An in-graph task reaches that fork by the
same path a top-level one does.

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

In-graph tasks count as zero tasks of the run itself. The last one to complete
finishes the one outer Graph task, publishes that task's `task_states` byte, wakes
external consumers, and contributes one to the host-visible completion count.

Localization or materialization failure is fail-fast: the Scheduler latches an
error instead of leaving an already-submitted outer Graph unable to complete.

## Current unsupported cases

Conditions detected before an outer shell is accepted use the ordinary path:

- an empty Graph boundary;
- variable ChipTensor shape or metadata;
- changed boundary aliasing;
- runtime-allocated boundary outputs;
- more than 32 boundary Tensors;
- more than 16 Definitions;
- insufficient task-window or known cache-hit heap capacity.

The following constructs are discovered only while a thread records the
first Definition. Because its outer shell is already in the task/dependency
sequence, they assert in debug builds and fail the orchestration in release
builds:

- nested Graph recording;
- cross-boundary explicit dependencies that are not represented by a boundary
  ChipTensor's creator;
- an unclassifiable internal ChipTensor source, including a dispatch
  predicate's operand tensor;
- a dispatch predicate whose operand is the predicated in-graph task's own output;
- a dispatch predicate whose index vector leaves the operand tensor's extent;
- runtime allocation inside the Graph body;
- more than 1024 in-graph tasks;
- insufficient heap capacity while deferred shells are finalized.

An AICPU execution-pool or materialization failure happens after the outer
Graph has already been submitted. It therefore latches a Scheduler fatal error
instead of falling back; leaving the outer task pending would otherwise wedge
completion.

Explicit dependencies between recorded in-graph tasks are preserved when they
are otherwise supported; ordinary ChipTensor dependencies are always preserved.

## DFX

With L2 swimlane level 4:

- `Graph Execution` spans an outer Graph execution;
- `AICPU Scheduler` shows bounded `graph_prepare` slices separately from normal
  dispatch;
- existing Scheduler and Worker lanes show the expanded internal tasks.

The scene coverage under `tests/st/{a2a3,a5}/host_build_graph/graph_execution`
includes AIV fanin/fanout DAGs, architecture-native AIC/AIV decoder-style DAGs,
and three-slot multi-block MIX/SPMD Graphs. Every scene invokes the same fixed
Graph three times: one recording execution followed by two outer-Graph
submissions. The full-model example at
`examples/a2a3/host_build_graph/qwen3_14b_decode` records one Qwen3-14B decoder
layer and replays its Definition for the remaining 39 layers.
`tests/st/{a2a3,a5}/host_build_graph/graph_predicated_dispatch` covers dispatch
predicates on both operand sources, giving each invocation its own gate buffer
and gate scalar so a replay that reused the recorded operand address would read
the recording invocation's gates.
