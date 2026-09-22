# Graph Execution

Graph Execution is available only in the `host_build_graph` runtime. A Graph is
a composite incore task: it occupies one task window slot and completes once,
like any other task of the run, but contains a recorded DAG of sub-tasks. It
is a container in the same sense an SPMD task is — SPMD expands one slot into
`logical_block_num` blocks, a Graph expands one slot into its recorded sub-tasks
— where an AIC, AIV or MIX task is a leaf that dispatches straight to cores.

Every invocation places exactly one `GRAPH` task in the host task window. On a
first miss, the caller immediately submits an outer task shell keyed by Graph
identity while a recording thread records the DAG off the ordinary submit path. Internal
submissions build host-only sub-task metadata and assign output addresses from the
recording's own address space instead of consuming task-window slots or heap.
Later calls for the same in-flight identity submit more shells without waiting
for recording, and a call for a *different* identity opens its own recording on
its own thread rather than waiting. At orchestration completion, the caller joins
every recording and fills each shell's heap range and the device address of the
Definition object it replays.
Cached invocations submit the same one `GRAPH` task directly — a cache hit never
waits on a recording. In both cases the device Scheduler expands the saved
topology and dispatches the sub-tasks; the Host Orchestrator never submits
them as tasks of the run itself.

Boundary contracts are checked before an in-flight shell is accepted. Once a
shell has entered the task/dependency sequence, an unsupported construct found
by asynchronous recording is terminal for that orchestration; it cannot be
replayed as ordinary tasks without rolling back already assigned task IDs and
TensorMap producers.

## API

A Graph boundary uses `GraphTaskArgs`; a sub-task's arguments use
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
  The body's `Arg`s are stack locals while the boundary lives on the in-flight
  entry, so their relative addresses are a platform accident, not a guarantee.
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
- Boundary ChipTensor shape, stride, dtype, size, direction, contiguity, and
  alias partition must match the first invocation.
- A parameter's view origin (`start_offset`) may move between invocations, but
  only by moving its whole alias partition: what the contract compares is each
  parameter's origin *relative to the lowest-numbered parameter sharing its
  buffer*. A recorded tensor derived from a parameter stores its origin relative
  to that parameter, so a uniform shift of the partition is absorbed when the
  tensor is rebound; a shift of one member against another is not, because the
  overlap geometry recording inferred from those origins is baked into the
  Definition. A sliding-window caller that hands the same cache a different
  slice each invocation therefore keeps reusing its Definition.
- The buffers a boundary names must be pairwise identical or disjoint. Two
  parameters either share a buffer exactly — same address and same size, as two
  views of one tensor do — or share nothing. This one is a **precondition, not a
  checked property**: hazard tracking groups by buffer address, so two buffers
  that partially overlap at different addresses would be called unrelated memory
  and the edges between them would be lost. A runtime-allocated buffer satisfies
  it for free, since one allocator hands out disjoint blocks; boundary storage is
  caller-owned, though, and a caller-provided device address
  (`ChipTensor.make(..., child_memory=True)`) is outside that guarantee, so for
  those the caller supplies the precondition. Proving it per invocation would
  cost an ordering of the addresses, which is the whole cost of a sort.
- Two things about a boundary's buffers *are* checked, because each is
  answerable from one address alone. None of them may be empty. And two
  parameters at one address must name one size — a parameter's recording-space
  window is reserved once per address and sized from the first parameter to
  claim it, so its group has to agree on how wide that is. Either refusal sends
  the invocation down the ordinary path.
- Everything else about the body's shape is the author's declaration, not a
  checked property. A Graph key asserts that two invocations record the same
  topology: the same task count, the same kernel selection, the same edge set,
  and the same overlap geometry within an alias partition. The runtime validates
  none of it.
- Internal task scalars with no boundary source are fixed Definition data.
- Boundary storage is caller-owned. `INPUT`, `INOUT`, `OUTPUT_EXISTING`, and
  `NO_DEP` are supported. A boundary `TensorCreateInfo` tagged `OUTPUT` is not.
- Early-resolve hints apply while recording the first invocation. Replayed
  sub-tasks use the saved completion topology without the hint.
- Every tensor a recorded task uses must come from the Graph's own boundary — a
  parameter, or a view derived from one — or from another sub-task's output.
  A tensor that entered the body any other way, such as a global or one produced
  before the Graph, is refused by name: recording is abandoned and the bind fails
  rather than baking a Definition that would rebind that tensor to someone else's
  buffer. Pass it as a boundary parameter instead.
- A recorded task may not depend on a task submitted before the Graph. The only
  way to order a body behind one is to pass that task's output as a boundary
  parameter, which the outer Graph task then depends on like any other argument.
  An explicit dependency naming a task outside the Graph is refused by name.
- A recorded task may carry a dispatch predicate. Submit resolves a predicate
  into an absolute GM address, which no Definition can hold, so the Definition
  stores the operand tensor plus the element index within it, and materialize
  rebinds the tensor and resolves the pair per execution. The operand may be a
  boundary ChipTensor or another sub-task's output, but not the consuming
  task's own output; the predicate itself creates no dependency, exactly as on
  the ordinary path, so the caller still declares one on the operand's producer.

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

- `std::vector` for sub-tasks, tensors, scalars, fanins, and pending uploads;
- `std::unordered_map` for the per-run Definition cache;
- `std::unordered_map` for the recordings in flight, keyed by Graph identity and
  holding each entry by `std::unique_ptr`, guarded by a mutex and completion
  condition while the recording threads publish their Definitions.

The cache stores at most 16 Definitions and allocates each entry to its actual
serialized size. Published and in-flight entries count against the same limit,
since an in-flight one has already claimed its identity. No fixed maximum-size
recording array is copied on a cache hit.

A recording addresses its body in a space of its own, starting at
`GRAPH_RECORD_BASE = PACKED_OUTPUT_ALIGN`. The boundary's formal parameters come
first, one address per buffer — parameters sharing a buffer share an address —
each claiming as much room as the argument it stands for. A sub-task's
packed outputs are bumped from the end of that region. Both are positions in the
recording's own space, not addresses of anything.

Moving the parameters into that space is what makes a recording closed. The body
derives every tensor it uses from a parameter or from another sub-task's
output, so no address the caller owns reaches the Definition — the caller's real
addresses travel with the outer shell's own arguments instead. The base is
non-zero so that no recorded object sits at address 0, which a task slot uses as
its "has no packed output" sentinel, and it is `PACKED_OUTPUT_ALIGN` specifically
because that is the finest granularity any recorded address takes.

Parameters sharing a buffer must land on one address, and parameters over
different buffers on different ones, because hazard tracking groups by buffer
address: splitting one buffer across two addresses drops the WAR/WAW edges
between its views, and merging two buffers onto one invents edges the body never
had. `graph_alias_partition` settles that grouping in one hashing pass over the
addresses — argument order is the scan order, so each parameter's representative
is simply the first parameter that reached its address. What the pass does *not*
do is prove the distinct addresses name non-overlapping memory; that is a
precondition the allocator supplies (see "Supported dynamic and static data").

**Classification is by provenance, not by address range.** Each boundary tensor
is stamped with `TaskId::Space::PARAM` and its parameter index, each recorded
output with `Space::SUB_TASK`, and views propagate the stamp — so a tensor's
`owner_task_id` says which of the two cases it is. A tensor owned by neither,
meaning one that entered the body without passing through the boundary, is
refused by name and the Graph is abandoned rather than recorded. Ranges could not
answer this: a recording's space starts just above zero and a real device address
is 48-bit, so the two overlap, and an address test would attribute a foreign
tensor to whichever parameter it happened to land on.

**A recorded tensor is its own relocation record.** Nothing travels beside it
saying where its storage comes from: its owner already says that, and only its
buffer address has to move. A parameter's is dropped — replay takes the buffer
from this invocation's argument — and a body tensor's is stored relative to the
recording's output region, which is an exact affine image of the heap a replay
commits, so replay adds that heap's base and nothing else. Everything else in the
tensor is carried absolutely, because the boundary contract pins the fields a
replay would otherwise have to supply.

### First-miss host threading

`graph_begin` computes the Graph identity before the body is recorded. On a
cache miss, the calling thread allocates a zero-heap outer task shell, records
its boundary dependency edges, captures the boundary into the in-flight entry,
and returns. A recording thread reads that entry's parameter list, records the
sub-tasks in the recording's own address space, and builds and hashes the
Definition. The first call waits only
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
job queue; each boundary lives on its own in-flight entry. The eight prewarmed
workers cover
a workload that cuts a forward pass into up to eight Definitions without creating
threads or allocating boundary storage between shell submissions. A ninth or later
concurrent miss grows one worker per additional job, up to the 16-Definition
limit, so the prewarm does not turn into an eight-recording concurrency cap.
Growth happens inside the submission that needs it, so it lands on the submitting
thread: a workload whose Definition count exceeds the prewarmed count pays a
`pthread_create` (measured 32-74 us each) in the middle of its submission burst.
Recording touches no shared allocator state and each recording classifies Tensor
sources only against its own sub-tasks and its own boundary, so two
recordings cannot see each other's addresses even though both address their
bodies from `GRAPH_RECORD_BASE`: a tensor's provenance names a parameter or a
task of the recording that stamped it, and nothing resolves it anywhere else.
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
captured boundary under that mutex. Avoiding the redundant acquire prevents the
short main-thread submit loop from starving a recording thread before it can
enter its private state, and the handle makes recording into another identity's
state unrepresentable rather than merely unlikely.

Calls for the same identity while recording is in flight follow the same shell
submission path on the calling thread. Their task IDs and TensorMap producers
therefore enter the ordinary program-order sequence while its recording thread is
executing `record_sub_task` and `build_definition`.

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

Commit is therefore a barrier at exactly one point, orchestration completion. It
waits for **every** recording to leave `RECORDING`, then walks deferred shells in
original submission order, reserves each shell's real heap block using its
Definition's `required_heap`, patches the task descriptor and the shell's
Definition address, and lets the image be prepared. That wait is on recording
*state*, not on the recorder pool: a job returns from `graph_end` before its own
captures are destroyed, so the host orchestration entry's scope guard joins the
jobs belonging to this bind's `RuntimeContext` separately, on both normal return
and exception unwinding, before the build state those jobs borrow goes out of
scope. Completion includes capture destruction. An independent bind can finish
while another bind's recorder is still running; both continue to share the pool's
worker and queue capacity. A scope transition is deliberately
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
`alloc_tensors` on the main lane overlapping `record_sub_task` and
`build_definition` on the recording lanes.

At `graph_end`, recording is compacted into one contiguous, pointer-free POD
Definition. It contains:

- sub-task order and AIC/AIV/MIX/SPMD kernel metadata;
- `root_indices` plus both directions of the immutable topology:
  fanin CSR and fanout CSR;
- each sub-task's early-dispatch verdicts (`ED_FLAG_CANDIDATE` when every
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
- one packed-heap offset per sub-task;
- each sub-task's ChipTensors, stored relative to whatever replay rebases
  them against and carrying the owner that says which that is;
- fixed scalar values plus boundary-scalar source indices.

The recorded boundary itself — the parameters, their directions, and the alias
partition they form — is **not** in the image. It is held beside it, on the host,
because only the host reads it: it is what a later invocation is matched against,
while materialize takes the boundary from the outer task's own payload.

The header carries the object magic, the Graph key, and the image size. The
framing check requires the magic, a size that admits a `GraphDefinition`, a
`total_bytes` equal to the header's own, and a key equal to the header's before
any section offset is read out of the image.

All references are 32-bit offsets from the Definition base. A tensor is stored as
`TensorData`, the 96-byte base the 64-byte-aligned runtime `Tensor` derives from:
the image is built in a byte vector, which has nowhere to put an over-aligned
element. The upload is therefore one contiguous copy with no raw Host pointers
and no relocation pass.

Before materialization, the Scheduler re-checks the object framing and validates
section ranges, topology indices, sub-task heap offsets, and the
outer heap extent. A tensor's own geometry is not re-checked there: the host
resolved each one against the parameter or the producing block it came from and
refused the body otherwise, so what materialize validates is only the parameter
index that reaches an array of the execution. Invalid wire data is rejected
before an offset participates in pointer arithmetic.

There is no cache schema version. The cache is per run and starts empty, so a
persistent-format version would currently have no effect.

## Cache hit and memory

For a cache hit, the Host Orchestrator:

1. validates the fixed boundary contract;
2. reserves one task-window slot;
3. reserves one heap block large enough for every internal intermediate, plus
   the Definition's `execution_storage_bytes` for sub-task storage and its
   argument pools;
4. computes only external fanin and boundary tensormap effects;
5. emits one outer `GRAPH` task;
6. stores boundary values in the outer task's ordinary compact argument pools.

The outer Graph's tensor region holds ordinary `Tensor` values, one pool slot per
boundary parameter — the same element type and stride every other task's argument
pool uses. That uniformity is what lets the shared-memory restack walk every
task's tensors in one pass rather than switching element stride on the `GRAPH`
kind. Graph scheduling never dispatches the outer payload as a kernel payload;
device materialization reads the boundary from it directly.

Sub-tasks consume no task-table slots. Their descriptor, payload, slot
state, argument pools, and completion states live in the tail of the outer `GRAPH` task's
own heap block, past `required_heap`:
`[GraphExecution][ChipTaskStorage...][tensor pool][scalar pool][task_states]`. The state
array is last because a byte needs no alignment, so appending it moves no other region. One
`TaskAllocator::alloc` covers both the packed outputs and this execution storage,
so they are reclaimed together without a separate device allocation or release path.

A sub-task's payload holds no argument array of its own — it names each region by
a delta, like any other payload. Its pools are the last two regions of the execution
storage, sized by the Definition's `tensor_arg_count` / `scalar_arg_count` and indexed by
the task's own `tensor_offset` / `scalar_offset`, so its arguments occupy the same
span in the pool as in the Definition's arg table. There is no fanin region:
sub-task dependencies come from the Definition's fanin CSR, so such a task's `fanin_count`
stays 0 and its fanin delta unbound.

The Host computes the execution-storage size before allocating the outer task's
heap. It points the outer slot's existing `graph_context` at the shared device
Definition and compacts the outer payload's tensor/scalar regions with every
other task's argument pools. The copied arena zone and compact shared-memory
image travel in one H2D. During the parallel initial classify, the Scheduler
constructs `GraphExecution` in the outer heap tail, binds it to that Definition
and the outer payload, and replaces `graph_context` with the execution pointer.
Sub-task storage remains untouched until bounded materialization begins.

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

- `graph_prepare_queue`: materialize the saved sub-tasks even while external
  fanin is still pending;
- `graph_ready_queue`: signal that the outer Graph's external fanin is ready.

Core-owning Scheduler threads pop at most one item from each queue per loop. A
prepare call expands at most four sub-tasks and requeues unfinished work,
interleaving Graph expansion with normal scheduling.

Preparation and external readiness set two bits in one atomic activation gate.
Whichever operation sets the second bit activates the saved root sub-tasks
exactly once.

Internal dependency readiness borrows the completion-state polling idea, but
dependency wiring remains an Orchestrator responsibility:

- recording constructs both fanin and fanout CSR in the immutable Definition;
- materialization builds each sub-task's runnable state from the Definition,
  moving each Tensor's buffer onto this invocation's argument or onto the graph
  heap this execution was given;
- materialization registers each non-root on one producer selected from its
  saved fanin CSR, scanning the row from its tail so the bet lands on the
  producer likeliest to complete last. A row holds the consumer's deduplicated
  operand order, so the tail is exactly the deepest producer only on an
  early-dispatch candidate's row, which recording sorts by producer index;
  elsewhere the direction is a heuristic;
- a sub-task's completion truth is its execution's own `task_states` byte,
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
other candidate in its per-shape one. A sub-task reaches that fork by the
same path a top-level one does.

### What a `sync_start` cohort is scoped to

A cohort is **one task's blocks**, never a set of tasks. Everything the
rendezvous reads — `staged_core_mask`, `running_slot_count` and
`early_dispatch_state` — lives in that one task's `TaskPayload`, so
`try_launch_sync_start_cohort` decides a launch from that task alone. A body
root and a top-level task therefore never rendezvous with each other, and two
`sync_start` roots in the same body do not either. This holds however the
members were staged: the scope is the task, not the submission site.

What changes for a root inside a body is only **which event supplies each half**
of the rendezvous. Both halves must still hold before any doorbell rings, and
both mean the same thing they do at top level:

- *every gated core occupies a running slot*. For a top-level candidate Tier 0
  stages the cohort once its producer has published. For a body root Tier 0
  stages it once `stage_graph_roots_early` has enqueued it, which happens on the
  shell's early release rather than on any publish of its own.
- *the producer released*, i.e. `early_dispatch_state == DISPATCHED`. At top
  level that is the producer completing. For a body root it is the shell's
  producers completing, which lets `activate_graph_task` open the external gate
  and `graph_route_ready_roots` route the root.

A root has no producer inside the body, so the second half is the shell's
dependency rather than its own — which is the same dependency the shell stands
for. The all-or-nothing contract is unchanged; only its trigger moved.

The one thing two cohorts do share is the **global drain**, and it is a mutual
exclusion rather than a rendezvous: `sync_start_pending` admits one
capacity-short cohort at a time, whether it came from a body or from the top
level. A cohort that loses that race is cancelled back to the ordinary ready
path, not merged into the winner.

A body root that needs the whole device is legal and does not deadlock against
its siblings, for three independent reasons. Early staging runs only on an idle
pass with every ready queue drained, so it never takes cores from ready work.
Gated staging counts pending slots as available (`include_pending`), so a cohort
can stage behind running tasks instead of waiting for them to retire. And
sibling roots carry no waits-for edge between them — they are all gated on the
same external event — so no cycle exists for them to close. The worst outcome is
a lost pre-stage: a cohort that cannot be placed is cancelled and takes the
ordinary `ready_sync_queues` path, where the ready drain serves it.

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

Sub-tasks count as zero tasks of the run itself. The last one to complete
finishes the one outer Graph task, publishes that task's `task_states` byte, wakes
external consumers, and contributes one to the host-visible completion count.

Localization or materialization failure is fail-fast: the Scheduler latches an
error instead of leaving an already-submitted outer Graph unable to complete.

## Current unsupported cases

Conditions detected before an outer shell is accepted use the ordinary path:

- an empty Graph boundary;
- variable ChipTensor shape, stride, dtype, size, direction, or contiguity;
- a boundary whose alias partition changed, or whose members moved relative to
  each other within a partition;
- a boundary naming an empty buffer, or naming one address at two buffer sizes;
- runtime-allocated boundary outputs;
- more than `GRAPH_MAX_TENSOR_ARGS` (128) boundary Tensors;
- more than 16 Definitions;
- insufficient task-window or known cache-hit heap capacity.

The following constructs are discovered only while a thread records the
first Definition. Because its outer shell is already in the task/dependency
sequence, they assert in debug builds and fail the orchestration in release
builds:

- nested Graph recording;
- an explicit dependency naming a task outside the Graph;
- a ChipTensor that reached the body without coming through the boundary,
  including a dispatch predicate's operand tensor;
- a dispatch predicate whose operand is the predicated sub-task's own output;
- a dispatch predicate whose index vector leaves the operand tensor's extent;
- runtime allocation inside the Graph body;
- more than 1024 sub-tasks;
- insufficient heap capacity while deferred shells are finalized.

An AICPU execution-pool or materialization failure happens after the outer
Graph has already been submitted. It therefore latches a Scheduler fatal error
instead of falling back; leaving the outer task pending would otherwise wedge
completion.

Explicit dependencies between recorded sub-tasks are preserved when they
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
