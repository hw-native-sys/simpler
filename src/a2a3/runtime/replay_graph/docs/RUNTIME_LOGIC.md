# Graph Execution Runtime Logic (`replay_graph`)

The runtime directory and configuration value remain `replay_graph` for
compatibility. The user-facing capability is **Graph Execution**: a repeated
sub-DAG is represented by one outer `GRAPH` task in the Task Window and is
prepared as scheduler-owned nodes as soon as that outer task is submitted.
Its saved roots are activated only after the outer task's dependencies become
ready.

In this document, an *arena graph* is the batch published by
`rt_graph_boundary()`. A *Graph Execution* is the reusable sub-DAG submitted by
`rt_submit_graph()`; the two concepts can be nested.

## 1. Execution Contract

`replay_graph` pipelines device orchestration and scheduling at explicit graph
boundaries:

1. One AICPU orchestrator builds graph N in the active graph arena.
2. `rt_graph_boundary()` publishes graph N and releases its task publish gates.
3. Scheduler threads execute graph N while the orchestrator builds graph N+1
   in the other arena.
4. The orchestrator reuses an arena only after its tasks have completed and the
   following graph has finished adding cross-graph dependency edges.

The final `rt_orchestration_done()` call publishes any remaining tasks and
marks the task stream complete. Scheduler threads start polling immediately
after runtime initialization; there is no whole-orchestration dispatch barrier.

## 2. Lifecycle

Task state is monotonic:

```text
PENDING -> COMPLETED
```

There is no `CONSUMED` state. Graph-level arena controls provide the reuse
contract instead:

```text
FREE -> BUILDING -> RUNNING -> DONE -> FREE
```

`exec_done` means every task in the graph completed. `dep_closed` means the
following graph can no longer append consumers to this graph's producers. Both
conditions must hold before the physical task and heap arena is reused.

Scopes preserve nesting, manual-dependency, DFX, and profiling semantics. A
scope does not publish a graph and does not own an arena; only
`rt_graph_boundary()` changes graph ownership.

## 3. Memory Ownership

### Shared memory

`PTO2SharedMemoryHeader` contains:

- the monotonic logical task count;
- `task_descriptors[]`, `task_payloads[]`, and `slot_states[]`;
- `task_slot_map[]`, which maps a logical task id to a physical arena slot;
- final orchestration, output, fatal-error, and stall state.

Logical task ids remain dense and monotonic. Physical slots come from one of
two half-window arenas, so task id and slot are not generally identical.

### Orchestrator arena

The orchestrator owns:

- `PTO2TaskAllocator`, with two task-slot and heap bump arenas;
- `PTO2DepListPool`, which holds fanout edges for the full invocation;
- `fanin_seen_epoch[]`, used for per-submit producer deduplication;
- `PTO2TensorMap`, used for automatic tensor dependency discovery;
- immutable Graph Definitions and their one-time definition state.

The dependency pool and TensorMap remain monotonic for the invocation. Only
task slots and output heap bytes use graph-level ping-pong reuse.

### Graph Execution memory

A cached Graph Definition stores the kernels, argument sources, resource
shapes, topologically ordered task templates, CSR fanout, initial fanin counts,
root indices, and output offsets. Capture has a fixed 1024-node upper bound,
but cache publication copies only compact node headers plus the actual tensor
templates, tensor-source records, scalars, scalar-source records, edges, roots,
and output offsets into one variable-length allocation. Capture-only maximum
arrays, including per-node fanin lists, are not retained. Each cache hit creates
a `PTO2GraphExecution` containing a Definition reference, the current dynamic
Args snapshot, and invocation-specific node state. Scheduler threads materialize
node descriptors and payloads outside the Task Window from a prepare queue in
bounded four-node slices, independently of the outer task's external readiness.
Each slice runs after completion handling and ready-task dispatch, then an
unfinished execution returns to the prepare queue. Thus a large Definition can
be prepared concurrently with existing AICore work without blocking one
scheduler's completion polling for the duration of the whole Graph. No
per-execution internal edge objects are created.

Retired Graph Execution blocks return to a bounded best-fit pool. A later
submission first prefers a block last materialized from the same Definition and
node count, then falls back to the smallest capacity that fits. Both paths
preserve the node-storage allocation and constructed node skeletons. On the
Definition-affine path, re-prepare retains immutable kernel, shape, topology,
and static-scalar fields and patches only task identity, output addresses,
runtime state, boundary tensors, internal addresses, and dynamic scalars. A
non-affine reuse rewrites the static fields too. This keeps general-purpose
`free`, `posix_memalign`, placement construction, and repeated static payload
copies out of the steady-state submit/reclaim cycle. The pool retains at most
16 MiB across 64 blocks and releases all cached blocks at invocation teardown.

Only the outer `GRAPH` descriptor, its aggregate output heap range, and its
coarse external dependencies are placed in the Task Window. Consequently an
N-node Graph Execution consumes one task slot. It still consumes dynamic node
storage proportional to N and output heap bytes equal to the aligned sum of
the nodes' outputs.

### Scheduler arena

The scheduler owns ready queues, early-dispatch queues, async wait state, and
profiling counters. It updates graph completion counters after the completed
task's fanout walk is finished, so an arena cannot be reused while a scheduler
still reads one of its slots.

## 4. Allocation

`PTO2TaskAllocator` divides the configured task window and heap equally between
two graph arenas. Each successful allocation:

1. reserves the next logical task id;
2. reserves the next physical slot in the active arena;
3. writes the logical-id-to-slot mapping;
4. aligns and reserves output storage in the active heap arena;
5. release-publishes the new logical task count.

One graph must fit in one half of the task window and heap. Dependency-pool and
TensorMap capacities must hold the complete orchestration invocation.

Submitting a cached Graph Definition reserves one outer task slot and one
contiguous output range. Internal node outputs are slices of that range. The
immutable CSR topology belongs to the Definition and internal dependencies do
not consume the global dependency pool.

## 5. Submit And Concurrent Wiring

Every non-inline task starts with one synthetic publish dependency:

```text
fanin_count = 1
fanin_refcount = 0
```

`submit_task` then discovers explicit and TensorMap dependencies. For each
unique pending producer, it:

1. increments the consumer's `fanin_count` before publishing the edge;
2. pushes the consumer onto the producer's atomic `fanout_head` stack;
3. treats the dependency as already satisfied if completion closed the stack
   before the compare-and-swap succeeded.

Publishing the count before the edge prevents a concurrent producer completion
from making an incompletely built consumer ready. A producer completion
atomically exchanges `fanout_head` with a CLOSED sentinel. An orchestrator
append therefore either joins the list before closure or observes completion
and omits the runtime edge.

Graph Definition capture records the logical dependency before the completion
check. The cached topology therefore does not depend on whether a prior arena
graph producer happened to finish while the next graph was being defined.

## 6. Graph Boundary

`rt_graph_boundary()` performs the publication protocol:

1. seals the active graph's logical task range;
2. reads the incrementally maintained count of inline-completed allocation
   tasks, without scanning Task Window slots;
3. publishes the range and graph completion state;
4. releases each pending task's synthetic publish dependency;
5. builds and compacts any pending Graph Definition while Scheduler/AICore can
   already execute the newly published miss tasks;
6. closes dependency wiring for the graph in the next arena;
7. waits until that arena has both `exec_done` and `dep_closed`;
8. resets the reusable task and heap bumps and starts building there.

The boundary release is the graph-freeze point. A task can receive completed
producer notifications while it is being built, but it cannot enter a ready
queue until its publish dependency is released.

## 7. Scheduling And Completion

Scheduler threads poll ready queues while orchestration is active. For a normal
completion, the scheduler:

1. marks the producer `COMPLETED`;
2. atomically closes and snapshots its fanout stack;
3. increments each consumer's `fanin_refcount`;
4. routes a consumer exactly once when
   `fanin_refcount == fanin_count`;
5. updates the producer graph's completion count after the fanout walk.

Dummy and deferred-completion tasks use the same final completion hook, so they
participate in arena reuse accounting without special cases.

A `GRAPH` task is control work and is never dispatched to AICore. Submission
adds it to a scheduler prepare queue. Scheduler threads cooperatively advance
one bounded slice per loop after completion, dummy resolution, normal dispatch,
and early dispatch. A per-execution claim prevents concurrent writers, while
requeueing partial executions provides round-robin progress across Graphs.
Materialization reads dynamic values from the current `TaskArgs` snapshot even
if the outer task still has unresolved external fanin. The state progresses
from `SUBMITTED` through `MATERIALIZING` to `PREPARED`, but no node is routed
before activation.

When the normal dependency path makes the outer task ready, Scheduler changes
the execution from `PREPARED` to `ACTIVE` and routes only the saved root indices
to the existing C/V/MIX/Dummy queues in topological order. A race between
prepare and readiness is joined by an activation-request flag, so roots are
routed exactly once. Internal completion walks the saved CSR fanout and
releases a consumer when its per-execution fanin count is satisfied. Internal
node completions do not advance the stream task count. The final internal node
completes the outer `GRAPH` task, releases its external consumers, and advances
the stream count once.

The scheduler exits only after final orchestration completion and the total
completed task count reaches the final logical task count.

## 8. Graph Definition And Execution

For the complete user-facing contract and examples, see
[Graph Execution API Guide](GRAPH_EXECUTION_API.md).

The preferred orchestration API uses an existing `L2TaskArgs` as the complete
dynamic boundary:

```cpp
static void attention(const L2TaskArgs &args) {
    L0TaskArgs qkv_args;
    qkv_args.add_input(args.tensor(0).ref());
    qkv_args.add_output(args.tensor(1).ref());
    qkv_args.add_scalar(args.scalar(0));
    rt_submit_task(qkv_kernels, qkv_args);
}

L2TaskArgs args;
args.add_input(input);
args.add_output(output);
args.add_scalar(layer_id);
rt_submit_graph(&attention, args);
```

On a Definition miss, the function executes normally and the runtime captures
an immutable, topologically ordered Graph Definition. Scalars forwarded
directly from `args.scalar()` retain their dynamic source index; constants
created inside the function are frozen in the Definition. On a hit, the
function is skipped and the orchestrator submits one outer `GRAPH` task
containing the Graph Execution address. It does not scan or wire internal
nodes. The returned `task_id` identifies that outer task on a hit; it
is invalid during the first define-and-execute call because that call submitted
the original nodes directly.

The outer `L2TaskArgs` forms the Graph Execution's external contract. The runtime
aggregates all accesses for each boundary tensor:

- read only becomes `INPUT`;
- write only becomes `OUTPUT_EXISTING`;
- read and write becomes `INOUT`;
- dependency-free access remains `NO_DEP`.

The outer task waits for all aggregated input producers. All later consumers
of an aggregated output depend on the outer task, so externally the Graph
Execution has coarse completion semantics. Internally, nodes keep their exact
recorded dependencies and can run in parallel when the DAG permits.

Definitions are process-local. Their keys include the runtime schema,
callable-content hash, explicit ID or function address, boundary tensor
metadata and direction, and boundary scalar count. Each execution reads dynamic
scalar values from their source positions in the current `L2TaskArgs` without
changing the structural key. The `enable_graph_cache` configuration name remains
for runtime compatibility.

Each active Graph Execution pins its Definition cache slot until all nodes
retire. Cache replacement selects only unpinned slots, so concurrent executions
can share topology without observing a Definition overwrite.

## 9. Early Dispatch

The publish dependency also participates in `dispatch_fanin`. Boundary
publication accounts for that dependency before releasing completion fanin.
Internal producer launches can then advance `dispatch_fanin` to the task's
final `fanin_count` and retain the existing early-dispatch behavior.

Graph Execution nodes currently disable early resolve. They enter the existing
ready queues only after their internal completion fanin reaches zero. The
outer `GRAPH` task still participates in the normal publication protocol.

A cross-graph producer may publish before or during the boundary operation. In
that race, early staging is opportunistic; completion readiness remains exact
because it uses the atomic fanout-close protocol and `fanin_refcount`.

## 10. TensorMap And Data Access

TensorMap entries remain valid for the invocation and are reset by epoch between
runtime invocations. Newer producer entries shadow prior graph entries for the
same tensor region.

An entry stores a logical producer task id. Resolving that id through
`task_slot_map` is valid only while the slot's task descriptor still carries the
same logical id. A mismatch means the producer's arena has already completed
and been reused, so the dependency is already satisfied and must be skipped.
This snapshot check prevents an old TensorMap entry or cached external fanin
from being attached to the unrelated task that now occupies the physical slot.

`get_tensor_data` and `set_tensor_data` can wait for producer completion, but
there is no consumer-retirement state. `set_tensor_data` therefore provides WAW
protection, not WAR protection.

## 11. Capacity Failures

The following fail fast:

- one graph exceeds half of the task window;
- one graph exceeds half of the output heap;
- the invocation exceeds the dependency pool;
- the invocation exceeds TensorMap capacity;
- a Graph Definition exceeds its fixed capture capacities.

Graph Definition capacity failure falls back to normal orchestration when no
runtime state has been partially submitted. Graph Execution allocation failure
before the outer task is allocated also falls back. Failure after Task Window
publication is fatal, matching ordinary submit allocation semantics.

## 12. Invariants

- Stream tasks use `ring_id == 0`; scheduler-owned Graph nodes use synthetic
  `ring_id == 1` identities for diagnostics and never enter `task_slot_map`.
- Logical task ids are dense; `task_slot_map` resolves physical slots.
- A stream task has one publish dependency plus its unique pending producers;
  internal Graph nodes have only internal fanin.
- Graph node materialization may run before the outer task is dependency-ready,
  but only in bounded slices after completion and dispatch; root activation may
  not precede outer readiness.
- An arena graph is dispatchable only after `rt_graph_boundary()` releases its
  publish dependencies.
- Fanout append and completion-close are atomic and cannot lose an edge.
- A resolved dependency slot is used only when its task-id snapshot matches the
  requested logical producer id.
- A physical arena is reused only when `exec_done && dep_closed`.
- Dependency-pool and TensorMap storage are not reclaimed within an invocation.
- One Graph Execution contributes exactly one logical stream task and one Task
  Window slot, independent of its internal node count.
- The outer task exposes coarse boundary dependencies; immutable internal
  topology stays in the pinned Graph Definition.
- Concurrent executions share Definition topology but have independent fanin
  counters, node states, dynamic Args snapshots, and output storage.
