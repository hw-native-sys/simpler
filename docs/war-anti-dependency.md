# WAR (Write-After-Read) Anti-Dependencies

**Decision record for issue #1306.**

When a pure `INPUT` reader of a buffer is followed by a later task that
overwrites that same buffer, the runtime does **not** guarantee a
write-after-read (WAR) ordering on its own. This is a deliberate performance
trade-off, not a bug. Express the ordering explicitly — the recommended way is
a manual `add_dep`, not promoting the reader to `INOUT`.

## The scenario

```text
R : reads  X   (add_input(X))        ── task R
W : writes X   (add_inout(X) / add_output_existing(X))  ── task W, submitted later
```

`W` overwrites `X` while `R` may still be reading it. Correctness requires
`W` to wait for `R` (WAR / anti-dependency). The automatic dependency
generator does not create this edge for a pure `INPUT` reader.

## Why the runtime does not track this automatically

The automatic dep-gen in
[`dep_compute.h`](../src/a5/runtime/tensormap_and_ringbuffer/runtime/dep_compute.h)
tracks the two hazards that a producer-keyed map answers in O(1):

- **RAW** (read-after-write): an `INPUT`/`INOUT` looks up the current writer
  of `X` and takes an edge on it.
- **WAW** (write-after-write): an `INOUT`/`OUTPUT_EXISTING` replaces the
  writer entry for `X` and takes an edge on the prior writer.

Tracking **WAR** for pure readers is fundamentally different: a writer would
have to find *every reader that is still in flight*, which means keeping a
reader set per buffer and walking it on every write. Recording each pure read
as its own tensormap entry and walking the whole same-buffer chain on every
write is an `O(chain)` cost paid on the orchestration hot path — for an edge
that most workloads never need, because their reads are already ordered ahead
of the overwrite by the RAW/WAW chain that produced the new data.

The orchestrator contract states this directly:
[`docs/orchestrator.md` §7 "Semantics"](orchestrator.md#semantics) — *"WAR is
not tracked directly … Simultaneous read and write races are a user bug, not a
scheduler concern."* This document is the how-to for the cases where you own
that WAR ordering.

## Expressing the WAR edge — two options

### Option A — promote the reader to `INOUT`

Change the reader's argument from `add_input(X)` to `add_inout(X)`. An `INOUT`
access is treated as a writer: it registers `X` in the tensormap, so the later
write takes a WAW edge on it, and the host-side `set_tensor_data` path becomes
aware of the reader through the producer's `fanout_refcount`.

**Cost — unnecessary read serialization.** Because `INOUT` is a write, two or
more readers of the same buffer no longer run concurrently: each becomes the
tensormap writer in turn, so the second reader takes a WAW edge on the first
and they serialize. A workload that reads `X` from several tasks in parallel
loses that parallelism purely to satisfy the anti-dependency. Reach for this
only when the reader genuinely also writes `X`, or when you specifically need
`set_tensor_data` on the host side to observe the reader.

### Option B — manual `add_dep` (recommended)

Capture the reader's task id and make the later writer depend on it
explicitly. This creates exactly the WAR edge and nothing else — the reads
stay pure `INPUT` and run concurrently.

```cpp
// Reader: keep it a pure INPUT.
CoreTaskArgs r_args;
r_args.add_input(ext_X);
r_args.add_inout(ext_Y);
TaskOutputTensors r = rt_submit_aic_task(FUNC_READ_X, r_args);

// Later writer: depend on the reader so the overwrite waits for the read.
// add_dep() lives on the convenience wrapper CoreTaskArgsWithDeps<N>.
CoreTaskArgsWithDeps<> w_args;
w_args.add_inout(ext_X);
w_args.add_dep(r.task_id());          // <-- the WAR edge R -> W
rt_submit_aic_task(FUNC_WRITE_X, w_args);
```

`add_dep` is the convenience layer over the primitive
`CoreTaskArgs::set_dependencies(ptr, count)`; both are documented in
[`arg_with_deps.h`](../src/a5/runtime/tensormap_and_ringbuffer/orchestration/arg_with_deps.h).
Multiple readers each contribute one `add_dep(reader.task_id())` on the writer
and still run in parallel with each other — only the writer waits.

At L3 and above, the Python API offers the same two dependency names as L2
with opaque host task handles. This WAR edge is ordering-only, so it uses
`add_dep_wait` rather than the retaining `add_dep`:

```python
reader = orch.submit_next_level(reader_handle, reader_args, cfg, worker=rank)
writer_args.add_dep_wait(reader)
orch.submit_next_level(writer_handle, writer_args, cfg, worker=rank)
```

The handle is valid only in the run that returned it. Host `TaskArgs` stores an
unbounded dynamic list, so there is no `CoreTaskArgsWithDeps<N>`-style fixed
capacity.

**`add_dep` is a task-to-task edge only — it is invisible to the host-side
`set_tensor_data`.** `set_tensor_data(X)` is a host write, not a task in the
graph, so a "writer depends on reader" edge places no constraint on it. Left to
itself, its only channel for discovering in-flight readers is buffer-keyed: it
looks up `X`'s producer in the TensorMap and waits for that producer *and its
consumers* (`wait_for_tensor_ready` with `wait_for_consumers=true`, see
[`runtime_core.cpp`](../src/a5/runtime/tensormap_and_ringbuffer/runtime/runtime_core.cpp)).
`add_dep` writes no TensorMap entry for `X` and does not touch that producer's
fanout, so a reader wired only through `add_dep` is not discovered by
`set_tensor_data` on its own. A host-side write that must wait for such a
reader names it in `set_tensor_data`'s `deps` argument: the accessor waits for
every declared task and its retained consumers. That is the same task id `add_dep`
already captured, so no extra bookkeeping is needed — see
[`SCALAR_DATA_ACCESS.md`](../src/a5/runtime/tensormap_and_ringbuffer/docs/SCALAR_DATA_ACCESS.md)
§3.3. Ordering-only (`DEP_WAIT`) consumers must be named directly: their
producer releases the fanout reference at wiring, before the reader completes.

That guard is a `tensormap_and_ringbuffer` property. `host_build_graph` has no
host-side WAR guard to be visible to: its orchestration runs to completion before
the device starts, and `set_tensor_data` rejects any tensor with a producer
outright (see
[`SCALAR_DATA_ACCESS.md`](../src/a5/runtime/host_build_graph/docs/SCALAR_DATA_ACCESS.md)).

## Recommendation

Prefer **Option B (`add_dep`)**. It is precise (one edge, no side effects on
the tensormap), and it preserves read parallelism. Use **Option A (`INOUT`)**
only when the tensor is semantically read-modify-write anyway, or when the
reader must also be visible to every accessor in the graph without threading
its id through each `set_tensor_data` call (see the `set_tensor_data` note in
[`orchestration_api.h`](../src/a5/runtime/tensormap_and_ringbuffer/orchestration/orchestration_api.h)).

| Concern | Option A: `INOUT` | Option B: `add_dep` |
| ------- | ----------------- | ------------------- |
| Creates the WAR edge | ✓ | ✓ |
| Keeps readers concurrent | ✗ (readers serialize as WAW) | ✓ |
| Visible to host `set_tensor_data` | ✓ | ✓ when the id is passed in `deps` |
| Extra tensormap entry per read | ✓ | ✗ |
| Effort | change one arg tag | capture id + one `add_dep` |

## See also

- [`docs/orchestrator.md` §7](orchestrator.md#semantics) — TensorMap RAW/WAW/WAR
  semantics.
- [`docs/manual-scope.md`](manual-scope.md) — manual scopes, where automatic
  dep tracking is off and `add_dep` is the primary ordering tool.
