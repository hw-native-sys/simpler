# Prepared capture resources and lifetime

For A5 Scalar scheduling, apply the [shared hardware and scheduling premises](../zh-cn/design/a5-hbg-graph-task/build-execute-separation.md#a5-hardware-principles). A future STARS event must retain its actual resource through all notification and observation uses; it cannot be reset or released merely because a task returned. Mapped SPRs are not a substitute without an allocation and release protocol.

[Execution and API](native-capture-common-contract.md) ·
[HBG state restoration](native-hbg-resource-reset.md)

## Owners and backing

A prepared execution refers to immutable `PreparedRunResources`; the runner
separately tracks `PreparedRunUse` obligations. Destroying a Host run does not
remove an unresolved device, queued-wait, binding or external-use obligation.
A completed execution is not authorization to destroy a model's backing.

| Resource | Ownership and last use |
| -------- | ---------------------- |
| Caller context, stream and external tensors | Caller-owned; valid through all recorded and in-flight uses. |
| Callable code and registration | Simpler retains the actual callable lease; referenced code cannot be unregistered. Unrelated callables remain independent. |
| Runtime descriptor and initial HBG execution image | Onboard Program and Kernel identify both sections by offset and length in the RTS launch packet. RTS owns the snapshot for execution and retains it with a captured model for replay. |
| HBG writable execution arena | Simpler shares the current allocation across ordered runs. Each prepared run retains the allocation named by its launch packet. On A2/A3 and A5 the entry restores the complete RTS-delivered image to this arena before scheduling. |
| Pinned sources for explicit copies | Simpler submission storage retains pinned sources for the lifetime of recorded copies and any in-flight DMA. |
| Shared state and scheduler resources | Simpler retains the shared AICPU/AICore backing, each workspace lease, and scheduler notification and counter owners until the last use. Both device roles receive the same shared addresses. |
| HBG GM heap | Preparation retains a fixed heap lease through its last use, including external model uses. |
| Internal streams, fences and caller-order events | Actual stream generation and event storage survive their recording/execution/wait references. |
| Program input staging | Owned by the Program run through observation and cleanup. |

Host orchestration scratch and draft mirrors can be reused after extraction has
ended their readers. RTS launch snapshots and the Simpler-owned sources of
recorded DMA carry their respective replay lifetimes. Borrowed device addresses
in either carrier refer to separately retained backing.

Host snapshots and pinned DMA sources retain their per-run contents. The writable
HBG arena and ordinary A5 scheduler storage can serve multiple retained runs at
the same address because each execution initializes them in stream order. When
a graph outgrows the current allocation, previous runs keep their recorded
addresses through their last use. A5 scheduler storage read back for profiling
remains private to its run.

HBG heap leases protect allocation lifetime, not historical scratch contents.
Same-context ordered preparations share one fixed-capacity allocation even while
models remain retained. Capacity and base remain fixed from the first successful
prepare allocation until context teardown. Oversized graphs fail before device
submission. Neither normal retirement nor workspace pressure trims the heap.
Failed finalization preserves the preparation and its heap lease for retry.
Heap sharing does not apply to
caller-owned persistent tensors, immutable images or retained diagnostic evidence.

## Framework-owned runs

A framework model ticket holds the submitted run directly. Prepare snapshots
the execution image. The run owns the backing that the submission references
while its pipeline slot can accept another run. A model destroy notification
allows the framework to finalize each run after its last use. The same
notification can serve multiple nodes in one model.

## Native caller-owned runs

Prepare each Kernel node with `simpler_prepare_caller_run` and submit it with
`simpler_submit_run_on_stream`. Keep the original run storage, borrowed
context, stream and tensors valid until checked finalization. A successful
submission returns execution capacity while preserving the backing named by
the launch. Other nodes can reuse the slot.

After destroying or cancelling the model, end outstanding uses on every
execution stream and call `simpler_finalize_run` for each original run. A
partial submission needs the same cleanup. A failed stream drain or reader
settlement leaves the run available for retry. Call finalization from a thread
that can use the borrowed device context and synchronize streams.

For diagnostics, the original run owns its window and associated collector
resources. Before finalization, stop other producers sharing the pool. After
finalization, flush diagnostics to observe artifact publication. See the
[ScopeStats contract](../dfx/scope-stats.md#native-tmr-capturereplay-collection)
and the [PMU contract](../dfx/pmu-profiling.md#native-a3-capturereplay).
