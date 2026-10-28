# A5 `host_build_graph` resident scheduler: per-task argument capture

Status: implemented by the resident boundary hooks and shared resumable producer.
The CPU contracts, public Program simulator payloads and native cross-compilation
provide separate checks. Onboard visibility, error recovery and the framework
corpus require qualification against the final source/native identity; simulator
success alone does not establish those properties.

## Capture contract and identity

A run-boundary snapshot is insufficient: if task A writes X and task B reads and
changes X, a pre-run snapshot misses A's value and a post-run snapshot loses B's
input. The resident AICore scheduler therefore pauses tasks at their real
`BEFORE_DISPATCH` and `AFTER_COMPLETION` boundaries while an AICPU producer copies
selected arguments. It uses the common eager Program/Kernel and per-run
`CallConfig` paths, without a new capture executor or public capability query.

The supported resident path remains single-block, with one execution of each
task in the run and one to three active subtasks. Existing graph admission and
whole-graph legacy fallback remain in force. Unsupported SPMD/cohort semantics
are not supplied by this diagnostic protocol.

The scheduler's `slots[3][2]` belong to independent lanes. Mix participants can
occupy different slot indices, while unrelated tasks can use the same numeric
slot in different lanes. A cluster/slot side state cannot identify the task.
Instead, a checked dense map covers the actual graph's task IDs and points to
one state record per selected task. The host verifies the existing dense graph
identity and task-window bounds before publication. No reusable slot claim,
occupancy counter or additional execution generation is introduced.

A phase is identified by the full 64-bit run epoch, full task ID and stage.
The publisher's existing dispatch generation additionally qualifies the seal.
A request tick tracks resumable copy progress within that phase; it is not a
substitute for phase identity. Peers accept a terminal seal by phase identity,
while the publisher requires the matching tick before advancing its request.
The BEFORE and AFTER phases may elect different publishers.

The acknowledgement's existing 64-bit status word commits its dispatch
generation, stage and state together. The owner publishes an invalid zero
commit before overwriting the body, publishes the body and echoed tick, then
publishes the packed commit. Each publication uses the existing cache flush
and write barrier. No acknowledgement or phase record grows.

Readers load commit, tick, body, commit again, and tick last. Explicit device
DSBs order these groups; the device-direct load helper's order argument does
not provide C++ acquire semantics. A snapshot needs equal nonzero commits,
equal nonzero full-width ticks, valid packed fields and matching body identity.
The final tick read rejects a same-state progress update crossing the body
read. A same-tick stop changes the committed state to Aborted and still wins
over an already-departed participant's task-local consumed proof. A late
departed participant can finish from that task-local proof when the reusable
acknowledgement is invalid or belongs to another task.

## Participant rendezvous and visibility

The graph's active-subtask mask defines all participants. A BEFORE arrival sets
one atomic bit. The last active arrival publishes a request; no participant
executes until the input records have been committed and flushed, the producer
has sealed the phase, and every active participant has consumed that seal.

After kernel execution, each participant drains its pipeline and writes back
only physical cache lines of selected output/INOUT views. View traversal uses
the actual shared Tensor and dtype helpers, including start offsets and
non-unit inner strides. It does not flush an allocation or the entire cache.
Every participant then arrives at AFTER. The producer copies outputs only after
all active participants have arrived; no participant publishes task completion
until all have consumed the output seal.

One phase produces one argument set with the task's active function IDs. Mix
participants do not independently duplicate its metadata or payloads. Existing
FULL/HYBRID/PARTIAL selection remains authoritative: HYBRID includes unselected
argument metadata with zero payload bytes, PARTIAL omits unselected tasks and
arguments, and OFF has no diagnostic rendezvous.

Before any worker receives READY, one AICPU initializer reads the actual callable
signatures, derives selected output masks through the shared AFTER iterator,
and publishes immutable task state. This is the only task-line writeback.
After READY, mutable task words use existing 64-bit GM atomics/device-direct
loads and stores. Cache invalidation can observe immutable fields; cached
writeback must never overwrite another participant's protocol update. Request
and acknowledgement lines each have one writer.

The existing shared dtype mapping is used on host and device. Its CCEC
qualification permits both targets; there is no second dtype table or copied
Tensor wire layout.

## Bounded producer and memory

The common producer exposes try-begin, copy-chunk, commit and abandon operations.
A reservation holds a snapshot and byte cursor at a stable address. Only one
record per lane can be open: metadata count is the publication watermark, so
committing a later record must not expose an unfinished earlier one. Count
advances only after the full selected payload is copied. Abandon rolls back an
open reservation and records loss exactly once.

Each resident service step copies at most 64 KiB, including strided gathers.
Every positive byte budget is valid, even when it splits an element. The copy
invalidates each actual source span before reading. Metadata/arena pressure
returns `WouldBlock` without spinning or incrementing loss accounting. A
nonblocking flush never publishes an open record. Existing blocking entrypoints
reuse the same copy/commit machinery while preserving their bounded waiting and
loss behavior on other backends and the A5 legacy fallback.

Workers are partitioned over actual AICPU service threads. Each partition uses
a bounded observation-order FIFO and at most one open reservation. A request
can be retried without preventing the thread from observing stop/error state.
Only copied bytes, committed arguments or participant departure advance progress;
polling and `WouldBlock` do not refresh the existing watchdog. Concurrent AICPU
progress timestamps use a monotonic, wrap-aware update.

| Allocation | Size and lifetime |
| ---------- | ----------------- |
| Header | 128 bytes per enabled run |
| Task map | 4 × actual task count N bytes |
| Selected task state | 64 × selected task count M bytes |
| Requests and acknowledgements | 128 × actual worker count W bytes |
| Observed ticks and pending FIFO | 12 × W bytes |
| Alignment | Checked 64-byte alignment between regions |
| AICPU service objects | Actual service-thread count × `sizeof(ResidentArgsDumpService)` |

Thus GM storage is approximately `128 + 4N + 64M + 140W`, plus alignment,
with N bounded by the validated task count/window and M ≤ N. It is a tail of the
existing charged runtime arena, outside the original compact SM image bounds.
Capacity and overflow failures occur before metadata publication. The retained
arena/bank owns it through retirement; compact-image rebasing never treats the
tail as SM content. OFF or no selected resident task allocates no image or
service array. Each AICPU service includes the actual reservation size (152 bytes
in the qualified CPU build); sizing uses `sizeof`, not a fixed 64-byte estimate.

## Abort, close and accounting

An error or expired no-progress deadline flags the task aborted. A service
finishes its bounded synchronous chunk, abandons an open reservation and
publishes an Aborted acknowledgement. No aborted phase authorizes execution or
completion. Waiting uses the existing remaining no-progress deadline; abort
does not start another full timeout interval. An unresponsive producer takes
the fatal run-shutdown path instead of fabricating a seal.

Each AICPU participant stops its own producer before worker retirement. Normal
closure requires all participants to finish cleanly, every reservation closed,
no aborted diagnostic task, and the existing all-core retirement proof. A run
that cannot establish these facts withholds settled terminal accounting.
Committed records remain available and loss is distinguished from unknown
counts. Private service storage with an open reservation is retained rather
than freed or overwritten by another run; the retained allocation's own count
is used during cleanup. Existing host/native unknown-outcome ownership remains
in force. Allocation lifetime alone does not prove that a copy stopped.

## Verification and limits

CPU tests exercise exact identity, missing/late participants, independent lane
slots, one publisher, all-departure release, physical strided views, bounded
chunking, back-pressure, rollback, premature commit, close accounting and charged
arena-tail admission. Executable mutations must fail these assertions; a source
pattern match is not stop or visibility evidence.

The existing Program scene verifies a producer writes 5, its consumer reads 5
and writes 6 in the same backing under FULL/HYBRID/PARTIAL. The existing Mix
scene checks one complete task record set with all active function IDs and
independent per-lane numerical payloads. Joined-launch attribution keeps its
refilled run identities and payload checks. The framework queue consumer keeps
OFF/FULL/HYBRID/OFF, exact positive payload checks, owners, guards and caller-stream
reuse after Worker close on all four architecture/runtime combinations.

Actual CCEC AIC/AIV compilation and AICPU linking check the production dependency
surface. Onboard tests must separately establish cache visibility, task-boundary
ordering, partial/error shutdown and native ABI results for the final candidate.
The historical ordinary A5 scheduler timeout remains unclassified; successful
ArgsDump qualification does not classify or fix that earlier event.
