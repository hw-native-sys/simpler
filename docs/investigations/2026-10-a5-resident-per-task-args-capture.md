# A5 HBG task-time argument capture

A5 HBG's AICore scheduler owns task dispatch and completion. Its AICPU
participants copy selected arguments on request; they do not schedule tasks.
The public selection modes and artifact format are described in
[ArgsDump](../dfx/args-dump.md).

## Task boundaries

Each selected task has one BEFORE latch, a returned-participant count and one
AFTER latch. The task's existing logical block count and active AIC/AIV mask
supply the expected participant count; zero and counts above `INT16_MAX` are
rejected during planning.

The first participant claims BEFORE, publishes its worker/generation and sends
a copy request. No participant enters the kernel before the BEFORE seal. Later
blocks can observe this same latch after earlier blocks retire: there is no
barrier requiring all blocks to be resident together.

Every participant finishes its kernel pipelines and writes the selected output
cache lines back before incrementing the returned count. The increment and its
ordering barrier publish that participant's writes. Only the final return
requests AFTER. It holds its existing scheduler completion until the copy is
sealed, so normal block retirement cannot release the task's dependents early.
A failed output publication does not count as a return.

The scheduler's existing GM atomics own mutable phase words. A5 atomic-written
state requires atomic confirmation rather than a bare `ld_dev` observation;
explicit DSBs order metadata and payload publication. AICPU invalidates and
reads these words but never writes back their cache line. Its output-role mask
occupies a separate immutable cache line and is published before READY.

## Identity and source addresses

Host planning uses current task metadata for ordinary tasks and the existing
native Graph plan for materialized child arguments. Requests name the private
scheduler task reference; the captured record carries the public `TaskId`,
including a native child's parent/local identity. No business identity or
cross-invocation sequence is added.

A participant has one request and one acknowledgement line. The complete
existing run epoch, task reference, stage and dispatch generation identify the
phase. A request tick advances for each bounded copy step. A consumer probes
both commit and tick around the acknowledgement body, so a reused line cannot
produce a torn or stale seal.

AICPU validates complete tensor/scalar source regions before invalidating or
reading them. Native child tensors are read only after AICore bootstrap has
materialized them. Output roles are determined from registered callable
signatures before READY without dereferencing unbound child tensors.

## Storage and back-pressure

The table is a tail of the existing scheduler-state allocation and is charged
to that same reservation and pending publication. It adds no resource owner.
The image contains a 128-byte header, a 4-byte selected-index map per scheduler
task, a 192-byte entry per selected task, and per-worker 64-byte request and
acknowledgement lines plus tick/FIFO storage. Arrays are aligned and checked
for arithmetic overflow. Every actual AICPU participant has one copy-service
object with a disjoint worker-request partition.

Payloads use the existing per-lane metadata buffers and bounded arena. One open
reservation per lane protects the metadata publication watermark. Copying is
resumable at 64 KiB, including strided gathers. FIFO service permits other
requests to proceed between chunks. A full metadata buffer or unreclaimed
payload returns WouldBlock; it is neither a completed copy nor a dropped
record. Only actual copy/publication progress extends the watchdog budget.

PARTIAL selects marked tasks and arguments. HYBRID records all task metadata
but copies marked payloads. FULL copies every selected argument. All modes use
the same boundary and reservation protocol.

## Abort and retirement

A waiting participant observes the existing scheduler error and a bounded
progress watchdog. On abort it marks the task failed and waits for the service
acknowledgement, or the bounded abort timeout; neither outcome is a seal.
The copy service abandons only its own open record and reports Aborted. A
committed earlier record remains a record rather than being silently rolled
back.

Each AICPU participant stops its service before shutdown. Diagnostic closure
requires both a stopped service and the existing core-retirement proof.
Unknown retirement preserves resources and leaves accounting unsettled; a
completion-gate arrival alone does not authorize reading or releasing them.

Dump-disabled execution carries a zero image address and follows its normal
launch/completion path. Unsupported AICPU-scheduled shapes are refused only
when argument dumping is requested. AICPU scheduling itself is outside this
feature's support and tests.
