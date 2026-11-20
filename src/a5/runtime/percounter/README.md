# Percounter acquisition and cohort entry

The Host emits two order lists as projections of one explicit `global_order_id`:
AIC and a vector-domain list containing ordinary AIV and MIX tasks. Graph bodies
occupy their shell's place in that order; their counter IDs are separate
identities. Every block of a task has the same rank. A block index identifies an
invocation and is not a scheduling order.

The Host derives that rank from a virtual resource schedule using the actual
AIC/AIV topology, active masks, dependency edges and block counts. Each block
has unit cost; ready tasks compete by b-level, and a task's first virtual start
sets its rank. This supplies an acquisition order, not physical core assignments
or a prediction of kernel duration. Graph bodies use the same planning rule;
their shell reserves the body’s resource types for its virtual duration, and
the body stays contiguous at the shell’s position. Planning state remains on
the Host and adds no fields or bytes to the uploaded region.

`PERCOUNTER_PENDING_CAPACITY` defines X. Each lane owns X fixed positions.
AIV0 can claim either ordinary AIV or MIX into any free position, using one
occupancy bitmap. MIX position j owns SSBUF record j. AIC and AIV1 query these
records directly until they latch their own launch; AIV1 skips MIX entries in the
vector list without consuming their tickets. All records and publication words
fit inside the cluster's last 1 KiB of SSBUF. X=1 is supported; an occupied
coordinator then has no spare position to claim another task.

Every SPMD task and every MIX task has private block claim sectors. Ordinary
and `sync_start` SPMD use the same acquisition rule: a physical owner holds at
most one unretired block of that task. A free slot selects the earliest eligible
task this owner does not already hold. It can skip a held task even while other
blocks remain unclaimed. The new-task cursor moves forward monotonically;
each owned slot retains its task-counter index. On physical retirement the
slot checks that counter before accepting later work, and claims another
block there if any remain. Multiple retired slots are revisited in rank order. Different tasks can occupy the other slots without waiting
for earlier dependencies or execution. MIX claims come only from AIV0, so its
one-held-block rule applies per cluster. A sync task has no valid blocks left
after retirement: launch requires every block's entry commitment.

Intervals between task boundaries contain only ordinary single-block tasks.
Each nonempty interval uses one ordered fetch-add counter. Each claim takes
one position, leaving the other positions available to other lanes even
when the claimant has multiple free slots. The next boundary opens when the
interval is fully claimed, independently of execution or retirement.
Overshooting adds stay within their task or interval counter and cannot
consume a different kind.

Each list ends with one tail segment. The task-counter index is the global
segment index minus the list index. Each boundary task reserves four 128-byte
claim sectors and a separate 128-byte closed-sector mask. Sector `p` assigns
block `p + ticket * 4`. An owner starts at its cluster's sector and tries the
remaining sectors cyclically when that sector is exhausted. The final valid
ticket of each nonempty sector adds that sector's bit without waiting for a
returned value; the Host carries the bits of sectors the block count cannot
reach. A boundary holding all four bits is exhausted. Its segment summary bit
is added afterwards by a lane that reads the full mask while skipping the
boundary, and a compare-exchange on a flag above the sector bits names that
single publisher, so the add never carries into the neighbouring segment's
position. This preserves unique block assignment even when block and owner
counts are not divisible by four. A sync boundary whose block count equals its
list's eligible owner count skips the ticket entirely: block `rank` belongs to
the owner of rank `rank`, so no owner draws and none waits on another's draw.
No owner touches the closed-sector mask there, so such a boundary never reports
itself exhausted and carries no segment summary bit. A mask filled by block
assignment would complete once the owners of the highest blocks had claimed,
stating that the boundary was finished while it still held a block reserved for
a lower rank that had yet to scan it. Its two readers answer that question
directly instead: a lane skips the boundary on the static verdict itself, and
the publication floor releases it when that lane's own claim cursor passes.
Only a full-width sync boundary qualifies: binding a block to one owner
gives up the ticket path's freedom to place it on whichever owner arrives first,
which a cohort taking every owner has nothing to spend and a narrower one does.
A sync task counts its arrivals in one STARS
CntNotify register when the Host could map one for that cohort: a block writes
the unit-increment command and an observer reads the register once, comparing it
against the block count. SRAM carries no same-address serialization, so one
counter costs a reader less than four partitions do. A cohort the Host could not
map keeps four independent 128-byte GM arrival sectors, selected by logical block
index modulo four, where every nonempty sector must reach its expected block
count before the cohort can start. Either way an over-arrival is permanent, and
the entry-commitment token, the per-slot entry flag and the refusal to re-claim
an exhausted task are what keep each block arriving once.
Ordinary AIC, ordinary AIV and MIX sync widths are checked against their actual
eligible owner counts. Nonempty single-block intervals reserve one ticket
sector each, bounded by the smaller of the ordinary single-block task count
and the segment count. Regular entries inherit `NO_SYNC` and need only the
immutable rank/mask pair.

Ordered acquisition prevents later tasks from filling every eligible owner's
positions while an earlier unclaimed task is absent. Skipping that task requires
already holding it; retiring it creates capacity for its remaining blocks.
For the minimum-rank unfinished task, internal predecessors are complete, and
ordinary waves or a width-valid sync cohort can make progress. This capacity
argument also requires the publication and startup protocol below, fair lane
progress, valid kernels and eventual completion of external deferred conditions.

Each segment carries a 64-bit task shape (task ID, block count, active mask and
sync flag). A closed boundary counter proves its preceding ordinary interval
fully claimed, so skipping that segment needs only the shape and counter reads.
Interval bounds and tickets are read only for an open segment; rank and sync
index are read after a successful task claim. This preserves interval
boundaries without repeatedly decoding full metadata for exhausted tasks.

Claimed intervals also publish monotone closure bits, one 128-byte summary
sector per 64 segments. The draw containing a tail interval's final entry adds
that segment's bit; a boundary's bit is added by the lane that wins its
publication flag and therefore trails the draw that exhausted it. Host
initializes empty-tail bits. Readers can skip a closed range with one word
read. An unobserved bit only causes ordinary counter checks. A closure bit asserts allocation, not execution,
dependency completion or SSBUF publication.

When the lane already holds work, an unsuccessful atomic draw returns control
to servicing those candidates before another draw, and advances its scan past
that boundary: a draw fails only once every sector reports its tickets issued,
which is exactly when every block of the boundary has an owner. An empty lane
can continue through exhausted intervals immediately. There is no timing budget
or change to which task an empty position may acquire.

## Shared record lifetime

AIV0 writes a descriptor and release-publishes its generation. Each actual
participant acquire-observes that generation before inspecting the descriptor.
The lowest numbered participant probes fanin and publishes its generation in
`fanin_ready`; it does not wait for other participants to discover the record.
Successful fanin prefixes are cached. A coordinator outside the active mask,
including AIV0 for mask 0b101, neither executes nor contributes completion.

A cohort may enter only after its lane has no unfinished lower-ranked cohort
and the ordinary sync and MIX publication boundaries exclude undiscovered
lower-ranked work. The MIX boundary is acquired before scanning the records
it covers; an acquired bound is cached until a candidate needs a larger one.
Record discovery also uses one acquired publication-sequence ceiling, so a
scan cannot admit a newer same-task notification after missing an earlier one.
The shared head is selected by `(global_order_id, sequence)`. Retirement can
publish another block of an earlier task after a later task's notification;
publication sequence alone therefore cannot determine task priority. A single
active record needs no comparison, while a new publication invalidates the
cached head before selection.
The Host records the first and next relevant boundary for three startup domains:
ordinary AIC sync, ordinary AIV sync and MIX. These indexes reside in the existing
control and task-metadata storage; they are not task-acquisition lists. Each
publication cursor caches its open boundary's block count, rank and successor.
Repeated checks read only that task's closed-sector mask. Once all blocks are claimed,
the cursor follows the successor directly, independently of the fresh-task cursor
and execution completion. Graph-body indexes are available before materialization.
Each participant then publishes an irrevocable `enter_ready` token for the
current generation, identified locally by candidate, entry and generation.
Pure ordinary ready work can still run while that commitment waits. For sync
tasks, a block contributes one start arrival after its entry commitments; all
blocks must arrive before any enters its kernel. For a MIX block, its lowest active
lane observes the cohort's arrival count and publishes start_ready with the
record generation. Other members consume that SSBUF notification. The local
idle handshake still gates execution. The uploaded table carries one address
per cohort, the register's SET/read address; its ADD alias is a fixed
displacement of that address, so a lane resolves the whole slot with one load,
once per cohort it visits, and a repeated readiness probe reads no table at
all.

Once these entry conditions hold, each MIX participant finishes its current
ordinary kernel and publishes `enter_idle` for the generation. This second
promise reserves its UB and pipelines: it cannot start another ordinary kernel
until this MIX invocation returns. All actual participants must publish idle
before any starts the MIX kernel. ENTER_READY alone is insufficient because
another core can publish it during an ordinary kernel. Waiting for all entry
conditions before reserving the cores preserves ordinary work needed by earlier
cohorts. The idle handshake is cluster-local and adds no GM start-arrival atomic.

Each participant stores its entry, function address, generation and active
mask in lane-local state before it release-publishes `enter_idle`. Its kernel
then uses that snapshot. A waiting participant can start after all idle tokens
match or after the shared descriptor is released. Release proves every
participant has latched that generation, even if its kernel has yet to start.

AIV0 releases the publication and its claim slot on return from its own MIX
kernel. When AIV0 coordinates without executing, it releases the publication
once every active lane has latched the launch. The next claim may publish a new
generation while peers execute their previous snapshots. A peer returning to
the scheduler ignores an already latched generation and can claim ordinary
work. The same task ID can recur with a new generation for another SPMD block.

Generation checks enclose shared descriptor reads, including nonparticipants
that can race reuse. SSBUF has no atomics. Each token has one writer.

The Host plans monotonic CntNotify chains for top-level multi-block tasks.
A task can extend a chain whose last task is a strict DAG ancestor, as long as
the accumulated credit count fits in 32 bits. Planning collapses ancestry
through credit-free nodes with one relevant frontier and bounds indirect
search to 256 fanin edges per candidate. It creates a separate chain when no
reusable tip is found within that search. Tasks assigned to a mapped chain
share one SRAM register and have cumulative thresholds. The AICPU resets each
mapped chain once before READY. Each physical subtask
increments it once after synchronous completion or after its deferred condition
is satisfied. Each MIX lane publishes its own credit. The fanin observer caches the register address and target
after the first probe, and tests one MMIO value against the expected subtask
count including the prefix of its chain. The AICPU uses the same threshold
for top-level sinks. Later contributions preserve the completion proof for
earlier tasks in the chain.

Unmapped tasks and Graph body tasks publish into the GM counter in their
task record. The Host presets it to `1 - total_subtasks`. Each physical
subtask publishes one credit when its completion becomes final, and the
consumer waits for a positive value. Graph shells receive the sum of their
body sink subtask credits.

The Host folds top-level dependency-only joins with at most one live
consumer when the merged fanin fits the task limit. The consumer inherits
the join's predecessors. For a terminal join, its remaining predecessors
retain their completion obligations through the sink check. Shared joins
and Graph body joins retain their own counters and scheduling positions.
The logical task table remains available to dependency export.

The Host omits predecessors already complete at bind from device fanin rows.
The logical DAG still determines ordering and sink discovery. A task-table
fanin entry points to a GM counter or tags a CntNotify descriptor. Graph body
fanin entries carry local producer indices. Ordinary slots and the MIX
observer's SSBUF record retain the current producer and its SRAM threshold
while waiting. Ready observations advance the producer cursor.

The state allocation is aligned to 128 bytes. The uploaded image includes
the primary completion counters and claim tickets. AICore invalidates its
data cache after READY and before reading the uploaded image.
The SSBUF record remains 128 bytes. Each boundary task uses 640 device bytes for
claim state. Immutable control fields use ordinary loads after the lane's entry
cache invalidation; mutable progress and dependency fields retain bypass or atomic
accesses.
Host, AICPU and AICore share one current layout.
AIV0 publishes cluster-local MIX closure only after all MIX claim boundaries are
closed and every record it owns has retired. A closing publication sequence
carries this single-writer SSBUF token; peers acquire it before disabling MIX
discovery. Startup clears it before publishing readiness.

Lanes publish task completion only; they keep no whole-graph progress count and
reach no completion verdict of their own. A lane can end its scheduler loop when
its local list is exhausted, all owned positions and retirement revisits are
empty, its MIX domain is closed (or the run has no MIX), and it has no pending
MIX launch or deferred completion record — that is a statement about the lane, not about the run. It
then enters the register EXIT wait. Active lanes poll EXIT and the shared error
word at the existing observation interval; they do not reread the run-wide
completion and async counts or maintain another progress watchdog.

The AICPU supervisor owns the completion verdict: it waits for every top-level
sink the Host found to read published and for `async_registered ==
async_settled`, then the ordinary retirement path signals EXIT. AICPU also
retains watchdog and EXIT/nonce-handshake ownership. Initialization and startup
waits retain their local time bounds; a started run is supervised by AICPU.

Lanes check EXIT and the shared GM error word at startup and periodically
during scheduling. AICPU also observes errors and broadcasts EXIT. Neither
publication is an immediate admission fence: a lane may start work before its
next observation. EXIT does not preempt a running business kernel. A kernel
that cannot return relies on the device operator timeout and the platform's
failed-run teardown. Such a run fails, and its poisoned Worker cannot accept
another run. This failure contract does not promise graceful cohort retirement.

## Task arguments and execution contexts

The Host image packer directly produces one immutable argument vector for
all executions of each ordinary task. Tensor slots contain device addresses of
128-byte tensor descriptors; scalar slots contain their 64-bit values. The vector
has 48 slots (384 bytes), with no context suffix pointers. Functions remain in
the task counter. Ordinary vectors are part of the task image. Graph vectors are
part of each invocation's device-only execution storage. In both layouts the
payload's Scalar span aliases the vector's Scalar tail; there is no second
runtime Scalar pool or scheduler-owned argument copy. Outer Graph boundary
Scalars remain in their own image span because they feed materialization rather
than a kernel and can exceed kernel argument limits. Vectors remain resident
until the run retires, including deferred completions.

Static Graph order, shape and total-subtask completion seeds are available at
bind. Graph body fanin references encode local producer indices. The Host
presets each GM counter before upload. AICPU materialization binds each
body task's descriptors and writes Tensor pointers and Scalars directly to its
final vector, then publishes
`metadata_ready` after writing back both. It does not wait for fanin. Lanes never
read body arguments before that publication. Graph roots observe shell fanin and
sinks also update the shell counter. The Host publishes the Definition carrier
used by the AICPU materializers.

Each physical C/V0/V1 lane owns a 64-byte SSBUF context slot in the cluster's
`[2816, 3008)` window, inside the scheduler's existing last-KiB reservation.
The slot contains an 8-byte LocalContext and a 48-byte GlobalContext. Startup
initializes the physical AIV role, run identity and lane-owned async registration
scratch reference once. Immediately before dispatch, the selected block's logical
index and width replace LocalContext. The input acquisition barrier also orders
these stores. A claimed or waiting candidate does not change the current context.
The slot remains stable until the synchronous kernel call returns.

Percounter owns its kernel-facing `intrinsic.h` and `async_kernel_api.h`. Context
accessors read SSBUF directly, independent of `args`; standalone AIV kernels use
the physical mixed launch's subblock register to select V0 or V1. Simulation
injects process-owned context binding/access callbacks into the runtime and each
child kernel, isolating threads and loaded DSOs. Kernels must be rebuilt for the
selected runtime's current ABI. HBG retains its own GM-context ABI.

AICore dispatch uses the final vector's GM address. Ordinary tasks derive it
from the image base; Graph metadata publishes the address before readiness.
Dispatch uses the shared argument vector in place. Each lane issues
nonblocking preloads for the populated argument prefix and the contiguous
Tensor descriptors during ordinary or SPMD resolve, or after first publishing
MIX entry readiness. Graph body preloads follow metadata publication. Dispatch
orders dependency observations and the lane's context writes with a DSB.
Predicate evaluation retains its own cache-line maintenance.
Successful ordinary fanin is cached by task identity across blocks.
After sixteen idle passes, repeated failed fanin probes run once every four
passes. Newly resolved tasks probe immediately; shared publication, cohort entry,
claims and deferred completion remain serviced on each scheduling pass. Dependency-only
blocks without predicates publish completion after fanin without a kernel call;
a barrier orders the observed fanin before completion publication.

## Deferred completion

Deferred completion waits for peer-card data. Same-card producers and consumers
use explicit task dependencies; COUNTER waits do not replace those edges. A lane
initializes its registration scratch once per run. A kernel can register up
to four conditions; at return, only actual registrations are copied into
persistent records, allowing the payload and scratch to be reused immediately.
Each record retains its block's actual completion address across slot and
payload reuse. Each owner polls its records and publishes completion only after all conditions
hold. Independent records can settle out of registration order. A lane cannot
stop polling while it owns an unsettled record. Global run completion still
requires all physical blocks and all registered conditions to have completed.

Each MIX participant contributes completion when its own subtask finishes.
A deferring participant's persistent record contributes once when the
condition holds. Graph-sink
contributions also update the shell's counter.

An ordinary slot is held through its lane's physical return. A MIX publication
slot follows the launch handoff described above. A released slot's counter index is acquisition bookkeeping, not an additional
claimed block or a receiver task copy. A publication-floor cursor advances past
a boundary only once this lane can no longer claim from it, which a ticket
boundary proves by exhausting its sectors and a statically assigned one by this
lane's new-task cursor having passed it — that cursor passes a boundary only
after this lane has taken the block reserved for it, or established that it owns
none there. Exhaustion is the weaker fact on a static boundary and is not
available on it; the floor never uses the two interchangeably.
An unready task is not evicted into a GM pending queue. A record for a kernel
that has already returned holds only its deferred completion conditions; it is
not another unstarted scheduling candidate.

## Chip swimlanes (L1–L4)

`enable_chip_swimlane` uses the standard profiling levels:

| Level | Recorded data |
| ----- | ------------- |
| 0 | Disabled; the uninstrumented lane implementation is compiled separately |
| 1 | Kernel execution and final input acquisition, streamed through the shared collector |
| 2 | L1 plus the Scheduler View with task/block milestones |
| 3 | L2 plus nested AICore Scheduler function intervals of at least 1 microsecond |
| 4 | L3 with every instrumented function interval retained, individual GM/SSBUF operations, cache issues and barriers, and Host orchestration phases |

The Scheduler View groups events by task and block, including Graph body tasks.
Its `Cluster 00 / 2` style labels name additional display rows used when block
lifetimes overlap in the same physical cluster. Marker details identify the
actual worker, cluster and C/V0/V1 role.

| Milestone | Device boundary |
| --------- | --------------- |
| Claim start | The claiming lane has space in its local slots and starts the scan and ticket acquisition that eventually owns the block. Passes that acquire nothing emit no marker. |
| Claim | The claiming lane acquires the task/block in its local scheduler state. For MIX, the timestamp precedes publication of the entry to the participating lanes in SSBUF. |
| Ideal ready | The converter estimates when the last predecessor issues its final completion contribution. It can precede Claim start. The converter leaves the Claim-to-Ready span intact when dependency or completion evidence is missing. |
| Ready | A lane selects the task/block after its dependency, ordering and applicable MIX or sync-start gates. Each participating MIX lane records its own Ready. Argument prefetch can occur earlier. |
| Kernel end | The executing lane returns from its kernel, before the output-store barrier. The marker also carries that lane's kernel start. Dependency-only and skipped entries have no kernel interval. |
| Completion issued | A lane issues its final completion credit to GM or CntNotify. Deferred settlement records its own publisher. |

Completion markers describe individual lane credits. A task's total completion may
include contributions from several blocks. A consumer observes completion when
its own dependency checks succeed.

The task-event capture lists Host-completed task IDs and the dependency-only
controls folded during binding. The converter removes Host-completed
predecessors and expands folded controls in the captured dependency graph
to recover the producer credits the consumer actually waits on.

Each active worker has its own **AICore Scheduler** lane, grouped by cluster
and C/V0/V1 role. Function names are percounter's own; task IDs retain the HBG
layout shared by the two runtimes. The standard converter nests kernel
intervals inside execution/service functions and computes exclusive time without
double-counting parents. Worker View retains the original kernel intervals.

L2 reserves up to 65,536 task-timing records per worker, bounded by the graph's
block count, and up to 16,384 scheduler task events per worker. Each worker
partition starts on its own cache line. Task-timing overflow fails the export.
Milestone overflow is reported as `percounter_task_events.dropped`. L3/L4
additionally reserve 8 MiB per worker slot. A compact
Scalar row occupies one 8-byte word: a 32-bit start offset from the worker's
window, a 16-bit duration, an 8-bit phase and a 7-bit nesting depth. This retains
the full clock precision. Intervals outside those bounds use five words with
the complete timestamps, task reference, phase/depth and call count. The buffer
holds up to 1,048,576 compact rows or 209,715 extended rows. Only headers are
initialized/uploaded; the Host downloads committed words after device retirement
and derives the row count while decoding. Scalar overflow is exported as
`capture.truncated=true` with a dropped count; the analyzer does not claim a
complete time partition for that worker. L0/L1 allocate neither record tail.
Profiling storage belongs to the run's scheduler-state allocation and is reset
on reuse; each profiled AICore entry resets its local recorder binding.

L4 memory leaves preserve the protocol's existing accesses and barriers. Returning
atomics and loads end when the result is locally consumable; stores and atomics
whose results are unused record issue time. DCCI issue and DSB completion are
separate events. L2+ task records are staged locally and written after the
business completion signal; Scalar records use private cached storage. A terminal
cache clean and barrier publish the records before retirement. L1 batches cache
publication at full-buffer handoff and final partial-buffer retirement, retaining
the shared collector's bounded buffer rotation.
The detailed stream starts after the startup handshake and topology acquisition;
it ends at record publication, excluding the final retirement wait. Profiling
perturbs execution, so use level 0 for latency measurements.

`enable_chip_swimlane=1` records each executed kernel through the shared
`ChipSwimlaneAicore` collector. The standard `swimlane_converter` accepts
`runtime=percounter` and decodes the shared HBG task identity, including Graph
body tasks. Host runtime selection reports mode 5.

The receive timestamp is entry to execution using the task's immutable arguments;
start is immediately before the kernel call, and end follows its output-store
barrier. Local setup therefore includes execution-time cache acquisition and
record reservation, but excludes argument packing and fanin waits.
Predicated-off and dependency-only tasks do not emit kernel records.

Task identity is captured from the selected task when its kernel returns and
retained in the pending timing record. Publication copies that captured identity;
it does not reread a reused task or execution context. No async token is stored
in the argument vector. A self-scheduling core reserves from its cached buffer without
observing the shared head while space remains; only the first reservation and
full-buffer replacement need to observe it. This rule belongs to the
self-scheduled path, whose core alone publishes the full count.

Each buffer holds `PLATFORM_AICORE_BUFFER_SIZE` records. A self-scheduling core
publishes its committed count only when full or when the lane exits with a
partial tail, and waits for replacement when a full buffer has no space.
The shared AICPU collector uses the existing rotation, ready queue, Host drain
and free-buffer refill paths; there is no per-run 1024-record limit. The final
partial buffer uses the same flush path. A replacement wait that exceeds the
platform DFX backpressure budget reports a runtime timeout. L1 does not record
scheduler phases. Profiling disabled uses the same runtime binary.

The common device-timing output also reports `AICore` wall independently of L1:
the earliest actual kernel start to the latest actual kernel end. It includes
scheduling and dependency waits between those endpoints, while excluding lane
setup, the handshake and final scheduler drain. See
[Host trace timing](../../../../docs/dfx/host-trace.md) for the metric boundary.

## Deferred completion regression tests

Default A5 packages include percounter, and the CPU suite builds the default,
X=2 and X=4 configurations. Deferred peer-notification scenes remain manual
because they require two devices. An explicit build can select only this runtime:

```bash
python -m simpler_setup.build_runtimes --platforms a5 --runtime percounter
pytest tests/st/a5/percounter/defer_wait --platform a5 --manual include
```

Each peer-notification scene requires exactly two A5 devices and checks the
received payload before its local consumers complete. Submit each case in its
own allocation after preparing its kernels. Waiter counts just below, at and
above the AIV count exercise physical return without tying up execution slots.
CPU protocol tests inject external completion from their test driver; they do
not schedule a later local kernel as the hidden notification producer.
