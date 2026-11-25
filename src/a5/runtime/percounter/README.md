# A5 calendar scheduler

The Host places each kernel block and role on an AICore and uploads a calendar
for each physical lane. A lane executes its rows in order. Ordinary tasks and
Graph invocations can appear in the same submission. The Host partitions the
outer DAG by dependency and uses a separate range for each Graph body.

## Image and execution

| Storage | Contents and operations |
| ------- | ----------------------- |
| GM | Per-lane range indexes, calendar rows, bound kernel entries and arguments, and dependency descriptors. AICPU materializes a Graph's arguments and publishes a prefix watermark after flushing the bindings. |
| SSBUF | Per-cluster rendezvous tokens for MIX roles and eligible blocks of the same consumer, and the block context read by each kernel. |
| CntNotify SRAM | Run-wide START and EXIT, arrivals for explicit `sync_start`, logical DONE counts, defer indicators, and range tail arrivals. |
| SPR | Each core acknowledges quiescence through its own COND nonce. |

Several invocations can share the placement of a Graph body. Each invocation
has its own argument bindings and completion targets. A lane waits for the
required prefix of bindings before reading a body range. At a range end, it
checks the exit dependencies assigned by the Host before publishing its tail
arrival. A Graph shell receives completion credits from its body's sinks and
publishes completion only when all those credits have arrived.

Within a range, a lane checks the dependencies attached to its row and calls the
bound kernel with the block index and block count in its invocation context.
A MIX block uses a local SSBUF rendezvous across its participating roles. An
explicit `sync_start` also waits for the entire cohort's CntNotify START count.
The kernel's output barrier precedes completion publication.

### Graph boundaries

Graph invocations in a Percounter run execute serially. The first range on each
lane in the phase following a Graph waits for that Graph's shell to reach
logical completion, even when the submitted DAG has no dependency between the
two invocations.

For kernels inside a Graph, every deferred wait (`defer_wait`) must settle
within that invocation. A deferred wait must never depend on a later Graph
invocation making progress. Express dependencies between Graph invocations as
explicit DAG edges.

## CntNotify capacity

The Host requires a complete CntNotify lease for START, DONE, range tail and
fan-in counters. Quota exhaustion fails bind before launch. This makes the
counter demand visible at the point of failure so it can be investigated.
The `complete CntNotify lease required` error reports `requested`, `received`
and `status`. Check the requested slot count and the SDK allocation error in
the preceding `scheduler notification acquire failed` message when diagnosing
quota exhaustion. The AICPU clears the calendar counters before it releases the lanes.

One additional CntNotify SET slot carries START and EXIT for all workers in the
run. Each lane samples its previous value before publishing its startup report.
After discovery and calendar initialization, the AICPU leader publishes the next
even generation with a direct store to SET. The AICPU publishes EXIT to the
same address with the low bit set. If discovery fails, it first records that
EXIT generation for each worker without a validated register window. The Host
upload clears these per-worker slots before launch. A late worker that first
samples the current EXIT can acknowledge its matching slot and return. The
run still reports incomplete retirement for those missing windows.
The lease outlives every worker using it. Retirement still requires each
discovered core to echo its fresh COND nonce before the AICPU releases that core.
The calendar leaves DATA_MAIN_BASE available during startup, execution and teardown.

## Completion and defer

One CntNotify word holds a 24-bit completion count and up to eight sticky defer
bits. Tasks can share the count when the Host can prove their completions advance
in order. Each executable task has its own defer bit within the slot.

1. After the output barrier, an execution with no registered signal issues a
   DONE ADD in CntNotify SRAM. The scheduler can take this path after observing
   that a single monotonic counter wait is already satisfied. A pending
   deferred execution publishes one record in GM and sets its task's defer bit
   in CntNotify SRAM.
2. A successor reads the DONE word in SRAM. It checks the count first. If the
   count is short and the task's defer bit is set, it visits the task's
   registration records in GM and can claim any unowned record with a CAS.
3. The owner alone polls the signals in that record. Once all of them are
   satisfied, it issues one DONE ADD for the producing block and role. If the
   producer is a Graph sink, the owner also issues its shell credit.
4. The remaining successors observe the DONE threshold and can proceed. A
   phase exit services registrations for tasks without a polling successor in
   that phase.

Each lane's invocation context contains an SSBUF marker for access to its
registration scratch. The kernel sets the marker before validating the deferred
registration context. After the output fence, the scheduler checks the GM
registration only when the marker is set and clears it on a successful check.

A single predecessor can have multiple registrations when different blocks or
roles defer. Each registration has its own owner and can contain up to four
signals.

For an independent multi-block consumer with multiple predecessors, the Host
can assign the dependency checks to distinct consumer lanes. Each assigned lane
must be able to reach the consumer's wait without relying on other consumer rows
to execute. A lane with multiple assignments visits its pending predecessors
in turn. Each completed predecessor contributes once to a shared CntNotify
counter. The rest of the consumer lanes only wait for that counter. MIX and
synchronized SPMD retain their per-block entry protocol and dependency checks.

The Host can omit a wait when the calendar proves that the lane has already
observed the required completion, including through a retained dependency.
Within a cluster, eligible consumer blocks can share an input-ready observation
through SSBUF. The lane assigned to service a deferred producer continues to
poll it directly.

A single ordinary phase without Graph invocations can fuse the DONE counts of
independent producers that have one common remaining consumer. Each producer
keeps its own defer bit and a lane that can service its registrations. The
consumer waits for the combined count. Wide groups use a bounded,
cluster-dependent interval between unsuccessful reads to reduce contention
when many lanes poll the same CntNotify counter. A helper checks promptly when
at most two credits remain or its producer has published a defer bit.
The Host proves fusion eligibility while planning and reuses the result on
cache hits. The independence check has a phase-wide edge budget. Groups
without a completed proof use ordinary fan-in.

## Online PGO and tracing

The first run of a shape executes a calendar generated from the submitted DAG
and execution shapes. Sampled rows record kernel start, output barrier
completion, defer registration and defer settlement. These samples distinguish
physical lane occupancy from logical completion.

The Host scores the sampled placement and three bounded candidate calendars.
It considers block submission order, longer-output blocks with wait-aware
priority, and a variant that can fill an idle interval with ready work from the
same resource class. The backfill must preserve the selected task's estimated
start and placements. The planner commits both groups before considering
overlapping resource classes. The evaluator replays each placement's lane and
wait events, including the follower's periodic DONE probe, fused producer
counts, START, completion and lane retirement. The worker caches the calendar
with the smallest estimated end time in memory.

A repeated Graph in the same submission samples its last matching invocation.

Chip Swimlane is a separate diagnostic stream. It can display actual kernel runs
and scheduler events for all invocations, including the rows for which online
PGO did not request samples.
