# A5 Host Build Graph calendar scheduler

The Host places each kernel block and role on an AICore and uploads a calendar
for each physical lane. A lane executes its rows in order. Ordinary tasks and
Graph invocations can appear in the same submission. The Host partitions the
outer DAG by dependency and uses a separate range for each Graph body.

## Image and execution

| Storage | Contents and operations |
| ------- | ----------------------- |
| GM | Per-lane range indexes, calendar rows, bound kernel entries and arguments, and dependency descriptors. AICPU materializes a Graph's arguments and publishes a prefix watermark after flushing the bindings. |
| SSBUF | Per-cluster rendezvous tokens for MIX roles and eligible blocks of the same consumer, and the block context read by each kernel. |
| SoC SRAM | Run-wide START and EXIT, logical DONE counts, defer indicators, and range tail arrivals. |
| SPR | Each core acknowledges quiescence through its own COND nonce. |

Several invocations can share the placement of a Graph body. Each invocation
has its own argument bindings and completion targets. A lane waits for the
required prefix of bindings before reading a body range. At a range end, it
checks the exit dependencies assigned by the Host before publishing its tail
arrival. A Graph shell receives completion credits from its body's sinks and
publishes completion only when all those credits have arrived.

Within a range, a lane checks the dependencies attached to its row and calls the
bound kernel with the block index and block count in its invocation context.
A MIX block uses a local SSBUF rendezvous across its participating roles.
For `sync_start`, the Host places the entire cohort on distinct compatible lanes.
Each block can enter after its own lane and dependencies are ready. The kernel
handles any synchronization between blocks of the cohort.
The kernel's output barrier precedes completion publication.

### Graph boundaries

Graph invocations in an A5 HBG run execute serially. The first range on each
lane in the phase following a Graph waits for that Graph's shell to reach
logical completion, even when the submitted DAG has no dependency between the
two invocations.

For kernels inside a Graph, every deferred wait (`defer_wait`) must settle
within that invocation. A deferred wait must never depend on a later Graph
invocation making progress. Express dependencies between Graph invocations as
explicit DAG edges.
If an independent ordinary task produces the signal on the same device, make it
a predecessor of the Graph so the Host schedules it in an earlier range.
If that task must first observe the defer registration inside the Graph, put it
inside the same Graph so both operations can execute before the shell completes.
Place a task that publishes a deferred condition before any same-lane task that
waits for that condition and express the constraint in the submitted ordering.

### Peer DMB return hints

On A5 onboard, the planner can attach DMB return hints to existing waits in
a single ordinary top-level phase. Selection uses the DAG, placement and
available AIV senders. A hint does not change the counted dependencies, MIX
rendezvous or `sync_start` resource rules. Graph and predicated execution use
the ordinary calendar wait.

After the kernel's output fence, the producer publishes either a SoC SRAM DONE
credit or its actual defer registration. Selected AIV rows then write a token
to the subscriber's DATA_MAIN_BASE via MTE3. For multiple receivers, AIV
members that complete directly can broadcast once they observe the task's
DONE threshold. A returning deferred member can prompt handling of its pending
registration. AICPU binds receivers and prepares batch addresses before START.

The receiver services its original unresolved DONE and defer waits on entry,
on a DMB change, and after 300 SYS_CNT cycles without a check. A deferred
producer can return and send its hint before the dependency is satisfied. A
lost hint delays a check only until the next periodic probe.
A cross-core single-input wait on an AIC producer uses the same immediate
and periodic SRAM checks without accessing DMB. When hints are disabled, the
same selected receiver rows keep that polling cadence.

See the [peer DMB design](../../../../docs/zh-cn/design/percounter-peer-dmb.md).

## SoC SRAM capacity

The CANN `aclrtCntNotify` resource leases 32-bit words in SoC SRAM. The device
accesses each word through three 32-bit MMIO aliases. SET writes or reads the
word. ADD issues one atomic increment per command. OR sets a mask of bits
atomically. ADD and OR have no return value. A consumer observes the resulting
word with a separate load from SET. The kernel output fence precedes either
completion command on the producer.

The Host leases all required SoC SRAM words for DONE, range tail and
fan-in counters. Quota exhaustion fails bind before launch. This makes the
counter demand visible at the point of failure so it can be investigated.
The `complete SoC SRAM lease required` error reports `requested`, `received`
and `status`. Check the requested slot count and the SDK allocation error in
the preceding `scheduler notification acquire failed` message when diagnosing
quota exhaustion. The AICPU clears the calendar counters before it releases the lanes.

One additional SoC SRAM SET slot carries START and EXIT for all workers in the
run. Each lane samples its previous value before publishing its startup report.
After discovery and calendar initialization, the AICPU leader publishes the next
even generation with a direct store to SET. The AICPU publishes EXIT to the
same address with the low bit set. The high bit is set when any core registers
a real deferred condition. Consumers sample this sticky bit after their waits.
A lane that sees it invalidates its data cache before subsequent kernels so
ordinary rows can also observe data changed through asynchronous execution. If discovery fails, it first records that
EXIT generation for each worker without a validated register window. The Host
upload clears these per-worker slots before launch. A late worker that first
samples the current EXIT can acknowledge its matching slot and return. The
run still reports incomplete retirement for those missing windows.
The lease outlives every worker using it. START and EXIT use CntNotify.
When a calendar binds peer notifications, the AICPU samples the subscribed
workers' DATA_MAIN_BASE values before START. Those registers are reserved for
peer notification tokens during execution. The AICPU retains the peer group
through teardown until all discovered cores acknowledge their fresh COND nonces.

## Completion and defer

One SoC SRAM word holds a 24-bit completion count and up to eight sticky defer
bits. Tasks can share the count when the Host can prove their completions advance
in order. Each executable task has its own defer bit within the slot.

1. After the output barrier, an execution with no registered signal issues a
   DONE ADD in SoC SRAM. The scheduler can take this path after observing
   that a single monotonic counter wait is already satisfied. A pending
   deferred execution publishes one record in GM and sets its task's defer bit
   in SoC SRAM.
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
roles defer. Each registration has its own owner and can contain up to 64
signals.

For an independent multi-block consumer with multiple predecessors, the Host
can assign the dependency checks to distinct consumer lanes. Each assigned lane
must be able to reach the consumer's wait without relying on other consumer rows
to execute. A lane with multiple assignments visits its pending predecessors
in turn. Each completed predecessor contributes once to a shared SoC SRAM
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
when many lanes poll the same SoC SRAM counter. A helper checks promptly when
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
counts, completion and lane retirement. The evaluator predicts completion
from the latest ready block in a cooperative cohort and uses the measured
remaining duration from its latest kernel entry. The worker caches the calendar
with the smallest estimated end time in memory.

A repeated Graph in the same submission samples its last matching invocation.

A Graph submission can supply `GraphSubmitOptions::calendar_hash` to share an
online PGO plan across argument values. The Worker scopes each bucket to a
registered Callable and a Graph ID. Each distinct scheduling structure and
core topology in the bucket trains its own plan. A later bind can reuse a
published plan and binds its current arguments and counter addresses. Without
the option, Graphs use the exact argument-derived key. The caller controls
which arguments share a bucket and should compare warm timings for its
representative members.

Within a registered Callable, a Graph ID and equal Config must record the same
body structure in each bind. Put construction choices such as a shape-derived
block count in Config equality. Before any Calendar lookup or training, bind
reports the first structural difference against the previously recorded body.
The check applies to both explicit buckets and the exact argument path.

Calendar cache diagnostics label each `MISS`, `REUSE`, `HIT` and `STORE`
as `kind=graph_bucket`, `kind=graph_args` or `kind=outer`. The `key` field
identifies the lookup key in logs. A successful `STORE` records one trained
plan, and `REUSE` denotes an in-flight initial plan.

Chip Swimlane is a separate diagnostic stream. It can display actual kernel runs
and scheduler events for all invocations, including the rows for which online
PGO did not request samples.

Tasks tagged with `CoreTaskArgs::set_task_timing_slot` record kernel start and
return with Chip Swimlane disabled. For each slot, the AICPU merges the earliest
start and latest return across retired lanes. Tasks in Graph bodies contribute
their own tags. Repeated tags produce one window on the device phase timeline.
