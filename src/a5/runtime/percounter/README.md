# A5 calendar scheduler

The Host places each kernel block and role on an AICore and uploads a calendar
for each physical lane. A lane executes its rows in order. Ordinary tasks and
Graph invocations can appear in the same submission. The Host partitions the
outer DAG by dependency and uses a separate range for each Graph body.

## Image and execution

| Storage | Contents and operations |
| ------- | ----------------------- |
| GM | Per-lane range indexes, calendar rows, bound kernel entries and arguments, and dependency descriptors. AICPU materializes a Graph's arguments and publishes a prefix watermark after flushing the bindings. |
| SSBUF | Per-cluster rendezvous tokens for the participating roles of a MIX block and the block context read by each kernel. |
| CntNotify SRAM | START arrivals for explicit `sync_start`, logical DONE counts, defer indicators, and range tail arrivals. |
| SPR | Core bring-up, exit and error handling through the shared platform lifecycle. |

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

## Completion and defer

One CntNotify word holds a 24-bit completion count and up to eight sticky defer
bits. Tasks can share the count when the Host can prove their completions advance
in order. Each executable task has its own defer bit within the slot. The Host
requires a complete CntNotify lease to bind the START, DONE, range and fan-in
counters. If the CntNotify quota cannot supply all the slots, the Host rejects
the calendar before launch. The AICPU clears the leased slots before it releases
the lanes.

1. After the output barrier, an execution with no registered signal issues a
   DONE ADD in CntNotify SRAM. An execution with registered signals publishes
   one record in GM and sets its task's defer bit in CntNotify SRAM.
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

## Online PGO and tracing

The first run of a shape executes a calendar generated from the submitted DAG
and execution shapes. Sampled rows record kernel start, output barrier completion, defer
registration and defer settlement. The Host uses these samples to estimate
physical lane occupancy and logical completion separately, then plans a calendar
for reuse. A repeated Graph in the same submission samples its last matching
invocation. The cache is in memory, belongs to the worker and is released with
the worker.

Chip Swimlane is a separate diagnostic stream. It can display actual kernel runs
and scheduler events for all invocations, including the rows for which online
PGO did not request samples.
