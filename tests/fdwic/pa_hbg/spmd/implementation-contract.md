# A5 resident scheduler implementation contract

See the [architecture](../a5-hbg-spmd-design.md) for scope. The rules below
describe the retained implementation, not a proposed development sequence.
Paths and identifiers refer to `src/a5/runtime/host_build_graph/` unless noted.

## Ownership and admission

1. Existing ready pop or direct refill uniquely owns a logical task. Cooperative
   tasks leave the intrusive queue once; block distribution does not requeue the
   same node. Generic pop is not an expected-task conditional removal operation.
2. A task-local `admission_claim_lock` protects a recheck of UNCLAIMED and the
   publication of CLAIMED. Only its owner may release that lock. PREPARED and
   ADMITTED are published by the same admission owner. RETRY preserves a pending
   candidate; errors must not silently drop it.
3. Host-preallocated records freeze epoch, shape, roles and membership. The
   short commit lock assigns `order_key`, publishes the complete record and log
   cell, then advances a contiguous visible tail. It never spans a kernel,
   partner wait or task completion. Reserving a number does not publish a record.
4. Each worker projects the log onto its memberships and services the earliest
   unclosed obligation. A published membership cannot grow or be reordered.
5. Ordinary, direct and GM Mix entries all pass the final entry gate. A prepared
   payload is not execution permission. A later ordinary candidate can be parked
   to release rebuildable payload storage for an earlier obligation; its owner
   and already-evaluated predicate remain local, without duplicate enqueue.

An ordinary single-block permit linearizes at its final critical bypass read of
the published tail, after servicing the observed prefix. A new cooperative
record published after that read must tolerate the already-permitted finite
kernel. Each later kernel needs a fresh permit; filling two slots does not grant
two durable permissions. Epoch zero is a valid identity, not an unset sentinel.

Two finite direct chains can delay discovery of queued cooperative work. The
current contract guarantees progress for the assumed finite valid workload, not
a fixed upper bound on discovery latency.

## Block claims and local closure

| Path | Ownership operation | Closure requirement |
| ---- | ------------------- | ------------------- |
| Ordinary N>1 | FAA on `claim_tickets`; t<N owns block t and confirms a valid claim in `next_block` | Strongly confirm `next_block == N`, then retire all local obligations |
| N=1 or `sync_start` | Bounded CAS on `next_block` | No outstanding local execution/publication/resource work |
| `sync_start` member | At most one successful claim at a distinct execution position | A claimant may close after retirement; an unclaimed member waits for exhaustion |
| Mix AIV | Consumes its AIC's published descriptor, never independently claims a block | Observes and consumes its task/cluster close seal |

Ordinary ticket ownership is irreversible. An invalid tail ticket creates no
payload, completion contribution or block owner. Each candidate position takes
at most one invalid ticket, so terminal `claim_tickets` is in `[N, N+P]`, where P
is the candidate AIC/cluster count or twice that count for pure AIV. Ticket
exhaustion alone is insufficient: the last valid claimant may not yet have
confirmed `next_block`. No claim can be transferred to a worker that passed the
task. Preparation failure fails the run instead of reissuing the owned block.

`LocalContext.block_num` remains the original N; `block_idx` is the owned index.
Tensor pointers are not adjusted a second time. Pure C/V retains at most two
unretired claims per worker and one executing kernel. A `sync_claimed` bit
survives early return until that obligation closes, preventing a second claim.

## Mix group lifecycle

```text
claim block -> publish descriptor/sequence -> all three resource lanes arrive
            -> active lanes prepare and execute -> each lane releases
            -> AIC aggregates one block completion -> mailbox reusable
```

One cluster has one in-flight group. The descriptor binds task, order, block,
epoch, cluster, group sequence and fixed slot-to-lane placement. Each lane owns
its acknowledgment region. ARRIVED, kernel return and RELEASED are distinct;
inactive lanes release resources without inventing a kernel execution.

The AIC waits for all required releases before replacing the descriptor. After
it can claim no more work and its groups have retired, it publishes the task's
close seal: last group sequence plus one, or one for a zero-work cluster. AIVs
must consume through that sequence and close their resources before advancing.
Seeing the global claim range exhausted cannot prove that the local AIC has
published its last group. Seals remain allocated per task/cluster until run end,
so mailbox reuse cannot erase a slower reader's closure evidence.

## Completion and errors

Pure C/V retirement contributes one block; a Mix AIC aggregates one block after
lane release. Only the completion increment observing old value N-1 releases
the logical task's wake-list and successors. Duplicate execution is rejected by
ownership and epoch/slot/group identities, not merely by checking a final count.
No-kernel services complete logically without allocating payload slots, changing
kernel generations or consuming another task's trace prefix.

The first-error CAS owns the failure record. Publish contextual identity before
the status, stop new unsafe entries and never manufacture successful completion.
Unknown epoch/order/block/group/worker/lane fields remain explicitly unknown;
their validity bitmap is distinct from numeric values, including valid epoch 0.
Prepare/complete failures identify the actual slot, not whichever slot is nearby.
AICPU reads and logs the full identity at bootstrap, execution and terminal exits.

An empty queue/log, exhausted cursor or completed logical count alone does not
authorize teardown. Outstanding groups, local obligations and trace publication
must drain. A still-running kernel cannot be canceled by clearing its GM state.
Finalize failure poisons the Worker lane and retains borrowed buffers until the
existing close/recovery contract permits release. Retry does not silently reuse
that failed Worker. Error and sequence overflow do not wrap into a new success.

## Memory publication domains

`scheduler_memory.h` defines the actual A5 instructions. Host/simulator C++
memory-order arguments alone do not prove device ordering. Keep these domains
separate; an observed control word does not automatically acquire its payload.

| State | Writer / reader contract |
| ----- | ------------------------ |
| Ownership locks, valid claim/completion counters, tickets | Existing atomic CAS/exchange/FAA; critical confirmation remains atomic |
| Admission phase, log cells and committed tail | `scheduler_publish_signal` uses `st_dev` plus DSB; readers use `ld_dev` |
| Admission record and bitmap | Owner prepares and publishes cache lines before phase/log/tail; reader invalidates and validates after observing publication |
| Mix descriptor | AIC publishes cached payload before atomic sequence; receiver confirms sequence atomically, invalidates payload and checks identity |
| Mix arrival/release and close seal | Atomic generation-bearing publication and critical confirmation |
| First-error status | Isolated bypass signal after identity publication; AICPU writer additionally follows its identity/status writeback contract |

Bypass signal lines have one active writer, complete publication before ownership
handoff, and exclude AICore RMW, cached stores and dirty writeback during the run.
The compact admission log is wholly in this domain: writing one cell must not
write back stale neighboring cells. Conversely, `ld_dev` is only a prefilter for
atomic-published ownership words; use atomic confirmation at protocol boundaries.
Never generalize a signal optimization to cursor, acknowledgment or seal words.

Host sizes all storage with checked products/alignment and uses offsets in device
images, not Host pointers. Important layout contracts have C++ static assertions:

| Object | Retained layout |
| ------ | --------------- |
| TaskControl | 512 bytes; valid claims at 128, completed blocks at 256, admission signal at 384, each isolated by 128 bytes |
| Admission control | Lock and tail on separate 128-byte regions |
| Admission record | 64-byte header plus worker bitmap; stride rounded to 128 bytes |
| Group mailbox | 640 bytes: descriptor, sequence and three independent lane regions |
| Close seals | 8 bytes per Mix task per candidate cluster, retained until run teardown |
| RunControl | 512 bytes; extended identity uses reserved bytes 336-383, error signal begins at 384 |

Topology facts may be cached until run reinitialization. A negative ready probe
may suppress duplicate discovery only in the same fill call, after all applicable
queues were checked and no pending/candidate/local READY work exists. It grants
neither ownership nor entry permission, does not cache emptiness across turns,
and does not hide invalid scheduler metadata or delay obligation servicing.

## Native evidence and compatibility

Execution trace indices include block and active subtask rank. Kernel entry
captures identity from the actual payload; Host export preserves independent
`hbg-execution-identity-v1` records. Predicate skips use `executed=false`.
Compact worker IDs in swimlanes are not physical core IDs in topology records.

L4 additionally exports `hbg-native-protocol-v5`: frozen graph/members, admission,
claims, entry permits, LocalDone, group publication/release, completion and terminal
state. Readers in `simpler_setup/tools/hbg/` check per-worker sequences, exact
execution joins, ownership and causal edges from observed values and atomic
tickets. Cross-core timestamps are not a total-order proof.

Versions v1-v4 remain strictly readable. Ticket thresholds are N>2P in v2,
N>=2P in v3 and N>1 in v4/v5; N1/sync_start never use tickets. v5 also requires
the raw admission claim lock, zero at successful termination. Missing fields
are not synthesized, and old captures without protocol objects remain unverified.
The strict CSA gate additionally matches independent dep-gen, original kernels,
block counts, all fan-in and the full task/block/slot set. Truncated/missing
evidence fails the gate even when golden values match. Failed runs require their
failure evidence, not a fabricated successful `protocol_verified=true`.
