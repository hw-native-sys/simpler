# A5 resident scheduler implementation contract

See the [architecture](../a5-hbg-spmd-design.md) for scope. The rules below
describe the retained implementation, not a proposed development sequence.
Paths and identifiers refer to `src/a5/runtime/host_build_graph/` unless noted.

## Ownership and admission

1. Existing ready pop or direct refill uniquely owns a logical task. Cooperative
   tasks leave the intrusive queue once; block distribution does not requeue the
   same node. Generic pop is not an expected-task conditional removal operation.
2. Ready pop or direct resolve transfers unique ownership to the pending
   admission owner. Bootstrap publishes the immutable INITIALIZED record before
   the candidate is visible. Commit publishes ADMITTED; RETRY preserves the
   candidate, and errors must not silently drop it.
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
| Ordinary single wave, 1<N<=eligible positions, without `sync_start` | Membership rank assigns one block per worker or MIX cluster; no shared claim cursor | Physical retirement and MIX close seals complete the local obligation |
| Ordinary multi-wave N>eligible positions | FAA on `claim_tickets`; t<N owns block t and confirms a valid claim in `next_block` | Confirm exhaustion atomically or through the final claimant's certificate, then retire local obligations |
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

On the shared-cursor paths, only the final valid claimant publishes `claim_seal=N`. The certificate occupies
an isolated atomic-only region and is published with atomic exchange. A positive
bypass hint is confirmed atomically before it can authorize closure; a missing
certificate retains the original strong cursor-read path. Reservation tickets
that overshoot N cannot publish this certificate.

`LocalContext.block_num` remains the original N; `block_idx` is the owned index.
Tensor pointers are not adjusted a second time. Pure C/V retains at most two
unretired claims per worker and one executing kernel. A `sync_claimed` bit
survives early return until that obligation closes, preventing a second claim.

## Mix group lifecycle

```text
claim block -> publish descriptor/sequence -> all required resource lanes arrive
            -> active lanes prepare and execute -> each lane releases
            -> AIC aggregates one block completion -> mailbox reusable
```

One cluster has one in-flight group. The descriptor binds task, order, block,
epoch, cluster, group sequence and fixed slot-to-lane placement. Each lane owns
its acknowledgment region. ARRIVED, kernel return and RELEASED are distinct.
The resource mask is `active_mask | AIC`. A 1C1V group coordinates two lanes;
1C2V and 2V coordinate three, including the AIC coordinator. Unused AIV lanes do
not acquire this group's resource obligation. All workers still participate in
run bootstrap and shutdown.

Each AIV observes its own assignment sequence. The descriptor records the
previous group that used each lane, so skipping an unrelated two-lane group
does not manufacture a missing arrival or permit reuse of a live descriptor.

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
| Claim certificate | Final valid claimant publishes by atomic exchange in an isolated region; positive hints require atomic confirmation |
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
| TaskControl | 384 bytes; claim tickets at 24, valid claims at 128, completed blocks at 192, admission signal at 256 and atomic claim certificate at 320 |
| Admission control | 128 bytes; lock and tail on separate 64-byte regions |
| Admission record | 64-byte header including the Host-planned per-type block prefix at offset 40; bitmap follows, stride rounded to 64 bytes |
| Group mailbox | 384 bytes: 128-byte descriptor, 64-byte publication region and three 64-byte lane acknowledgments |
| Close seals | One 64-byte atomic-only cell per Mix task per candidate cluster, retained until run teardown |
| RunControl | 448 bytes; dynamic counters begin at 192, error payload at 256, identity at 320 and error signal at 384 |

Topology facts may be cached until run reinitialization. A negative ready probe
may suppress duplicate discovery only in the same fill call, after all applicable
queues were checked and no pending/candidate/local READY work exists. It grants
neither ownership nor entry permission, does not cache emptiness across turns,
and does not hide invalid scheduler metadata or delay obligation servicing.

AICPU discovery validates compact worker identities and complete clusters before
READY publishes an immutable topology marker. Workers use that proof to build
their private membership cache; general layouts retain the checked scan. Static
configuration validation is reusable only inside one bounded service call.
Error, epoch, phase, slot and task identity checks remain live.

## Native evidence and compatibility

Execution trace indices include block and active subtask rank. Kernel entry
captures identity from the actual payload; Host export preserves independent
`hbg-execution-identity-v1` records. Predicate skips use `executed=false`.
Compact worker IDs in swimlanes are not physical core IDs in topology records.

L4 additionally exports `hbg-native-protocol-v12` (v14 for Host-seeded ordinary CSR, or v13 for native graph identity): frozen graph/members, admission,
claims, entry permits, LocalDone, group publication/release, completion and terminal
state. Readers in `simpler_setup/tools/hbg/` check per-worker sequences, exact
execution joins, ownership and causal edges from observed values and atomic
tickets. Cross-core timestamps are not a total-order proof.

Only the current v12/v13/v14 schemas are accepted. Producer and reader must use
matching contracts; historical and experimental formats are rejected. Admission
records and close seals use 64-byte alignment; shared admission control separates
its lock and tail into 128-byte regions. Ordinary multi-block tasks require
admission. Single waves use membership-based block assignment; larger waves use
tickets. N1 and sync_start do not use tickets. Terminal validation checks the
completion total and the closure evidence for the selected claim policy.
MIX membership follows its resource mask and records per-lane predecessor sequences.

Native Graph v13 includes `node_map` and `graph_source` for an independent topology
audit. All schemas validate member_rank_offset against a per-resource-type block
prefix reconstructed from frozen task shapes; v14 also applies cluster alignment
to even partial AIV waves in mixed ordinary graphs. All tasks use the current cooperative
admission rules. Execution identities use only the strict v1 format.
The strict CSA gate additionally matches independent dep-gen, original kernels,
block counts, all fan-in and the full task/block/slot set. Truncated/missing
evidence fails the gate even when golden values match. Failed runs require their
failure evidence, not a fabricated successful `protocol_verified=true`.
