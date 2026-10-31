# A2/A3 ready sync-start prestaging proposal

**Status:** design for review; no runtime implementation or new timing result.
**Issue:** [#2727](https://github.com/hw-native-sys/simpler/issues/2727), under
[#2631](https://github.com/hw-native-sys/simpler/issues/2631).
**Source reviewed:** `4321b3d2137d72d1bd6cee7c6f964345a90cb4f8`.

## Existing launch chain and evidence

The motivating HCA consumer has 24 full MIX blocks. Three A3 scheduler owners
have eight complete MIX groups each. The cohort cannot take the single-owner
ready fast path. Its dispatch sequence is:

```text
completion updates the owner's tracker
  -> request / current-attempt ACK
  -> coordinator checks global idle capacity
  -> each owner selects groups and claims its block range
  -> each owner prepares its whole batch, then wmb and publication
  -> stage-done / running seed when gated / software reopen
  -> resume completion and ordinary dispatch
```

The ready global path begins preparation only after idle capacity fits the
whole cohort. Within each owner, an already-prepared subtask waits for the rest
of that owner's batch before publication. The owners themselves work in parallel.
Ready Core payloads start on pickup; the final software reopen is not their
hardware launch signal. Gated payloads use a separate high-half doorbell.

The connected source entry points are
[`dispatch_shape`](../../src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_dispatch.cpp),
[`handle_drain_mode` / `stage_sync_start_cores`](../../src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_completion.cpp),
[`CoreTracker::count_available_blocks`](../../src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_types.h),
and [`try_launch_sync_start_cohort`](../../src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler.h).
They use `LaneMask` at the reviewed revision.

| Evidence and identity | Observation | Meaning |
| --------------------- | ----------- | ------- |
| Historical `15f5cbd9`, A3, one successful L4 round | Owner 0: 6.20us prefix + 6.80us Prepare + 0.58us Publish + 1.90us tail = 15.48us | 13.00us precedes this owner's publish phase; not 15.48us of MMIO |
| Separate `b5862247` BASE/M4 diagnostic calls, one per arm | Successful-round envelope 15.72 -> 14.48us; first entry to first publish 13.90 -> 12.84us; owner-mean Prepare 7.00 -> 6.053us | Local observations, not a repeated local-performance distribution |
| M4 historical same-card L0, 300 calls per arm | Device mean 789.489 -> 784.563us (-4.926us / -0.624%); P95 -9.979us; max -10.480us | Whole-entry improvement observed across separate batches; Host conditions differ |
| DEBUG-call guards added to historical M4, 300 calls per arm | Device mean -1.365us; P99 +0.092us; max +4.500us | No new drain endpoints; retained as a small, unadopted follow-up |

M4 skips 24 unused ready MIX idle-count helpers and three post-stage counter
RMWs in that full-cohort case. It leaves the parameter, slot, ACK, capacity and
publication sequence in place. Ordinary BGEMM/manual-attention mean regressions
remain recorded; M4 has no general adoption verdict. None of these historical
measurements validates this proposal or the reviewed main revision.

The existing Drain DFX record is conditional on reaching stage. Capacity-failed
attempts and follower exits before `stage_go` are omitted. A successful 15us bar
does not bound first-request-to-launch time or the optimization opportunity.

## Proposed admission and state handoff

Reuse the existing gated MIX split placement after a ready task has entered
global drain. Prepare while old running tasks finish, then let their FIN
observations promote pending subtasks and complete the existing cohort rendezvous.
This changes when preparation occurs, rather than predicting task duration.

```text
current ready: wait for all idle -> prepare batch -> publish -> execute
proposal:     prove all placements -> gated prepare/publish
                                     overlaps old running execution
              last required promotion -> one cohort doorbell -> execute
```

The first implementation should require all of these after the original ACK:

- Ready sync-start, no predicate, full `LaneMask` for AIC+AIV0+AIV1, multiple owners.
- No prior claimed block, staged mask, launch, or completed drain state.
- Canonical ready state: `early_dispatch_state == DISPATCHED` and
  `early_sync_drain_state == READY`. Readiness through the direct zero-fanin or
  pre-completed-producer submit routes does not establish these markers; keep
  those routes on their original behavior in this first scope.
- PMU single-issue mode off. Use the same `is_pmu_enabled()` policy that supplies
  `pmu_active` in the dispatch loop, with its existing non-DFX fallback.
- `0 < idle_blocks < B` and `eligible_split_blocks >= B`, using full-mask tracker
  classification. Preserve zero-idle rejection and the existing all-idle path.
- `B > P`, where `P` is the sum of complete physical MIX group counts owned by
  all non-coordinator owners. This is also a task-lifetime condition below.

Do not change `allow_early_resolve` or producer wiring to admit this ready task.
The original HCA gather is unflagged and prevents unresolved early admission;
the consumer's own flag only controls propagation to its successors.

Proposed control flow, not an implemented diff:

```text
after original attempt-tagged ACK, coordinator:
    identify the existing early-gated mode
    count original idle capacity for a ready task
    if narrow ready admission holds and split capacity fits:
        CAS drain state READY -> READY | OWNER | ARMED
        on failure: retain original ready insufficient-capacity retry
    otherwise: retain original capacity decision and retry
    reset original seed/done fields; release-store stage_go = 1

peer:
    acquire stage_go, honoring completed/attempt-change exits

every owner, including coordinator:
    read OWNER only after the stage_go handoff
    call original stage_sync_start_cores(..., gated = OWNER)
    publish original running contribution and done bit

coordinator:
    acquire all done bits; publish running seed when gated
    reopen original software gate
    retry original cohort rendezvous; finish original drain ownership
```

Today each peer reads `gated` before waiting for `stage_go`. Merely changing the
coordinator to set OWNER after ACK permits peers to prepare different payload
modes. Move the common ownership read after that release/acquire handoff.
The ownership CAS must accept exactly READY, not NONE or an old OWNER/COMPLETE.
Existing early ownership remains unchanged. No new shared layout or Core encoding
is proposed: the current `SyncStartDrainState` stays 64 bytes.

`finish_early_sync_drain` sets COMPLETE; it does not clear OWNER. The proposal
retains that terminal state through this task generation. A stale entry cannot
re-adopt it through the exact-READY CAS. Normal next-submit payload initialization
is still responsible for resetting the state.

## Resource, ordering and lifetime proof obligations

All owners stop ordinary tracker mutation between ACK and stage. Core FIN can
arrive during this interval, but the CPU tracker does not change itself. Thus
the capacity decision and the existing unique range claims remain within the
same frozen ownership interval. Each owner prepares only its own slots.

| Event | Retained requirement and code consequence |
| ----- | ----------------------------------------- |
| Before successful admission | No new seq, slot, payload or range mutation. An insufficient attempt advances the original attempt and returns to completion polling. A failed ownership CAS does not call cancel/requeue. |
| Gated stage | `force_gate` produces the original nonzero source pointer. Split placement uses a running slot for an idle used lane and a legal pending slot for a busy used lane. Existing pending occupancy and per-owner range bounds apply. |
| Payload publication | Retain the prepare `wmb`, per-Core register writes, doorbell-table writes and staged-mask atomics. Every owner, including claim-zero owners, contributes done. |
| Seed and reopen | Owner work precedes its done release; coordinator acquires all done bits, publishes the running seed, then releases reopen. No completion poll can promote a new pending subtask before that seed is visible. |
| Old running FIN | Existing `pending_gated` also recognizes DISPATCHED sync-start tasks. Case 3.3 completes the old running task and promotes the gated pending slot without waiting for its ACK. Retain one running-count increment per promotion and the one-winner launch latch. |
| New cohort launch | Seed is read before the stable mask. Ring only when every staged used lane is in a running slot and the already-ready task is DISPATCHED. Propagate its dispatch fanin only after launch is complete. |
| Coordinator post-reopen tail | Preserve a physical completion obligation owned by the coordinator until its own completion loop resumes; OWNER is not a reference-count pin. |
| Fatal / teardown | Preserve all completed/attempt exits and original EXIT/join/quarantine boundaries. Published payloads are not rolled back. Clearing CPU fields alone must not certify Core quiescence or permit buffer/arena reuse. |

The coordinator condition has a concrete bound. Each peer can claim at most its
physical full-MIX group count in this one split-placement pass. With an initially
unclaimed `B`-block cohort and `B > P`, peers cannot claim every block, so the
coordinator must stage at least `B - P > 0` blocks. Its Core FINs are retired by
its own completion loop, which cannot resume before its current drain tail ends.
Other owners therefore cannot complete and consume this task during that tail.
For the historical 24/8+8+8 layout, the bound is eight coordinator blocks.

Do not call `cancel_early_sync_drain` to undo the new ready adoption: its READY
branch pushes the task again, whereas the active drain already owns this task.
Select adoption only after capacity fits, with no capacity-retry branch after
the successful ownership CAS. Fatal exits retain the original teardown contract.

Before implementation acceptance, map every return after adoption to the actual
normal/fatal cleanup caller and test task reuse after supported termination.
Missing Core exit confirmation remains a reuse failure, not a recoverable
permission to clear the gate.

## Work moved and connected progress

For the 24 full-MIX/72-subtask example, compare against the reviewed main path,
not against an uncommitted combination of M4 and the proposal:

| Processing | Frequency and change |
| ---------- | -------------------- |
| Admission | Ready-state/shape/PMU/width checks once per eligible attempt at the coordinator; an additional split-capacity pass when original idle capacity is insufficient. Retry count must be measured. |
| Ownership | One successful ready-owner CAS per adopted cohort; one common post-go ownership read per owner. No per-block adoption CAS. |
| Arguments | AICPU skips 72 copies of 17 argument elements; the existing gated Core path performs those 1,224 element writes. Work is moved and may overlap; it is not erased. |
| Gated metadata | 72 doorbell-table entries and local mask accumulation, then one seq_cst OR per nonempty mask word per owner. Original payload/register publication remains. |
| Launch | Up to 72 pending-to-running count increments, determined by actual placement; one successful launch latch and 72 high-half doorbells for the full cohort. These are added costs relative to immediate ready launch. |
| Retained preparation | Function lookup, seq/buffer selection, per-Core contexts, tracker/slot updates, MIX running counts and the existing global running contribution remain. M4's two guards are evidence only, not included code. |

Early occupancy can delay independent RAW or ROPE work needed by the same merge.
For that join, `merge_start >= max(CMP_end, RAW_end, ROPE_end)` plus the actual
completion/publication interval. Prior SS selection experiments already changed
the last necessary predecessor from CMP to RAW; their post-CMP gap cannot be
charged to a slow ready queue. Keep the original task dependencies and require
that prestaging does not create a resource cycle with still-required work.

The existing [ready-path arg-offload investigation](../investigations/2026-07-aicore-fills-all-args-ready-path.md)
measured a setup increase from 349ns to 1,356ns per subtask on a different
`paged_attention_unroll` workload. That is a concrete counterexample to treating
the moved copies as a free saving. An idle Core can fill while its cohort waits;
a busy Core cannot fill its pending task's arguments until the previous task
returns. In particular, the last such Core may add fill latency after the last
required old FIN. Establish the target's actual fill/gate interval instead of
subtracting CPU preparation time directly from cohort start.

Retain the generation-tagged ACK tree and fixed coordinator from the
[drain retry ABA investigation](../investigations/2026-07-sync-start-drain-retry-aba.md).
This proposal changes admission and the post-go mode read, not attempt identity
or coordinator election. Old-attempt exits remain part of the correctness gate.

Fallback paths are all-idle ready, zero-idle ready, partial MIX, predicate, PMU,
direct-submit readiness without canonical markers, insufficient split capacity,
and a cohort that peers could stage without the coordinator. Each retains its
existing source path; no claim that an unchanged path has zero code-layout cost.

## Review and validation gates

This Draft proposes a bounded A2/A3 mechanism. It does not implement the changes,
close #2727, restart stopped candidates, or predict a speedup.

1. Audit the actual minimal diff: exact-READY ownership, the shared post-go load,
   original early behavior, PMU and all fallback exits, coordinator capacity bound,
   duplicate entries, post-clear lifetime, fatal and reuse. Reuse the corresponding
   constraints from [#2682](https://github.com/hw-native-sys/simpler/issues/2682).
2. Validate the real function chain: one ready adoption, peer observes the selected
   mode, zero-claim owner, insufficient retry, CAS loss, mixed running/pending lanes,
   each promotion once, last promotion launch, all-idle launch, stale COMPLETE,
   supported fatal exits and next-task reuse. Include original early and PMU negatives.
3. Inspect actual HCC for extra flag loads, root scans, atomics, loops, spills,
   barriers and unchanged boundary layouts before new timing. Source counts are
   not instruction counts, retries or microseconds.
4. Bind a low-write record to one TaskId/run/attempt: first request, capacity
   failures, chosen mode/go, owner preparation, publish, seed/reopen, last promotion,
   cohort doorbell and Core start. Keep CPU/Core domains separate. Check both launch
   savings and added gated wait/doorbell cost, including all high and reverse calls.
5. Only after correctness and a real overlap window are established, measure
   matched L0 target groups of 100 calls per arm, repeated three times with order
   balanced. Keep numerical checks outside timing, preserve all highs, and compare
   RAW/CMP/ROPE merge plus whole-entry Device. Broaden regression in proportion to
   the adopted path; unchanged cases need not gain.

Source maps, state checks and documentation validation are the current review
evidence. There is no new candidate binary, simulation result, NPU run or measured
benefit for the proposed prestaging path.
