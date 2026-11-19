# A5 SPMD: one fetch-add for an exact full-residency sync cohort

A `sync_start` cohort handed out its block indices through a bounded
compare-exchange loop on the cohort's `next_block` cursor: one plain read of the
cursor, then a compare-exchange, retried up to eight times before the caller had
to come back later. That cursor is the cohort's launch path, so every core that
joins the wave passes through it and each retry contests the line the other
joiners are reading.

When the cohort is an exact full residency — as many logical blocks as eligible
claim positions — each position can claim exactly once, so the retry loop has
nothing to resolve. One fetch-add on `next_block` then returns the block index
and publishes the claim in the same operation.

## Conclusion

The invariant below holds, so the change is confined to one branch of
`scheduler_claim_ordered_cursor`. The block index stays the fetch-add return
value, `claim_tickets` stays zero on this branch, the actual last valid claim
publishes `claim_seal`, and `next_block` still counts only real claims.

Everything else is untouched: the ordinary ticket path, partial-sync CAS, the
`sync_claimed` gate, admission and MIX-caller semantics, error sites, and every
existing test expectation.

## Why the shape is exact

`scheduler_block_claim_positions(mask, clusters)` is the same expression the
admission record uses to size a sync cohort:

| active shape | eligible claim positions | where |
| ------------ | ------------------------ | ----- |
| AIC only | one per cluster | `scheduler_types.h` positions = clusters; admission members = AIC lanes |
| AIV only | two per cluster, either V lane | positions = 2 × clusters; admission members = all AIV lanes |
| any Mix mask (AIC+AIV, both AIV, all) | one coordinator per cluster | positions = clusters; members include all lanes, but only `is_mix_coordinator()` (`self_lane == 0`, the AIC) calls `scheduler_gm_mix_claim` |

So `requires_sync_start && current_block_num == positions` means every position
has exactly one block and every block has exactly one position. The invariants
are already enforced by existing code:

- **One claim per local:** `sync_claimed` is checked before allocation and set by
  the winning claim; a repeat returns `RETRY`. In the new branch it is set
  immediately after the successful fetch-add, before protocol recording.
- **No extra caller:** admission membership is written from the frozen topology;
  a non-member cannot adopt the record. Mix non-coordinators fail
  `scheduler_gm_mix_claim` before the cursor.
- **N > 1 and topology:** sync with `logical_block_num == 1` is rejected at
  record validation, so an admissible full-sync cohort has at least two
  positions; admission already rejects `block_num > positions`.
- **Malformed/overshoot:** the fetch-add can return `block >= count` only for an
  invalid extra claimant. The branch reports `BLOCK_CURSOR_INVALID` and returns
  `ERROR` instead of handing out an out-of-range index or silently inventing an
  exhaustion answer; the cursor was already advanced by that invalid claim, so a
  fatal report is the truthful outcome. Valid positions cannot reach it.
- **Certificate:** only `block + 1 == count` publishes the seal; the confirmed
  cursor/seal therefore describe the actual claims, not a planned rank.

## File changes

```text
 src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_admission.h | +34
 tests/ut/cpp/a5/runtime/host_build_graph/test_hbg_admission.cpp        | +235
 tests/ut/cpp/a5/runtime/host_build_graph/test_hbg_ordered_execution.cpp | rename
 docs/investigations/a5-spmd-full-sync-claim-results.md                 | new (this file)
```

The new branch sits before the bounded compare-exchange retry loop and recomputes
`positions` itself, so the ordinary ticket path and the retry loop are untouched.
`HbgOrderedTicketCursor.SyncStartTasksKeepTheBoundedCasPath` is renamed: it
already built an exact full-residency cohort, so it exercises the fetch-add
branch now and must not claim otherwise.

## Tests added

`HbgFullSyncClaim` in `test_hbg_admission.cpp`, using the existing
topology-shaped admission fixture (4 clusters, 12 workers):

| case | asserts |
| ---- | ------- |
| `EveryEligiblePositionClaimsOneUniqueBlockAndTheActualLastClaimSeals` (6 masks: AIC, AIV, AIC+AIV0/1, both AIV, all) | claimant count equals block count; unique dynamic indices; seal only at the final actual claim; `claim_tickets == 0`; confirmed cursor sealed at count; repeated calls from a claimant return `RETRY`; no completion |
| `AnExtraReservationPastTheSealedCohortFailsInsteadOfReturningAClaim` | an injected extra claimant past a sealed full cohort gets `ERROR`, no block index, error published |
| `PartialSyncCohortKeepsTheBoundedRetryAllocator` | 2 blocks over 4 AIC positions stays on CAS; remaining positions get `EXHAUSTED`, no error |
| `OrdinaryFullWidthCohortKeepsTheTicketAllocator` | non-sync full-width cohort still claims uniquely, seals, and exhausts cleanly |
| `IndependentOrdinaryTaskStillClaimsWhileTheSyncCohortSealIsComplete` | a nonmember passes the sealed sync record and claims an independent vector task; the cohort certificate is unchanged |

Only the second case asserts the new allocator mechanism; the other four are
guards that must pass on both headers.

## Verification

CPU only: `test_a5_hbg_admission` passes 312/312 with the change and 311/312 with
the branch removed, the single failure being the new mechanism case. That case
therefore pins the allocator rather than the surrounding behaviour. `clang-format`
was applied to both changed source files.

On device the A/B runs two AICore objects built from the same tree — one with this
branch, one with only the branch removed — against an identical host runtime,
AICPU kernel and dispatcher, so the samples isolate the allocator. Thirteen
`(source, case)` rows that contain a full-sync cohort each interleave two arms per
object (`b1 c1 b2 c2`), one fresh process per arm and one level-3 capture per arm;
all 104 rank replays match their reference. Median interval between the cohort's
32 coordinator claims:

| source | case | cohorts | main µs/claim | candidate µs/claim | change |
| ------ | ---- | ------: | ------------: | -----------------: | -----: |
| PyPTO-lib | `c1a_full` | 1 | 2.465 | 0.103 | −95.8 % |
| PyPTO-lib | `c1a_reindex` | 1 | 1.731 | 0.086 | −95.0 % |
| PyPTO-lib | `engram` | 5 | 1.875 | 0.095 | −94.9 % |
| PyPTO-lib | `c2a_full` | 1 | 1.888 | 0.114 | −93.9 % |
| PyPTO-lib | `layers_19_21` | 1 | 1.651 | 0.107 | −93.5 % |
| DSL | `engram`, `moe`, `layers_19_21`, `c1a_*`, `c2a_*` (8 rows) | 45 | 1.09–2.19 | 0.11–0.29 | −77.4…−93.6 % |

Geomean over all thirteen rows is −91.1 %. Whole-cohort dispatch ramp falls
76.4 % (geomean), the PyPTO-lib rows 86–91 %. On the same arms, measured
`device_wall` improves by 56 µs (`c2a_full`), 69 µs (`c1a_full`), 88 µs
(`c1a_reindex`) and 181 µs (`engram`); the DSL rows carry one cohort each off the
critical path and stay inside the vehicle's noise.

## Limits

- These are level-3 captures on one device and two ranks: they show a shorter
  claim and a shorter ramp under per-phase scheduler records. The regular
  configuration was not measured, and no gain is claimed for it.
- Only the exact-residency shape changes. A cohort that is not on the critical
  path, or that the dispatcher resolves before its claim is contended, has
  nothing to gain.
- The exact-position argument is source-level. The extra-claimant case asserts
  fail-safe behaviour for a state production cannot reach; it is not a workload
  observation.
- Resource and dependency exclusivity, 96-lane mapping and COND quiescence are
  untouched here and remain unestablished by this change.
