# A5 HBG terminal claim-count memoization

## Ticket-owned ordinary execution records, 2026-11-16

Retain the ordinary lifecycle candidate on `3c9127783` under the user's
atomic-reduction rule. A successful ordinary block FAA fixes its sole owner;
the execution record is then an owner-written diagnostic, not a second
cross-core ownership gate. A ticket-owned flag distinguishes that contract.
The owner validates its local lifecycle, writes the diagnostic, and publishes
the isolated line with DCCI and DSB. Cooperative reservation, arming, acquisition
and retirement retain atomic transitions. No record mixes these two access
contracts, no `st_dev` is added, and all shared task completion/async operations
remain atomic. AICPU error snapshots retain cache invalidation; Host collection
and the canonical C-zero retirement inference keep their existing formats.

The focused ordinary test removes the initial execution-record CAS and checks
successful return, cancellation, stale-handle rejection and non-completion of
the remaining block. Production ordinary partial returns also replace their
record Exch with diagnostic publication. These reductions add ordinary stores,
DCCI and DSB; they are not free and do not imply fewer total memory operations.
All seven focused CPU suites, the 84.07-second A5 build and sixteen device IR
scans pass.

Twelve paired AICore L0 mean durations sum to **20.4327887 -> 20.4940688 ms
(+0.299911%)**. This is a measured regression, retained because it is below one
percent with a proven ownership-atomic reduction. Latest C1A reindex increases
1.807%, and latest layers 19-21 increases 1.128%; the acceptance rule applies to
the twelve-case sum, not every individual row. No stable latency improvement or
net L3 primitive-count change is claimed.

Eleven replay cases pass numerical validation; native MoE is timing-only.
Each arm uses one warmup and ten samples on automatically allocated matched
devices. Every two-device round rendezvouses on Device before timing and takes
the slower rank; all eight arm audits pass. No retry or sample removal was used.
Evidence is under `artifacts/c2a-handoff-20261115/ticket-owned-execution/`, its
`-l0/` and `-comm-l0/` siblings, and
`artifacts/atomic-reduction-20261115/ticket-owned-execution12.json`.

## Bounded fill completion observations, 2026-11-16

Reject the worker-private four-entry negative completion cache on `58bf35339`.
It lived only inside one bounded fill, shared pending observations across token
and issue probes, reset after cooperative progress, and let canonical DONE
override a pending entry. A focused real-fill check reduced reads of a shared
predecessor from four to one and observed completion in the next fill. The
ownership, ordered and GM MIX CPU suites passed; the 88.01-second A5 build and
all sixteen device IR scans passed.

Paired twelve-case AICore L0 mean sums increased from **20.5193698 to
20.7866075 ms (+1.302368%)**, exceeding the user's one-percent retention limit.
Eight non-communication cases increased 0.243482%; four communication cases
increased 1.780039%. DSL layers 19-21 changed 8.8039083 -> 9.0070668 ms.
This result does not distinguish local cache lookup cost from the effect of
deferring a fresh observation; no causal mechanism is claimed.

All eleven replay cases passed numerical checks; native MoE was timing-only.
Both arms used the same automatically allocated devices, one warmup and ten
samples. Every dual-device sample retained the Device rendezvous and slower-rank
aggregation; all eight arm audits passed. No retry or sample removal was used.
The source/runtime snapshot and rejected patch are under
`artifacts/c2a-handoff-20261115/fill-completion-reuse/`; the complete matrix is
`artifacts/atomic-reduction-20261115/fill-completion-reuse12.json`.
Production code and the candidate-specific test were restored. Binding the
static wait address while retaining every fresh observation is a separate trial.

## Decision

Reject the terminal claim-count cache. It reduced repeated cursor observations
in SWA L3 but increased the sum of sixteen L0 means by 3.17%.

The retained baseline for that trial re-read the admission claim cursor.
A local reservation does not substitute for global claim exhaustion or its seal.

## Candidate and evidence

The candidate cached only an atomic observation of `next_block == block_num`,
optionally accompanied by the valid `claim_seal`. Adoption and obligation
retirement cleared the private proof; errors, dynamic data dependencies and
sync-start entry checks were unchanged. The private structure did not grow.

This followed SWA traces showing repeated claim-count checks during the same
obligation, and `tests/atomic_probe/ATOMIC_USAGE_GUIDE.md` section 4.6 in the
`simpler-fdwic-swimlane-deps` checkout: same-address atomic reads are contended
operations, not free scalar loads. That evidence motivates fewer reads but
does not predict end-to-end speedup.

The baseline is local `0b5ee78e5` over upstream `7688a860d`, plus explicit
resident-entry atomic-store-mode initialization. The candidate adds only the
claim proof. Both arms use frozen source and binaries; kernels and inputs match.
A5 onboard, CANN 9.3.0; one warmup plus ten L0 measurements per arm. Device time
is `chip.run.runner_run.device_wall`; two-rank samples use the slower rank per
round. Each pair uses the same automatically allocated devices. All numerical
checks passed; no timed samples were removed. Communication cases are included.

| Case | Baseline ms | Candidate ms | Change | Devices |
| ---- | ----------: | -----------: | -----: | ------- |
| latest__c1a_reindex | 0.765604 | 0.757027 | -1.12% | 7 |
| latest__c1a_reuse | 0.674075 | 0.645797 | -4.20% | 0 |
| latest__c2a_full | 0.680352 | 0.691522 | +1.64% | 7 |
| latest__engram | 1.167640 | 1.175749 | +0.69% | 0 |
| latest__layers_19_21 | 4.303244 | 5.030891 | +16.91% | 0, 7 |
| latest__moe | 1.984563 | 1.492141 | -24.81% | 0, 7 |
| latest__swa | 0.512892 | 0.515337 | +0.48% | 3 |
| pypto_lib__c2a_full | 1.569189 | 1.549021 | -1.29% | 5 |
| pypto_lib__layers_19_21 | 15.489094 | 15.050557 | -2.83% | 1, 2 |
| pypto_lib__moe | 3.625605 | 3.661719 | +1.00% | 1, 2 |
| pypto_lib_dsl__c2a_full | 1.857136 | 1.964603 | +5.79% | 3 |
| pypto_lib_dsl__layers_19_21 | 10.310880 | 10.500324 | +1.84% | 3, 4 |
| pypto_lib_dsl__moe | 1.761116 | 1.735021 | -1.48% | 3, 4 |
| pypto_lib_dsl_all_sync64__c2a_full | 2.012180 | 2.475511 | +23.03% | 5 |
| pypto_lib_dsl_all_sync64__layers_19_21 | 10.500586 | 12.013564 | +14.41% | 5, 6 |
| pypto_lib_dsl_all_sync64__moe | 2.064361 | 1.897058 | -8.10% | 5, 6 |
| **Sum** | **59.278517** | **61.155843** | **+3.17%** | — |

SWA L3 recorded 4,524 → 4,344 atomic fetch-add calls and 344.76 → 180.84 µs
of long cursor-check spans. These are totals across cores, not wall-clock
savings. SWA L0 was 0.512892 → 0.515337 ms.

The full-sync C2A L0 regression was not a single outlier: the ten-sample mean
rose 23.03%. A diagnostic L3 pair on device 5 did not reproduce that difference
(2.936830 → 2.911414 ms). Its longest dependency gaps include independent
kernels and cohort admission. Neither that trace nor source inspection proves
the cause of the L0 regression. Do not attribute it to atomic contention,
compiler layout or card differences without further evidence.

Both runtimes passed the actual device IR scan, 16/16 translation units with
zero findings. Admission, GM MIX and admission-progress CPU tests passed.
Local reproducibility artifacts live under
`artifacts/hbg-sixteen-optimization-20261115/`: `startup-store-mode/` and
`claim-proof/` contain frozen source, hashes and runtime binaries;
`claim-proof-eval/tasks.json`, `summary.json`, per-job results and captures
retain the full paired measurements; `claim-proof-rejected.patch` preserves
the rejected diff. These large artifacts are local, not shipped in Git.

## Ordinary monotone ticket trial

A separate trial adapted `cross_core_only_scheduler::TakeSharedExecTicket` to
ordinary tasks and non-sync MIX: discard an excess ordinal without decrementing
the shared cursor, and cache exhaustion after the last valid or first excess
ticket. The cache remained keyed to the current task. Sync-start valid counters,
claim seals, fixed ownership and data readiness were unchanged. The DFX audit
distinguished ordinary raw ticket bounds from exact cooperative valid counts.

The deterministic stale-open-snapshot test failed before and passed after this
change. Ownership, MIX, deferred progress and catalog checks passed; 689 Python
causality checks passed. The A5 build took 536.08 seconds, with 16/16 device
translation units scanned and zero IR findings. All sixteen paired hardware
cases passed numerical validation, using one warmup and ten L0 measurements.

Reject the trial: sixteen-case mean sums increased from 56.4976080 to
57.1059750 ms (+1.076801%). SWA improved 0.63% and initial C2A full improved
5.97%, but initial layers 19–21 regressed 3.51%, initial MoE regressed 3.72%,
and C1A reuse regressed 12.71%. Communication cases remain in the aggregate.
SWA L3 returned FAA calls fell from 4317 to 4132, while its profiled device
duration increased from 1.374579 to 1.387540 ms. Fewer atomics did not establish
an aggregate speedup. No cause for the per-case regressions is claimed.

Production code and the added tests were reverted. Local evidence is under
`artifacts/hbg-sixteen-optimization-20261115/`: `monotone-tail-ticket/`,
`tail-ticket-eval/summary.json`, `tail-ticket-cpp-tests.log`,
`tail-ticket-py-tests.log`, and `tail-ticket-rejected.patch`.

## Follow-up experiments

Each report retains its own decision, measurements and evidence locations.
The measurements are incremental comparisons and must not be added together.

- [Local excess-ticket suppression with rollback retained](2026-11-a5-hbg-task-state/excess-ticket-suppression.md)
- [Local closure from an owned sync_start block](2026-11-a5-hbg-task-state/owned-sync-start-closure.md)
- [Sync MIX closure through the existing SSBUF frontier](2026-11-a5-hbg-task-state/mix-ssbuf-frontier.md)
- [Prefer canonical DONE when observing closed supply](2026-11-a5-hbg-task-state/canonical-done-observation.md)
- [Confirm exhausted ordinary supply before reserving another ticket](2026-11-a5-hbg-task-state/ordinary-supply-exhaustion.md)
- [Publish ordinary supply after Claim, before payload receipt](2026-11-a5-hbg-task-state/supply-before-receipt.md)
- [Confirm async FINISH with the canonical CAS](2026-11-a5-hbg-task-state/async-finish-cas.md)
- [Reuse the completion writer's own DONE proof](2026-11-a5-hbg-task-state/completion-writer-proof.md)
- [Fold single-consumer ordinary control nodes](2026-11-a5-hbg-task-state/ordinary-control-folding.md)
