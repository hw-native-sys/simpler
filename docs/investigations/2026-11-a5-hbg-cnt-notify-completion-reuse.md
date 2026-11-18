# CntNotify completion bitmaps with within-run DAG bank reuse

Date: 2026-11-17. Verdict: rejected under the current 0.5% aggregate regression limit.

## Candidate and result

The candidate replaced some per-return GM completion accounting with SRAM bitmaps,
then reused physical bitmap banks within one run. The retained baseline includes
DONE notifications (`0a4ce448c`), remote GM MIX handshakes (`55e4f36f1`) and the
Host mapping pool (`92c2befbd`). The separately rejected task-flag/admission-tail
candidate is absent from this baseline.

Twelve paired AICore L0 mean times sum to **20.2993002 -> 21.5089332 ms
(+5.95899%)**. All twelve cases regress; the completion bitmap and within-run
bank candidate is not retained. MIX handshakes and the Host mapping pool remain.
The independent SDK allocation fallback correction remains at `3840763e8`.

## Initial allocation failure

The first bitmap candidate allocated independent words for every eligible task.
Both ranks of both layers cases failed in their first prepare (`generation=1`,
`run_epoch=1`) before timed execution. Their SDK plogs report:

- Driver: `type=CntNotify`, `alloc_num=2048`, `allocatable_num=0`, `ret=-28`.
- Runtime: inner code `0x7020023`, `EE1023`, too many CntNotify objects.
- ACL: **207005**, whereas the original fallback handled only 207009.

The C API converted the thrown allocation error to a generic preparation failure.
The 2,048 count is the allocation state observed in these runs, not a universal
hardware quota claimed for every context. Failure on the first preparation does
not demonstrate a multi-run pool leak. Both 207005 and 207009 are now treated as
capacity failure only at `aclrtCntNotifyCreate`; get-ID and mapping errors remain
errors, and incomplete banks cannot grant partial bitmap completion.

These first preparations are the benchmark's **reference full-prefix graph**,
not the selected layers' timed subgraph. `production/runner.cpp` uses phase 0 for
`reference`, phase 1 for `prefixN`, and phase 2 for `runN`. Their resource counts
must remain separate.

## Within-run reuse mechanism and resource counts

Host planning propagates ancestors through the actual DAG, including non-kernel
control nodes. Host-completed/folded nodes follow their existing completion
semantics, so skipped incoming edges cannot create false ancestry. In topological
order, best-fit chooses a bank only if its previous user is a strict ancestor of
the current task. Capacity is the maximum word demand along that chain. Independent
tasks do not share a bank merely because catalog order puts one first; this adds
no execution dependencies.

Each bank owns bitmap words and an owner notification. Device handoff uses a
run-scoped lease sequence, resets the words, and publishes the next active owner.
A task retains a separate snapshot, with publication state isolated from payload
cache lines. Logical completion, including deferred conditions, remains required;
issuance completion alone cannot permit reuse. Physical resources are allocated
in contiguous owner-plus-words intervals. Every resource in a bank must map before
any task on that chain uses it; otherwise the complete chain falls back to GM.
Cross-run Host mapping pooling and within-run bank sharing are distinct lifetimes.

The following are final hardware-run Host records. All listed banks are fully
mapped. These counts exclude the existing DONE and MIX notifications.

| Graph / benchmark phase | Tasks | Original bitmap words | Banks | Reused bitmap words | Owner words | Total bank resources |
| ----------------------- | ----: | --------------------: | ----: | ------------------: | ----------: | -------------------: |
| latest SWA / timed run | 28 | 41 | 4 | 9 | 4 | 13 |
| DSL C2A full / timed run | 42 | 49 | 7 | 17 | 7 | 24 |
| DSL layers / reference | 1521 | 2167 | 19 | 105 | 19 | 124 |
| DSL layers / prefix | 1315 | 1881 | 19 | 105 | 19 | 124 |
| DSL layers / timed run | 206 | 286 | 7 | 32 | 7 | 39 |
| latest layers / reference | 1429 | 2381 | 40 | 146 | 40 | 186 |
| latest layers / prefix | 1235 | 2068 | 36 | 139 | 36 | 175 |
| latest layers / timed run | 194 | 313 | 10 | 26 | 10 | 36 |

Reuse resolves the observed preparation-capacity issue. It does not establish a
scheduling performance benefit.

## Paired performance comparison

Each case uses one warmup and ten timed samples with profiling disabled. The
metric is `chip.run.runner_run.aicore_wall`: last actual kernel end minus first
actual kernel begin on the same device. Inputs and kernels are unchanged. Devices
are automatically assigned; each revision pair shares its allocation. Two-device
cases rendezvous on Device before timing and use the per-round rank maximum.

| Case | Baseline ms | Candidate ms | Change | Allocated devices |
| ---- | ----------: | -----------: | -----: | ----------------- |
| latest__swa | 0.255943 | 0.294107 | +14.911% | 2 |
| pypto_lib__c2a_full | 1.077375 | 1.156630 | +7.356% | 0 |
| pypto_lib_dsl__c2a_full | 1.505912 | 1.554622 | +3.235% | 1 |
| pypto_lib_dsl_all_sync64__c2a_full | 1.562294 | 1.605198 | +2.746% | 3 |
| latest__c2a_full | 0.329109 | 0.368249 | +11.893% | 2 |
| latest__c1a_reindex | 0.332605 | 0.381064 | +14.570% | 3 |
| latest__c1a_reuse | 0.247114 | 0.279432 | +13.078% | 2 |
| latest__engram | 0.966384 | 1.066130 | +10.322% | 4 |
| pypto_lib_dsl__moe | 1.423980 | 1.479962 | +3.931% | 4, 5 |
| pypto_lib_dsl__layers_19_21 | 8.787330 | 9.112584 | +3.701% | 6, 7 |
| latest__layers_19_21 | 3.088374 | 3.424397 | +10.880% | 0, 1 |
| latest__moe | 0.722882 | 0.786558 | +8.809% | 3, 5 |
| **Twelve-case sum** | **20.2993002** | **21.5089332** | **+5.959%** | - |

All eleven numerically validated cases pass. Native MoE remains in its established
**timing-only** mode (`numerical_validation:false`); execution success is not a
numerical-validation result. Functional success does not override the measured
aggregate regression.

## L3 evidence and limits

The candidate DSL C2A full L3 converts successfully, with complete Scalar records
for all **96 workers**. It records bitmap handoff, snapshot and accounting scopes.
The raw capture lacks the native protocol, bitmap snapshot and condition evidence
needed for a full hardware bitmap protocol check. No protocol-checker pass is
claimed from this capture.

Bank-owner observation/reset, generation handoff, bitmap scanning, snapshots and
final accounting are additional execution-path work to investigate. Their exact
contributions to the 5.959% L0 regression were not isolated. L3 durations contain
nested and overlapping cross-core work; summing them is not serialized delay or
wall-time savings. Thresholded scope counts are not total function invocation
counts. Profiling also changes the schedule, and the captures do not identify
every primitive target address. These results do not prove that SRAM itself is
slower than GM.

Reconsider only after a revised handoff/accounting protocol shows less work on the
execution path, preserves deferred completion and generation correctness, and
passes another full paired twelve-case L0 comparison. Lower resource consumption
or fewer instances of one atomic instruction alone is insufficient.

## Reproduction record

- Final samples: `artifacts/atomic-reduction-20261115/cnt-bitmap-reuse12.json`.
- Paired outputs and Host resource-count records:
  `artifacts/c2a-handoff-20261115/cnt-bitmap-reuse-l0/` and
  `cnt-bitmap-reuse-comm-l0/` under the same parent.
- Frozen source/runtime: `artifacts/c2a-handoff-20261115/cnt-bitmap-reuse/`;
  baseline runtime: `cnt-resource-pool/` under the same parent.
- Rejected patch and hashes:
  `artifacts/cnt-next-20261116/bitmap-reuse-rejected-full.patch` and
  `bitmap-reuse-rejected-manifest.json`.
- Initial full-graph candidate: `bitmap-full-graph-before-reuse.patch` in the same
  artifact directory.
- All four initial SDK failures, with exact plog paths and line numbers:
  `artifacts/cnt-next-20261116/completion-bitmap-capacity-triage.json`.
- L3 analysis: `artifacts/cnt-next-20261116/reuse-l3-analysis.md`;
  raw: `trace-bitmap-reuse/case/rank0/run1/chip_swimlane_records.json`;
  converted: `bitmap_reuse__c2a_dsl__L3.json`; counts: `reuse-l3-counts.json`, all
  under the same artifact parent.
- Related rejected experiment: [task flags and admission tail](2026-11-a5-hbg-cnt-notify-task-flags.md).

Local artifacts retain measurement provenance and are not shipped as repository
source. The rejected completion experiment must not be described as current
production behavior.
