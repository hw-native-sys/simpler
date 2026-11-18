# CntNotify task flags and admission-tail migration

Date: 2026-11-16/17. Verdict: rejected under the current 0.5% aggregate regression limit.

## Candidate and result

The candidate combined SUPPLY_CLOSED, DONE and ADMITTED as independent bits in
one task-owned CntNotify register. BIT_WR publishes a mask without overwriting
another event; admission commit_tail uses a separate SET/read register. It keeps
all claim and commit-lock arbitration, publication barriers and GM resource fallback.
The baseline includes the retained GM MIX notification change (55e4f36f1).

Real A5 probes verified non-consuming reads, reset, 96 concurrent bit publishers
and repeated idempotent writes. All eleven numerical benchmark cases pass;
native MoE retains its established timing-only mode. Functional correctness did
not establish a performance benefit.

Twelve paired AICore L0 mean times sum to **20.3368670 -> 20.6939601 ms
(+1.7559%)**. Each case uses one warmup and ten samples, automatic allocation,
and identical binaries/inputs except the candidate runtime. Two-device cases
rendezvous on Device before timing and use the per-round rank maximum.

| Case | Baseline ms | Candidate ms | Change |
| ---- | ----------: | -----------: | -----: |
| latest__swa | 0.257829 | 0.261833 | +1.553% |
| pypto_lib__c2a_full | 1.076966 | 1.093912 | +1.573% |
| pypto_lib_dsl__c2a_full | 1.492568 | 1.495697 | +0.210% |
| pypto_lib_dsl_all_sync64__c2a_full | 1.557151 | 1.546068 | -0.712% |
| latest__c2a_full | 0.329348 | 0.336589 | +2.198% |
| latest__c1a_reindex | 0.333236 | 0.341492 | +2.478% |
| latest__c1a_reuse | 0.247118 | 0.251957 | +1.958% |
| latest__engram | 0.972654 | 0.980269 | +0.783% |
| pypto_lib_dsl__moe | 1.427828 | 1.443561 | +1.102% |
| pypto_lib_dsl__layers_19_21 | 8.835907 | 8.961705 | +1.424% |
| latest__layers_19_21 | 3.084536 | 3.220450 | +4.406% |
| latest__moe | 0.721726 | 0.760428 | +5.362% |

## Evidence and limits

A paired L3 capture of DSL C2A full, on the same automatically assigned device,
has complete Scalar records. Bypass loads grow 125784 -> 129906; CAS calls grow
636 -> 678; returned FAA calls grow 1766 -> 1830. Visible admission-refresh
scopes grow 1318 -> 1451, ordered-poll scopes 611 -> 733, and issue-fanin scopes
1557 -> 2400. These are actual observed trajectory counts, not claims that the
source introduced one new atomic per event. Cross-core inclusive times are not
additive wall-time savings, and L3 does not carry each primitive's target address.

Source inspection also identifies a new lookup before admission-tail queries:
load the graph task count, locate the immutable notification-table entry, then
read SRAM. The previous path directly read the fixed GM tail address. Caching
that immutable address is a possible follow-up, but was not measured here.
The experiment does not establish that SRAM itself is slower than GM, nor that
any one lookup caused the entire regression. More frequent scheduling visits and
additional address lookup must be examined together before revisiting this path.

## Reproduction record

- Frozen source, runtime, IR and hashes: `artifacts/c2a-handoff-20261115/cnt-bit-flags/`.
- Twelve-case samples: `artifacts/atomic-reduction-20261115/cnt-bit-flags12.json`.
- Rejected patch and verdict: `artifacts/cnt-next-20261116/flags-rejected.patch`
  and `flags-verdict.json`.
- Paired L3 raw: `artifacts/cnt-next-20261116/trace-{cnt-mix,cnt-bit-flags}/`.
- Counts: `artifacts/cnt-next-20261116/flags-l3-comparison.json`.
- The requested latest PyPTO-lib SWA base/MIX/MIX+flags L3 triplet is separately
  retained under `artifacts/cnt-next-20261116/swa-l3-three/`.

The separate GM MIX change is retained at +0.224% by explicit user instruction
and the subsequently updated 0.5% threshold. The rejected task-flag migration is
not present in the working production implementation.
