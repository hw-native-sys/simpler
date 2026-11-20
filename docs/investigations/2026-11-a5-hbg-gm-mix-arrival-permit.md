# A5 HBG GM MIX arrival permission and ACK reuse

## Current MIX entry and retirement

AIV0 takes its assigned cluster block for a fixed-owner wave or claims a
dynamic ticket, then publishes the block identity and active lane mask in a
cluster SSBUF bank. Each active member accepts the same odd generation,
observes the task's fan-in with its own prefix cursor, satisfies its local
ordering obligations and publishes its own ARRIVED. A descriptor or an
arrival on one lane does not certify another lane's fan-in. The additional
fan-in observations can increase GM reads relative to the coordinator trial.

When any active lane has a nonzero Host-planned order floor from an earlier
pure sync-start cohort, the members exchange ORDER_READY before publishing
ARRIVED. In that case, or when the run has an async layout, each member must
observe the complete active ARRIVED set for its generation before entry.
Otherwise a member may publish its payload after its own ARRIVED. Its peer
may still be executing unrelated ordinary work. A kernel that waits for that
peer on an in-kernel event occupies its core until the peer reaches the event.
The order floor constrains earlier conflicting duties; it neither bounds this
delay nor guarantees simultaneous entry.

Every successful active lane contributes its own physical P on the CntNotify
path. The accountant (AIV0 when active, otherwise AIC) waits for matching FIN
from every active lane before retiring the bank. On the GM path it also
contributes the active lane count at that point. It release-publishes
`accounted_generation` to AIV0. After AIV0 sees its current odd generation
acknowledged, `scheduler_cluster_store_release` already issues a DSB before
storing the next even `armed_generation`. A separate DSB immediately before
that helper would repeat the same ordering. The DSB **after** the even store
remains: an observer can still be sampling the old odd generation, so the
invalidation must precede overwrite of the descriptor. The new payload is
published by a subsequent release store of its new odd generation.

## Current scheduler compared with upstream main

The following measurements compare the complete scheduler change against upstream
`9cf7491e5` on A5 with CANN 9.3.0. Each case reused its compiled model and
inputs across two frozen runtime builds. Both builds ran on the same allocated
device or device pair, with one warmup followed by ten L0 samples per build.
The run order alternated across cases. For two-rank cases, each device completed
a peer rendezvous before timing. Each sample uses the slower rank's AICore
wall duration.

| Case | Base (ms) | PR (ms) | Change | Devices |
| ---- | --------: | ------: | -----: | ------- |
| `latest__swa` | 0.203955 | 0.185787 | -8.91% | 1 |
| `pypto_lib__c2a_full` | 0.975231 | 0.973886 | -0.14% | 2 |
| `pypto_lib_dsl__c2a_full` | 1.419222 | 1.402445 | -1.18% | 7 |
| `pypto_lib_dsl_all_sync64__c2a_full` | 1.412688 | 1.376215 | -2.58% | 0 |
| `latest__c2a_full` | 0.270215 | 0.241756 | -10.53% | 1 |
| `latest__c1a_reindex` | 0.295063 | 0.269624 | -8.62% | 2 |
| `latest__c1a_reuse` | 0.202048 | 0.189660 | -6.13% | 3 |
| `latest__engram` | 0.106377 | 0.101294 | -4.78% | 0 |
| `pypto_lib_dsl__moe` | 1.260845 | 1.239232 | -1.71% | 2, 7 |
| `pypto_lib_dsl__layers_19_21` | 8.100796 | 7.954047 | -1.81% | 0, 3 |
| `latest__layers_19_21` | 2.202250 | 2.153074 | -2.23% | 2, 7 |
| `latest__moe` | 0.495745 | 0.497899 | +0.43% | 0, 3 |
| **Sum of case means** | **16.944434** | **16.584920** | **-2.12%** | |

Eleven replay cases matched their fused-prefix reference on both builds.
Native `latest__moe` checked routing counts and supplied timing only.
The sum weights each case mean once. These timings cover the combined changes
to MIX admission, fan-in, ownership and completion. Isolating the impact of
individual mechanisms or counting GM bus transactions requires separate
measurements.

## Historical coordinator-only trial

The first trial was removed after its twelve-case AICore L0 sum regressed
0.0731%. A later adaptation restored coordinator confirmation under a criterion
that allowed measured reductions in same-address atomics. The two comparisons
below describe those earlier implementations.

The candidate used the coordinator to confirm the arrival set and publish a
permission sequence through an additional 8-byte SSBUF field. Peers consumed
that permission. Per-worker arrival/release masks retained positive ACK
observations within a group. Self-publication supplied the local ACK proof;
failed release remained distinct from success and masks reset on group changes.

The baseline is local `241af7b91` (private issuance cursors), after `d8a71ec3a` (MIX completion aggregation). Both arms use frozen A5 runtime binaries, unchanged business kernels and inputs, one warmup and ten measurements per case. Devices are allocated automatically and paired arms use the same devices. Every dual-device launch rendezvous precedes the measured run; each sample uses the slower rank's AICore-local wall duration. Eleven cases validate replay outputs; latest native MoE remains performance-only.

Five focused C++ suites pass, including all 652 MIX checks after updating the explicit coordinator-to-peer ordering. The actual device IR scan covers 16/16 translation units with zero findings. C2A full L4 export and protocol conversion also pass.

## Historical paired AICore L0 results

| Case | Baseline ms | Candidate ms | Change | Devices |
| ---- | ----------: | -----------: | -----: | ------- |
| latest__swa | 0.3103409 | 0.3230783 | +4.104% | 1 |
| pypto_lib__c2a_full | 1.1803912 | 1.1847535 | +0.370% | 0 |
| pypto_lib_dsl__c2a_full | 1.5767262 | 1.5509702 | -1.634% | 3 |
| pypto_lib_dsl_all_sync64__c2a_full | 1.6114378 | 1.5851287 | -1.633% | 2 |
| latest__c2a_full | 0.3900541 | 0.3956579 | +1.437% | 1 |
| latest__c1a_reindex | 0.3774132 | 0.3801868 | +0.735% | 3 |
| latest__c1a_reuse | 0.2932383 | 0.3016940 | +2.884% | 0 |
| latest__engram | 0.9968131 | 0.9958341 | -0.098% | 2 |
| pypto_lib_dsl__moe | 1.4795339 | 1.4889674 | +0.638% | 4,5 |
| pypto_lib_dsl__layers_19_21 | 9.0713424 | 9.1043110 | +0.363% | 6,7 |
| latest__layers_19_21 | 3.5178777 | 3.5084740 | -0.267% | 1,2 |
| latest__moe | 0.8277235 | 0.8296483 | +0.233% | 4,5 |
| **Sum** | **21.6328923** | **21.6487042** | **+0.0731%** | — |

## Historical trace observations

The paired C2A full L3 captures on automatically allocated device 1 record 5,415 -> 4,728 returned FAA spans inside the business kernel window, while that window changes 607.188 -> 614.561 microseconds. The t26-to-t29 kernel gap decreases 40.310 -> 34.942 microseconds. Reduced primitive counts and this shorter local gap do not establish an unprofiled whole-graph gain. Profiling perturbation and changed execution interleaving are not isolated by this pair; no cause for the aggregate result is claimed.

Reproduction evidence is local under `artifacts/c2a-handoff-20261115/`: `private-cursor-final/`, `gm-ack-go/`, their gate wrappers, `gm-ack-go-l0/`, `gm-ack-go-comm-l0/`, and `trace-gm-ack/`. `artifacts/atomic-reduction-20261115/gm-ack-go12.json` contains all timed samples; `gm-ack-go-rejected.patch` preserves the complete candidate. The following adaptation restored coordinator confirmation with additional startup and exit checks.

## Historical coordinator adaptation

The same rendezvous, execution-return and retirement calls measure ACK-specific
FAA operations for all four MIX lane masks. Two resource members change from
4 arrival + 2 release reads to 1 + 1; three change from 9 + 3 to 2 + 2. The
checks include an inactive AIC coordinator for a pure-AIV MIX. Existing GM MIX,
startup and admission-progress suites pass. Synthetic startup workers bind the
same SSBUF state as real bootstrap; missing SSBUF is an error rather than a
second acknowledgement protocol.

The inactive-coordinator exit drain uses the same arrival-permission helper.
Protocol records describe the logically confirmed arrival set; only actual
atomic primitives produce FAA spans. SSBUF permission polls reuse the existing
arrival-readiness profiling phase. Validation artifacts for that adaptation are under
`artifacts/atomic-ownership-20261116/`.

That adaptation built in 70.01 seconds with a clean 16/16 A5 device IR
scan. Twelve paired AICore L0 mean durations sum to 21.4235579 ms before and
21.5307014 ms after (+0.500120%). The adaptation was retained for its measured ACK-demand reduction; its L0 sum
regressed. Each arm
uses one warmup and ten samples, automatic paired devices and a pre-timing
device rendezvous for two-rank cases. Eleven replay cases pass numerical
validation; native MoE remains timing-only. Full evidence is
`artifacts/atomic-reduction-20261115/gm-ack-owner12.json`.
