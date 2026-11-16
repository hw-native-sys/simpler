# A5 HBG GM MIX arrival permission and ACK reuse

## Decision

Do not retain this implementation: the twelve-case AICore L0 sum did not improve. The measured increase is 0.0731%; this small difference is not evidence of a stable regression, but it does not meet the retention criterion.

## Protocol and validation

The candidate uses the existing coordinator to confirm the group arrival set, then publishes its sequence through an additional 8-byte SSBUF field. Peers consume that permission. Per-worker arrival/release masks retain positive ACK observations within a group; self-publication supplies the local ACK proof. Failed-release status remains distinct from successful completion, and masks reset on group changes. Shared GM descriptor and global sync-start admission semantics are unchanged.

The baseline is local `241af7b91` (private issuance cursors), after `d8a71ec3a` (MIX completion aggregation). Both arms use frozen A5 runtime binaries, unchanged business kernels and inputs, one warmup and ten measurements per case. Devices are allocated automatically and paired arms use the same devices. Every dual-device launch rendezvous precedes the measured run; each sample uses the slower rank's AICore-local wall duration. Eleven cases validate replay outputs; latest native MoE remains performance-only.

Five focused C++ suites pass, including all 652 MIX checks after updating the explicit coordinator-to-peer ordering. The actual device IR scan covers 16/16 translation units with zero findings. C2A full L4 export and protocol conversion also pass.

## Paired AICore L0 results

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

## Trace observations

The paired C2A full L3 captures on automatically allocated device 1 record 5,415 -> 4,728 returned FAA spans inside the business kernel window, while that window changes 607.188 -> 614.561 microseconds. The t26-to-t29 kernel gap decreases 40.310 -> 34.942 microseconds. Reduced primitive counts and this shorter local gap do not establish an unprofiled whole-graph gain. Profiling perturbation and changed execution interleaving are not isolated by this pair; no cause for the aggregate result is claimed.

Reproduction evidence is local under `artifacts/c2a-handoff-20261115/`: `private-cursor-final/`, `gm-ack-go/`, their gate wrappers, `gm-ack-go-l0/`, `gm-ack-go-comm-l0/`, and `trace-gm-ack/`. `artifacts/atomic-reduction-20261115/gm-ack-go12.json` contains all timed samples; `gm-ack-go-rejected.patch` preserves the complete candidate. Production code and its test migrations are removed.
