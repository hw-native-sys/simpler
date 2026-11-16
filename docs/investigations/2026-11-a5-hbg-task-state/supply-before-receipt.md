# Publish ordinary supply after Claim, before payload receipt

Retain a separate candidate on baseline `bd4c27262`. The last ordinary
ticket owner publishes its supply seal after immutable execution-index
validation and the unique Claim CAS, before receiving Call/Binding cache
lines. In the current implementation, the token remains TICKETED with its
claimed handle until payload validation finishes, then enters WAIT_FANIN.
The existing combined acquisition entry still claims and receives for
active MIX/cohort callers, including its cached-handle behavior. MIX and
sync-start keep their own responsibility-registration publication points.

The claim and receipt steps have separate helpers and DFX scopes. Scalar
phase 142, its compact-codec bound, Host export bound and Python label are
updated together. No shared state or ownership transfer is added. A claimed
handle finishes bounded payload receipt even if supply publication reports
an error, because cancellation inspects its Call. Failed receipt marks the
prepared entry FAILED; neither supply closure nor failed preparation grants
logical DONE or consumer execution.

The motivating trace contains sixteen matched ordinary last-owner receipts
per rank inside the kernel window, averaging about 2.03 us of payload
receipt after Claim. This is a potential publication delay, not a predicted
wall-time saving. A deterministic CPU test observes the first real Call
cache invalidation. The baseline has seal zero and cannot prepare the
dependent task at that boundary. The candidate has a registered owner and
seal, lets another worker prepare the successor, and keeps that successor
waiting for real DONE. An invalid Call exercises failure and cancellation
of both unstarted owners. Five focused C++ suites and sixty scalar-profile
Python checks pass. The A5 build takes 74.01 seconds with 16/16 clean IR units.

Twelve paired AICore L0 means sum to **21.6737907 -> 21.6588182 ms
(-0.069081%)**: eight cases improve and four regress. The regressions are
latest C2A full +1.603%, DSL MoE +1.037%, DSL layers 19-21 +0.219%, and
full-sync C2A +0.210%. This batch shows only a small aggregate reduction.
Each arm has one warmup and ten samples on automatically allocated paired
devices. Eleven replay cases validate numerically; native MoE remains
timing-only. All eight communication arms pass the device-rendezvous audit.

The initial DSL layers 19-21 pair failed before timing: the HCCL log reports
port 60000 already in use during gate communication initialization. Its
evidence is retained under `failed_attempts/`; only that complete pair was
rerun. No timed sample was selected or discarded to improve a result.

A paired DSL MoE L3 capture on automatically allocated devices 0 and 1 has
complete profiles and numerical validation on both ranks. Counts in each
rank's first-to-last kernel window, summed across ranks, are:

| Primitive | Baseline calls | Candidate calls |
| --------- | -------------: | --------------: |
| Returned FAA | 11,586 | 11,434 |
| CAS | 2,098 | 2,061 |
| Atomic store issue | 6,364 | 6,366 |
| Non-returning FAA issue | 62 | 62 |
| **All atomic calls** | **20,110** | **19,923** |
| Bypass load | 284,078 | 274,074 |
| DSB ALL | 89,110 | 86,412 |

Recorded atomics decrease 0.930%. All sixteen matched last owners per rank
have a store issue between Claim and payload receipt, versus zero before.
The source identifies this as the supply publication path, but raw records
have no atomic addresses or returned values, and issue time is not remote
visibility time. Neither arm captures a dependent ordinary-token, group-
reservation or payload-receipt start overlapping that small receipt interval.
Short scopes can be omitted. Do not claim that device-side preparation
overlap has been demonstrated by this capture, or attribute the L0 change
solely to earlier publication.

Local evidence: `artifacts/atomic-reduction-20261115/early-owned-supply12.json`,
the frozen `artifacts/c2a-handoff-20261115/early-owned-supply/` source/runtime,
and `artifacts/atomic-continuous-20261116/` with red/green logs, the receive
tail audit and retry/gate records. Its `early-supply-l3/` folder contains
`comparison.json`, `overlap-analysis.json` and `overlap-all-preparations.json`.
Native source and tests match the frozen
build; the final investigation prose was added after measurement.

[Task-state experiment index](../2026-11-a5-hbg-terminal-claim-cache.md)
