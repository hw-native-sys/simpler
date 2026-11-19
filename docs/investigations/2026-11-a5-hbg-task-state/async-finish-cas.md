# Confirm async FINISH with the canonical CAS

Retain a separate candidate on baseline `3db3f0e36`. The unique FINISH owner
constructs its expected canonical completion word from immutable task geometry
(total physical contributions N) and the already confirmed join result
(registered debt D). After validating those bounds, one CAS changes `(N, 0, D)`
to `(N, 0, 0)`. Remove the preliminary FAA(0) on that same completion word.
The CAS still confirms actual zero remaining computation and exact debt; static
metadata and join ownership alone do not grant DONE.

Run identity, unique finish ownership, publication barriers and DFX completion
causality are unchanged. A mismatched total, outstanding computation, wrong
debt or repeated finish fails without rewriting the gate. No shared state,
notification replicas, kernel or frontend changes are introduced. This uses
the existing canonical atomic authority; it does not substitute a weak load
for a positive completion proof. The same-address measurements in
`atomic_probe/ATOMIC_USAGE_GUIDE.md` section 4.6 motivate removing redundant
requests, but do not predict this candidate's end-to-end gain.

The existing invalid-owner regression now also covers a syntactically valid
but wrong immutable total, and nonzero remaining computation. The wrong-total
case fails against the baseline because it accepts that corrupted total; it
passes with the direct expected-state CAS. Four focused C++ suites pass:
async control, admission progress, execution ownership and GM MIX. The A5
build takes 76.01 seconds, with zero IR findings across 16/16 units.

Twelve paired AICore L0 mean durations sum to **21.6240982 -> 21.6194148 ms
(-0.021658%)**: six improve and six regress. The aggregate change is tiny;
it is not evidence of a stable latency improvement. The largest regressions
are full-sync C2A +1.229%, SWA +0.821% and latest layers 19-21 +0.514%.
Retention follows the demonstrated redundant-atomic reduction and the
less-than-one-percent aggregate-regression criterion. Each arm has one
warmup and ten samples on automatically allocated paired devices. Eleven
replay cases pass numerical checks; native MoE remains timing-only. All eight
communication arms pass the Device rendezvous audit; no pair required a retry
and no timed sample was excluded.

A paired DSL MoE L3 capture on automatically allocated devices 0 and 1 passes
numerical validation and has complete Scalar streams on both ranks. The six
captured FINISH scopes, three per rank, contain:

| Primitive in FINISH | Baseline calls | Candidate calls |
| ------------------- | -------------: | --------------: |
| Returned FAA | 6 | 0 |
| CAS | 6 | 6 |
| Bypass load | 6 | 6 |
| DSB ALL | 12 | 12 |

Across both rank-local first-to-last kernel windows, all recorded atomics
change 19,893 -> 19,750, returned FAA 11,423 -> 11,257, CAS 2,080 -> 2,101,
and bypass loads 278,198 -> 280,073. Only the six FINISH FAA removals are
directly attributable to this edit. Other aggregate differences reflect the
observed interleaving and profiling; no claim ties the entire atomic-count
change or the small L0 difference to those six reads.

Local evidence: `artifacts/atomic-reduction-20261115/finish-direct-cas12.json`,
the frozen `artifacts/c2a-handoff-20261115/finish-direct-cas/` source/runtime,
and `artifacts/atomic-continuous-20261116/` with red/green logs, gate audit,
`finish-direct-cas-l3/comparison.json` and the per-case `progress-results.tsv`.
Native source, tests and the design match the frozen build; this investigation
entry was added after measurement.

[Task-state experiment index](../2026-11-a5-hbg-terminal-claim-cache.md)
