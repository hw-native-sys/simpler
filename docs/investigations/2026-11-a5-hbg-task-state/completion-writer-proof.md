# Reuse the completion writer's own DONE proof

Retain a separate candidate on baseline `94eee15ae`. Successful ordinary and
MIX contributions returning TASK_LAST, the winning control-node completion CAS,
and successful async FINISH now remember DONE in the existing sixteen-entry
worker-private task-proof cache. The same core can reuse the returned atomic
fact for subsequent supply and dependency checks instead of confirming it
again with FAA(0). The production change adds one cache update at each of
these four publication points, without a new shared notification, cache entry,
layout field, polling path or kernel change.

Partial contributions, TASK_LAST_DEFERRED, failed CAS and resource-only
retirement do not establish this proof. Cache identity, run reset and collision
rules remain intact; collisions only discard knowledge. Pending completion
counts, resource release, error gates and DFX publication keep their existing
responsibilities. A successful canonical publication establishes the fact even
if a later unrelated operation fails; run errors still prevent execution.

The motivating DSL MoE trace has fifteen inferred ordinary last-return owners
per rank, excluding captured async FINISH tasks. It captures four and six
subsequent FAA operations in matching supply scopes. This identifies an
opportunity, not an exact saved-call forecast: raw records have no addresses or
returned values, short scopes may be omitted, and private-cache eviction is
not modeled. The audit also considers only single-predecessor fanin rows when
attributing a dependency read; an inferred physical last return alone is not
proof that async debt was absent at that instant.

A deterministic AIC/AIV test completes the actual producer and checks its
consumer's dependency through the runtime. The baseline performs one FAA for
each variant; the candidate performs zero and still advances the fanin prefix.
Existing partial-return, MIX, control-node and async checks cover the proof's
publication boundaries. All five focused C++ suites pass. The A5 build takes
76.01 seconds, with zero IR findings across 16/16 units.

Twelve paired AICore L0 mean durations sum to **21.6388103 -> 21.5690454 ms
(-0.322406%)**: eight improve and four regress. Regressions are latest MoE
+1.980%, full-sync C2A +1.091%, DSL C2A +0.145% and latest layers 19-21 +0.134%.
Each arm uses one warmup and ten samples on automatically allocated paired
devices. Eleven replay cases validate numerically; native MoE remains
timing-only. All eight communication arms pass the Device rendezvous audit.
No retry or timed-sample exclusion was needed.

A fresh paired DSL MoE L3 capture on automatically allocated devices 4 and 5
passes numerical validation and has complete Scalar streams. It matches
fifteen inferred ordinary last-return owners per rank in each arm. Captured
same-owner FAA after these returns changes from three/two to zero/zero.
This is a visible subset, subject to the attribution limits above; it is not
an exhaustive address-level proof. Counts across both rank-local first-to-last
kernel windows are:

| Primitive | Baseline calls | Candidate calls |
| --------- | -------------: | --------------: |
| Returned FAA | 11,272 | 11,337 |
| CAS | 2,084 | 2,125 |
| Atomic store issue | 6,346 | 6,301 |
| Non-returning FAA issue | 62 | 62 |
| **All atomic calls** | **19,764** | **19,825** |
| Bypass load | 280,900 | 280,141 |

Aggregate recorded atomics increase 0.309%, despite eliminating the targeted
reconfirmation in the deterministic check and the matching trace subset.
Do not describe this capture as a net atomic-count reduction, or attribute
its other count differences to the four cache updates. Retention follows the
verified redundant-read removal and the paired twelve-case L0 gate; the
measured wall-time reduction does not prove a stable or isolated latency cause.

Local evidence: `artifacts/atomic-reduction-20261115/owned-done-proof12.json`,
the frozen `artifacts/c2a-handoff-20261115/owned-done-proof/` source/runtime,
and `artifacts/atomic-continuous-20261116/` with red/green logs, the gate audit,
`owned-done-audit.json`, `owned-done-l3/comparison.json`, its baseline/candidate
owner audits and the per-case `progress-results.tsv`. Native source, tests and
the design match the frozen build; this investigation entry was added after
measurement.

[Task-state experiment index](../2026-11-a5-hbg-terminal-claim-cache.md)
