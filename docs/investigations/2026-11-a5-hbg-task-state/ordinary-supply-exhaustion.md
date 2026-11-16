# Confirm exhausted ordinary supply before reserving another ticket

Retain a separate candidate on baseline `72c1c0633`. Ordinary token refill
uses the existing ordinary-MIX preflight: a bypass hint at or beyond N is
confirmed by one FAA(0). Confirmation invalidates the current issue snapshot
and sets only the existing task-private exhaustion bit. It does not publish
a supply seal, advance the catalog, grant completion or allocate a token.
Valid tickets are irrevocable, and excess-ticket rollback cannot remove a
valid owner. The final shared cursor remains exact; raced excess allocations
still use the original rollback. The optional continuation hint is unchanged.

The existing AIC/AIV exhausted-supply test fails on the baseline, counting
two FAA operations, and passes with one after the change. Eight subsequent
refill/continuation checks add no FAA. No successor is allocated before the
real seal; publishing the seal then permits the next task normally. Ordered,
dispatch and GM MIX CPU suites pass. The A5 build takes 76.02 seconds, with
all 16 IR units clean. This differs from the rejected monotone-counter trial:
it avoids a known-excess reservation without changing the shared protocol.

Twelve paired AICore L0 means sum to **21.5145294 -> 21.5892693 ms
(+0.347393%)**. Three cases improve and nine regress; the largest relative
regressions are latest C2A full +1.300%, C1A reuse +1.243% and native MoE
+1.107%. Eleven replay cases validate numerically; native MoE is timing-only.
Each arm uses one warmup and ten samples on automatically allocated paired
devices. All eight communication arms pass the per-launch device-rendezvous
audit; samples use the maximum rank-local kernel span. Retention follows the
agreed atomic-reduction rule and aggregate regression below 1%, not an
observed speedup.

The matched DSL MoE L3 capture on automatically allocated devices 1 and 3
has complete profiles and numerical validation on both ranks. Recorded calls
inside each rank's first-to-last kernel window, summed across ranks, are:

| Primitive | Baseline calls | Candidate calls |
| --------- | -------------: | --------------: |
| Returned FAA | 12,250 | 11,599 |
| CAS | 2,060 | 2,102 |
| Atomic store issue | 6,369 | 6,397 |
| Non-returning FAA issue | 62 | 62 |
| **All atomic calls** | **20,741** | **20,160** |
| Bypass load | 276,235 | 294,043 |
| DSB ALL | 87,235 | 91,546 |

FAA decreases 5.314% and all recorded atomics decrease 2.801%, while bypass
loads increase 6.447% and DSB ALL increases 4.942%. The new hint adds a load
to live allocation attempts. Aggregate L3 counts also include changes in
polling and observation timing; they do not establish which changes caused
the L0 regression or identify exact atomic target addresses. The deterministic
test establishes the specific two-to-one exhausted-claim reduction.

Local evidence: `artifacts/atomic-reduction-20261115/ordinary-terminal-prefilter12.json`,
the frozen `artifacts/c2a-handoff-20261115/ordinary-terminal-prefilter/`
source/runtime, and `artifacts/atomic-continuous-20261116/` with red/green
logs, gate audit and `ordinary-terminal-l3/comparison.json`. The final native
source and tests match the frozen build; only design/investigation prose was
refined after compilation.

[Task-state experiment index](../2026-11-a5-hbg-terminal-claim-cache.md)
