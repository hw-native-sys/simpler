# Prefer canonical DONE when observing closed supply

Retain a separate candidate on baseline `9f98be9f3`. On a task-proof cache
miss, observe canonical completion before consulting the supply seal. A
strongly confirmed DONE certificate serves both supply closure and later
dependency checks. A pending completion still falls back to the existing
claim-seal confirmation; `C=0,D>0` never grants DONE. HostDone handling,
task/run identity, cache collision behavior, ownership, and kernel entry
checks are unchanged. No private or shared storage is added.

This addresses distinct observations, not repeated polling of one cached
task. The baseline DSL MoE traces have 399 and 405 pure-task supply FAA calls
after all captured physical return scopes for their task. These are potential
opportunities, not proven DONE observations: raw records contain no target
address or returned value, and asynchronous debt can outlive physical return.
The audit excludes activity outside the first-to-last kernel window and
requires every physical return scope to be present for this count.

The existing task-proof test fails before the change because a completed task
with a published supply seal retains only SUPPLY_CLOSED; after the change it
retains DONE for the subsequent fanin check. The same test preserves pending
async debt, colliding task identities and new-run isolation. Dispatch,
ordered-execution and GM MIX CPU suites pass. The native A5 build takes
76.01 seconds and passes all 16 IR checks.

The twelve paired AICore L0 means sum to **21.5005798 -> 21.5894029 ms
(+0.413120%)**. Three cases improve and nine regress. The largest relative
regressions are C1A reuse +2.758%, SWA +2.371%, and initial PyPTO-lib C2A full
+1.522%. This is below the agreed 1% aggregate rejection threshold, but is
not a measured speedup. Each arm has one warmup and ten samples on paired
automatically allocated devices. Eleven replay cases validate numerically;
native MoE remains timing-only. All eight communication arms pass the
per-launch device-rendezvous audit; samples use the slower rank-local span.

The paired DSL MoE L3 run on automatically allocated devices 4 and 5 has
complete scalar profiles and numerical validation on both ranks. Counts
inside the kernel window, summed across the two ranks, are:

| Primitive | Baseline calls | Candidate calls |
| --------- | -------------: | --------------: |
| Returned FAA | 12,481 | 12,182 |
| CAS | 2,088 | 2,094 |
| Atomic store issue | 6,450 | 6,400 |
| Non-returning FAA issue | 62 | 62 |
| **All atomic calls** | **21,081** | **20,738** |
| Bypass load | 288,710 | 288,622 |
| DSB ALL | 91,138 | 90,600 |

Recorded atomic calls decrease 1.627%, while L0 regresses 0.413%. A
sealed-but-unfinished task incurs an additional canonical pending hint with
this preference. These aggregate traces do not isolate that load's latency
or establish the cause of the L0 change; they also do not prove per-address
contention. Retention follows the agreed redundant-atomic reduction rule.

Local evidence: `artifacts/atomic-reduction-20261115/terminal-first-supply12.json`,
the frozen `artifacts/c2a-handoff-20261115/terminal-first-supply/` source/runtime,
and `artifacts/atomic-continuous-20261116/` with `terminal-supply-audit.json`,
red/green test logs, gate audit and `terminal-supply-l3/comparison.json`.
Post-build investigation text does not alter the measured runtime.

[Task-state experiment index](../2026-11-a5-hbg-terminal-claim-cache.md)
