# Local excess-ticket suppression with rollback retained

A narrower trial kept the ordinary ticket rollback and formal supply seal. A
worker-private bit remembered a strongly observed excess claim for the current
issuance task and suppressed further allocation attempts until the task changed.
It did not advance the catalog or establish completion. MIX and sync-start
registration were unchanged, and the bit used existing LocalState padding.

The deterministic stale-open test measured 72 FAA operations on the baseline
and two on the candidate (one initial refill plus eight refill/continuation
rounds), for both AIC and AIV, no advance before the seal, and normal next-task
allocation after publication. All 149 ordered CPU checks passed. The A5 build
took 70.01 seconds; the IR scan covered 16/16 units with zero findings.

The selected twelve AICore L0 mean durations summed to
21.5799021 ms before and 21.6660406 ms after (+0.399161%). Eight cases regressed,
four improved; fewer failed claims did not establish an aggregate benefit.
Each arm used one warmup and ten measurements on paired automatically allocated
devices; two-device launches rendezvoused before timing. Eleven replay cases
passed numerical checks; native MoE remained performance-only. One baseline
communication initialization failed on a port-60000 listen conflict before any
sample, so only that paired case was rerun and the failed attempt retained.

Retain this narrower candidate under the explicitly revised acceptance rule:
it eliminates proven redundant atomic requests without changing the shared
protocol, even though this batch did not improve L0. Do not report it as a
measured speedup. A change in contention timing is a hypothesis, not an
established explanation of the regression. Evidence: the frozen
`ticket-exhaustion-demand` source/runtime and paired runs under
`artifacts/c2a-handoff-20261115/`, and the full twelve-case report at
`artifacts/atomic-reduction-20261115/ticket-exhaustion-demand12.json`.

[Task-state experiment index](../2026-11-a5-hbg-terminal-claim-cache.md)
