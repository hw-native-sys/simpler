# Sync MIX closure through the existing SSBUF frontier

Retain the closure-channel cleanup on baseline `0ec8256ae` under the agreed
atomic-reduction rule, not as a measured speedup. Remove the per-task/cluster
GM close-seal arrays, the coordinator's publishing CAS, and peers' confirming
FAA reads. The coordinator already publishes its monotonically retired
admission prefix in SSBUF `sync_frontier[0]`; peers consume that same prefix.
No notification replicas or extra shared/private state are introduced.
In this experiment admission-local state shrank from 184 to 176 bytes.
The current layout is 168 bytes after upstream removed fixed stride metadata.

The AIC coordinator, including the inactive AIC of vector-only MIX, advances
only after its group is IDLE, its slots are empty, and error/epoch checks pass.
Zero-work coordinators still strongly confirm global ticket exhaustion.
The GM publication barrier precedes frontier advancement. A peer acquires
the current-run frontier, rejects values beyond admission capacity, and
requires its own group and slots to have retired. A late peer can consume a
closed prefix even after another lane's later task reuses the mailbox.
This is local admission retirement, not TaskDone or a global claim seal.
Cluster INIT acquisition, rather than the epoch value alone, excludes stale
SSBUF from an earlier run.

DFX records `close_frontier` and the actual coordinator retirement that
precedes the peer observation. It does not fabricate a GM atomic event or
retain the removed close-cell layout. Existing startup fixtures now clear
SSBUF only before cluster INIT, matching the real initialization ownership;
late AIV arrivals must not erase their coordinator's published frontier.

The four-mask coordinator test fails on the baseline and passes after the
change: closing an owned sync position performs zero GM atomics rather than
one CAS. Layout, admission, MIX, late-start/exit and export CPU suites pass,
as do 1,077 Python protocol, causality, scalar-profile and failure-fixture
checks. A native A5 build takes 76.01 seconds and passes all 16 IR checks.
The existing eight-block, two-AIV-lane sync-start hardware test passes on
automatically allocated device 4, including exact output and co-residency.

All twelve paired L0 cases complete. Eleven replay cases validate numerically;
the native MoE case remains explicitly timing-only. Each arm uses one warmup
and ten timed samples on the same automatically allocated devices. Both ranks
rendezvous on Device before each two-device launch; the sample is the maximum
of rank-local first-kernel-start to last-kernel-end spans. The eight paired
communication arms all pass the rendezvous-log audit.

The sum of twelve AICore L0 means changes from **21.4158391 to 21.5496156 ms
(+0.624661%)**. All twelve cases regress in this batch; the largest relative
changes are SWA +2.004%, native MoE +1.913%, and latest layers 19-21 +1.757%.
The aggregate remains below the agreed 1% rejection threshold. Retention is
justified by removing redundant protocol operations, not an observed speedup.

A paired DSL MoE L3 capture on automatically allocated devices 4 and 5 has
complete scalar profiles and numerical validation on both ranks. Inside each
rank's kernel window, totals across the two ranks are:

| Primitive | Baseline calls | Candidate calls |
| --------- | -------------: | --------------: |
| Returned FAA | 12,846 | 12,401 |
| CAS | 2,246 | 2,062 |
| Atomic store issue | 6,354 | 6,421 |
| Non-returning FAA issue | 62 | 62 |
| Bypass load | 273,546 | 278,450 |
| DSB ALL | 86,155 | 88,544 |

Total recorded atomic calls decrease from 21,508 to 20,946. The accompanying
load/barrier increases do not establish the cause of the L0 regression:
L3 perturbs execution, omits short parent scopes, and carries no atomic target
addresses or returned values. The code and focused test establish which
closure operations are removed; aggregate trace counts show the net observed
traffic, not a per-address contention proof.

Reproduction evidence is local: `artifacts/atomic-reduction-20261115/` holds
`mix-frontier-close12.json`; `artifacts/c2a-handoff-20261115/` holds the frozen
`mix-frontier-close` source/runtime and paired L0 outputs. The frozen runtime
is unchanged by later comment, documentation and duplicate-test cleanup;
`delivery-manifest.json` and `delivery-source/` preserve that exact overlay.
`artifacts/atomic-continuous-20261116/` holds the smoke result, gate audit,
logs, `mix-frontier-l3/comparison.json` and full raw captures. These artifacts
are not shipped in Git.

[Task-state experiment index](../2026-11-a5-hbg-terminal-claim-cache.md)
