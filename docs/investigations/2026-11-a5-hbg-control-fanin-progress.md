# A5 HBG control-node fanin progress

## Single-entry trial

A worker-private task ref and fanin prefix retained already confirmed DONE
predecessors when the general sixteen-entry task-proof cache collided. The
control scan still visited eight nodes per opportunity and did not wait on a
particular control. A deterministic collision case removed four repeated FAA
reads; another independent control continued to progress.

The twelve paired AICore L0 mean sums changed from 21.4156622 to
21.5043764 ms (+0.414249%). This is below the one-percent rejection threshold,
but paired DSL MoE L3 did not establish a reduction in the targeted control
traffic. FAA attributed to captured control scopes changed from 578 to 617
on rank 0, and from 633 to 636 on rank 1. Total in-window FAA changed from
7326 to 7375 and 7419 to 7321 respectively.

The single slot forgets its prefix when another unfinished control is checked.
The workload has several such controls. Short function scopes may be omitted
from L3 and changed interleaving affects counts, so these captures do not prove
the cause of the L0 regression or identify every atomic address. They do show
that the isolated CPU reduction did not demonstrate the intended real-workload
control-path reduction.

Supersede the single-entry candidate with a bounded, task-keyed private control
progress cache. It must preserve scan fairness, exact identity, run lifetime and
collision safety; collisions may only forget evidence. The proposed capacity is
sixteen, matching the existing private proof-cache scale, with no model-name or
task-ID specialization. No single-entry production commit is retained.

Evidence: `artifacts/atomic-reduction-20261115/control-fanin-prefix12.json`,
`artifacts/atomic-continuous-20261116/control-prefix-l3/comparison.json`, and the
source/runtime snapshot under `artifacts/c2a-handoff-20261115/control-fanin-prefix/`.

## Task-keyed private cache

Retain sixteen packed private entries per worker (128 bytes), using the existing
private proof-cache capacity. Exact task identity gates prefix reuse; hash
collisions only forget progress, and zero progress cannot evict another node's
positive proof. Run initialization clears all entries. Graph validation bounds
refs to INT32_MAX and fanin counts to the low32-bit field. Scan order and budget
remain unchanged; there is no shared election or notification state.

The existing two focused tests cover proof-cache collisions, multiple pending
controls, identity collisions and independent progress. The multiple-control
case changes three redundant FAA reads under the single-entry design to zero.
All 152 ordered and 19 inline-service checks pass. The A5 build takes 76.01
seconds, with zero IR findings across 16/16 translation units.

Twelve paired AICore L0 mean durations sum to 21.3625372 ms before and
21.4278457 ms after (+0.305715%). This is below the one-percent rejection
threshold and is recorded as a regression, not a measured latency improvement.
Each arm uses one warmup and ten samples with automatically allocated paired
devices; two-rank cases rendezvous on Device before timing. Eleven replay cases
pass numerical checks; native MoE remains performance-only.

A fresh paired DSL MoE L3 capture records in-kernel-window FAA counts of
7266 -> 6857 on rank0 and 7346 -> 6950 on rank1. FAA attributed to captured
control scopes changes from 606 -> 268 and 641 -> 315; the visible task32 fanin
scopes change from 369 -> 98 and 371 -> 103. Both ranks have complete Scalar
streams. Function attribution still has the short-scope limitation above, and
profiling counts do not predict the L0 latency change.

Current evidence: `artifacts/atomic-reduction-20261115/control-fanin-cache12.json`,
`artifacts/atomic-continuous-20261116/control-cache-l3/comparison.json`, and the
frozen source/runtime under `artifacts/c2a-handoff-20261115/control-fanin-cache/`.
The inline-service fixture also binds the cluster signal required by the current
GM MIX protocol; its old null-signal failure was reproduced with baseline
headers before the fixture correction.
