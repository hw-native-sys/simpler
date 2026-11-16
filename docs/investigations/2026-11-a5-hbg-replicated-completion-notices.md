# A5 HBG replicated completion notices

## Decision

Remove the four-notice implementation at the user's request. The measured
twelve-case AICore L0 sum improved, but replicating completion state does
not establish an acceptable scheduler-wide ownership and observation model.
The optimization priority is to reduce atomic requests to the same address,
including contended and repeated requests, then reduce total atomic calls.
Do not retain a fixed number of replicated completion words as the solution.

## Experiment and evidence

The candidate marked producers with at least 16 potential readers and
published four epoch-tagged completion words on separate 128B regions.
Consumers selected a word by worker ID and confirmed it with returned FAA(0).
Canonical C/D/N completion remained authoritative. The task control grew
from 640B to 1152B, and selected producers issued four additional exchanges.

Paired twelve-case mean sums were 21.6288058 ms before and 21.3918099 ms
after (-1.0957%). Runs used one warmup and ten measurements, automatic
device allocation, and a device rendezvous before every dual-device launch.
Each dual-device sample used the maximum rank-local AICore wall duration.
Eleven cases validated replay outputs; native MoE was performance-only.
The experiment supports investigating shared-address pressure; it does not
quantify that pressure's share of the total workload duration.

Raw samples remain in
`artifacts/atomic-reduction-20261115/distributed-done12.json`; the frozen
candidate and L3/L4 captures remain under
`artifacts/c2a-handoff-20261115/distributed-done/` and `trace-notices/`.
Implementation commit `97c346b95` also includes a control-only Graph epoch
fallback checked by CPU regression after the device measurements.

The retained runtime returns to `54050e1e3`: private issuance cursors and
MIX completion aggregation remain, with one canonical completion word and
the existing worker-local DONE cache. No new performance measurements were
run for this removal. The experiment's performance table is historical,
not a measurement of the retained code after removal.
