# Fold single-consumer ordinary control nodes

Retain a separate candidate on baseline `271067157`. Ordinary inline control
nodes have no kernel execution; they join predecessor completion. At Host bind,
fold a zero/single-consumer control into its live consumer when the merged fanin
fits the existing 128-entry limit. Keep shared joins and overflow shapes live.
The worklist also handles control chains and newly eligible joins. Direct
completion-driven fanout was considered, but would require another bounded
runtime service protocol; this candidate instead removes eligible join work
before dispatch.

The pass runs after the original graph ordering, preserving physical catalogs,
schedule ranks and their original level metadata. Consumers inherit every real
predecessor, and duplicate inherited edges are removed. Repack the live rows
inside the existing fanin pool; zero/single-consumer contraction cannot increase
the logical edge count. Keep an aliased source image intact if a separate-row
repack cannot fit. Tensor/scalar regions, Host Tensor lifetime bookkeeping and
business kernels are untouched. No Device storage or runtime notification state
is added. The resident's existing bounded fanin reduction runs afterward.

Folded nodes use the existing Host-completed metadata representation. Subtract
them from the catalog control count and include them in inline-completed run
accounting; the AICPU still requires all real tasks to complete. Removing a
terminal no-consumer join cannot finish the run while a real task remains.
Actual defer_wait kernels and their debt remain; consumers now wait directly
on those predecessors. MIX and sync-start resource contracts are unchanged.
Native Graph ENTER/EXIT/CONTROL_DUMMY retain their current instance handling;
this pass operates only on the ordinary packed image.

The motivating DSL MoE has three pure joins, each with one consumer. In its
previous L3, captured fanin FAA for those controls total 225/311 by rank, and
bypass loads directly attributed to control scopes total 7,513/11,555. These
are captured subsets, not a predicted wall-time benefit. The shared-address
measurements in `atomic_probe/ATOMIC_USAGE_GUIDE.md` section 4.6 motivate
removing the requests rather than replicating completion words.

A failing contraction test becomes green with the pass: a control chain is
removed, its consumer inherits the real predecessors, and nonzero async debt
still blocks that consumer until canonical DONE. Shared and over-limit joins
remain live. All fourteen task-catalog checks pass, as do native-graph-plan
and protocol-export CPU suites. Review removed a vacuous assertion on an
unmodified plan and reran the fourteen catalog checks. The native build takes
78.01 seconds, with zero IR findings across 16/16 units. Device runtime source
hashes match the baseline; production changes are confined to Host planning.

Twelve paired AICore L0 mean durations sum to **21.5357806 -> 21.4962053 ms
(-0.183765%)**, with six faster and six slower cases. Latest C2A full regresses
2.165% and DSL MoE regresses 1.197%; latest layers 19-21 improves about 1.10%.
Each arm uses one warmup and ten samples on automatically allocated paired
devices. Eleven replay cases pass numerical validation; native MoE is still
timing-only. All eight communication arms pass the Device-rendezvous audit.
No retry or timed-sample exclusion was needed. This is a small observed
aggregate reduction, not proof of a stable speedup in every workload.

A paired DSL MoE L3 capture on automatically allocated devices 0 and 2 passes
numerical validation with complete Scalar streams. For each rank, all 913
`(task, block, lane, executed)` identities match exactly across the two arms.
Captured fanin FAA for the three controls changes from 246/297 to zero/zero.
Their enclosing control-service scopes contain 550 returned FAA, 37 CAS and
20,250 bypass loads before the change, and none of those operations afterward.
A few outer control scopes still include completion-count flushes; they are
not evidence that the folded join service still runs.

Counts across both rank-local first-to-last kernel windows are:

| Primitive | Baseline calls | Candidate calls |
| --------- | -------------: | --------------: |
| Returned FAA | 11,363 | 10,993 |
| CAS | 2,105 | 2,082 |
| Atomic store issue | 6,322 | 6,411 |
| Non-returning FAA issue | 62 | 56 |
| **All atomic calls** | **19,852** | **19,542** |
| Bypass load | 280,748 | 274,187 |
| DSB ALL | 88,058 | 91,909 |

Recorded atomics decrease 1.562%. Some traffic increases, including stores
and barriers; do not equate removed control-scope traffic with the net change
or multiply primitive counts by microbenchmark latency to predict L0 savings.
Short scopes may be omitted and raw records carry no atomic addresses/values.
The unchanged execution identities and real numerical checks substantiate that
the reduction is not obtained by skipping business work.

Local evidence: `artifacts/atomic-reduction-20261115/fold-inline-controls12.json`,
the frozen `artifacts/c2a-handoff-20261115/fold-inline-controls/` source/runtime,
and `artifacts/atomic-continuous-20261116/` with red/green logs, gate audit,
`post-owned-done-demand.json`, `fold-inline-l3/comparison.json` and the per-case
`progress-results.tsv`. Native source and design match the measured build.
Only the reviewed test assertion removal and this investigation/index prose
were changed after compilation; the modified catalog suite was rerun.

[Task-state experiment index](../2026-11-a5-hbg-terminal-claim-cache.md)
