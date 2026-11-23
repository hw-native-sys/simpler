# A5 HBG b-level priority with fixed execution owners

## Decision

Reject unconditional descending b-level priority. Keep the existing topological
generation order with b-level and original ID as tie breakers. Existing deferred
notification usage must continue to make progress without new frontend metadata
requirements, ownership transfer or a larger pending window.

The current scheduler assigns eligible exact-width waves deterministic block
owners while retaining this topological task order.

Same-device SWA L1 motivated the trial: both runtimes executed 578 instances
with identical static kernel IDs and block counts. HBG scheduled independent
tasks 23/26 at 45.960/68.413 µs relative to its first kernel; percounter scheduled
them at 159.071/197.991 µs. HBG task 8 spanned 77.647–129.456 µs, versus
49.485–58.467 µs. These profiled windows suggest a priority difference, not a
proven prediction of L0 savings.

## Progress counterexample

The existing `DeferredPredecessorLeavesIndependentCounterPublisherRunnable`
CPU test has a separate waiter W, five ordinary successors B depending on W,
and an independent notification publisher P submitted after them. W physically
returns after registering its condition; its completion debt still waits for P.
The test does not supply a static notification edge for that runtime condition.

Pure b-level puts W first, then B, then P. The test fails at the publisher-order
assertion: P has rank 6 and the first B has rank 1. Under the fixed four-token
window, issuing four waiting B tasks per worker leaves the fifth B at the shared
issuance head and P unreachable. This is the protocol consequence of the order;
no device hang was induced, and the CPU test stopped at the ordering assertion.

Adding a root-depth tie breaker only handles terminal B nodes. If B has a long
successor chain, its b-level remains higher than an independent publisher's;
that is not a general repair. Supplying new metadata in this test would change
the usage it protects rather than establish progress for that usage.

Kahn catalog and notification-planning checks passed, but the deferred-progress
check failed. No hardware performance result is claimed. The Host-only build
took 43.851 seconds and reused byte-identical device binaries; the candidate
manifest is marked rejected so it cannot be submitted by the benchmark launcher.

Local evidence is under `artifacts/hbg-sixteen-optimization-20261115/`:
`blevel-cpp-tests.log`, `blevel-priority-rejected.patch`, the frozen
`blevel-priority/` build, and `percounter-swa-current/` paired L0/L1/L3 captures.
Production code, tests and the unified design were restored to `90b8c7624`.

## Ready-token check before a new admission proposal

A narrower device-side trial retained Host order and existing commitments. Before
installing a new local admission proposal, an uncommitted worker checked whether
one of its received ordinary tokens already had complete fanin. Such a token
deferred the proposal; a committed cohort or named MIX retained priority.

The focused AIV/MIX ownership test, deferred-progress tests, all sixteen numerical
checks and two L3 captures passed. The actual A5 build took 530.08 seconds; its
IR scan covered 16/16 device translation units with zero findings.

Reject this trial too: paired sixteen-case L0 sums increased from 56.7443149 to
57.3343735 ms (+1.039855%). SWA changed from 0.5144214 to 0.5136925 ms, while
initial layers 19–21 increased from 15.1539189 to 15.6452610 ms. All communication
cases remain in the sum. A shorter SWA L3 task-8 tail does not override the L0
aggregate. The source and added test were reverted.

Evidence: `ready-priority-eval/summary.json`, the frozen
`ready-before-admission/` build, and `ready-priority-rejected.patch` under the
same artifact root. The unified design no longer proposes this extra check.

## Known ordered duties before new discovery

A second device priority trial skipped new cohort/MIX discovery while the worker
held a pending admission or an observed cohort. Existing cluster reservations,
ARM, lane preparation, formal group service and ordered processing continued.
The ordinary-token fast path already ran at the resident loop's first checkpoint;
this trial did not add another token scan. It retained the bounded preclaim
window from `d688a3d4b` and used `718121fd2` as its baseline.

SWA L3 motivated the change: late task-8 workers 27, 28 and 29 queried the issue
head for another 1.94–2.16 µs before retrying their existing admission and entering
the preclaim path. The change removed that intervening discovery. Five focused
CPU targets passed, including deferred progress, MIX and native Graph. The A5
build took 494.07 seconds, with zero findings across all 16 device IR units.
All sixteen numerical comparisons and the SWA L4 protocol audit passed.

Reject the trial: paired L0 sums increased from 57.2122144 to 57.7861192 ms
(+1.003116%). Native MoE increased from 1.6268509 to 2.2670433 ms (+39.35%).
Communication variability limits causal attribution, but that case remains in
the agreed aggregate. Latest C2A full also regressed by 17.13%; SWA improved by
0.42% and reindex by 0.80%. No alternate rows replace the original measurements.

The local trace improvement is insufficient: reindex task-24 entry spread fell
from 63.045 to 54.939 µs and task-27 spread from 52.167 to 41.291 µs in the paired
L3 captures. A known obligation can still be waiting for peers, so suppressing
all new discovery also removes useful lookahead during that wait. The experiment
does not isolate how much of the aggregate change comes from this trade-off.

Restore the production code and unified design to `718121fd2`. Retain
`known-obligation-first-rejected.patch`, the frozen `known-obligation-first/`
build and `known-obligation-first-eval/` results under the same artifact root.
The rejected manifest cannot be used by the normal benchmark launcher.

A separate same-device reindex comparison uses HBG `718121fd2` and percounter
frozen at `aa6d1bb1b` on upstream base `7688a860d`; the percounter subtree matches
that upstream commit. It records identical static kernel IDs, block counts and
709 executions. HBG tasks 24/27 have native `group_kind=1`: they use the GM
sync-MIX protocol. Their L3
entry spreads are 52.164/49.961 µs, versus percounter's task-wide
5.677/3.746 µs. Tails after the last entry are similar. Percounter does not export
block execution identities in that capture, so this is a task-wide comparison,
not a matched per-block causal proof. Evidence is in
`percounter-reindex-current/`; this comparison does not replace the sixteen-case
retention measurements or historical percounter reference values.

An additional same-device L1 pair confirms the spread with Scalar profiling
disabled. This is a separate automatic device allocation from the L0/L3 pair:

- Task 24: entry spread is 34.291 µs for HBG and 2.372 µs for percounter;
  tails after the last entry are 13.339 and 13.421 µs respectively.
- Task 27: entry spread is 31.392 µs for HBG and 2.707 µs for percounter;
  tails after the last entry are 27.374 and 27.747 µs respectively.

HBG's explicit execution identities also expose within-group entry skew: the
three members of a task-24 group have a median spread of 6.9035 µs (maximum
13.610); task 27 has 9.6125 µs (maximum 17.539). Percounter's task-wide spreads
bound any group's spread without inventing missing block identities. This
supports investigating sync-MIX handoff, but does not yet attribute the delay to
a specific primitive. Current `scheduler_gm_mix_prepare_lane` already prepares
its owned Call/Binding before all peers have arrived; do not propose that same
early preparation as a missing feature.
