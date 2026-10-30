# A5 completion batching: shortcuts and singleton polling

**Date**: 2026-10-28
**Verdict**: the two historical shortcuts remain dropped. The later direct-grouping
implementation selects per-core singleton polling over shared-barrier polling.

## Singleton group scan

Review of [#2752](https://github.com/hw-native-sys/simpler/pull/2752) found that
completion batching improves the targeted attention workloads but slows the
ordinary alternating-matmul/add and BGEMM benchmarks. Both benchmarks submit
tasks with one physical subtask. Such a task cannot legitimately contribute
two completion events, so searching earlier events for the same task cannot
combine its credits.

The isolated experiment `ece1eb48` adds
`if (slot->total_required_subtasks != 1)` around that pointer search in
`append_completion`. It preserves the existing path for multi-subtask tasks
and does not change ordering, barriers, slab checks or callbacks. The field is
immutable after task publication, and outstanding physical completions keep
the referenced slot alive during classification.

### Measurement

Baseline `1f12a442`, batching `c35db013`, experimental child `ece1eb48`; all use
ISA `dcebd22` and CANN 9.3.0 on A5. Each comparison holds one device through
`task-submit`, preserves inputs, executable compute images and original golden
checks, and uses two fresh processes per variant. The table contains means of
200 measured **Effective** execution windows, without tail trimming.

Runtime order is baseline/batching/experiment/experiment/batching/baseline,
with 110 calls per process and the first 10 treated as warmup. Attention order
is batching/experiment/experiment/batching, using the unchanged
pypto-lib `eba9beaa` manifest: B12/S6/history8192/TP1, 100 measured calls and
10 warmups, PyPTO `3562aa1e`.

| Workload | Baseline (us) | Batching (us) | Skip scan (us) | Change vs batching |
| -------- | ------------: | ------------: | -------------: | -----------------: |
| Alternating matmul/add, Case1 | 1406.054 | 1442.172 | 1447.531 | +0.37% |
| BGEMM, Case0 | 1341.958 | 1459.353 | 1446.462 | -0.88% |
| SWA | — | 611.684 | 608.893 | -0.46% |
| C1 full | — | 1915.270 | 1921.858 | +0.34% |
| C2 full | — | 1894.585 | 1885.712 | -0.47% |

All numerical gates pass. The experimental runtime still trails the baseline
by 2.95% on Case1 and 7.79% on BGEMM. The small mixed-direction changes relative
to batching do not establish a useful improvement. The pointer-search count
alone therefore does not explain the observed whole-graph regression.

## Scratch transition initialization

A second independent child of `c35db013`, `8f5d0378`, removes the five default
boolean initializers from A5 TMR's `SlotTransition` and explicitly value-initializes
the local result in `decide_slot_transition`. Every used scratch observation is
assigned a complete transition before application. This removes 40 redundant
`strb wzr` instructions from the polling prologue: the function shrinks from
3872 to 3712 bytes, with its 624-byte stack frame unchanged.

The same paired runtime protocol produces the following independent comparison;
do not pool it with the preceding experiment.

| Workload | Baseline (us) | Batching (us) | Initialize on classification (us) | Change vs batching |
| -------- | ------------: | ------------: | --------------------------------: | -----------------: |
| Alternating matmul/add, Case1 | 1400.292 | 1437.852 | 1458.544 | +1.44% |
| BGEMM, Case0 | 1339.784 | 1418.509 | 1373.620 | -3.16% |

All golden checks pass. BGEMM improves in this comparison but remains 2.53%
slower than baseline; Case1 becomes worse and remains 4.16% slower than baseline.
The change is not adopted, and attention is not remeasured for this rejected
candidate. Fewer generated instructions do not establish a whole-graph win or
identify the complete cause of the batching tradeoff.

## Historical disposition

Keep both shortcuts out of the PR. Reconsider them only with a workload and
repeatable whole-graph measurement that demonstrate a benefit. The PR review
record contains the broader capacity comparison, current-runtime measurements
and raw-sample artifact links.

## Direct grouping with per-core singleton polling

**Date**: 2026-10-29

The revised implementation replaces the prior-event pointer scan with a direct
ring/task-slot lookup and tracks running/pending multi-subtask slots in
`CoreTracker`. An isolated comparison keeps that implementation fixed and
changes only the outer polling selector:

- A (`97ce5732`) uses per-core polling when an owner has only singleton tasks.
  Mixed batches preserve their original event order and skip singleton lookup.
- B (`9217a893`) always uses the eight-observation shared-barrier path, including
  singleton-only owners. Singleton events still skip grouping.

Both are based on main `0cf5b3d5`, ISA `217bb989` and CANN 9.3.0. On the existing
alternating-matmul/add Case1 (1,000 independent singleton tasks), four complete
fresh processes run A/B/B/A on one allocated A5 device. Original golden checks,
input/compute identity and runtime/cache guards pass. Each process retains all
110 calls; the means below include 200 measured calls per variant after ten
warmups per process, with no tail trimming.

| Polling strategy | Device mean (us) | Effective mean (us) |
| ---------------- | ---------------: | ------------------: |
| A: per-core singleton polling | 1403.056 | 1360.308 |
| B: shared barrier for singleton owners | 1459.832 | 1416.477 |

A reduces Device by 3.89% and Effective by 3.97% relative to B. Both A process
means are lower: Device 1404.506/1401.605 us versus B 1472.082/1447.582 us.
The high samples remain included. This comparison is the complete A/B/B/A
subsequence of a larger control matrix interrupted by its allocation timeout;
the incomplete main/control comparison is not used as baseline acceptance.
Separate complete baseline comparisons and raw evidence are linked from the PR.

Separate instrumented captures confirm that both ordinary workloads still
publish one completion RMW per singleton task. A sends none of their events
through grouping. It performs more read barriers, so this is a measured
whole-path tradeoff, not a fixed cost assigned to a barrier or pointer comparison.
Generated A singleton polling has a 288-byte frame versus the shared path's
640-byte frame; its batch scratch initialization is absent. Diagnostics and code
size do not substitute for the uninstrumented comparison above.

Select A and keep B out of the PR. The historical scan-only shortcut left the
batch polling structure in place; these results do not turn that rejected
shortcut into an independently proven fix. Grouping remains linear, and the
owner-level selector avoids reordering mixed-task completion callbacks.

### Remaining cost

Complete comparisons against the same main revision remove the prior wide-task
and BGEMM regressions: mean Device changes by -3.84% and -0.96%, respectively.
The three manifest attention cases retain Device improvements of 8.94–15.11%.
However, a two-task short graph regresses in two independent balanced comparisons:
Device 69.815 to 72.056 us (+3.21%) and 69.453 to 70.855 us (+2.02%). The second
comparison also increases Effective from 29.487 to 31.052 us (+5.31%). Broad
performance acceptance therefore remains open.

Warm geometry rechecks do not explain that regression: their reset-phase span
increases by about 0.05 us in the first comparison and decreases in the second.
Scheduling and graph-build spans increase, but overlap and cannot be added or
assigned to individual instructions. Raw per-call phases and untrimmed statistics
remain in the PR evidence.

The lookup uses 64 KiB per scheduler at default ring geometry, retained across
runs. Initial allocation increases the observed cold reset span from about 9 us
to 159–170 us; full cold Device increases by about 28–30 us. Each comparison has
only two first calls per variant, so these are descriptive observations rather
than a fixed startup-cost estimate. A maintainer must accept the remaining
short-graph and startup tradeoffs, or further optimization must address them,
before treating performance acceptance as complete.

### Singleton completion helper

An isolated child of A, `67ae7344`, specializes `complete_slot_task` on the
polling mode. The singleton specialization compiles out deferred-credit
bookkeeping while retaining physical error checks and deferred registration.
Independent review, 57 normal and 57 ASan/UBSan test executions, native GCC 7
compilation and source/cache/input guards pass.

A separate short-graph comparison uses main/A/C/C/A/main, with the same
110-call protocol and 200 measured calls per variant. All numerical checks pass.

| Variant | Device mean (us) | Effective mean (us) |
| ------- | ---------------: | ------------------: |
| Main | 69.227530 | 30.079325 |
| A: selected implementation | 69.494705 | 29.288280 |
| C: specialized completion helper | 69.927630 | 29.985785 |

C is 0.62% slower than A in Device and 2.38% slower in Effective, so it is dropped.
Generated helper size also changes through inlining decisions and does not
establish a timing benefit. Attention and native error replay are not repeated
for this rejected candidate.

In this third comparison, A is +0.39% Device and -2.63% Effective relative to main.
Keep this variation alongside the two earlier regressions; do not pool the runs
or discard the unfavorable comparisons. Broad non-regression remains unproven.
