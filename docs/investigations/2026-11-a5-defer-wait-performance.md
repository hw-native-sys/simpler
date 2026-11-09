# A5 deferred-completion performance experiments

These runtime experiments use frozen compute binaries and original goldens.
Unless stated otherwise, comparisons use L1 device-wall timing. Outlined O3
builds are private experimental artifacts; default production compiler flags
and inline declarations remain unchanged.

## Runtime candidates

### Empty-guard outlining follow-up

A later experiment kept empty sealing and pre-registration service guards inline,
and moved only nonempty processing into two static noinline helpers. The prior
outlined binary saved 13 registers before its early-return guards. The split
passed 10 relevant C++ targets, 47 selected A5 items and 56 capture audits.

On the same A5 device, separate ten-round L1 comparisons measured CSA at
2.719 versus 2.714 ms and SWA at 1.045 versus 1.036 ms for the prior and split
candidates. Both retained the same compute binaries and original goldens. The
small differences do not explain the remaining 20% CSA and 16% SWA regression;
baseline variation also limits their significance. The split is not retained.
Its source, binaries and raw comparisons remain under the sibling
`logs/defer-wait-model-perf-20261108/` directory as the `v6` experiment.

### Existing owner-ready queue diagnostic

A Host-only diagnostic selected the existing owner-ready queue through the
published summary, retaining the full v8 async implementation and its AICore
binary. On one allocated device, ten L1 rounds measured CSA at 2.577 ms with
the current pool and 2.599 ms with the owner queue; SWA measured 1.016 ms with
either queue. Original goldens passed in every round. Historical same-device
values were 2.255 ms and 0.893 ms, respectively. CPU compilation ran concurrently
on separate CPU cores, so these are diagnostic results, not final qualification.

Five selected A5 test items passed, including two L1 async cases, task shapes,
forward progress and shared-channel SDMA. The queue switch did not establish a
benefit and is not retained. Evidence is in the same sibling log directory under
`queue-diag-v8-*` and `queue-owner-v8-probe`.

### Historical error-reporter function boundary

Restoring only the historical `static noinline` declaration on the canonical
error reporter took 2185.41 seconds to build, versus 1814.25 seconds for the
preceding v8 release. The ordinary L1 dispatch-loop symbol grew from 398,536 to
687,380 bytes. Ten paired L1 rounds measured CSA at 2.269 versus 2.622 ms and
SWA at 0.877 versus 1.022 ms for the historical baseline and this candidate.
Compiler processes were suspended during this measurement. All original
goldens and ten relevant C++ targets passed; Qwen and the full A5 acceptance
suite were not rerun for this candidate.

This declaration change established neither a build-time nor a runtime benefit
and is reverted. Error contents and failure behavior are unchanged. The larger
ordinary specialization is an observed compiler result; its cause has not been
proved. The frozen source, commands and raw evidence remain under `v9-*` in the
same sibling log directory.

Compiler stack samples during that build showed `GVNHoist` inside CCEC's legacy
pass manager. An isolated follow-up added `-mllvm -enable-gvn-hoist=false` and
time tracing, but a later sample still showed the same pass running. The switch
did not bypass the observed slow phase in this CANN 9.3.0 compiler. The experiment
was stopped before completion; no successful build-time or runtime result is
claimed, and the flag is not adopted. Exact commands and samples are retained
under `v9-no-hoist`.

### Fixed-stride condition binding

A candidate validated the fixed 64-entry allocation at mount and derived token
addresses from the group index, avoiding the 40-byte identity read during bind.
Nonempty sealing still validated identity before publishing CHECK. It passed
10 relevant C++ targets, 49 selected A5 items and 56 protocol capture audits.

Three-arm, ten-round L1 comparisons used the same device, frozen compute binaries
and original goldens. CSA medians were 2.261 ms for the historical baseline,
2.596 ms for v8 and 2.579 ms for the candidate. SWA measured 0.873, 1.013 and
1.010 ms. The v8 and candidate arms used the same O3 outlined profile. Qwen and
the candidate's production inline profile were not rerun. The sub-1% gains did
not justify added fixed-layout coupling and moving record-identity rejection
past kernel entry, so this change is reverted. Evidence remains under `v10-*`
in the same sibling log directory.

### Per-core last-task map cache

An isolated candidate cached the last validated immutable task map in each
core-local view (32 additional bytes), retaining first-use validation and every
execution-group identity check. Two existing C++ targets passed. On one locked
A5 device, ten L1 rounds using the same frozen compute binaries and original
goldens measured CSA at 2.554 ms for the v12 outlined control and 2.542 ms for
the candidate (0.47% lower); SWA measured 0.994 and 0.995 ms (0.11% higher).
All 60 baseline/control/candidate captures preserved execution identities and
passed the existing reader. Qwen and the full A5 acceptance selection were not
rerun because the initial comparison did not establish a useful consistent
Device-wall benefit. The cache is not retained.

Evidence is under `logs/defer-wait-model-perf-20261108/v13b-*`. An earlier private
`v13` binary was discarded before hardware measurement: fallback formatter
settings had reordered includes. The measured `v13b` uses the repository's
explicit formatting configuration and changes only the intended cache logic.

### Last-task async completion decision boundary

An isolated v15 candidate kept the partial-block guard inline and moved only
the existing TASK_LAST map/join decision into a static noinline helper. It
retained every read, validation and failure path. Two existing C++ targets
passed; device compilation took 669.01 seconds on the shared host.

Ten same-device L1 rounds measured CSA at 2.510 ms for the v14 outlined
control and 2.499 ms for the candidate; SWA measured 0.984 ms for both
(0.05% lower for the candidate). Original goldens passed. The small changes
do not justify the additional function boundary, so it is not retained.
Qwen and the full A5 acceptance selection were not rerun for this candidate.
Evidence remains in `logs/defer-wait-model-perf-20261108/v15-*`.

### Historical single-fanin completion shortcut

A private v16 restored the old single-fanin shortcut in the shared completion
SELECT phase. An existing CPU work-budget test failed before the change
(200 predecessor observations versus an expected 128) and passed afterwards;
four related C++ targets passed. Ten same-device L1 rounds measured CSA at
2.517 ms for the v14 outlined control and 2.518 ms for the candidate, and SWA
at 0.986 ms for both. Original goldens passed; this candidate is not retained
and did not trigger another full A5 acceptance sweep or release rebuild.

Existing dependency export explains its limited coverage: among 81 exported
CSA tasks (79 with kernels and two dependency-only nodes), only two
single-dependency tasks have a kernel task as their predecessor; among 24
SWA kernel tasks, none do.
Most exported incoming sets contain multiple predecessors, including creator
nodes. These are deduplicated exported edges, not measured wake-list visits.
One L1 dependency capture per model preserved the original outputs and execution
identities. Evidence remains under `v16-*` and `graph-census*` in the same log
directory.

A private v17 follow-up reused the completion caller's known completed producer
inside the general resolver, retaining every other predecessor check. Four
related C++ targets passed; the synthetic work-budget test reduced predecessor
observations from 200 to 127. The real waiter metadata retains the current
fanin index, so this does remove a possible repeated observation; the test
count does not establish how often these models take that path.

Ten same-device L1 rounds measured CSA at 2.510548 ms for the v14 control and
2.506416 ms for v17 (0.16% lower), and SWA at 0.977093 and 0.978541 ms
(0.15% higher). Original goldens and all 60 capture identities/reader checks
passed. There was no consistent useful Device-wall improvement, so v17 is not
retained. No full A5 sweep, Qwen run or production release rebuild was performed
for this candidate. Evidence remains under `v17-*` in the same log directory.

### Moving bind and return function boundaries

A private v18 changed four declarations: bind and return execution wrappers
became static noinline functions, and their existing runtime helpers became
inline. Reads, checks and shared layouts were unchanged. Two related C++
targets passed; the outlined O3 device build took 670.48 seconds.

Two ten-round same-device L1 comparisons reversed arm order. CSA improved
from 2.512283 to 2.485515 ms and from 2.505755 to 2.476143 ms (1.07% and
1.18%). SWA improved from 0.973813 to 0.961319 ms in the first comparison,
but regressed from 0.970784 to 0.982441 ms in the repeat (1.20% higher).
All original goldens and 120 capture identity/reader checks passed. The mixed
SWA result does not support retaining the boundary change; it remains private.
No Qwen run, full A5 sweep or production inline build was performed for v18.

Existing L1 captures also show the dominant repeated CSA MIX task's same-core
kernel-to-kernel gaps increased by about 3.61 us on AIC and 3.84 us on AIV
against the historical baseline. These are consecutive executions of the same
task/subtask, not an additive Device-wall breakdown. Evidence is under `v18-*`
and `repeated-task-gap-summary.json` in the same log directory. Further
diagnostics target binding and empty-return subphases instead of more small
function-boundary candidates.
