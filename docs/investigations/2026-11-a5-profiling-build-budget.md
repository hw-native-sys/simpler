# A5 HBG profiling and build budgets

The targets are less than 1% L0 profiling overhead, a complete cold A5 HBG
runtime build below 300 seconds, and L3 AICore business-kernel window below
1.3 times L0 without removing the existing trace detail. AICPU trace collection
is outside this L3 acceptance window. The build target is achieved by the retained
compiler boundaries below. The complete runtime targets remain open.

## Retained compilation boundaries

Keep cooperative filling, own-slot filling and execution-token refill as
separate, internally linked functions. Their primitive operations remain inline.
This bounds optimization of the large bootstrap and steady-state callers.
MIX pending progress and service keep their idle checks inline and call a
shared body only when work exists. Dispatch preparation and publication also
have separate call frames. This removes repeated large bodies from the callers.
Enabled Scalar scope constructors and close remain inline; primitive observers
retain their call boundaries. Scope types have internal linkage so AIC and AIV
objects cannot share an observer binding accidentally. Disabled scopes are empty.

A fresh complete A5 HBG build took **49.662 seconds** with the normal `-O3 -g`,
ccache disabled and a build parallelism limit of 16. The source, configuration,
logs and binary hashes are retained separately from the active checkout. The A5
IR scanner found no escaping high-alignment stack objects in all 16/16 device
translation units. IR review time is separate from compilation time.

Merely outlining the enabled Scalar scopes took 469.800 seconds and did not
meet the build target. A single profiled AIV unit with line-table-only debug
information still took 323.161 seconds; the analogous `-O2` probe exceeded
300 seconds and was stopped. Neither compiler-flag experiment is retained.

## Runtime checks

The current PyPTO-lib cases reuse their business binaries and inputs. L0
regression checks compare the fixed fwic-graph baseline with the candidate,
both using DFX-enabled runtime and orchestration ABIs. Each automatically
allocated job runs baseline, candidate, candidate, baseline; every arm uses
one warmup and ten measurements. The table averages the two batches per arm.

| Case | Device | Baseline L0 Device, ms | Candidate L0 Device, ms | Change |
| ---- | ------ | ---------------------: | ----------------------: | -----: |
| SWA | 0 | 0.507833 | 0.507168 | -0.13% |
| C2A full | 1 | 0.685914 | 0.654949 | -4.51% |

These are two-case results, not a full-matrix acceptance claim. C2A baseline
batch means vary from 0.723348 to 0.648479 ms; its aggregate speedup should not
be read as a stable 4.51% effect.

The L3 comparison uses each candidate's own L0 business-kernel window. These
are separate jobs from the regression table, with two ten-measurement L0
batches around one ten-measurement L3 batch and one warmup per arm:

| Case | Device | L0 AICore, ms | L3 AICore, ms | Increase |
| ---- | ------ | ------------: | ------------: | -------: |
| SWA | 0 | 0.369286 | 0.504449 | 36.60% |
| C2A full | 7 | 0.489881 | 0.695043 | 41.88% |

Both cases pass numerical validation and have no dropped or truncated Scalar
records. Each L3 capture contains 96 distinct worker streams; SWA has all 578
task executions. Every measured aggregate endpoint exactly matches the first
begin and last end of executed task records. The semantic-function threshold
remains 1000 cycles. The **30% L3 target remains unmet**.

Earlier comparisons between `SIMPLER_DFX=0` and `SIMPLER_DFX=1` runtimes are
**invalid** because the same orchestration binary was reused. `CoreTaskArgs`
has a conditional `DumpArgSelection` member: its sizes are 648 and 808 bytes,
respectively, and it crosses the orchestration/runtime interface by reference.
Correct output alone does not validate that mixed-ABI comparison. A complete
DFX-disabled comparison requires matching orchestration builds. The invalid
measurements are retained as rejected evidence and are not acceptance results.

The historical [ordered-fill outlining experiment](../zh-cn/design/a5-hbg-graph-task/rejected-optimizations.md#ordered-fill-outline)
failed a synchronous dual-AIV diagnostic. The baseline and initial compilation
boundaries were compared on the same automatically allocated device and both
passed `test_cohort_blocks_land_on_distinct_clusters`. The retained candidate
also passed on automatically allocated device 7, including exact outputs,
distinct core placement and simultaneous co-residency of all 16 subtasks.
The two small business kernels were compiled before device allocation and
reused throughout.

Scalar recorder, header, execution-ownership and GM-MIX CPU checks passed.
The historical compilation-boundary report at `00986257e` (rebased as
`0e1199a77`) recorded 144 of 147 ordered-execution cases passing. Its three
failures were also reproduced with the unchanged baseline headers and test source:

- `PreparedOrdinaryTokenYieldsItsSlotAndPreservesPredicateDecision`;
- both shapes of `ACooperativeCandidatePublishesWhileAnOrdinaryTokenIsHeld`.

Those baseline assertions were not changed or counted as successful checks in
that investigation. The later final snapshot `8717a12fb` updates the ordered
fixtures to match the retained service boundaries and passes **147/147** cases.
Its evidence is `artifacts/ordered-contract-tests-20261115/test-final.log`; the
144/147 result is historical, not the final snapshot's validation status.

Local evidence is under `artifacts/dfx-budget/`: `compiler/`,
`dispatch-boundaries/`, `dispatch-boundaries-baseline-l0/`,
`dispatch-boundaries-accept/` (candidate rows only), `cohort-final/`, `base-cpu/`,
the focused CPU logs,
and the separate earlier scope/compiler-flag probes. Hardware records retain
the actual automatically allocated device IDs.

## L0 business-kernel endpoints

HBG records each participating core's first business-kernel begin and latest
business-kernel end in local state. The end clock is read after each executed
kernel, because the scheduler cannot know in advance which will be last.
There is no shared atomic or printing in this path. After scheduling, each core
flushes its pair through the existing 64-byte handshake before retirement.
The AICPU reduces the retired reports and the host emits one `aicore_wall` span
with both raw endpoints and the AICore clock frequency.

The reduction excludes cores with two zero endpoints, rejects incomplete
pairs, and emits no span for a wholly idle run. Percounter uses the same
business-kernel endpoints and idle-core reduction. Historical percounter
scheduler-loop windows include startup and retirement work and cannot be
compared directly to these endpoints.

The paired SWA check used automatically allocated device 0, existing business
binaries, one warmup and ten measurements per arm:

| Runtime | L0 Device wall, ms | L0 AICore, ms | L3 AICore, ms |
| ------- | -----------------: | ------------: | ------------: |
| Compilation boundaries | 0.508885 | unavailable | unavailable |
| With kernel endpoints | 0.511827 | 0.370503 | 0.524688 |

The added endpoint accounting increased L0 Device wall by **0.58%** in this
paired check. This is endpoint overhead, not proof that total DFX overhead is
below 1% across the case matrix. L3 AICore inflation is **41.62%**, so the
30% target remains unmet. In every measured L3 capture, the aggregate raw
endpoints exactly match the earliest begin and latest end of executed task
records. All 578 SWA executions remain present with no dropped Scalar records.

The endpoint reduction CPU test and 16/16 A5 IR checks pass. Evidence is in
`window-final-window/latest__swa/`. The endpoint runtime used an incremental
repair of a failed build, so its build duration is not presented as a fresh
cold-build result.

The endpoint-only runtime also passes real two-device native MoE at L0 using
its existing callable and all 27 kernel binaries: dynamic counts `(0, 6)`,
`(6, 0)`, and `(72, 72)` all pass the three golden outputs, including ten
measured rounds of the last shape. The actual devices were 1 and 2. Evidence
is in `native-endpoint-check/`; the original host `get_tensor_data` remains.

## Rejected recording changes

A 32-slot local write buffer increased SWA AICore inflation to 54.20%. Inlining
primitive reads increased it to 49.11% and grew one profiled AIV unit's code
from 757,808 to 1,034,872 bytes. Inlining all primitives exceeded the 300-second
build budget and was stopped. These changes are not retained. Compiling only
the profiled resident units at `-O2` also failed to improve the L3 ratio; the
retained code uses the normal `-O3 -g` configuration.

Reducing trace prefetch from sixteen cache lines to four also failed to improve
the ratio: SWA AICore inflation was 37.42%. The original lookahead is retained.

A diagnostic that removes the bypass-load result dependency before its end
clock measures issue time instead of completion time. Its SWA AICore window
still increases by 35.35% (0.368735 to 0.499074 ms). It does not meet the budget,
does not establish that this dependency is the main remaining cost, and changes
the trace contract. The diagnostic is not retained or counted as acceptance.
