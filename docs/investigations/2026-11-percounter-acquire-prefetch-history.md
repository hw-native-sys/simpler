# Historical percounter acquisition experiments

These measurements describe superseded implementations. References
to a selected candidate in this record refer to those earlier binaries.
Device-0-inclusive results are excluded from acceptance. See the
[current investigation](2026-11-percounter-acquire-prefetch.md) for the merged
vector implementation and its measured validation scope.

## Three-list reference before the merged design

`b71-selected-unified-x1-p1` applies one held-block-per-task rule to ordinary
and sync SPMD. After retirement, both can claim again; sync entry implies all
valid blocks have already been claimed. Single-block intervals retain sharded
batch tickets. SPMD counters are shared across all eligible owners, avoiding
static cross-task shard fragmentation without restricting distinct-task claims.

Same-device main measurements are frozen and reused. The interrupted repeat
matrix is not resumed: only missing or changed candidate evidence is collected.
The current complete timing samples cover all eighteen cases in
`outputs/percounter-acquire-20261114/verification/selected-final/report.json`.
SWA regresses by 1.31% (lib) and 1.41% (DSL); DSL C2A full regresses by 3.35%.
Engram improves by about 3%, below the required meaningful-gain threshold of
5%. Candidate-only repeated communication samples establish larger reductions:
DSL MoE 17.43%-18.15%, lib layers 43.89%-43.97%, and DSL layers 20.44%-20.95%,
each against its reused same-device baseline. Lib MoE's passing retry measures
1.897951 ms against 2.978871 ms (-36.29%); its earlier failure remains open.
`candidate-repeatability.json` retains both candidate means for each repeat.
Passing retries do not close the fault investigation.
Every per-case mean, p50/p95 and per-rank sample remains in that report.

The same-device selected L1 traces localize the lib SWA delay: cache-stage end
moves earlier by 9.33 us, but attention starts 4.14 us later; the interval
between those task envelopes grows by 13.47 us. Attention's own envelope is
nearly unchanged (189.48 to 189.61 us). DSL attention instead starts earlier
but its envelope grows from 170.02 to 201.81 us. These are task-envelope
observations, not proof of pure scheduler overhead or the cause of kernel
slowdown. Per-round evidence is `selected-final/task-attribution.json`.

Lib C1A reindex has passing timing samples but also repeated FIXPIPE error 169
in `_proj_q_a`. Lib MoE also failed in that kernel during prefix execution; its later full
retry passed.
Neither is accepted as functionally correct. The first hardware fault precedes
the supervisor's no-progress timeout; that timeout alone does not establish a
scheduler deadlock. A related earlier unified build reproduces the same fault.
The acceptance workload and business kernel are unchanged; `_proj_q_a` has
code hash `0x80039b66fe6635d3`. Separate instruction-level diagnostic copies are
excluded from every acceptance comparison. A single cold-start functional
control on pristine main reproduces the same
kernel hash, PC and error on device 6. A supervisor snapshot of the failing
candidate shows all twenty unique block indices, block count twenty and scalar
rows value 72 in GM. These observations narrow the fault but do not prove what
the kernel actually read. Root cause remains open.

Task-local sharding, closed-shard summaries, metadata caches and local token
caches did not establish a repeatable 5% wall reduction and are not selected.
Lower FAA time alone did not make task-local sharding beneficial. Earlier
unified-X1 communication measurements remain pilot evidence, separate from
the current selected-binary results. Their L1 traces show no overlap between
different sync tasks; concurrent sync-task kernels do not explain those gains.

## Historical baseline and measurement (not current acceptance)

`cc9-verified-baseline` was built in a pristine source archive using its own
project-local venv. Its audit compares 871 source blobs with the pinned Git
commit and checks the actual CMake source directories. Runtime arms retain
source patches and binary hashes. An earlier arm named `cc9-baseline` was built
from the modified workspace because an editable import finder overrode a
manually inserted source path. That arm and all comparisons using it are
invalidated; they are not inputs to the tables below. The build helper now
asserts the imported source root before compiling.

The frozen matrix contains nine cases in each of `pypto_lib` and
`pypto_lib_dsl`, including MoE and layers 19-21. The historical genuine main and
GO-idle candidate completed all eighteen in that sweep. Original sync flags are retained;
newer half/all-sync exports are a different workload. Historical comparisons
against `56ddac9e8` and `3bcaf8d4f` remain in raw evidence but do not validate
performance against current main.

Each case uses two warmups and ten measured L1 rounds per arm. For each round,
compute the earliest business-kernel start through latest end separately on
each rank, then take the maximum rank duration. Do not subtract clocks from
different devices, sum kernel durations, or substitute Host/Device wall.
MoE and layers use two cards; the other cases use one. The four configurations
are bracketed by genuine-main runs on the same physical device or pair, with
rotated candidate order and one concrete case/arm per allocation.

All 108 captures pass golden and capture audits: runtime, ranks, expected
execution sets, task shapes, lane roles, record uniqueness, clock frequency,
and measured rounds. Raw samples, p50/p95 and baseline drift are retained in
`outputs/percounter-acquire-20261114/verification/go-idle-config-sweep/`.
This sweep predates the Graph counter atomic-initialization review fix and
later fast-launch/owner-distribution experiments; final selection requires
matching source and binary verification.

## Historical GO-idle configuration sweep (superseded claim policy)

Times are milliseconds. The X=2/P=1 candidate has ten measured samples per case;
the bracketing baseline has twenty. Negative changes are faster. Small changes
are not established gains; in particular, the lib c2a_reuse baseline drifted
by 3.39% between its two controls.

| Flavor | Case | Main mean | X=2/P=1 mean | X=2/P=1 p50 | X=2/P=1 p95 | Change |
| ------ | ---- | --------- | ------------ | ----------- | ----------- | ------ |
| pypto_lib | c1a_full | 0.983052 | 0.997194 | 0.993356 | 1.030566 | +1.44% |
| pypto_lib | c1a_reindex | 1.024830 | 1.018483 | 1.017456 | 1.024756 | -0.62% |
| pypto_lib | c1a_reuse | 0.867937 | 0.870772 | 0.870980 | 0.875852 | +0.33% |
| pypto_lib | c2a_full | 1.099283 | 1.076098 | 1.075418 | 1.087652 | -2.11% |
| pypto_lib | c2a_reuse | 0.862198 | 0.881215 | 0.880353 | 0.888563 | +2.21% |
| pypto_lib | engram | 0.953700 | 0.898303 | 0.897601 | 0.903529 | -5.81% |
| pypto_lib | layers_19_21 | 15.879818 | 8.836440 | 8.783632 | 9.079675 | -44.35% |
| pypto_lib | moe | 2.979941 | 2.090643 | 2.071192 | 2.217397 | -29.84% |
| pypto_lib | swa | 0.486386 | 0.498464 | 0.498318 | 0.505574 | +2.48% |
| pypto_lib_dsl | c1a_full | 1.472849 | 1.574268 | 1.573216 | 1.613209 | +6.89% |
| pypto_lib_dsl | c1a_reindex | 1.503384 | 1.445014 | 1.429519 | 1.489985 | -3.88% |
| pypto_lib_dsl | c1a_reuse | 1.417432 | 1.411386 | 1.412701 | 1.433165 | -0.43% |
| pypto_lib_dsl | c2a_full | 1.454516 | 1.562359 | 1.569392 | 1.602272 | +7.41% |
| pypto_lib_dsl | c2a_reuse | 1.388213 | 1.388972 | 1.389771 | 1.418737 | +0.05% |
| pypto_lib_dsl | engram | 0.953560 | 0.901917 | 0.902249 | 0.905294 | -5.42% |
| pypto_lib_dsl | layers_19_21 | 10.745095 | 8.476549 | 8.490873 | 8.537158 | -21.11% |
| pypto_lib_dsl | moe | 1.670715 | 1.408897 | 1.414099 | 1.466579 | -15.67% |
| pypto_lib_dsl | swa | 0.999537 | 1.007540 | 0.998114 | 1.064082 | +0.80% |

The earlier independent true-baseline pilot also found roughly 5%-6% engram
improvement, 31% lib MoE improvement and 44% lib layers improvement. The sweep
repeats those gains. SWA and several full-attention cases still regress;
large communication-case improvements do not erase those regressions.

All four configurations below use the same algorithm. Values are percentage
changes from the two bracketing main controls; configurations must not be
selected separately for each case merely to report the minimum.

| Flavor | Case | X=2/P=1 | X2/P2 | X4/P1 | X=4/P=2 |
| ------ | ---- | ------- | ----- | ----- | ------- |
| pypto_lib | c1a_full | +1.44% | +2.46% | +1.49% | +6.85% |
| pypto_lib | c1a_reindex | -0.62% | -0.39% | -0.23% | -0.39% |
| pypto_lib | c1a_reuse | +0.33% | +0.63% | +0.80% | +0.97% |
| pypto_lib | c2a_full | -2.11% | +0.97% | -2.09% | +1.68% |
| pypto_lib | c2a_reuse | +2.21% | -1.60% | +2.29% | -0.93% |
| pypto_lib | engram | -5.81% | -5.39% | -5.97% | -5.30% |
| pypto_lib | layers_19_21 | -44.35% | -43.92% | -43.32% | -43.11% |
| pypto_lib | moe | -29.84% | -30.46% | -26.87% | -30.42% |
| pypto_lib | swa | +2.48% | +2.50% | +2.35% | +3.15% |
| pypto_lib_dsl | c1a_full | +6.89% | +6.67% | +4.14% | -0.88% |
| pypto_lib_dsl | c1a_reindex | -3.88% | -1.77% | -3.37% | -1.03% |
| pypto_lib_dsl | c1a_reuse | -0.43% | -1.69% | -2.89% | -1.60% |
| pypto_lib_dsl | c2a_full | +7.41% | +7.76% | +2.01% | +3.99% |
| pypto_lib_dsl | c2a_reuse | +0.05% | +6.57% | -1.15% | -0.41% |
| pypto_lib_dsl | engram | -5.42% | -5.33% | -5.58% | -5.53% |
| pypto_lib_dsl | layers_19_21 | -21.11% | -21.03% | -19.84% | -20.21% |
| pypto_lib_dsl | moe | -15.67% | -16.98% | -18.08% | -15.61% |
| pypto_lib_dsl | swa | +0.80% | +1.27% | +3.19% | +1.41% |

Larger windows or a second payload do not win consistently. A second payload
preserves a prepared cohort while ordinary work uses the other buffer, but
selection, identity checks, writeback and GM storage remain costs. The CPU
payload test checks independent identities; it does not establish overlap of
scalar scheduling and a kernel on the same lane.

## Historical SWA attribution and rejected shortcuts

In the separately paired `go-fast-pilot` captures, lib SWA's direct producer
kernel-end to attention-first-start interval falls from about 7.7 us on main
to 4.9 us with GO-idle. Its upstream stage ends later and attention itself lasts
longer, so the complete window still regresses. For DSL SWA, the interval falls
from about 21.3 us to 8.5 us, while mean attention kernel duration rises from
about 143.9 us to 177.0 us. Sparse cluster placement also changes. This is a
correlation to investigate, not proof of a memory-contention cause.

These intervals exclude runtime completion publication and untraced
dependency-only tasks; they must not be labeled pure scheduler overhead.
Per-task/per-round data are under `verification/swa-stage-analysis/`, beneath
the same output root. Genuine-main and GO-idle representative PNG/SVG/Perfetto
traces are under `swa-l1-cc9/pypto_lib/`, with median-round selection and source
provenance recorded beside each image.

An inline fast-launch trial bypasses repeated candidate scans after own idle.
It passes the pilot's functional checks but makes both SWA flavors about 1.7%
slower than its matched parent. It is not selected. Static generated instruction
count rises from 54,490 to 62,520; these are code-size diagnostics, not dynamic
execution counts. IR confirms the dispatch helper is inlined, so outlined-call
overhead cannot explain that trial. The explicitly outlined and preset-owner trials are historical experiments;
the preset-owner eligibility rule is rejected by the acquisition contract.

Reducing the ordinary ownership window to one after a long kernel is unsafe:
a lane can hold a later-shard consumer while an earlier-shard producer remains
unclaimed. Reducing the batch is safe; removing the available slot is not.
The source-specific reserved positions remain available under saturation.
Stealing before home exhaustion and unconditional large initial batches can
concentrate SPMD waves on too few cores. The current path uses home-first claims, same-task uniqueness while a block
is held, and immutable/fanin/payload caches. Distinct-task acquisition uses
free allowed slots without a readiness or idle-pass gate.
Readiness never determines whether a task is eligible for its initial claim.

Failed dependency probes use a bounded one-in-four cadence after sixteen idle
passes. Entry, completion, publication and async progress remain serviced each
pass. A one-in-sixteen cadence regressed SWA by about 14% and is rejected.
GM coherent reads, reader fences and SSBUF release/acquire boundaries remain;
static instruction reduction alone does not justify deleting them.

## Merged-order optimization trials

These lib SWA pilots use device 6, two warmups and ten measured L1 rounds against
the same 0.4862898 ms main baseline. They are predecessor-arm observations, not
current acceptance or independent-repeat evidence.

| Mechanism/configuration | Mean ms | Change |
| ----------------------- | ------: | -----: |
| Initial merged order X=2/P=1 | 0.5248253 | +7.92% |
| Monotone fresh cursor plus retirement revisit X=2/P=1 | 0.5274623 | +8.47% |
| Compact segment shape X=2/P=1 | 0.5124850 | +5.39% |
| Closure summaries X=2/P=1 | 0.5091732 | +4.71% |
| Return after failed draw when holding work X=2/P=1 | 0.5051656 | +3.88% |
| Same mechanism X=1/P=1 | 0.4854653 | -0.17% |
| Same mechanism X=1/P=2 | 0.4905496 | +0.88% |
| Same mechanism X=4/P=2 | 0.5141242 | +5.72% |
| Host/AICPU argument metadata X=1/P=1 | 0.4857608 | -0.11% |

The X=1/P=1 one-draw predecessor completed all 18 case/flavor checks on permitted
devices. Its MoE and layers gains did not establish ordinary SWA improvement.
Increasing capacity or adding a second payload did not produce a stable benefit;
X is the held-task capacity and P is the count of reusable prepared payloads,
not a per-workload tuning parameter. Subsequent preparation and publication-cursor
refinements retain one global configuration. The current investigation records
the matched current-source results and their remaining scope limits.

The gains persist across capacities in the same one-draw implementation,
so selecting X=1 is not the origin of the large gain against main:

| Configuration | lib MoE change | lib layers change |
| ------------- | -------------: | ----------------: |
| X=1/P=1 | -39.72% | -47.22% |
| X=2/P=1 | -38.58% | -46.40% |
| X=1/P=2 | -39.99% | -47.19% |
| X=4/P=2 | -36.09% | -45.34% |

## General-path follow-up candidates

After the rollback checkpoint `afa9206a8`, three candidates were measured on
device 6 with two warmups and ten L1 rounds per SWA flavor. The original main
means remain 486.2898 us (lib) and 987.8475 us (DSL); main was not remeasured.

| Candidate | lib mean us | Change | DSL mean us | Change |
| --------- | ----------: | -----: | ----------: | -----: |
| Local unresolved/waiting/ready sets | 486.1128 | -0.04% | 980.7364 | -0.72% |
| Sets plus prepared full-window waiting | 485.3576 | -0.19% | 974.3555 | -1.37% |
| Host virtual ready/resource ordering | 485.0583 | -0.25% | 986.0333 | -0.18% |

None reached the required repeatable 5% ordinary-scheduling improvement.
The two device-loop experiments were reverted; frozen sources and captures
remain under the task output. The Host ordering candidate uses the same
AICore allocated sections and relocations as the rollback checkpoint and adds
no upload bytes. It moves an independent branch roughly 400 us earlier in lib
SWA without substantially shortening the attention/output chain. Early task
placement alone therefore does not establish an end-to-end gain.

The state-set and waiting candidates passed CPU, TSan, full simulation and
device smoke checks. A negative control that skips event service before the
prepared-window return fails the new MIX-notification test. These checks prove
that specific progress property, not a performance benefit. Complete absolute
mean/p50/p95 pilot data are in `verification/{state-sets,event-loop,resource-order}-pilot.json`
under `outputs/percounter-acquire-20261114/`.

After the four-sector completion and claim implementation (`def2de899`),
the following lib SWA feasibility candidates did not establish a
further improvement. Each retained the business kernels and L1 boundaries,
with two warmups and ten measured rounds. These are pilot means, not matched
effect estimates; all changes were reverted.

| Candidate | Device | Mean us |
| --------- | -----: | ------: |
| Omit explicit Host-inline-completed fanin edges | 1 | 447.548 |
| Point partitioned fanin references directly at component groups | 2 | 448.122 |
| Four device-initialized argument-metadata replicas per task | 1 | 449.862 |
| Identify pooled payloads only by task/block entry | 1 | 447.312 |
| Align the payload array to independent 128-byte sectors | 1 | 450.369 |
| Eight claim sectors for every task | 1 | 452.730 |
| Four claim sectors for narrow tasks, 32 for wide tasks | 1 | 465.096 |
| Width-aware groups with issue-only closure bits and idempotent summary | 1 | 464.682 |
| Closure bits also skip exhausted groups during acquisition | 1 | 462.448 |

Payload count had been independent from candidate capacity to reuse argument
buffers across candidates. Earlier P=1/2 experiments did not compare this pool
against fixed slot binding. A subsequent fixed-binding prototype mapped AIV0
slot j directly to payload j; AIC/AIV1 used separate buffers for their X local
positions and X shared-record positions. Payload initialization moved to each
AICore before claims, reducing Host upload bytes. It removed pool lookup and
replacement while retaining publication, entry and retirement protocols.

The device-1 independent comparison measured 447.733/447.591/454.251 us
(mean/p50/p95) for the committed X1/P1 pool and
449.387/447.042/462.613 us for fixed binding at X=1. Fixed binding at X=2
measured 463.792/465.742/477.182 us. Replacing its parameter identity tags
with slot-lifetime valid bits at X=1 measured 452.826/453.555/461.300 us.
These prototypes were also reverted; the data do not establish a universal
advantage for pooling. Captures, configurations and complete statistics are
indexed by `verification/slot-bound-payload.json` under the same output root.
These feasibility candidates received no CPU/simulation acceptance claim.

A separate X=1 capacity-only comparison retained the original pool selection,
protection and preparation, with AIC/AIV0/AIV1 capacities 2/1/2. It preserved
the primary uploaded payloads and added only device-initialized second buffers.
The device-1 ABBA means were 448.621 us for the committed pool and 447.684 us
for 2/1/2: 0.21% overall, with pair directions +0.58% and -0.16%.
The 444.662 us pilot did not repeat. A complete 18-case comparison passed all
36 runs: nine candidate means improved and nine regressed. The 16 matched-device
pairs had a 0.41% geometric mean regression; lib c2a_reuse regressed 3.47%.
This capacity change was reverted. Complete data are indexed by
`verification/payload-capacity-212.json` and `verification/payload-capacity-212-full/`
under the same output root.

Transposing completion storage to `components[sector][task]` preserves the
allocation size, uploaded bytes and logical component membership. A device-1
ABBA measured control means 451.231/449.192 us and transposed means
447.896/450.595 us: 450.212 versus 449.246 us overall, a 0.21% observed mean
improvement with opposite pair directions. The retained layout is task-major:
fanin observes one task's components together and fixed component strides avoid
task-count-dependent addressing. The transposed trial was reverted; its full
mean/p50/p95 data are in `verification/sector-major-abba.json`.

Additional admission/dispatch stop checks were evaluated on base `42e0d7b97`
against the frozen PR source `cc2fea4da`. Repeated shared-GM error checks raised
latest C2A L3 bypass loads from 27,205 to 30,236; restricting the checkpoints
reduced them to 28,078 but did not remove the L0 regression. These diagnostic
counts are not elapsed-time savings. A register-only boundary variant completed
all twelve RESULTS configurations: geometric mean latency rose 0.14%, with
individual changes from -1.51% to +1.24%. Independent same-device ABBA measured
DSL half-sync C2A at -0.10% and all-sync64 C2A at +0.76%.
Moving the register check to every loop iteration also failed to remove the
cost: a separate device-2 all-sync64 ABBA measured 1500.656 versus 1518.206 us
(+1.17%). Its four runs are retained in `confirmation-v5.json` alongside the
other review experiments.

The extra hot-path checks were not retained. Periodic EXIT/error observation
and platform timeout recovery remain the failure contract; they do not promise
immediate admission stop or graceful retirement of an incomplete cohort.
The retained runtime binaries match the twelve measured control binaries
byte-for-byte. Full trial data and the identity proof are under
`outputs/percounter-acquire-20261114/ci-feedback-3114/`, including
`report-v4.json`, `confirmation-v4.json`, `report-final.json` and
`final-runtime-identity.json`. The final report contains actual control samples,
not an invented second measurement of the identical binary.
