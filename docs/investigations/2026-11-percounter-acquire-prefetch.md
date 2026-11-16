# A5 percounter acquisition: fixed slots and MIX entry

The implementation uses the user-approved merged AIV/MIX order, X generic
AIV0 slots and retirement revisits. The working configuration is X=1 with one
DispatchPayload per lane. It is rebased onto main
`cbe15ec86e559ccca2a02dfb78f228db57ee25ba`, including its kernel-bounded
L0 work window. The measured snapshot is `8de12840d` on `1568c3da1`;
the final rebase changes no percounter source. The historical optimization measurements below use the
earlier `b71ba64fdd3f369613616828c83ece1aea37ee0b` baseline and its L1–L4
DFX integration; they are not measurements against the rebased main.

The rebased implementation preserves the first actual kernel start and last
actual kernel end through the role-specialized lane loop by referencing the
caller's configuration. A lane-entry regression covers the returned endpoints;
the default CPU suite passes 82 tests with six capacity-dependent skips.
On main `1568c3da1`, all 18 original frozen L1 cases pass on both arms.
Lib SWA measures 481.474 us for main and 449.209 us for the candidate on
device 1. Sixteen means improve and two regress. Full statistics are in
`outputs/percounter-acquire-20261114/verification/rebase1568-l1-full/`.

The separate `RESULTS.md` matrix uses its 12 selected configurations and
PyPTO-lib `a6a9a2d49047`, with pure L0, one warmup and ten measured rounds.
All 24 measurements pass; every comparison uses the same device or pair.
All 12 means improve. Their sum is 25,255.456 us for main and 17,919.511 us
for the candidate, a 29.05% reduction in the suite aggregate, not end-to-end
model latency. Latest-lib SWA measures 235.162 versus 185.697 us, a 21.03%
improvement. This is a different workload and timing contract from the old
lib SWA optimization target. Eleven replay cases retain numerical checks;
native MoE runs original Host control, nonzero weights and 72/72 routing
without numerical validation, as specified by the benchmark. Complete
mean/p50/p95, devices and raw evidence are indexed by
`outputs/percounter-acquire-20261114/results1568-l0/report.json`.

The MIX shared head uses task rank before publication sequence. A deterministic
regression publishes a later MIX successor before a retired predecessor slot
revisits its next block. Sequence-only selection stalls at X=2 and X=4;
rank-first selection completes every invocation and preserves fanin. The CPU
matrix covers all four MIX masks, six peer step orders and ordinary or legal
sync successors. X=2 and X=4 tests are registered in CTest. Default A5 builds
include percounter so automatically collected scene tests have their binaries.

A same-device ABBA on the latest-lib SWA, holding P=1 and the cbe15ec86 base,
compares the submitted a0b90c0be implementation with the rank-first correction.
L0 uses one warmup and ten measured rounds per run. Combined before/after
means are 185.720/184.402 us at X=1 on device 5, 184.024/183.594 us at X=2
on device 2, and 191.625/187.891 us at X=4 on device 5. Improvements are
0.71%, 0.23% and 1.95%, below the 5% meaningful-gain threshold. X=1 p95
increases from 191.502 to 194.593 us in this batch. This is correctness-fix
validation, not a claim of another substantial performance improvement.
Evidence is under `outputs/percounter-acquire-20261114/review3114-sort/`.

The selected candidate is `45b-claim-shards4-completion4-x1-p1`, with Host
ready/resource ordering and four completion components partitioned by block
index modulo four. Components occupy independent device-only sectors in
task-major order, `components[task][sector]`;
consumers retain successful component observations and advance cyclically.
The array adds 512 bytes per task, initialized in
parallel by AICore before an epoch barrier. The uploaded layout does not grow.
Task-private block claims use four independent sectors and a closed-sector
count. Their 640-byte records also initialize on AICore, outside the upload
prefix. Cross-sector acquisition covers all blocks without assigning physical
owners at bind. A unique final sector closure publishes each segment summary.
`b71-merged-boundary-links-x1-p1` is the earlier acquisition checkpoint whose
results are retained below.

The four-component device-6 lib SWA pilot measured 451.400 us against
486.290 us on frozen main. Independent device-4 measurements reproduced the
improvement: the two-component implementation measured 461.544 us and four
components measured 451.645 us. Eight components measured 457.869 us on the
same device. All retain X=1/P=1 and one completion contribution per block.
The final source also requests a 128-byte-aligned state allocation. Its
independent device-4 comparison measured 458.658 us for two components and
450.769 us for four; the separate device-1 sample measured 462.267 us.
Four claim sectors then improved the device-1 ABBA comparison: control means
452.562/453.954 us versus candidate means 447.537/444.179 us. The averages are
453.258 and 445.858 us, a 1.63% increment. Two claim sectors measured 450.188 us
in a separate device-1 pilot and did not establish a further improvement.
Increasing X alone regressed: same-device X=1/2/4 measurements were
460.654/481.017/520.015 us. L3 attributes additional cost to control and
preparation; it does not establish P=1 as superior to one payload per slot.

The retained four-component, four-claim-sector implementation passes all
18 frozen DSV4.1 cases, including MoE and layers_19_21 on two devices for each
flavor. Its lib SWA mean/p50/p95 is 450.135/449.978/459.816 us on device 4;
frozen main is 486.290/484.281/499.899 us on device 6, a 7.43% mean reference
improvement. Sixteen case means improve against frozen main. DSL c2a_reuse
and DSL c1a_reuse regress 2.84% and 0.71%, respectively. These cross-device
differences are reference comparisons, not matched-device acceptance.
Complete statistics, actual devices and captures are indexed by
`outputs/percounter-acquire-20261114/verification/final-retained/`.
The preceding two-component array measurements remain in
`verification/paired-array-full/` and `verification/paired-array-independent.json`
under the same output root. The claim-partitioned SWA captures are under
the candidate-named pilot directory and
`claim-abba-{1,2,3,4}-20261116`; the matched comparison is recorded in
`verification/claim-shards4-abba.json`. X comparisons are recorded in
`verification/array-x-capacities.json` under the same output root.
The continuing ordinary-scheduling target is a repeatable lib SWA L1 AICore
wall of at most 440 us, with the frozen workload and measurement boundary.
This target is not yet met.

For this source, CPU tests passed at X=1 (80 pass, 6 capacity skips), X=2 and
X=4 (86 pass each), with TSan. The completion test leaves each component
unfinished in turn and checks every physical observer's starting position.
The claim test checks unique block coverage and exactly one segment closure
across ragged block counts and different owner counts.
The full simulation suite passes 23 standalone cases and 10 scene classes.
Its mixed-wave stress case exposed a 64-byte state-allocation request despite
128-byte-aligned records; the group-address encoding requires the stronger
alignment. Requesting `alignof(PercounterControl)` fixes both the isolated
reproduction and the complete suite without changing uploaded byte counts.
The predecessor's negative
control that checks only the first completion component fails the
early-consumer test; both-component gating remains covered by the migrated
tests. The A5 IR scan covered
five of six objects without findings; the remaining wrapper has the recorded
tooling limitation. Every preceding two-component matrix case passed on its first attempt.

The [runtime contract](../../src/a5/runtime/percounter/README.md) describes the
implemented protocol. The [historical experiments](2026-11-percounter-acquire-prefetch-history.md)
retain superseded three-list and merged-list candidates. Device 0 and captures
containing it are excluded from acceptance. Fourteen audited main captures on
device 6 and four dual-card captures on devices 2/6 provide the frozen baseline;
its authoritative list is
`outputs/percounter-acquire-20261114/verification/nonzero-devices/measurement-plan.json`.
No baseline was remeasured for the metadata and cursor refinements.

## Capacity and cursor invariants

The Host projects one explicit global task rank into an AIC order and an
AIV/MIX order. AIV0 claims either kind into X generic fixed positions; AIV1
skips MIX claims but participates through SSBUF. MIX position j owns record j.
Receivers query shared records without persistent task-descriptor copies.

Every SPMD task and every MIX task has private claim sectors. A physical
owner holds at most one unretired block of the same task; the MIX owner is the
cluster's AIV0. Successful acquisition advances that owner's fresh-task cursor
immediately, regardless of remaining blocks, fanin or execution. Free positions
can claim subsequent distinct tasks. No pending-wave, ordinary-pending or
partial-MIX-owner restriction applies.

Each fixed slot retains its claim-counter index. Physical retirement marks it
for revisit; retired positions are revisited in rank order before reuse for
later work. A valid remaining block reuses that position. A closed counter
allows the position to continue along the fresh cursor. This avoids repeated
searches for held tasks without losing subsequent SPMD waves.

An owner cannot fill every position solely with later tasks while skipping an
earlier eligible task: it must first claim or already hold that task. Retirement
creates capacity for its remaining blocks. For the minimum-rank unfinished task,
internal DAG predecessors are complete; ordinary waves can retire and width-valid
sync cohorts can acquire their required owners. Progress also requires the
publication/entry protocol, fair scheduling, valid kernels and eventual external
completion. This capacity argument does not prove hardware memory ordering.

Ordinary single-block intervals use one ordered batch counter. A batch is
bounded by free positions and cannot cross a MIX or SPMD boundary. AIV1 never
consumes a skipped MIX ticket. Task-private counters and typed intervals prevent
a stale observation from consuming a different type of work.

Publication cursors remain separate from the fresh cursor. They advance only
when all blocks of the relevant task have been claimed. Each caches its open
boundary's immutable block count, rank and successor; repeated checks read only
its closed-sector count. Stale counter observations delay advancement, while immutable
metadata cannot invalidate the cache. The Host supplies first/next segment indexes
for AIC sync, AIV sync and MIX boundaries. These links use existing control and
task-metadata storage and are available for Graph bodies before materialization.
They remove intervening segment classification without changing acquisition order
or adding uploaded bytes.

## Publication, entry and retirement

AIV0 publishes each local MIX record before its release-published MIX boundary.
Receivers acquire the boundary before discovering records it covers. Discovery
uses one acquired publication-sequence ceiling and checks generations around
shared descriptor reads. A boundary proves allocation/publication progress,
not execution completion.

The lowest active participant resolves fanin and publishes readiness without
waiting for other participants to discover the record. Successful fanin prefixes
are retained. Each participant commits ENTER_READY only after excluding smaller
uncompleted or undiscovered cohorts. Pure ordinary ready work can continue while
that commitment waits. Sync tasks additionally require one start arrival per block.

ENTER_READY does not prove the participant's UB and pipelines are idle. After
all entry conditions hold, each participant returns from its ordinary kernel and
publishes ENTER_IDLE; it starts no more ordinary kernel until this MIX invocation
returns. Every active participant must be idle before MIX starts. This retains
the ordinary producers needed to resolve earlier cohorts without overlapping a
peer's ordinary kernel and MIX resource use.

FINISHED follows the participant's final payload use. The lowest active lane
aggregates the actual subtask count after every FINISHED, then publishes completed.
Only that completion allows AIV0 to release the position and reuse its record.
All tokens are generation-qualified. A nonparticipating AIV0, including mask 0b101,
coordinates without executing or contributing a fictitious completion.

Only actual defer_wait registrations survive physical retirement. Their records
hold completion conditions, not unstarted candidates. Same-card work uses explicit
DAG dependencies; the user-rejected later-notifier case is not counted as passing.
Deferred MIX shares are subtracted before FINISHED and restored after their
conditions hold, independently of shared-record reuse.

## Preparation and instruction costs

The Host resolves ordinary-task callable entries, argument counts and span bases
once per task. AICPU materialization publishes equivalent Graph-body metadata
before metadata_ready. AICore copies values and fills its block context without
reparsing generic payloads or looking up callable tables for each block.
The metadata occupies 44 previously reserved bytes of the 128-byte task counter;
compiled before/after layout probes show zero added upload bytes.

Payload publication writes back the control line, populated argument prefix and
global-context line, then fences. Cached payloads acquire inputs after readiness.
A payload freshly prepared after readiness in the same dispatch step can reuse
that preparation's data acquisition. No cached acquisition crosses iterations.

Compact 24-byte claim segments and monotone closure summaries allow exhausted
ranges to be skipped. Only the final valid claimant publishes a closure bit;
overshoots cannot close a different task. A failed atomic draw returns to servicing
held work; an empty owner can continue through exhausted ranges immediately.
These mechanisms reduce repeated reads and preparation, but instruction-count or
gap reductions alone do not establish an AICore-wall gain.

L3 exposed a 26.10 us acquisition call in the first merged candidate, with 77 bypass
loads spanning QKV predecessor completion. Its L1 launch spread grew from 2.01 us
to 26.73 us. Monotone acquisition/revisit removed repeated held-task walks. The
lib SWA remains effectively flat against main; reducing local preparation and
boundary reads has not established a 5% SWA wall gain. The original optimization
goal remains open. This candidate excludes the subsequent fanin-address cache
experiment and uses X=1/P=1; neither a larger window nor a short-path instruction
reduction alone established an end-to-end benefit.

## Validation and measured scope

Bounded SC models cover 19,200 single-block DAG configurations (6,034,516 states),
1,584 SPMD/sync capacity configurations (17,213,302 states), and 30 explicit
publication/ENTRY/IDLE/FIN configurations (43,132 states). They retain missing
publication, missing entry and missing idle negative controls. The unreduced
state-bound-limited attempt is excluded. Evidence is under
`outputs/percounter-acquire-20261114/analysis/unified-vector-list/`.

The restored source passes 70 CPU contracts at X=2/P=1 and X=4/P=2.
At X=1/P=1, 64 pass and 6 require multiple positions and skip; the same 64 pass
under ThreadSanitizer. Python model/DFX checks pass all 45 tests. The exact
runtime snapshot previously passed 23 standalone simulation cases and 10 scene
classes; restoring its source and verified binary hashes reuses that evidence.
Two additional producer-reuse CPU tests pass against the restored source.
The shared Host seam is unchanged from its five binding CPU checks and six HBG
simulation checks. Full pre-commit checks all 34 existing/new files with the
actual simulation compile databases and correct GCC headers; no checks are disabled.
Source restoration and validation records are `verification/rollback-source-audit.json`
and `verification/rollback-validation.json` under the task output directory.

The current IR scan parses 5/6 device translation units with zero findings.
System llvm-dis crashes on the unchanged AIV SIMT wrapper. Its allocated sections
and non-debug relocations match the previous artifact. This is a scoped scan,
not a fully clean scanner result. Hardware evidence remains independent.

Performance uses two warmups and ten measured L1 rounds per process. Every round
uses the earliest-to-latest business-kernel window on each rank, then its maximum
across ranks; clocks on different devices are never subtracted. The complete
restored-candidate matrix passes all eighteen frozen original lib/DSL cases,
including golden, task shape, execution set, lane-role and record uniqueness checks.
This frozen workload retains original sync flags; it does not cover the subsequently
curated latest-lib/all-sync matrix. The paired main measurements were reused.
Raw per-rank rounds and both sides' mean/p50/p95 are in
`verification/rollback-full/report.json` under the task output directory.

| Flavor | Case | Main mean ms | Candidate mean ms | Change | Main p50 ms | Candidate p50 ms | Main p95 ms | Candidate p95 ms |
| ------ | ---- | ------------ | ----------------- | ------ | ----------- | ---------------- | ----------- | ---------------- |
| pypto_lib | swa | 0.486290 | 0.487478 | +0.24% | 0.484281 | 0.485005 | 0.499899 | 0.500456 |
| pypto_lib_dsl | swa | 0.987847 | 0.998405 | +1.07% | 0.983426 | 0.996145 | 1.012070 | 1.019156 |
| pypto_lib | c2a_full | 1.094188 | 1.109269 | +1.38% | 1.093519 | 1.107942 | 1.100372 | 1.116326 |
| pypto_lib_dsl | c2a_full | 1.494619 | 1.446543 | -3.22% | 1.490029 | 1.445628 | 1.523548 | 1.479120 |
| pypto_lib | c2a_reuse | 0.841698 | 0.831006 | -1.27% | 0.841545 | 0.830597 | 0.847547 | 0.835188 |
| pypto_lib_dsl | c2a_reuse | 1.365283 | 1.351696 | -1.00% | 1.364235 | 1.350249 | 1.391911 | 1.396478 |
| pypto_lib | c1a_full | 0.978581 | 0.981433 | +0.29% | 0.979228 | 0.980109 | 0.986043 | 0.986940 |
| pypto_lib_dsl | c1a_full | 1.550499 | 1.452851 | -6.30% | 1.555427 | 1.450350 | 1.581066 | 1.486855 |
| pypto_lib | c1a_reindex | 1.020379 | 1.001035 | -1.90% | 1.019769 | 1.002498 | 1.025440 | 1.005873 |
| pypto_lib_dsl | c1a_reindex | 1.502843 | 1.480087 | -1.51% | 1.505946 | 1.472307 | 1.547737 | 1.512956 |
| pypto_lib | c1a_reuse | 0.866884 | 0.857391 | -1.09% | 0.866376 | 0.858237 | 0.874216 | 0.862626 |
| pypto_lib_dsl | c1a_reuse | 1.385306 | 1.369240 | -1.16% | 1.386204 | 1.369417 | 1.419504 | 1.398947 |
| pypto_lib | engram | 0.948128 | 0.911524 | -3.86% | 0.947594 | 0.910643 | 0.953792 | 0.920859 |
| pypto_lib_dsl | engram | 0.947103 | 0.915323 | -3.36% | 0.948230 | 0.915965 | 0.955236 | 0.920172 |
| pypto_lib | moe | 3.032159 | 1.768825 | -41.66% | 3.054733 | 1.765700 | 3.535285 | 1.836330 |
| pypto_lib_dsl | moe | 1.681709 | 1.361697 | -19.03% | 1.643181 | 1.363103 | 1.813082 | 1.378170 |
| pypto_lib | layers_19_21 | 15.789710 | 8.004214 | -49.31% | 15.726427 | 7.993768 | 16.399528 | 8.104446 |
| pypto_lib_dsl | layers_19_21 | 10.691380 | 8.364982 | -21.76% | 10.754530 | 8.370017 | 11.081642 | 8.409298 |

Negative changes are faster. The restored candidate's SWA means are 0.24% (lib)
and 1.07% (DSL) above main in this batch. Its preceding lib pilot was 0.56% below
main. These small changes establish neither a meaningful improvement nor strict
zero regression. MoE and layers have large reductions; they do not satisfy the
unmet ordinary-scheduling goal on SWA. No overall completion is claimed.

The predecessor `b71-merged-cached-boundary-x1-p1` has a separate 50/50
directed/DFX matrix and repeat batches in `verification/current-report.json` and
`verification/b71-merged-cached-boundary-x1-p1-hardware.json`. Those runs validate
that earlier snapshot; they are not relabeled as tests of the current binary.
Its L1 PNG/SVG/Perfetto comparisons remain in
`swimlanes/b71-merged-cached-boundary-x1-p1/`. All captures stay inside the task
output directory.

Historical FIXPIPE169 root cause remains unproven; current original-kernel
passing captures are not a root-cause fix. A predecessor-arm smoke also failed
at rtSetDevice(6) initialization with 507033 before runtime launch. That failure
is retained separately; subsequent successful initialization does not explain it.
