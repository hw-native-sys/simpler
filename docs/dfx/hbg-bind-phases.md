# The `host_build_graph` bind phases

`host_build_graph` builds the whole task graph on the host before the device
executes anything, so the host-side **`bind` stage** — argument staging,
orchestration, the Graph Definition, and every H2D copy — is a first-class cost.
`bind` is the `chip.run.bind` `[STRACE]` span both runtimes emit; only this one
subdivides it into **segments**, one `chip.run.bind.<segment>` span each. This
page is what those segments are, how to measure them on the two decode networks that exercise
them, and the traps that make a measurement wrong rather than merely noisy.

For the marker grammar and the tool's other views, see
[host-trace.md](host-trace.md). For what the runtime records inside `host_orch`,
see [host_build_graph's profiling levels](../../src/a2a3/runtime/host_build_graph/docs/profiling_levels.md).

**A new lesson about measuring these phases belongs on this page.** The
[`hbg-bind-phases`](../../.claude/skills/hbg-bind-phases/SKILL.md) skill holds the
invocation and nothing else — it is loaded into context on every use — and
[`hbg_bind_phases`](../../simpler_setup/tools/hbg_bind_phases.py) gets a comment
only once the lesson is an invariant its code depends on.

## What the segments are

`SIMPLER_HBG_BIND_BREAKDOWN_ENABLE=1` makes the runtime emit one
`chip.run.bind.<segment>` `[STRACE]` span per segment per bind, at depth 2 inside
the `chip.run.bind` span:

| Segment | What it covers |
| ------- | -------------- |
| `args` | staging readable caller tensors H2D into slices of the slot's retained staging buffer, and exposing their existing host buffers to orchestration; pure outputs skip the H2D and the view. The buffer grows to the high-water packed size and is reused, so a steady-state workload allocates no device memory here |
| `arena_build`, `static_arena`, `gm_heap`, `shared_mem`, `runtime_init` | arena layout, GM heap and shared-memory bring-up |
| `host_orch` | **all** orchestration: every task submitted, every in-graph task recorded, the Definition built |
| `graph_upload` | one H2D of the block holding every Definition object, and binding each Graph task to the one with its key. The recorders built the objects in that block's host staging during `host_orch`, so this segment writes their headers and copies in only what did not fit |
| `arena_h2d` | one H2D of the arena's copied zone and the shared-memory image |
| `host_view_close` | closing per-run tensor-access regions and any optional device mappings; the current bind path installs none (`count=0 bytes=0`) |

The **control plane** is `host_orch + graph_upload + arena_h2d`: everything
between "the caller's data is in place" and "the device can start". It is what
the < 1 ms target applies to. `args` is excluded because it scales with the
caller's tensor bytes, not with the graph. `host_view_close` stays excluded so
current reports remain comparable with older logs, although the current bind path
has no device mappings to close.

**Two kinds were retired from the set, not merely from the output.** `relocate`
and `sm_h2d` dated from when the shared-memory image was relocated and copied on
its own; it now travels inside the single `arena_h2d` copy as that segment's
`sm=`. Neither had a recording site for as long as the change has been in, so
both were removed from `HostPhaseKind` — a log that predates the change still
carries their lines, but no current tool totals them. The table above is what a
current run emits: ten segments, three of them control plane.

**The control plane is a sum of costs, not an interval.** Its three segments are
not adjacent: `static_arena`, `shared_mem` and `gm_heap` run between
`graph_upload` and `arena_h2d`, so the segments do not form one contiguous
window. Sum the ones the bind has; do not subtract two timestamps.

**A bind runs its segments in one order and emits them in another.** Execution is
the sequence `runtime_maker.cpp` calls them in, which each span's `ts` records:

```text
args, arena_build, runtime_init, host_orch, graph_upload,
static_arena, shared_mem, gm_heap, arena_h2d, host_view_close
```

That order is what puts `static_arena`, `shared_mem` and `gm_heap` between
`graph_upload` and `arena_h2d`, which is why the control plane is a sum and not an
interval. Emission is `HostPhaseKind` order, all of it at the end of the bind —
which matters to nothing, because each span carries its own `ts` and its own
`(pid, inv)`. Reading a log by line order is not how the segments are grouped.

## Prerequisites

```bash
python3 -m venv --system-site-packages .venv     # once per worktree
source .venv/bin/activate
pip install --no-build-isolation -e .            # after every source change
.claude/skills/onboard-arch-precheck/check.sh a2a3   # exit 0 ⇒ this box can run a2a3
```

Both cases are onboard-only and must run through `task-submit`, which holds the
device lock for the whole job (see
[`.claude/rules/running-onboard.md`](../../.claude/rules/running-onboard.md)).

## The two cases

| Property | qwen3-14b decode | DeepSeek-V4 FLASH decode |
| -------- | ---------------- | ------------------------ |
| Path | `examples/a2a3/host_build_graph/qwen3_14b_decode/` | `examples/a2a3/host_build_graph/deepseek_v4_flash_decode/` |
| Entry point | standalone `main.py`, which owns its L2 `Worker` | standalone `main.py`, which owns its L3 `Worker` |
| Devices | 1 | 2 (EP2/TP2) |
| Host tasks (`host_orch tasks=`) | 47 | 129 |
| Graph submissions (`graph_upload submissions=`/`defs=`) | 40, of 1 Definition | 86, of 8 Definitions |
| Graph boundary | 26 tensors | 118 tensors, 31 scalars |
| First-run compile | seconds | **minutes** (369 kernel sources + an 11.6k-line orchestration) |
| Parameters | device memory; valid fixture streamed once before all rounds | child memory, and `--skip-golden` leaves it uninitialized |
| Marked | `manual` | `manual`, and it has no golden |

The entry point decides how a case's output is captured, which is what the recipe
below has to work around.

**The two count rows name the markers they come from, because they are properties
of the cases and the cases get edited.** They read 47 / 129 tasks and 40 / 86
submissions on `4d31f482`; they previously read 1131 tasks and 20 replays for
DeepSeek-V4, from before its orchestration moved most task submission onto the
recording threads. Re-read them from a current log rather than trusting this
table — a bind's `host_orch` and `graph_upload` spans carry both.

## Recipe A — stable numbers, many rounds

The ready-made invocation for either case lives in the
[`hbg-bind-phases`](../../.claude/skills/hbg-bind-phases/SKILL.md) skill;
`python -m simpler_setup.tools.hbg_bind_phases <log>` turns its log into
per-segment statistics. This section is what the switches mean and why the traps
below exist.

Six rounds is the working minimum: this box is shared, and a single bind has been
seen to land 3.5× off its own minimum. Which statistic to read depends on the
question. For "how long does this path take", take the **minimum of the per-round
sums** — the quietest bind is the closest this box gets to the machine's own cost.
For "did a change move it", see [Comparing two branches](#comparing-two-branches)
below; the answer there is not a minimum.

Four switches and one flag make the measurement, and each is load-bearing:

| Switch | Why |
| ------ | --- |
| `SIMPLER_HBG_BIND_BREAKDOWN_ENABLE=1` | emits the segment spans at all |
| `--log-level timing` | pins the level the report is read at, and is the only way it reaches the `[stamp]` line; TIMING is already the default, so this records a condition rather than enabling one |
| `TORCH_DEVICE_BACKEND_AUTOLOAD=0` | keeps CPU golden imports from loading `torch_npu`; the `torch_backend_autoload` timing record confirms the effective setting and observed module state |
| `SIMPLER_SKIP_DEVICE_RUN=1` | returns at `simpler_launch_run`, so the host path is measured without a working device run |
| `--skip-golden` | with the device skipped the outputs a golden check compares are never produced. Qwen still streams its valid fixture once before the measured rounds; on dsv4 the flag also skips the 42.6 GiB-per-rank fixture upload |

**`--log-level` overrides a level that is already TIMING; it does not turn the
records on.** There is one control point and it is the Python logger named
`simpler`: [`python/simpler/_log.py`](../../python/simpler/_log.py) sets it to
TIMING at import when nothing else has, and `Worker` snapshots that logger's
effective level once — feeding it to the host log and to every forked chip
child, which is why the same level governs the host and device sides.
`configure_logging()` in
[`simpler_setup/log_config.py`](../../simpler_setup/log_config.py) is the only
way a CLI moves that snapshot; there is no environment variable.

The recipe passes `--log-level timing` anyway, because **the level is the one
measurement condition with no record of its own in the log.** Autoload state has
`torch_backend_autoload`; the level has nothing, so a run that omits the flag
leaves it implicit in whichever default that commit compiled in. Two arms of a
cross-commit A/B could then run at different levels with neither `[stamp]`
showing it — the same silent-mismatch failure the autoload record exists to
prevent. Passing it puts the level in the stamp, where a `diff` of the two first
lines catches a mismatch.

SceneTest and the standalone Qwen driver emit one `torch_backend_autoload`
record per interpreter after torch-dependent argument preparation and before
the first dispatch. `effective` reports the environment's dispatch-time intent
using torch's current private autoload predicate; `torch_npu_loaded` reports the
observed module state and is the authoritative field if the environment changed
after torch was imported. `raw` is the JSON-encoded environment value (`null`
when unset); values longer than 64 characters carry their first 64 characters
and `raw_truncated=true`, keeping the record single-line and bounded.

**The dsv4 driver is neither of those two, so a dsv4 log carries no such
record** — `hbg_bind_phases` prints "backend-autoload state must be established
before comparing this log" on every dsv4 measurement. The check below that this
is the one condition already known to have produced a wrong number therefore has
no in-log witness on that case; the `[stamp]` line is all a dsv4 A/B has, so both
arms must be read off it by hand.

**`SIMPLER_SKIP_DEVICE_RUN` is presence-based.** `SIMPLER_SKIP_DEVICE_RUN=0` still
skips; `unset` it. It is a temporary handle from the dsv4 bring-up and is deleted
once that case's device execution works.

Both cases now run through standalone `main.py` drivers. Qwen owns its L2
`Worker` in the invoked process, while dsv4 owns its L3 `Worker`; neither needs
module-runner `--runtime` / `--level` forwarding to expose the segment spans.

A 2-rank case emits one bind per rank per round, so six rounds is twelve binds.
Pass `--rounds` to the parser so it infers the rank count and drops one cold bind
*per rank* rather than one in total.

A skipped run still writes `host_phase_records.jsonl`, so Recipe B works without
touching the device. Every phase in that artifact is produced on the host during
bind, so the skip path writes it exactly as the device-run teardown does; what
gates it is Recipe B's three conditions, none of which is the device.

### Reading the segments out

The parser does this grouping; read it out by hand only to check something it does
not report. The log lands in `outputs/hbg_bind_stats_<sha>.log` unless `-o` names
it:

```bash
grep -oE 'name=chip\.run\.bind\.[a-z0-9_]+ ts=[0-9]+ dur=[0-9]+[^[]*' outputs/hbg_bind_stats_<sha>.log
```

The character class has to admit digits, or it drops every `arena_h2d` — the only
H2D left, and the one that itemizes the whole upload.

Each span carries `ts` (a `CLOCK_MONOTONIC` timestamp), `dur`, `pid` and `inv`,
plus the segment's own attributes — the six kernel counters on all of them, then
`tasks=` and `heap_used=` on `host_orch`, `defs=`, `bytes=`, `submissions=` and
`spilled=` on `graph_upload`, and `arena_h2d`'s itemized upload. **A bind is one
`(pid, inv)`**, so grouping needs no inference from order: `inv` is the run epoch
the enclosing `chip.run.bind` allocated, and two ranks writing one stream cannot
be confused for each other. Sum the control-plane segments **within each bind**
and take the minimum of those sums. Never sum minima taken across binds; that
total belongs to no bind and can point the wrong way (see below).

**`spilled=` should be 0 on every bind but the first.** It counts the Definition
objects the recorders could not build inside the retained staging, which
`graph_upload` then has to copy in. The first bind of a process has nothing
retained and so spills all of them; a later bind that still spills means the run's
Definitions outgrew the high-water mark the previous one left, and the copies are
back. It is not spelled `copied=` on purpose: on `arena_h2d` that name means a
zone, not a count.

**A segment's `bytes=` is what that segment itself copied, so no copy is counted
twice.** `graph_upload` counts the Definition objects it uploads, which are all it
copies; `arena_h2d`'s `bytes=` is its single copy, exactly partitioned by the
`copied=` and `sm=` beside it. A Graph invocation's boundary values are inside that
`sm=`: they live in the outer Graph task's ordinary argument pools, so they travel
with every other task's arguments rather than as a population of their own.
(`shared_mem`'s `bytes=` is the image the arena grew to hold, which `arena_h2d`
then ships as `sm=`; that is the one figure two segments both report, and neither
is a copy count of the other.)

The first bind of each rank is warm-up and belongs in neither statistic; drop it
explicitly rather than letting a minimum quietly exclude it.

### Was the segment running or waiting?

Every line also carries the kernel counters, which say what a duration cannot. Each
covers the same span the duration does.

| Field | Source | What it says |
| ----- | ------ | ------------ |
| `cpu_ns` | the bind thread's `CLOCK_THREAD_CPUTIME_ID` | how much of the segment that thread spent **running**, so `dur_ns - cpu_ns` is what it spent **off CPU** — blocked *and* runnable-but-preempted, which the two `*csw` counters separate |
| `rec_cpu_ns` | every recording worker's CPU clock, summed | how many threads' worth of work ran **alongside** — a ratio to `dur_ns`, never something to subtract from it |
| `minflt` / `tminflt` | `getrusage` `RUSAGE_SELF` / `RUSAGE_THREAD` | minor faults for the whole process, and the bind thread's share of them; the difference is the recorders' |
| `nvcsw` / `nivcsw` | `getrusage` `RUSAGE_SELF` | voluntary and involuntary context switches **for the whole process**, so the recorders' switches are in here too. A high `nivcsw` says the box was loaded, not that this code was; neither counter isolates the bind thread — `tminflt` is the only thread-scoped field |

**`rec_cpu_ns` and `tminflt` are Linux-only, and read zero elsewhere.** Sampling
another thread's CPU clock needs `pthread_getcpuclockid` and a per-thread fault count
needs `getrusage(RUSAGE_THREAD)`; Darwin provides neither. That costs nothing where it
matters — a bind is profiled on the silicon it binds to — but on a `*sim` platform
built for macOS a `reccpu` of 0 means "not measurable here", not "nothing ran
alongside". `cpu_ns` uses `CLOCK_THREAD_CPUTIME_ID` for the calling thread and works
everywhere.

```bash
python -m simpler_setup.tools.phase_time_split <log>          # cold and warm, per segment
python -m simpler_setup.tools.phase_time_split <log> --phase host_orch
```

**CPU time comes from per-thread clocks and never from rusage.** `ru_utime` and
`ru_stime` are accounted per scheduler tick — 10 ms at `CLK_TCK=100` — so on a segment
of a millisecond they quantise to either zero or a whole tick, and the values look
plausible one at a time while being noise. A per-thread clock reads the scheduler's
running total in nanoseconds; measured against a busy and a sleeping thread over a
1.29 ms window it resolved both to under 10 µs. The three counters stay on rusage
because they are event counts, which do not quantise.

**On-CPU sends you somewhere different from off-CPU.** A segment that is mostly
on-CPU is running, so split it further by the fault count and by the syscalls inside
its window. A segment that is mostly off-CPU is waiting, and `nvcsw` versus `nivcsw`
says whether it blocked or merely lost the CPU. Reading a fault count without this
split is what let a page-fault tail be chased three times on a segment whose faults
were on threads with spare parallelism — see
[`docs/investigations/2026-08-host-orch-phase-tail-is-page-faults.md`](../investigations/2026-08-host-orch-phase-tail-is-page-faults.md).

Device wall clock for the same rounds comes from the `[STRACE]` markers, on a run
that did not skip the device:

```bash
grep -oE 'device_wall ts=0 dur=[0-9]+' <log> | \
  awk -F'dur=' '{printf "%.2f\n", $2/1e6}' | sort -n
```

### Comparing two branches

A branch comparison is a different measurement from a single reading, and two of
its failure modes have already produced wrong answers on this box.

**Both arms must be the same ruler, and the log is the only witness you get.** A
baseline missing `TORCH_DEVICE_BACKEND_AUTOLOAD=0` produced a wrong number once:
it alone paid for `torch_npu` grabbing a device on import, and the difference was
attributed to the branch. Compare the `torch_backend_autoload` timing record in
both logs; the measurement recipe produces
`setting=0 raw="0" raw_truncated=false effective=disabled torch_imported=true torch_npu_loaded=false`.
`hbg_bind_phases` prints every distinct record above its table and warns when the
log does not carry one.

The recipe also echoes the command it is about to run, verbatim, as the log's
first line, and `hbg_bind_phases` prints that line above the table:

```bash
diff <(head -1 base.log) <(head -1 measure.log)   # must differ only in the commit
```

An arm with no `[stamp]` line cannot take part in a comparison.

**Interleave the conditions; never run one after the other.** `base` then
`measure` attributes every drift in host load to the branch, and the drift is
larger than most effects worth measuring. Alternate instead —
`base, measure, base, measure` — which gives one minimum-of-sums per arm per
repetition, and require the delta between them to **agree in sign across the
repetitions**. A repetition that disagrees says the run was contended, not that
the effect is small: on one dsv4 bind `graph_upload` came out +0.46 ms against
−0.20 ms on the other three, and the same bind carried a run whose `sm_h2d` was
5.93 ms against a 0.6 ms norm.

**One statistic decides: the minimum of the per-bind sums.** Sum the
control-plane segments *within* each bind, take the minimum across the warm
binds, and compare those. A min of sums is not a sum of mins and the two can
disagree in sign — each segment's minimum comes from whichever bind was quietest
*for that segment*, so summing per-segment minima produces a total no bind
achieved. On one dsv4 comparison the sum-of-minima moved −0.30 ms while the
minimum-of-sums moved +0.16 ms, from the same log. `hbg_bind_phases` reports the
minimum-of-sums as its `total` row; never assemble a total by hand from the
per-phase `min` column.

The median and the max in that table are **not** a second decision rule. Read
them for one thing only: a change that lowers the minimum while widening the
range has made the cost less predictable, which is a cost of its own and worth
reporting alongside the minimum.

**Judge a segment the diff does not touch.** `host_orch`'s own scatter on dsv4
spans 2.6–4.9 ms across binds of an unmodified `main` — wider than most changes
being tested — so a ±0.5 ms difference there is not resolvable by comparing
durations however many rounds are run. When a segment matters and its scatter
swamps it, instrument the mechanism instead: a sub-counter around the suspected
work answers in one run what a duration comparison cannot answer in ten.

## Recipe B — one round with a swimlane

The summed `host-orch phase=` lines cannot be placed on a timeline inside
`host_orch`: they are cost shares. The per-event view comes from the runtime's
per-producer record pool, written to `outputs/<case>_<ts>/host_phase_records.jsonl` —
one record per orchestrator operation, each with its own interval.

Three conditions must all hold, and the first two produce an empty result
silently:

1. **`SIMPLER_HBG_HOST_PHASE_RECORDS_ENABLE=1`**, which is what arms the pool.
   `SIMPLER_HBG_BIND_BREAKDOWN_ENABLE` does not: it gates the summed
   the segment spans alone, so Recipe A's environment collects no records.
2. **A diagnostic flag must be on**, because that is what makes
   `CallConfig.output_prefix` non-empty. `--enable-scope-stats` is the cheapest
   choice when only Host phase records are needed. `--enable-chip-swimlane` is
   also supported for same-host L3 and writes each ChipWorker capture below a
   separate `rankN/dN` directory, but it collects substantially more data.
3. **`--rounds` must be 1.** `rounds > 1` force-disables every diagnostic flag —
   this one does warn, `<flag> disabled: --rounds > 1` per flag
   ([`simpler_setup/scene_test.py`](../../simpler_setup/scene_test.py)), but the
   warning sits in a log whose run otherwise passed.

The device run may be skipped or may fail; neither costs you the artifact. Every
phase recorded is host work done during bind, so both `SIMPLER_SKIP_DEVICE_RUN`
and the device-run teardown write it. That is what lets a case whose device
execution does not complete still yield its prepare timing — and it is why a
swimlane for a case that hangs on device is cheaper to take with the variable set
than to take by waiting out the stall.

**The flag that satisfies condition 2 also moves the log.** A non-empty
`CallConfig.output_prefix` redirects every host-log record — segment spans
and `[STRACE]` spans alike — from stderr into `outputs/<case>_<ts>/host.<pid>.log`,
one file per process ([`python/simpler/worker.py`](../../python/simpler/worker.py)
sets the directory on the L3 submit path and in the forked chip child;
[`src/common/log/host_log.cpp`](../../src/common/log/host_log.cpp) opens the file).
So Recipe A's `grep -c 'name=chip.run.bind\.' "$LOG"` reports **0** for a Recipe B run that
worked perfectly, and the finisher must read the prefix's own logs. Measured on a
2-rank dsv4 run: `$LOG` alone yields `No [STRACE] markers found` and drops every
phase record, while `$LOG` plus the prefix's logs attaches all 4186 of them.

The skill's timeline mode is this recipe; it finishes with

```bash
D=outputs/<case>_<ts>
# The clock anchors are split: the invoking process wrote its own to $LOG, each
# chip child wrote its own under $D. Concatenating keeps every pid alignable.
cat "$LOG" "$D"/host.*.log > "$D/bind_timeline.log"
python -m simpler_setup.tools.strace_timing "$D/bind_timeline.log" \
    --host-phase-records "$D/host_phase_records.jsonl" \
    --swimlane "$D/host_swimlane.json"
```

Load the JSON in [Perfetto](https://ui.perfetto.dev) or `chrome://tracing`. Each
rank is its own pid lane, and every record is drawn inside its
`chip.run.bind`. The same command with the dsv4 log and its records file gives
the two-rank version; `dropped` in the artifact's header says whether the pool
truncated anything (it is 0 for both cases: 337 records for qwen, 1887 per rank
for dsv4).

## Reading the result

**The first round is cold, and not by a little.** On dsv4's first bind
`static_arena` spends 97.8 ms allocating the 2 GiB ring heap against 0.002 ms on
later binds, and `host_orch` runs 8% long. Recipe B therefore describes
*structure* — the order of operations, the per-operation distribution — while
Recipe A gives the numbers.

**Per-operation records show tails the sums hide.** dsv4's `submit_task` has a
median of 0.28 µs and a maximum of 54.49 µs, a 195× spread that a
`total_ns / count` mean reports as 0.68 µs.

**Not all of `host_orch` is instrumented.** The gaps between records — fanin
computation, tensormap registration, scope bookkeeping — are 22% of `host_orch`
on qwen and 25–30% on dsv4, and they are the second-largest item inside it.
Subtract the records' sum from the segment to see them.

**`heap_used=` on `host_orch` is the run's real GM footprint**, so it is the
metric for any change to allocation or to the Graph expansion pool. It is exact
and repeatable — byte-identical across runs — which makes it a better regression
signal than any duration on a shared box.

## Traps

| Trap | Symptom | What to do |
| ---- | ------- | ---------- |
| Any diagnostic flag on (so, every Recipe B run) | `$LOG` has no `[STRACE]` markers at all, run passes | the non-empty `output_prefix` moved the host log to `outputs/<case>_<ts>/host.<pid>.log`; grep and parse those too |
| A `SceneTestCase` with `device_count > 1` run through the module runner | log has zero segment spans, test passes | give the child command `--runtime <rt> --level 3`; a standalone `main.py` case needs nothing |
| `SIMPLER_SKIP_DEVICE_RUN=0` | run still skips the device, "PASSED" means nothing ran | `unset` the variable |
| `--rounds 6` with `--enable-scope-stats` | no `outputs/<case>_<ts>/` artifacts, plus a `disabled: --rounds > 1` warning | one round for artifacts, many rounds for numbers |
| Only `SIMPLER_HBG_BIND_BREAKDOWN_ENABLE` set for Recipe B | segment spans present, no `host_phase_records.jsonl` | the records are a separate switch: also export `SIMPLER_HBG_HOST_PHASE_RECORDS_ENABLE=1` |
| Comparing a log with no `[stamp]` first line | the parser says so above the table | re-run it through the recipe; conditions cannot be recovered from memory |
| Subtracting timestamps for the control plane | ~300 ms instead of ~3 ms | sum the segments; `arena_h2d` is not adjacent |
| Summing per-segment minima by hand | a total no bind achieved; can invert the sign | read the tool's `total` row — the minimum of the per-bind sums |
| `--rounds 1` for numbers | the tool refuses: every bind is a rank's warm-up | six rounds; `--keep-first` only to look at the cold bind deliberately |
| Single bind, or comparing across differently-loaded moments | swings of 3.5× | six rounds, compare minima, keep an untouched segment as a control |
| Reading a **warm** bind as a **steady-state** one | the first few warm binds carry several hundred more `minflt` than the last ones, and a per-bind average built from them is a warm-up figure wearing a steady-state label | the parser drops the cold bind per rank, not the decay after it: on dsv4 `host_orch`'s `minflt` runs 989, 983 (cold), then 114, 173, 54, 3, 13, 13, 11, 8. Take **twelve** rounds and read the last binds, or check the count column is flat before quoting a duration |
| Dividing a total by the bind count | a per-bind figure no bind ever had, inflated by the cold and decaying ones | bucket by bind first and look at the sequence; only quote a mean over binds whose counts already agree |
| `base` then `measure`, sequentially | a load drift reads as the branch's effect | interleave the arms and require the sign to agree per repetition |
| Stale build | mass collection errors, or a `launch_aicpu_num (0)` failure | `pip install --no-build-isolation -e .` after every `HEAD` move |

## Reference numbers

Both columns are one measurement session on `main` at **`777d4171`**, host
`host_build_graph`, on one a2a3 die for qwen and two for dsv4, four rounds each with
the warm-up bind dropped — 3 warm binds for qwen and 6 for dsv4, since a
2-rank case emits one bind per rank per round and the parser drops one cold bind per
rank. Durations are the full **range** across those
binds; `heap_used`, every `bytes=` and every count are exact and repeat
byte-identically.

**Warm, not steady-state.** Dropping one cold bind per rank does not reach the steady
state: the decay behind it takes about six binds on dsv4, so a four-round session spends
most of its warm binds inside it. Both columns therefore sit somewhere in the warm-up,
which is another reason to treat them as orientation rather than as a baseline — see the
trap table above.

**For orientation, not thresholds, and pinned to a commit for a reason.** The
machine's other tenants move every duration here, so the range is the point: a
change smaller than the range beside it cannot be demonstrated by comparing two
runs. The counts move too — they are properties of the cases, and the cases are
edited. Re-measure rather than trusting this table; the recipes above are the part
meant to outlive it.

| Measurement | qwen3-14b decode | dsv4 FLASH decode |
| ----------- | ---------------- | ----------------- |
| control plane | 1.11–1.53 ms | 3.63–6.81 ms |
| `host_orch` | 0.44–0.75 ms (47 tasks) | 2.60–4.91 ms (1131 tasks) |
| `graph_upload` | 0.56–0.96 ms / 40 submissions, 232,320 B † | 0.39–1.14 ms / 20 submissions, 671,144 B † |
| `sm_h2d` † | 0.067–0.068 ms / 233,799 B | 0.54–0.98 ms / 5,620,195 B |
| `arena_h2d` † | 0.035–0.039 ms / 632 B | 0.03–0.10 ms / 632 B |
| `heap_used` | 127,673,344 | 2,038,508,544 |
| device wall | 39.3 ms | does not complete yet (`sched_error_code=5 INVALID_ARGS`) |
| `args` (excluded) | 1.37 s / 40.9 GB, 19 of 20 staged | 1.48 s / 45.8 GB, 77 of 92 staged |
| `host_view_close` (excluded, legacy mapping path) | 0.25 s / 40.9 GB | 0.28 s / 45.8 GB |

† The three upload rows are the markers as they read at that commit, before the
upload was restructured: `graph_upload`'s `bytes=` then also counted the Graph
submission block, `sm_h2d` was still a copy of its own, and `arena_h2d` was the
copied zone alone. A run today has no `sm_h2d` kind at all, counts only the
Definition objects in `graph_upload`, carries no submission block, and ships both
remaining regions in `arena_h2d` — so the same case reports different figures for
the same work.

**dsv4's `args` and `host_view_close` rows no longer describe that case at this
scale.** Both are per-byte costs over what a bind stages, and dsv4's parameters
now live in child memory: allocated once before the first round, and passed
through without malloc, H2D or a host view. What still crosses is
`num_tokens_per_owner`, the one caller tensor the host orchestrator has to read —
so a bind stages **1 of its 92 tensors, 8 bytes**. On `dcf7559e8`, 12 binds
(`--rounds 6`, both ranks) measure `args` at 0.036–0.075 ms and
`host_view_close` at 0.0012–0.0030 ms with `count=0 bytes=0`, against 1.48 s and
0.28 s over 45.8 GB above. The same run peaks at 1.31 GiB of host RSS across the
whole process tree under `--skip-golden`, and at 23.4 GiB when the fixture is
streamed in, where the row above cost ~45.5 GB per rank. qwen still stages its
fixture.

The rows also describe the legacy mapping behavior at the pinned commit. A
current bind uses the caller's existing host buffers as its
orchestration views, so it performs no `halHostRegister` calls and reports
`host_view_close count=0 bytes=0`. On Qwen3-14B this makes the close marker
20.12–24.73 us instead of the 0.25 s shown above. The old `args` figure included
20 registrations in addition to staging 19 tensors H2D; current `args` retains
the H2D work but removes that registration side.

Three of these deserve reading together. `host_orch` is the whole story on dsv4 —
839 `submit_task`, 743 `record_in_graph_task` and 272 `alloc_tensors` per bind against qwen's
5, 277 and 2 — and its 2.3 ms of scatter is why a claim about it needs a
sub-counter rather than a stopwatch. At the pinned commit, `args` plus
`host_view_close` are two orders of magnitude above everything else while being
excluded from the control plane: they are staging and legacy mapping costs over
the ~41–46 GB of weights, not graph dispatch. Current qwen runs retain the
staging cost in `args` but close no mappings; moving dsv4's parameters to child
memory left its bind staging one 8-byte tensor, whose caller-buffer view also
needs no mapping. And dsv4's device wall is absent because the case did not
complete on device at the pinned commit — it is a completion case with no golden
whose host path is what these numbers describe, which is also why
`SIMPLER_SKIP_DEVICE_RUN` appears in its recipe.
