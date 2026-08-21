# The `host_build_graph` bind phases

`host_build_graph` builds the whole task graph on the host before the device
executes anything, so the host-side **`bind` stage** — argument staging,
orchestration, the Graph Definition, and every H2D copy — is a first-class cost.
`bind` is the `chip.run.bind` `[STRACE]` span both runtimes emit; only this one
subdivides it into **phases**, one `bind phase=<p>` line each. This page is what
those phases are, how to measure them on the two decode networks that exercise
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

`SIMPLER_HBG_BIND_BREAKDOWN_ENABLE=1` makes the runtime emit one `bind phase=`
line per segment per pass at `LOG_TIMING`:

| Segment | What it covers |
| ------- | -------------- |
| `args` | staging the caller's host tensors and mapping them for host access |
| `arena_build`, `static_arena`, `gm_heap`, `shared_mem`, `runtime_init` | arena layout, GM heap and shared-memory bring-up |
| `host_orch` | **all** orchestration: every task submitted, every Graph node recorded, the Definition built |
| `graph_upload` | the Definition objects and the replay boundary images |
| `relocate` | pointer relocation inside the shared-memory image |
| `sm_h2d` | the task descriptors and payloads, host → device |
| `arena_h2d` | the runtime arena's copied zone |
| `host_view_close` | unmapping the host views taken in `args` |

The **control plane** is `host_orch + graph_upload + relocate + sm_h2d +
arena_h2d`: everything between "the caller's data is in place" and "the device
can start". It is what the < 1 ms target applies to. `args` and
`host_view_close` are excluded — they scale with the caller's tensor bytes, not
with the graph.

**The control plane is a sum of costs, not an interval.** `arena_h2d` runs
*after* `host_view_close`, hundreds of milliseconds later, so the segments do not
form one contiguous window. Sum the five; do not subtract two timestamps.

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
| Devices | 1 | 2 (EP2/TP2) |
| Level | 2 | 3 |
| Host tasks | 47 | 1131 |
| Graph replays | 40, of a 277-node Definition | 20, of a 743-node Definition |
| Graph boundary | 26 tensors | 118 tensors, 31 scalars |
| First-run compile | seconds | **minutes** (369 kernel sources + an 11.6k-line orchestration) |
| Marked | `manual` | `skip_golden` |

The level decides how a case's output is captured, which is what the recipe below
has to work around.

## Recipe A — stable numbers, many rounds

The ready-made invocation for either case lives in the
[`hbg-bind-phases`](../../.claude/skills/hbg-bind-phases/SKILL.md) skill;
`python -m simpler_setup.tools.hbg_bind_phases <log> --rounds N` turns its log into
per-phase statistics. This section is what the switches mean and why the traps
below exist.

Six rounds is the working minimum: this box is shared, and a single pass has been
seen to land 3.5× off its own minimum. Which statistic to read depends on the
question. For "how long does this path take", take the **minimum of the per-round
sums** — the quietest pass is the closest this box gets to the machine's own cost.
For "did a change move it", see [Comparing two branches](#comparing-two-branches)
below; the answer there is not a minimum.

Four switches and one flag make the measurement, and each is load-bearing:

| Switch | Why |
| ------ | --- |
| `SIMPLER_HBG_BIND_BREAKDOWN_ENABLE=1` | emits the `bind phase=` lines at all |
| `SIMPLER_LOG_LEVEL=TIMING` | the level they are emitted at |
| `TORCH_DEVICE_BACKEND_AUTOLOAD=0` | otherwise `torch_npu` grabs a device on import, which a host-only measurement does not want — and nothing in the log records whether it was set |
| `SIMPLER_SKIP_DEVICE_RUN=1` | returns at `simpler_launch_run`, so the host path is measured without a working device run |
| `--skip-golden` | with the device skipped the outputs a golden check compares are never produced, so a checking case such as qwen otherwise fails at validation with the whole measurement already complete in the log |

**`SIMPLER_SKIP_DEVICE_RUN` is presence-based.** `SIMPLER_SKIP_DEVICE_RUN=0` still
skips; `unset` it. It is a temporary handle from the dsv4 bring-up and is deleted
once that case's device execution works.

**A multi-device case must be invoked as its own L3 child command**
(`--runtime host_build_graph --level 3`). A case whose `device_count > 1` is
otherwise dispatched by the module runner as one subprocess per class, and
`run_jobs` captures that subprocess's stdout and prints it only on failure — so a
passing run yields a log with **no** `bind phase=` lines at all. qwen is
`level=2` and needs no child command.

A 2-rank case emits one pass per rank per round, so six rounds is twelve passes.
Pass `--rounds` to the parser so it infers the rank count and drops one cold pass
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
grep -oE 'bind phase=[a-z_]+ start_ns=[0-9]+ dur_ns=[0-9]+[^[]*' outputs/hbg_bind_stats_<sha>.log
```

Each line carries `start_ns` (a `CLOCK_MONOTONIC` timestamp) plus the segment's
own attributes — `tasks=` and `heap_used=` on `host_orch`, `bytes=` on every H2D
segment, `count=` on `graph_upload`. Group the lines into passes — `arena_h2d` is
the last segment of a pass, so it closes one — then sum the five control-plane
segments **within each pass** and take the minimum of those sums. Never sum
minima taken across passes; that total belongs to no pass and can point the wrong
way (see below).

The first pass of each rank is warm-up and belongs in neither statistic; drop it
explicitly rather than letting a minimum quietly exclude it.

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
attributed to the branch. Nothing in this repo reads that variable — it is
`torch_npu`'s own — so no log line records whether it was set. The recipe
therefore echoes the command it is about to run, verbatim, as the log's first
line, and `hbg_bind_phases` prints that line above the table:

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
the effect is small: on one dsv4 pass `graph_upload` came out +0.46 ms against
−0.20 ms on the other three, and the same pass carried a bind whose `sm_h2d` was
5.93 ms against a 0.6 ms norm.

**One statistic decides: the minimum of the per-pass sums.** Sum the five
control-plane segments *within* each pass, take the minimum across the warm
passes, and compare those. A min of sums is not a sum of mins and the two can
disagree in sign — each segment's minimum comes from whichever pass was quietest
*for that segment*, so summing per-segment minima produces a total no pass
achieved. On one dsv4 comparison the sum-of-minima moved −0.30 ms while the
minimum-of-sums moved +0.16 ms, from the same log. `hbg_phase_stats` reports the
minimum-of-sums as its `total` row; never assemble a total by hand from the
per-phase `min` column.

The median and the max in that table are **not** a second decision rule. Read
them for one thing only: a change that lowers the minimum while widening the
range has made the cost less predictable, which is a cost of its own and worth
reporting alongside the minimum.

**Judge a segment the diff does not touch.** `host_orch`'s own scatter on dsv4
spans 2.6–4.9 ms across passes of an unmodified `main` — wider than most changes
being tested — so a ±0.5 ms difference there is not resolvable by comparing
durations however many rounds are run. When a segment matters and its scatter
swamps it, instrument the mechanism instead: a sub-counter around the suspected
work answers in one pass what a duration comparison cannot answer in ten.

## Recipe B — one round with a swimlane

The summed `bind phase=` lines cannot be placed on a timeline inside
`host_orch`: they are cost shares. The per-event view comes from the runtime's
per-producer record pool, written to `outputs/<case>_<ts>/host_phase_records.jsonl` —
one record per orchestrator operation, each with its own interval.

Three conditions must all hold, and the first two produce an empty result
silently:

1. **`SIMPLER_HBG_HOST_PHASE_RECORDS_ENABLE=1`**, which is what arms the pool.
   `SIMPLER_HBG_BIND_BREAKDOWN_ENABLE` does not: it gates the summed
   `bind phase=` lines alone, so Recipe A's environment collects no records.
2. **A diagnostic flag must be on**, because that is what makes
   `CallConfig.output_prefix` non-empty. `--enable-scope-stats` is the cheapest
   for an L3 case; `--enable-chip-swimlane` raises `NotImplementedError` for
   `level=3` (per-chip-process filename collision).
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

The skill's timeline mode is this recipe; it finishes with

```bash
python -m simpler_setup.tools.strace_timing <log> \
    --host-phase-records outputs/<case>_<ts>/host_phase_records.jsonl \
    --swimlane host_swimlane.json
```

Load the JSON in [Perfetto](https://ui.perfetto.dev) or `chrome://tracing`. Each
rank is its own pid lane, and every record is drawn inside its
`chip.run.bind`. The same command with the dsv4 log and its records file gives
the two-rank version; `dropped` in the artifact's header says whether the pool
truncated anything (it is 0 for both cases: 337 records for qwen, 1887 per rank
for dsv4).

## Reading the result

**The first round is cold, and not by a little.** On dsv4's first pass
`static_arena` spends 97.8 ms allocating the 2 GiB ring heap against 0.002 ms on
later passes, and `host_orch` runs 8% long. Recipe B therefore describes
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
| dsv4 run through the module runner's L3 phase | log has zero `bind phase=` lines, test passes | run the L3 child command directly (Recipe A) |
| `SIMPLER_SKIP_DEVICE_RUN=0` | run still skips the device, "PASSED" means nothing ran | `unset` the variable |
| `--rounds 6` with `--enable-scope-stats` | no `outputs/<case>_<ts>/` artifacts, plus a `disabled: --rounds > 1` warning | one round for artifacts, many rounds for numbers |
| Only `SIMPLER_HBG_BIND_BREAKDOWN_ENABLE` set for Recipe B | `bind phase=` lines present, no `host_phase_records.jsonl` | the records are a separate switch: also export `SIMPLER_HBG_HOST_PHASE_RECORDS_ENABLE=1` |
| Comparing a log with no `[stamp]` first line | the parser says so above the table | re-run it through the recipe; conditions cannot be recovered from memory |
| Subtracting timestamps for the control plane | ~300 ms instead of ~3 ms | sum the five segments; `arena_h2d` is not adjacent |
| Summing per-segment minima by hand | a total no pass achieved; can invert the sign | read the tool's `total` row — the minimum of the per-pass sums |
| `--rounds 1` for numbers | the tool refuses: every pass is a rank's warm-up | six rounds; `--keep-first` only to look at the cold pass deliberately |
| Single pass, or comparing across differently-loaded moments | swings of 3.5× | six rounds, compare minima, keep an untouched segment as a control |
| `base` then `measure`, sequentially | a load drift reads as the branch's effect | interleave the arms and require the sign to agree per repetition |
| Stale build | mass collection errors, or a `launch_aicpu_num (0)` failure | `pip install --no-build-isolation -e .` after every `HEAD` move |

## Reference numbers

Both columns are one measurement session on `main` at **`777d4171`**, host
`host_build_graph`, on one a2a3 die for qwen and two for dsv4, four rounds each with
the warm-up pass dropped — 3 steady-state passes for qwen and 7 for dsv4, since a
2-rank case emits one per rank. Durations are the full **range** across those
passes; `heap_used`, every `bytes=` and every count are exact and repeat
byte-identically.

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
| `graph_upload` | 0.56–0.96 ms / 40 uploads, 232,320 B | 0.39–1.14 ms / 20 uploads, 671,144 B |
| `sm_h2d` | 0.067–0.068 ms / 233,799 B | 0.54–0.98 ms / 5,620,195 B |
| `arena_h2d` | 0.035–0.039 ms / 632 B | 0.03–0.10 ms / 632 B |
| `heap_used` | 127,673,344 | 2,038,508,544 |
| device wall | 39.3 ms | does not complete yet (`sched_error_code=5 INVALID_ARGS`) |
| `args` (excluded) | 1.37 s / 40.9 GB, 19 of 20 staged | 1.48 s / 45.8 GB, 77 of 92 staged |
| `host_view_close` (excluded) | 0.25 s / 40.9 GB | 0.28 s / 45.8 GB |

Three of these deserve reading together. `host_orch` is the whole story on dsv4 —
839 `submit_task`, 743 `record_node` and 272 `alloc_tensors` per bind against qwen's
5, 277 and 2 — and its 2.3 ms of scatter is why a claim about it needs a
sub-counter rather than a stopwatch. `args` plus `host_view_close` are two orders of
magnitude above everything else while being excluded from the control plane: they
are per-byte costs over the ~41–46 GB of weights each case stages, so they belong
to getting the weights resident, not to dispatching a graph. And dsv4's device wall
is absent because the case does not complete on device on this commit — it is a
`skip_golden` completion case whose host path is what these numbers describe, which
is also why `SIMPLER_SKIP_DEVICE_RUN` appears in its recipe.
