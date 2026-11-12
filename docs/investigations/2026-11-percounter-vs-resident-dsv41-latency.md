# percounter on DSV4.1: the claim range, and a probe that faked a 45% win

`tests/fdwic/dsv41_scheduler` runs real DeepSeek-V4.1 regions on two A5 devices with
the original kernel binaries. percounter started this investigation between **2.11x
faster and 37% slower** than the resident scheduler across its eighteen cases, and the
spread is now attributed: a lane drew four order-list positions at a time and withheld
the ones it had not reached. Taking one position per draw moves sixteen of the eighteen
by 7% to 48% and puts percounter ahead on all of them.

Most of this entry is what the measurement cost. Seven mechanisms were killed before the
right one, one of them because it was tested in the only direction that confirms it, and
a probe inserted to time a different effect produced a 45% improvement that turned out
to be an artifact of the probe itself.

## What was measured

Device wall `chip.run.runner_run.device_wall` at profiling level 0, one warmup and five
measured rounds, on both case flavours. The runner executes three callables per
iteration — a reference, the *prefix* that produces the slice's real inputs, then the
case — so a median over a run's spans mixes three populations. Separate them by phase
position before reading anything; the method note at the end says why magnitude will not
do it.

`aicore_kernel.o` was rebuilt from a dropped cache for every arm and its md5 recorded
with the measurement; see the method note at the end for why that is not optional.

## The answer: one position per draw

`base` is four positions per draw with the long-kernel switch; `one` is one position per
draw; `3 us -> 1 us` lowers that switch's threshold instead. Microseconds, p50.

| case | base | one | `3 us -> 1 us` | resident |
| ---- | ---- | --- | -------------- | -------- |
| **pypto_lib** | | | | |
| `swa` | 925.2 | **617.9** | 925.1 | 1095.1 |
| `c2a_full` | 2170.8 | **1287.2** | 1968.6 | 1829.4 |
| `c2a_reuse` | 1818.2 | 1024.9 | **992.7** | 1434.8 |
| `c1a_full` | 1991.4 | **1182.0** | 1198.2 | 1754.5 |
| `c1a_reindex` | 1923.4 | **1168.3** | 1169.3 | 1744.7 |
| `c1a_reuse` | 1747.0 | 1005.5 | **999.2** | 1451.6 |
| `moe` | 9766.9 | **2113.4** | 7344.6 | 3961.0 |
| `engram` | 1233.9 | **1122.8** | 1189.2 | 1477.5 |
| `layers_19_21` | 19042.3 | **9821.0** | 17187.3 | 19465.3 |
| **pypto_lib_dsl** | | | | |
| `swa` | 1357.8 | **1048.8** | 1050.6 | 1498.0 |
| `c2a_full` | 2514.6 | 1573.4 | **1567.4** | 2069.7 |
| `c2a_reuse` | 1748.1 | 1448.3 | **1436.1** | 1841.0 |
| `c1a_full` | 2219.2 | 1625.0 | **1571.2** | 1980.9 |
| `c1a_reindex` | 1835.8 | **1576.5** | 1581.4 | 1986.4 |
| `c1a_reuse` | 1706.7 | 1502.0 | **1463.8** | 1865.6 |
| `moe` | 3930.3 | 1593.2 | **1450.7** | 2090.0 |
| `engram` | 1224.8 | **1099.6** | 1201.1 | 1460.6 |
| `layers_19_21` | 11635.6 | 8869.3 | **8744.5** | 12223.2 |

All eighteen improve, by 9.0% to 78.4%, with the largest gains where a lane had the
most to withhold: `moe` 9766.9 -> 2113.4 us and `layers_19_21` 19042.3 -> 9821.0 on
`pypto_lib`.

Against the resident scheduler the change moves the verdict, not the margin: `base` won
three of nine on `pypto_lib` and six of nine on the DSL flavour; `one` wins **eighteen
of eighteen**, by 1.22x to 1.98x. All seventy-two runs passed on both ranks.

What a range costs is availability. A lane that draws four positions is committed to
them: if none is ready it waits on one while the other three go unrun, and no idle lane
can take them. On a graph whose wall is dependency-bound that is what bounds the wall,
and the ticket atomics the range saves do not come close to paying for it.

## Why the threshold could not be tuned instead

The rule being removed read "a lane whose last kernel ran at least 3 us draws one
position at a time". Lowering it to 1 us is the `3 us -> 1 us` column, and it is
**flavour-dependent**: on the DSL cases it lands within a few percent of one-per-draw,
and 9% ahead of it on `moe`, while on `pypto_lib` it leaves most of the win behind —
`swa` +50%, `c2a_full` +53%, `layers_19_21` +75% and `moe` +248% against the
one-per-draw arm. `c2a_full` under that threshold is visibly bimodal, `[1271.5 .. 2017.9]`, which is what a threshold sitting
inside the distribution looks like.

The same operators built two ways want different thresholds, so no constant works. The
predictor is the problem, not its value: how long the *previous* kernel ran says little
about whether the *next* positions are ready, and a sub-microsecond kernel stalls on an
unready producer exactly as a long one does.

## The probe that faked a 45% win

Before the claim range was suspected, a pure spin — no memory traffic — was inserted
after each kernel to separate a timing effect from the swimlane writes. At 1 us it took
`c1a_reuse` from 1833 us to 1017 us, and the dose response was a cliff: 100 ns and
300 ns did nothing, 1000 ns gave the whole win, 3000 ns gave no more. Moving the spin
around the lane loop produced a position map that read like a real memory effect.

It was not. The loop computed the long-kernel classification *after* the kernel returned,
from `get_sys_cnt_aicore() - kernel_start`. Every position where the spin helped lies
inside that interval; every position where it did nothing lies outside it. The spin was
not separating anything — it was inflating the lane's own measurement of its kernel and
flipping the draw from four positions to one.

| spin position | inside `now - kernel_start`? | p50 |
| ------------- | ---------------------------- | --- |
| before the seal | yes | 1017.4 |
| after the seal | yes | 1017.8 |
| after the draw | no | 1862.7 |
| before the cache invalidate | no | 1858.1 |
| after the cache invalidate | no | 1848.1 |
| before the kernel | no | 1857.8 |

Two further probes placed after the classification — one before the completion publish,
one after it — confirmed it: 1849.5 and 1859.3 against a 1834.6 baseline, no effect
either way. Forcing one position per draw with no spin anywhere then reproduced the win
outright at 1001.9, and forcing four reproduced the loss at 2049.6.

Three explanations were argued for that 1 us before the right one, and all three are
wrong: lane de-synchronisation (a uniform per-lane delay does not change phase); a cold
first memory access after a kernel (the spin helps equally on both sides of the seal's
load, which excludes that load, and `OUT_OF_ORDER_STORE_BARRIER()` is already
`dsb(DSB_DDR)` immediately after the kernel); and contention on the ticket or completion
atomic (both probes above sit upstream of those atomics and do nothing).

**The transferable lesson:** a probe inserted upstream of a timestamp the code itself
takes does not observe the system, it steers it. Before believing a probe's result,
work out what the probe is inside of.

## What this overrides, and what was not re-measured

The two constants removed carried their own recorded measurements, taken on workloads
this investigation did not run:

- `PERCOUNTER_CLAIM_BATCH = 4` was chosen as the worst-case-best across the `mixed_*`
  DAGs and `paged_attention` Case1, where **B=1 was 3.11x off** each shape's own best
  (B=2 1.76x, B=4 1.39x, B=8 1.96x). The constant stays at 4 because it also blocks the
  order list across ticket groups, but it no longer sizes a draw.
- `PERCOUNTER_LONG_KERNEL_CYCLES = 3 us` was measured on qwen3 decode at
  **29.60 -> 23.02 ms** device wall.
- `PERCOUNTER_SHORT_LIST_PER_LANE = 64` made a lane's first draw a single position on a
  short list: `paged_attention_unroll` Case2 -18%, `manual_scope` Case2 -10%,
  `batch_paged_attention` Case1 -5%; applied to every list it cost `paged_attention`
  B64 +5% and Case1 +1.6%.

**None of those shapes was re-run here.** If one position per draw costs them what the
first bullet says, the answer is not a constant at all but a batch that reacts to what
the lane observes — shrink to one when a drawn range turns out to hold no ready
position, grow again when every draw finds work. That has a feedback signal instead of a
time constant, and on these eighteen cases it should converge to one.

## Resident's prefix: 239 ms in 38 stalls (open)

Separately, the prefix phase — the preceding layers, 80042 executions over 10622 tasks,
an identical token multiset on both schedulers — runs 144.3 ms on percounter and
350.6 ms on the resident scheduler, 2.43x, reproduced at level 0 over six rounds per arm.

The swimlane span matches the device wall on both (144.0 / 350.0), so the gap is inside
the execution window. It is not throughput: total busy core-time is 1368.6 ms against
1494.6 ms, 9%. It is idleness — the resident scheduler spends **76.2%** of its span with
all 96 cores idle, against 11.3% for percounter, and 91.6% of that idle time is **38
discrete stalls over 100 us**, totalling 244 ms, the largest 28.9 ms. Measured over the
same nineteen task transitions, the resident scheduler spends 239.3 ms where percounter
spends 1.0 ms; on true successor edges it is 25 ms against 5.8 us.

Every stall brackets a single-block task, and the task that starts after it carries a
*lower* `dispatch_generation` than the one that finished before — it was staged early on
a gated core and waited there. That points at the speculative early-dispatch path.

**Do not read this as a percounter win.** Subtracting the stalls leaves the resident
scheduler at 110.7 ms against percounter's 131.4 ms, because it extracts more
parallelism while it runs (17.96 cores on average against 10.72). That subtraction is
not sound either: the high concurrency may be the stalls' own doing, work piling up while
nothing runs and releasing in a burst. Which scheduler schedules this graph better is
**not answered**. Settling it needs the stall explained first, and a level-2 run on the
resident scheduler would show whether its AICPU is dispatching during those 25 ms or
stuck elsewhere.

## Six explanations that did not survive

**1. Resident's speculative early dispatch, on the cases.** Zero `early_dispatch` phase
records across all nine `pypto_lib` cases at swimlane level 3; every block went through
ordinary `dispatch`. Candidacy requires *every* producer to carry
`allow_early_resolve()` (`orchestrator.cpp`), which this callable does not set. It does
appear to be involved in the prefix stalls above, which is a different question.

**2. Topology.** `swa` is 25 tasks, 29 edges, 599 blocks, depth 14; `c1a_reuse` is 24,
29, 551, depth 13 — the same shape to within a few percent, and they sat at opposite
ends of the original spread.

**3. Task granularity.** The DSL flavour decomposes the same work far more coarsely
(`moe` 479 tasks to 32, `layers_19_21` 1527 to 159), predicting that both coarse-flavour
winners flip. `moe` flipped, `layers_19_21` did not, and three other cases flipped the
opposite way.

**4. Per-block argument materialisation.** percounter calls
`scheduler_materialize_task_payload_resolved` once per block with neither
`shared_arguments` nor `write_arguments`; the resident scheduler parses once per task
behind an `args_observed` latch. Giving percounter the same per-lane cache moved five
cases by -3.0% to +3.9%, one of them slower.

**5. The whole-data-cache invalidate per dispatch.** Both runtimes do it — percounter in
`scheduler_observe_data_cache`, the resident scheduler as
`dcci(exec_payload, ENTIRE_DATA_CACHE)` — so it cannot be what separates them.

**6. A resident-side utilisation threshold.** Core utilisation split the nine
`pypto_lib` cases perfectly and scored five of nine out of sample on the DSL cases. It
was fitted on the data it explained. Over all eighteen points the association survives
as a trend (Spearman -0.789), not a per-case predictor.

The seventh was the claim range itself, and it was dismissed because it was tested in one
direction only: raising the long-kernel threshold out of reach made every case slower,
which was read as "the switch is already paying for itself". The opposite direction —
taking the switch's branch always — was the answer and went untried.
**A knob that gets worse when you turn it one way is not thereby at its optimum.**

## Method notes

`pip install --no-build-isolation -e .` does **not** rebuild percounter: its
`build_config.py` sets `BUILD_BY_DEFAULT = False`, and the installer neither builds it
nor says it skipped it. Two complete nine-case sweeps were attributed to a binary that
did not contain the code under test. Build with
`python simpler_setup/build_runtimes.py --platforms a5 --runtime percounter` after
dropping `build/cache/a5/onboard/percounter`, and record the object's hash on both sides
of every sweep.

A `comm_init failed` / `507033` at `ChipWorker` init aborts a run before any kernel
executes. It occurred once in seventy-two runs on a shared box and passed on re-run; it
is not a scheduler signal.

Separate a run's device-wall spans **by phase position, never by magnitude**. They come
out as one reference followed by a (prefix, case) pair per round, and on `swa`, `moe`,
`engram` and the DSL `c2a_full` the two phases are the same size, so a magnitude split
mixes them without saying so. An earlier draft of this entry did exactly that and
reported `swa` at 100 us and `moe` at 937 us; both were prefix spans, and the error
turned `swa`'s 33% gain into an apparent 4% loss, which then read as the one case the
change did not help. `layers_19_21`, whose phases differ by 6x, comes out the same
either way and is the control that shows the positional split is right.

The entry it replaced already carried "a device-wall median mixes the prefix and the
case" as its own method note, so the warning was present and still not enough: a note
that says *separate them* does not stop you reaching for the separator that is wrong.
