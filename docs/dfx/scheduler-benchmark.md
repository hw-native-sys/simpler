# Scheduler Benchmark

How long one run took, and where the time went — from a chip-swimlane capture
and the `deps.json` of the same run. The numbers read the dependency graph, the
run's kernel durations and the cores, and two of them the run's own placement
and order, but never a scheduler's rules. So runs of any scheduler on either
runtime (`host_build_graph`, `tensormap_and_ringbuffer`) are measured by one
rule and can be compared directly.

```text
measured ── dispatch cost ──> squeezed ── schedule cost ──> best  ...  bound
```

```bash
python -m simpler_setup.tools.scheduler_benchmark <chip_swimlane_records.json> --deps <deps.json>
```

## 1. Inputs

Run the case with a level-1 chip swimlane and dep_gen, for example:

```bash
python <test_file.py> -p a5 -d <device> --enable-chip-swimlane 1 --enable-dep-gen
```

Its output directory holds both inputs. The capture supplies, per block, the
core it ran on and its start and end ([Chip Swimlane Profiling](chip-swimlane-profiling.md));
`deps.json` supplies the tasks and their edges ([dep_gen](dep-gen.md)). Level 1
is enough: higher levels add scheduler records this tool does not read.

`deps.json`'s `runtime` names the task-id layout, and both runtimes are read the
same way. A capture that holds several runs (`run_epoch`) is reported run by
run, each against the same `deps.json`.

Both inputs must be in the current format, and anything else is refused with
what is missing: every AICore record has its seven columns, `run_epoch`
included; the capture's `metadata.runtime` matches `deps.json`'s; and every
task, Graph body tasks included, carries `kernel_ids`, `block_num`,
`sync_start` and `predicated`. `graph_definitions` and `graph_shells` appear only when the run
replayed a Graph.

**The capture must be complete.** A capture that carries `metadata.collection`
must report verdict `published`, `processing_complete` and
`handoff_identity_verified` true, and no loss count above zero; any other is
refused, since lost records would read as work that never ran. Only the
collection path that retains runs writes `metadata.collection`; a capture from
the default path carries no such report and is taken as complete.

**A predicated task without rows counts as skipped.** A task whose dispatch
predicate fails is retired without reaching a core, so it leaves no AICore row.
`deps.json` records that a task has a predicate, not how it evaluated, and a
level-1 capture records no skip. So a predicated task with no row at all counts
as skipped: it holds no core and ends the moment it is ready, and the report
says how many tasks counted so. A task that left rows for only some of its
blocks, and a task without a predicate that left none, are refused as
incomplete.

## 2. The numbers

| Name | What it is |
| ---- | ---------- |
| **`measured`** | the run, from its first AICore start to its last end |
| **`squeezed`** | the same cores running the same blocks in the same order, every gap removed |
| **`best`** | a near-optimal order and placement, found rather than proven |
| **`bound`** | no schedule of this DAG with these durations on these cores can be shorter |

`squeezed` keeps every decision the run made — which core ran each block, and in
which order each core ran its blocks — and removes only the waiting. `best` and
`bound` bracket the optimum rather than name it: the shortest schedule on a
fixed core count is NP-hard to find.

> **The shortest possible schedule is in `[bound, best]`.** A narrow bracket
> means the workload's scheduling problem is settled; a wide one means it is not.

The two steps split the time a run spent above `best`:

| Step | Answers |
| ---- | ------- |
| `measured − squeezed` | **dispatch cost** — what reacting slowly cost, with every decision held |
| `squeezed − best` | **schedule cost** — what the cores it chose and the order it ran cost |

Two schedulers with the same `measured` can have nothing else in common: one may
dispatch in a microsecond and place badly, the other the reverse. The step a run
loses its time in says which.

Two references stand beside them; neither is subtracted from anything:

| Reference | Answers |
| --------- | ------- |
| **`fifo`** | first-in-first-out by ready time. It reads the DAG and the run's kernel durations and none of the run's decisions, so two runs of one workload differ in `fifo` only as far as their kernel durations do |
| **`sameOrder`** | this run's start order, on any free core: how much of the schedule cost was the cores chosen rather than the order |

`sameOrder` can land on either side of `squeezed`: greedy list scheduling has
known anomalies, and freeing the cores can lose as well as win.

## 3. Reading the report

An a5 `host_build_graph` batch paged-attention run:

```bash
python -m simpler_setup.tools.scheduler_benchmark <records> --deps <deps> --label batch_pa_case2
```

```text
== batch_pa_case2 run 1 (host_build_graph; 2,048 units on 32 clusters)
  measured        5,533.1 us   what the run took
                  1,175.8 us     - dispatch cost
  squeezed        4,357.3 us   the same cores in the same order, every gap removed
                    531.4 us     - schedule cost
  best            3,825.9 us   a near-optimal order and placement
  bound           3,825.5 us   no schedule of this DAG can be shorter
                               the shortest schedule is in [bound, best]

  fifo            4,332.3 us   first-in-first-out by ready time; none of this run's decisions
  sameOrder       4,346.0 us   this run's start order, any free core
```

`best` is within 0.01% of `bound`, so this workload's optimum is known: 3.83 ms.
Of the 1.71 ms the run spent above it, 1.18 ms was dispatch and 0.53 ms the
decisions it made. `fifo` and `sameOrder` sit next to `squeezed`: the order it
ran in was no better than first-in-first-out, and the graph has one long chain
that both delayed. A run with predicated tasks that left no row gets a line
under the header saying how many counted as skipped (§1).

**`schedule cost` is not always the policy's.** It measures the decisions a run
made. A scheduler limited by its own dispatch rate keeps few blocks in flight,
finds the same few cores idle every time, and piles work onto them; that costs
it nothing while dispatch is the limit, but `squeezed` keeps the pile-up and
serializes it. One a5 paged-attention run kept its AIC cores busy 3.3% of the
run and its AIV cores 1.6% (each type's kernel time from the capture's rows,
over its cores times `measured`), and read `measured` 53,266.8 us, `squeezed`
5,330.9 us and `best` 1,736.1 us. Most of that 3.6 ms `schedule cost` is the
dispatch rate showing through the idle cores, not an order or a placement the
policy chose. Read `schedule cost` as policy only when the run kept a fair share
of its cores busy; a run whose `dispatch cost` dominates is judged by that first.

## 4. How the numbers are computed

Every block of a task is a *unit*, except that a MIX block and all the blocks of
a sync-start task are each one unit that starts as a whole. A schedule gives
each unit one start and its cores, and every schedule passes one validator
before it is reported: each unit runs once, no consumer starts before all of its
producers end, no core runs two units at once, a MIX block's slots share one
cluster, and a sync-start task's blocks share one start. Kernel durations are
the captured ones throughout. A task without a kernel, or a predicated task
that left no row (§1), holds no core and ends the moment its producers have
ended; a skipped task the runtime still sent to a core keeps its brief row,
like any other.

**`squeezed`** keeps each core's captured sequence: every unit starts the moment
its producers have ended and the unit before it on each of its cores has ended,
a longest-path pass over that constraint graph. It is never longer than
`measured`, except where the capture started a MIX block's slots or a
sync-start task's blocks apart: the model starts them together, and
`dispatch cost` can then read slightly negative.

**`fifo`, `sameOrder` and `best`** are zero-overhead list schedules: at every
event the ready unit with the smallest priority that fits starts at once, on the
lowest-numbered free cores of its type (for MIX, the lanes its mask names, on
one cluster). The priority
is the ready time for `fifo`, the captured start for `sameOrder`, and for
`best`:

- **By longest remaining chain.** Each task carries its b-level, the longest
  chain of durations from it to the end of the graph, its own included. A task
  that takes a microsecond but gates a 20 us chain delays everything behind it;
  one with nothing after it can wait.
- **Then forward-backward improvement**, up to 16 rounds. A round schedules the
  DAG with every edge reversed, the units that ended latest first; mirrors that
  into a forward priority; and schedules forward again. A round is kept only if
  it is shorter, and rounds stop at the first that is not.
- **Then the shortest of all.** The search can lose to a schedule it did not
  try, so `best` is the shortest of its result, `squeezed`, `fifo` and
  `sameOrder`, and `schedule cost` never reads negative.

**`bound`** schedules nothing. Two numbers per task come from the DAG alone —
`head`, the longest path of durations before it, and `tail`, the longest chain
of durations after it — and hold under any schedule. For every unit set with
`head >= h` and `tail >= q`, on the cores of one type:

```text
bound >= h + ceil(their work / cores of that type) + q
```

since none of them starts before `h`, their work has to fit on those cores, and
whichever ends last still has `q` to go. The tool tries every `q` and every
distinct `h` (when there are 128 or more, an evenly spaced sample of 64 to 96),
and takes the largest result or the critical path, whichever is longer.

Worked case — two cores, `A(2) -> B,C,D,E (4 each) -> F(3)`:

| Argument | Value |
| -------- | ----: |
| critical path, `A -> B -> F` | 9 |
| all the work over two cores, `21 / 2` | 11 |
| `{B,C,D,E}`: `head 2 + 16 / 2 + tail 3` | **13** |

13 is also achievable, so it is the optimum, and only the three-part argument
reaches it.

**Barrier segments.** A kernel-less task that every core-holding task either
precedes or follows is a barrier: nothing runs beside it, so the segments it
cuts the graph into never overlap, and `bound` is the sum of the segments'
bounds. A Graph task's entry and exit are barriers whenever consecutive Graph
tasks chain. On a decode graph that replays one layer 40 times, the naive
`max(critical path, work / cores)` falls 14% short of `bound`: it assumes the
40 layers can run mixed together, and they cannot.

## 5. Known limits

- **One capture, one round.** A single run carries no interval; compare runs of
  one workload, captured the same way.
- **Durations are fixed.** Kernels are replayed at their captured durations, so
  a schedule that raises concurrency does not slow bandwidth-bound kernels the
  way the device would. On an a5 decode graph, operators ran 5–9% longer
  (median over operators) in the busiest third of the run than in the quietest;
  on paged attention, no measurable difference.
- **Any free core can take the next unit.** `best`, `fifo` and `sameOrder` assume
  that. A mechanism that holds resources — `host_build_graph`'s early dispatch,
  which stages a block on a core before its producers end, or
  `tensormap_and_ringbuffer`'s bounded task ring and heap — can make them
  unreachable for that scheduler; `bound` stays valid.
- **On `tensormap_and_ringbuffer`, `dispatch cost` includes orchestration.** The
  orchestrator runs on the device beside the scheduler, so a task may not be
  submitted until well into the run, while every number here assumes the whole
  graph is known from the start.
- **`best` and `bound` describe the graph a run executed.** The two runtimes can
  record one workload as different graphs — the barriers around a Graph task
  exist only on `host_build_graph` — so compare runs on different runtimes by
  their steps, not by their `bound`.
- **Observation is not free.** A level-1 capture slows the run a little; the
  numbers describe the profiled run.
- **A lost record can read as a skip.** A capture from the default collection
  path does not report lost AICore records, so a predicated task whose every
  record was lost counts as skipped. A loss that takes every row of a task
  without a predicate, or only some of one task's blocks, is refused as
  incomplete.
- **Large graphs take seconds.** Finding the barriers walks the whole graph once
  per kernel-less task. An a5 decode graph of 11,000 tasks, about 500 of them
  kernel-less, takes 13 s in all; a paged-attention graph of 65,000 tasks, none
  kernel-less, 25 s. A graph with thousands of kernel-less tasks takes longer.
- **The core numbering is the runtime's.** Core ids index `metadata.core_types`
  with every AIC worker first, then each cluster's AIV workers in turn, as the
  schedulers number them. A capture whose AIC workers do not all come first is
  refused; which cluster an AIV worker belongs to cannot be read from
  `core_types`, and is taken from that numbering.
