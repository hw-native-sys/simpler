# percounter against the resident scheduler on DSV4.1: seven explanations that did not survive

`tests/fdwic/dsv41_scheduler` runs real DeepSeek-V4.1 regions on two A5 devices with
the original kernel binaries. Across its eighteen cases percounter is sometimes twice
as fast as the resident scheduler and sometimes a third slower, and which way it goes
is **not explained by any of the seven mechanisms tried here**. This records the
measurements, so the next person spends their hardware time on something else.

## What was measured

One isolated case per run at profiling level 0, one warmup and five rounds, three arms
interleaved — percounter, resident, percounter again — with the two percounter arms'
difference taken as the drift floor of a shared machine. Ratios below are
`resident / percounter` device wall, so **above 1 means percounter is faster**.

The runner executes three callables per iteration: a reference, then the *prefix* that
produces the slice's real inputs, then the case. Only the case is the measurement —
the prefix is setup, it runs at level 0 regardless, and the README states its timing is
never part of a case's scheduling numbers. Its cost tracks the layer index (0.9 ms for
layer 0, 156 ms for layer 24), so mixing it into a comparison swamps everything else.

percounter ran from `aicore_kernel.o` md5 `a1e2c917`, 1,718,824 bytes, verified
unchanged before and after every sweep.

| case | pypto_lib | drift | DSL | drift |
| ---- | --------- | ----- | --- | ----- |
| `swa` | **2.11** | 2.1% | **2.04** | 0.8% |
| `engram` | 1.18 | 1.4% | 1.20 | 0.4% |
| `moe` | 1.20 | 0.0% | **0.73** | 2.4% |
| `layers_19_21` | 1.17 | 2.2% | 1.05 | 1.4% |
| `c1a_reindex` | 0.92 | 2.1% | 1.09 | 3.0% |
| `c1a_reuse` | 0.79 | 0.1% | 1.05 | 5.8% |
| `c2a_reuse` | 0.81 | 7.1% | 1.05 | 1.8% |
| `c1a_full` | 0.85 | 0.1% | 0.87 | 4.1% |
| `c2a_full` | 0.81 | 1.2% | 0.77 | 1.1% |

All eighteen pass on both schedulers. `engram` is the same workload in both flavours
(12 tasks / 960 executions) and measured 1.18 and 1.20 in two independent sweeps, which
is the method's own control.

## The seven

**1. Resident's speculative early dispatch.** Resident keeps a running and a pending
slot per core and can stage a task whose producers have published but not finished; the
gated core assembles its own argument vector while it waits, so the producer's
completion costs one doorbell write and no scheduler work. It is the obvious candidate
and it **never fires here**: zero `early_dispatch` phase records across all nine
pypto_lib cases at swimlane level 3, every block through ordinary `dispatch`.
Candidacy requires *every* producer to carry `allow_early_resolve()`
(`orchestrator.cpp`), which this callable does not set.

**2. percounter's long-kernel draw switch.** A lane whose last kernel ran longer than
`PERCOUNTER_LONG_KERNEL_CYCLES` (3 us) draws one position at a time instead of four,
and the losing cases sit on that threshold. Raising the constant out of reach made
every case **slower** — `c1a_reindex` +27%, `c1a_full` +14%, `moe` +10%. The switch is
already paying for itself.

**3. Topology.** `swa` is 25 tasks, 29 edges, 599 blocks, depth 14; `c1a_reuse` is 24,
29, 551, depth 13. They are the same graph shape to within a few percent and their
ratios are 2.11 and 0.79.

**4. Task granularity.** The DSL flavour decomposes the same work far more coarsely
(`moe` 479 tasks to 32, `layers_19_21` 1527 to 159), predicting that both coarse-flavour
winners flip. `moe` flipped, hard — 1.20 to 0.73, the largest deficit measured.
`layers_19_21` did not. Three other cases flipped the *opposite* way.

**5. Per-block argument materialisation.** percounter calls
`scheduler_materialize_task_payload_resolved` once per block with neither
`shared_arguments` nor `write_arguments`, re-parsing a task's payload header and
rewriting its argument vector on the core for every block; resident parses once per task
behind an `args_observed` latch and shares the result. The work is entirely
block-independent, and `c1a_reuse` has 24 tasks across 583 blocks. Giving percounter the
same per-lane cache moved five cases by −3.0% to +3.9% — inside the drift band, one of
them slower.

**6. The whole-data-cache invalidate per dispatch.** Both runtimes do it —
percounter in `scheduler_observe_data_cache`, resident as
`dcci(exec_payload, ENTIRE_DATA_CACHE)` — so it cannot be what separates them. It
remains a real cost for percounter's deferred-completion seal, which is a different
entry's subject.

**7. A resident-side utilisation threshold.** Core utilisation measured from resident's
own swimlane separated the nine pypto_lib cases perfectly: every case at or below 27.1%
had percounter faster, every case at or above 28.2% had it slower, nine of nine, stable
across two rounds. On the nine DSL cases it scored **five of nine**. The threshold was
fitted on the data it explained. What survives is weaker and worth keeping: over all
eighteen points the association is Pearson −0.657, Spearman −0.789, and the four misses
all sit at 31.8–33.6% with ratios of 1.05–1.09, which is a tie at this drift. Utilisation
is a trend, not a per-case predictor.

## What is established

On `c1a_reuse` resident's cores run 16 us kernels with a **median 104 us gap between
consecutive tasks on the same core**. The wall is dependency-bound, not dispatch-bound,
and no per-dispatch cost of the half-microsecond scale that hypotheses 2, 5 and 6 argue
over can account for the 383 us that separates the two schedulers there. That is the
strongest reason to stop looking at dispatch cost.

## The remaining lead

If the wall is spent waiting, what decides it is how quickly a consumer starts once its
producer finishes. The two schedulers differ structurally there — resident's AICPU
observes a FIN and dispatches, percounter's lane spins on a counter and then resolves,
materialises and launches on the core — and **it is the one quantity not measured here**,
because percounter emits no swimlane records at any profiling level.

Measure that first. Everything else in this entry is downstream of it.

## Two method notes

`pip install --no-build-isolation -e .` does **not** rebuild percounter: it sets
`BUILD_BY_DEFAULT = False`, and the installer neither builds it nor says it skipped it.
Two complete nine-case sweeps were attributed to a binary that did not contain the code
under test before the object's size disagreed with an independently recorded number.
Build it with `python simpler_setup/build_runtimes.py --platforms a5 --runtime percounter`
after dropping `build/cache/a5/onboard/percounter`, and record the object's hash with the
measurement.

A median taken over a run's device-wall spans mixes the prefix and the case, which differ
by one to two orders of magnitude. Separate them before reading anything.
