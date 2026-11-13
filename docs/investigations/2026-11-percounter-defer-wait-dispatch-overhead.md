# percounter defer_wait: where the per-dispatch overhead is, and what did not fix it

The measurements below describe earlier revisions. The current percounter
implementation has been rebased and functionally validated; refreshed DSV4.1
performance acceptance is deferred to follow-up work. These historical figures
are not a performance claim for the current PR head.

The historical COUNTER `defer_wait` support on `feat/percounter-defer-wait` costs a percounter
graph about 6% even when no kernel registers anything. This records what the cost was
measured to be, which parts were attributed, and the attempts that did not reduce it,
so the next person does not repeat them.

## Measurement

`tests/st/a5/host_build_graph/multi_core_dag` case `mixed_multi_root_4096` — 4096 tasks,
no registration anywhere — run with `runtime="percounter"`, 15 rounds, first dropped,
each phase inside its own `task-submit` device lock on a5. The number is the runtime's
own `chip.run.runner_run.device_wall` span.

| build | device wall (mean) | range | `aicore_kernel.o` |
| ----- | ------------------ | ----- | ----------------- |
| base (`616f42f04`, none of the code present) | 256.2 us | [252.3, 259.4] | 1,336,848 |
| branch, every hot-path addition short-circuited | 260.8 us | [254.9, 266.5] | 1,370,280 |
| branch | 274.0 us | [271.1, 277.2] | 1,718,824 |
| branch, measured again afterwards | 270.5 us | [266.4, 274.0] | 1,718,824 |

The base and branch ranges do not overlap. Drift between the two measurements of the
same branch binary is 1.7%, so the 5.6% mean gap is not drift.

## What the cost is made of

This decomposition uses the **second branch measurement, 270.5 us**, against
the same 256.2 us base. Each component below uses the base as denominator.

- **Code present but not executed: +1.8%** (260.8 against 256.2). The short-circuited
  build keeps every added function, type and include and skips only their bodies.
- **Code executed: +3.8 percentage points** ((270.5 − 260.8) / 256.2).
  Together with the 1.8 points above, the total is 5.6%.
- Of the executed part, roughly a third of the *total* gap is the seal's read of
  `record->count` / `record->error_code`: a separate build that skips that read alone
  measured 265.6 us against the branch's 269.3 us in the same session. The read is a
  cold GM load on every dispatch, because `percounter_observe_data_cache` invalidates the
  whole data cache at the top of `percounter_run_task`, so the record's line is never
  resident when the seal reaches it.

That read cannot be removed without knowing before the kernel runs whether it will
register, which is not visible at bind. Making it visible means a mandatory user
declaration, which the design rules out. The other route is the whole-cache invalidate
itself, which is a separate pre-existing cost and not specific to this feature.

## The include this measured, and why it went anyway

**Dropping `scheduler_async_control.h`.** The branch reached for one predicate,
`SchedulerAsyncCompletionObserver`, by including a header that also carries the
resident scheduler's leases, directory and resolver. Measured as a size question
it is not worth doing: replacing the predicate with a fifteen-line local COUNTER
poll took `aicore_kernel.o` from 1,718,824 to 1,691,208 — 27 KiB of the 382 KiB
the branch adds, so the include is not where the size went, and 27 KiB does not
pay for a second copy of a predicate.

Size was the wrong question to ask of it. What the include costs is a dependency:
it was the only edge from the percounter AICore translation unit into the resident
scheduler's *control* code, where every other edge that unit has reaches a layout
or wire-format header. The poll now lives in `scheduler/async_completion.h`,
beside the GM accesses `gm.h` already keeps for the same reason — a lane reaches
it from every checkpoint, where the sibling reaches it from a scheduler lane that
is already carrying phase instrumentation.

What is **not** copied is the formats. `CompletionToken`, the `COMPLETION_TYPE_*`
constants, `AsyncGroupData` and `DeferredCompletionEntry` are still included from
host_build_graph, because a backend writes those records and both runtimes read
them; a fork there would be a layout fork. The drift this does accept is narrower
and worth naming: if the meaning of a COUNTER, an SDMA event record or a CCU event
changes, two polls now have to follow it instead of one.

## Two things that did not work

**Marking the cold bodies `noinline`.** `percounter_async_seal`, `..._settle`,
`..._tick_pending` and the slow half of `..._acquire` are reached from several sites,
so letting each expand everywhere looked like the obvious source of the 382 KiB.
Marking them `__attribute__((noinline))` took the object the other way, from 1,718,824
to **2,680,784**: these are `inline` functions in a header, and refusing to inline them
leaves an out-of-line copy per translation unit where inlining had collapsed them.

**Splitting the tick into an inline test and an out-of-line walk.** Keeps the hot path
to one local word, which is the right shape, but measured no size or time difference on
its own once `noinline` was off. Not kept, because it adds a function for nothing.

## Where the remaining size goes

Unknown. The branch adds roughly 250 lines of AICore source and 382 KiB of object,
and neither the include nor inlining of the four async functions accounts for it.
`aicore_kernel.o` is built from several translation units; the next step would be
per-TU sizes or a symbol-level diff, which was not done.

## When to re-open

- If the whole-cache invalidate per dispatch is replaced by a payload-line invalidate
  (the separate optimisation already recorded against percounter), re-measure: the
  seal's read becomes a hit and roughly a third of this gap should disappear on its own.
- If a per-task "may register" declaration ever becomes available for another reason,
  the seal's read and the record reset both become conditional and this entry's whole
  subject goes away. Do not introduce such a declaration *for* this.
