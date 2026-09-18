# Porting a2a3's full AICore retirement protocol to A5

**Date**: 2026-09-17
**Verdict**: dropped. The part of the a2a3 protocol worth porting is batched
EXIT + a shared deadline, proposed in #2288; A5's `shutdown()` itself still
retires one core at a time. The next step of the a2a3 protocol makes
`shutdown()` faster in isolation and buys no net time; the remaining steps have
nothing on A5 to act on.

## Question

a2a3 retires a scheduler's cores through `retire_cores()` →
`platform_retire_aicore_group()`: claim each core, broadcast EXIT, sweep ACKs
round-robin, close each window (IDLE, `FAST_PATH_ENABLE=CLOSE`, readback), drain
once, then release a GM return gate per core. A5's
`SchedulerContext::shutdown()` signals, waits for and restores one core at a
time. Since a2a3 already paid for a full group protocol, porting it looks like
an obvious way to shorten A5's exit — and it is the change a reader of the two
trees reaches for first.

## What was tried

Three of the a2a3 steps have an A5 counterpart, and the first of those is not
needed on the normal path; the rest act on hardware state A5 does not have:

| a2a3 step | A5 | result |
| --------- | -- | ------ |
| `core_retired_` atomic claim | normal path does not need it: each scheduler owns a disjoint core set | correctness topic, not measured |
| broadcast EXIT + shared deadline | implementable | "batched" below, proposed in #2288 |
| round-robin ACK sweep, deferred IDLE, post-broadcast `wmb` | implementable | "ExitSync" below |
| IDLE then `FAST_PATH_ENABLE=CLOSE` | absent: A5 init only writes `DATA_MAIN_BASE=IDLE` | nothing to port |
| CLOSE readback + drain | absent: A5 has no CLOSE | nothing to port |
| GM `post_close_release` return gate | absent: an A5 worker ACKs and exits on EXIT | nothing to port |

So the fullest A5 port is ExitSync: batched EXIT plus the sweep, the deferred
IDLE reset and the `wmb`. It differs from #2288 in `scheduler_cold_path.cpp`
only; the `platform_regs` split is shared, so the ACK-wait strategy is the one
variable.

Setup: nine DeepSeek-V4-Pro graphs (pypto-lib #873), A5 (`Ascend950PR`) on a
development card, frozen base `e63903c3`. Each block runs
`per-core baseline → batched → ExitSync → per-core baseline` with the two
candidates swapped in the second of three blocks; 6 warmup + 20 measured calls
per process; 27 paired blocks, 135 processes, all passing. All arms carry the
same fixed `AicpuPhase` stamps (#2287's `Shutdown`, plus the broadcast loop), so
stamping cost cancels in each difference. The two candidates sit in the same
bracket and are differenced directly, not through two population medians.

The judging segment is `window_to_gb_end` (`graph_build` end −
`max(orch_end, sched_end)`). Injecting known delay at `shutdown()` entry gives
it a transfer slope of **1.000** to `device_wall` — a microsecond saved in the
tail is a microsecond off the run — so a real saving should reach it in full.

## Result

Medians, µs:

| segment | per-core | batched | ExitSync |
| ------- | -------- | ------- | -------- |
| `shutdown` | 13.759 | 9.857 | **7.885** |
| of which EXIT broadcast | — | 2.067 | 2.035 |
| `shutdown` end → `graph_build` end | 2.209 | 2.361 | **4.702** |
| `window_to_gb_end` | 15.245 | 11.656 | 11.922 |
| `device_wall` | 201.420 | 198.370 | 198.816 |

Transfer from the inner segment to the outer one:

| vs per-core | `shutdown` gain | `window_to_gb_end` gain | transfer |
| ----------- | --------------- | ----------------------- | -------- |
| batched | +3.907 (27/27) | +3.790 (27/27) | **97%** |
| ExitSync | +5.908 (27/27) | +3.497 (27/27) | **59%** |

ExitSync against batched, same block (positive = ExitSync faster):

| segment | median | ExitSync faster | range |
| ------- | ------ | --------------- | ----- |
| `shutdown` | +1.945 | 27/27 | [+1.582, +2.635] |
| `shutdown` end → `graph_build` end | **−2.355** | **0/27** | [−3.186, −1.529] |
| `window_to_gb_end` | −0.306 | 8/27 | [−1.248, +0.386] |
| `device_wall` | −0.979 | 9/27 | [−21.886, +7.589] |

The `shutdown` saving reappears, entirely, in the segment after it, where
ExitSync is slower in every block (27/27). The net differences sit below
`window_to_gb_end`'s ~1 µs visibility floor (its run-to-run noise is ~0.24 µs)
and a repeat round measured −0.108 (10/27), so the accurate reading is **no
measurable net gain**, not a slowdown.

A diagnostic round added a span around the last thread's `runtime_destroy()`
and a COND-read counter, and ruled out the two obvious mechanisms:

| metric | per-core | batched | ExitSync |
| ------ | -------- | ------- | -------- |
| `runtime_destroy` | 0.659 | 0.707 | 0.657 |
| COND reads per thread | 41 | 20 | 20 |

- The cost does not land in `runtime_destroy` (0.66–0.71 µs, ExitSync not
  slower, 17/27), so "an ACK is not a finished exit" does not explain it.
- The sweep does not read more: both candidates issue **exactly 20** COND reads
  (no block differs).

What remains in that segment is a `LOG_INFO` and one atomic increment. The
explanation consistent with the data is thread dispersion: the segment is
`max(graph_build end) − max(shutdown end)` across threads, and it grows when the
thread that leaves `shutdown` last is not the one that finishes last. That is an
inference — the fixed-phase buffer reduces across threads, so per-thread end
times would be needed to confirm it.

The same data also puts a price on the ACK wait: batched EXIT cuts COND reads
from 41 to 20 per thread (7 cores each), and the ~7.7 µs wait comes to
~0.39 µs per read amortised — a poll plus the wait on the AICore, not the raw
LDR cost in [`mmio-performance.md`](../hardware/mmio-performance.md). The floor
is one read per core, so ~5 µs per call is left, reachable only through fewer
polls per core or a non-COND ACK channel; the latter is the question
[2026-06-cond-vs-gm-notification](2026-06-cond-vs-gm-notification.md) already
deferred.

## Why not (now)

Porting further adds a per-thread ACK array, a second pass over the cores and a
barrier to the exit path, and returns nothing measurable: the one saving it
makes is given back in full in the next segment (slower in 27/27). The remaining
a2a3 steps are tied to hardware state — the FAST_PATH window and its GM return
gate — that A5 does not have, so they are not "not yet ported" but have no
target.

## When to reconsider

- **An A5 abnormal-exit path needs per-core budgets.** #2288 trades each core's
  own timeout for one group deadline, which the normal path never reaches
  (~20 polls against a 1 s budget). a2a3's sweep exists so that "a core is only
  abandoned once the deadline passes with it still silent". If fault injection
  on A5 shows late cores starved by an early one, the sweep comes back — as a
  correctness change, judged on recovery behaviour, not on this benchmark.
  #1710 argues the other side for fatal shutdown, where every core may be
  dead: one group deadline caps that teardown at a single timeout, where the
  per-core loop pays one per core.
- **The dispersion explanation is confirmed and can be avoided.** A variant that
  keeps ExitSync's shorter `shutdown` without widening the gap after it would
  reopen the question.
- **A5 gains a window-close or GM completion mechanism** comparable to a2a3's,
  giving the remaining steps something to act on.
