# AICore first-task cold-start: pre-warm dispatch path in `aicore_execute`

**Date**: 2026-06-05
**Verdict**: dropped — per-core long tail is NoC routing latency, not I-cache miss; software warmup buys ~0.3 µs on avg with no movement on the slow-core max

## Question

In `aicore_execute` (`src/a2a3/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp`)
the main poll loop has a "task arrived" branch that, on the first real
task, hits cold I-cache lines for `dcci(exec_payload, ENTIRE_DATA_CACHE)`,
`get_sys_cnt_aicore()`, and `write_reg(RegId::COND, MAKE_ACK_VALUE)`. The
rest of the loop body (the idle-spin path on `read_reg(DATA_MAIN_BASE)`)
is warmed naturally during AICPU init while AICore spins.

Proposal: between Phase 1 handshake exit and the main loop entry, run a
short warmup sequence that touches the same instructions the task-arrived
branch will hit, so the first real task pays no cold-I-cache load on the
dispatch→start critical path.

The warmup cannot call `execute_task` directly because the payload's
`function_bin_addr` may not be deterministic at that point (AICPU's
`memset(payload_per_core_, 0)` at `scheduler_cold_path.cpp:899` happens
*after* `handshake_all_cores` at line 888, so AICore could observe stale
GM contents during the gap). Limit warmup to the three non-payload-data
instructions: `dcci(payload, ENTIRE_DATA_CACHE)` (cache-management op,
data not consumed), `get_sys_cnt_aicore()` (intrinsic, no payload read),
`write_reg(RegId::COND, AICORE_IDLE_VALUE)` (the IDLE value matches the
existing Phase 2 write and does not disturb the AICPU protocol).

## What was tried

Patch in `aicore_executor.cpp`, inserted right before the Phase 4 main
loop:

```cpp
dcci(payload, ENTIRE_DATA_CACHE);
volatile uint64_t warmup_ts = get_sys_cnt_aicore();
(void)warmup_ts;
write_reg(RegId::COND, AICORE_IDLE_VALUE);
```

Workload: `examples/a2a3/tensormap_and_ringbuffer/spmd_serial_chain_mix/`
— 4 chained MIX tasks (AIC + AIV0 + AIV1), `block_num=24`, 72 AICore
cores per task, each task busy-waits 50 µs in `get_sys_cnt()`. Run on
a2a3 onboard via `task-submit --device auto --device-num 1 --run "...
--enable-l2-swimlane 2"`. Compared three runs, all with the
batched-publish dispatch optimization (PR #989) and the eager swimlane
head resolve (also in #989). Head-OH per task = `start_time_us −
dispatch_time_us`, joined per `(core_id, reg_task_id)` from the level-2
`l2_swimlane_records.json`.

## Result

Per-task head-OH (us) across 72 cores:

| Run | t0 min | t0 avg | t0 max | t1 avg | t2 avg | t3 avg |
| --- | -----: | -----: | -----: | -----: | -----: | -----: |
| Lazy head resolve, no warmup | 1.42 | 3.87 | 6.36 | 0.48 | 0.43 | 0.29 |
| Eager head resolve, no warmup | 0.36 | 2.61 | 5.62 | 0.35 | 0.39 | 0.23 |
| Eager + warmup | **0.32** | **2.29** | 5.64 | 0.36 | 0.36 | 0.31 |

Warmup vs eager-only: t0 avg −0.32 µs (12% reduction), t0 min −0.04 µs
(into noise), **t0 max unchanged** (within run-to-run variance).
Steady-state t1–t3 unchanged. t0 max for the slowest cores is essentially
the same as without warmup.

Per-core breakdown for the warmup run reveals the cost is **bimodal**:

```text
Δ = t0 − avg(t1..t3) over 72 cores
min = −0.03 µs  (warmup fully effective; t0 ≈ steady state)
p25 = 0.53 µs
p50 = 1.77 µs
p75 = 3.39 µs
max = 5.31 µs   (slowest cores still pay ~5 µs first-task tail)
```

Bottom-6 cores: t0 ∈ [0.32, 0.34] µs — equal to their own t1–t3 average.
Top-6 cores: t0 ∈ [4.84, 5.64] µs — unchanged from no-warmup runs.

Top-6 slowest cores by t0: 71, 60, 61, 21, 66, 67 — predominantly AIV
cores at higher cluster offsets, no strong correlation with publish-loop
position (the 72 handles are published over ~360 ns total, an order of
magnitude smaller than the 5 µs tail).

## Why not (now)

- **Warmup targets the wrong layer.** start_time is stamped at
  `aicore_executor.cpp` line 146, *before* `execute_task` and the
  kernel binary execution. The cold cost inflating head-OH is *not*
  AICore I-cache miss on the task-arrived branch — those instructions
  are 3 cache lines and warming them via dcci/sys_cnt/write_reg moves
  the avg by only 0.32 µs. The remaining 4–5 µs tail on the slowest
  cores is the NoC + FFTS routing latency from AICPU's
  `write_reg(DATA_MAIN_BASE)` to the core seeing the update in its
  `read_reg` spin — pure hardware propagation delay, not software-
  resolvable.
- **The avg-only win does not justify code on the hot path.** The
  warmup is 3 instructions on every kernel entry, including the
  `l2_swimlane_enabled == false` case where head-OH measurement is not
  even being collected. Trading 3 hot-path instructions for a 0.32 µs
  avg-only win that the slow-core tail (the only thing that matters
  for end-to-end wall time) doesn't see is a poor trade.
- **DFX-path priorities apply.** This is a profiling / cold-start
  observation path; accuracy and simplicity outrank micro-optimization
  (see `.claude/rules/...` DFX guidance: "do not propose
  micro-optimizations on profiling / swimlane / diagnostics paths").

## When to reconsider

- A workload appears whose end-to-end wall time is meaningfully
  affected by the t0 head-OH average (rather than the tail). In our
  measurement the warmup wins 0.32 µs on a single task whose own
  duration is 50 µs — far below any user-visible threshold.
- The slow-core 5 µs tail is shown to be I-cache-driven after all
  (e.g. by a future AICore generation with a different NoC layout
  where the tail collapses but the avg cold-load is still present).
  Re-run the level-2 swimlane and check whether per-core t0 cost
  becomes uniform; if so, warmup becomes worth re-evaluating.
- A separate AICore startup-perf investigation lands and wants a
  central place to attach further warmup-style fixes. This
  investigation's measurement and rationale would be the starting
  point.

## Per-core cold-start distribution as a known property

Independent of whether warmup is added back, the per-core variance is a
property of the dispatch path that future profiling work should expect:

- **First-task head-OH varies 0.3–5.6 µs across cores on a 72-core
  MIX batch.** Half the cores have first-task head-OH ≈ steady-state;
  the other half see 3–5 µs of cold tail.
- The variance source is NoC / FFTS routing, not the AICore-side
  dispatch code path.
- Steady-state (t1+) head-OH is 0.2–0.5 µs across all cores,
  dominated by publish-loop position (~5 ns per `write_reg`, ~360 ns
  total across 72 handles).
- A profile that observes "AICore startup is slow for some cores" on
  the first task of a run should not assume software warmup will fix
  it.

## Follow-up — receive_time DFX field + what propagation/local_setup actually mean

PR #1004 adds an AICore-side `receive_time` capture immediately after
`read_reg(DATA_MAIN_BASE)` returns the new task_id (before the per-task
`dcci(payload, ENTIRE_DATA_CACHE) + write_reg(COND, ACK)` pair). With it
the swimlane reports two new per-task fields:

- `local_setup_us` = start_time − receive_time (AICore-side dcci + ack).
- `propagation_us` = receive_time − dispatch_ts.

The cleanest reading **only applies to IDLE-phase dispatches** (where
AICPU writes the new task_id to a core whose RUNNING slot is empty —
core was actually idle and polling). In that case:

- `propagation_us` ≈ AICPU→AICore SoC delivery + AICore's next poll iter
  picking up the new value. Hardware-bound at the ns scale per the
  cold-start measurement above.
- `local_setup_us` ≈ dcci + ack cost on AICore. Empirically 0–0.02 µs on
  warm cache, 0.5–1 µs on cold first task.

For **PENDING-phase dispatches** (AICPU pre-loads next task_id while
RUNNING slot is still busy), `dispatch_ts` is captured at pre-load time,
which can be tens of µs before the prior task FINs. AICore picks up the
pending task immediately after prior FIN. So:

- `propagation_us` here is **not** AICPU→AICore delay. It is mostly
  "remaining exec time of the prior task at the moment of pre-load",
  i.e. a measurement artifact of dual-buffer scheduling, not a physical
  delivery delay.
- `local_setup_us` interpretation still holds (dcci + ack on AICore).

How to know which kind of dispatch a record represents: compare
`dispatch_ts` against the prior task's `end_time` on the same core
(both available in the swimlane records). `dispatch_ts ≥ prior.end` →
IDLE-phase, propagation reflects real NoC delivery. `dispatch_ts <
prior.end` → PENDING-phase, propagation is a pre-load offset.

Level-3 capture on `vector_example` (5 tasks, all IDLE-dispatched):

| Task | head_OH | propagation | local_setup |
| ---- | ------: | ----------: | ----------: |
| Cold first on core 5 | 2.12 µs | 1.20 µs | 0.92 µs |
| Warm on core 5 | 0.16 µs | 0.16 µs | 0.00 µs |
| Cold first on core 6 | 1.66 µs | 0.80 µs | 0.86 µs |
| Warm on core 5 | 0.34 µs | 0.34 µs | 0.00 µs |
| Warm on core 5 | 0.32 µs | 0.32 µs | 0.00 µs |

Cold tasks pay both halves; warm tasks have local_setup at cycle-zero
(dcci on already-invalidated cache is free) and propagation dominates.
**This pattern is specific to the IDLE-phase regime.** Mid-workload
traces (qwen3 decode_layer) include both IDLE and PENDING dispatches
interleaved, and aggregating `propagation_us` without separating them
yields confusing distributions — the "long tail" in such aggregates is
usually PENDING pre-load offset, not slow NoC.

Future warmup proposals targeting `local_setup_us` can now be evaluated
directly against captured data. Proposals targeting `propagation_us`
should first filter to IDLE-dispatched records to avoid attacking a
measurement artifact.

## References

- PR #989: dispatch path batched publish (where the eager swimlane head
  resolve lives).
- PR #1004: AICore receive_time DFX field (the measurement that
  confirmed the split above).
- PR #988: `spmd_serial_chain_mix` example used for measurement.
- Issue #545 comment #2: SPMD dispatch stagger.
- `.claude/rules/...` DFX priority guidance.
