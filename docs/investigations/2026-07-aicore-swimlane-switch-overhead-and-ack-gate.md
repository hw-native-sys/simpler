# L2 swimlane AICore: switch-overhead source + FIN-early reorder & ACK-gate

**Date**: 2026-07-08
**Verdict**: measured — the ~0.8 µs inter-task "switch" is the per-task
record write-back (`dcci(record, CACHELINE_OUT) + dsb`) on AICore's own
timeline; it is **inherent, not reducible by reordering**. Two changes
shipped alongside the measurement: (1) sample the swimlane `end_time`
after an early FIN so the record write no longer sits on the
completion-signal path; (2) an AICPU-side **ACK-gate** on buffer
rotation, required because that early FIN otherwise races the record
flush at BUFFER_SIZE boundaries.

## Question

Switching between two back-to-back AICore tasks shows a ~0.7–1.0 µs gap
on the L2 swimlane in normal dispatch (not in speculative early
dispatch). Where does it come from, and can moving the FIN write earlier
(release the core to AICPU sooner) shrink it?

## What was tried

Ran `qwen3_14b_decode` (`StressBatch16Seq3500`, block_dim auto → 72
cores, 1088 AICore tasks) with `--enable-l2-swimlane 4` on a2a3 silicon
via `task-submit`. Decomposed each core's inter-task gap from the
records (`aicore_tasks = [core, token, reg_task_id, start, end,
receive_to_start_cycles]`, 50 MHz → 20 ns/tick), classifying a gap as a
**switch** when the next task was already dispatched before the previous
ended (`dispatch(K+1) < end(K)`, i.e. work was waiting) vs a **WAIT**
gap (core idle, scheduler/dep-bound). Switches split at `receive_time`
(`= start − receive_to_start_cycles`):

- `prev_tail = receive(K+1) − end(K)` — AICore's own post-task work:
  `record_task` (the `dcci(record, OUT) + dsb`) + loop-around pickup.
- `setup = receive_to_start_cycles(K+1)` — payload `dcci(ENTIRE)` + ACK.

## Result

| metric (steady state) | mean | p50 | p90 | p99 |
| --------------------- | ---- | --- | --- | --- |
| STEADY switch (`disp < end`) | 0.81 | 0.80 | 1.00 | 1.22 |
| ↳ prev_tail (record + loop) | 0.52 | 0.50 | 0.66 | 0.88 |
| ↳ setup (payload dcci + ack) | 0.29 | 0.28 | 0.40 | 0.60 |
| first switch / core (cold) | 1.39 | 1.48 | 1.86 | 1.98 |
| WAIT gap (core idle) | 126 | 54 | 381 | 698 |

(µs; n = 581 steady switches, 403 wait gaps, 32 cores.)

The switch's dominant half is **prev_tail ≈ 0.5 µs = the record
write-back `dcci(record, CACHELINE_OUT) + dsb`**. `setup` (~0.28 µs) is
the payload invalidate + ACK. The **WAIT gap (p99 ~700 µs)** dwarfs the
whole switch — decode latency is a scheduler/dep-graph question, not a
DFX one.

**The switch gap is largely profiling self-cost, not a real hardware
task switch.** With the swimlane on, every task issues three
`get_sys_cnt_aicore()` SPR reads (receive / start / end), each ~100–200
ns, and one record write-back. The `receive` read lands entirely inside
the switch gap; the `end`/`start` reads contribute their boundary
fractions. So the ~0.8 µs decomposes roughly as **get_sys_cnt reads
(~0.2–0.4 µs) + record write-back `dcci+dsb` (~0.4–0.5 µs)** — i.e. the
gap is dominated by the instrumentation itself. This is an *observer
effect*: with the swimlane off none of these reads or the record write
happen, so the gap vanishes — which is exactly why the overhead "only
appears on the swimlane" and not in a non-profiled / early-dispatch run.
`prev_tail`'s "record + loop" therefore includes the `receive` read, and
`setup` includes part of the `start` read; neither is pure dcci/ack.

**Moving FIN earlier does not shrink the AICore switch.** `record_task`
still runs between `end(K)` and `receive(K+1)`, so prev_tail is
unchanged. What the early FIN *does* change: AICPU observes completion
~0.5 µs sooner (before the record write), decoupling the record flush
from the completion signal. That is a pipelining nicety on the AICPU
side, not a switch-gap reduction.

## What shipped

1. **`end_time` after an early FIN** (`aicore_executor.cpp`, tmr):
   op order is now `ACK → pmu_begin → start → execute → FIN → end →
   pmu_end → dump → record`. FIN fires right after `execute_task`;
   `record_task` runs last, off the completion path. The three swimlane
   timestamps are gated on `l2_swimlane_enabled`.

2. **AICPU ACK-gate on rotation** (`l2_swimlane_collector_aicpu.cpp`,
   shared a2a3). Because tmr now writes FIN *before* the record, the
   completion-before-dispatch invariant no longer proves the old
   buffer's tail record has drained at a BUFFER_SIZE rotation — AICPU
   could enqueue a buffer whose last record's `dcci+dsb` is still in
   flight (host then discards it as `start_time == 0`). Fix: at
   rotation, publish the new buffer to AICore but **stash** the
   just-filled buffer; release it to the host only when AICPU observes
   AICore ACK the new buffer's first task (`reg_task_id == gate`). By
   AICore's single-threaded program order that ACK is emitted only
   after the previous task's `record_task + dsb`, so the tail record is
   proven drained. `host_build_graph` writes the record *before* FIN,
   so it has no such window — it does not wire the ACK hook and relies
   on the next-rotation / run-end backstop.

## Why not chase the switch further

The 0.5 µs is a cache-line clean-out + `dsb` on a DFX-only path; it is
sub-noise against the ~700 µs WAIT tail and self-correcting (the host
tolerates a dropped tail record). The per-task `dcci(head)` poll was
separately measured as effectively free, and deferring the record
`wmb`/`dsb` already measured as sub-noise
(`2026-06-l2-swimlane-defer-wmb.md`). The accuracy-over-performance rule
for profiling paths applies.

## When to reconsider

- If a workload runs **> 1024 AICore tasks on a single core** the
  ACK-gate's boundary path becomes live (qwen decode's ~15 tasks/core
  never crosses it, so the ST smoke and this run exercise only the
  common path). Re-measure buffer-tail record loss there to confirm the
  gate eliminates the `start_time == 0` discards.
- If the record write-back moves off HBM (e.g. an on-chip swimlane
  staging buffer), the 0.5 µs prev_tail would shrink and the switch
  decomposition should be re-run.

## References

- [2026-06 — L2 swimlane: defer per-task wmb to rotation](2026-06-l2-swimlane-defer-wmb.md)
- `.claude/rules/codestyle.md` (no logging on AICPU hot paths),
  `docs/hardware/mmio-performance.md` (dcci/dsb costs).
