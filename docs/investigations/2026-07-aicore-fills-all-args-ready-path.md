# AICore-side arg materialization for ALL dispatches (not just the gated path)

**Date**: 2026-07-11
**Verdict**: measured and rejected for the ready path — making the AICore
fill its own `args[]` for **every** task adds **~1.0 µs to each task's
setup** (`receive → start`), on the critical path, because a ready task has
no idle doorbell gate to hide the fill behind. Offloading is only a win on
the `not_ready` (early-dispatch) path, where the AICore is already spinning
at the gate and absorbs the fill for free. The shipped design keeps the
AICPU filling `args[]` for ready tasks and offloads only gated ones.

## Question

The tmr dispatch handoff writes the kernel `args[]` (tensor GM pointers +
scalar values) into the per-core `PTO2DispatchPayload` on the AICPU. For
gated (`not_ready` / early-dispatch) tasks we already offload that fill to
the AICore — it does it during its doorbell wait, off the AICPU dispatch
path (see the folded-gate `src_payload` design in the dispatch cold-write
PR).

Natural follow-on: why not offload the fill for **all** tasks and take the
`args[]` write off the AICPU entirely? What does that cost the AICore?

## What was tried

An experiment build (not merged) that:

- Adds a separate `gated` flag to `PTO2DispatchPayload` — with all tasks
  filling from source, `src_payload` is always non-zero and can no longer
  double as the gate flag (the shipped design folds `not_ready` into
  `src_payload == 0`).
- AICPU `build_payload` writes only `src_payload = &PTO2TaskPayload` and
  `gated`, and **never** fills `args[]`.
- AICore `aicore_executor` fills `args[0..num_args)` from `src_payload` for
  **every** task (moved out of the gate branch), then gates only when
  `gated`.

Measured `paged_attention_unroll` (`--enable-l2-swimlane`, 1024 AICore
tasks, a2a3 silicon) and read each task's AICore setup —
`receive_to_start_cycles` from `l2_swimlane_records.json::aicore_tasks`
(50 MHz → 20 ns/cycle).

## Result

| build | AICore setup (`receive → start`) |
| ----- | -------------------------------- |
| baseline (AICPU fills args) | mean **349 ns** / p50 320 ns |
| experiment (AICore fills **all**) | mean **1356 ns** / p50 1340 ns |
| **delta** | **+~1.0 µs/task (+50 cycles, ~4×)** |

Correctness held (`paged_attention_unroll` passes), so the AICore fill is
functionally equivalent — it just costs ~1 µs. The fill is AICore GM work:
after the per-task whole-cache `dcci(ENTIRE_DATA_CACHE)`, the first reads of
the source `PTO2TaskPayload` (counts, scalars) miss to HBM, then the loop
computes `&tensors[i]` and writes `args[]` — all on a core whose GM-access
latency is high. On the **ready** path there is no doorbell wait to overlap
it with, so the whole ~1 µs lands on the `receive → start` setup.

## Why the shipped design offloads only the gated path

- **Gated (`not_ready`) tasks are staged before their producer completes**;
  the AICore then spins at the doorbell gate. The ~1 µs fill runs inside
  that otherwise-idle wait → net ~zero, and it takes the `args[]` write off
  the AICPU dispatch path (the win we wanted).
- **Ready tasks execute on pickup** — no gate, no idle window. Offloading
  the fill there is a straight +~1 µs/task regression on the AICore
  critical path. For an AICore-bound workload (e.g. `paged_attention`,
  1024 tasks) that is a large, unhideable cost.

So the asymmetry is intrinsic: offload where there is idle time to spend
(gated), keep it on the AICPU where the AICore would otherwise stall
(ready).

## When to reconsider

- If a future protocol gives ready tasks an idle window between pickup and
  the point their inputs are guaranteed visible (today the kernel's input
  `dcci` runs right after ACK), the fill could hide there — re-measure.
- If the AICPU dispatch path becomes the proven end-to-end bottleneck for a
  ready-dominated workload AND the AICore has spare setup budget (setup ≪
  the WAIT/exec time), the trade could flip — but weigh against the ~1 µs
  measured here, and note it scales with per-task arg count.

## References

- Dispatch cold-write PR (folded-gate `src_payload`, AICore fill on the
  gated path, init prefill, deferred-slab-at-completion, prefetch):
  hw-native-sys/simpler#1328.
- [2026-07 — L2 swimlane AICore switch-overhead](2026-07-aicore-swimlane-switch-overhead-and-ack-gate.md)
  — the baseline ~0.28–0.35 µs AICore setup (`payload dcci + ACK`) this
  experiment adds onto.
