# Replacing COND with GM+dcci for AICore→AICPU notification

**Date**: 2026-06-15
**Verdict**: deferred — COND wins on single-event latency, GM wins on
wide-polling rounds. Production keeps COND for ACK / FIN. GM remains
open for future hint-style channels.

## Question

The production scheduler reports AICore completion through the
`COND` SPR register: AICore `set_cond(FIN | task_id)`, AICPU
`LDR COND` (MMIO, Device-nGnRE). One AICPU thread polling a group of
cores costs ~95 ns × N as the cores grow. GM-with-dcci looks
attractive as an alternative — AICPU's cache stays warm while no
AICore writes, so its polling LDR can collapse to a few ns. Could we
shave the polling-round cost by 2–3× by moving FIN signalling onto
GM-and-coherency?

## What was tried

Two phases on `experiment/dmb-64bit-probe`, both AICPU-only on the
read side, AICore-only on the produce side:

- **Phase 13 — polling rate**. AICore tight-loops `counter++ + dcci`
  on its handshake field while `hammer_go == 13`. AICPU does
  10000-LDR sweeps in four modes: same GM field; rotating 9 GM
  fields; same COND register; rotating 9 COND registers. Per-LDR
  cost is reported and a `freshness` check confirms AICPU's reads
  actually see AICore's updates (proving the AICPU-side cache fills
  through the coherency protocol — see
  [cache-coherency.md](../hardware/cache-coherency.md)).
- **Phase 14 — E2E single-event latency**. AICore (only `block_idx
  == 0`) captures `tw = sys_cnt` immediately before either (mode 0)
  writing `p14_seq` + dcci, or (mode 1) `write_reg(COND, FIN | seq)`,
  throttled to ~1 µs / iter. AICPU polls the path under test, captures
  `t_obs = sys_cnt` on first change, reads paired `p14_tw`, and
  computes `latency = t_obs - tw`. Both producer and consumer use the
  same system counter, so the subtraction is well-defined. 100 samples
  each.

## Result

a3 silicon, vector_example config (3 AIC + 6 AIV), Phase 13 readings:

| Test | Per-LDR | Extrapolated to 24-core round |
| ---- | ------- | ----------------------------- |
| GM same field | **3 ns** (L1 hit) | 24 × 3 = ~72 ns |
| GM rotating 9 fields | **41 ns** | 24 × 41 ≈ **~984 ns** |
| COND same register | **105 ns** | 24 × 105 ≈ 2520 ns |
| COND rotating 9 registers | **100 ns** | 24 × 100 ≈ **~2400 ns** |

Phase 14 readings (single-producer, single-consumer, 100 samples):

| Path | avg | min | max |
| ---- | --- | --- | --- |
| **COND** (set_cond → AICPU LDR) | **600 ns** | **180 ns** | 2020 ns |
| **GM + dcci(SINGLE_CACHE_LINE)** | **1040 ns** | 980 ns | 1380 ns |

The cost asymmetry is reproducible: GM polling-round is ~2.5× faster
than COND, GM single-event latency is ~1.7× *slower* than COND.

## Why not (now)

Neither path dominates the other across the use cases of the
scheduler:

- **Single-event latency** is what ACK / FIN signalling lives or
  dies on. The `set_cond` path puts a new value in the COND register
  in tens of nanoseconds, and the AICPU's polling thread catches it
  on its next LDR (~95 ns minimum). The GM path adds AICore-side
  `dcci`-and-HBM-commit (~150–300 ns) plus AICore-side `dcci` on
  AICPU's read (~few hundred ns to invalidate and refetch). COND
  wins this by ~1.7× on average and ~5× on the min.
- **Wide polling sweeps** matter when one AICPU thread tracks many
  cores' liveness. With GM, the AICPU's cache stays warm while no
  AICore has written; a 24-core sweep collapses to ~1 µs total when
  the cache lines fit in L1. COND has no such cache and stays at
  ~95 ns / LDR × N cores. GM wins this by ~2.5×.

Production has both kinds of paths. The ACK / FIN signalling is the
load-bearing one — its latency directly bounds task dispatch
throughput. So **COND stays as the FIN channel.**

GM-coherent polling remains attractive for *hint* channels — paths
where the producer rate is much slower than the consumer poll rate,
and a missed cache hit on the consumer side is not catastrophic
(e.g. liveness checks, profiling progress counters). None of those
channels currently exist in the runtime; opening one would be a new
design, not a swap.

## When to reconsider

- A workload that consistently shows the single-thread COND-polling
  round (24 × 95 ≈ 2.3 µs) at the top of an AICPU profile, AND a
  design constraint that prevents adding scheduler threads to
  parallelise the sweep (which is the orthogonal cheaper fix).
- A hint channel proposal where consumer poll rate ≫ producer write
  rate, so the L1-warm GM path actually buys most of its 2.5×.
- Silicon-side relaxation of the `nR` attribute (would let
  single-thread LDR pipeline, narrowing or eliminating COND's
  polling penalty) — currently not on the roadmap.

Until then, **default to COND for new completion signals**; introduce
GM-coherent only for genuinely hint-shaped data flows.

## References

- **Standalone reproduction**:
  [`tools/cann-examples/aicore-notification-perf/`](../../tools/cann-examples/aicore-notification-perf/) —
  Phase 13 + Phase 14 packaged as three independently-built programs
  (AICore producer, AICPU consumer, host launcher). Build + run end-
  to-end without touching the production runtime.
- Branch: `experiment/dmb-64bit-probe` — original Phase 13 / 14 in
  `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_cold_path.cpp`
  and the matching producer in
  `src/a2a3/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp`.
- Per-LDR cost table propagated to
  [`docs/hardware/mmio-performance.md`](../hardware/mmio-performance.md).
- Coherency protocol it relies on:
  [`docs/hardware/cache-coherency.md`](../hardware/cache-coherency.md)
  — AICore → AICPU is automatic after AICore's `dcci`; AICPU does
  not need to invalidate.
