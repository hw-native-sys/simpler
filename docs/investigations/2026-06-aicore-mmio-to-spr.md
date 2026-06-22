# Letting AICore directly read or write the SPR MMIO window

**Date**: 2026-06-15
**Verdict**: dropped — both sub-paths blocked by hardware

## Question

When the scheduler design pushes more work onto AICore (e.g.
peer-to-peer ACK without going through AICPU, or AICore mutating its
own `DATA_MAIN_BASE` for fast-path tricks), the obvious shortcut is
to have AICore access the same SPR MMIO window AICPU uses — either:

1. **Cross-core read**: AICore on core A loads
   `peer.reg_addr + DMB_OFFSET` to observe core B's dispatch state
   without round-tripping through AICPU.
2. **Self-write**: AICore on core A writes its own
   `DATA_MAIN_BASE` either via SPR instruction (`MOV DATA_MAIN_BASE,
   x`) or via MMIO STR at `own.reg_addr + DMB_OFFSET`, to publish
   completion or self-reset.

Both are tempting because the MMIO window is already mapped on AICPU
and the addresses are known.

## What was tried

Two probe phases were added to the experiment scaffold on
`experiment/dmb-64bit-probe`:

- **Phase 10** — AICPU pre-seeds each core's DMB with a unique 64-bit
  magic via MMIO, publishes per-core `reg_addr` into Handshake fields,
  and signals AICore (via `hammer_go = 10`). Each AICore reads its
  own DMB via SPR (sanity), then attempts a 64-bit `LDR` from each
  peer core's `reg_addr + DMB_OFFSET` via the AICore-side LSU.
  Per-stage marker fields let AICPU detect a stall without crashing
  the host process.
- **Phase 11** — AICPU pre-seeds each core's DMB with a sentinel.
  AICore tries `MOV DATA_MAIN_BASE, x` (compile-time probe of SPR
  write), then an MMIO STR to its own `reg_addr + DMB_OFFSET`.

Source: `src/a2a3/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp`
and `.../runtime/scheduler/scheduler_cold_path.cpp` on the experiment
branch (both phases kept under `#if 0` after the verdicts below).

## Result

**Phase 10 — cross-core MMIO read**: All 9 cores (3 AIC + 6 AIV) on
the test config reached stage 1 (own-SPR read succeeded), advanced
the stage marker to 2 (about to issue the first cross-core MMIO
load), then stuck. AICPU's 5 M-iter spin (~200 ms) timed out with
every core at stage 2. ~2 seconds later the CCECPU monitor declared
op-timeout and killed `aicpu-sd`. The chip stayed in `Critical`
state in `npu-smi info` until reset. The first cross-core MMIO load
itself hangs the AICore — the LSU cannot route to the SPR window.

**Phase 11 — SPR write of DMB**: The build failed at AICore
compilation with:

```text
error: invalid operand for instruction
        __asm__ volatile("MOV DATA_MAIN_BASE, %0\n" : : "l"(spr_magic));
        MOV DATA_MAIN_BASE, X6
```

The CCEC backend encodes `DATA_MAIN_BASE` as a SPR mnemonic that is
valid only as a *source* in `MOV`. There is no destination encoding.

The MMIO-STR fallback for Phase 11 was not run because Phase 10
already established that any AICore-side LSU access to the SPR
window hangs the chip — symmetrically, a STR would also hang.

## Why not (now)

Two independent hardware-level walls:

- **AICore LSU cannot reach the SPR MMIO window** (Phase 10). This
  is not a permission / driver issue — the slot of the chip's
  interconnect that maps to AIC_CTRL only accepts transactions from
  AICPU. AICore-side loads / stores at those addresses time out and
  ultimately bring down `aicpu-sd`.
- **DMB is read-only from AICore at the SPR level** (Phase 11). The
  AICore SPR encoding does not include a destination slot for DMB,
  so even if hardware could implement it, the compiler can't emit
  the instruction.

So there is no path — direct or indirect — by which AICore can
observe peer state or mutate its own DMB without round-tripping
through AICPU.

## When to reconsider

- A chip family where the AICore LSU is given a window into the SPR
  space (post-silicon design change, not a software lever).
- A new CCEC release that adds an SPR-write encoding for DMB. Search
  the CCEC built-ins for `__builtin_cce_set_data_main_base` or
  equivalent — its presence would lift the Phase 11 wall.

Until either of those happens, **route any AICore-originating signal
through GM with `dcci`, and let AICPU forward into DMB by MMIO STR.**

## References

- Branch: `experiment/dmb-64bit-probe` — Phase 10/11 code under
  `#if 0`.
- Build log for Phase 11 SPR write rejection: `error: invalid operand
  for instruction / MOV DATA_MAIN_BASE, X6`.
- Device log for Phase 10 hang:
  `[P10] TIMEOUT — some cores stuck. Per-core stages: ... stage=2` ×
  9, followed by
  `[ERROR] CCECPU ... HandleTaskTimeout ... op name[simpler_aicpu_exec] ...`.
- Constraint propagated to [`docs/hardware/mmio-performance.md`](../hardware/mmio-performance.md)
  and [`.claude/rules/ascend.md`](../../.claude/rules/ascend.md).
