# Ascend Architecture Quick Reference

See [docs/hardware/chip-architecture.md](../../docs/hardware/chip-architecture.md) for the **hardware substrate** (Host CPU / AICPU / AICore tiers, PCIe boundary, cache-coherency entry point) — shared across a2a3 and a5. See [docs/chip-level-arch.md](../../docs/chip-level-arch.md) for the **software three-program model** (host `.so` + AICPU `.so` + AICore `.o`) layered on top: full diagram, API layers, execution flow, handshake protocol. See [docs/hierarchical_level_runtime.md](../../docs/hierarchical_level_runtime.md) for the L0–L6 level model and component composition, and [docs/task-flow.md](../../docs/task-flow.md) for end-to-end task data flow.

## Key Concepts

- **Three programs**: Host `.so`, AICPU `.so`, AICore `.o` — compiled independently, linked at runtime
- **Two runtimes** under `src/{arch}/runtime/`: `host_build_graph`, `tensormap_and_ringbuffer`
- **Two platform backends** under `src/{arch}/platform/`: `onboard/` (hardware), `sim/` (simulation)

## Hardware Units

- **AIC** = **AICore-CUBE**: Matrix computation unit for tensor operations (matmul, convolution)
- **AIV** = **AICore-VECTOR**: Vector computation unit for element-wise operations (add, mul, activation)
- **AICPU**: Control processor for task scheduling and data movement (not a worker type — acts as scheduler)

For the full hardware tier model (Host CPU / AICPU / AICore), off-chip vs on-chip boundary, and end-to-end task flow, see [docs/hardware/chip-architecture.md](../../docs/hardware/chip-architecture.md). For chip-specific counts and sizes, see `src/a2a3/docs/` and `src/a5/docs/`.

## Hard Hardware Constraints (do not re-propose)

These were each verified on a3 silicon. Re-deriving them costs at
least a chip-hang round-trip per item; prefer the linked doc.

- **AICore has no path to write `DATA_MAIN_BASE`.** The SPR write
  instruction `MOV DATA_MAIN_BASE, x` is rejected by the CCEC backend
  at compile time, and an AICore-side MMIO STR to the SPR window
  hangs the chip (CCECPU kills `aicpu-sd` after the 50 s op-timeout).
  DMB is hardware-unidirectional: AICPU writes, AICore SPR-reads.
- **AICore's LSU cannot reach the SPR MMIO window.** Any load or
  store from an AICore to `reg_addr + offset` hangs that core, including
  cross-core attempts and self-core attempts. Don't design protocols
  where one AICore directly observes or mutates another core's
  control register; route through AICPU instead.
- **AICPU single-thread LDR COND is strictly serial.** The MMIO
  region is `Device-nGnRE` (driver source: `pgprot_device()`), so the
  `nR` attribute makes one LDR drain before the next can issue —
  ~95 ns per LDR, no way to pipeline. Polling N cores from one thread
  costs ~95 ns × N. Multi-thread cross-core polling, on the other
  hand, scales linearly and is the only way to shrink the round.

Details + measurement provenance in
[docs/hardware/mmio-performance.md](../../docs/hardware/mmio-performance.md).
"We tried it and dropped it" history for the AICore-side MMIO ideas
sits in
[docs/investigations/2026-06-aicore-mmio-to-spr.md](../../docs/investigations/2026-06-aicore-mmio-to-spr.md).
