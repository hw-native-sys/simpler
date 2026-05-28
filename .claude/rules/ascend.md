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
