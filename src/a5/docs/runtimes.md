# Runtime Variants (a5)

Two runtime implementations live under `src/a5/runtime/`, each providing a different graph-building strategy. The `RUNTIME_CONFIG.runtime` field in `kernel_config.py` selects which runtime to use.

## Comparison

| Feature | host_build_graph | tensormap_and_ringbuffer |
| ------- | ---------------- | ------------------------ |
| Graph built on | Host CPU | AICPU (device) |
| Task storage | Fixed `Task[]` array | Ring buffer (`PTO2TaskDescriptor[]`) |
| Dependencies | Explicit edges | Auto-derived via TensorMap |
| Use case | Development, debugging | Production workloads |

## host_build_graph

See [host_build_graph/docs/RUNTIME_LOGIC.md](../runtime/host_build_graph/docs/RUNTIME_LOGIC.md).
Graph recording and replay are documented in
[GRAPH_EXECUTION.md](../runtime/host_build_graph/docs/GRAPH_EXECUTION.md).

## tensormap_and_ringbuffer (PTO2)

See [tensormap_and_ringbuffer/docs/](../runtime/tensormap_and_ringbuffer/docs/):

- [RUNTIME_LOGIC.md](../runtime/tensormap_and_ringbuffer/docs/RUNTIME_LOGIC.md) — Full system design
- [MULTI_RING.md](../runtime/tensormap_and_ringbuffer/docs/MULTI_RING.md) — Multi-ring buffer architecture
- [SUBMIT_BY_CLUSTER.md](../runtime/tensormap_and_ringbuffer/docs/SUBMIT_BY_CLUSTER.md) — Cluster submission design
- [SCALAR_DATA_ACCESS.md](../runtime/tensormap_and_ringbuffer/docs/SCALAR_DATA_ACCESS.md) — Scalar data access patterns
- [profiling_levels.md](../runtime/tensormap_and_ringbuffer/docs/profiling_levels.md) — Profiling levels
- [device_log_profiling.md](../runtime/tensormap_and_ringbuffer/docs/device_log_profiling.md) — Device log profiling guide
- [pmu-profiling.md](../../../docs/dfx/pmu-profiling.md) — PMU design and per-task CSV output
- [chip-swimlane-profiling.md](../../../docs/dfx/chip-swimlane-profiling.md) — chip swimlane and scheduler-phase profiling
- [args-dump.md](../../../docs/dfx/args-dump.md) — Per-task argument capture
