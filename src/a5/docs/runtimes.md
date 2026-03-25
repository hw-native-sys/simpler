# Runtime Variants (a5)

Two runtime implementations live under `src/a5/runtime/`, each providing a different graph-building strategy. The `RUNTIME_CONFIG.runtime` field in `kernel_config.py` selects which runtime to use.

## Comparison

| Feature | host_build_graph | tensormap_and_ringbuffer |
|---------|-----------------|--------------------------|
| Graph built on | Host CPU | AICPU (device) |
| Task storage | Fixed `Task[]` array | Ring buffer (`PTO2TaskDescriptor[]`) |
| Dependencies | Explicit edges | Auto-derived via TensorMap |
| Use case | Development, debugging | Production workloads |

## host_build_graph

See [host_build_graph/docs/RUNTIME_LOGIC.md](../runtime/host_build_graph/docs/RUNTIME_LOGIC.md).

## tensormap_and_ringbuffer (PTO2)

See [tensormap_and_ringbuffer/docs/](../runtime/tensormap_and_ringbuffer/docs/):
- [RUNTIME_LOGIC.md](../runtime/tensormap_and_ringbuffer/docs/RUNTIME_LOGIC.md) — Full system design
- [MULTI_RING.md](../runtime/tensormap_and_ringbuffer/docs/MULTI_RING.md) — Multi-ring buffer architecture
- [SUBMIT_BY_CLUSTER.md](../runtime/tensormap_and_ringbuffer/docs/SUBMIT_BY_CLUSTER.md) — Cluster submission design
- [profiling_levels.md](../runtime/tensormap_and_ringbuffer/docs/profiling_levels.md) — Profiling levels
- [device_log_profiling.md](../runtime/tensormap_and_ringbuffer/docs/device_log_profiling.md) — Device log profiling guide
