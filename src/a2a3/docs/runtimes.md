# Runtime Variants (a2a3)

Three runtime implementations live under `src/a2a3/runtime/`, each providing a different graph-building strategy. The `RUNTIME_CONFIG.runtime` field in `kernel_config.py` selects which runtime to use.

## Comparison

| Feature | host_build_graph | aicpu_build_graph | tensormap_and_ringbuffer |
|---------|-----------------|-------------------|--------------------------|
| Graph built on | Host CPU | AICPU (device) | AICPU (device) |
| Task storage | Fixed `Task[]` array | Fixed `Task[]` array | Ring buffer (`PTO2TaskDescriptor[]`) |
| Dependencies | Explicit edges | Explicit edges | Auto-derived via TensorMap |
| Memory management | Host-side | Host + device malloc | Ring buffer heap (GM) |
| Concurrent build+schedule | No | Optional (`build_mode=1`) | Yes (always) |
| Profiling support | Basic | Basic | Multi-level hierarchy |
| Batch/streaming | No | No | Yes (flow control, back-pressure) |
| Thread model | N scheduler threads | 1 builder + N schedulers | 1 orchestrator + 3 schedulers |
| Use case | Development, debugging | Reduced host-device transfer | Production workloads |

## host_build_graph

The simplest runtime. The host CPU builds the complete task dependency graph before launching device execution. Orchestration runs on the host side.

- Task storage: fixed array (up to 131,072 tasks)
- Scheduling: AICPU receives the pre-built graph and dispatches by traversing dependencies
- No device-side orchestration overhead

See [host_build_graph/docs/RUNTIME_LOGIC.md](../runtime/host_build_graph/docs/RUNTIME_LOGIC.md) for details.

## aicpu_build_graph

Orchestration runs on an AICPU thread, building the task graph on device. Supports concurrent build + schedule (`build_mode=1`).

- Same task array as host_build_graph
- Device-side API: `add_task`, `add_successor_conditional`, `publish_task`, `device_malloc`
- Reduces host-device data transfer; graph can depend on device-side data

See [aicpu_build_graph/docs/RUNTIME_LOGIC.md](../runtime/aicpu_build_graph/docs/RUNTIME_LOGIC.md) for details.

## tensormap_and_ringbuffer (PTO2)

The primary production runtime. Uses ring buffers for task slots and output memory, with a TensorMap for automatic dependency tracking.

- Task storage: `PTO2TaskDescriptor[]` in shared memory ring buffer
- Memory: GM Heap ring for output buffer allocation
- Dependencies: automatically derived from tensor read/write patterns via TensorMap
- Thread model: 3 scheduler threads + 1 orchestrator thread on AICPU
- Multi-ring: HeapRing, TaskRing, and DepPool split into 4 independent instances for nested scope isolation
- Supports streaming, flow control, large batch sizes, and multi-level profiling

See [tensormap_and_ringbuffer/docs/](../runtime/tensormap_and_ringbuffer/docs/):
- [RUNTIME_LOGIC.md](../runtime/tensormap_and_ringbuffer/docs/RUNTIME_LOGIC.md) — Full system design
- [MULTI_RING.md](../runtime/tensormap_and_ringbuffer/docs/MULTI_RING.md) — Multi-ring buffer architecture
- [SUBMIT_BY_CLUSTER.md](../runtime/tensormap_and_ringbuffer/docs/SUBMIT_BY_CLUSTER.md) — Cluster submission design
- [profiling_levels.md](../runtime/tensormap_and_ringbuffer/docs/profiling_levels.md) — Profiling levels
- [device_log_profiling.md](../runtime/tensormap_and_ringbuffer/docs/device_log_profiling.md) — Device log profiling guide

## Shared Components

Runtime-specific shared files are in `src/a2a3/runtime/common/`:
- `pto_ring_buffer.h/cpp` — Ring buffer data structures (HeapRing, TaskRing, DepListPool)
- `pto_submit_types.h` — Subtask types, MixedKernels, ResourceShape

Cross-architecture shared files are in `src/common/task_interface/`:
- `data_type.h` — DataType enum and element size helpers
- `task_arg.h` — Task argument tagged union (host↔device data interface)
