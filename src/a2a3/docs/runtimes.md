# Runtime Variants (a2a3)

Two runtime implementations live under `src/a2a3/runtime/`, each providing a different graph-building strategy. The `RUNTIME_CONFIG.runtime` field in `kernel_config.py` selects which runtime to use.

## Comparison

| Feature | host_build_graph | tensormap_and_ringbuffer |
| ------- | ---------------- | ------------------------ |
| Graph built on | Host CPU | AICPU (device) |
| Task storage | Fixed `Task[]` array | Ring buffer (`PTO2TaskDescriptor[]`) |
| Dependencies | Explicit edges | Auto-derived via TensorMap |
| Memory management | Host-side | Ring buffer heap (GM) |
| Concurrent build+schedule | No | Yes (always) |
| Profiling support | Basic | Multi-level hierarchy |
| Batch/streaming | No | Yes (flow control, back-pressure) |
| Thread model | N scheduler threads | 1 orchestrator + 3 schedulers |
| Use case | Development, debugging | Production workloads |

## host_build_graph

The simplest runtime. The host CPU builds the complete task dependency graph before launching device execution. Orchestration runs on the host side.

- Task storage: fixed array (up to 131,072 tasks)
- Scheduling: AICPU receives the pre-built graph and dispatches by traversing dependencies
- No device-side orchestration overhead

See [host_build_graph/docs/RUNTIME_LOGIC.md](../runtime/host_build_graph/docs/RUNTIME_LOGIC.md) for details.

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
- [pmu-profiling.md](../../../docs/pmu-profiling.md) — PMU design and per-task CSV output
- [l2-swimlane-profiling.md](../../../docs/l2-swimlane-profiling.md) — L2 swimlane and scheduler-phase profiling
- [tensor-dump.md](../../../docs/tensor-dump.md) — Per-task tensor I/O capture

## Shared Components

Ring buffer and submit type definitions are duplicated per-runtime (not in a shared `common/` directory):

- `{runtime}/runtime/pto_ring_buffer.cpp` — Ring buffer data structures (HeapRing, TaskRing, DepListPool)
- `{runtime}/runtime/pto_runtime2_types.h` — Task descriptor types, resource shapes

Cross-architecture shared files are in `src/common/task_interface/`:

- `data_type.h` — DataType enum and element size helpers
- `tensor_arg.h` — ContinuousTensor type (host↔device data transport)
- `task_args.h` — TaskArgs template (separated tensor/scalar argument storage)
