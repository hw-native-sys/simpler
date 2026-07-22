# Runtime Variants (a2a3)

Three runtime implementations live under `src/a2a3/runtime/`, each providing a different graph-building strategy. The `RUNTIME_CONFIG.runtime` field in `kernel_config.py` selects which runtime to use.

## Comparison

| Feature | host_build_graph | tensormap_and_ringbuffer | replay_graph |
| ------- | ---------------- | ------------------------ | ------------ |
| Graph built on | Host CPU | AICPU (device) | AICPU (device) |
| Task storage | Fixed `Task[]` array | Ring buffer (`PTO2TaskDescriptor[]`) | Ping-pong task-window arenas + Graph Execution storage |
| Dependencies | Explicit edges | Auto-derived via TensorMap | Explicit + TensorMap, wired in submit |
| Memory management | Host-side | Ring buffer heap (GM) | Monotonic task/heap/dep pools |
| Concurrent build+schedule | No | Yes (always) | Yes, across explicit graph boundaries |
| Profiling support | Basic | Multi-level hierarchy | Multi-level hierarchy |
| Batch/streaming | No | Yes (flow control, back-pressure) | Arena-sized batches; repeated sub-DAGs collapse to one outer task |
| Thread model | N scheduler threads | 1 orchestrator + 3 schedulers | 1 orchestrator + scheduler threads |
| Use case | Development, debugging | Production workloads | Graph-boundary pipelining and Graph Execution |

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
- [pmu-profiling.md](../../../docs/dfx/pmu-profiling.md) — PMU design and per-task CSV output
- [l2-swimlane-profiling.md](../../../docs/dfx/l2-swimlane-profiling.md) — L2 swimlane and scheduler-phase profiling
- [args-dump.md](../../../docs/dfx/args-dump.md) — Per-task argument capture

## replay_graph

The replay runtime builds arena-sized graph batches on AICPU and pipelines their construction with scheduler dispatch. It uses one logical task stream, two task/output arenas, monotonic dependency allocation, and immutable fanout edges built directly by `submit_task`.

- Task lifecycle: `PENDING -> COMPLETED`; no `CONSUMED` or in-replay reclaim
- Capacity contract: one arena graph fits half the task window and heap; dependency and TensorMap pools cover the invocation
- Synchronization: `rt_graph_boundary()` publishes an arena graph and allows orchestration to build the next one concurrently
- Graph Execution: `rt_submit_graph()` captures a repeated C/V/MIX/Dummy sub-DAG as a Graph Definition, then represents each later execution with one outer Task Window task
- Scheduler expansion: a ready outer `GRAPH` task expands to scheduler-owned nodes and preserves their internal dependencies
- Failure mode: task-window/heap capacity exhaustion is a fatal sizing error rather than back-pressure

See the
[Graph Execution API Guide](../runtime/replay_graph/docs/GRAPH_EXECUTION_API.md)
for orchestration usage and
[Runtime Logic](../runtime/replay_graph/docs/RUNTIME_LOGIC.md) for the full
design.

## Shared Components

Runtime-specific allocator and submit type definitions are kept per-runtime (not in a shared `common/` directory):

- `{runtime}/runtime/pto_ring_buffer.{h,cpp}` — ring allocators for TMR; monotonic allocators for replay_graph
- `{runtime}/runtime/pto_runtime2_types.h` — Task descriptor types, resource shapes

Cross-architecture shared files are in `src/common/task_interface/`:

- `data_type.h` — DataType enum and element size helpers
- `tensor.h` — unified strided `Tensor` type + `TensorArgType` (host↔device data transport)
- `task_args.h` — TaskArgs template (separated tensor/scalar argument storage)
