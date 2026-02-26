# Runtime Layer

## Purpose

The runtime layer (`src/runtime/`) implements the task dependency graph and scheduling strategy. Three variants exist, each building on the previous one's ideas with increasing sophistication.

All three variants produce the same three programs (Host `.so`, AICPU `.so`, AICore `.o`) and follow the same handshake protocol (see [README.md "Handshake Protocol"](../README.md#handshake-protocol)), but they differ in **where the graph is built** and **how tasks are stored and scheduled**.

## Variant Comparison

| Aspect | `host_build_graph` | `aicpu_build_graph` | `tensormap_and_ringbuffer` (RT2) |
|--------|-------------------|--------------------|---------------------------------|
| Graph built by | Host CPU | AICPU thread 3 (on-device) | AICPU thread 3 (on-device) |
| Task storage | `Task[]` fixed array | `Task[]` fixed array | `TaskRing` (circular ring buffer) |
| Dependency model | Manual `add_successor()` | Manual `add_successor()` | Automatic via `TensorMap` lookup |
| Build-schedule overlap | No (build first, then schedule) | Yes (concurrent build ‖ schedule) | Yes (orchestrator ‖ scheduler) |
| Orchestration location | Host-side (linked into host `.so`) | AICPU-side (`dlopen`'d `.so`) | AICPU-side (`dlopen`'d `.so`, PTO2 ops table) |
| Dispatch mechanism | Handshake buffer → `Task*` | Handshake buffer → `Task*` | Handshake buffer → `PTO2DispatchPayload*` |
| Memory management | `MemoryAllocator` per-alloc | `MemoryAllocator` + `dlopen` | Ring buffers (`HeapRing`, `TaskRing`) — zero allocation after init |

## Variant 1: `host_build_graph`

The simplest variant. Good for debugging and small graphs.

**How it works:**
1. Host builds the full task graph via an orchestration function (compiled as `.so`, loaded via `dlopen`)
2. Host copies the entire `Runtime` struct (containing `Task[]` array) to device memory
3. AICPU scans for initially-ready tasks (`fanin == 0`) and dispatches them to AICore via handshake buffers
4. When an AICore finishes a task, AICPU decrements `fanin` of all successors and dispatches newly-ready tasks
5. AICPU sends quit signal when all tasks complete

**Key files:**
- `host_build_graph/runtime/runtime.h` — `Task` struct, `Runtime` class, `Handshake` struct
- `host_build_graph/host/runtime_maker.cpp` — Host-side init, orchestration loading, tensor copy-back
- `host_build_graph/aicpu/aicpu_executor.cpp` — AICPU task scheduler
- `host_build_graph/aicore/aicore_executor.cpp` — AICore poll-execute loop

For detailed execution flow, see [README.md "Execution Flow"](../README.md#execution-flow).

## Variant 2: `aicpu_build_graph`

Moves graph construction to the device. Enables dynamic graphs that depend on device-side computation results.

**Key differences from variant 1:**
- Host embeds the orchestration `.so` binary into the `Runtime` struct (`aicpu_orch_so_storage[]`)
- AICPU thread 3 materializes the `.so` to a temp file, `dlopen`s it, and resolves the build function
- An `AicpuBuildApi` function-pointer table is passed to the orchestration function, providing `add_task()`, `publish_task()`, `add_successor()`, etc. This avoids symbol resolution issues across `.so` boundaries
- **Concurrent build ‖ schedule:** Builder publishes tasks via `published` atomic flag; scheduler threads consume published tasks while the builder continues adding more
- `add_successor_conditional()` handles the race where a predecessor finishes before its fanout is fully wired

**Key additions over variant 1:**
- `build_mode` field in `Runtime` (controls whether AICPU runs builder + scheduler or scheduler only)
- `orch_args[]` — Wider argument marshaling for device orchestration
- `kernel_addrs[]` — `func_id → GM address` lookup table (host fills after uploading kernels)
- `AicpuBuildApi` — Device-side function pointer table

**Key files:**
- `aicpu_build_graph/runtime/runtime.h` — Extended `Runtime` with build API, orchestration storage, and concurrent atomics

## Variant 3: `tensormap_and_ringbuffer` (RT2)

A complete redesign for production workloads. Zero-allocation steady state, automatic dependency discovery, and multi-core scheduling.

**Architecture:**

```
PTO2SharedMemory (device GM)
├── SharedMemoryHeader     Flow control pointers (orchestrator ↔ scheduler)
├── TaskDescriptor[]       Ring buffer of task slots
└── DepListPool            Ring buffer for dependency list entries

PTO2 Orchestrator State (AICPU thread 3, private memory)
├── TensorMap              Hash table: tensor region → producer task ID
├── ScopeStack             RAII buffer lifecycle tracking
└── HeapRing               Bump allocator for output buffers

PTO2 Scheduler State (AICPU threads 0-2, private memory)
├── ReadyQueues            Per-worker-type circular queues (AIC, AIV)
└── Handshake buffers      One per AICore worker
```

**Key data structures:**

| Structure | Purpose | Design |
|-----------|---------|--------|
| `TaskRing` | Task slot allocation | Fixed window, modulo wrap-around, implicit reclamation via `last_task_alive` |
| `HeapRing` | Output buffer allocation | O(1) bump allocator, wrap-around, back-pressure via spin-wait when full |
| `DepListPool` | Dependency list entries | Ring buffer for linked-list nodes, implicit reclamation with task ring |
| `TensorMap` | Dependency discovery | Hash table with chaining. Maps `(base_ptr, offset, size)` → producer task. Lazy invalidation: entries stale when `producer_task_id < last_task_alive`. Chain truncation on first stale entry. Hash by `base_ptr` only for overlap detection. |
| `ScopeStack` | Buffer lifecycle | RAII-based `scope_begin()` / `scope_end()` brackets. Enables early buffer release when a scope's outputs are no longer needed. |

**How it works:**
1. Host creates `PTO2SharedMemory` (ring buffers + task descriptors) in device GM
2. AICPU thread 3 (orchestrator) builds the graph on device via a `PTO2RuntimeOps` function-pointer table
3. `pto2_submit_task()` automatically discovers dependencies by querying the `TensorMap` for each input tensor's producer
4. Ready tasks (with all dependencies satisfied) are pushed to per-worker-type `ReadyQueues`
5. AICPU threads 0-2 (schedulers) dequeue from ready queues and dispatch via `PTO2DispatchPayload` in the handshake buffer
6. AICore executes the kernel and signals completion
7. Scheduler decrements fanin of successors and enqueues newly-ready tasks

**Task lifecycle:** `PENDING → READY → RUNNING → COMPLETED → CONSUMED`

**Key files:**
- `tensormap_and_ringbuffer/runtime/pto_runtime2.h` — Main PTO2Runtime interface and ops table
- `tensormap_and_ringbuffer/runtime/pto_shared_memory.h` — Shared memory layout
- `tensormap_and_ringbuffer/runtime/pto_ring_buffer.h` — HeapRing, TaskRing, DepListPool
- `tensormap_and_ringbuffer/runtime/pto_tensormap.h` — TensorMap with lazy invalidation
- `tensormap_and_ringbuffer/runtime/pto_scheduler.h` — Scheduler state machine
- `tensormap_and_ringbuffer/runtime/pto_orchestrator.h` — Orchestrator state
- `tensormap_and_ringbuffer/orchestration/pto_orchestration_api.h` — Orchestration `.so` API

## Common Concepts

### Handshake Buffer

All three variants use the same cache-line-aligned `Handshake` struct for AICPU-AICore communication:

```c
struct Handshake {
    volatile uint32_t aicpu_ready;       // AICPU → AICore: scheduler ready
    volatile uint32_t aicore_done;       // AICore → AICPU: core ready
    volatile uint64_t task;              // AICPU → AICore: task pointer
    volatile int32_t  task_status;       // 0 = idle, 1 = busy
    volatile int32_t  control;           // 0 = continue, 1 = quit
    volatile CoreType core_type;         // AIC or AIV
    // ... profiling fields
} __attribute__((aligned(64)));
```

One handshake buffer per AICore worker. See [README.md "Handshake Protocol"](../README.md#handshake-protocol) for the full state machine.

### Task Structure

Variants 1 and 2 use a `Task` struct with:
- `func_id` — Kernel function ID
- `args[RUNTIME_MAX_ARGS]` — Up to 16 `uint64_t` arguments
- `function_bin_addr` — Kernel address in device GM memory
- `core_type` — `AIC` or `AIV`
- `std::atomic<int> fanin` — Predecessor count (decremented atomically)
- `fanout[RUNTIME_MAX_FANOUT]` — Up to 512 successor task IDs

RT2 uses `PTO2TaskDescriptor` with a similar structure but stored in a ring buffer instead of a fixed array.

### build_config.py

Each variant has a `build_config.py` that declares include paths and source directories for the three targets (host, aicpu, aicore). Selected at build time by `RUNTIME_CONFIG.runtime` in the example's `kernel_config.py`.

## Design Principles

1. **Lock-free handshake** — AICPU and AICore communicate via `volatile` fields + cache-line alignment. No mutexes, no atomic CAS on the hot path (single-word reads/writes are naturally atomic on ARM64).

2. **Cache-line alignment** — `Handshake` is `__attribute__((aligned(64)))` to prevent false sharing between cores.

3. **Concurrent build ‖ schedule** (v2, RT2) — The graph builder publishes tasks as they become ready; scheduler threads consume them in parallel. No need to wait for the entire graph to be built before scheduling begins.

4. **Lazy invalidation** (RT2) — TensorMap entries are not explicitly removed. Instead, entries with `producer_task_id < last_task_alive` are considered stale and skipped during lookup. Chain truncation drops the rest of a chain on the first stale entry.

5. **Zero-allocation steady state** (RT2) — Ring buffers (`HeapRing`, `TaskRing`, `DepListPool`) wrap around and implicitly reclaim space. After initialization, no `malloc`/`free` calls are needed.

6. **Back-pressure** (RT2) — When `TaskRing` or `HeapRing` is full, the orchestrator spin-waits until space is freed by the scheduler. This prevents overcommitting memory.

7. **Scope-based lifecycle** (RT2) — `scope_begin()` / `scope_end()` brackets group tasks and their output buffers. When a scope's outputs are consumed, its heap space can be reclaimed.

## See Also

- [README.md](../README.md) — Execution flow and handshake protocol for `host_build_graph`
- [.ai-instructions/coding/architecture.md](../.ai-instructions/coding/architecture.md) — Three-program model deep-dive
