# TPUSH/TPOP Simulation Support

## Overview

TPUSH/TPOP enables direct data transfer between AIC (Cube) and AIV (Vector)
cores within the same cluster via an on-chip VEC_FIFO, bypassing global memory.

In simulation, the hardware VEC_FIFO is replaced by a software ring buffer
(`SharedState`) managed per cluster per pipe configuration. The simulation
platform provides the necessary context so that pto-isa's CPU simulation
backend can locate the correct ring buffer without any runtime involvement.

## Architecture

### Hardware Model

Each cluster has a fixed set of physical VEC_FIFO channels. AIC writes
(TPUSH), AIV reads (TPOP). The channels persist across task dispatches —
they are not created or destroyed per task.

### Simulation Model

The simulation mirrors this with per-cluster, per-pipe-configuration
`SharedState` instances:

```text
cluster 0 ──┬── pipe_key A → SharedState (mutex + ring buffer + slot storage)
             └── pipe_key B → SharedState (if kernel uses multiple pipe types)
cluster 1 ──┬── pipe_key A → SharedState
             └── pipe_key B → SharedState
...
```

Each `SharedState` contains:

- A mutex and condition variable for producer/consumer synchronization
- `next_producer_slot` / `next_consumer_slot` ring indices
- `occupied` counter (number of filled slots)
- `local_slot_storage[SlotNum]` — the actual data buffers
- `remaining_consumers[SlotNum]` — per-slot consumer reference count (for split mode)

### Key Properties

**Fixed channel count.** The number of `SharedState` entries is determined at
compile time by the kernel's `TPipe<...>` template instantiations, not by the
number of tasks. A typical kernel has 1 pipe type; a complex kernel may have
2–3 (e.g., C2V + V2C).

**Persistent across tasks.** Like hardware VEC_FIFO, the `SharedState` is not
reset between task dispatches. Tasks on the same cluster share the same ring
buffer, enabling cross-task pipelining.

**Bounded by SlotNum.** The ring buffer has `SlotNum` slots (a compile-time
constant, e.g., 2 for bgemm). When all slots are occupied, the producer
(TPUSH) blocks until a consumer (TPOP + TFREE) frees a slot.

### Pipe Key

Each `TPipe<FlagID, DirType, SlotSize, SlotNum, LocalSlotNum>` instance
produces a unique 64-bit `pipe_key`:

```text
pipe_key = FlagID:8 | DirType:8 | SlotNum:8 | LocalSlotNum:8 | SlotSize:32
```

Combined with `cluster_id`, this uniquely identifies a `SharedState` within a
device. The `SharedState` size (`sizeof(SharedStateStorage)`) depends on
`SlotSize` and `SlotNum`, so different `pipe_key` values always map to
different allocations.

## Cross-Task Pipelining

The simulation correctly supports AIC running ahead of AIV across task
boundaries:

```text
task N:   AIC TPUSH → slot 0, occupied=1
task N+1: AIC TPUSH → slot 1, occupied=2
task N+2: AIC TPUSH → allocate() blocks (occupied >= SlotNum)

          AIV TPOP task N  → reads slot 0
          AIV TFREE task N → occupied=1, AIC unblocked
          AIV TPOP task N+1 → reads slot 1
          ...
```

This matches hardware behavior: the VEC_FIFO is a persistent physical channel,
and back-pressure naturally throttles the producer when the FIFO is full.

### Split Mode (TILE_UP_DOWN)

For C2V split mode, AIC pushes one full tile, and both AIV0 and AIV1 consume
their respective halves from the same slot:

1. AIC `record()` sets `remaining_consumers[slot] = 2`
2. AIV0 `wait()` + `free()` → remaining: 2→1
3. AIV1 `wait()` + `free()` → remaining: 1→0, slot released

The slot is not freed until **both** AIVs have consumed it. This matches the
hardware behavior where both AIV sub-cores read from the same VEC_FIFO entry.

## Platform Integration

### Context Plumbing

The simulation platform sets up per-thread identity without any runtime
involvement:

1. **`aicore_execute_wrapper`** (in `kernel.cpp`): computes `cluster_id` and
   `subblock_id` from `physical_core_id` and `core_type`, sets them in
   per-thread TLS via function pointer injection from `cpu_sim_context.cpp`.

2. **`DeviceRunner::upload_kernel_binary`**: after dlopen'ing each kernel SO,
   injects `pto_sim_get_subblock_id` and `pto_sim_get_pipe_shared_state`
   function pointers via `pto_sim_register_hooks`.

3. **pto-isa `TPush.hpp`**: calls the injected hooks to locate the correct
   `SharedState` for the current cluster and pipe configuration.

### Core Layout

```text
physical_core_id [0..block_dim-1]           = AIC of cluster i
physical_core_id [block_dim..3*block_dim-1] = AIV pairs
                                              (AIV0_c0, AIV1_c0, AIV0_c1, AIV1_c1, ...)
```

- `cluster_id`: AIC → `physical_core_id`; AIV → `(physical_core_id - block_dim) / 2`
- `subblock_id`: AIC → 0; AIV → `(physical_core_id - block_dim) % 2`

### Lifecycle

- `DeviceRunner::run()` start: `clear_cpu_sim_shared_storage()` frees all
  `SharedState` entries for the current device.
- `DeviceRunner::finalize()`: same cleanup.
- `pto_cpu_sim_release_device()`: destroys the entire device context including
  all pipe states.

During a run, `SharedState` entries are lazily allocated on first access and
persist until the run ends. The total count is bounded by
`block_dim × pipe_type_count`, which is small (typically < 100).

## Runtime Isolation

Runtime code (`aicore_executor.h`, `aicpu_executor.cpp`) has **zero
simulation awareness**. All sim context setup is handled by the platform layer:

- No `CPU_SIM_SET_*` macros in runtime code
- No `platform_set_cpu_sim_task_cookie` calls
- No sim-specific includes in runtime headers
- Onboard platform has no empty stubs for sim functions

## Current Constraints

The simulation correctly handles:

- Multiple tasks sharing the same VEC_FIFO (cross-task pipelining)
- Back-pressure when the FIFO is full
- Split-mode consumption by two AIV sub-cores
- Multiple pipe types within a single kernel

**Current execution model constraint:** the runtime dispatches a complete
AIC+AIV task group to a cluster and waits for all cores to finish (write FIN)
before reusing that core for a new task. A kernel using TPUSH/TPOP must run
to completion — **preemption of a partially-executed C↔V kernel is not
supported**. If a kernel is interrupted mid-TPUSH/TPOP and a different
C↔V kernel is scheduled on the same cluster, the new kernel will see the
previous kernel's ring buffer state (stale slot data, incorrect producer/
consumer indices), leading to data corruption or deadlock.

This constraint matches the current hardware execution model where AIC/AIV
tasks are non-preemptible.

## Validation

Validated with `examples/a5/tensormap_and_ringbuffer/bgemm/`:

- AIC: TLOAD + TMATMUL + TPUSH (cube-to-vector via VEC_FIFO)
- AIV: TPOP + TADD + TSTORE (accumulate result from cube)
- Grid: 4×4×4, Batch: 2, block_dim: 3 (3 clusters, 128 MIX tasks)

## File Structure

| Responsibility | File |
| -------------- | ---- |
| Per-device pipe shared state + TLS | `src/common/sim_context/cpu_sim_context.cpp` |
| Per-thread core identity setup | `src/{arch}/platform/sim/aicore/kernel.cpp` |
| Hook injection into kernel SOs | `src/{arch}/platform/sim/host/device_runner.cpp` |
| pto-isa hook registration API | `pto-isa/include/pto/common/cpu_stub.hpp` |
| pto-isa TPUSH/TPOP implementation | `pto-isa/include/pto/cpu/TPush.hpp`, `TPop.hpp` |
