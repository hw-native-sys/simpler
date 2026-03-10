# PTO Runtime2 Profiling Levels

This document describes the profiling macro hierarchy and logging control in the PTO Runtime2 system.

## Overview

PTO Runtime2 uses a hierarchical profiling system with compile-time macros to control profiling code compilation and log output. The `enable_profiling` runtime flag controls data collection (performance buffers, shared memory writes) but does NOT control log output.

## Profiling Macro Hierarchy

```
PTO2_PROFILING (base level, default=0)
├── PTO2_ORCH_PROFILING (orchestrator, default=0, requires PTO2_PROFILING=1)
├── PTO2_SCHED_PROFILING (scheduler, default=0, requires PTO2_PROFILING=1)
└── PTO2_TENSORMAP_PROFILING (tensormap, default=0, requires PTO2_PROFILING=1)
```

### Compile-Time Validation

Each sub-level macro requires `PTO2_PROFILING=1`:

```cpp
#if PTO2_ORCH_PROFILING && !PTO2_PROFILING
#error "PTO2_ORCH_PROFILING requires PTO2_PROFILING=1"
#endif

#if PTO2_SCHED_PROFILING && !PTO2_PROFILING
#error "PTO2_SCHED_PROFILING requires PTO2_PROFILING=1"
#endif

#if PTO2_TENSORMAP_PROFILING && !PTO2_PROFILING
#error "PTO2_TENSORMAP_PROFILING requires PTO2_PROFILING=1"
#endif
```

## Profiling Levels

### Level 0: No Profiling (PTO2_PROFILING=0)

**What's compiled:**
- Debug/diagnostic logs (always present)
- Progress tracking
- Stall detection
- Deadlock/livelock detection

**What's NOT compiled:**
- All profiling counters
- All profiling logs
- Performance data collection

**Log output:** 11 DEV_ALWAYS logs (debug/diagnostic only)

---

### Level 1: Basic Profiling (PTO2_PROFILING=1)

**What's compiled:**
- All profiling counters (cycles, task counts, loop counts)
- Basic profiling summaries
- Scheduler summary output
- Orchestration completion time

**What's NOT compiled:**
- Detailed phase breakdowns
- TensorMap statistics

**Log output:** 13 DEV_ALWAYS logs
- 11 debug/diagnostic logs (always present)
- 2 basic profiling summaries:
  - Orchestration completion time
  - Total submitted tasks

**Scheduler output:**
```
Thread X: Scheduler summary: total_time=XXXus, loops=XXX, tasks_scheduled=XXX
```

**Note:** Scheduler summary always prints when `PTO2_PROFILING=1`, regardless of `enable_profiling` flag.

---

### Level 2: Scheduler Detailed Profiling (PTO2_SCHED_PROFILING=1)

**Requires:** `PTO2_PROFILING=1`

**What's compiled:**
- All Level 1 features
- Detailed scheduler phase counters
- Phase-specific statistics (complete, scan, dispatch, idle)
- Hit rate tracking (complete poll, ready queue pop)

**Log output:** 18 DEV_ALWAYS logs (11 debug + 2 basic + 7 scheduler detailed - 2 replaced)
- Replaces scheduler summary with detailed breakdown

**Scheduler output:**
```
Thread X: === Scheduler Phase Breakdown: total=XXXus, XXX tasks ===
Thread X:   complete       : XXXus (XX.X%)  [fanout: edges=XXX, max_degree=X, avg=X.X]  [fanin: edges=XXX, max_degree=X, avg=X.X]
Thread X:     poll         : XXXus (XX.X%)  hit=XXX, miss=XXX, hit_rate=XX.X%
Thread X:     otc_lock     : XXXus (XX.X%)  work=XXXus wait=XXXus  atomics=XXX
Thread X:     otc_fanout   : XXXus (XX.X%)  work=XXXus wait=XXXus  atomics=XXX
Thread X:     otc_fanin    : XXXus (XX.X%)  atomics=XXX
Thread X:     otc_self     : XXXus (XX.X%)  atomics=XXX
Thread X:     perf         : XXXus (XX.X%)
Thread X:   dispatch       : XXXus (XX.X%)  [pop: hit=XXX, miss=XXX, hit_rate=XX.X%]
Thread X:     poll         : XXXus (XX.X%)
Thread X:     pop          : XXXus (XX.X%)  work=XXXus wait=XXXus  atomics=XXX
Thread X:     setup        : XXXus (XX.X%)
Thread X:   scan           : XXXus (XX.X%)
Thread X:   idle           : XXXus (XX.X%)
Thread X:   avg/complete   : XXXus
Thread X: Scheduler summary: total_time=XXXus, loops=XXX, tasks_scheduled=XXX
```

---

### Level 3: Orchestrator Detailed Profiling (PTO2_ORCH_PROFILING=1)

**Requires:** `PTO2_PROFILING=1`

**What's compiled:**
- All Level 1 features
- Detailed orchestrator phase counters
- Per-phase cycle tracking
- Atomic operation counters
- Wait time tracking

**Log output:** 30 DEV_ALWAYS logs (11 debug + 2 basic + 1 scheduler summary + 17 orchestrator detailed - 1 replaced)
- Replaces basic orchestration completion with detailed breakdown

**Orchestrator output:**
```
Thread X: === Orchestrator Profiling: XXX tasks, total=XXXus ===
Thread X:   sync_tensormap : XXXus (XX.X%)
Thread X:   task_ring_alloc: XXXus (XX.X%)  work=XXXus wait=XXXus  atomics=XXX
Thread X:   param_copy     : XXXus (XX.X%)  atomics=XXX
Thread X:   lookup+dep     : XXXus (XX.X%)
Thread X:   heap_alloc     : XXXus (XX.X%)  work=XXXus wait=XXXus  atomics=XXX
Thread X:   tensormap_ins  : XXXus (XX.X%)
Thread X:   fanin+ready    : XXXus (XX.X%)  work=XXXus wait=XXXus  atomics=XXX
Thread X:   finalize+SM    : XXXus (XX.X%)  work=XXXus wait=XXXus  atomics=XXX
Thread X:   scope_end      : XXXus  atomics=XXX
Thread X:   avg/task       : XXXus
```

**Note:** Orchestrator logs always print when `PTO2_ORCH_PROFILING=1`, regardless of `enable_profiling` flag.

---

### Level 4: TensorMap Profiling (PTO2_TENSORMAP_PROFILING=1)

**Requires:** `PTO2_PROFILING=1` AND `PTO2_ORCH_PROFILING=1`

**What's compiled:**
- All Level 3 features
- TensorMap lookup statistics
- Hash chain walk tracking
- Overlap check counters

**Log output:** 34 DEV_ALWAYS logs (30 from Level 3 + 4 tensormap)

**TensorMap output:**
```
Thread X: === TensorMap Lookup Stats ===
Thread X:   lookups        : XXX, inserts: XXX
Thread X:   chain walked   : total=XXX, avg=X.X, max=X
Thread X:   overlap checks : XXX, hits=XXX (XX.X%)
```

---

## Runtime Flag: enable_profiling

The `runtime->enable_profiling` flag controls **data collection**, NOT log output.

### When enable_profiling=true:
- Performance buffers are allocated and written
- Per-task timing data is collected
- Phase profiling data is recorded
- Orchestrator summary is written to shared memory

### When enable_profiling=false:
- No performance data collection
- No shared memory writes
- Logs still print (controlled by macros only)

### Usage:
```cpp
// Initialize runtime with profiling enabled
runtime->enable_profiling = true;
```

---

## Common Profiling Configurations

### Development (default)
```bash
# No profiling overhead
PTO2_PROFILING=0
```

### Basic Performance Monitoring
```bash
# Minimal overhead, summary logs only
PTO2_PROFILING=1
PTO2_ORCH_PROFILING=0
PTO2_SCHED_PROFILING=0
```

### Scheduler Performance Analysis
```bash
# Detailed scheduler breakdown
PTO2_PROFILING=1
PTO2_ORCH_PROFILING=0
PTO2_SCHED_PROFILING=1
```

### Orchestrator Performance Analysis
```bash
# Detailed orchestrator breakdown
PTO2_PROFILING=1
PTO2_ORCH_PROFILING=1
PTO2_SCHED_PROFILING=0
```

### Full Profiling (maximum overhead)
```bash
# All profiling features enabled
PTO2_PROFILING=1
PTO2_ORCH_PROFILING=1
PTO2_SCHED_PROFILING=1
PTO2_TENSORMAP_PROFILING=1
```

---

## Setting Profiling Macros

### At compile time:
```bash
# In CMakeLists.txt or build command
add_definitions(-DPTO2_PROFILING=1)
add_definitions(-DPTO2_ORCH_PROFILING=1)
```

### In source code (before including headers):
```cpp
#define PTO2_PROFILING 1
#define PTO2_ORCH_PROFILING 1
#include "pto_runtime2_types.h"
```

---

## Log Output Summary

| Level | Macro Settings | DEV_ALWAYS Count | Description |
|-------|---------------|------------------|-------------|
| 0 | `PTO2_PROFILING=0` | 11 | Debug/diagnostic only |
| 1 | `PTO2_PROFILING=1` | 13 | Basic summaries |
| 2 | `+PTO2_SCHED_PROFILING=1` | 18 | Scheduler detailed |
| 3 | `+PTO2_ORCH_PROFILING=1` | 30 | Orchestrator detailed |
| 4 | `+PTO2_TENSORMAP_PROFILING=1` | 34 | TensorMap stats |

---

## Implementation Notes

### Key Principles

1. **Macros control compilation and logging**
   - `#if PTO2_PROFILING` controls whether profiling code is compiled
   - Logs print when macro is enabled, regardless of runtime flag

2. **Runtime flag controls data collection**
   - `enable_profiling` controls performance buffer allocation
   - Controls shared memory writes for host-side export
   - Does NOT control log output

3. **Consistent behavior across components**
   - Scheduler logs: macro-controlled only
   - Orchestrator logs: macro-controlled only
   - Data collection: runtime flag controlled

### Code Locations

- Macro definitions: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2_types.h`
- Scheduler profiling: `src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp` (lines 770-835)
- Orchestrator profiling: `src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp` (lines 1035-1105)
- TensorMap profiling: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_tensormap.h`

---

## Performance Impact

### Compilation overhead:
- Level 0: No overhead
- Level 1: Minimal (counter increments, basic arithmetic)
- Level 2-4: Low to moderate (additional counters, cycle measurements)

### Runtime overhead:
- Logging: Negligible (device logs are asynchronous)
- Data collection (`enable_profiling=true`): Low to moderate
  - Performance buffer writes
  - Shared memory updates
  - Per-task timing measurements

### Recommendation:
- Use Level 0 for production
- Use Level 1-2 for performance monitoring
- Use Level 3-4 for detailed performance analysis only
