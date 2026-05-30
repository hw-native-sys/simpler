# Scope Stats — Per-scope Resource Usage Peaks

## 1. Background & Motivation

When a model runs out of task windows, heap, or dep/fanin pool entries,
the failure message tells you *which* resource is exhausted but not
*which scope* caused the peak. Without per-scope attribution, debugging
requires binary-searching the orchestration code to find the offending
scope — slow and error-prone.

Scope stats captures the peak resource usage (heap bytes, task
in-flight, tensormap entries) for every `PTO2_SCOPE` region, so the
output directly tells you which scope drove each resource to its
high-water mark.

## 2. Overview

- **One row per scope exit.** Peaks are sampled continuously inside the
  scope and flushed to a shared buffer on `scope_end`.
- **Per-ring breakdown.** Each ring's task allocator heap/task-window
  is tracked independently.
- **NDJSON output.** A `scope_stats.jsonl` lands under the per-task output
  prefix: line 1 is run metadata (version / fatal / dropped / total), each
  subsequent line is one per-scope record.
- **Runtime-gated.** Controlled by `--enable-scope-stats` (bit 4 of
  `enable_profiling_flag`). When off, every probe is a single bool
  load — no measurement overhead.
- **T&R runtime only.** See §6 for why.

Enable in one line:

```bash
python tests/st/<case>/test_<name>.py -p a2a3 -d 0 --enable-scope-stats
```

## 3. Architecture

### 3.1 Layering

Scope stats uses a clean platform-provides / runtime-calls pattern:

```text
platform/include/aicpu/scope_stats_collector.h
    Pure-value API declarations. No runtime types cross this boundary.

platform/src/aicpu/scope_stats_collector.cpp
    Owns all collector state (depth stack, peak arrays, shared buffer).
    Implements scope lifecycle (begin/end), peak comparison logic,
    capacity registration, and shared buffer record writes.

runtime (pto_orchestrator.cpp, pto_scheduler.h)
    Calls platform APIs at instrumentation points, passing extracted
    values (ring_id, heap_bytes, tasks_in_flight, etc.) as plain
    integers. No scope_stats source files in the runtime directory.
```

### 3.2 Platform API

Header:
[`src/a2a3/platform/include/aicpu/scope_stats_collector_aicpu.h`](../../src/a2a3/platform/include/aicpu/scope_stats_collector_aicpu.h)

All entry points are `extern "C"` and take primitive types only — no
runtime structs cross the boundary, so the same collector links into
any runtime that wants to wire it up. Symbol resolution is unconditional
(see §3.4), so callers do not need to guard the call sites.

Single-producer contract: all `*_peaks` updates use non-atomic
read-max-write and assume the orchestrator thread is the only writer.
Concurrent callers may lose peaks silently — that is acceptable for
diagnostic data and saves an atomic on the hot path.

#### Setter symbols (host → AICPU init)

```cpp
void set_scope_stats_enabled(bool enable);
void set_platform_scope_stats_base(uint64_t scope_stats_data_base);
```

`kernel.cpp` calls both at kernel entry from `KernelArgs`. `enable`
mirrors the host's `--enable-scope-stats` flag; `scope_stats_data_base`
is the device-visible address of the `ScopeStatsDataHeader` region the
host `ScopeStatsCollector` allocates in `init_scope_stats()`.
`set_platform_scope_stats_base` doubles as the device-side init: it maps
the shared header + per-instance buffer state and resets collector-local
state (the host owns the shared header / free_queue, so it is not zeroed
here). When `enable=false` every probe early-returns after one bool load —
that is the off-cost.

#### Capacity registration (runtime → AICPU init)

```cpp
void scope_stats_set_ring_capacity(int ring_id, int32_t window_cap,
                                   uint64_t heap_cap);
void scope_stats_set_tensormap_capacity(int32_t cap);
```

Called once per ring at orchestrator init / scheduler attach. Caps are
copied verbatim into the buffer header so the host JSON can render
`used/cap` ratios without a second device→host query. `ring_id` outside
`[0, PTO2_SCOPE_STATS_MAX_RING_DEPTH)` is silently dropped.

#### Scope lifecycle (runtime → AICPU per-scope)

```cpp
void scope_stats_set_pending_site(const char *file, int line);
void scope_stats_begin(int ring_id, uint64_t heap_bytes,
                       int32_t tasks_in_flight, int32_t tensormap_used);
void scope_stats_end(int ring_id, uint64_t heap_bytes,
                     int32_t tasks_in_flight, int32_t tensormap_used);
void scope_stats_on_fatal();
```

A scope costs exactly two collector calls — `begin` and `end` — each
carrying that boundary's sample for the scope's own ring. The runtime gates
both on the local weak `is_scope_stats_enabled()` stub first, so a disabled
run pays neither the cross-`.so` calls nor the cross-agent `active_count()`
read that feeds them (same idiom as `is_dep_gen_enabled`). This replaces the
former `on_begin` + `update_peaks` + `on_end` fan-out (which crossed the
`.so` ~5× per scope plus a separate `set_pending_site`).

- `begin` pushes the depth/site and **seeds** the peak with the begin
  sample. The begin sample is load-bearing: a prior same-ring scope may not
  have released its resources yet, and that residual is part of this scope's
  true high-water mark.
- `end` folds the end sample into the peak (`max`), emits one record, and
  tears down the depth/site bookkeeping.

`PTO2_SCOPE()` expansion calls `set_pending_site(__FILE__, __LINE__)` before
`begin` so `end` can stamp the record with the source location — the
basename copy (`copy_basename`) keeps the JSON readable without forcing the
host to chase a device pointer into the orchestration `.so`'s string table.
Peaks are **not** propagated to enclosing scopes: each scope samples only its
own ring (`ring_id = min(depth, MAX_RING_DEPTH-1)`), so its usage appears
only in its own record. `on_fatal` sets `header.fatal_latched`, surfacing as
`"fatal": true` in the JSON; the host treats that as "diagnostic-only past
this point" but still emits whatever records made it.

### 3.3 Comparison with other profiling subsystems

| Feature | Layer | Runtime scope | Why |
| ------- | ----- | ------------- | --- |
| PMU | platform only | all runtimes | reads hardware registers (platform) |
| L2 swimlane | platform only | all runtimes | reads AICore ring buffers (platform) |
| dep_gen | platform only | all runtimes | traces `submit_task` (runtime-agnostic) |
| tensor dump | platform only | all runtimes | dumps tensor data (platform) |
| **scope stats** | **platform API + runtime call sites** | **T&R only** | runtime extracts values, platform tracks peaks |

### 3.4 Symbol resolution flow

```text
kernel.cpp (platform, shared by all runtimes)
    ├── set_scope_stats_enabled(flag)
    └── set_platform_scope_stats_base(addr)

For host_build_graph AICPU .so:
    kernel.cpp ──links──> platform collector
    → symbols resolve, .so loads, scope_stats is enabled but
      no runtime call sites invoke begin/end → no records

For tensormap_and_ringbuffer AICPU .so:
    kernel.cpp ──links──> platform collector
    runtime call sites invoke update/capacity APIs
    → full peak tracking active
```

## 4. Data Flow

```text
Host                              AICPU (T&R runtime)
─────                             ─────────────────────
ScopeStatsCollector                platform scope_stats_collector_aicpu.cpp
  allocate header + buffer pool      set_platform_scope_stats_base(addr)
  pre-fill free_queue                  └─ map header + buffer state
  set kernel_args fields             set_scope_stats_enabled(true)
  start mgmt + poll threads          scope_stats_aicpu_set_orch_thread_idx()
  launch kernel                      runtime: scope_stats_set_ring_capacity()
      │                              runtime: scope_stats_set_tensormap_capacity()
      │                                  │
  mgmt thread:                       on PTO2_SCOPE begin:
   refill free_queue,                  runtime: scope_stats_begin()
   drain ready_queue   ◀──ready──┐       └─ seed peak with begin sample
      │                          │   on PTO2_SCOPE end:
  poll thread:                   │     runtime: scope_stats_end()
   append records to memory      │       ├─ fold end sample, emit record
      │                          │       └─ append to current buffer;
      │                          └────────── push full buffer to ready_queue
      │                                  │
  stop() (drain + join)              orch exit: scope_stats_aicpu_flush_buffers()
  reconcile_counters()
  write_jsonl()
```

## 5. Output: `scope_stats.jsonl`

The host streams full buffers off the device during the run and, after
`stop()` + `reconcile_counters()`, emits `<output_prefix>/scope_stats.jsonl`
(NDJSON). Line 1 is run metadata; each subsequent line is one record.
Schema (version 3):

```json
{"version": 3, "fatal": false, "dropped": 0, "total": 2}
{"site": "example_orchestration.cpp:77", "depth": 1, "ring": 1, "task_window": 4, "heap": "8192/268435456", "tensormap": "5/65536"}
{"site": "kernel.cpp:80", "depth": 0, "ring": 0, "task_window": 1, "heap": "4096/268435456", "tensormap": "5/65536"}
```

Metadata line (line 1) fields:

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `version` | int | Always `3` |
| `fatal` | bool | `true` iff `scope_stats_on_fatal()` fired during the run |
| `dropped` | uint | Records dropped on device (free_queue empty / ready_queue full); `0` on a healthy run |
| `total` | uint | Total `scope_end` events the device attempted to record (collected + dropped) |

Per-record lines: one JSON object per `scope_end`, oldest-first. The
effective record count is disk-bounded — full buffers stream off the
device continuously, so there is no fixed-ring wrap. Fields:

| Field | Type | Description |
| ----- | ---- | ----------- |
| `site` | `"basename:line"` | Source location of the `PTO2_SCOPE()` call |
| `depth` | int | Nesting depth (0 = root scope inside the executor) |
| `ring` | int | Ring this scope used (`min(depth, MAX_RING_DEPTH-1)`); indexes the cap arrays for `heap` |
| `task_window` | int | Peak task-window slots in use on `ring` |
| `heap` | `"used/cap"` | Peak heap bytes in use on `ring` |
| `tensormap` | `"used/cap"` | Peak tensormap entries in use |

A scope only ever touches its own `ring`, so a single ring's peak is
recorded rather than a per-ring array (the other rings were always zero).
The `cap` denominators come from `scope_stats_set_ring_capacity` /
`scope_stats_set_tensormap_capacity` snapshots, so they always reflect
the values the runtime actually configured for that run.

A worked example is in
[`tests/st/a2a3/tensormap_and_ringbuffer/dfx/scope_stats/test_scope_stats.py`](../../tests/st/a2a3/tensormap_and_ringbuffer/dfx/scope_stats/test_scope_stats.py)
— it runs the `vector_example` orchestration with `--enable-scope-stats`
and asserts the resulting NDJSON for the depth=0 / depth=1 records the
outer-executor + inner `PTO2_SCOPE` produce.

## 6. Future: Cross-runtime Support

If host_build_graph adds scope-like concepts in the future, extending
scope_stats only requires adding the same platform API call sites in
the HBG runtime — no platform changes needed. The platform collector
is already runtime-agnostic; it accepts plain values and has no
knowledge of T&R types.
