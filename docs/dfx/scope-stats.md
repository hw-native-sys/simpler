# Scope Stats — Per-scope Resource Usage Peaks

Scope stats records the peak resource usage — task-window slots, heap
bytes, tensormap entries — for every `PTO2_SCOPE` region in an
orchestration, so you can see *which scope* drove each resource to its
high-water mark. When a model runs out of task windows, heap, or
tensormap entries, the failure tells you *which* resource is exhausted
but not *where*; scope stats gives you the where.

It is a diagnostic-only, opt-in feature for the
**tensormap_and_ringbuffer (T&R)** runtime. When disabled (the default)
it costs a single bool load per probe.

## 1. Quick Start

The full workflow is three steps: enable the flag, run, then turn the
resulting `scope_stats.jsonl` into an HTML report.

### Step 1 — Run with `--enable-scope-stats`

Pass the flag to any T&R example or scene test:

```bash
python tests/st/<case>/test_<name>.py -p a2a3 -d 0 --enable-scope-stats
```

The flag is bit 4 of `enable_profiling_flag`; on a T&R run it turns on
per-scope peak tracking. On other runtimes the flag is accepted but
produces no records.

### Step 2 — Locate the output

The run writes `scope_stats.jsonl` under the per-task output prefix
(the same directory other profiling artifacts land in for that run).
It is NDJSON: line 1 is run metadata, every subsequent line is one
scope sample. See [§3](#3-output-scope_statsjsonl) for the schema.

### Step 3 — Visualize with `scope_stats_plot.py`

```bash
python tools/scope_stats_plot.py <output_prefix>/scope_stats.jsonl
# writes <output_prefix>/scope_stats.html

# or send the report elsewhere:
python tools/scope_stats_plot.py path/to/scope_stats.jsonl --out-dir /tmp/report
```

| Argument | Required | Meaning |
| -------- | -------- | ------- |
| `jsonl` | yes | Path to a `scope_stats.jsonl` produced by Step 1 |
| `--out-dir DIR` | no | Where to write `scope_stats.html` (default: next to the input) |

The output is a single self-contained `scope_stats.html` — the SVG
charts are inlined (no matplotlib, no external JS/CDN), so it opens
offline and is trivial to share. Open it in any browser.

### Reading the report

The report groups charts by **ring**, and within each ring draws one
line chart per resource (task_window, heap, tensormap). The x-axis is
scopes in begin/end order; hover a point to see its source site. Each
ring header lists the per-resource max capacity, and three metrics are
plotted per ring/resource:

| Metric | Formula | What it tells you |
| ------ | ------- | ----------------- |
| `scope_high_water` | `end.head − begin.tail` | Highest occupancy the scope held — residual not yet released on entry, plus what this scope added. The true peak. |
| `real_occupancy` | `end.head − end.tail` | What is still occupied at scope exit (the live level on leaving). |
| `scope_alloc` | `end.head − begin.head` | How far the allocation frontier advanced — this scope's own net (unreleased) allocation. |

`head` is the ring's allocation frontier (`heap_top`); `tail` is its
released boundary (`heap_tail`). Ring-buffer occupancy is non-negative,
so a negative delta is folded back by one buffer length
(`(head + cap − tail) % cap`) — a wrap past the buffer end. `tensormap`
is reported as a single in-use value (no head/tail), so it shows only
the `real_occupancy` curve.

## 2. What gets captured

- **The whole orchestration, automatically.** You do not mark a region
  to profile — instrumentation lives inside the `PTO2_SCOPE` macro
  itself, so *every* scope in the orchestration is recorded once the
  flag is on. The executor wraps the orchestration entry in a root
  `PTO2_SCOPE` (`depth` 0), and every user-written `PTO2_SCOPE` nests
  under it, so the report covers the entire orch scope tree end to end.
- **One sample per scope boundary.** Each `PTO2_SCOPE` emits a `begin`
  record on entry and an `end` record on exit. The plot tool pairs them
  by site to compute the metrics above.
- **Per-ring.** Each ring's task allocator heap / task-window is tracked
  independently; a scope only ever touches its own ring
  (`ring = min(depth, MAX_RING_DEPTH−1)`).
- **Capacity denominators.** The `*_max` capacities in the metadata line
  are snapshots of what the runtime actually configured for that run, so
  `used/cap` ratios are exact.
- **Disk-bounded, no wrap.** Full buffers stream off the device during
  the run, so the record count is bounded only by disk — there is no
  fixed-ring overwrite.

## 3. Output: `scope_stats.jsonl`

NDJSON. Line 1 is run metadata; each subsequent line is one scope
sample (`begin` or `end`). Schema version 4:

```json
{"version": 4, "fatal": false, "dropped": 0, "total": 4, "task_window_max": [8, 4], "heap_max": [268435456, 268435456], "tensormap_max": 65536}
{"site": "example_orchestration.cpp:77", "phase": "begin", "depth": 1, "ring": 1, "task_window_start": 0, "task_window_end": 0, "heap_start": 0, "heap_end": 0, "tensormap": 0}
{"site": "example_orchestration.cpp:77", "phase": "end", "depth": 1, "ring": 1, "task_window_start": 0, "task_window_end": 4, "heap_start": 0, "heap_end": 8192, "tensormap": 5}
```

Metadata line (line 1):

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `version` | int | Schema version (`4`) |
| `fatal` | bool | `true` iff a fatal was latched during the run; records past it are diagnostic-only |
| `dropped` | uint | Records dropped on device (free_queue empty / ready_queue full); `0` on a healthy run |
| `total` | uint | Total records the device attempted (collected + dropped) |
| `task_window_max` | int[] | Per-ring task-window capacity (indexed by `ring`) |
| `heap_max` | int[] | Per-ring heap-byte capacity (indexed by `ring`) |
| `tensormap_max` | int | Tensormap entry capacity (scalar) |

Per-sample lines, oldest-first:

| Field | Type | Description |
| ----- | ---- | ----------- |
| `site` | `"basename:line"` | Source location of the `PTO2_SCOPE()` call |
| `phase` | `"begin"`/`"end"` | Scope entry or exit sample |
| `depth` | int | Nesting depth (0 = root scope inside the executor) |
| `ring` | int | Ring this scope used; indexes `task_window_max` / `heap_max` |
| `task_window_start` | int | Task-window ring tail at this boundary |
| `task_window_end` | int | Task-window ring head at this boundary |
| `heap_start` | uint | Heap ring tail (released boundary) in bytes |
| `heap_end` | uint | Heap ring head (allocation frontier) in bytes |
| `tensormap` | int | Tensormap entries in use |

`start`/`end` are the ring's tail/head pointers at that boundary — see
the metric formulas in [§1](#reading-the-report) for how a scope's peak
and occupancy are derived from a paired begin/end.

---

## 4. Internals

This section is for maintainers wiring scope stats into a runtime; users
do not need it. There is **no public C/C++ API** — the only external
interfaces are the `--enable-scope-stats` flag and the plot tool above.

### 4.1 Layering

Scope stats uses a platform-provides / runtime-calls pattern:

```text
platform/include/aicpu/scope_stats_collector_aicpu.h
    Pure-value API declarations. No runtime types cross this boundary.

platform/src/aicpu/scope_stats_collector_aicpu.cpp
    Owns all collector state (depth stack, peak arrays, shared buffer).
    Scope lifecycle, peak comparison, capacity registration, record writes.

platform/src/host/scope_stats_collector.cpp
    Host side: allocates the shared header/buffer pool, streams full
    buffers off the device, reconciles counters, writes scope_stats.jsonl.

runtime (pto_orchestrator.cpp, pto_scheduler.h)
    Calls platform APIs at instrumentation points, passing extracted
    values (ring_id, heap_bytes, ...) as plain integers. No scope_stats
    source files live in the runtime directory.
```

### 4.2 AICPU platform API

Header:
[`src/a2a3/platform/include/aicpu/scope_stats_collector_aicpu.h`](../../src/a2a3/platform/include/aicpu/scope_stats_collector_aicpu.h)

All entry points are `extern "C"` and take primitive types only, so the
same collector links into any runtime that wires it up. Symbol
resolution is unconditional (see §4.4), so call sites need no guards.

Single-producer contract: all `*_peaks` updates use non-atomic
read-max-write and assume the orchestrator thread is the only writer.
Concurrent callers may lose peaks silently — acceptable for diagnostic
data, and it saves an atomic on the hot path.

```cpp
// Host → AICPU init (called from kernel.cpp at kernel entry)
void set_scope_stats_enabled(bool enable);
void set_platform_scope_stats_base(uint64_t scope_stats_data_base);

// Runtime → AICPU init (once per ring at orchestrator init / scheduler attach)
void scope_stats_set_ring_capacity(int ring_id, int32_t window_cap, uint64_t heap_cap);
void scope_stats_set_tensormap_capacity(int32_t cap);

// Runtime → AICPU per-scope
void scope_stats_set_pending_site(const char *file, int line);
void scope_stats_begin(int ring_id, uint64_t heap_bytes, int32_t tasks_in_flight, int32_t tensormap_used);
void scope_stats_end(int ring_id, uint64_t heap_bytes, int32_t tasks_in_flight, int32_t tensormap_used);
void scope_stats_on_fatal();
```

`enable` mirrors the host's `--enable-scope-stats` flag;
`scope_stats_data_base` is the device-visible address of the shared
header the host allocates. `set_platform_scope_stats_base` doubles as
device-side init (maps the header + per-instance buffer state, resets
collector-local state). When `enable=false` every probe early-returns
after one bool load.

A scope costs exactly two collector calls — `begin` and `end` — each
carrying that boundary's sample for the scope's own ring. The runtime
gates both on the local weak `is_scope_stats_enabled()` stub first, so a
disabled run pays neither the cross-`.so` calls nor the cross-agent
`active_count()` read (same idiom as `is_dep_gen_enabled`).
`PTO2_SCOPE()` expansion calls `set_pending_site(__FILE__, __LINE__)`
before `begin`; `copy_basename` keeps the JSON readable without forcing
the host to chase a device pointer into the orchestration `.so`'s string
table. `on_fatal` sets `header.fatal_latched`, surfacing as
`"fatal": true`; the host still emits whatever records made it.

`ring_id` outside `[0, PTO2_SCOPE_STATS_MAX_RING_DEPTH)` is silently
dropped. Caps are copied verbatim into the buffer header so the host can
render `used/cap` without a second device→host query.

### 4.3 Comparison with other profiling subsystems

| Feature | Layer | Runtime scope | Why |
| ------- | ----- | ------------- | --- |
| PMU | platform only | all runtimes | reads hardware registers |
| L2 swimlane | platform only | all runtimes | reads AICore ring buffers |
| dep_gen | platform only | all runtimes | traces `submit_task` |
| tensor dump | platform only | all runtimes | dumps tensor data |
| **scope stats** | **platform API + runtime call sites** | **T&R only** | runtime extracts values, platform tracks peaks |

### 4.4 Symbol resolution

`kernel.cpp` (platform, shared by all runtimes) always calls
`set_scope_stats_enabled` / `set_platform_scope_stats_base`, so the
collector symbols resolve into every AICPU `.so`. Only the T&R runtime
adds the `begin`/`end`/capacity call sites, so only it produces records;
host_build_graph links the collector but never invokes it.

### 4.5 Data flow

```text
Host                              AICPU (T&R runtime)
─────                             ─────────────────────
ScopeStatsCollector                platform scope_stats_collector_aicpu.cpp
  allocate header + buffer pool      set_platform_scope_stats_base(addr)
  pre-fill free_queue                set_scope_stats_enabled(true)
  set kernel_args fields             runtime: scope_stats_set_ring_capacity()
  launch kernel                      runtime: scope_stats_set_tensormap_capacity()
      │                                  │
  poll thread:                       on PTO2_SCOPE begin/end:
   append records to memory  ◀──┐      runtime: scope_stats_begin()/end()
      │                         │         └─ emit record, append to buffer;
  stop() (drain + join)         └──────────── push full buffer to ready_queue
  reconcile_counters()               orch exit: flush remaining buffers
  write_jsonl()
```

A worked example is in
[`tests/st/a2a3/tensormap_and_ringbuffer/dfx/scope_stats/test_scope_stats.py`](../../tests/st/a2a3/tensormap_and_ringbuffer/dfx/scope_stats/test_scope_stats.py)
— it runs the `vector_example` orchestration with `--enable-scope-stats`
and asserts the resulting NDJSON.

### 4.6 Future: cross-runtime support

If host_build_graph adds scope-like concepts, extending scope_stats only
requires adding the same platform call sites in HBG — no platform
changes. The collector is already runtime-agnostic: it accepts plain
values and has no knowledge of T&R types.
