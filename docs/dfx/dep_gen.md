# dep_gen — Complete Per-Submit Dependency Graph (Tensor-Annotated)

## 1. Background & Motivation

The swimlane profiler's per-task `fanout[]` array is the obvious place to
read "which tasks did task X feed into?" — but it is **structurally
incomplete on real hardware**.

Each producer task carries its own `L2SwimlaneAicpuTaskRecord.fanout[RUNTIME_MAX_FANOUT]`,
populated by the AICPU scheduler at the moment it wires a downstream
consumer. If a producer has already finished and transitioned to
`PTO2_TASK_COMPLETED` by the time a later submit wants to register a
dependency on it, the consumer's edge has nowhere to go — the record is
sealed, the slot is closed, and the edge is silently dropped. This is
not a bug in fanout itself; fanout is "successors known at runtime", not
"successors discoverable from the orchestrator's input". Race window in
[#599](https://github.com/hw-native-sys/simpler/issues/599); see also the
PR #500 archive for the historical attempt to fix this in-place.

`dep_gen` sidesteps the race by capturing the **inputs** to every
`Orchestrator::submit_task` call into a host-resident record stream
(no on-disk hop) and replaying them offline through the same
`compute_task_fanin` / `register_task_outputs` primitives the device
orchestrator uses. The host replay sees every submit — there is no
"already retired" producer because nothing retires during replay. The
output, `deps.json`, is a strict superset of fanout: every fanout edge
appears in deps.json, and the edges fanout dropped due to the race
appear too.

---

## 2. Overview

- **Capture point.** `pto_orchestrator::submit_task` writes a
  `DepGenRecord` (task_id, scope flag, tensor blobs, arg types,
  explicit deps) into a per-thread shared-memory ring buffer for every
  call when `enable_dep_gen` is on.
- **In-memory drain.** The host `DepGenCollector` (background mgmt
  thread, ProfilerBase machinery shared with PMU / L2 Perf / Tensor
  Dump) drains the ring into a `std::vector<DepGenRecord>` resident on
  the runner. No `submit_trace.bin` lands on disk — the host already
  has the records once the run ends, and going through the filesystem
  would just be extra I/O.
- **Host replay (dual-pass, self-checking).** After `reconcile_counters()`
  confirms a clean trace (no drops, no leftovers),
  `dep_gen_replay_emit_deps_json` runs every record back through *two*
  parallel host-resident `PTO2TensorMap` instances that evolve in lockstep:
  - **Oracle pass** drives the canonical `compute_task_fanin` template
      from `pto_dep_compute.h` and collects the producer-id set the
      runtime would have emitted.
  - **Annotated pass** runs an inlined mirror of STEP A
      (creator retention) + STEP B (tensormap lookup) against the second
      map, with a wider callback so each edge gets recorded with its
      tensor metadata (producer/consumer shape + offset, dtype, version).
  Per-record semantics mirror runtime `submit_task` exactly: STEP 1
  (explicit deps), STEP 3 (creator retention + tensormap lookup),
  STEP 4 (register outputs). Per-successor dedup matches
  `PTO2FaninBuilder::append_fanin_or_fail`. After both passes finish per
  record, the replay asserts the two producer-id sets are equal; if they
  diverge, `deps.json` is not written and the function returns non-zero.
  This is the guarantee against silent shotgun modifications — anyone
  who changes `compute_task_fanin` semantics will trip the gate
  immediately and know to update the annotated mirror.
- **Output.** `<output_prefix>/deps.json` — strided-Tensor schema with
  `tasks[]`, `tensors[]`, and tensor-annotated `edges[]` (see §4).

---

## 3. How to Enable

`dep_gen` is gated by `CallConfig.enable_dep_gen` (alongside
`enable_l2_swimlane`, `enable_dump_tensor`, `enable_pmu`). The CLI flag
is `--enable-dep-gen`:

```bash
# Standalone
python test_my_case.py --platform <a2a3|a5> --enable-dep-gen --enable-l2-swimlane

# Pytest
pytest tests/st/... --platform <a2a3|a5> --enable-dep-gen --enable-l2-swimlane
```

The `--enable-l2-swimlane` flag is independent but recommended in pair
because:

- `deps.json` is the dep_gen artifact.
- `l2_swimlane_records.json` (from swimlane) is the timing artifact;
  `merged_swimlane.json` (the Perfetto trace) uses `deps.json` for
  dependency arrows when both files exist.
- The "fanout ⊆ deps" validation gate fires only when both files are
  present.

For perf-sensitive runs where you'd rather measure each profiler in
isolation, see the **split workflow** described in
[l2-swimlane-profiling §3.5](l2-swimlane-profiling.md#35-dependency-arrows-from-dep_gen)
— one dep_gen capture per topology, then any number of swimlane
runs that the converter joins back to that captured graph.

When `--enable-dep-gen` is on with any other diagnostic flag, an
`output_prefix` directory must be set (the runtime throws otherwise).
The standard SceneTest path
(`outputs/<TestName>_<case>_<timestamp>/`) handles that automatically.

---

## 4. Output: `deps.json`

```json
{
  "tasks": [
    {"task_id": "0",          "scope": "auto", "args": []},
    {"task_id": "4294967296", "scope": "auto", "args": [
      {"idx": 0, "type": "INPUT", "tensor_id": "13451765318376212391",
       "dtype": "FLOAT32", "shape": [16384],
       "start_offset": "0", "strides": [1]}
    ]}
  ],
  "tensors": [
    {"tensor_id": "13451765318376212391",
     "buffer_addr": "29204938752", "version": 0,
     "dtype": "FLOAT32", "buffer_numel": "16384"}
  ],
  "edges": [
    {"pred": "0", "succ": "4294967296", "arg": 0, "source": "creator",
     "tensor_id": "13451765318376212391", "consumer_dtype": "FLOAT32",
     "consumer_shape": [16384],
     "consumer_start_offset": "0", "consumer_strides": [1]},
    {"pred": "4294967296", "succ": "4294967298", "arg": 0, "source": "tensormap",
     "overlap": "covered",
     "tensor_id": "9514117477438350967", "consumer_dtype": "FLOAT32",
     "consumer_shape": [16384],
     "consumer_start_offset": "0", "consumer_strides": [1],
     "producer_shape": [16384],
     "producer_start_offset": "0", "producer_strides": [1]}
  ]
}
```

All 64-bit unsigned fields (`task_id`, `tensor_id`, `pred`, `succ`,
`buffer_addr`) are serialized as JSON **strings**, not numbers. Many
JavaScript-based JSON parsers can only safely represent integers up to
`Number.MAX_SAFE_INTEGER` (2^53 − 1); `tensor_id` (FNV-1a hash) and
`buffer_addr` (hardware address) routinely exceed that limit and would
silently lose precision if encoded as numbers. Python consumers pass
these through `int(v)` which accepts either form, so the schema is
JS-safe without burdening Python.

Task ids encode `(ring_id << 32) | local_id` — the same layout as
`PTO2TaskId::raw`:

```python
ring = (raw >> 32) & 0xFF
local = raw & 0xFFFFFFFF
```

### `tasks[]`

One entry per task observed in the trace. `scope` is `"manual"` when the
submit happened inside a manual scope (no automatic dependency wiring)
and `"auto"` otherwise. Tools that only need task-pair edges can ignore
this block.

### `tensors[]`

One entry per unique `(buffer_addr, version)` pair touched by the trace.
`tensor_id` is a stable FNV-1a 64-bit hash of that pair — identical
inputs across runs yield the same id, making `deps.json` files diffable.
`buffer_numel` is the element count of the **underlying buffer**, not the
slice; per-edge slice geometry (`shape` + `start_offset` + `strides`)
lives in the `edges[]` entries.

### `edges[]`

Each edge is `{pred, succ}` plus annotation. Fields:

| Field | Type | When present | Meaning |
| ----- | ---- | ------------ | ------- |
| `pred`, `succ` | uint64 (string) | always | `PTO2TaskId::raw` of producer and consumer |
| `arg` | int32 | always | Consumer's arg-slot index; `-1` for `explicit` source |
| `source` | string | always | `explicit` (from `explicit_deps[]`), `creator` (`owner_task_id` retention), or `tensormap` (overlap lookup hit) |
| `overlap` | string | `source=tensormap` | `covered` (producer slice fully contains consumer slice) or `other` |
| `tensor_id` | uint64 (string) | not `explicit` | Identity of the underlying tensor; cross-references `tensors[]` |
| `consumer_dtype` | string | not `explicit` | Element type the consumer reads as |
| `consumer_shape` | uint32 array | not `explicit` | Per-dim element count of the consumer slice |
| `consumer_start_offset` | uint64 (string) | not `explicit` | Element offset of the consumer slice into the buffer |
| `consumer_strides` | uint32 array | not `explicit` | Per-dim stride (in elements) of the consumer slice; runtime invariant > 0 |
| `producer_shape` | uint32 array | `source=tensormap` | Per-dim element count of the producer slice |
| `producer_start_offset` | uint64 (string) | `source=tensormap` | Element offset of the producer slice |
| `producer_strides` | uint32 array | `source=tensormap` | Per-dim stride of the producer slice; runtime invariant > 0 |

A single `(pred, succ)` pair can appear in `edges[]` multiple times if
the producer drives the consumer through multiple slots, multiple
sources, or multiple tensormap matches (different slice / version). For
"is task X a successor of task Y at all?" questions, project edges down
to the `(pred, succ)` set; for "what specifically did Y feed into X?",
keep the full annotation.

`deps.json` can have `"edges":[]` when the workload's tasks have no
inter-task data dependencies (e.g. embarrassingly parallel kernels
under scope_end barriers). `tasks[]` and `tensors[]` still list every
observed task and tensor — that is not an error.

---

## 5. Visualizing — `deps_to_graph.py`

`simpler_setup/tools/deps_to_graph.py` turns `deps.json` into a
self-contained pan/zoom HTML page (Graphviz SVG + inline vanilla-JS
drag-pan + wheel-zoom). Open the file in any browser, no internet
needed:

```bash
# Newest deps.json under outputs/
python -m simpler_setup.tools.deps_to_graph

# Specific path
python -m simpler_setup.tools.deps_to_graph outputs/.../deps.json

# Big graphs: use force-directed layout (recommended >1000 nodes)
python -m simpler_setup.tools.deps_to_graph deps.json --engine sfdp

# Show per-edge tensor slice annotations on the arrows.
# Off by default — the bare task-pair graph stays readable on dense workloads;
# turn this on when you actually need to inspect which slice an edge carries.
python -m simpler_setup.tools.deps_to_graph deps.json --show-tensor-info
```

`--show-tensor-info` rewrites every task as an HTML-table node with two
compartments:

- **Top rows (blue)** are the consumed inputs (slots of type `INPUT`
  and `INOUT`).
- **Middle header** is the task identity (`(ring, local) · func_name`),
  background-colored by core_type (aic / aiv / mix / alloc).
- **Bottom rows (orange)** are the produced outputs (`INOUT`,
  `OUTPUT_EXISTING`, and `OUTPUT` slots).

Each arg row carries a 4-line block:

```text
arg<i> <ARG_TYPE>[ ?] <Tname>:<dtype>
storage:      <buffer_numel> elems   # underlying buffer size
shape:        [...]                  # slice this slot accesses
strides:      [...]                  # per-dim element strides
start_offset: <N> (elem)             # slice start in the underlying buffer
```

`<Tname>` is `T<idx>` from `tensors[]` order, so two slots referencing
the same underlying buffer share a name across the whole graph.
`raw != shape` is a visual cue that the slot accesses a sub-region.
The trailing `?` on an `OUTPUT` row marks a tensor_id that the viewer
backfilled from a downstream `creator` edge (the runtime hadn't
materialized a Tensor at `submit_task` time, so the raw blob was zero).

Edges in this mode route from the producer's `out_<idx>` row directly
to the consumer's `in_<arg>` row, so "which output of X feeds which
input of Y" reads off the picture. `explicit`-source edges render as
dashed grey arrows; `tensormap` edges whose `overlap` is not `covered`
carry a small red label so partial-overlap cases stand out.

Node visual encoding (legend top-right of the rendered HTML):

| Shape + color | Meaning |
| ------------- | ------- |
| Blue rounded box | AIC (cube) — kernel ran on the matrix unit |
| Orange ellipse | AIV (vector) — kernel ran on the vector unit |
| Green diamond | mix — single `submit_task` with `MixedKernels` spanning both core types |
| Gray dashed note | alloc — task from `alloc_tensors` (got a task_id, references downstream via `owner_task_id`, but never dispatched a kernel so has no perf record) |

Labels read as `(ring, local) · func_name · core_type-implicit-via-shape`.
When a colocated `l2_swimlane_records.json` is present the func_id is enriched
with the kernel name via the sibling `name_map_<case>.json` (written by
SceneTest's `_dump_name_map`).

Browser controls in the HTML viewer:

- **drag** → pan
- **wheel** → zoom about cursor
- **`f` key** → fit to view
- **`r` key** → reset to 1:1

The HTML scales to graphs the browser's SVG renderer can handle — in
practice, ~50k nodes with `--engine sfdp`. Past that, you want a
canvas/WebGL viewer (Cytoscape.js, sigma.js), which is out of scope
for this tool.

---

## 6. Relationship to `fanout[]` + Validation Gate

When checking fanout coverage, project annotated edges down to a
`{(pred, succ)}` set first — the per-edge annotation distinguishes
sources / args / slices, so the raw `edges[]` count is a superset of the
underlying task-pair count.

`deps.json` (projected) is a **superset** of the fanout edges in
`l2_swimlane_records.json`:

| Edge source | Captures | Drops on race? |
| ----------- | -------- | -------------- |
| `task.fanout[]` (L2SwimlaneAicpuTaskRecord) | Successors known at producer-retire time | **Yes** — sealed when producer retires |
| `deps.json` (this feature) | Every consumer → producer reachable via tensormap / explicit_deps | No — replay sees every submit |

`tests/st/{a2a3,a5}/tensormap_and_ringbuffer/dfx/dep_gen/test_dep_gen.py`
enforces the 6-edge expectation against the `vector_example`
orchestration as a validation gate: any edge missing from `deps.json`
is a replay-side regression and fails the test. Cases where
`deps - fanout ≠ ∅` are the dep_gen sweet spot — those are
exactly the race-window edges fanout dropped. The
`swimlane_converter.py` uses `deps.json` (when present) as the source
of flow events in the Perfetto trace, and flags any edge whose
`pred.end_time > succ.start_time` as `hb_violation` (rendered as a
distinct flow event name so Perfetto colors it apart from regular
dependencies).

---

## 7. Big-Fanin Submits: Overflow Chain

A `DepGenRecord` carries up to **64** explicit_deps inline. Submits with
more than 64 explicit deps spill into a chain of `DepGenOverflowRecord`
slots that overlay normal record slots in the buffer.

| Submit `explicit_dep_count` | Chain shape | Per-submit slots used |
| --------------------------- | ----------- | --------------------- |
| `0 ≤ dc ≤ 64` | base only — fast path, no chain bookkeeping | 1 |
| `65 ≤ dc ≤ 390` | base (64 deps) + 1 overflow (up to 326 deps) | 2 |
| `391 ≤ dc ≤ 716` | base + 2 overflow | 3 |
| general | base + `⌈(dc − 64) / 326⌉` overflow | `1 + ⌈(dc − 64) / 326⌉` |

**Wire format.** An overflow record reinterprets the same 2624-byte slot
as `{ task_id (8) + flags (4) + dep_count (2) + _reserved (2) + deps[326] }` —
no tensor blobs (those live on the base record). Replay distinguishes
the two views by `flags & DEP_GEN_FLAG_OVERFLOW`. Chain records always
share the base's `task_id`, and the last one sets
`DEP_GEN_FLAG_LAST_OVERFLOW` so replay knows the chain is complete
without peeking ahead.

**Atomicity.** The AICPU writer reserves *all* chain slots up front: if
the current buffer can't hold the full chain, it switches buffer first.
`buf->count` is published with a single store at the end, so the host
either sees the old count (chain invisible) or the new count with the
full chain committed. Chain records are therefore always contiguous
within one buffer — replay scans linearly and ties them back to the
base via `task_id`.

**Truncation tail.** A submit whose chain exceeds the buffer's slot
budget (`PLATFORM_DEP_GEN_RECORDS_PER_BUFFER = 32` slots → roughly
`64 + 31 × 326 = 10170` deps max in the best case) is logged via
`LOG_ERROR` and truncated to the largest dc that fits. Runtime
correctness is unaffected — `Arg::set_dependencies` keeps the full dep
list; only the dep_gen replay graph loses the tail.

---

## 8. Architecture Touchpoints

| Layer | File | Role |
| ----- | ---- | ---- |
| Shared-mem layout | `src/{a2a3,a5}/platform/include/common/dep_gen.h` | `DepGenRecord` (2624 B base, cache-line aligned, ≤64 inline explicit_deps) + `DepGenOverflowRecord` chain view (≤326 deps per slot) + SPSC ring + per-thread ready queue. Byte-identical layout across platforms. |
| AICPU writer | `src/{a2a3,a5}/platform/{include,shared}/aicpu/dep_gen_collector_aicpu.{h,cpp}` | Single-instance write path; weak-fallback exported to host build. a5 reuses the a2a3 source verbatim — the writer accesses its own device-side view of shared memory, independent of how host↔device transport is implemented. |
| Host collector | `src/{a2a3,a5}/platform/{include/host,shared/host}/dep_gen_collector.{h,cpp}` | `ProfilerBase<DepGenCollector, DepGenModule>` — drains ring → `records_` vector. On a5 (no SVM) it uses the base `alloc_paired_buffer`, which malloc's a host shadow + `copy_to_device`'s it and registers it via `add_malloc_shadow` so teardown can free it; `reconcile_counters` explicitly `copy_from_device`'s the BufferState before reading, and `finalize` lets `BufferPoolManager::clear_mappings()` release all shadows as the single source of truth. |
| Capture call site | `src/{a2a3,a5}/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp` `submit_task_common` | One conditional block that snapshots inputs into the ring when `is_dep_gen_enabled()`; fires for both `submit_task` and `submit_dummy_task`. Dep-only tasks land in the record stream with valid tensor/dep info but no kernel_id field (the schema does not carry kernel_id), so replay treats them as ordinary dep nodes — viewers do not currently distinguish dummy from real tasks. |
| Replay | `src/{a2a3,a5}/runtime/tensormap_and_ringbuffer/host/dep_gen_replay.{h,cpp}` | Pure CPU; runs dual-pass differential replay — `compute_task_fanin` (oracle) + inlined STEP A/B mirror (annotated) against two `PTO2TensorMap` instances. Emits `deps.json` when both passes agree per record. Platform-agnostic — a5 reuses the a2a3 source verbatim. |
| Device-runner hookup | `src/{a2a3,a5}/platform/{onboard,sim}/host/device_runner.cpp` | post-`reconcile_counters` calls `dep_gen_replay_emit_deps_json(records.data(), records.size(), deps_path)` |
| Viewer | `simpler_setup/tools/deps_to_graph.py` | `deps.json` → pan/zoom HTML |
| Test | `tests/st/{a2a3,a5}/tensormap_and_ringbuffer/dfx/dep_gen/test_dep_gen.py` + `test_dep_gen_chain.py` | Smoke test + 6-edge validation against `vector_example` orchestration (both platforms share byte-identical orchestration code). |

Supported on both a2a3 and a5. The a5 host collector differs from a2a3 only in its host↔device transport path (a5 has no SVM, so all transfers go through `profiling_copy_to_device` / `profiling_copy_from_device` instead of relying on `halHostRegister`'s shared mapping); the AICPU writer, shared-memory ABI, runtime call site, and replay are platform-agnostic.
