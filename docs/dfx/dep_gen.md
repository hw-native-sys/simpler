# dep_gen — Complete Per-Submit Dependency Graph

## 1. Background & Motivation

The swimlane profiler's per-task `fanout[]` array is the obvious place to
read "which tasks did task X feed into?" — but it is **structurally
incomplete on real hardware**.

Each producer task carries its own `L2PerfRecord.fanout[RUNTIME_MAX_FANOUT]`,
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
- **Host replay.** After `reconcile_counters()` confirms a clean trace
  (no drops, no leftovers), `dep_gen_replay_emit_deps_json` runs every
  record back through a host-resident `PTO2TensorMap`. Per-record
  semantics mirror runtime `submit_task` exactly: STEP 1 (explicit
  deps), STEP 3 (creator retention + tensormap lookup), STEP 4
  (register outputs). Per-successor dedup matches
  `PTO2FaninBuilder::append_fanin_or_fail`.
- **Output.** `<output_prefix>/deps.json` —
  `{"version":1,"edges":[[pred_raw,succ_raw], ...]}`.

---

## 3. How to Enable

`dep_gen` is gated by `CallConfig.enable_dep_gen` (alongside
`enable_l2_swimlane`, `enable_dump_tensor`, `enable_pmu`). The CLI flag
is `--enable-dep-gen`:

```bash
# Standalone
python test_my_case.py --platform a2a3 --enable-dep-gen --enable-l2-swimlane

# Pytest
pytest tests/st/... --platform a2a3 --enable-dep-gen --enable-l2-swimlane
```

The `--enable-l2-swimlane` flag is independent but recommended in pair
because:

- `deps.json` is the dep_gen artifact.
- `l2_perf_records.json` (from swimlane) is the timing artifact;
  `merged_swimlane.json` (the Perfetto trace) uses `deps.json` for
  dependency arrows when both files exist.
- The "fanout ⊆ deps" validation gate fires only when both files are
  present.

When `--enable-dep-gen` is on with any other diagnostic flag, an
`output_prefix` directory must be set (the runtime throws otherwise).
The standard SceneTest path
(`outputs/<TestName>_<case>_<timestamp>/`) handles that automatically.

---

## 4. Output: `deps.json`

```json
{
  "version": 1,
  "edges": [
    [0,           4294967296],
    [0,           4294967297],
    [4294967296,  4294967298],
    [4294967297,  4294967298],
    [4294967298,  4294967299],
    [0,           4294967299]
  ]
}
```

Each edge is `[pred_raw, succ_raw]` where the raw uint64 encodes
`(ring_id << 32) | local_id` — the same layout as `PTO2TaskId::raw`. To
decode:

```python
ring = (raw >> 32) & 0xFF
local = raw & 0xFFFFFFFF
```

Edges are de-duplicated within a single successor's fanin (matches the
runtime's `append_fanin_or_fail` contract) but **not** globally —
distinct successors with the same predecessor each get their own edge.

`deps.json` can be empty (`"edges":[]`) when the workload's tasks have
no inter-task data dependencies (e.g. embarrassingly parallel kernels
under scope_end barriers). That is not an error.

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
```

Node visual encoding (legend top-right of the rendered HTML):

| Shape + color | Meaning |
| ------------- | ------- |
| Blue rounded box | AIC (cube) — kernel ran on the matrix unit |
| Orange ellipse | AIV (vector) — kernel ran on the vector unit |
| Green diamond | mix — single `submit_task` with `MixedKernels` spanning both core types |
| Gray dashed note | alloc — task from `alloc_tensors` (got a task_id, references downstream via `owner_task_id`, but never dispatched a kernel so has no perf record) |

Labels read as `(ring, local) · func_name · core_type-implicit-via-shape`.
When a colocated `l2_perf_records.json` is present the func_id is enriched
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

`deps.json` is a **superset** of the fanout edges in
`l2_perf_records.json`:

| Edge source | Captures | Drops on race? |
| ----------- | -------- | -------------- |
| `task.fanout[]` (L2PerfRecord) | Successors known at producer-retire time | **Yes** — sealed when producer retires |
| `deps.json` (this feature) | Every consumer → producer reachable via tensormap / explicit_deps | No — replay sees every submit |

`tests/st/a2a3/tensormap_and_ringbuffer/dep_gen_capture/test_dep_gen_capture.py`
enforces `fanout ⊆ deps` as a validation gate: any edge in fanout that
is missing from deps is a replay-side regression and fails the test.
Cases where `deps - fanout ≠ ∅` are the dep_gen sweet spot — those are
exactly the race-window edges fanout dropped. The
`swimlane_converter.py` uses `deps.json` (when present) as the source
of flow events in the Perfetto trace, and flags any edge whose
`pred.end_time > succ.start_time` as `hb_violation` (rendered as a
distinct flow event name so Perfetto colors it apart from regular
dependencies).

---

## 7. Architecture Touchpoints

| Layer | File | Role |
| ----- | ---- | ---- |
| Shared-mem layout | `src/a2a3/platform/include/common/dep_gen.h` | `DepGenRecord` (2240 B, cache-line aligned) + SPSC ring + per-thread ready queue |
| AICPU writer | `src/a2a3/platform/{include,src}/aicpu/dep_gen_collector_aicpu.{h,cpp}` | Single-instance write path; weak-fallback exported to host build |
| Host collector | `src/a2a3/platform/{include/host,src/host}/dep_gen_collector.{h,cpp}` | `ProfilerBase<DepGenCollector, DepGenModule>` — drains ring → `records_` vector |
| Capture call site | `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp` `submit_task` | One conditional block that snapshots inputs into the ring when `is_dep_gen_enabled()` |
| Replay | `src/a2a3/runtime/tensormap_and_ringbuffer/host/dep_gen_replay.{h,cpp}` | Pure CPU; runs `compute_task_fanin` + `register_task_outputs` against a host `PTO2TensorMap` |
| Device-runner hookup | `src/a2a3/platform/{onboard,sim}/host/device_runner.cpp` | post-`reconcile_counters` calls `dep_gen_replay_emit_deps_json(records.data(), records.size(), deps_path, nullptr)` |
| Viewer | `simpler_setup/tools/deps_to_graph.py` | `deps.json` → pan/zoom HTML |
| Test | `tests/st/a2a3/tensormap_and_ringbuffer/dep_gen_capture/test_dep_gen_capture.py` | Smoke test + `fanout ⊆ deps` validation gate |

Currently a2a3 only; an a5 port is planned.
