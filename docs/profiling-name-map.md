# Profiling Name Map

## Problem

Profiling data (`l2_perf_records.json`) identifies tasks by numeric IDs
(e.g., `func_id: 0`).  Without a mapping, swimlane visualizations show
opaque labels like `func_0_a(t0)` instead of human-readable names like
`QK(t0)`.

## Design Principle: Each Level Owns Its Own Mapping

The runtime is organized in hierarchical levels (L2, L3, ...).  Each
level independently exports profiling data and a name mapping that
describes **its own tasks and the next level down**:

- **Perf data** declares which level it comes from.
- **Name map** declares which level it describes.
- The visualization tool matches them by level.  No cross-level
  merging or recursive expansion.

This keeps each level self-contained.  L3 treats L2 as an opaque
callable; L2 treats individual cores as opaque executors.

## Name Map Format

Every level uses the same structure:

```json
{
  "level": <int>,
  "orchestrator_name": "<display name or null>",
  "callable_id_to_name": {
    "<id>": "<name>",
    ...
  }
}
```

| Field | Meaning |
| ----- | ------- |
| `level` | Which level this mapping describes |
| `orchestrator_name` | Display name for the orchestrator at this level (optional, from `"name"` in CALLABLE) |
| `callable_id_to_name` | Maps next-level-down callable IDs to human-readable names |

### L2 (Orchestration + Incores)

`callable_id` = incore `func_id` (the integer assigned in the CALLABLE
spec).  These are the same IDs that appear in L2 perf data.

```json
{
  "level": 2,
  "orchestrator_name": "PagedAttn",
  "callable_id_to_name": {
    "0": "QK",
    "1": "SF",
    "2": "PV",
    "3": "UP"
  }
}
```

### L3 (Callables List)

`callable_id` = index in the `callables` list (0-based).  Both
ChipCallable entries and SubWorker entries share this index space.

```json
{
  "level": 3,
  "orchestrator_name": "run_dag",
  "callable_id_to_name": {
    "0": "vector_kernel",
    "1": "verify"
  }
}
```

## How to Define Names in SceneTest

Add optional `"name"` fields to the CALLABLE spec.  Entries without
`"name"` are omitted from the mapping and display with default labels.

### L2 Example

```python
@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestPagedAttention(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/orch.cpp",
            "function_name": "build_paged_attention_graph",
            "name": "PagedAttn",                          # optional
            "signature": [D.IN, D.IN, D.IN, D.OUT],
        },
        "incores": [
            {"func_id": 0, "name": "QK",  "source": "kernels/aic/qk.cpp",  "core_type": "aic"},
            {"func_id": 1, "name": "SF",  "source": "kernels/aiv/sf.cpp",  "core_type": "aiv"},
            {"func_id": 2, "name": "PV",  "source": "kernels/aic/pv.cpp",  "core_type": "aic"},
            {"func_id": 3, "name": "UP",  "source": "kernels/aiv/up.cpp",  "core_type": "aiv"},
        ],
    }
```

### L3 Example

At L3, callable entries already have a `"name"` field (used by
`CallableNamespace`).  These names are automatically included in the
mapping:

```python
@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestL3Group(SceneTestCase):
    CALLABLE = {
        "orchestration": run_dag,
        "callables": [
            {
                "name": "vector_kernel",                  # already required at L3
                "orchestration": {"source": "...", "function_name": "build_vec"},
                "incores": [{"func_id": 0, "source": "...", "core_type": "aiv"}],
            },
            {
                "name": "verify",                         # already required at L3
                "callable": verify_fn,
            },
        ],
    }
```

## Automatic Dump and Consumption

When `--enable-l2-swimlane` is used, SceneTest automatically:

1. Extracts the name mapping from `CALLABLE` via `_extract_name_map()`.
2. Writes `<output_prefix>/name_map_<ClassName_casename>.json`.
3. Passes the file to `swimlane_converter.py` via `--func-names`.

No manual steps are needed.  If no `"name"` fields are defined, no
mapping file is written and the tools fall back to default labels.

## Tool Usage

The `--func-names` flag is available on both visualization tools.  It
takes precedence over `-k` (kernel_config.py):

```bash
# Automatic (via SceneTest profiling)
pytest tests/st/... --platform a5onboard --enable-l2-swimlane

# Manual (paths land alongside l2_perf_records.json inside the same
# <output_prefix> directory)
python -m simpler_setup.tools.swimlane_converter \
    outputs/<case>_<ts>/l2_perf_records.json \
    --func-names outputs/<case>_<ts>/name_map_TestPA_basic.json

python -m simpler_setup.tools.deps_to_graph \
    outputs/<case>_<ts>/deps.json \
    --func-names outputs/<case>_<ts>/name_map_TestPA_basic.json
```

## File Layout

Each test case writes its diagnostic artifacts under
`CallConfig::output_prefix` (chosen by
`scene_test.py::_build_output_prefix` as
`outputs/<ClassName>_<case>_<YYYYMMDD_HHMMSS>/`). Filenames are fixed —
the per-case directory is the uniqueness boundary, so parallel runs
cannot collide.

```text
outputs/TestPA_basic_20260416_151301/
  l2_perf_records.json         # perf data (runtime)
  name_map_TestPA_basic.json   # name mapping (SceneTest)
  merged_swimlane.json         # Perfetto trace (converter)
```
