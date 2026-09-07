# Args Dump — Per-task Argument Capture

## 1. Background & Motivation

Numerical bugs (NaNs, wrong shapes, off-by-one offsets, mis-aligned
strides) are notoriously hard to reason about by reading kernel code:
the symptom shows up two tasks downstream, the suspect tensor is
gone, and re-running with `printf` distorts the timing that triggered
the bug in the first place.

Args Dump captures per-task tensor inputs/outputs plus scalar inputs during kernel
execution and writes them to disk for offline inspection. The host
pre-allocates the recording buffers, AICPU writes records during
execution, and the host exports a JSON manifest plus any level-selected binary
payload.
The result is a stable, replayable record of every dumped argument a kernel
saw, without the timing distortion of inline printing.

## 2. Overview

- **Per-task input/output capture.** Inputs snapshotted before
  dispatch, outputs snapshotted after FIN; `INOUT` tensors at both
  stages.
- **Logical shape preserved.** Records carry dtype, shape,
  `strides`, `start_offset`, and `is_contiguous` so logical views are
  reconstructable.
- **Manifest + level-dependent binary payload.** Once at least one record is
  collected, every enabled level writes `args_dump.json`. Levels 1 and 2 then
  create `args.bin` eagerly; hybrid Level 3 creates it lazily only when an
  `Arg::dump(...)`-selected tensor contributes payload. Tensor payload records
  carry `bin_offset` / `bin_size`, while scalar values remain manifest-only.
- **Unified scalar args.** Scalar values are emitted as
  `kind: scalar`, `stage: before_dispatch`, zero-dim records in
  `args_dump.json`; there is no separate args-only manifest. Their
  final `dtype` is registered at submit time in a dump-only per-task side table
  and resolved by AICPU when writing each scalar dump record.
- **Cross-architecture metadata.** The flags, manifest schema, and scalar
  metadata are wired through both runtime variants on `a2a3` and `a5`.
  Real tensor-payload consumption is currently supported only on the
  `a2a3` platform family; see the platform scope below.

Enable in one line (`2` = full dump, every task):

```bash
python tests/st/<case>/test_<name>.py -p a2a3sim --dump-args 2
```

## 3. How to Use

### 3.1 Enable Args Dump

`--dump-args` takes an optional **level**:

| Level | Meaning |
| ----- | ------- |
| `0` (or flag absent) | off — zero overhead |
| `1` (bare `--dump-args`) | **partial** — only args marked with `CoreTaskArgs::dump(...)` (see §3.2) |
| `2` (`--dump-args 2`) | **full** — every task's tensor inputs/outputs and scalar args |
| `3` (`--dump-args 3`) | **hybrid** — every task's tensor/scalar metadata goes to `args_dump.json`; tensors marked with `CoreTaskArgs::dump(...)` also contribute payload to `args.bin` |

> **Tensor-payload platform scope:** levels 1/2 and marked level 3 are
> currently supported for payload consumers only on `a2a3` / `a2a3sim`.
> On `a5` / `a5sim`, payload truth remains outside the supported consumer
> contract while [#1560](https://github.com/hw-native-sys/simpler/issues/1560)
> is open. Do not use an a5 dump as a source of tensor truth. A marked
> level-3 payload still produces `args.bin` on that platform family, but it is
> not trustworthy for a Core swimlane replay that restores it.
> Plain level-3 metadata, including inline scalar values, does not consume
> tensor payload and remains available.

The hybrid Level 3 is the capture mode used by
`python -m simpler_setup.tools.core_swimlane` to build a Core swimlane simulator replay:
it provides complete per-task argument metadata without dumping every tensor's
element data. It reuses the exact task/argument mask produced by
`CoreTaskArgs::dump(...)` for Level 1: every argument still gets a JSON record,
while only marked tensors contribute bytes to `args.bin`.
Scalar values already live inline in the manifest and never need payload copy.
With no `dump(...)` markers, the AICPU skips all arena payload copies, the
manifest's `bin_file` is `null`, and every `bin_size` is `0`.

Level 3 deliberately inherits Level 1's selector semantics; it does not add a
second mask. `tensormap_and_ringbuffer` registers selection metadata in its
AICPU per-task table. `host_build_graph` embeds the same mask, ambiguity flags,
and scalar dtypes in each H2D task image, including cached in-graph task
definitions. The device collector consumes either source identically. On `a5`,
however, the resulting tensor bytes remain untrusted under #1560, so payload
restoration stays outside this change until that issue is fixed.

```bash
# Standalone runner
python tests/st/<case>/test_<name>.py -p a2a3sim --dump-args 2  # full
python tests/st/<case>/test_<name>.py -p a2a3 -d 0 --dump-args   # partial (level 1)

# pytest
pytest tests/st/<case> --platform a5sim --dump-args 2
# a5 host_build_graph has no examples — use the scene test
pytest tests/st/a5/host_build_graph/dump_args --platform a5sim --dump-args 2
pytest tests/st/<case> --platform a2a3sim --dump-args 2
pytest tests/st/a2a3/host_build_graph/vector_example --platform a2a3sim --manual include --dump-args 2
```

The level sets `CallConfig::enable_dump_args` (0/1/2/3). The host then
allocates dump storage, publishes its base address through
`kernel_args.dump_data_base`, and sets `SIMPLER_DFX_FLAG_DUMP_ARGS`
(levels 1, 2, and 3) in each worker handshake's `enable_profiling_flag` for
the enable/disable decision. The **partial / full / hybrid**
distinction is carried as a `DumpArgsLevel` in the dump shared-memory header
(`DumpDataHeader::dump_args_level`, host-written before launch) rather
than in the profiling-flag bitmask. The on-device AICPU reads the storage base
via `set_platform_dump_base()`, the enable bit via
`set_dump_args_enabled(SIMPLER_GET_DFX_FLAG(...))`, and latches the mode
from the header level before task dispatch (`PARTIAL` → selective,
`HYBRID` → all metadata, payload copy controlled by the
same per-task mask as `PARTIAL`). The runtime resolves the mask before each
record reaches the platform writer, so the writer does not depend on a second
task-id lookup. Because the
level is decided host-side **before any task is
dispatched**, it is latched up front — there is no dependence on task
submission order. AICore executors read the same `SIMPLER_DFX_FLAG_DUMP_ARGS`
bit to insert a `pipe_barrier(PIPE_ALL)` before FIN when dump is on, so
`AFTER_COMPLETION` snapshots see the kernel's final writes.

### 3.2 Partial Dump — Select Specific Args

Partial dump (level 1) captures only the tasks whose `CoreTaskArgs` is marked
with `dump(...)`; every unmarked task is skipped. Within a marked task,
only the selected tensor/scalar args are recorded. Mark the arguments on
the relevant `CoreTaskArgs` before submission:

```cpp
CoreTaskArgs args;
args.add_input(x);
args.add_input(y);
args.add_output(z);
float scale = 1.0f;
args.add_scalar(scale);
args.dump(x, z, scale);
rt_submit_aiv_task(FUNC_ADD, args);
```

`dump(...)` selects arguments from the current `CoreTaskArgs`; it does not execute
a dump immediately. The selected tensors and scalar lvalues must already
belong to that `CoreTaskArgs`. Scalar selection is by the lvalue passed to
`add_scalar(...)`, not by scalar value. Temporaries such as
`args.dump(1.0f)` are rejected because they do not identify a previously
added scalar slot. If the same scalar lvalue is added more than once,
`dump(lvalue)` selects the first matching scalar arg and marks that scalar
entry with `arg_index_ambiguous: true` in `args_dump.json`; use distinct
local variables when you need to select a later duplicate.

The runtime uses the tensor direction already provided by `add_input()`,
`add_output()`, or `add_inout()` to decide when each selected tensor is
captured:

- input tensors are dumped before dispatch.
- output tensors are dumped after completion.
- inout tensors follow the existing inout dump behavior.
- scalar args are dumped before dispatch.

Selective dump comes at two granularities, both expressed with the same
`dump(...)` marker:

- **Arg granularity** — `dump(x, z, scale)` selects specific tensor and
  scalar arguments of the task (the example above).
- **Task granularity** — `dump()` with no arguments selects the whole
  task (every tensor and scalar argument on the `CoreTaskArgs`), without
  enumerating them:

  ```cpp
  CoreTaskArgs args;
  args.add_input(x);
  args.add_input(y);
  args.add_output(z);
  args.add_scalar(scale);
  args.dump();        // whole task: every tensor/scalar arg on this CoreTaskArgs
  rt_submit_aiv_task(FUNC_ADD, args);
  ```

Partial vs full is chosen by the **dump level** (§3.1), latched host-side
before any dispatch — not inferred from the markers, so it never depends
on task submission order. At level 1, tasks without a marker are skipped
and marked tasks dump only their selected arguments. At level 2, the
markers are ignored and every task's tensors/scalars are dumped. At level
3, every task's tensors/scalars are dumped as metadata and the same markers
control which tensor records carry payload. A5 tensor payload remains
unsupported under [#1560](https://github.com/hw-native-sys/simpler/issues/1560).
With no `--dump-args` (level 0) dump is off entirely.

If you run at level 1 but place no `dump(...)` markers anywhere, the
collector receives no records and exports no manifest — that is the deliberate
"only what I marked" contract. Use `--dump-args 2` when you want everything.

### 3.3 Output

The dump artifacts land under the per-task output prefix
(`CallConfig::output_prefix`, set by
`scene_test.py::_build_output_prefix` to
`outputs/<ClassName>_<case>_<YYYYMMDD_HHMMSS>/` for SceneTest runs):

```text
<output_prefix>/
└── args_dump/
    ├── args_dump.json  # unified argument manifest (`--dump-args`)
    └── [args.bin]      # eager at levels 1/2; present at level 3 only with selected tensor payload
```

Filenames are fixed (no per-file timestamp) — the directory is the
per-task uniqueness boundary. Once at least one record reaches the collector,
levels 1/2 create `args.bin`, which may be empty when no tensor payload was
selected or recorded. Hybrid Level 3 with records but no selected tensor
payload emits only `args_dump.json`; otherwise it also emits `args.bin`.
On a5, a present and correctly sized `args.bin` does not establish payload
truth while [#1560](https://github.com/hw-native-sys/simpler/issues/1560) is
open.

#### `args_dump.json` — Unified manifest

`args_dump.json` is the manifest; its `bin_file` field points at
the sibling binary payload. A plain level-3 capture has `bin_file: null`, no
`.bin`, and `bin_size: 0` for every entry. A level-3 capture with marked tensors
has `bin_file: "args.bin"`; only the `CoreTaskArgs::dump(...)`-marked tensor
records have a non-zero `bin_size`.

Example manifest (one input tensor captured before dispatch):

```json
{
  "run_dir": "args_dump",
  "bin_format": {
    "type": "logical_contiguous",
    "byte_order": "little_endian"
  },
  "dump_args_level": 2,
  "total_args": 1,
  "before_dispatch": 1,
  "after_completion": 0,
  "input_args": 1,
  "output_args": 0,
  "inout_args": 0,
  "truncated_args": 0,
  "dropped_records": 0,
  "bin_file": "args.bin",
  "args": [
    {
      "task_id": "0x0000000200000a00",
      "func_id": [0],
      "role": "input",
      "stage": "before_dispatch",
      "arg_index": 0,
      "kind": "tensor",
      "dtype": "float32",
      "shape": [16384],
      "strides": [1],
      "start_offset": 0,
      "is_contiguous": true,
      "truncated": false,
      "bin_offset": 0,
      "bin_size": 65536
    }
  ]
}
```

Key fields:

- `dump_args_level` — the capture mode (`0` / `1` / `2` / `3`).
- `task_id` — runtime task identity. Use to correlate with swimlane / PMU
  output.
- `func_id` — **array** of the task's active-subtask kernel ids (its mix
  membership). A single-kernel task is `[N]`; a cooperative or packed mix is
  `[i, j, ...]` (one entry per active subtask). The dump emits each payload
  tensor **once** with this array, so a mix is recoverable by grouping on
  `task_id` and reading the func set. `[-1]` when unknown (the
  `host_build_graph` dump path, which does not thread func_id). Note: the
  array is the task's mix set, **not** a claim that each listed kernel reads
  this specific slot.
- `arg_index` — payload argument index (the slot this tensor occupies).
  Tensor entries use the payload tensor index; scalar entries use
  `payload.tensor_count + scalar_index`. When scalar lvalue selection matched
  multiple scalar slots, the selected first-match scalar entry carries
  `arg_index_ambiguous: true`.
- `role` / `stage` — `input` / `output` / `inout`, captured
  `before_dispatch` / `after_completion`.
- `kind` — `tensor` for arena-backed payloads, `scalar` for
  zero-dim `before_dispatch` scalar args stored directly in the
  manifest.
- `dtype` / `shape` / `strides` / `start_offset` /
  `is_contiguous` — logical view geometry. Tensor `bin_size` is
  `numel × elem_size` of the *logical* view, gathered if
  non-contiguous; scalar entries have `bin_size = 0`.
- `bin_offset` — byte offset into `args.bin` where the
  payload starts.
- `truncated` — set when the tensor exceeded arena size; see §7.
- Top-level `dropped_records` surfaces aggregate metadata-buffer loss — useful
  for spot-checking a run.

### 3.3 Inspect with `dump_viewer`

The viewer auto-picks the latest `outputs/*/args_dump` directory
when invoked without arguments. It loads `args_dump.json` and
uses its `bin_file` field to find the payload:

```bash
# List every dumped arg in the latest run
python -m simpler_setup.tools.dump_viewer

# Filter and save matching args to human-readable .txt files
python -m simpler_setup.tools.dump_viewer --stage before --role input --export

# Export one specific entry by its manifest index
python -m simpler_setup.tools.dump_viewer --index 42

# Pin to a specific dump directory
python -m simpler_setup.tools.dump_viewer outputs/<case>_<ts>/args_dump \
    --task 0x0000000200000a00 --export
```

Exported `.txt` files include metadata headers, a row-major overview
with aligned columns, and a detail listing with multi-dim indices —
diff-friendly against golden tensors and pasteable into a
spreadsheet.

### 3.4 Add dump support to a new test

For `tensormap_and_ringbuffer`, each incore declares a `signature` (each
tensor's `ArgDirection`) in its `CALLABLE` entry. The dump maps signature
entry `i` to payload tensor slot `i` **positionally** — there is no
`arg_index`. The signature gives every captured tensor a direction (which
sets its capture stage); geometry (shape, dtype, strides) is derived
automatically from the payload.

The signature must cover the task's full payload: for a **mix task**
(multiple subtasks sharing one `args[]`) every cooperating incore declares
the **complete** task signature, in payload order. A narrower task that
dispatches only a prefix of that layout (e.g. a standalone kernel, or a
2-subtask mix of a 3-subtask shape) supplies a prefix — the dump records
the slots present and skips the rest. The dump is driven by the **widest**
active subtask's signature, so at least one cooperating incore must declare
the full-width layout.

A `signature` usually lists only tensor directions — scalars are
independent (added via `add_scalar`) and are typically omitted. If a
`SCALAR` direction *is* listed it must come after every tensor entry; the
dump skips `SCALAR` entries (they do not consume a positional tensor slot).

Both `host_build_graph` and `tensormap_and_ringbuffer` share the
same code path: the AICPU dump-collector reads geometry directly from
the payload's tensor region, which carries shape and offset info
alongside the device-packed arguments. For allocated output tensors,
`alloc_tensors` supplies a `TensorCreateInfo` at orchestration time
that is embedded in the payload by submit. For tensors that already
exist (inputs passed by the caller), the geometry travels directly on
the Tensor object and needs no explicit per-task registration.

Full template:
[`tests/st/a5/host_build_graph/dump_args`](../../tests/st/a5/host_build_graph/dump_args/)
(and the `a2a3` mirror at
`tests/st/a2a3/host_build_graph/dump_args`).

## 4. Capabilities

What you can read out of `args_dump.json` and, when present, `args.bin`:

- **Per-task input snapshots** (`role: input`, `stage:
  before_dispatch`) — what each kernel was given.
- **Per-dispatch scalar args** (`kind: scalar`, `stage:
  before_dispatch`) — the raw per-task scalar-slot values AICPU
  handed to the kernel, tagged with the dump-only dtype captured at
  submit time.
- **Per-task output snapshots** (`role: output`, `stage:
  after_completion`) — what each kernel produced. The barrier
  ensures these reflect the kernel's final writes.
- **`INOUT` deltas** — same arg captured at both stages; diff
  before vs after to see exactly what the kernel modified.
- **Logical view reconstruction** — `shape` / `strides` /
  `start_offset` / `is_contiguous` plus the gathered
  logical-contiguous payload.
- **Per-task identity** — `task_id`, `stage`, `role`, and `arg_index`
  identify each dumped argument within a task.
- **Loss accounting** — a per-record `truncated` flag plus aggregate
  `dropped_records` in the summary.

## 5. Design Highlights

`CoreTaskArgs::dump(...)` selection state is compiled only when
`SIMPLER_DFX=1`. With `SIMPLER_DFX=0`, the public API remains
available but acts as a no-op: no dump-only `CoreTaskArgs` state is stored and
submit does not propagate dump metadata.

### 5.1 Common device-side structures

Both architectures share the same device-side layout, published via
`kernel_args.dump_data_base`:

```text
DumpDataHeader                                  (host init, AICPU reads)
├── queues  [MAX_AICPU_THREADS][READYQUEUE_SIZE]
├── queue_heads / queue_tails (per-thread)
├── num_dump_threads
├── records_per_buffer
├── magic = 0x44554D50 ("DUMP")
└── dump_args_level  (DumpArgsLevel: 0=off, 1=partial, 2=full, 3=hybrid; AICPU latches before dispatch)

DumpBufferState[num_dump_threads]               (per-thread)
├── free_queue {buffer_ptrs[SLOT_COUNT], head, tail}
├── current_buf_ptr            (AICPU active DumpMetaBuffer*)
├── current_buf_seq
├── arena_base / arena_size    (per-thread arena pointers)
├── arena_write_offset         (AICPU cursor in the current arena reuse cycle)
├── published_payload_count    (payloads committed to Host through RQ entries)
├── completed_payload_count    (published watermark acknowledged by Host)
└── dropped_record_count

DumpMetaBuffer pool (rotated)                   (BUFFERS_PER_THREAD per thread)
└── ArgsDumpRecord records[RECORDS_PER_BUFFER] + count   ← 128 B each

arena_data (per-thread, circular byte buffer)
  default = BUFFERS_PER_THREAD × RECORDS_PER_BUFFER × AVG_TENSOR_BYTES
          = 8 × 256 × 64 KiB = 128 MiB per thread
```

These structs are binary-identical between a2a3 and a5
(`static_assert`-checked). `dump_data_base` flows through
`KernelArgs`, not `Runtime` — AICPU reads it from
`k_args->dump_data_base` in `kernel.cpp` and passes it to
`set_platform_dump_base()`. Dump enablement is propagated
separately via the umbrella bitmask `KernelArgs::enable_profiling_flag`
(`bit0 = SIMPLER_DFX_FLAG_DUMP_ARGS`); the AICPU kernel entry calls
`set_dump_args_enabled()` with the decoded bit, so device-side
code does not infer "dump enabled" from `dump_data_base != 0`.

Each record is fixed at 128 B (two cache lines) — see
`ArgsDumpRecord` in
[`args_dump.h`](../../src/common/platform/include/common/args_dump.h).

### 5.2 Where dump calls are wired in

Each runtime's scheduler dispatch code calls
`dump_args_for_task` at two points in the per-task state machine
(for `tensormap_and_ringbuffer`, this is in
`runtime/scheduler/scheduler_completion.cpp` and
`runtime/scheduler/scheduler_dispatch.cpp`):

```text
┌──────────────────────────────────────┐
│ per-task dispatch:                   │
│   if enable_dump_args {              │
│     dump_args_for_task(              │
│         BEFORE_DISPATCH);            │
│   }                                  │
│   dispatch(task);                    │
│   wait FIN;                          │
│   if enable_dump_args {              │
│     dump_args_for_task(              │
│         AFTER_COMPLETION);           │
│   }                                  │
│   retire(task);                      │
└──────────────────────────────────────┘
```

`dump_args_for_task` first collects the task's active-subtask kernel ids
(the `func_id` array), then walks the **widest active subtask's** callable
signature **once**. Signature entry `i` maps to payload tensor slot `i`
positionally; for each non-scalar entry it builds a `ArgsDumpInfo`
(dtype + shape + strides + start offset + device address) for that slot
and calls `dump_arg_record` for entries whose role matches the current
stage — stamping every record with the **whole** func_id array. Each
payload tensor is thus emitted **once** (not duplicated per subtask); a
mix is recoverable as the func set carried on each slot. A signature entry
beyond the payload is skipped (memory safety — a prefix-dispatched task);
payload slots the signature does not reach are left undumped with a soft
completeness warning. (The `host_build_graph` dump path is a separate
overload that also maps positionally and stamps `func_id = [-1]`.) Scalar
values are dumped separately from the flat payload `scalars[]`, once per
task (with the same func_id array); their dtype table is registered at
submit time and emitted as a dump-only metadata record at
`BEFORE_DISPATCH`.

When dump is enabled, AICore executors also issue
`pipe_barrier(PIPE_ALL)` after kernel execution and before writing
the FIN handshake. This closes the ordering gap where
`AFTER_COMPLETION` snapshots could observe output buffers before
all device-side writes were globally visible. Older
implementations could capture stale output data; the current
implementation fixes this in the runtime, not in each individual
kernel. The barrier is gated on `SIMPLER_DFX_FLAG_DUMP_ARGS`, so
non-dump runs keep the original cheaper completion path.

### 5.3 Tensor metadata registration

AICPU has device addresses and sizes — the logical shape, dtype,
and view geometry come from the runtime. Both `host_build_graph`
and `tensormap_and_ringbuffer` share the same code path: the
AICPU dump-collector reads geometry directly from
the payload's tensor region, which carries shape and offset info
alongside the device-packed arguments. For allocated output tensors,
`alloc_tensors` supplies a `TensorCreateInfo` at orchestration time
that is embedded in the payload by submit. For tensors that already
exist (inputs passed by the caller), the geometry travels directly
on the Tensor object and needs no explicit per-task registration.

When metadata is missing or inconsistent, the task is skipped for
dump and a single `LOG_WARN` is emitted (guarded by
`try_log_dump_args_layout_mismatch` to avoid log flooding);
normal execution continues.

### 5.4 a2a3 — shared-memory streaming

`halHostRegister` maps device memory into host virtual address
space so the host can read device buffers directly.
`ArgsDumpCollector` runs split mgmt threads and collector shards on top of a
[`BufferPoolManager<DumpModule>`](../../src/common/platform/include/host/buffer_pool_manager.h):
drain/refill shards poll SPSC ready queues and refill free queues from
shard-local recycled lanes **while kernels are still executing**. Collector
shards drain the host hand-off queues into `on_buffer_collected`, then the
replenish thread routes done buffers to same-kind lanes below their recycled
watermarks. Modules that declare no watermark keep origin-shard routing. Any
remaining watermark deficit is batch-allocated without writing device free
queues.

Each collector appends metadata to its own vector, so collectors do not
serialize on the manifest accumulator. Captured payloads share one `args.bin`:
the narrow writer lock assigns the next binary offset and enqueues the payload
in the same order, while metadata insertion stays outside that lock. Export
joins the writer, folds the shard-local vectors together, and sorts the merged
metadata by `(task_id, stage, arg_index, role)` before writing JSON.

```text
        HOST                                         DEVICE
┌──────────────────────────┐               ┌──────────────────────────┐
│ ArgsDumpCollector      │               │ AICPU thread             │
│                          │               │                          │
│ initialize()             │  alloc +      │ dump_args_init()         │
│   rtMalloc + halRegister │──register────>│   read DumpDataHeader    │
│   build DumpDataHeader   │              │   cache per-thread ptrs  │
│                          │               │                          │
│ start()                  │               │ per-task run loop:       │
│   ┌────────────────────┐ │               │   BEFORE_DISPATCH        │
│   │ drain/refill shard │ │               │     dump_arg_record()    │
│   │ + replenish thread │ │ SPSC ready    │     → write to arena     │
│   │   poll ready queue │<┼──queues──────<│     → append record      │
│   │   refill freeQ     │─┼──free queue──>│     → push to ready_q    │
│   └────────────────────┘ │               │   dispatch kernel        │
│   ┌────────────────────┐ │               │   wait FIN               │
│   │ collector shard    │ │               │   AFTER_COMPLETION       │
│   │   reads arena via  │ │ shared mem    │     dump_arg_record()    │
│   │   host mapping     │<┼──mapping─────<│                          │
│   └────────────────────┘ │               │                          │
│                          │               │ dump_args_flush()        │
│ stop()                   │               │   log per-thread stats   │
│   join mgmt → collectors │               └──────────────────────────┘
│ reconcile_counters()     │
│   recover leftovers      │
│   + dropped accounting   │
│                          │
│ export_dump_files()      │
│   → <output_prefix>/     │
│     args_dump/           │
│       args_dump.json     │
│       args.bin           │
└──────────────────────────┘
```

**Lifecycle** (`device_runner.cpp`):

```text
init_args_dump()
  dump_collector_.initialize(..., output_prefix_)
  kernel_args_.args.dump_data_base = dump_collector_.get_dump_shm_device_ptr()
start()                          ← spawn split mgmt threads (drain/refill
                                   + replenish), then collector shards
launch AICPU / AICore
rtStreamSynchronize              ← wait for kernel completion
stop()                           ← join mgmt/replenish after final drain,
                                   then signal collector shards and join them
reconcile_counters()             ← recover leftover current buffers
                                   + dropped accounting
export_dump_files()
```

[`ArgsDumpCollector`](../../src/common/platform/include/host/args_dump_collector.h)
on a2a3 inherits from
[`profiling_common::ProfilerBase<ArgsDumpCollector, DumpModule>`](../../src/common/platform/include/host/profiler_base.h):
the base class owns split mgmt threads, collector shards, and the
`BufferPoolManager<DumpModule>` they share. `ArgsDumpCollector`
only supplies the dump-specific pieces — the `DumpModule` trait
that describes the shared-memory layout, `initialize` that
allocates and pre-fills free queues, an `on_buffer_collected`
callback that gathers payload bytes and appends metadata to a shard-local
record list, plus `reconcile_counters` / `export_dump_files` /
`finalize`. The mgmt/collector threading, buffer pooling, and `Module`
trait pattern are shared with PMU and ChipSwimlane — see
[profiling-framework.md](profiling-framework.md) for the
framework reference.

### 5.5 a5 — same framework, host-shadow transport

a5's `ArgsDumpCollector` derives from
`ProfilerBase<ArgsDumpCollector, DumpModule>` and shares the
split mgmt + collector shard structure with a2a3. The architectural
deviation from §5.4 is the **transport channel**: a5 has no
`halHostRegister`, so each device buffer is paired with a
host-shadow `malloc()` and the framework synchronizes selected fields via
`profiling_copy.h` (`rtMemcpy` onboard, `memcpy` in sim).
`MemoryOps` therefore carries five callbacks (`alloc` / `reg` /
`free_` / `copy_to_device` / `copy_from_device`). The mgmt threads
pull queue indices and ready entries on demand, then push back only
the fields host modified (advanced `queue_heads[q]`, refilled
`free_queue.tail` and `buffer_ptrs[slot]`) via
`BufferPoolManager::write_range_to_device`. Each popped
`DumpMetaBuffer` is pulled inside `ProfilerAlgorithms::process_entry`.
The per-thread arena lives outside the shm region, so
`on_buffer_collected` separately refreshes `arena_write_offset` and
copies the arena bytes. The freeze-release predicate refreshes each
thread's `published_payload_count` and compares it with that thread's writer
completion count. Once every thread's counts match, Host writes each published
watermark back as `completed_payload_count`; AICPU requires this acknowledgement
before reusing that thread's arena.

```text
        HOST                                         DEVICE
┌──────────────────────────┐               ┌──────────────────────────┐
│ ArgsDumpCollector      │               │ AICPU thread             │
│   : ProfilerBase<...>    │               │                          │
│                          │               │                          │
│ initialize()             │  alloc + reg  │ dump_args_init()         │
│   rtMalloc shm           │──+ shadow────>│   read DumpDataHeader    │
│   per-thread arenas      │   memset 0    │   cache per-thread ptrs  │
│   per-thread             │   + push 0s   │                          │
│   DumpMetaBuffers        │               │ per-task run loop:       │
│   register_mapping(s)    │               │   BEFORE_DISPATCH        │
│                          │               │     dump_arg_record()    │
│ start(thread_factory)    │               │   dispatch kernel        │
│   split mgmt starts      │               │   wait FIN               │
│   collector shards start │               │   AFTER_COMPLETION       │
│                          │               │     dump_arg_record()    │
│ mgmt polling:            │               │   if buffer full:        │
│   refresh queue fields   │<──memcpy─────<│     push ready entry,    │
│   for each ready entry:  │               │     pop next from free_q │
│     copy buf from device │<──memcpy─────<│                          │
│     resolve host ptr     │               │ dump_args_flush():       │
│     push to host ready_q │               │   push remaining buffers │
│   advance queue_heads,   │               │   to ready_q             │
│     refill free_queues   │               │                          │
│   write_range_to_device  │──memcpy──────>│                          │
│     for each modified    │               │                          │
│     field                │               │                          │
│                          │               │                          │
│ collector shard:         │               │                          │
│   wait_pop_ready         │               │                          │
│   on_buffer_collected →  │               │                          │
│     copy arena slice     │<──memcpy─────<│                          │
│     extract DumpedArgs   │               │                          │
│     queue to writer thrd │               │                          │
│   notify_copy_done       │               │                          │
│                          │               │                          │
│ rtStreamSynchronize      │               │                          │
│ stop()                   │               │                          │
│   join mgmt + collectors │               │                          │
│ reconcile_counters()     │               │                          │
│   recover leftovers      │               │                          │
│   + dropped accounting   │               │                          │
│ export_dump_files()      │               │                          │
│   → <output_prefix>/     │               │                          │
│     args_dump/...        │               │                          │
└──────────────────────────┘               └──────────────────────────┘
```

**Lifecycle** (`device_runner.cpp`):

```text
init_args_dump()
  dump_collector_.initialize(num_dump_threads, ..., output_prefix_)
  kernel_args_.args.dump_data_base = dump_collector_.get_dump_shm_device_ptr()
dump_collector_.start(thread_factory)   ← split mgmt + collector shards
launch AICPU / AICore
rtStreamSynchronize
dump_collector_.stop()                  ← join mgmt + collectors, drain final batch
dump_collector_.reconcile_counters()    ← recover leftover current buffers
                                          + dropped accounting
dump_collector_.export_dump_files()
dump_collector_.finalize()
```

[`ArgsDumpCollector`](../../src/common/platform/include/host/args_dump_collector.h)
on a5 inherits the same CRTP base
([`profiling_common::ProfilerBase`](../../src/common/platform/include/host/profiler_base.h))
as a2a3 and parameterizes
[`BufferPoolManager`](../../src/common/platform/include/host/buffer_pool_manager.h)
with `DumpModule`. The only a5-specific glue is the 5-callback
`MemoryOps`, targeted shared-field refreshes, and the on-demand arena copy
inside `on_buffer_collected`.

a5 normally relies on per-thread AICPU flush (`dump_args_flush`) to move
the current buffer into the ready queue. If a hang/op-timeout reaps AICPU
before that flush runs, `reconcile_counters` recovers a non-empty
`current_buf_ptr` host-side before export, then accumulates each thread's
`dropped_record_count` for the final anomaly report.

### 5.6 a2a3 vs a5 at a glance

| Aspect | a2a3 | a5 |
| ------ | ---- | -- |
| Device-side layout | identical (same `DumpDataHeader` / `DumpMetaBuffer` / arena shape, `static_assert`-checked) | |
| AICPU recording logic | identical | |
| Buffer model | rotating pool (free + ready queues per thread) | identical |
| Host threads | split mgmt + collector shards, streams during execution | identical |
| Host-class shape | `ProfilerBase<ArgsDumpCollector, DumpModule>` | identical |
| Host transport | `halHostRegister` shared memory | host-shadow `malloc` + targeted `rtMemcpy`/`memcpy` |
| `MemoryOps` callbacks | 3 (`alloc`, `reg`, `free_`) | 5 (+ `copy_to_device`, `copy_from_device`) |
| Arena access | direct via SVM | targeted `copy_from_device` inside `on_buffer_collected` |
| Tensor-payload support | Supported | payload consumers remain unsupported while #1560 is open |
| `reconcile_counters` | recover leftover current buffers + dropped accounting | identical |
| Lifecycle | `initialize` → `start` → `stop` → `reconcile_counters` → `export_dump_files` → `finalize` | identical |

## 6. Overhead

Args Dump is opt-in and zero-overhead when disabled — without
`--dump-args` the host does not allocate dump storage and AICPU /
AICore skip the dump-specific code paths. The `pipe_barrier(PIPE_ALL)`
before FIN is also gated on the same handshake bit.

With `--dump-args`, AICPU records level-selected tensor/scalar metadata and
copies tensor payload according to the level table in §3.1. Scalar values stay
manifest-only. When tensor payload is selected, the per-task overhead is
dominated by:

- The `BEFORE_DISPATCH` / `AFTER_COMPLETION` payload memcpy into
  the per-thread arena (contiguous fast-path; logical traversal for
  non-contiguous views).
- The completion `pipe_barrier(PIPE_ALL)` before writing FIN, which
  serializes all device-side writes for dumped tasks.
- The arena and metadata writes themselves; host drain/replenish and
  collector work runs concurrently with the stream on both architectures.
  a5 additionally pays `rtMemcpy`/`memcpy` transport cost to keep host
  shadows in sync.

For interactive debugging, total memory pressure is what to watch:
the default per-thread arena is 128 MiB
(`8 × 256 × 64 KiB`), so a 7-thread run reserves ~896 MiB on
device.

## 7. Limitations

Three pressure or failure conditions exist when dump buffers run out
of space. Successful arena backpressure preserves payload; truncation
and record discard surface in the JSON manifest plus the
`dump_args_flush` log line so users can detect and diagnose them.

### 7.1 Truncation (`truncated = true`)

**Trigger:** a single tensor's logical payload (`numel × elem_size`)
exceeds the entire per-thread arena size.

**Mechanism (identical on a2a3 and a5):** before copying, AICPU
compares `bytes` against `arena_size`. When `bytes > arena_size`,
only `arena_size / 2` bytes are copied and the record is flagged
`truncated = 1`.

```text
bytes = numel × elem_size
if bytes > arena_size:
    copy_bytes = arena_size / 2     ← half the arena
    truncated  = true
```

**Effect:** the tensor entry has `"truncated": true` and `bin_size`
is smaller than the full tensor. The payload contains the first
`arena_size / 2` bytes of the **logical** layout (gathered or
contiguous), enough for statistical sampling.

**Tuning:** raise `PLATFORM_DUMP_AVG_TENSOR_BYTES` (arena grows
proportionally) so the arena is at least as large as the biggest
tensor you need to inspect.

### 7.2 Arena payload freeze release

`arena_write_offset` remains monotonic and physical writes use `% arena_size`.
Before reserving an offset or copying payload bytes, AICPU checks whether the
arena is already one full physical cycle deep or whether the new payload would
cross the physical end. If so, it seals/publishes the current metadata buffer
and raises `fq_contended`; the host then opens, drains, and releases the existing
freeze cycle.

At each existing RQ publish point, AICPU counts the non-empty tensor payload
records in that metadata buffer. The host's single writer increments the
originating thread's completion count only after `args.bin` accepts a payload.
`backpressure_release_ready()` therefore holds an existing queue freeze until
each thread's written count equals its published count. The common framework
still independently requires all RQs drained and FQs refilled. Host then
acknowledges each completed per-thread watermark. The triggering thread requires
that acknowledgement, even if the common queue freeze has already released,
before aligning its monotonic logical offset to the next physical arena boundary
and writing the pending payload from the arena start.
The cursor is never reset and no reclaimed/published arena offset is needed.

### 7.3 Record discard (`dropped_record_count` / `dropped_records`)

**Trigger:** the metadata record buffer (not the payload arena) is
full and no replacement buffer is available.

**Mechanism (identical on a2a3 and a5):** each thread rotates through
multiple metadata buffers via an SPSC free queue. When a buffer
fills (256 records), AICPU tries to:

1. Enqueue the full buffer to the per-thread ready queue (for the
   host mgmt thread to pick up).
2. Pop a fresh buffer from the free queue.

If the ready queue is full or the free queue is empty, AICPU raises
the corresponding DFX contention signal and waits at the existing
bounded freeze gate while the host drains/refills the queues. If the
host-crash timeout expires, the current records are accounted as
dropped.

```text
// Reuse current buffer — account for lost records
account_dropped_records(state, cur_buf.count)
cur_buf.count = 0          ← reset and reuse
dropped_record_count += N  ← tracks total lost records
```

The same fallback applies during `dump_args_flush()` at end of
execution if the ready queue is full.

**Effect:** `dropped_records` in the manifest summary shows how
many argument records were lost. Dropped arguments do not appear in
the `args[]` array at all.

**Tuning:** raise `PLATFORM_DUMP_BUFFERS_PER_THREAD` (more
rotation buffers) and/or `PLATFORM_DUMP_READYQUEUE_SIZE` (deeper
host hand-off queue).

### 7.4 Summary matrix

| Condition | Flag | Metadata | Payload | a2a3 | a5 |
| --------- | ---- | -------- | ------- | ---- | -- |
| Tensor > arena | `truncated` | Preserved | Partial (`arena/2` bytes) | Same | Same |
| Arena host writer falls behind | none on success | Preserved | Preserved after bounded freeze | Same | Same |
| Record buffer full, no free buffer | `dropped_records` summary | Lost | Lost | After freeze timeout | Same |

### 7.5 Configuration knobs

All defaults live in each platform's `platform_config.h`. The values below
match between `a2a3` and `a5` except for the platform thread cap:

| Constant | Default | Effect |
| -------- | ------- | ------ |
| `PLATFORM_DUMP_RECORDS_PER_BUFFER` | 256 | Max records per metadata buffer |
| `PLATFORM_DUMP_BUFFERS_PER_THREAD` | 8 | Arena size multiplier and SPSC free queue depth |
| `PLATFORM_DUMP_AVG_TENSOR_BYTES` | 64 KiB | Arena size multiplier |
| `PLATFORM_DUMP_MAX_DIMS` | 5 | Upper bound on shape / offset arrays |
| `PLATFORM_MAX_AICPU_THREADS` | a2a3: 4; a5: 5 | Maximum number of dump-producing threads |

Per-thread arena =
`BUFFERS_PER_THREAD × RECORDS_PER_BUFFER × AVG_TENSOR_BYTES`
= `8 × 256 × 65536` = **128 MiB**.

## 8. FAQ and Debug Guide

**No `args_dump/` directory or `args_dump.json` in the output.** Check that
`--dump-args` was passed; without it (level 0) the host does not allocate dump
storage. The collector exports files only after receiving at least one record,
so the same result is expected when the kernel dispatches no arguments or when
Level 1 has no `CoreTaskArgs::dump(...)` markers. Add markers (§3.2), or pass
`--dump-args 2` for a full dump.

**a5 `args.bin` fails known-value validation.** Treat this as the payload-truth
problem tracked by [#1560](https://github.com/hw-native-sys/simpler/issues/1560)
while that issue remains open; it is not evidence that the recorded tensors
were actually zero. Use `a2a3` / `a2a3sim` for any payload consumer. Plain
level-3 metadata remains usable on a5 because it does not read tensor bytes.

**Manifest has tasks but expected tensor records are missing.** AICPU received
a payload whose tensor count or metadata did not match what the orchestrator
registered. Look for a `LOG_WARN` from `try_log_dump_args_layout_mismatch` — it
identifies the first mismatched task, then is suppressed to avoid
log flooding. A scalar-only task has no tensor records but still carries its
scalar records in the manifest. For `host_build_graph`, ensure
`set_tensor_info_to_task` (or
`add_task_with_tensor_info`) was called for every task. For
`tensormap_and_ringbuffer`, ensure every task that expects a tensor output calls
`alloc_tensors` with a `TensorCreateInfo`, and every input tensor
is added via `add_input` / `add_output` / `add_inout`.

**`AFTER_COMPLETION` data looks stale or partially written.** This
should not happen with the runtime barrier in place — AICore
issues `pipe_barrier(PIPE_ALL)` before FIN when dump is enabled.
If you see it, verify the executor saw `SIMPLER_DFX_FLAG_DUMP_ARGS`
set in the handshake (a missing handshake bit silently disables
the barrier).

**`truncated_args > 0` in summary.** A tensor exceeded the
per-thread arena (default 128 MiB). Bump
`PLATFORM_DUMP_AVG_TENSOR_BYTES` to extend the arena and rerun.

**`dropped_records > 0` in summary.** Metadata-buffer pressure.
On a5 raise `PLATFORM_DUMP_RECORDS_PER_BUFFER`; on a2a3 raise
`PLATFORM_DUMP_BUFFERS_PER_THREAD` and/or
`PLATFORM_DUMP_READYQUEUE_SIZE`.

**Viewer reports "no `outputs/*/args_dump` directory found".**
Either the run did not produce one (see first question), or the
viewer's working directory differs from the run's. Pass the
explicit dump-dir path to the viewer:
`python -m simpler_setup.tools.dump_viewer outputs/<case>_<ts>/args_dump`.

**The run hung (AICore op-timeout / scheduler timeout) — is there
still a dump?** Yes. The host exports the manifest even though `run()` returns
the timeout error. Within the selected level's record scope, it contains task
inputs captured at `BEFORE_DISPATCH`, outputs from tasks that completed before
the hang, and any recoverable partial output from an in-flight task. This rests
on the timeout ordering — the three budgets are tuned so the **AICPU detects the
hang first**, dumps, and only then the hardware/host timeouts fire:

```text
SCHEDULER_TIMEOUT_MS (20 s)           <  PLATFORM_OP_EXECUTE_TIMEOUT_US (45 s)  <  PLATFORM_STREAM_SYNC_TIMEOUT_MS (50 s)
   AICPU declares hang,                   STARS reaps the AICore op              host stream sync gives up
   flushes + dumps in-flight              and poisons the context                and surfaces the error
```

These defaults can be overridden without rebuilding by setting
`SIMPLER_SCHEDULER_TIMEOUT_MS`, `SIMPLER_OP_EXECUTE_TIMEOUT_US`, and
`SIMPLER_STREAM_SYNC_TIMEOUT_MS`. Invalid values, or onboard combinations that
break the ordering above, are ignored with a warning and fall back to the
defaults. The onboard host also requires stream-sync to cover the scheduler
budget plus a 1.5 s scheduler-arming guard for cold init work before the
no-progress timer starts. This guard covers fixed/cold costs such as kernel
registration, orchestration SO dlopen, runtime init, and AICore handshake.
It cannot know the graph-specific maximum orchestration producer wall time, so
callers that raise scheduler/op timeouts must also size
`SIMPLER_STREAM_SYNC_TIMEOUT_MS` for their worst-case orchestration window. Sim
builds do not have STARS or ACL stream-sync timeouts, but scheduler overrides
are still parsed and applied independently so slow CPU-sim kernels can raise
the no-progress budget without onboard-only ordering limits. CI restores the
old fast-fail values through these env vars: 2 s scheduler, 3 s op-execute,
and 4 s stream-sync for onboard jobs; 5 s scheduler for sim jobs.

- **Device-side graceful flush (primary).** At 20 s of no progress
  the AICPU declares the hang, runs the end-of-loop flush, *and*
  dumps the **partial output** of every task still RUNNING on a core
  — written at the `after_completion` stage, reflecting current GM,
  so you can see how far a stuck kernel got and how much it wrote.
  Because this fires before STARS reaps the op (45 s), it
  normally completes while the kernel is still wedged (and host-side
  recovery below backstops the rest). Best-effort:
  only AICore writes already drained to GM are visible (the stuck
  core issued no `pipe_barrier`, so writes still in its cache are
  not), and a read may be torn if the core is mid-write. To tell a
  running task's partial output apart from a genuine completion,
  cross-reference the `[STALL …] state=RUNNING` / `SHUTDOWN_SNAPSHOT`
  log lines — those task_ids are the in-flight ones.
- **Host-side recovery (backstop).** If the AICPU is killed before
  it finishes flushing (it loses the race, or `emergency_shutdown`
  blocks on the wedged core), its last buffer is left un-flushed on
  the device (`current_buf_ptr != 0`). The host's `reconcile_counters`
  detects this and recovers those records directly from the device
  buffer + arena (the device is not reset until *after* the export),
  so whatever the AICPU managed to record — including in-flight
  output written before the kill — still reaches the manifest.

This ordering is load-bearing: if the timeouts were inverted (STARS
reaping before the AICPU's budget, as in earlier versions), the
device-side dump would never run on a real AICore hang and you would
only recover what was already in the buffer. The chain lives in
`platform_config.h`, which holds all three (`PLATFORM_SCHEDULER_TIMEOUT_MS`,
surfaced as `SCHEDULER_TIMEOUT_MS` — 20 s, one value for onboard and sim,
`PLATFORM_OP_EXECUTE_TIMEOUT_US` /
`PLATFORM_STREAM_SYNC_TIMEOUT_MS`). The env overrides use those constants as
their unset fallback and keep the `#897` distributed-skew trade-off.

## 9. Related docs

- [profiling-framework.md](profiling-framework.md) — shared
  host-side collector framework (a2a3 only).
- [chip-level-arch.md](../chip-level-arch.md) — host / AICPU /
  AICore program boundaries this feature spans.
- [task-flow.md](../task-flow.md) — where AICPU dispatch and
  completion sit in the per-task state machine.
- [hierarchical-level-runtime.md](../hierarchical-level-runtime.md)
  — how L2 (this feature) relates to L3+ composition.
