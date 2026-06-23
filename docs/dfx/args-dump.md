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
execution, and the host exports a JSON manifest plus a binary payload.
The result is a stable, replayable record of every dumped argument a kernel
saw, without the timing distortion of inline printing.

## 2. Overview

- **Per-task input/output capture.** Inputs snapshotted before
  dispatch, outputs snapshotted after FIN; `INOUT` tensors at both
  stages.
- **Logical shape preserved.** Records carry dtype, shape,
  `strides`, `start_offset`, and `is_contiguous` so logical views are
  reconstructable.
- **Manifest + binary payload.** `--dump-args` writes a JSON
  manifest plus one `.bin` payload per run; tensor entries carry
  `bin_offset` / `bin_size`, while scalar entries stay manifest-only.
- **Unified scalar args.** Scalar values are emitted as
  `kind: scalar`, `stage: before_dispatch`, zero-dim records in
  `args_dump.json`; there is no separate args-only manifest. Their
  final `dtype` is registered at submit time in a dump-only per-task side table
  and resolved by AICPU when writing each scalar dump record.
- **Cross-architecture.** Same flags and on-disk layout family on
  `a2a3` and `a5`. Both runtimes are wired through.

Enable in one line (`2` = full dump, every task):

```bash
python tests/st/<case>/test_<name>.py -p a5sim --dump-args 2
```

## 3. How to Use

### 3.1 Enable Args Dump

`--dump-args` takes an optional **level**:

| Level | Meaning |
| ----- | ------- |
| `0` (or flag absent) | off — zero overhead |
| `1` (bare `--dump-args`) | **partial** — only args marked with `L0TaskArgs::dump(...)` (see §3.2) |
| `2` (`--dump-args 2`) | **full** — every task's tensor inputs/outputs and scalar args |
| `3` (`--dump-args 3`) | **full, JSON only** — every task's tensor/scalar metadata (shape, dtype, strides, scalar values) to `args_dump.json`; no payload captured, no `.bin` file |

Level 3 exists for consumers that need only the per-task argument metadata,
not the tensor element data — e.g. feeding the manifest into tooling. The AICPU
skips the arena payload copy entirely (treating every tensor like a scalar
for payload purposes), so it incurs none of full mode's device→host
bandwidth or arena pressure, and the manifest's `bin_file` is `null` with
every `bin_size` at `0`.

```bash
# Standalone runner
python tests/st/<case>/test_<name>.py -p a5sim --dump-args 2     # full
python tests/st/<case>/test_<name>.py -p a2a3 -d 0 --dump-args   # partial (level 1)

# pytest
pytest tests/st/<case> --platform a5sim --dump-args 2
pytest examples/a5/host_build_graph/vector_example --platform a5sim --dump-args 2
```

The level sets `CallConfig::enable_dump_tensor` (0/1/2/3). The host then
allocates dump storage, publishes its base address through
`kernel_args.dump_data_base`, and sets `PROFILING_FLAG_DUMP_TENSOR`
(levels 1, 2, and 3) in each worker handshake's `enable_profiling_flag` for
the enable/disable decision. The **partial / full / full-json-only**
distinction is carried as a `DumpTensorLevel` in the dump shared-memory header
(`DumpDataHeader::dump_tensor_level`, host-written before launch) rather
than in the profiling-flag bitmask. The on-device AICPU reads the storage base
via `set_platform_dump_base()`, the enable bit via
`set_dump_args_enabled(GET_PROFILING_FLAG(...))`, and latches the mode
in `dump_args_init()` from the header level (`PARTIAL` → selective,
`FULL_JSON_ONLY` → metadata-only, payload copy suppressed). Because the
level is decided host-side **before any task is
dispatched**, it is latched up front — there is no dependence on task
submission order. AICore executors read the same `PROFILING_FLAG_DUMP_TENSOR`
bit to insert a `pipe_barrier(PIPE_ALL)` before FIN when dump is on, so
`AFTER_COMPLETION` snapshots see the kernel's final writes.

### 3.2 Partial Dump — Select Specific Args

Partial dump (level 1) captures only the tasks whose `L0TaskArgs` is marked
with `dump(...)`; every unmarked task is skipped. Within a marked task,
only the selected tensor/scalar args are recorded. Mark the arguments on
the relevant `L0TaskArgs` before submission:

```cpp
L0TaskArgs args;
args.add_input(x);
args.add_input(y);
args.add_output(z);
float scale = 1.0f;
args.add_scalar(scale);
args.dump(x, z, scale);
rt_submit_aiv_task(FUNC_ADD, args);
```

`dump(...)` selects arguments from the current `L0TaskArgs`; it does not execute
a dump immediately. The selected tensors and scalar lvalues must already
belong to that `L0TaskArgs`. Scalar selection is by the lvalue passed to
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
  task (every tensor and scalar argument on the `L0TaskArgs`), without
  enumerating them:

  ```cpp
  L0TaskArgs args;
  args.add_input(x);
  args.add_input(y);
  args.add_output(z);
  args.add_scalar(scale);
  args.dump();        // whole task: every tensor/scalar arg on this L0TaskArgs
  rt_submit_aiv_task(FUNC_ADD, args);
  ```

Partial vs full is chosen by the **dump level** (§3.1), latched host-side
before any dispatch — not inferred from the markers, so it never depends
on task submission order. At level 1, tasks without a marker are skipped
and marked tasks dump only their selected arguments. At level 2, the
markers are ignored and every task's tensors/scalars are dumped. At level
3, every task's tensors/scalars are dumped as metadata only (no payload,
no `.bin`). With no
`--dump-args` (level 0) dump is off entirely.

If you run at level 1 but place no `dump(...)` markers anywhere, the
manifest is empty — that is the deliberate "only what I marked" contract.
Use `--dump-args 2` when you want everything.

### 3.3 Output

The dump artifacts land under the per-task output prefix
(`CallConfig::output_prefix`, set by
`scene_test.py::_build_output_prefix` to
`outputs/<ClassName>_<case>_<YYYYMMDD_HHMMSS>/` for SceneTest runs):

```text
<output_prefix>/
└── args_dump/
    ├── args_dump.json  # unified argument manifest (`--dump-args`)
    └── args.bin   # raw tensor payload (`--dump-args`)
```

Filenames are fixed (no per-file timestamp) — the directory is the
per-task uniqueness boundary. `--dump-args` emits both files in the
same `args_dump/` directory.

#### `args_dump.json` — Unified manifest

`args_dump.json` is the manifest; its `bin_file` field points at
the sibling binary payload. At level 3 (full, JSON only) no payload is
captured: `bin_file` is `null`, no `.bin` is written, and every entry's
`bin_size` is `0`.

Example manifest (one input tensor captured before dispatch):

```json
{
  "run_dir": "args_dump",
  "bin_format": {
    "type": "logical_contiguous",
    "byte_order": "little_endian"
  },
  "total_args": 1,
  "before_dispatch": 1,
  "after_completion": 0,
  "input_args": 1,
  "output_args": 0,
  "inout_args": 0,
  "truncated_args": 0,
  "dropped_records": 0,
  "dropped_overwrite": 0,
  "bin_file": "args.bin",
  "args": [
    {
      "task_id": "0x0000000200000a00",
      "func_id": 0,
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
      "overwritten": false,
      "bin_offset": 0,
      "bin_size": 65536
    }
  ]
}
```

Key fields:

- `task_id` — runtime task identity. Use to correlate with swimlane / PMU
  output.
- `func_id` — kernel id of the subtask that declared this arg (via its
  incore `signature` + `arg_index`). The dump walks each active subtask
  independently, so a tensor shared by a cooperative mix is emitted **once
  per declaring subtask**, each under its own `func_id`; the sharing is
  recoverable as the same `(task_id, arg_index)` appearing under more than
  one `func_id`. Scalars are stamped with the first active subtask's
  `func_id`. `-1` when unknown (the `host_build_graph` dump path, which does
  not thread func_id).
- `arg_index` — payload argument index. Tensor entries use the payload
  tensor index; scalar entries use `payload.tensor_count + scalar_index`.
  When scalar lvalue selection matched multiple scalar slots, the selected
  first-match scalar entry carries `arg_index_ambiguous: true`.
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
- `truncated` / `overwritten` — set when the tensor exceeded arena
  size or was overwritten by a later task; see §7.
- Top-level `dropped_records` / `dropped_overwrite` counters
  surface aggregate loss — useful for spot-checking a run.

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

For `tensormap_and_ringbuffer`, each incore declares two parallel,
equal-length arrays in its `CALLABLE` entry — `signature` (each
tensor's `ArgDirection`) and `arg_index` (the absolute payload slot it
maps to). The dump reads them to give every captured tensor a direction
(which sets its capture stage) and a `func_id`; geometry (shape, dtype,
strides) is still derived automatically from the payload. `arg_index`
is mandatory and must equal `signature` in length — `CoreCallable.build`
raises otherwise. A tensor shared by a cooperative mix is declared by
every subtask that uses it, and dumped once per subtask.

A `signature` usually lists only tensor directions — scalars are
independent (added via `add_scalar`) and are typically omitted. If a
`SCALAR` direction *is* listed it must come after every tensor entry;
the dump skips `SCALAR` entries and `arg_index` indexes only the tensor
payload, so a scalar entry's `arg_index` value is immaterial (it exists
only to keep the two arrays equal-length).

`host_build_graph` has no incore signatures, so it needs explicit
`TensorInfo` wiring instead (geometry is still automatic):

```cpp
// In orchestration C++ (host_build_graph only)
TensorInfo info_a = make_tensor_info_from_tensor_arg(orch_args.tensor(0));
TensorInfo info_b = make_tensor_info_from_tensor_arg(orch_args.tensor(1));
TensorInfo info_f = make_tensor_info_from_tensor_arg(orch_args.tensor(2));

int t0 = add_task(runtime, args_t0, 4, /*func_id=*/0, CoreType::AIV);
TensorInfo t0_info[] = {info_a, info_b, info_f};
set_tensor_info_to_task(runtime, t0, t0_info, 3);

// Or in one call
int t1 = add_task_with_tensor_info(
    runtime, args_t1, /*num_args=*/3, /*func_id=*/1, CoreType::AIV,
    t1_info, /*tensor_count=*/1);
```

Full template:
[`tests/st/a5/host_build_graph/dump_tensor`](../../tests/st/a5/host_build_graph/dump_tensor/)
(and the `a2a3` mirror at
`tests/st/a2a3/host_build_graph/dump_tensor`).

## 4. Capabilities

What you can read out of `args_dump.json` + `args.bin`:

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
- **Loss accounting** — `truncated` / `overwritten` per-record
  flags, plus aggregate `dropped_records` / `dropped_overwrite` in
  the summary.

## 5. Design Highlights

`L0TaskArgs::dump(...)` selection state is compiled only when
`PTO2_PROFILING=1`. With `PTO2_PROFILING=0`, the public API remains
available but acts as a no-op: no dump-only `L0TaskArgs` state is stored and
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
└── dump_tensor_level  (DumpTensorLevel: 0=off, 1=partial, 2=full, 3=full_json_only; AICPU latches in dump_args_init)

DumpBufferState[num_dump_threads]               (per-thread)
├── free_queue {buffer_ptrs[SLOT_COUNT], head, tail}
├── current_buf_ptr            (AICPU active DumpMetaBuffer*)
├── current_buf_seq
├── arena_base / arena_size    (per-thread arena pointers)
├── arena_write_offset         (AICPU monotonic cursor)
└── dropped_record_count

DumpMetaBuffer pool (rotated)                   (BUFFERS_PER_THREAD per thread)
└── TensorDumpRecord records[RECORDS_PER_BUFFER] + count   ← 128 B each

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
(`bit0 = PROFILING_FLAG_DUMP_TENSOR`); the AICPU kernel entry calls
`set_dump_args_enabled()` with the decoded bit, so device-side
code does not infer "dump enabled" from `dump_data_base != 0`.

Each record is fixed at 128 B (two cache lines) — see
`TensorDumpRecord` in
[`tensor_dump.h`](../src/a2a3/platform/include/common/tensor_dump.h).

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

`dump_args_for_task` walks **each active subtask's** callable
signature. For each non-scalar entry it reads the parallel `arg_index`
to locate the absolute payload tensor slot that entry describes, builds
a `TensorDumpInfo` (dtype + shape + strides + start offset + device
address) for that slot, and calls `dump_arg_record` for entries whose
role matches the current stage — stamping each record with that
subtask's `func_id`. A tensor shared by a cooperative mix is thus
emitted once per declaring subtask, each under its own `func_id`. A
declared `arg_index` beyond the payload is skipped with a warning
(memory safety); payload slots that no subtask declares are left
undumped with a soft completeness warning. (The `host_build_graph` dump
path is a separate overload that maps each signature entry to its slot
positionally and stamps `func_id = -1`.) Scalar values are dumped
separately from the flat payload `scalars[]`, once per task (stamped
with the first active subtask's `func_id`); their dtype table is
registered at submit time and emitted as a dump-only metadata record at
`BEFORE_DISPATCH`.

When dump is enabled, AICore executors also issue
`pipe_barrier(PIPE_ALL)` after kernel execution and before writing
the FIN handshake. This closes the ordering gap where
`AFTER_COMPLETION` snapshots could observe output buffers before
all device-side writes were globally visible. Older
implementations could capture stale output data; the current
implementation fixes this in the runtime, not in each individual
kernel. The barrier is gated on `PROFILING_FLAG_DUMP_TENSOR`, so
non-dump runs keep the original cheaper completion path.

### 5.3 Tensor metadata registration

AICPU has device addresses and sizes — the logical shape, dtype,
and view geometry come from the runtime. Each runtime exposes
metadata through a slightly different path, but they all converge
on `TensorInfo` (see
[`tensor_info.h`](../src/a5/runtime/host_build_graph/runtime/tensor_info.h)):

- **`host_build_graph`** — two orchestration-side APIs:
  - `add_task()` → `set_tensor_info_to_task(task_id, info[], count)`
  - `add_task_with_tensor_info()` (single-call convenience wrapper)

  See
  [`dump_tensor_orch.cpp`](../../tests/st/a5/host_build_graph/dump_tensor/kernels/orchestration/dump_tensor_orch.cpp)
  for both styles in one file.
- **`tensormap_and_ringbuffer`** — runtime layer fills `TensorInfo`
  from `PTO2TaskPayload::tensors[]` directly. The ring buffer
  carries `PTO2TaskPayload` which already contains shape/offset
  arrays, so no orchestration API is needed.

When metadata is missing or inconsistent, the task is skipped for
dump and a single `LOG_WARN` is emitted (guarded by
`try_log_dump_args_layout_mismatch` to avoid log flooding);
normal execution continues.

### 5.4 a2a3 — shared-memory streaming

`halHostRegister` maps device memory into host virtual address
space so the host can read device buffers directly.
`TensorDumpCollector` runs two background threads on top of a
[`BufferPoolManager<DumpModule>`](../src/common/platform/include/host/buffer_pool_manager.h):
a mgmt thread that polls SPSC ready queues and recycles full
metadata buffers **while kernels are still executing**, plus a
poll thread that drains the L2 hand-off queue into
`on_buffer_collected`.

```text
        HOST                                         DEVICE
┌──────────────────────────┐               ┌──────────────────────────┐
│ TensorDumpCollector      │               │ AICPU thread             │
│                          │               │                          │
│ initialize()             │  alloc +      │ dump_args_init()         │
│   rtMalloc + halRegister │──register────>│   read DumpDataHeader    │
│   build DumpDataHeader   │              │   cache per-thread ptrs  │
│                          │               │                          │
│ start()                  │               │ per-task run loop:       │
│   ┌────────────────────┐ │               │   BEFORE_DISPATCH        │
│   │ mgmt thread        │ │               │     dump_arg_record()    │
│   │ (BufferPool driver)│ │ SPSC ready    │     → write to arena     │
│   │   poll ready queue │<┼──queues──────<│     → append record      │
│   │   recycle buffers  │─┼──free queue──>│     → push to ready_q    │
│   └────────────────────┘ │               │   dispatch kernel        │
│   ┌────────────────────┐ │               │   wait FIN               │
│   │ poll thread        │ │               │   AFTER_COMPLETION       │
│   │   reads arena via  │ │ shared mem    │     dump_arg_record()    │
│   │   host mapping     │<┼──mapping─────<│                          │
│   └────────────────────┘ │               │                          │
│                          │               │ dump_args_flush()        │
│ stop()                   │               │   log per-thread stats   │
│   join mgmt → join poll  │               └──────────────────────────┘
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
init_tensor_dump()
  dump_collector_.initialize(..., output_prefix_)
  kernel_args_.args.dump_data_base = dump_collector_.get_dump_shm_device_ptr()
start()                          ← spawn mgmt thread (drains L1 ringbuffer)
                                   then spawn poll thread (consumes L2 queue)
launch AICPU / AICore
rtStreamSynchronize              ← wait for kernel completion
stop()                           ← join mgmt (its final-drain pass into L2
                                   has poll as the consumer), then signal
                                   poll and join it
reconcile_counters()             ← recover leftover current buffers
                                   + dropped accounting
export_dump_files()
```

[`TensorDumpCollector`](../src/a2a3/platform/include/host/tensor_dump_collector.h)
on a2a3 inherits from
[`profiling_common::ProfilerBase<TensorDumpCollector, DumpModule>`](../src/common/platform/include/host/profiler_base.h):
the base class owns the mgmt thread, the poll thread, and the
`BufferPoolManager<DumpModule>` they share. `TensorDumpCollector`
only supplies the dump-specific pieces — the `DumpModule` trait
that describes the shared-memory layout, `initialize` that
allocates and pre-fills free queues, an `on_buffer_collected`
callback that gathers payload bytes into the in-memory record
list, plus `reconcile_counters` / `export_dump_files` /
`finalize`. The mgmt/poll threading, buffer pooling, and `Module`
trait pattern are shared with PMU and L2Swimlane — see
[profiling-framework.md](../profiling-framework.md) for the
framework reference.

### 5.5 a5 — same framework, host-shadow transport

a5's `TensorDumpCollector` derives from
`ProfilerBase<TensorDumpCollector, DumpModule>` and shares the
mgmt + poll thread structure with a2a3. The single behavioral
deviation from §5.4 is the **transport channel**: a5 has no
`halHostRegister`, so each device buffer is paired with a
host-shadow `malloc()` and the mgmt loop synchronizes the two via
`profiling_copy.h` (`rtMemcpy` onboard, `memcpy` in sim).
`MemoryOps` therefore carries five callbacks (`alloc` / `reg` /
`free_` / `copy_to_device` / `copy_from_device`); the mgmt loop
mirrors the entire shm region (`DumpDataHeader` + per-thread
`DumpBufferState`) device → host at the top of every tick, then
pushes back only the fields host modified (advanced
`queue_heads[q]`, refilled `free_queue.tail` and
`buffer_ptrs[slot]`) via `BufferPoolManager::write_range_to_device`.
The bulk `mirror_shm_to_device` is **not** called from the mgmt
loop: it would race with AICPU writes to device-only fields
(`current_buf_ptr`, `total/dropped` counters, `queue_tails`,
`free_queue.head`) and roll them back. Each popped
`DumpMetaBuffer` is still pulled on demand inside
`ProfilerAlgorithms::process_entry`. The per-thread arena lives
outside the shm region, so `on_buffer_collected` issues an
additional `copy_from_device` for the arena bytes referenced by
the buffer's records.

```text
        HOST                                         DEVICE
┌──────────────────────────┐               ┌──────────────────────────┐
│ TensorDumpCollector      │               │ AICPU thread             │
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
│   mgmt_thread starts     │               │   wait FIN               │
│   poll_thread starts     │               │   AFTER_COMPLETION       │
│                          │               │     dump_arg_record()    │
│ mgmt every 10us tick:    │               │   if buffer full:        │
│   copy_from_device(shm)  │<──memcpy─────<│     push ready entry,    │
│   for each ready entry:  │               │     pop next from free_q │
│     copy buf from device │<──memcpy─────<│                          │
│     resolve host ptr     │               │ dump_args_flush():       │
│     push to L2 ready_q   │               │   push remaining buffers │
│   advance queue_heads,   │               │   to ready_q             │
│     refill free_queues   │               │                          │
│   write_range_to_device  │──memcpy──────>│                          │
│     for each modified    │               │                          │
│     field                │               │                          │
│                          │               │                          │
│ poll thread:             │               │                          │
│   wait_pop_ready         │               │                          │
│   on_buffer_collected →  │               │                          │
│     copy arena slice     │<──memcpy─────<│                          │
│     extract DumpedTensors│               │                          │
│     queue to writer thrd │               │                          │
│   notify_copy_done       │               │                          │
│                          │               │                          │
│ rtStreamSynchronize      │               │                          │
│ stop()                   │               │                          │
│   join mgmt + poll       │               │                          │
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
init_tensor_dump()
  dump_collector_.initialize(num_dump_threads, ..., output_prefix_)
  kernel_args_.args.dump_data_base = dump_collector_.get_dump_shm_device_ptr()
dump_collector_.start(thread_factory)   ← mgmt + poll threads
launch AICPU / AICore
rtStreamSynchronize
dump_collector_.stop()                  ← join mgmt + poll, drain final batch
dump_collector_.reconcile_counters()    ← recover leftover current buffers
                                          + dropped accounting
dump_collector_.export_dump_files()
dump_collector_.finalize()
```

[`TensorDumpCollector`](../src/a5/platform/include/host/tensor_dump_collector.h)
on a5 inherits the same CRTP base
([`profiling_common::ProfilerBase`](../src/common/platform/include/host/profiler_base.h))
as a2a3 and parameterizes
[`BufferPoolManager`](../src/common/platform/include/host/buffer_pool_manager.h)
with `DumpModule`. The only a5-specific glue is the 5-callback
`MemoryOps`, the per-tick shm mirror, and the on-demand arena copy
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
| Host threads | mgmt + poll, streams during execution | identical |
| Host-class shape | `ProfilerBase<TensorDumpCollector, DumpModule>` | identical |
| Host transport | `halHostRegister` shared memory | host-shadow `malloc` + per-tick `rtMemcpy`/`memcpy` |
| `MemoryOps` callbacks | 3 (`alloc`, `reg`, `free_`) | 5 (+ `copy_to_device`, `copy_from_device`) |
| Arena access | direct via SVM | `copy_from_device` inside `on_buffer_collected` |
| `reconcile_counters` | recover leftover current buffers + dropped accounting | identical |
| Lifecycle | `initialize` → `start` → `stop` → `reconcile_counters` → `export_dump_files` → `finalize` | identical |

## 6. Overhead

Args Dump is opt-in and zero-overhead when disabled — without
`--dump-args` the host does not allocate dump storage and AICPU /
AICore skip the dump-specific code paths. The `pipe_barrier(PIPE_ALL)`
before FIN is also gated on the same handshake bit.

With `--dump-args`, AICPU records full `BEFORE_DISPATCH` /
`AFTER_COMPLETION` tensor payloads plus manifest-only
`BEFORE_DISPATCH` scalar args. The per-task overhead is dominated by:

- The `BEFORE_DISPATCH` / `AFTER_COMPLETION` payload memcpy into
  the per-thread arena (contiguous fast-path; logical traversal for
  non-contiguous views).
- The completion `pipe_barrier(PIPE_ALL)` before writing FIN, which
  serializes all device-side writes for dumped tasks.
- The arena and metadata writes themselves; the host transport
  cost is taken concurrently on a2a3 (mgmt + poll threads) or after
  the stream finishes on a5.

For interactive debugging, total memory pressure is what to watch:
the default per-thread arena is 128 MiB
(`8 × 256 × 64 KiB`), so a 7-thread run reserves ~896 MiB on
device.

## 7. Limitations

Three failure modes exist when dump buffers run out of space. All
three surface in the JSON manifest plus the `dump_args_flush`
log line so users can detect and diagnose them.

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

### 7.2 Overwrite (`overwritten = true`)

**Trigger:** the circular arena wraps around and AICPU writes new
payload data over a region whose metadata record has already been
emitted but whose payload has not yet been consumed by the host.

**a2a3 mechanism:** the arena is a monotonic-offset circular buffer.
`arena_write_offset` grows without bound; the actual write position
is `offset % arena_size`. When the host processes a record it
compares the record's `payload_offset` against a high-water mark:

```text
high_water = max payload_offset seen so far (maintained per-thread)
if high_water > arena_size:
    oldest_valid = high_water − arena_size
    if record.payload_offset < oldest_valid:
        overwritten = true
```

Because a2a3 uses shared memory and a background reader, the host
can drain arena data while the kernel is still running. Overwrite
happens only when AICPU writes faster than the host can read —
many large tensors arrive in rapid succession without the host
keeping up.

**a5 mechanism:** same arithmetic, but detection happens in
`collect_all()` after `rtStreamSynchronize`:

```text
write_offset = arena_header.write_offset   (total bytes ever written)
if write_offset > arena_size:
    oldest_valid = write_offset − arena_size
    if record.payload_offset < oldest_valid:
        overwritten = true
```

a5 collects only after the stream finishes, so the entire execution
window's data must fit in the arena. If total payload bytes written
across all tasks exceed `arena_size`, the earliest payloads are
overwritten.

**Effect:** overwritten records appear with `"overwritten": true`
and zero payload bytes in the binary file. Metadata (shape, dtype,
task_id) is preserved — only the raw data is lost.

**Tuning:** raise `PLATFORM_DUMP_BUFFERS_PER_THREAD` (arena grows
proportionally) so total payload fits, or reduce the number of
tasks being dumped.

### 7.3 Record discard (`dropped_count` / `dropped_records`)

**Trigger:** the metadata record buffer (not the payload arena) is
full and no replacement buffer is available.

**a5 mechanism (simple):** each thread has a single `DumpBuffer`
with `capacity = RECORDS_PER_BUFFER` (default 256). When `count >=
capacity`, subsequent `dump_arg_record()` calls increment
`dropped_count` and return immediately — no metadata, no payload
is stored for that tensor.

```text
if buf.count >= buf.capacity:
    buf.dropped_count++
    return              ← tensor silently skipped
```

**a2a3 mechanism (rotating buffers):** each thread rotates through
multiple metadata buffers via an SPSC free queue. When a buffer
fills (256 records), AICPU tries to:

1. Enqueue the full buffer to the per-thread ready queue (for the
   host mgmt thread to pick up).
2. Pop a fresh buffer from the free queue.

If the ready queue is full or the free queue is empty, AICPU
spin-waits up to `DUMP_SPIN_WAIT_LIMIT` (1 000 000 iterations) to
give the host mgmt thread (driving `BufferPoolManager<DumpModule>`)
time to replenish. If the wait expires:

```text
// Overwrite current buffer — account for lost records
account_dropped_records(state, cur_buf.count)
cur_buf.count = 0          ← reset and reuse
dropped_record_count += N  ← tracks total lost records
```

The same fallback applies during `dump_args_flush()` at end of
execution if the ready queue is full.

**Effect:** `dropped_records` in the manifest summary shows how
many tensor records were lost. Dropped tensors do not appear in
the `tensors[]` array at all.

**Tuning:** raise `PLATFORM_DUMP_BUFFERS_PER_THREAD` (more
rotation buffers) and/or `PLATFORM_DUMP_READYQUEUE_SIZE` (deeper
host hand-off queue).

### 7.4 Summary matrix

| Condition | Flag | Metadata | Payload | a2a3 | a5 |
| --------- | ---- | -------- | ------- | ---- | -- |
| Tensor > arena | `truncated` | Preserved | Partial (`arena/2` bytes) | Same | Same |
| Arena wraps, old data overwritten | `overwritten` | Preserved | Lost (zero bytes in bin) | Rare (concurrent drain) | Likely if total data > arena |
| Record buffer full, no free buffer | `dropped_count` | Lost | Lost | After spin-wait fallback | Immediate when count ≥ capacity |

### 7.5 Configuration knobs

All defaults live in
[`platform_config.h`](../src/a2a3/platform/include/common/platform_config.h)
and match between `a2a3` and `a5`:

| Constant | Default | Effect |
| -------- | ------- | ------ |
| `PLATFORM_DUMP_RECORDS_PER_BUFFER` | 256 | Max records per DumpBuffer (a2a3: per metadata buffer) |
| `PLATFORM_DUMP_BUFFERS_PER_THREAD` | 8 | Arena size multiplier (a2a3: also SPSC free queue depth) |
| `PLATFORM_DUMP_AVG_TENSOR_BYTES` | 64 KiB | Arena size multiplier |
| `PLATFORM_DUMP_MAX_DIMS` | 5 | Upper bound on shape / offset arrays |
| `PLATFORM_MAX_AICPU_THREADS` | 7 | Number of dump-producing threads |

Per-thread arena =
`BUFFERS_PER_THREAD × RECORDS_PER_BUFFER × AVG_TENSOR_BYTES`
= `8 × 256 × 65536` = **128 MiB**.

## 8. FAQ / Debug Guide

**No `args_dump/` directory in the output.** Check that
`--dump-args` was passed; without it (level 0) the host does not
allocate dump storage. Verify the run wrote to the expected
`<output_prefix>` and that the kernel actually executed (a kernel
that exits early before any task dispatch produces an empty
manifest).

**`args_dump/` exists but the manifest captured nothing.** You are
most likely at the default partial level (`--dump-args` = level 1)
with no `L0TaskArgs::dump(...)` markers in the orchestration, so every task is
skipped. Add markers (§3.2), or pass `--dump-args 2` for a full dump.

**Manifest has tasks but `tensors[]` is empty.** AICPU received a
task whose `TensorInfo` was missing or inconsistent. Look for a
`LOG_WARN` from `try_log_dump_args_layout_mismatch` — it
identifies the first mismatched task, then is suppressed to avoid
log flooding. For `host_build_graph`, ensure
`set_tensor_info_to_task` (or
`add_task_with_tensor_info`) was called for every task.

**`AFTER_COMPLETION` data looks stale or partially written.** This
should not happen with the runtime barrier in place — AICore
issues `pipe_barrier(PIPE_ALL)` before FIN when dump is enabled.
If you see it, verify the executor saw `PROFILING_FLAG_DUMP_TENSOR`
set in the handshake (a missing handshake bit silently disables
the barrier).

**`truncated_args > 0` in summary.** A tensor exceeded the
per-thread arena (default 128 MiB). Bump
`PLATFORM_DUMP_AVG_TENSOR_BYTES` to extend the arena and rerun.

**`dropped_overwrite > 0` in summary.** On a5, the run produced
more total payload than fits in the arena; on a2a3, the host
mgmt/poll threads couldn't keep up. Reduce the number of dumped
tasks (filter by `func_id` upstream) or increase
`PLATFORM_DUMP_BUFFERS_PER_THREAD`.

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
still a dump?** Yes, and it includes the hung kernel's own partial
output. The host exports the manifest even though `run()` returns the
timeout error, and you get the inputs of every dispatched task
(captured at `BEFORE_DISPATCH`) plus the `AFTER_COMPLETION` output of
every task that completed before the hang. This rests on the timeout
ordering — the three budgets are tuned so the **AICPU detects the
hang first**, dumps, and only then the hardware/host timeouts fire:

```text
SCHEDULER_TIMEOUT_MS (2 s, onboard)  <  PLATFORM_OP_EXECUTE_TIMEOUT_US (3 s)  <  PLATFORM_STREAM_SYNC_TIMEOUT_MS (4 s)
   AICPU declares hang,                   STARS reaps the AICore op              host stream sync gives up
   flushes + dumps in-flight              and poisons the context                and surfaces the error
```

- **Device-side graceful flush (primary).** At 2 s of no progress
  the AICPU declares the hang, runs the end-of-loop flush, *and*
  dumps the **partial output** of every task still RUNNING on a core
  — written at the `after_completion` stage, reflecting current GM,
  so you can see how far a stuck kernel got and how much it wrote.
  Because this fires ~1 s before STARS reaps the op (3 s), it
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
`spin_hint.h` (`PLATFORM_SCHEDULER_TIMEOUT_MS`, surfaced as
`SCHEDULER_TIMEOUT_MS` — 2 s onboard, 5 s in sim where there is no STARS to
race) and `platform_config.h` (`PLATFORM_OP_EXECUTE_TIMEOUT_US` /
`PLATFORM_STREAM_SYNC_TIMEOUT_MS`), along with the `#897` distributed-skew
trade-off.

## 9. Related docs

- [profiling-framework.md](../profiling-framework.md) — shared
  host-side collector framework (a2a3 only).
- [chip-level-arch.md](../chip-level-arch.md) — host / AICPU /
  AICore program boundaries this feature spans.
- [task-flow.md](../task-flow.md) — where AICPU dispatch and
  completion sit in the per-task state machine.
- [hierarchical_level_runtime.md](../hierarchical_level_runtime.md)
  — how L2 (this feature) relates to L3+ composition.
