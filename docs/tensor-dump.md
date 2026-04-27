# Tensor Dump — Runtime Tensor Capture

Tensor Dump captures per-task tensor inputs and outputs during kernel
execution and exports them to disk for offline inspection. It is a
runtime observability feature: host pre-allocates buffers on device,
AICPU writes records during execution, host collects data and exports
JSON manifest + binary payload.

Supported on both architectures (`a2a3` / `a5`) and all three runtimes
(`host_build_graph`, `aicpu_build_graph`, `tensormap_and_ringbuffer`).
Opt-in via `--dump-tensor` — zero overhead when disabled.

The **primary design** (a2a3) uses shared memory (`halHostRegister`) +
background threads for concurrent collection during execution. The a5
implementation uses a **temporary memcpy-based fallback** (batch
collect-after-sync) because a5 hardware does not yet support
host-pinned shared memory mapping. Device-side data structures and
AICPU recording logic are **identical** across both platforms — only the
host-side collection transport differs.

---

## 1. What gets captured

For every task that AICPU dispatches:

- **`TensorDumpRole`** — per formal callable signature (`IN` / `OUT` /
  `INOUT`).
- **`TensorDumpStage`** — `BEFORE_DISPATCH` (inputs snapshotted before
  the kernel runs) and `AFTER_COMPLETION` (outputs snapshotted after
  the kernel reports FIN). `INOUT` tensors are captured at both stages.
- **Metadata** — `task_id`, `subtask_id` (AIC / AIV0 / AIV1), `func_id`,
  `arg_index`, `dtype`, `shapes`, `raw_shapes`, `offsets`,
  `is_contiguous`.
- **Payload bytes** — copied from the tensor's device buffer into a
  per-thread circular arena. Non-contiguous views are gathered via
  logical traversal; contiguous views take a fast-path memcpy.

Each record is a fixed 128 B (two cache lines) — see `TensorDumpRecord`
in [`tensor_dump.h`](../src/a2a3/platform/include/common/tensor_dump.h).

---

## 2. Architecture

### 2.1 Common device-side structures

Both platforms share the same device-side layout, published via
`kernel_args.dump_data_base`:

```text
DumpSetupHeader                                 (host init, AICPU reads)
├── num_dump_threads
├── records_per_buffer
├── magic = 0x44554D50 ("DUMP")
├── dump_buffer_ptrs  [MAX_AICPU_THREADS]  ──> DumpBuffer    (per-thread)
├── arena_header_ptrs [MAX_AICPU_THREADS]  ──> DumpArenaHeader
├── arena_data_ptrs   [MAX_AICPU_THREADS]  ──> arena bytes
└── arena_sizes       [MAX_AICPU_THREADS]

DumpBuffer (per-thread, 64 B header + records[])
  ├── count          (AICPU writes)
  ├── capacity       (host sets)
  ├── dropped_count  (AICPU increments when full)
  └── TensorDumpRecord records[capacity]      ← 128 B each

DumpArenaHeader (per-thread)
  ├── write_offset   (AICPU monotonic cursor)
  └── arena_size     (host sets)

arena_data (per-thread, circular byte buffer)
  default = BUFFERS_PER_THREAD × RECORDS_PER_BUFFER × AVG_TENSOR_BYTES
          = 8 × 256 × 64 KiB = 128 MiB per thread
```

These structs are binary-identical between a2a3 and a5 (enforced by
`static_assert`). `dump_data_base` flows through `KernelArgs`, not
`Runtime` — AICPU reads it from `k_args->dump_data_base` in
`kernel.cpp` and passes it to `set_platform_dump_base()`. Dump
enablement itself is propagated separately via the per-worker
handshake field `enable_profiling_flag` (`bit0 =
PROFILING_FLAG_DUMP_TENSOR`), so device-side code must not infer
"dump enabled" from `dump_data_base != 0`.

### 2.2 a2a3 — shared-memory + background thread (primary design)

This is the canonical architecture. `halHostRegister` maps device memory
into host virtual address space so the host can read device buffers
directly without `rtMemcpy`. A `DumpMemoryManager` background thread
polls SPSC ready queues and recycles full metadata buffers **while
kernels are still executing**.

```text
        HOST                                         DEVICE
┌──────────────────────────┐               ┌──────────────────────────┐
│ TensorDumpCollector      │               │ AICPU thread             │
│                          │               │                          │
│ initialize()             │  alloc +      │ dump_tensor_init()       │
│   rtMalloc + halRegister │──register────>│   read DumpSetupHeader   │
│   build DumpDataHeader   │              │   cache per-thread ptrs  │
│                          │               │                          │
│ start_memory_manager()   │               │ per-task run loop:       │
│   ┌────────────────────┐ │               │   BEFORE_DISPATCH        │
│   │ DumpMemoryManager  │ │               │     dump_tensor_record() │
│   │ background thread  │ │ SPSC ready    │     → write to arena     │
│   │   poll ready queue │<┼──queues──────<│     → append record      │
│   │   recycle buffers  │─┼──free queue──>│     → push to ready_q    │
│   └────────────────────┘ │               │   dispatch kernel        │
│                          │               │   wait FIN               │
│ poll_and_collect()       │               │   AFTER_COMPLETION       │
│   concurrent thread      │ shared mem    │     dump_tensor_record() │
│   reads arena via host   │<──mapping────<│                          │
│   mapping (no memcpy)    │               │                          │
│                          │               │ dump_tensor_flush()      │
│ signal_execution_complete│               │   log per-thread stats   │
│ stop_memory_manager()    │               └──────────────────────────┘
│ drain_remaining_buffers()│
│ scan_remaining_buffers() │
│                          │
│ export_dump_files()      │
│   → outputs/tensor_dump_ │
│     YYYYMMDD_HHMMSS/     │
│       <same-name>.json   │
│       <same-name>.bin    │
└──────────────────────────┘
```

**Execution flow** (`device_runner.cpp`):

```text
init_tensor_dump()
  dump_collector_.initialize(...)
  kernel_args_.args.dump_data_base = dump_collector_.get_dump_shm_device_ptr()
start_memory_manager()           ← spawn background polling thread
launch AICPU / AICore
spawn collector_thread           ← poll_and_collect() concurrent with execution
rtStreamSynchronize              ← wait for kernel completion
signal_execution_complete()      ← tell background thread to drain
stop_memory_manager()
drain_remaining_buffers()        ← pick up any stragglers
scan_remaining_dump_buffers()    ← scan partial records still on device
export_dump_files()
```

Key classes (a2a3):

- [`DumpMemoryManager`](../src/a2a3/platform/include/host/tensor_dump_collector.h) —
  background thread: polls device ready queues, hands full buffers to
  main thread, recycles them back to device free queue.
- [`TensorDumpCollector`](../src/a2a3/platform/include/host/tensor_dump_collector.h) —
  main thread: `initialize` / `start_memory_manager` /
  `poll_and_collect` / `signal_execution_complete` /
  `drain_remaining_buffers` / `export_dump_files` / `finalize`.

### 2.3 a5 — memcpy batch (temporary fallback)

a5 hardware does not yet support `halHostRegister`. All device-to-host
transfers use `rtMemcpy` / `memcpy`, which requires the device to have
stopped writing. Collection happens **only after**
`rtStreamSynchronize`. No background threads, no SPSC queues.

This is a temporary simplification; the a5 implementation should migrate
to the a2a3 shared-memory design once `halHostRegister` becomes
available on a5 hardware.

```text
        HOST                                         DEVICE
┌──────────────────────────┐               ┌──────────────────────────┐
│ TensorDumpCollector      │               │ AICPU thread             │
│                          │               │                          │
│ initialize()             │  alloc +      │ dump_tensor_init()       │
│   rtMalloc / malloc      │──copy────────>│   read DumpSetupHeader   │
│   build DumpSetupHeader  │               │   cache per-thread ptrs  │
│   copy to device         │               │                          │
│                          │               │ per-task run loop:       │
│ (no background thread)   │               │   BEFORE_DISPATCH        │
│                          │               │     dump_tensor_record() │
│ ── kernel execution ──   │               │   dispatch kernel        │
│                          │               │   wait FIN               │
│ rtStreamSynchronize      │               │   AFTER_COMPLETION       │
│                          │               │     dump_tensor_record() │
│ collect_all()            │  batch        │                          │
│   2-step per thread:     │<──memcpy─────<│ dump_tensor_flush()      │
│   1. copy DumpBuffer hdr │               │   log per-thread stats   │
│      read count          │               └──────────────────────────┘
│   2. copy records+arena  │
│                          │
│ export_dump_files()      │
│   → outputs/tensor_dump_ │
│     YYYYMMDD_HHMMSS/     │
│       <same-name>.json   │
│       <same-name>.bin    │
└──────────────────────────┘
```

**Execution flow** (`device_runner.cpp`):

```text
init_tensor_dump()
  dump_collector_.initialize(...)
  kernel_args_.args.dump_data_base = dump_collector_.get_dump_setup_device_ptr()
launch AICPU / AICore              ← no background thread
rtStreamSynchronize                ← wait for kernel completion
collect_all()                      ← batch memcpy all buffers back
export_dump_files()
```

Key class (a5):

- [`TensorDumpCollector`](../src/a5/platform/include/host/tensor_dump_collector.h) —
  `initialize` / `collect_all` / `export_dump_files` / `finalize`.
  No `DumpMemoryManager`, no `start_memory_manager`.

### 2.4 Common: where dump calls are wired in

Each runtime's scheduler dispatch code calls `dump_tensors_for_task` at
two points in the per-task state machine (for `tensormap_and_ringbuffer`,
this is in `runtime/scheduler/scheduler_completion.cpp` and
`runtime/scheduler/scheduler_dispatch.cpp`):

```text
┌──────────────────────────────────────┐
│ per-task dispatch:                   │
│   if enable_dump_tensor {            │
│     dump_tensors_for_task(           │
│         BEFORE_DISPATCH);            │
│   }                                  │
│   dispatch(task);                    │
│   wait FIN;                          │
│   if enable_dump_tensor {            │
│     dump_tensors_for_task(           │
│         AFTER_COMPLETION);           │
│   }                                  │
│   retire(task);                      │
└──────────────────────────────────────┘
```

`dump_tensors_for_task` walks the formal callable signature, matches
each non-scalar slot to a `TensorDumpInfo` (dtype + shape + offsets +
device address), and calls `dump_tensor_record` for slots that match
the current stage (inputs `BEFORE`, outputs `AFTER`, inouts both).

When dump tensor is enabled, AICore executors also read
`enable_profiling_flag` and issue `pipe_barrier(PIPE_ALL)` after task
execution but before the FIN handshake. This closes the ordering gap
where `AFTER_COMPLETION` snapshots could observe output buffers before
all device-side writes were globally visible.

### 2.5 Common: tensor metadata registration

AICPU only has device addresses and sizes — it does **not** know the
logical shape / dtype / view geometry of each tensor unless the runtime
registers it. Each of the three runtimes exposes metadata through a
slightly different path, but they all converge on `TensorInfo` (see
[`tensor_info.h`](../src/a5/runtime/host_build_graph/runtime/tensor_info.h)):

- **`host_build_graph`** — two orchestration-side APIs:
  - `add_task()` → `set_tensor_info_to_task(task_id, info[], count)`
  - `add_task_with_tensor_info()` (single call convenience wrapper)

  See
  [`dump_tensor_orch.cpp`](../tests/st/a5/host_build_graph/dump_tensor_example/kernels/orchestration/dump_tensor_orch.cpp)
  for both styles in one file.
- **`aicpu_build_graph`** — runtime layer fills `TensorInfo` from
  `PTO2TaskPayload::tensors[]` directly. No orchestration API needed.
- **`tensormap_and_ringbuffer`** — identical to `aicpu_build_graph`;
  the ring buffer carries `PTO2TaskPayload` which already contains
  shape/offset arrays.

When metadata is missing or inconsistent, the task is **skipped for
dump** and a single `LOG_WARN` is emitted (guarded by
`try_log_tensor_dump_layout_mismatch` to avoid log flooding). Normal
execution is never affected.

---

## 3. Usage

### 3.1 Enable at runtime

The user-facing switch is still
`CallConfig::enable_dump_tensor`, but the runtime no longer relies
on `dump_data_base` as the enable signal.

Current propagation chain:

1. Python / test harness sets `CallConfig::enable_dump_tensor`.
2. Host `DeviceRunner` allocates dump storage and publishes its base
   address via `kernel_args.dump_data_base`.
3. Host also sets `PROFILING_FLAG_DUMP_TENSOR` in each worker
   handshake's `enable_profiling_flag`.
4. The onboard AICPU kernel calls:
   - `set_platform_dump_base(k_args->dump_data_base)` to publish the
     backing storage location
   - `set_enable_dump_tensor(GET_PROFILING_FLAG(...))` to publish the
     actual enable state
5. AICore executors read the same handshake bit to decide whether to
   insert the extra completion barrier before FIN.

When `enable_dump_tensor = false`, the host does not allocate dump
storage and neither AICPU nor AICore enters dump-specific paths.

**From a scene test** (`SceneTestCase.run_module` / pytest):

```bash
# Standalone runner
python tests/st/a5/host_build_graph/dump_tensor_example/test_dump_tensor_example.py \
    -p a5sim --dump-tensor

# pytest
pytest tests/st/a5/host_build_graph/dump_tensor_example --platform a5sim --dump-tensor
```

**From any scene test via pytest**:

```bash
pytest examples/a5/host_build_graph/vector_example --platform a5sim --dump-tensor
```

### 3.2 Output layout

```text
outputs/
└── tensor_dump_<YYYYMMDD_HHMMSS>/
    ├── tensor_dump_<YYYYMMDD_HHMMSS>.json
    └── tensor_dump_<YYYYMMDD_HHMMSS>.bin
```

The JSON file is the manifest; `bin_file` points at the sibling binary
payload file.

`tensor_dump.json`:

```json
{
  "run_dir": "tensor_dump",
  "bin_format": {
    "type": "logical_contiguous",
    "byte_order": "little_endian"
  },
  "total_tensors": 1,
  "before_dispatch": 1,
  "after_completion": 0,
  "input_tensors": 1,
  "output_tensors": 0,
  "inout_tensors": 0,
  "truncated_tensors": 0,
  "dropped_records": 0,
  "dropped_overwrite": 0,
  "bin_file": "tensor_dump.bin",
  "tensors": [
    {
      "task_id": "0x0000000200000a00",
      "subtask_id": 1,
      "role": "input",
      "stage": "before_dispatch",
      "func_id": 0,
      "arg_index": 0,
      "dtype": "float32",
      "shape": [16384],
      "raw_shape": [16384],
      "offsets": [0],
      "is_contiguous": true,
      "truncated": false,
      "overwritten": false,
      "bin_offset": 0,
      "bin_size": 65536
    }
  ]
}
```

### 3.3 Inspect with `dump_viewer`

The viewer auto-picks the latest `outputs/tensor_dump_*` directory
when invoked without arguments. It loads the `*.json` manifest found in
that directory and uses its `bin_file` field to locate the payload:

```bash
# List every dumped tensor in the latest run
python -m simpler_setup.tools.dump_viewer

# Filter and save matching tensors to human-readable .txt files
python -m simpler_setup.tools.dump_viewer --func 0 --stage before --role input --export

# Export one specific entry by its manifest index
python -m simpler_setup.tools.dump_viewer --index 42

# Pin to a specific dump directory
python -m simpler_setup.tools.dump_viewer outputs/<case>_<ts>/tensor_dump \
    --task 0x0000000200000a00 --export
```

Exported `.txt` files include metadata headers, a row-major overview
with aligned columns, and a detail listing with multi-dim indices —
safe to diff against golden tensors or pipe into a spreadsheet.

### 3.4 Add dump support to a new test

Only `host_build_graph` needs explicit wiring; the other two runtimes
pick up metadata automatically.

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

See the full template:
[`tests/st/a5/host_build_graph/dump_tensor_example`](../tests/st/a5/host_build_graph/dump_tensor_example/)
(and the `a2a3` mirror at `tests/st/a2a3/host_build_graph/dump_tensor_example`).

---

## 4. Configuration knobs

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

---

## 5. Memory-pressure behaviour

Three distinct failure modes exist when dump buffers run out of space.
All three are **safe** — they never crash the kernel or corrupt
execution — and all three surface in the JSON manifest plus the
`dump_tensor_flush` log line so users can detect and diagnose them.

### 5.1 Truncation (`truncated = true`)

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

**Effect:** the tensor entry in the manifest has `"truncated": true`
and `bin_size` is smaller than the full tensor. The payload contains
the first `arena_size / 2` bytes of the **logical** layout (gathered
or contiguous), enough for statistical sampling.

**Tuning:** increase `PLATFORM_DUMP_AVG_TENSOR_BYTES` (arena grows
proportionally) so that the arena is at least as large as the biggest
tensor you need to inspect.

### 5.2 Overwrite (`overwritten = true`)

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

Because a2a3 uses shared memory and a background reader, the host can
drain arena data **while the kernel is still running**. Overwrite
happens only when AICPU writes faster than the host can read — i.e.
many large tensors arrive in rapid succession without the host keeping
up.

**a5 mechanism:** same arithmetic, but detection happens in
`collect_all()` after `rtStreamSynchronize`:

```text
write_offset = arena_header.write_offset   (total bytes ever written)
if write_offset > arena_size:
    oldest_valid = write_offset − arena_size
    if record.payload_offset < oldest_valid:
        overwritten = true
```

Because a5 collects only after the stream finishes, the entire
execution window's data must fit in the arena. If total payload bytes
written across all tasks exceed `arena_size`, the earliest payloads
are overwritten.

**Effect:** overwritten records appear in the manifest with
`"overwritten": true` and zero payload bytes in the binary file.
Metadata (shape, dtype, task_id) is preserved — only the raw data
is lost.

**Tuning:** increase `PLATFORM_DUMP_BUFFERS_PER_THREAD` (arena grows
proportionally) so total payload fits, or reduce the number of tasks
being dumped.

### 5.3 Record discard (`dropped_count` / `dropped_records`)

**Trigger:** the metadata record buffer (not the payload arena) is
full and no replacement buffer is available.

**a5 mechanism (simple):** each thread has a single `DumpBuffer` with
`capacity = RECORDS_PER_BUFFER` (default 256). When `count >=
capacity`, subsequent `dump_tensor_record()` calls increment
`dropped_count` and return immediately — **no metadata, no payload**
is stored for that tensor.

```text
if buf.count >= buf.capacity:
    buf.dropped_count++
    return              ← tensor silently skipped
```

**a2a3 mechanism (rotating buffers):** each thread rotates through
multiple `DumpMetaBuffer`s via an SPSC free queue. When a buffer fills
(256 records), AICPU tries to:

1. **Enqueue** the full buffer to the per-thread ready queue (for the
   host background thread to pick up).
2. **Pop** a fresh buffer from the free queue.

If the ready queue is full or the free queue is empty, AICPU
spin-waits up to `DUMP_SPIN_WAIT_LIMIT` (1 000 000 iterations) to
give the host `DumpMemoryManager` time to replenish. If the wait
expires:

```text
// Overwrite current buffer — account for lost records
account_dropped_records(state, cur_buf.count)
cur_buf.count = 0          ← reset and reuse
dropped_record_count += N  ← tracks total lost records
```

The same fallback applies during `dump_tensor_flush()` at end of
execution if the ready queue is full.

**Effect:** `dropped_records` in the manifest summary shows how many
tensor records were lost. Individual dropped tensors do **not** appear
in the `tensors[]` array at all — they are gone without trace.

**Tuning:** increase `PLATFORM_DUMP_BUFFERS_PER_THREAD` (more
rotation buffers) and/or `PLATFORM_DUMP_READYQUEUE_SIZE` (deeper host
handoff queue).

### 5.4 Summary matrix

| Condition | Flag | Metadata | Payload | a2a3 | a5 |
| --------- | ---- | -------- | ------- | ---- | -- |
| Tensor > arena | `truncated` | Preserved | Partial (`arena/2` bytes) | Same | Same |
| Arena wraps, old data overwritten | `overwritten` | Preserved | Lost (zero bytes in bin) | Rare (concurrent drain) | Likely if total data > arena |
| Record buffer full, no free buffer | `dropped_count` | Lost | Lost | After spin-wait fallback | Immediate when count ≥ capacity |

---

## 6. Completion ordering

Older implementations could capture stale `AFTER_COMPLETION` output
data: AICPU observed FIN and dumped the tensor before all device-side
writes were guaranteed visible in GM.

The current implementation fixes this in the runtime, not in each
individual kernel. When dump tensor is enabled,
`aicore_executor.cpp` issues `pipe_barrier(PIPE_ALL)` immediately after
task execution and before writing the FIN handshake value. That makes
the `AFTER_COMPLETION` snapshot happen only after dump-relevant output
writes have drained.

This barrier is gated on `PROFILING_FLAG_DUMP_TENSOR`, so non-dump runs
keep the original cheaper completion path.

---

## 7. Related docs

- [chip-level-arch.md](chip-level-arch.md) — host / AICPU / AICore
  program boundaries this feature spans.
- [task-flow.md](task-flow.md) — where AICPU dispatch and completion
  sit in the per-task state machine.
- [hierarchical_level_runtime.md](hierarchical_level_runtime.md) — how
  L2 (this feature) relates to L3+ composition.
