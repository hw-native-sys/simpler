# Scalar Data Access — get/set_tensor_data Design

## 1. Overview

During task graph construction, orchestration sometimes needs to read InCore kernel results (for control-flow decisions) or write initial values into tensors. `get_tensor_data` / `set_tensor_data` provide **blocking** cross-layer data access, allowing orchestration to safely read and write tensor data.

**Core design principle**: Reuse the existing TensorMap dependency tracking for
automatic discovery. A caller-declared task list covers the accesses that
tracking does not record.

## 2. API

```cpp
// Blocking read: returns value at the given indices (default: raw uint64_t bits)
// Specify T for typed read: float val = get_tensor_data<float>(tensor, 1, idx);
template<typename T = uint64_t>
T get_tensor_data(const ChipTensor &tensor, uint32_t ndims, const uint32_t indices[],
                  const TaskId *deps = nullptr, uint32_t dep_count = 0);

// Blocking write: stores value at the given indices (type deduced from argument)
// Typed write: set_tensor_data(tensor, 1, idx, 42.0f);
// The tensor is const: the write targets buffer memory, never the descriptor.
template<typename T = uint64_t>
void set_tensor_data(const ChipTensor &tensor, uint32_t ndims, const uint32_t indices[], T value,
                     const TaskId *deps = nullptr, uint32_t dep_count = 0);
```

Both call into the runtime through the ops table — orchestration .so needs no runtime symbol linkage.

## 3. Blocking Interface Design

### 3.1 get_tensor_data Flow

```text
addr null-check → validate declared deps → TensorMap lookup → spin-wait producer COMPLETED
                → spin-wait each declared task COMPLETED → compute flat offset → memcpy read
```

- **addr null-check**: `buffer.addr == 0` means unallocated — log error, return 0
- **declared deps**: validate every id before waiting (§3.3); an invalid id is `INVALID_ARGS`, return 0
- **TensorMap lookup**: find producer task by `buffer.addr`
- **spin-wait**: wait until producer `task_state >= CHIP_TASK_COMPLETED`
- **No producer** (lookup callback never fires): skip the automatic producer wait; declared dependencies still apply
- **declared tasks**: wait for each declared task still holding its slot to reach `COMPLETED`

### 3.2 set_tensor_data Flow

```text
addr null-check → validate declared deps → TensorMap lookup → spin-wait producer COMPLETED
                → spin-wait consumers done → spin-wait each declared task COMPLETED
                → spin-wait that task's retained consumers done → memcpy write
```

One extra step versus get_tensor_data: wait for retained consumers to finish (`(fanout_refcount & ~FANOUT_SCOPE_BIT) >= (fanout_count & ~FANOUT_SCOPE_BIT)`, excluding the scope reference). The same consumer wait applies to each declared task, because a declared task is treated as a live accessor of the buffer.

### 3.3 Declared Dependencies

Automatic discovery reads `owner_task_id` and the TensorMap. Both are products
of automatic dependency generation, and both are switched off for two cases
(`dep_compute.h`):

- inside a **manual scope**, `register_task_outputs()` returns immediately, so
  nothing a task writes is registered;
- a tensor carrying **`manual_dep`** is skipped even in an automatic scope.

An external, host-staged tensor (invalid `owner_task_id`) submitted as `INOUT`
/ `OUTPUT_EXISTING` through either path is therefore invisible to both
accessors: `get_tensor_data` returns the pre-kernel bytes, and
`set_tensor_data` writes a value the kernel later overwrites or races a reader.
`deps` / `dep_count` express the missing ordering on the accessor itself:

```cpp
TaskOutputTensors outs = rt_submit_aiv_task(FUNC, task);
const TaskId dep = outs.task_id();
float v = get_tensor_data<float>(tensor, 1, idx, &dep, 1);
set_tensor_data(tensor, 1, idx, 9.0f, &dep, 1);
```

- `get_tensor_data` waits until every declared task reaches `COMPLETED`, in
  addition to the producers automatic discovery finds.
- `set_tensor_data` performs the same waits and additionally waits for each
  declared task's retained consumers. Treating the declared task as a live accessor is
  what makes WAR against an unregistered reader expressible.
- Validation runs before any wait: each id must be valid, its ring inside the
  ring domain, and its local id below the next allocation on that ring
  (`TaskAllocator::task_head()`) — the task must already be submitted in this
  run. Any other value is `SIMPLER_ERROR_INVALID_ARGS`, and the accessor does
  not touch memory.
- A declared task that has already been consumed adds no wait: consumption
  implies completion, and its slot may already hold a newer task.
- The defaults (`nullptr`, `0`) preserve the previous behavior, so existing
  call sites need no change.
- The parameter exists only in `tensormap_and_ringbuffer`. `host_build_graph`
  keeps its own `orchestration_api.h` without it: its orchestration runs to
  completion before the device starts, so a produced tensor has no value to
  order against, and both accessors reject producer tensors (#2080).

Only consumers with `DEP_RETAIN` remain represented in the fanout count until
completion. An ordering-only (`DEP_WAIT`) edge releases its reference at wiring;
passing its producer does **not** wait for that consumer. Pass every such reader
directly in `deps` before overwriting its buffer. This also applies to automatic
modifier edges, which are ordering-only.

### 3.4 Timeout

- Uses cycle counter (`get_sys_cnt_aicpu()`), checked every 1024 spins
- Threshold: `TENSOR_DATA_TIMEOUT_CYCLES`, derived from the platform default
  (15 s onboard, 30 s in simulation)
- On timeout: sets `orch.fatal = true`, preventing further task submission

## 4. Seeding a Runtime-Created Output

`TensorCreateInfo` carries no initial-value fill; a fresh allocation starts
with whatever the heap block last held. Two ways to give it defined content:

```cpp
// Orchestration-side write: alloc returns before any producer or consumer
// exists, so set_tensor_data finds nothing to wait for and writes at once.
// Suited to scalars and a handful of elements; each call writes one element.
TensorCreateInfo ci(shapes, ndims, dtype);
TaskOutputTensors outs = alloc_tensors(ci);
const ChipTensor &t = outs.get_ref(0);
set_tensor_data(t, 1, idx, 42.0f);

// Device-side fill: a task (a dedicated seed kernel, or a prefix of the
// first consumer) writes the buffer. Suited to whole-buffer initialization
// such as zeroing a large accumulator; task dependencies order every reader
// and writer after the seed.
```

**Constraint**: existing tensors are write targets only through `add_inout()`.

## 5. Scalar Dependencies via 1-Element Tensors

Traditional scalars (`CoreTaskArgs::add_scalar`) are one-way inputs with no TensorMap tracking. For cross-task scalar values, use a 1-element tensor as the carrier:

```cpp
uint32_t shapes[1] = {1};
TensorCreateInfo scalar_ci(shapes, 1, DataType::FLOAT32);

// Allocate, seed from orchestration, and keep the returned tensor
TaskOutputTensors outs = alloc_tensors(scalar_ci);
const ChipTensor &scalar_tensor = outs.get_ref(0);
uint32_t idx[1] = {0};
set_tensor_data(scalar_tensor, 1, idx, 77.0f);

// Orchestration-side blocking read (waits for kernel completion)
float val = get_tensor_data<float>(scalar_tensor, 1, idx);
```

**Advantage**: Fully reuses existing TensorMap (producer tracking, fanin/fanout dependencies) — no new infrastructure needed.

## 6. Data Hazard Analysis

Three actors:

- **Kernel**: InCore task submitted via add_input/add_output/add_inout (asynchronous execution)
- **Orch Read**: orchestration calls `get_tensor_data` (blocking read)
- **Orch Write**: orchestration calls `set_tensor_data` (blocking write)

### Hazard Matrix (earlier operation → later operation)

| # | Earlier Op | Later Op | Hazard | Guarantee | Safe? |
| - | ---------- | -------- | ------ | --------- | ----- |
| 1 | Kernel write (OUTPUT) | Orch Read | RAW | spin-wait producer COMPLETED | Yes |
| 2 | Kernel write (OUTPUT) | Orch Write | WAW | spin-wait producer COMPLETED | Yes |
| 3 | Kernel read (INPUT) | Orch Write | WAR | spin-wait fanout_refcount | **Needs INOUT, or declare the reader in `deps`** |
| 4 | Kernel read-write (INOUT) | Orch Read | RAW | spin-wait producer COMPLETED | Yes |
| 5 | Kernel read-write (INOUT) | Orch Write | WAW+WAR | spin-wait producer + consumers | Yes |
| 6 | Orch Write | Kernel read (INPUT) | RAW | blocking completes before next submit | Yes |
| 7 | Orch Write | Kernel write (OUTPUT) | WAW | same — serial guarantee | Yes |
| 8 | Orch Read | Kernel write (OUTPUT) | WAR | same — serial guarantee | Yes |
| 9–12 | Orch ↔ Orch | — | — | same-thread serial execution | Yes |

### Key Design Points

**Scenario #3 is the only case requiring special attention**:

TensorMap tracks only producers (OUTPUT/INOUT), not pure INPUT consumers. If a tensor is only registered via `add_input()`, TensorMap has no producer entry for it. `set_tensor_data`'s `wait_for_tensor_ready()` finds no matching producer (the lookup callback never fires) and returns immediately — but the kernel may still be reading → **WAR data race**.

**Solution**: For tensors that may later be written via `set_tensor_data`, pass the reader's task id in `deps` (§3.3). That orders the write after the reader without promoting it to a TensorMap writer. Alternatively, use `add_inout()` instead of `add_input()`: INOUT registers a producer entry in TensorMap, enabling `set_tensor_data` to track retained consumers through `fanout_refcount`, at the cost of serializing concurrent readers.

**Scenarios #6–8 serial guarantee**:

get/set_tensor_data are blocking calls, and orchestration is single-threaded serial submission. After a blocking operation completes, subsequent code (including task submissions) executes strictly afterward.

## 7. External Tensor Behavior

`make_tensor_external()` creates tensors with a pre-set `buffer.addr` (pointing to host-allocated device memory).

| Scenario | Behavior |
| -------- | -------- |
| External tensor never submitted as OUTPUT/INOUT | No TensorMap entry — get/set execute immediately |
| External tensor previously submitted as OUTPUT/INOUT | TensorMap has producer entry — get/set spin-wait |
| External tensor submitted as INPUT, then set_tensor_data | **WAR risk** — declare the reader in `deps`, or use INOUT instead (same as scenario #3) |

**Key rule**: If an external tensor will later be written via `set_tensor_data`, every prior kernel reader must either be declared in `deps` or access the tensor through `add_inout()`.
