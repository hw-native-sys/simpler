# Scalar Data Access — get/set_tensor_data Design

## 1. Overview

During task graph construction, orchestration sometimes needs to read InCore kernel results (for control-flow decisions) or write initial values into tensors. `get_tensor_data` / `set_tensor_data` provide **blocking** cross-layer data access, allowing orchestration to safely read and write tensor data.

**Core design principle**: Reuse the existing TensorMap dependency tracking mechanism — no new synchronization infrastructure.

## 2. API

```cpp
// Blocking read: returns value at the given indices (default: raw uint64_t bits)
// Specify T for typed read: float val = get_tensor_data<float>(tensor, 1, idx);
template<typename T = uint64_t>
T get_tensor_data(const ChipTensor &tensor, uint32_t ndims, const uint32_t indices[]);

// Blocking write: stores value at the given indices (type deduced from argument)
// Typed write: set_tensor_data(tensor, 1, idx, 42.0f);
// The tensor is const: the write targets buffer memory, never the descriptor.
template<typename T = uint64_t>
void set_tensor_data(const ChipTensor &tensor, uint32_t ndims, const uint32_t indices[], T value);
```

Both call into the runtime through the ops table — orchestration .so needs no runtime symbol linkage.

## 3. Blocking Interface Design

### 3.1 get_tensor_data Flow

```text
addr null-check → collect owner and overlapping writers → wait for each producer → compute offset → memcpy read
```

- **addr null-check**: `buffer.addr == 0` means unallocated — log error, return 0
- **Producer collection**: use `owner_task_id` for the allocation owner, then
  query TensorMap/OverlapMap for modifier writers overlapping the tensor view.
  One tensor can therefore require waiting for multiple tasks.
- **spin-wait**: wait until each tracked producer has
  `task_state >= CHIP_TASK_COMPLETED`.
- **No producer**: skip waiting only when there is neither a valid owner nor
  an overlapping writer. An empty lookup alone does not skip the owner wait.

### 3.2 set_tensor_data Flow

```text
addr null-check → collect owner and overlapping writers → wait for each producer and its consumers → memcpy write
```

For each tracked producer, wait for completion and then for its outstanding
consumers before moving to the next producer. Consumer counts occupy the low
bits; the scope reference is a flag, not a count of one:

```cpp
(fanout_refcount & ~FANOUT_SCOPE_BIT) >= (fanout_count & ~FANOUT_SCOPE_BIT)
```

### 3.3 Timeout

- Uses the platform system counter (`get_sys_cnt_aicpu()`), checked every 1024 spins.
- Defaults: 30 s in CPU simulation, 15 s onboard. The Host accepts
  `SIMPLER_TENSOR_DATA_TIMEOUT_MS` (1..2147483647 ms); invalid values warn and
  retain the backend default. Zero does not disable the deadline.
- The setting is latched at `Worker.init()`, via `InitArgs` onboard or the
  resident AICPU SO setter in simulation. Each scalar-access call reads the
  latched value once and converts it using `PLATFORM_PROF_SYS_CNT_FREQ`.
- Each producer gets its own timer, covering its upstream dependency latency.
  Writes start a separate timer for that producer's consumers. Unrelated
  completions never renew either timer.
- The scheduler watchdog has independent per-thread progress timers and checks
  task ownership before reporting a stall. Other threads completing work do not
  unconditionally renew every timer. At a scalar-wait polling checkpoint, an
  observed orchestration/scheduler error aborts the wait while preserving the
  recorded error code. Tensor expiry otherwise reports code 8.
- Failure sets `orch.fatal`; reads return zero and writes leave the element
  unchanged. Diagnostics show the budget, elapsed time and dependency identity.
- See [local timeout configuration](../../../../../docs/troubleshooting/local-timeout-defaults.md)
  for initialization, validation and the outer watchdog ordering.

## 4. Seeding a Runtime-Created Output

`TensorCreateInfo` carries no initial-value fill; a fresh allocation starts
with whatever the heap block last held. Two ways to give it defined content:

```cpp
// Orchestration-side write: alloc creates a hidden owner task that completes
// inline. Before any consumers are submitted, the wait finishes immediately.
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

// Read back the seeded value; no kernel was submitted in this example.
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
| 3 | Kernel read (INPUT) | Orch Write | WAR | spin-wait tracked producer fanout | **Requires a tracked producer; see below** |
| 4 | Kernel read-write (INOUT) | Orch Read | RAW | spin-wait producer COMPLETED | Yes |
| 5 | Kernel read-write (INOUT) | Orch Write | WAW+WAR | spin-wait producer + consumers | Yes |
| 6 | Orch Write | Kernel read (INPUT) | RAW | blocking completes before next submit | Yes |
| 7 | Orch Write | Kernel write (OUTPUT) | WAW | same — serial guarantee | Yes |
| 8 | Orch Read | Kernel write (OUTPUT) | WAR | same — serial guarantee | Yes |
| 9–12 | Orch ↔ Orch | — | — | same-thread serial execution | Yes |

### Key Design Points

**Scenario #3 is the only case requiring special attention**:

TensorMap tracks writer tasks, not standalone INPUT readers. For an external
tensor with no `owner_task_id` and no overlapping writer, an INPUT-only access
provides no producer slot whose fanout the scalar write can wait on. The write
can therefore race with a kernel still reading it. Runtime-created tensors also
have an owner slot; an empty TensorMap lookup does not by itself imply this risk.

**Solution**: For tensors that may later be written via `set_tensor_data`, use `add_inout()` instead of `add_input()`. INOUT registers a producer entry in TensorMap, enabling `set_tensor_data` to track all consumers through `fanout_refcount`.

**Scenarios #6–8 serial guarantee**:

get/set_tensor_data are blocking calls, and orchestration is single-threaded serial submission. After a blocking operation completes, subsequent code (including task submissions) executes strictly afterward.

## 7. External Tensor Behavior

`make_tensor_external()` creates tensors with a pre-set `buffer.addr` (pointing to host-allocated device memory).

| Scenario | Behavior |
| -------- | -------- |
| External tensor never submitted as OUTPUT/INOUT | No TensorMap entry — get/set execute immediately |
| External tensor previously submitted as OUTPUT/INOUT | TensorMap has producer entry — get/set spin-wait |
| External tensor submitted as INPUT, then set_tensor_data | **WAR risk** — must use INOUT instead (same as scenario #3) |

**Key rule**: If an external tensor will later be written via `set_tensor_data`, all prior kernel accesses must use `add_inout()`, not `add_input()`.
