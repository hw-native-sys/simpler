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
addr null-check → TensorMap lookup → spin-wait producer COMPLETED → compute flat offset → memcpy read
```

- **addr null-check**: `buffer.addr == 0` means unallocated — log error, return 0
- **TensorMap lookup**: find producer task by `buffer.addr`
- **spin-wait**: wait until producer `task_state >= CHIP_TASK_COMPLETED`
- **No producer** (lookup callback never fires): skip waiting, read immediately

### 3.2 set_tensor_data Flow

```text
addr null-check → writer/reader lookup → wait for conflicting accesses → memcpy write
```

In addition to the creator/writer and writer-consumer guards, wait for each overlapping registered reader to finish. Reader edges do not extend this wait to their consumers.

### 3.3 Timeout

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
| 3 | Kernel read (INPUT) | Orch Write | WAR | spin-wait registered reader COMPLETED | Yes |
| 4 | Kernel read-write (INOUT) | Orch Read | RAW | spin-wait producer COMPLETED | Yes |
| 5 | Kernel read-write (INOUT) | Orch Write | WAW+WAR | spin-wait producer + consumers | Yes |
| 6 | Orch Write | Kernel read (INPUT) | RAW | blocking completes before next submit | Yes |
| 7 | Orch Write | Kernel write (OUTPUT) | WAW | same — serial guarantee | Yes |
| 8 | Orch Read | Kernel write (OUTPUT) | WAR | same — serial guarantee | Yes |
| 9–12 | Orch ↔ Orch | — | — | same-thread serial execution | Yes |

### Access tracking

Ordinary INPUT arguments in automatic scopes register readers. A scalar write
waits for every overlapping reader to reach COMPLETED, including readers of
external storage with no producer. It does not wait for those readers' downstream
consumers. Scalar reads do not wait for other readers.

Manual scopes, manual_dep tensors and NO_DEP arguments do not publish accesses;
the caller remains responsible for synchronizing them. Blocking scalar calls
finish before subsequent submissions in the same orchestration thread.

## 7. External Tensor Behavior

| Prior registered access | Scalar read | Scalar write |
| ----------------------- | ----------- | ------------ |
| None and no owner | Immediate | Immediate |
| Writer | Wait for writer | Wait for writer and its consumers |
| Reader | No reader wait | Wait for reader completion |
