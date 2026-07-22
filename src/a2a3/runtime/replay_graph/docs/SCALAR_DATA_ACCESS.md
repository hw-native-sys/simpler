# Scalar Data Access — get/set_tensor_data Design

## 1. Overview

During task graph construction, orchestration sometimes needs to read InCore kernel results (for control-flow decisions) or write initial values into tensors. `get_tensor_data` / `set_tensor_data` provide **blocking** cross-layer data access, allowing orchestration to safely read and write tensor data.

**Core design principle**: Reuse the existing TensorMap dependency tracking mechanism — no new synchronization infrastructure.

## 2. API

```cpp
// Blocking read: returns value at the given indices (default: raw uint64_t bits)
// Specify T for typed read: float val = get_tensor_data<float>(tensor, 1, idx);
template<typename T = uint64_t>
T get_tensor_data(const Tensor& tensor, uint32_t ndims, const uint32_t indices[]);

// Blocking write: stores value at the given indices (type deduced from argument)
// Typed write: set_tensor_data(tensor, 1, idx, 42.0f);
template<typename T = uint64_t>
void set_tensor_data(Tensor& tensor, uint32_t ndims, const uint32_t indices[], T value);
```

Both call into the runtime through the ops table — orchestration .so needs no runtime symbol linkage.

## 3. Blocking Interface Design

### 3.1 get_tensor_data Flow

```text
addr null-check → TensorMap lookup → spin-wait producer COMPLETED → compute flat offset → memcpy read
```

- **addr null-check**: `buffer.addr == 0` means unallocated — log error, return 0
- **TensorMap lookup**: find producer task by `buffer.addr`
- **spin-wait**: wait until producer `task_state >= PTO2_TASK_COMPLETED`
- **No producer** (lookup callback never fires): skip waiting, read immediately

### 3.2 set_tensor_data Flow

```text
addr null-check → TensorMap lookup → spin-wait producer COMPLETED → memcpy write
```

Identical to `get_tensor_data`. In `tensormap_and_ringbuffer`, `set_tensor_data`
had an extra step — wait for all consumers to finish (a WAR drain) — but the
single-shot replay model dropped the `CONSUMED` lifecycle and the consumer
counter (`fanout_refcount`) it relied on, so the consumer-drain is gone. Both
get and set now wait only for the producer to reach `PTO2_TASK_COMPLETED`
(WAW safety). No workload currently calls `set_tensor_data`, so there is
nothing to regress; the WAR caveat below applies if one ever does.

### 3.3 Timeout

- Uses cycle counter (`get_sys_cnt_aicpu()`), checked every 1024 spins
- Threshold: `PTO2_TENSOR_DATA_TIMEOUT_CYCLES` (~10 s at 1.5 GHz)
- On timeout: sets `orch.fatal = true`, preventing further task submission

## 4. add_output with Initial Value

```cpp
TensorCreateInfo ci(shapes, ndims, dtype);
ci.set_initial_value(initial_value);
args.add_output(ci);
```

**Mechanism**:

1. `ci.set_initial_value(value)` marks the create-info with an initial value before submission
2. `add_output(ci)` stores a pointer to `ci` in `L0TaskArgs` (the original must remain valid until submit)
3. During payload init, the output tensor is materialized via `init_from_create_info()` which triggers the fill
4. Fill strategy:
   - Small buffer (< 64 B): element-by-element memcpy directly into dst
   - Large buffer (≥ 64 B): fill the first 64 bytes as a template block, then bulk-memcpy in 64 B chunks; partial tail copy for remainder

**Constraint**: existing tensors are write targets only through `add_inout()`.

## 5. Scalar Dependencies via 1-Element Tensors

Traditional scalars (`L0TaskArgs::add_scalar`) are one-way inputs with no TensorMap tracking. For cross-task scalar values, use a 1-element tensor as the carrier:

```cpp
uint32_t shapes[1] = {1};
TensorCreateInfo scalar_ci(shapes, 1, DataType::FLOAT32);

// Submit with initial value and keep the returned tensor
scalar_ci.set_initial_value(float_to_u64(77.0f));
L0TaskArgs args;
args.add_output(scalar_ci);
TaskOutputTensors outs = rt_submit_aiv_task(FUNC_NOOP, args);
const Tensor& scalar_tensor = outs.get_ref(0);

// Orchestration-side blocking read (waits for kernel completion)
uint32_t idx[1] = {0};
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
| 3 | Kernel read (INPUT) | Orch Write | WAR | none (consumer-drain dropped) | **Unprotected** |
| 4 | Kernel read-write (INOUT) | Orch Read | RAW | spin-wait producer COMPLETED | Yes |
| 5 | Kernel read-write (INOUT) | Orch Write | WAW | spin-wait producer COMPLETED (WAR part unprotected) | WAW only |
| 6 | Orch Write | Kernel read (INPUT) | RAW | blocking completes before next submit | Yes |
| 7 | Orch Write | Kernel write (OUTPUT) | WAW | same — serial guarantee | Yes |
| 8 | Orch Read | Kernel write (OUTPUT) | WAR | same — serial guarantee | Yes |
| 9–12 | Orch ↔ Orch | — | — | same-thread serial execution | Yes |

### Key Design Points

**WAR hazards (#3, #5) are no longer guarded at runtime**:

`set_tensor_data` only waits for the producer to finish writing (WAW). It
does **not** wait for downstream kernel consumers to finish reading, because
the single-shot replay model dropped the `CONSUMED` lifecycle and the
`fanout_refcount` consumer counter that the old WAR drain relied on. If a
kernel may still be reading a tensor when `set_tensor_data` overwrites it,
that is an unprotected WAR race. No workload currently uses `set_tensor_data`,
so this is dormant; if one is added that needs WAR safety, the consumer-drain
would have to be reintroduced. INOUT vs INPUT no longer changes this — both
hit the same dropped drain.

**Scenarios #6–8 serial guarantee**:

get/set_tensor_data are blocking calls, and orchestration is single-threaded serial submission. After a blocking operation completes, subsequent code (including task submissions) executes strictly afterward.

## 7. External Tensor Behavior

`make_tensor_external()` creates tensors with a pre-set `buffer.addr` (pointing to host-allocated device memory).

| Scenario | Behavior |
| -------- | -------- |
| External tensor never submitted as OUTPUT/INOUT | No TensorMap entry — get/set execute immediately |
| External tensor previously submitted as OUTPUT/INOUT | TensorMap has producer entry — get/set spin-wait producer COMPLETED |
| External tensor submitted as INPUT, then set_tensor_data | **WAR risk** — unprotected (consumer-drain dropped; same as scenario #3) |

**Key rule**: `set_tensor_data` only protects against the producer write
(WAW), not against in-flight kernel reads (WAR). The single-shot replay model
has no consumer-drain, so neither `add_input()` nor `add_inout()` makes a
later `set_tensor_data` WAR-safe.
