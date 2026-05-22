# L3/L2 Host-Device Communication

A second L3/L2 communication model is added beside the existing bounded
message channel. The runtime now exposes both **send/recv** for small control
messages and **shared-memory read/write + notify/wait** for data regions that
need explicit producer-consumer synchronization.

In this document, L3 means the host-side Worker / Orchestrator runtime that
issues control commands. L2 means the device-side execution boundary reached
through the chip child process and platform runtime. Today the L2-facing
implementation covers the AICPU broker / sim test paths, while the memory
layout is kept plain POD so future device-side participants can use the same
region format.

Current L3 shared-memory access is intentionally mailbox-mediated. The L3
parent does not mmap the chip child's host mapping and does not access the
device pointer directly. L3 `shared_memory_read` and `shared_memory_write`
are chunked mailbox RPC copy helpers over the child-owned region; they are
not a device-direct zero-copy data plane.

For the WorkerThread mailbox used to reach chip children, see
[worker-manager.md](worker-manager.md). For where the Orchestrator fits
in the hierarchical runtime, see
[hierarchical_level_runtime.md](hierarchical_level_runtime.md). For the
platform/runtime split, see [chip-level-arch.md](chip-level-arch.md).

---

## 1. Communication model

The PR deliberately keeps three semantics separate:

```text
Control:  send(route, payload, correlation_id) / recv()
Data:     read(offset, nbytes) / write(offset, bytes)
Sync:     notify(signal_id, value) / wait(signal_id, target)
```

### Responsibilities

- **Message channel**: small commands, events, and completion notifications.
  Messages carry `route`, `correlation_id`, and a byte payload.
- **Shared memory**: a CPU/NPU-visible data region addressed by offset.
  Reads and writes copy bytes only; they do not carry route metadata.
- **Signal slots**: software synchronization words associated with a shared
  memory region. Signals publish phase or sequence progress.

This split avoids forcing bulk data through the message path and avoids
hiding synchronization inside raw memory copies. Callers choose the primitive
that matches their protocol.

---

## 2. Message channel

The message channel is implemented by
`src/common/worker/host_device_channel.{h,cpp}` and exported through
`pto_runtime_c_api.h`.

```cpp
typedef struct {
    uint32_t lane_count_cpu_to_l2;
    uint32_t lane_count_l2_to_cpu;
    uint32_t lane_depth;
    uint32_t max_message_bytes;
    uint32_t flags;
} HostDeviceChannelConfig;
```

Each channel owns a shared control region:

```text
HostDeviceChannelHeader
  +-- cpu_to_l2 lanes[0..N-1]
  |     +-- HostDeviceLaneHeader
  |     +-- HostDeviceDesc[lane_depth]
  +-- l2_to_cpu lanes[0..M-1]
        +-- HostDeviceLaneHeader
        +-- HostDeviceDesc[lane_depth]
```

Each lane is a bounded SPSC ring. The CPU send path writes into
`cpu_to_l2` lanes; the CPU receive path consumes from `l2_to_cpu` lanes.
The L2 test helpers use the opposite direction so unit tests can exercise
both endpoints without real device code.

### Channel constraints

- `lane_depth` must be a non-zero power of two.
- `max_message_bytes` must be non-zero and no larger than
  `HDCH_MAX_INLINE_BYTES` (`256`).
- `timeout_us == 0` means non-blocking probe. If the selected operation
  cannot complete immediately, it returns a would-block error.
- Payloads are inline. This channel is not a streaming data plane.

---

## 3. Shared memory

The shared-memory model is implemented by
`src/common/worker/host_device_memory.{h,cpp}` and exported through
`pto_runtime_c_api.h`.

```cpp
typedef struct {
    uint64_t data_bytes;
    uint32_t signal_count;
    uint32_t flags;
} HostDeviceMemoryConfig;

typedef struct {
    uint64_t host_ptr;
    uint64_t device_ptr;
    uint64_t data_bytes;
    uint32_t signal_count;
    uint32_t flags;
} HostDeviceMemoryInfo;
```

The region starts with metadata, followed by cache-line-sized signal slots,
followed by the data region:

```text
offset 0:
  HostDeviceMemoryHeader
    magic
    version
    flags
    signal_count
    data_offset
    data_bytes
    total_bytes
    fatal_status

offset sizeof(HostDeviceMemoryHeader):
  HostDeviceSignalSlot[signal_count]   // each slot is 64 B

offset data_offset:
  data[data_bytes]                     // read/write offset 0 starts here
```

`HostDeviceMemoryInfo::host_ptr` and `device_ptr` both point at the data
region, not the header. All public `shared_memory_read` and
`shared_memory_write` offsets are relative to the data region.

### Memory constraints

- `data_bytes` must be non-zero.
- `signal_count` must be non-zero.
- The computed layout is 64-byte aligned and checked for integer overflow.
- Bounds checks use `offset <= data_bytes` and
  `nbytes <= data_bytes - offset`, so zero-length operations at the end of
  the region are valid.

---

## 4. Synchronization contract

Shared-memory read/write does not provide an implicit producer-consumer
protocol. The caller owns phase ordering.

### CPU produces, L2 consumes

```text
L3 CPU:
  shared_memory_write(mem, offset, bytes)
  shared_memory_notify(mem, signal_id, seq)

L2:
  wait(signal_id, seq)
  read/process data region
```

### L2 produces, CPU consumes

```text
L2:
  write/process data region
  notify(signal_id, seq)

L3 CPU:
  shared_memory_wait(mem, signal_id, seq, timeout_us)
  shared_memory_read(mem, offset, nbytes)
```

`notify` stores the signal value with release semantics. `wait` polls the
signal with acquire semantics until the observed value is greater than or
equal to the target. Signal values are intended to be monotonic sequence or
phase numbers; the implementation does not enforce monotonicity.

Use separate `signal_id` values for independent directions or pipeline
stages. V1 does not provide an MPMC queue or ownership protocol on top of
the data region.

---

## 5. Runtime and platform path

The public ABI lives in `src/common/worker/pto_runtime_c_api.h`. The relevant
entry points are:

```cpp
HostDeviceChannelHandle open_host_device_channel_ctx(
    DeviceContextHandle ctx, const HostDeviceChannelConfig *cfg);
int host_device_send_ctx(...);
int host_device_recv_ctx(...);

HostDeviceMemoryHandle open_host_device_memory_ctx(
    DeviceContextHandle ctx, const HostDeviceMemoryConfig *cfg);
int host_device_memory_info_ctx(...);
int host_device_memory_read_ctx(...);
int host_device_memory_write_ctx(...);
int host_device_memory_notify_ctx(...);
int host_device_memory_wait_ctx(...);
```

`ChipWorker` resolves these symbols unconditionally via `dlsym`. Runtime
variants that do not support the feature must export stubs rather than omit
symbols, so the worker has one stable ABI surface.

### Platform behavior

- `a2a3/sim`: uses a 64-byte-aligned host buffer; host and device pointers
  are the same simulated address.
- `a5/sim`: uses the same simulated allocation model as `a2a3/sim`.
- `a2a3/onboard`: allocates device-visible memory, registers the host mapping
  with `halHostRegister`, and initializes the common region layout.
- `a5/onboard`: exports unsupported stubs for this API surface.

The common `host_device_*` helpers own layout validation, bounds checks, ring
operations, read/write copies, and signal load/store semantics. Platform code
is responsible for allocation and mapping.

---

## 6. L3 control path

When Python code calls the API on a direct `ChipWorker`, the call reaches the
runtime C ABI in the same child process and `host_ptr` is a valid address in
that process.

When Python code calls the API through an L3 `Orchestrator`, the L3 parent
does **not** own the chip child's host mapping. The Orchestrator forwards the
operation through the target `WorkerThread` mailbox:

```text
Python Orchestrator
  |
  v
src/common/hierarchical/orchestrator.cpp
  |
  v
WorkerManager::get_worker(NEXT_LEVEL, worker_id)
  |
  v
WorkerThread::control_*()
  |
  v  CONTROL_REQUEST mailbox command
python/simpler/worker.py::_chip_process_loop
  |
  v
ChipWorker::{open_channel, shared_memory_write, ...}
  |
  v
runtime C ABI
```

`WorkerThread::mailbox_mu_` serializes task dispatch and control commands for
the same child mailbox. Different WorkerThreads still operate independently.

### L3 metadata semantics

`shared_memory_info` returns different `host_ptr` semantics depending on the
caller:

- Direct `ChipWorker`: `host_ptr` is the current process's data-region
  address.
- L3 `Orchestrator`: `host_ptr` is always `0`; callers must use
  `shared_memory_read` and `shared_memory_write`.

L3 reads and writes are mailbox RPC copies. Large transfers are split into
chunks no larger than `CTRL_PAYLOAD_CAPACITY`; the API materializes the full
`bytes` result for reads. This is a convenient control-plane data copy path,
not a zero-copy parent mapping and not a streaming transport.

The mailbox protocol is intentionally kept stable while chunking is handled
above it. A logical L3 read/write may issue multiple
`CTRL_SHARED_MEMORY_READ` or `CTRL_SHARED_MEMORY_WRITE` child requests, but
each individual mailbox payload still fits inside `CTRL_PAYLOAD_CAPACITY`.

---

## 7. Python API

Direct chip-worker API:

```python
ch = chip.open_channel(cpu_to_l2_lanes=1, l2_to_cpu_lanes=1)
chip.channel_send(ch, route=7, data=b"cmd", correlation_id=1)
payload, route, cid = chip.channel_recv(ch, capacity=256, timeout_us=1000)

mem = chip.open_shared_memory(data_bytes=4096, signal_count=2)
host_ptr, device_ptr, data_bytes, signal_count, flags = (
    chip.shared_memory_info(mem)
)
chip.shared_memory_write(mem, 0, b"payload")
chip.shared_memory_notify(mem, 0, 1)
chip.shared_memory_wait(mem, 1, 1, timeout_us=1000)
out = chip.shared_memory_read(mem, 0, 7)
```

L3 Orchestrator API adds `worker_id` and forwards through the selected
next-level worker:

```python
mem = orch.open_shared_memory(worker_id=0, data_bytes=4096, signal_count=2)
host_ptr, device_ptr, _, _, _ = orch.shared_memory_info(0, mem)
assert host_ptr == 0

orch.shared_memory_write(0, mem, 0, b"payload")
orch.shared_memory_notify(0, mem, 0, 1)
orch.shared_memory_wait(0, mem, 1, 1, timeout_us=1000)
out = orch.shared_memory_read(0, mem, 0, 7)
```

The `*_l2_for_test` methods exist only to simulate the L2 endpoint in tests.
They are not the production CPU-facing protocol.

---

## 8. Test coverage

Focused smoke tests cover both the protocol boundary and the supported
hardware path:

- `tests/ut/py/test_worker/test_host_device_comm_sim.py`
  - Runs on `a2a3sim` and `a5sim`.
  - Uses a real L3 `Worker` and child mailbox path.
  - Covers `open_channel`, `channel_send`, `channel_recv`, L3
    `open_shared_memory`, `shared_memory_info`, chunked
    `shared_memory_write/read`, `shared_memory_notify`, and
    `shared_memory_wait`.
- `tests/ut/py/test_worker/test_host_device_comm_hw.py`
  - Runs on `a2a3` onboard hardware.
  - Covers CPU/L2 channel traffic in both directions, chunked shared-memory
    round trip larger than `CTRL_PAYLOAD_CAPACITY`, notify/wait, and
    diagnostic failures for invalid config or oversized channel payloads.

These tests prove the current parent, nanobind/C++ binding, child handler,
and runtime backend agree on the mailbox command, offset, payload, and
metadata semantics. They do not claim device-direct zero-copy correctness;
that is outside the current L3 API contract.

---

## 9. Why this layering

### 9.1 Why keep send/recv and shared memory separate?

Small messages need routing metadata, correlation ids, and bounded queue
backpressure. Bulk data exchange needs an addressable buffer and explicit
phase synchronization. Combining both into one primitive would either make
messages carry unnecessary memory protocol state or make data exchange look
like a stream of oversized control packets.

### 9.2 Why explicit notify/wait?

Read/write only copies bytes. Different callers need different phase
protocols: one-shot handoff, ping-pong buffers, or multi-stage pipelines.
Explicit signal slots make the synchronization boundary visible and keep the
memory primitive from inventing hidden fences or ownership rules.

### 9.3 Why no direct L3 parent pointer?

The shared-memory region is opened by the chip child that owns the runtime
context. In PROCESS mode, the L3 parent talks to that child through a
pre-existing mailbox and does not have a valid mapping to dereference.
Returning `host_ptr = 0` prevents accidental parent-side pointer use and
forces L3 callers onto mailbox-mediated read/write operations.

### 9.4 Why unsupported stubs?

`ChipWorker` resolves the runtime ABI as a stable set of symbols. Exporting
unsupported stubs keeps loading deterministic on platforms that have not
implemented a backend yet, while still producing a clear runtime error if the
API is used.

---

## 10. Related

- [worker-manager.md](worker-manager.md) - WorkerThread mailbox,
  control commands, and child-process dispatch
- [hierarchical_level_runtime.md](hierarchical_level_runtime.md) -
  where L3/L2 workers fit in the level model
- [task-flow.md](task-flow.md) - task data movement through the
  hierarchical runtime
- `src/common/worker/host_device_channel.{h,cpp}` - message channel layout
  and SPSC ring implementation
- `src/common/worker/host_device_memory.{h,cpp}` - shared-memory layout,
  bounds checks, and signal semantics
