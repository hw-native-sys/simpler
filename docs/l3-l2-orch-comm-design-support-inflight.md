# L3-L2 In-Flight Control-Service Communication Design

This document designs a first-version communication mechanism between one
L3 Host Orchestrator and one L2 Device AICPU Orchestrator.

Implementation guide:
[l3-l2-orch-comm-impl.md](l3-l2-orch-comm-impl.md).

The target is in-flight interaction: after the L2 AICPU orchestration task has
launched, the L3 orchestration function can keep sending tensor payloads and
receiving outputs. The design therefore requires a control path that can make
progress while the task-dispatch mailbox is occupied by a running L2 task.

The first version keeps the bottom layer small:

- one L3 Orch and one L2 Orch;
- one or more GM communication regions owned by one L3 orchestration
  execution;
- one input slice and one output slice per first-version wrapper channel;
- one outstanding transfer at a time;
- synchronous payload and signal operations;
- tensor payloads are contiguous;
- tensor shape and dtype are fixed by task arguments or local protocol.

The mechanism exposes symmetric vocabulary on both sides:

- `payload_read`
- `payload_write`
- `notify`
- `wait`

The vocabulary is symmetric. The implementation is not. L3 payload operations
are Host DRAM to Device GM DMA operations executed by the chip child runtime
helper. L2 payload operations are GM-view operations used by AICPU
orchestration code to build runtime tensor views.

## Goals

The first version should support a closed-loop example:

```text
L3 Orch writes input tensor payload.
L3 Orch notifies data-ready(seq).
L2 Orch waits for data-ready(seq).
L2 Orch obtains input and output GM payload views.
L2 Orch launches AICore work.
AICore reads input and writes output.
L2 Orch waits for AICore completion.
L2 Orch notifies consumed(seq).
L3 Orch waits for consumed(seq).
L3 Orch reads output tensor payload.
L3 Orch validates golden output.
```

This proves the path:

```text
L3 Orch -> L2 Orch -> AICore -> L2 Orch -> L3 Orch
```

The first version should provide:

1. GM region allocation and release.
2. Region descriptor exposure to L3 and L2.
3. L3 payload write/read through synchronous DMA.
4. L3 signal notify/wait through the runtime helper.
5. L2 payload read as GM tensor views.
6. L2 narrow payload write for metadata/status bytes.
7. L2 signal notify/wait in AICPU orchestration code.
8. A streaming wrapper example with `DATA` and `STOP` semantics.

The design should keep the bottom GM endpoint primitive separate from the
streaming wrapper. The bottom layer does not know about `DATA`, `STOP`, ring
slots, tensor schema, or stream protocol. The first example may reserve a small
payload header for those wrapper-level semantics, but that header is still
payload data and must be written through the same shared Host-buffer payload
path as tensor data.

## Platform Support

The first version targets:

- `a2a3sim`: full support.
- `a5sim`: full support.
- `a2a3` onboard: full support.
- `a5` onboard: API stubs only, returning a clear not-supported error.

Simulation backends must preserve the same API, ordering, timeout, and poison
semantics as onboard. The physical implementation may be thread-based in sim,
but tests should still cover descriptor passing, control-path progress,
payload transfer, signal sequencing, and closed-loop behavior.

## Non-Goals

The first version does not cover:

- L3 Parent directly attaching to the NPU.
- Parent direct mapped-GM operations.
- Payload bytes copied through a mailbox or child DRAM staging buffer.
- Reusing the task dispatch mailbox for in-flight communication.
- Multiple L3 Orchestrators or multiple L2 Orchestrators.
- Multiple outstanding transfers.
- Multi-worker routing from one L3 orchestration execution.
- Multi-slot rings, credits, or queue-depth tuning.
- Async DMA tokens or overlap scheduling.
- Cancellation or protocol recovery after timeout.
- Dynamic tensor dtype/shape per round.
- Non-contiguous tensor views.
- Temporary Host tensor staging fallback.
- Encoding tile layout or AICore-specific alignment in the region descriptor.
- High-performance AICPU large tensor copy as the main output path.

## Why A Separate Control Path Is Required

There are two separate reasons not to use the existing task dispatch mailbox.

First, payload bytes must not flow through a mailbox. The target data path is:

```text
Parent shared Host tensor DRAM -> Device GM
```

The control path should carry only descriptors such as host pointer, GM offset,
size, sequence number, and operation kind. If tensor bytes were copied into a
mailbox or into child private DRAM first, the data path would become:

```text
Parent DRAM -> mailbox or child DRAM -> Device GM
```

That adds an unwanted Host-side copy and does not scale for large tensor
payloads.

Second, the existing task dispatch mailbox is busy while the L2 task is
running:

```text
Parent WorkerThread:
  write TASK_READY
  wait for TASK_DONE

chip child main loop:
  observe TASK_READY
  call ChipWorker.run(...)
  do not poll the dispatch mailbox again until run returns
  write TASK_DONE
```

If L3 Orch tried to send an in-flight `PAYLOAD_WRITE` control request over the
same mailbox, no child thread would service it while `ChipWorker.run()` is
active. A deadlock is then possible:

```text
L3 Orch waits for PAYLOAD_WRITE completion.
chip child main loop is blocked inside ChipWorker.run().
L2 Orch waits for the payload signal.
WorkerThread waits for TASK_DONE.
```

Therefore this design requires an independent control path. It may be
implemented as a second mailbox-like request/response queue, but it must be
independent from task dispatch and must be serviced by a runtime helper that
continues running while the L2 task is in flight.

## Architecture

The design has three planes:

```text
L3 Parent process
  L3 Orch Python API
  region handles and protocol state
  shared Host tensors

Parent -> child control path
  descriptor-only request/response queue
  no tensor payload bytes

chip child process
  ChipWorker / DeviceRunner
  runtime helper for DMA and Host-side signal operations
  device context and stream ownership

L2 AICPU orchestration
  descriptor consumption
  payload GM view construction
  signal wait/notify
  AICore task launch
```

The Parent does not directly touch Device GM. All L3-side Device GM
operations, including payload DMA and signal load/store, are executed by the
child runtime helper.

### Control Path Commands

The first version needs only synchronous commands:

```text
L3_L2_ORCH_COMM::ALLOC_REGION(payload_bytes)
L3_L2_ORCH_COMM::FREE_REGION(region_id)
L3_L2_ORCH_COMM::PAYLOAD_WRITE(region_id, offset, host_ptr, nbytes)
L3_L2_ORCH_COMM::PAYLOAD_READ(region_id, offset, host_ptr, nbytes)
L3_L2_ORCH_COMM::SIGNAL_NOTIFY(region_id, signal_slot, seq)
L3_L2_ORCH_COMM::SIGNAL_WAIT(region_id, signal_slot, seq, timeout)
```

`L3_L2_ORCH_COMM` is a dedicated command namespace for this interface group.
It distinguishes the in-flight L3-L2 orchestration control service from
existing task-dispatch and legacy communication interfaces; the operation kind
is interpreted only inside that namespace.

`ALLOC_REGION` returns the region descriptor. There is no separate
`GET_DESCRIPTOR` or `QUERY_REGION` command in the first version. The L3 region
handle caches the descriptor and can encode it into `TaskArgs` scalars for the
L2 orchestration task.

The control path carries only request descriptors and completion status.
Payload bytes never appear in the control path, including small wrapper
metadata such as `DATA` or `STOP` headers.

The first version serializes requests per chip worker:

```text
one request in flight per worker control path
request -> helper executes -> response
```

This matches synchronous API semantics and avoids request ids, out-of-order
responses, and multi-producer contention. A pending L3 `SIGNAL_WAIT` occupies
the per-chip control service until it succeeds or times out. Later commands for
any region on the same chip wait behind it.

### Control Service Bootstrap

The independent control service starts lazily on the first
`orch.create_l3_l2_region(worker_id, payload_bytes)` for the target chip
worker. That call blocks until the service is ready and `ALLOC_REGION`
completes. If bootstrap fails, no Host region handle is returned.

The existing task dispatch mailbox may be used only for this bootstrap step,
and only before the first L2 run that needs the L3-L2 control service. After
bootstrap, all region allocation, payload, and signal commands must use the
independent control service.

If the service has not been bootstrapped and the target worker is already
inside an L2 run, `create_l3_l2_region` fails with a clear bootstrap-ordering
error instead of waiting for the task mailbox to become idle. Once the service
is running, later `ALLOC_REGION` commands may execute while an L2 task is in
flight. A newly allocated descriptor is not visible to an already-running L2
task unless that task has an explicit wrapper protocol to receive it.

The service is a per-chip lazy singleton. Once started, it lives until the chip
child exits or the owning Worker closes; freeing a region does not stop the
service.

### Runtime Helper

The DMA and L3-side signal executor should live inside the host runtime /
`DeviceRunner` side of the chip child, not as a Python-level operation in the
chip child main loop.

This keeps device context and stream ownership local to the runtime component
that already owns allocation, copy, launch, and finalization. It also allows the
helper to service control requests while the chip child main loop is blocked in
`ChipWorker.run()`.

The helper should own the service thread, request queue, and control stream.
Payload and signal commands are synchronous at the L3 API boundary: the command
returns only after the control stream work is complete, synchronized, or has
failed. The first version does not expose async tokens or overlap scheduling.

The helper is responsible for:

- GM region allocation and release.
- Synchronous H2D and D2H DMA for payload operations.
- Host-side signal notify and wait against GM signal slots.
- Bounds checks for region offsets and sizes.
- Timeout handling for signal waits.
- Returning errors to the L3 API through the control response.

## Region Model

A communication region is an owned Device GM allocation managed by the child
runtime. User-visible semantics treat it as one region. The implementation may
use one GM allocation or multiple aligned allocations.

Region lifetime is bound to one L3 orchestration execution. A region may be
reused across multiple L2 orchestration runs submitted by that L3 orchestration
execution, but the first version does not promise reuse across multiple outer
L3 `Worker.run()` calls. Region handles from an L3 orchestration execution are
invalid after that orchestration function returns.

One L3 orchestration execution may own multiple live regions. These regions are
for independent logical channels or independent payload slices. They are not a
concurrency optimization: the per-chip control service still accepts one
request in flight, and each region protocol still has one outstanding transfer.

The descriptor contains only byte-oriented communication fields:

```text
magic/version
region_id
payload_base
payload_bytes
l3_to_l2_signal_base
l2_to_l3_signal_base
```

The descriptor is encoded as six `uint64_t` scalar slots in `TaskArgs`:

```text
scalar[i + 0] = magic_version
scalar[i + 1] = region_id
scalar[i + 2] = payload_base
scalar[i + 3] = payload_bytes
scalar[i + 4] = l3_to_l2_signal_base
scalar[i + 5] = l2_to_l3_signal_base
```

`region_id` is unique within one chip worker control-service lifetime and is
not reused until that chip child exits. L2 uses it only for diagnostics and
structured endpoint errors; it is not a device-side lookup key. L2 validates
descriptor ABI, payload bounds, signal base alignment, and signal protocol, but
descriptor liveness is a Host-side contract.

The descriptor does not contain:

- dtype
- shape
- stride
- tensor rank
- tile layout
- stream header layout
- ring layout

`payload_bytes` covers only the payload range:

```text
payload_base .. payload_base + payload_bytes - 1
```

The two directional signal slots are independent GM ranges and are not included
in payload bounds. `payload_read` and `payload_write` operate only on the
payload range and must not access signal slots.

Each signal slot is one 64-byte cache line. Signal values are little-endian
`uint64_t` sequence numbers. The initial value is zero. Valid sequence numbers
start at one and increase monotonically for the region lifetime. Signal slots
are initialized only at region allocation; L2 run boundaries do not reset them.

The first version has two directional signal slots:

```text
l3_to_l2_signal_base  # L3 notify, L2 wait
l2_to_l3_signal_base  # L2 notify, L3 wait
```

There is no separate stop signal slot. Stop is a wrapper-level opcode carried
in payload metadata and published through the normal L3-to-L2 signal.

### Region Release

`region.free()` releases the Host handle but does not immediately send
`FREE_REGION` to the control service. Physical GM release is deferred until the
owning L3 orchestration execution has drained all submitted L2 runs that could
hold the descriptor. At the end of an L3 orchestration execution, the runtime
marks all still-live regions released, drains submitted L2 work, and then
physically frees released or poisoned regions.

This delayed release applies equally to poisoned regions. Poisoning is not
device-side cancellation, signal reset, or recovery.

## Endpoint Semantics

### L3 Python Endpoint

Illustrative L3 API:

```python
region = orch.create_l3_l2_region(worker_id=0, payload_bytes=n)

region.payload_write(offset, host_tensor, nbytes=None)
region.payload_read(offset, host_tensor, nbytes=None)
region.notify(seq)
region.wait(seq, timeout=...)
descriptor = region.descriptor_scalars()
region.free()
```

Names are illustrative. The contract is what matters.

`create_l3_l2_region` is an L3 Orchestrator operation. The first call for a
chip worker lazily bootstraps the independent control service and allocates the
region. The call must happen before submitting the first L2 run that needs the
service.

`payload_write` publishes a contiguous Host tensor into the GM payload region.
It sends a descriptor command to the child runtime helper. The helper performs
synchronous H2D DMA from the shared Host tensor storage to Device GM. The call
returns only after DMA completion or error.

`payload_read` copies bytes from the GM payload region into a contiguous Host
tensor. The helper performs synchronous D2H DMA. The call returns only after
the destination tensor is readable by the L3 Parent.

`notify` stores the sequence value into the L3-to-L2 signal slot after the
relevant payload write has completed.

`wait` waits for the L2-to-L3 signal slot to equal the requested sequence. It
must have a timeout. Timeout poisons the region on the Host side.

After `free`, poison, or L3 orchestration function return, the Host handle
rejects payload operations, signal operations, and descriptor extraction.

### Host Buffer Requirements

The Host tensor storage used by L3 payload APIs must be visible to the child
process so the child helper can DMA directly from Parent DRAM to Device GM.

The first version should require pre-visible shared Host buffers, for example:

- `torch.Tensor.share_memory_()` before worker children are forked; or
- a runtime-provided shared Host tensor allocator.

Ordinary Parent-private tensors, temporary Python bytes, and post-fork buffers
that the child has not mapped are not valid zero-copy payload sources in the
first version. Supporting them would require a Parent-to-child staging copy,
which is explicitly outside the main path.

This applies to small wrapper metadata as well as tensor payloads. A wrapper
header such as `{seq, DATA}` or `{seq, STOP}` must live in a child-visible
shared Host buffer before `payload_write` copies it to the region. Inline
metadata bytes in the control request are future work.

### L2 C++ Endpoint

Illustrative L2 API:

```cpp
AicpuEndpoint ep(region_desc);

PayloadView input_view{};
PayloadView output_view{};
bool input_ok = ep.payload_read(input_offset, input_nbytes, &input_view);
bool output_ok = ep.payload_read(output_offset, output_nbytes, &output_view);
ep.payload_write(metadata_offset, &metadata, sizeof(metadata));
ep.notify(seq);
bool wait_ok = ep.wait(seq, timeout);
```

`PayloadView` is byte-oriented and carries only a GM address and byte length.
It does not carry dtype, shape, stride, or tensor role:

```cpp
struct PayloadView {
  uint64_t gm_addr;
  uint64_t nbytes;
};
```

`payload_read` validates bounds and writes a GM payload view into its output
parameter. The returned view can be combined with dtype and shape metadata
supplied by task args or local protocol conventions to build runtime tensor
views such as
`ContinuousTensor(child_memory=1)`. It does not copy data.

For large tensor output, L2 should obtain the output GM view through
`payload_read(output_offset, output_nbytes, &output_view)` and pass that view to
AICore or the runtime. AICore writes the output directly into the communication
region.

`payload_write` has a narrow first-version contract: it is byte-oriented and
intended for small metadata, status, actual output size, error code, test
patterns, or future fallback implementations. It is not the primary large
tensor output path.

`wait` waits on the L3-to-L2 signal slot. `notify` publishes on the L2-to-L3
signal slot after relevant payload writes and barriers.

L2 endpoint errors must carry structured metadata, at minimum:

```text
kind
op
region_id
seq
message
```

L3 uses `region_id` to poison only the corresponding Host region. Human-readable
error strings are diagnostics, not the attribution mechanism.

## Synchronization Semantics

The first version supports one outstanding transfer. L3 must not overwrite the
input slice for sequence `seq + 1` until L2 has notified completion for
sequence `seq`.

Single-round ordering:

```text
L3:
  payload_write(input_offset, host_input)
  notify(seq)

L2:
  wait(seq)
  payload_read(input_offset, input_nbytes, &input_view)
  payload_read(output_offset, output_nbytes, &output_view)
  submit AICore(input_view, output_view)
  wait for AICore completion
  notify(seq)

L3:
  wait(seq)
  payload_read(output_offset, host_output)
```

`notify(seq)` publishes all previous writes relevant to `seq`. `wait(seq)` is
the acquire point before reading data for that sequence.

More precisely, L3 `notify(seq)` publishes all previously completed
`payload_write` calls on the same region. It does not publish writes on other
regions, and it does not publish writes issued after the notify.

Cache and barrier responsibilities:

- L3 `payload_write` completes H2D DMA before L3 `notify`.
- After L2 `wait` succeeds, any AICPU-side read of Host-written payload bytes
  through the endpoint must invalidate the target payload range before loading.
- When AICore reads Host-written payload, the AICore kernel or runtime tensor
  cache-policy layer must ensure AICore-side visibility before loading.
- AICore output writes must be pushed to GM before L2 observes completion and
  publishes `notify(seq)`.
- L3 `payload_read` runs only after L3 `wait(seq)` succeeds.

Signal access should use plain load/store semantics. Atomic read-modify-write
operations on GM signal slots, such as CAS or fetch-add, are not part of the
first version.

The signal comparison is strict:

```text
notify(seq): store exactly seq
wait(seq):
  current == seq -> success
  current < seq  -> keep waiting until timeout
  current > seq  -> protocol error
```

`current > seq` poisons the Host-side region for L3 waits. On L2, it returns an
endpoint protocol error carrying `region_id`.

## Streaming Wrapper Example

The bottom layer does not define stream headers. The first example may build a
small wrapper protocol by reserving payload bytes:

```text
payload offset 0..63:  channel header
input_offset:          input tensor payload
output_offset:         output tensor payload
```

Example header:

```cpp
struct ChannelHeader {
  uint64_t seq;
  uint32_t opcode;    // DATA or STOP
  uint32_t reserved;
};
```

The fixed tensor schema is supplied through task args:

```text
input_offset
output_offset
shape
dtype
nbytes
```

Runtime flow:

```text
setup:
  L3 has pre-visible shared Host input/output tensors.
  L3 has pre-visible shared Host header storage.
  L3 creates one GM communication region inside the L3 orchestration execution.
  L3 passes descriptor scalars and tensor schema to L2.
  L3 submits one persistent L2 orchestration task.

for each seq:
  L3 writes input tensor into input slice.
  L3 writes wrapper header {seq, DATA} from shared Host header storage.
  L3 notifies seq.

  L2 waits seq.
  L2 reads wrapper header.
  L2 obtains input and output GM tensor views.
  L2 submits AICore work.
  AICore writes output view.
  L2 waits for AICore completion.
  L2 notifies seq.

  L3 waits seq.
  L3 reads output tensor from output slice.
  L3 checks golden.

teardown:
  L3 writes wrapper header {seq + 1, STOP} from shared Host header storage.
  L3 notifies seq + 1.
  L2 observes STOP and returns.
  Worker.run drains.
  L3 orchestration cleanup physically frees the region.
```

This wrapper has independent stop semantics without adding a third signal slot.
It still has only one outstanding transfer and does not implement a ring. STOP
does not require an L2-to-L3 signal acknowledgement. L2 observing STOP and
returning from the orchestration task is the acknowledgement; L3 observes it
through `Worker.run` drain/completion.

The same region may also be reused across multiple L2 orchestration runs
submitted by the same L3 orchestration execution, as long as only one transfer
is outstanding and physical free is delayed until all submitted L2 runs drain.
The streaming example uses one persistent L2 run because it proves in-flight
progress while the L2 task is actively waiting.

## Error Handling

All waits must accept a timeout. Defaults may be generous, but unbounded waits
must not hide protocol deadlocks.

Timeout behavior:

- L3 wait timeout raises a Python exception.
- L2 wait timeout returns an orchestration/runtime error.
- Timeout does not reset signal state.
- Timeout does not free the region.
- The Host-side region becomes poisoned.

A poisoned region rejects future L3 `payload_write`, `payload_read`, `notify`,
`wait`, and descriptor extraction operations. Only explicit free or L3
orchestration cleanup remains valid. Physical free remains delayed until
post-drain cleanup.

Failures that may corrupt protocol progress poison the corresponding region:

- L3 or L2 signal wait timeout;
- DMA failure after a payload command has been issued;
- signal notify failure;
- control service fatal error after the region is live;
- signal protocol error such as observing `current > seq`.

Pure pre-command validation failures do not poison:

- malformed API arguments;
- non-contiguous Host tensor detected before a command is issued;
- child-invisible Host buffer detected before a command is issued;
- out-of-bounds payload offset detected before a command is issued;
- descriptor extraction after release or poison.

When multiple live regions exist, L2 endpoint errors poison only the region
named by structured `region_id` metadata. Non-endpoint L2 run failures do not
automatically poison every live region.

The bottom layer should fail loudly on:

- out-of-bounds payload offsets;
- malformed region descriptors;
- non-shared or child-invisible Host buffers when detectable;
- use-after-free region handles;
- use of a poisoned region;
- DMA failure;
- signal wait timeout;
- L2 descriptor ABI mismatch;
- L2 endpoint error metadata without a usable `region_id`.

The first version does not recover protocol state after poison. Higher-level
stream wrappers may recreate the region and retry, but that is outside this
bottom-layer design.

## Testing Strategy

Testing should cover protocol units before the full closed-loop example.

### Platform Coverage

Tests should run on:

- `a2a3sim`
- `a5sim`
- `a2a3` onboard hardware where available

For `a5` onboard, tests should verify stub behavior:

- symbols are exported;
- region creation returns a clear not-supported error;
- failed stubs do not leave partial Host-side state.

### Unit Tests

1. Region descriptor validation:
   - descriptor magic/version;
   - payload bounds;
   - signal slot bounds;
   - use-after-free rejection.

2. Control request validation:
   - encode/decode each command kind;
   - reject malformed sizes and offsets;
   - ensure payload bytes are not embedded in control requests.

3. Signal helpers:
   - signal slots initialize to zero;
   - `notify(seq)` stores exactly the requested sequence;
   - stale values do not satisfy later waits;
   - wait timeout poisons the Host-side region.

4. Host buffer validation:
   - shared Host tensors are accepted;
   - child-invisible buffers fail when detectable;
   - non-contiguous tensors are rejected.

5. Region lifecycle:
   - explicit `free` marks the handle released;
   - forgotten free is cleaned up at L3 orchestration execution end;
   - released and poisoned regions are physically freed only after drain;
   - handles are invalid after the orchestration function returns.

### System Tests

1. Lazy bootstrap:
   - first `create_l3_l2_region` starts the independent control service;
   - bootstrap before the first L2 run succeeds;
   - bootstrap while the target worker is already busy fails clearly.

2. In-flight control path progress:
   - launch a persistent L2 orchestration task;
   - while it is running, perform L3 `payload_write`, `notify`, `wait`, and
     `payload_read`;
   - verify the request is serviced without using the task dispatch mailbox.

3. Closed-loop direct-output example:
   - fixed `float32` contiguous tensors;
   - one region;
   - one input slice and one output slice;
   - multiple sequence rounds;
   - AICore writes output directly into the output slice;
   - L3 reads output and checks golden for every round.

4. Stop semantics:
   - L3 sends `{seq, STOP}` in the wrapper header;
   - L2 exits cleanly;
   - `Worker.run` drains and returns.

5. Multi-region isolation:
   - create two regions in one L3 orchestration execution;
   - transfer distinct payloads and sequence values;
   - force an endpoint error carrying one `region_id`;
   - verify only the matching region is poisoned.

6. Poison behavior:
   - force an L3 wait timeout;
   - verify later payload and signal operations fail;
   - verify region free or L3 orchestration cleanup still succeeds.

7. Platform stub behavior:
   - run on `a5` onboard;
   - verify clear not-supported errors for region operations.

The main example should be readable rather than throughput-oriented. Performance
tests for async DMA, ring depth, or overlap belong to future work.
