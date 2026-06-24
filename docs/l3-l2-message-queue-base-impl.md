# L3-L2 Message Queue Base Queue Two-PR Implementation Plan

## 1. Scope And Platform Support

This document covers a two-PR delivery of the base bidirectional SPSC message
queue transport described in `l3-l2-message-queue-design.md`.

PR1 implements the core queue transport and primitive-compatible fast-path API:

- one input queue from L3 to L2;
- one output queue from L2 to L3;
- descriptor rings and payload arenas in one primitive L3-L2 region;
- `DATA`, `ERROR`, and input-only `STOP` descriptors;
- explicit output reserve/publish on L2;
- explicit input peek/release on L2;
- L3 enqueue, output ownership/dequeue, stop, and cleanup APIs;
- non-zero L3 buffers limited to primitive-compatible registered
  `orch.alloc(...)` host Tensors;
- two single-writer abort flags for timeout disambiguation;
- unit tests for ABI, layout, counters, zero-byte descriptors, queue
  mechanics, and fast-path APIs.

PR2 implements the usability and end-to-end layer:

- lazy internal staging for ordinary L3 host buffers;
- ordinary host-buffer enqueue and output read convenience paths;
- one base queue example with a small message-local AICore task.
- scene tests on supported platforms;
- final user-facing documentation cleanup.

Neither PR includes:

- the L2 input window helper;
- multiple active DATA input handles on L2;
- out-of-order input release;
- fragmented payload arenas;
- multiple outstanding producer reservations per direction;
- output-side STOP acknowledgement messages.

Supported across the two PRs:

- `a2a3` onboard;
- `a2a3sim`;
- `a5sim`.

Not supported:

- `a5` onboard.

The exact Python and C++ class names may change during implementation, but the
ABI, state transitions, and observable behavior in this document are base queue
requirements. Scope tags below identify whether a requirement lands in PR1 or
PR2.

## 2. Expected User Flow

The final base queue should be usable without exposing descriptor offsets,
counter offsets, or payload arena cursors to application code. PR1 supports
the same operation shape with primitive-compatible registered host Tensors for
non-zero L3 buffers. PR2 relaxes that buffer requirement with lazy staging.

Expected L3 shape:

```python
queue = orch.create_l3_l2_queue(
    worker_id=0,
    depth=8,
    input_arena_bytes=1 << 20,
    output_arena_bytes=1 << 20,
)

for payload in input_payloads:
    queue.input.enqueue(payload.buffer, nbytes=payload.nbytes, timeout=timeout_s)

queue.input.enqueue(None, nbytes=0, timeout=timeout_s)  # zero-byte DATA
queue.request_stop(timeout=timeout_s)

while not application_done:
    message = queue.output.peek(timeout=timeout_s)
    output_buffer = choose_buffer(message.payload_nbytes)
    queue.output.read_into(message, output_buffer)
    queue.output.release(message)
    handle_application_output(message)

queue.free()
```

If the application already owns a large enough output buffer, it may use the
convenience path instead:

```python
message = queue.output.dequeue_into(max_sized_output_buffer, timeout=timeout_s)
```

Expected base L2 shape:

```cpp
L3L2QueueEndpoint queue(desc_scalars, queue_args);
for (;;) {
    auto in = queue.input().peek(timeout);
    if (in.opcode == L3L2QueueOpcode::STOP) {
        queue.input().release(in);
        break;
    }

    auto out = queue.output().reserve(output_nbytes, timeout);
    launch_message_local_aicore_work(in.payload_view, out.gm_addr);
    wait_until_output_bytes_are_visible();
    queue.output().publish(out, L3L2QueueOpcode::DATA);
    queue.input().release(in);
}
```

Application payload schema, request IDs, final-output markers, and output
cardinality are application responsibilities. PR1 transport order does not
imply request correlation beyond FIFO order within each queue direction.

## 3. API Surface

PR1 must expose the semantic operations below. PR2 keeps the same operation
surface and only expands accepted L3 buffer types through lazy staging. Exact
class and method names may change during implementation, but the
implementation must not require users to manipulate descriptor slots, counter
offsets, payload arena offsets, or head/tail reconstruction state directly.

Required L3 Python surface:

```text
orch.create_l3_l2_queue(
    worker_id,
    depth,
    input_arena_bytes,
    output_arena_bytes,
) -> queue

queue.input.enqueue(buffer_or_none, nbytes, timeout)
queue.input.try_enqueue(buffer_or_none, nbytes)

queue.output.dequeue_into(buffer, timeout) -> message
queue.output.try_dequeue_into(buffer) -> message or no-progress

queue.request_stop(timeout)
queue.try_request_stop()
queue.free()
```

L3 message results must expose at least:

```text
seq
opcode
payload_nbytes
```

Convenience dequeue APIs may copy and release in one operation. PR1 must also
expose explicit output ownership APIs with these semantics:

```text
queue.output.peek(timeout) -> message_handle
queue.output.try_peek() -> message_handle or no-progress
queue.output.read_into(message_handle, buffer)
queue.output.release(message_handle)
```

Required L2 C++ surface:

```text
L3L2QueueEndpoint queue(desc_scalars, queue_args)

queue.input().peek(timeout) -> input_handle
queue.input().try_peek() -> input_handle or no-progress
queue.input().release(input_handle)

queue.output().reserve(nbytes, timeout) -> output_reservation
queue.output().try_reserve(nbytes) -> output_reservation or no-progress
queue.output().publish(output_reservation, opcode)
```

L2 input handles must expose at least:

```text
seq
opcode
payload_nbytes
payload_view or empty payload marker
```

L2 output reservations must expose at least:

```text
seq or publish sequence context
payload_offset
payload_nbytes
gm_addr for non-zero payload writes
```

The API must preserve these user-visible semantics:

- finite timeouts are required for blocking operations;
- `try_*` operations return no-progress without mutating shared state when the
  queue cannot make progress;
- ordinary timeout does not poison the queue unless peer abort is observed;
- zero-byte messages may pass `buffer_or_none == None`;
- PR1 non-zero L3 buffers must be primitive-compatible registered
  `orch.alloc(...)` host Tensors;
- PR2 L3 convenience APIs accept ordinary contiguous host byte spans and lazily
  stage them when they are not primitive-compatible registered tensors;
- primitive-compatible `orch.alloc(...)` host Tensors remain the fast path in
  both PRs;
- output ownership APIs are the recommended path for variable-size outputs,
  while `dequeue_into` remains valid when the caller supplies a large enough
  target buffer;
- after successful `request_stop`, L3 input enqueue rejects later input
  messages locally without poisoning;
- `ERROR` is an application-level message, not a transport exception;
- cleanup/free remains valid after local poison or remote-aborted terminal
  state.

## 4. L3 Host Buffer Contract And Lazy Staging

The primitive L3 payload APIs require a registered, child-visible
`orch.alloc(...)` host Tensor.

PR1 buffer contract:

- `nbytes == 0` accepts `buffer_or_none == None` and uses the zero-byte
  descriptor path;
- non-zero L3 input enqueue buffers must be primitive-compatible registered
  `orch.alloc(...)` host Tensors;
- non-zero L3 output read targets must be primitive-compatible registered
  `orch.alloc(...)` host Tensors;
- ordinary `bytes`, `bytearray`, `memoryview`, private tensors, and other
  non-registered host buffers are rejected before shared-state mutation;
- rejecting a non-registered buffer is a pre-mutation validation failure and
  does not poison or set an abort flag.

PR2 buffer contract:

- `nbytes == 0` accepts `buffer_or_none == None` and uses the zero-byte
  descriptor path;
- if the input buffer is a primitive-compatible registered `orch.alloc(...)`
  host Tensor, enqueue uses it directly as the zero-extra-host-copy fast path;
- otherwise enqueue accepts an ordinary readable contiguous host byte span,
  such as `bytes`, `bytearray`, `memoryview`, or a contiguous CPU tensor-like
  object the implementation can view as bytes;
- non-fast-path enqueue copies the user bytes into an internal registered
  staging Tensor, then issues primitive `payload_write` from that staging
  Tensor.

For L3 output read:

- if the output target is a primitive-compatible registered `orch.alloc(...)`
  host Tensor, `read_into` or `dequeue_into` uses it directly as the fast path;
- otherwise the target must be an ordinary writable contiguous host byte span;
- non-fast-path read first issues primitive `payload_read` into an internal
  registered staging Tensor, then copies from staging into the user target.

The staging Tensor is allocated lazily and owned by the queue handle. It may
grow when a later operation needs a larger staging span. The implementation
must not expose staging offsets or staging Tensor ownership to users.

If a payload is too large for the current staging Tensor, the queue should grow
or allocate staging before issuing any primitive command. Failure to allocate
staging is a pre-mutation validation/allocation failure: it rejects the
operation, does not publish descriptors, does not release descriptors, does not
poison, and does not set an abort flag.

Staging may add one host-to-host copy. Users that need the lowest host overhead
can pass primitive-compatible registered `orch.alloc(...)` host Tensors.

## 5. PR1 ABI Surface

The stable PR1 ABI is the L3/L2 shared contract. It is separate from exact
Python or C++ method names.

TaskArgs carry the primitive region descriptor followed by queue parameters:

```text
primitive desc[0..5]
queue_magic_version
depth
input_arena_bytes
output_arena_bytes
```

The queue ABI version covers:

- descriptor slot size and field order;
- opcode numeric values;
- deterministic payload layout derivation;
- counter offsets and meanings;
- head/tail low32 reconstruction rules;
- abort flag semantics;
- zero-byte descriptor canonical form;
- STOP and ERROR transport semantics.

Descriptor slot ABI:

```cpp
struct L3L2QueueDescSlot {
    uint64_t seq;
    uint64_t opcode;
    uint64_t payload_offset;
    uint64_t payload_nbytes;
};
static_assert(sizeof(L3L2QueueDescSlot) == 32);
```

Opcode ABI:

```text
0      invalid / never published
DATA   = 1
STOP   = 2
ERROR  = 3
```

Counter ABI:

```text
offset 0:   input_desc_tail       writer=L3
offset 64:  input_desc_head       writer=L2
offset 128: output_desc_tail      writer=L2
offset 192: output_desc_head      writer=L3
offset 256: l3_abort_flag         writer=L3
offset 320: l2_abort_flag         writer=L2
```

Layout validation ABI:

- `depth` must be a power of two and `depth <= 2^30`;
- queue capacity is `depth`, not `depth - 1`;
- descriptor slot size is 32 bytes;
- descriptor rings are 8-byte aligned;
- payload arena bases are 64-byte aligned;
- arena byte sizes are positive 64-byte multiples;
- `counter_bytes >= 384`.

The following are not PR1 ABI:

- exact Python class names;
- exact C++ helper class names;
- internal helper function names;
- polling backoff strategy;
- application payload schema;
- example payload format.

## 6. ABI And Layout

The descriptor slot ABI is the existing 32-byte format:

```cpp
struct L3L2QueueDescSlot {
    uint64_t seq;
    uint64_t opcode;
    uint64_t payload_offset;
    uint64_t payload_nbytes;
};
static_assert(sizeof(L3L2QueueDescSlot) == 32);
```

`payload_offset` is relative to the primitive payload base. For non-zero
message payloads, it points into the direction-local payload arena. It does not
point to the descriptor slot itself.

The layout helper must derive all payload and counter offsets. Python may
mirror the calculation, but tests must keep the Python calculation and the C/C++
helper in lockstep.

PR1 counter layout:

```text
offset 0:   input_desc_tail       writer=L3
offset 64:  input_desc_head       writer=L2
offset 128: output_desc_tail      writer=L2
offset 192: output_desc_head      writer=L3
offset 256: l3_abort_flag         writer=L3
offset 320: l2_abort_flag         writer=L2
```

`counter_bytes` must be at least 384. The abort flags are low-frequency
diagnostic signals, but they still use the same 64-byte stride as the
descriptor counters to preserve single-writer cache-line ownership.

All six counters are initialized to zero before submitting the persistent L2
run. Descriptor slots and payload bytes do not need to be zeroed for
correctness.

## 7. Primitive Command Mapping

The queue is a wrapper over the existing L3-L2 primitive commands. PR1 must not
add a new primitive command or bypass the primitive region lifetime model.

Descriptor rings live in the primitive payload region. Descriptor slot access
therefore uses the primitive payload APIs:

- L3 writes input descriptor slots with `L3L2OrchRegion.payload_write`;
- L3 reads output descriptor slots with `L3L2OrchRegion.payload_read`;
- L2 reads input descriptor slots with `L3L2OrchEndpoint::payload_read`;
- L2 writes output descriptor slots with `L3L2OrchEndpoint::payload_write`.

Message payload arena access also uses the primitive payload APIs when the
message payload is non-zero:

- L3 input enqueue writes non-zero input payload bytes with
  `L3L2OrchRegion.payload_write`;
- L3 output dequeue reads non-zero output payload bytes with
  `L3L2OrchRegion.payload_read`;
- L2 input consume obtains a non-zero input payload GM view with
  `L3L2OrchEndpoint::payload_read`;
- L2 output reserve returns a GM span in the output arena; L2 application code
  or AICore work writes that span before `publish`;
- PR1 does not require a separate L2 message-payload copy API. If an
  implementation uses `L3L2OrchEndpoint::payload_write` for a small L2-produced
  output payload, it is only a helper for filling the reserved output arena
  span before `publish`, not a separate transport path.

Queue counters use the primitive signal APIs:

- publishing descriptor tail, releasing descriptor head, and setting an abort
  flag use `SIGNAL_NOTIFY` / `signal_notify`;
- head/tail polling uses `SIGNAL_TEST` / `signal_test` snapshots;
- timeout disambiguation samples the peer abort flag with `SIGNAL_TEST`, for
  example `GE 1` against the peer flag address.

Only a matched `SIGNAL_TEST` snapshot may drive head/tail reconstruction,
descriptor replay, payload release, or payload reuse. A failed head/tail test
does not establish acquire ordering and its observed value must not update
local queue state. For abort flags, a matched `GE 1` test reports remote abort;
an unmatched test leaves the timeout as ordinary no-progress.

PR1 queue correctness must not depend on primitive `SIGNAL_WAIT`. Blocking
queue operations are wrapper-level bounded polling loops over `SIGNAL_TEST`
plus local queue-state checks.

## 8. Zero-Byte Message Rules

Zero-byte `DATA`, `ERROR`, and `STOP` descriptors are valid queue messages.
They still consume one descriptor slot and follow the normal descriptor
publication sequence.

For any descriptor with `payload_nbytes == 0`:

- `payload_offset` must be `0`;
- `payload_offset == 0` is a canonical sentinel, not a payload address;
- the message consumes no payload arena bytes;
- producer payload cursors do not advance;
- consumer payload cursors do not advance;
- payload wrap-padding replay is skipped for that descriptor;
- no message-payload arena copy/read/view is issued.

Descriptor-ring access is separate from message-payload arena access.
Descriptor slots live in the primitive payload region, so publishing or reading
a zero-byte message may still use primitive payload access for descriptor-ring
metadata. The rule above skips only the message payload arena path.

Consumer validation order must make the zero-byte path explicit:

```text
1. validate descriptor sequence;
2. validate opcode and direction legality;
3. if payload_nbytes == 0:
     require payload_offset == 0;
     skip direction-local arena range checks and payload replay;
   else:
     require payload_offset to be inside the direction-local arena;
     validate contiguous span and payload cursor replay.
```

This ordering matters because `payload_offset == 0` for a zero-byte output
descriptor usually is not inside the output arena. A consumer that runs arena
range validation before the zero-byte branch would reject a valid descriptor.

If a published descriptor has `payload_nbytes == 0` and `payload_offset != 0`,
the descriptor is invalid published state. The observing endpoint transitions
to `POISONED(local-infrastructure)` and sets its own abort flag.

## 9. Queue State And Abort Flags

PR1 uses two single-writer abort flags:

```text
l3_abort_flag: writer=L3, reader=L2
l2_abort_flag: writer=L2, reader=L3
```

Each flag is initialized to `0`. On local infrastructure poison, the endpoint
sets its owned flag to `1` with `NotifyOp.Set`. The flag never resets within a
queue lifetime. It is a terminal boolean, not an epoch and not a poison count.

Abort flags are for timeout disambiguation. PR1 does not require every wait
loop iteration to poll both data progress and abort progress. A blocking queue
operation that reaches its timeout samples the peer abort flag:

```text
peer abort_flag == 0:
  return ordinary timeout/no-progress;
  keep the local queue live;
  do not set the local abort flag.

peer abort_flag == 1:
  return remote-aborted transport failure;
  transition the local handle to a terminal remote-aborted state;
  do not publish descriptors or advance queue state;
  do not set the local abort flag solely because the peer flag was observed.
```

The implementation may represent terminal remote abort with the existing
`POISONED` state, but the reason must remain distinct:

```text
POISONED(local-infrastructure): set own abort_flag = 1
POISONED(remote-aborted):       do not set own abort_flag
```

This distinction prevents a peer abort observation from being amplified into a
new local infrastructure poison report.

## 10. Capacity, Counters, And Reconstruction

`depth` is the user-visible queue capacity. A queue created with `depth=N` can
hold `N` published, unreleased descriptors.

Validation rules:

- `depth` must be a power of two;
- `depth <= 2^30`;
- queue capacity is `depth`, not `depth - 1`.

Full and empty checks must use monotonic local `uint64_t` head/tail values, not
only masked ring indices:

```text
empty iff tail == head
full  iff tail - head == depth
invalid shared state iff tail - head > depth
```

The shared head/tail counters store only the low 32 bits. Each endpoint keeps
local `uint64_t` copies and reconstructs observed progress with signed 32-bit
delta semantics:

```text
delta = int32_t(observed_low32 - local_low32)
valid progress: 0 <= delta <= depth
```

`delta == depth` is valid. A peer may legally move from empty to full between
observations. Negative deltas or deltas larger than `depth` are inconsistent
shared state and poison the observing endpoint.

Descriptor slot validity does not depend on opcode or slot clearing. A
published descriptor is valid only when:

```text
slot.seq == expected_seq
expected_seq == local_head_or_tail + 1
slot_index == (expected_seq - 1) & (depth - 1)
```

Equivalent index calculations are allowed, but the sequence check must use the
full 64-bit `seq`. Descriptor slots do not need to be cleared before reuse.

Before a producer reuses released descriptor slots or payload arena bytes, it
must replay exactly the released FIFO prefix after observing head progress.
Replay must happen before slot reuse. Zero-byte descriptors in replay advance
descriptor state only and do not advance payload cursors.

## 11. Producer And Consumer Operation Details

Producer sequence:

```text
reserve -> fill/copy payload if payload_nbytes > 0 -> publish descriptor
```

Consumer sequence:

```text
peek/acquire descriptor -> read/view payload if payload_nbytes > 0
-> release descriptor and payload
```

Descriptor publication order:

1. reserve a descriptor slot and, for non-zero payloads, a contiguous payload
   arena span;
2. write or expose the payload bytes;
3. write descriptor fields other than `seq`;
4. write `seq` as the descriptor validity marker;
5. release-publish the tail counter.

Descriptor release order:

1. finish all uses of the message payload;
2. update local release and payload cursor state;
3. release-publish the head counter.

Each direction allows at most one outstanding producer reservation. Publishing
an unknown, stale, already-published, already-canceled, or cross-queue
reservation is a local ownership contradiction and poisons the queue.

The base queue has no reservation cancel. If a producer has successfully
reserved a non-zero payload span and later cannot safely publish either `DATA`
or application `ERROR`, it must poison the queue. If the queue remains
trustworthy, the application may publish an `ERROR` descriptor using the
reservation.

`STOP` is an input-queue descriptor. It consumes one input descriptor slot,
uses `payload_nbytes == 0` and `payload_offset == 0`, and is terminal for L3
input enqueue. After L3 successfully publishes `STOP`, later input `DATA`,
`ERROR`, or `STOP` attempts are rejected locally without poisoning. If L2 has
observed `STOP` and later observes another published input descriptor, the
descriptor is invalid published state and poisons the queue.

`ERROR` remains an application-level message. Receiving `ERROR` does not poison
the queue, set an abort flag, stop either direction, or imply transport abort.

## 12. Error Handling Rules

The guiding rule remains:

```text
Before shared-state mutation: reject, no poison, no abort flag.
After shared-state mutation or inconsistent shared-state observation:
  poison local infrastructure, set own abort_flag.
```

Pre-mutation validation failures do not poison and do not set abort flags:

- `try_enqueue` sees no descriptor or payload space;
- `try_request_stop` sees no input descriptor slot;
- a blocking operation times out under ordinary backpressure;
- payload size exceeds the arena before reservation mutates state;
- queue creation rejects invalid layout or reconstruction parameters;
- output buffer is too small before payload copy and before release;
- invalid API arguments are caught before shared state is touched;
- lazy staging allocation failure before primitive command issue;
- enqueue is attempted after L3 has already published `STOP`;
- application `ERROR` is sent or received normally.

Infrastructure poison sets the endpoint's own abort flag:

- descriptor sequence mismatch;
- invalid opcode observed in a published descriptor;
- `STOP` observed on the output queue;
- zero-byte descriptor with non-zero `payload_offset`;
- non-zero descriptor payload range outside its direction-local arena;
- head/tail reconstruction observes impossible progress;
- payload replay observes impossible state;
- payload copy failure after command issue;
- counter notify failure;
- control-service response timeout after command issue;
- L2 endpoint fatal error for this region;
- reservation, publish, or release ownership state becomes contradictory.

Ordinary timeout is ambiguous until the peer abort flag is sampled. A timeout
with peer abort flag `0` is not poison. A timeout with peer abort flag `1`
transitions the local handle to terminal `remote-aborted` without setting the
local abort flag.

Cleanup and `free()` remain valid and idempotent after both local
infrastructure poison and remote-aborted terminal state.

## 13. Example

PR2 adds one base queue example:

```text
examples/a2a3/tensormap_and_ringbuffer/l3_l2_message_queue/
```

The example should demonstrate the intended user shape, not every edge case.
It must show:

- L3 creating a queue with `depth > 1`;
- multiple variable-size input `DATA` messages;
- one zero-byte `DATA` message;
- a persistent L2 loop;
- L2 processing at most one active DATA input at a time;
- one small message-local AICore task;
- L2 publishing one output `DATA` per input `DATA`;
- L3 publishing `STOP`;
- L3 continuing to dequeue outputs after `STOP` according to application final
  output rules;
- L2 releasing the `STOP` descriptor and returning from the persistent run.

The example should not demonstrate:

- the L2 input window;
- multiple active input messages;
- one input producing multiple outputs;
- multiple inputs producing one output;
- out-of-input-order output publish;
- application `ERROR` protocol design;
- abort flag failure paths.

The zero-byte `DATA` message should exercise the descriptor-only message path.
It should not require a child-visible zero-byte host buffer.

## 14. Test Plan

Both PRs require automated tests for their review-driven boundaries. A manual
review checklist is not enough.

PR1 test scope:

- ABI and layout;
- descriptor/counter protocol;
- zero-byte descriptor handling;
- capacity, full/empty, wrap, and low32 reconstruction;
- abort flag semantics;
- L2 endpoint API;
- L3 fast-path API with primitive-compatible registered host Tensors.

PR2 test scope:

- lazy internal staging for ordinary L3 host buffers;
- registered Tensor fast path remains no-staging;
- staging allocation failure is pre-mutation and non-poisoning;
- base queue example and scene coverage.

Suggested C++ unit test category:

```text
tests/ut/cpp/common/test_l3_l2_message_queue.cpp
```

Suggested C++ unit tests:

- `LayoutAssignsAbortFlagsAfterDescriptorCounters`
- `LayoutRequiresCounterBytesForSixCounters`
- `DescriptorSlotEncodingIsStable`
- `ZeroByteDescriptorUsesCanonicalOffset`
- `ZeroByteDescriptorWithNonZeroOffsetPoisons`
- `CapacityEqualsDepthAllowsNPublishedDescriptors`
- `CapacityEqualsDepthRejectsNthPlusOneDescriptor`
- `FullAndEmptyUseMonotonicCountersNotMaskedIndices`
- `Low32ReconstructionAcceptsDeltaEqualDepth`
- `Low32ReconstructionHandlesCounterWrap`
- `Low32ReconstructionRejectsNegativeDelta`
- `Low32ReconstructionRejectsDeltaGreaterThanDepth`
- `ReplaySkipsPayloadCursorAdvanceForZeroByteDescriptors`
- `ReplayBeforeSlotReuseAfterFullQueueWrap`
- `LocalInfrastructurePoisonSetsOwnAbortFlag`
- `RemoteAbortObservationDoesNotSetOwnAbortFlag`
- `OrdinaryTimeoutDoesNotSetAbortFlag`
- `ApplicationErrorDoesNotSetAbortFlag`
- `PreMutationValidationFailureDoesNotSetAbortFlag`

Suggested Python unit test category:

```text
tests/ut/py/test_l3_l2_message_queue.py
```

Suggested Python unit tests:

- `test_layout_matches_cpp_helper`
- `test_counter_offsets_include_abort_flags`
- `test_zero_byte_enqueue_skips_payload_arena_copy`
- `test_zero_byte_dequeue_skips_payload_arena_read`
- `test_enqueue_rejects_ordinary_host_bytes_before_pr2_staging`
- `test_output_read_rejects_ordinary_buffer_before_pr2_staging`
- `test_enqueue_accepts_ordinary_host_bytes_with_lazy_staging`
- `test_enqueue_registered_tensor_uses_fast_path_without_staging`
- `test_output_read_into_ordinary_buffer_uses_lazy_staging`
- `test_staging_allocation_failure_does_not_poison`
- `test_timeout_with_peer_abort_flag_reports_remote_aborted`
- `test_timeout_without_peer_abort_flag_returns_timeout`
- `test_remote_aborted_terminal_state_rejects_later_operations`

Suggested scene/example tests:

```text
examples/a2a3/tensormap_and_ringbuffer/l3_l2_message_queue/
```

Suggested scene cases:

- `variable_size_messages`: enqueue/dequeue several non-zero `DATA` messages;
- `zero_byte_data`: send one zero-byte `DATA` and verify one corresponding
  output is produced without payload arena bytes;
- `depth_capacity`: with `depth=N`, publish `N` inputs before backpressure;
- `fifo_stop`: publish `STOP`, drain outputs, and verify L2 exits;
- `small_aicore_work`: each non-zero input launches message-local AICore work;
- `l2_abort_flag_timeout_disambiguation`: force an L2 local infrastructure
  poison, then verify L3 timeout reports remote-aborted instead of ordinary
  timeout.

The scene test matrix should include the PR1 supported simulation platforms
where practical:

- `a2a3sim`;
- `a5sim`.

Hardware execution should include `a2a3` onboard when device access is
available through the repository's `task-submit` workflow.
