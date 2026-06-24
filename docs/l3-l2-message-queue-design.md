# L3-L2 SPSC Message Queue Design

## 1. Goal

This document proposes the functional shape of an L3-L2 SPSC message queue
wrapper built on top of the existing `docs/l3-l2-orch-comm.md` primitives.

The feature goal is to let one L3 orchestrator exchange a sequence of input
and output messages with one persistent L2 orchestrator run. L3 can enqueue
task inputs and dequeue task outputs while the L2 run stays alive. This avoids
stopping the L2 run after every task and then paying host/device finish and
init costs again for the next task.

The target shape has two layers:

- a base bidirectional queue transport with input and output queues;
- an L2-side input window helper that lets L2 hold multiple input messages
  concurrently without changing the L3 API or the transport ABI.

The base transport should land first for reviewability. The input window can
then be added as an L2 helper policy on top of the same descriptor ABI, region
layout, counter layout, and L3 queue API.

The queue wrapper does not change the primitive L3-L2 communication service.
It uses the existing region descriptor, payload byte range, and `int32_t`
signal counter primitives.

## 2. Existing Primitive Constraints

The primitive L3-L2 communication layer provides:

- one region descriptor containing payload and counter base/size fields;
- contiguous payload byte access through `PAYLOAD_READ` and `PAYLOAD_WRITE`;
- address-based `int32_t` signal counters through `SIGNAL_NOTIFY`,
  `SIGNAL_TEST`, and `SIGNAL_WAIT`;
- region lifetime, release, and poison state handling.

The primitive layer deliberately does not define queue layout, stream headers,
opcodes, tensor schema, descriptor rings, STOP semantics, or typed tensor
metadata. The message queue wrapper owns those protocol choices.

The primitive layer requires only 4-byte alignment for counter addresses inside
the registered counter range. The queue wrapper places high-frequency shared
counter signals at 64-byte strides so counters written by different agents do
not share a cache line.

## 3. Public Functional Shape

L3 creates one bidirectional queue object:

```python
queue = orch.create_l3_l2_queue(
    worker_id=0,
    depth=8,
    input_arena_bytes=1 << 20,
    output_arena_bytes=1 << 20,
)
```

The L3-visible queue API exposes an input queue and an output queue. L3 sends
ordinary application messages to L2 through the input queue and receives
ordinary application messages from L2 through the output queue.

The wrapper computes:

- descriptor ring sizes;
- payload section offsets;
- counter offsets;
- total region payload bytes;
- total counter bytes.

The user does not pass internal descriptor offsets, arena offsets, or counter
offsets.

The queue owns one `L3L2OrchRegion`. The L2 task receives the primitive region
descriptor plus queue layout scalars through `TaskArgs`.

The intended L3 API shape is illustrative, but the semantics are part of the
transport contract:

```python
queue.input.enqueue(host_buffer, nbytes=None, timeout=timeout_s)
message = queue.output.dequeue_into(host_buffer, timeout=timeout_s)
handle = queue.output.peek(timeout=timeout_s)
queue.output.read_into(handle, host_buffer)
queue.output.release(handle)
queue.request_stop(timeout=timeout_s)
queue.free()
```

The output ownership APIs `peek`, `read_into`, and `release` are part of the
base L3 API. They are the recommended path for variable-size outputs because
the caller can inspect `payload_nbytes` before choosing or allocating a target
buffer. Convenience APIs such as `dequeue_into` may copy and release in one
operation when the caller already has a large enough target buffer. Core APIs
that hand ownership to the caller require explicit release.

`queue.free()` releases the L3 queue handle. It rejects later queue operations,
but it does not synchronously free device memory. Physical cleanup follows the
underlying region lifetime model.

The L3 public queue API accepts ordinary contiguous host byte spans for
convenience enqueue and output read operations. When the supplied buffer is
already a primitive-compatible registered `orch.alloc(...)` Tensor, the queue
uses it as the zero-extra-host-copy fast path. Otherwise the queue lazily
stages through an internal registered host Tensor before issuing the primitive
payload command, then copies between that staging Tensor and the user buffer.
Zero-byte DATA and ERROR messages may pass `None` as the buffer. Staging hides
the primitive child-visible Tensor requirement from ordinary queue users, but
may add one host-to-host copy.

The L2 input window extension is not visible to L3. It is an L2 helper policy
that controls how many DATA input messages L2 may hold concurrently before
releasing them in FIFO-safe order.

## 4. Non-Goals

- Multiple L2 orchestrators.
- Multi-producer or multi-consumer queues.
- Shared input/output payload allocator.
- Split payload spans across arena wrap.
- Dtype, shape, stride, tensor rank, or tile layout interpretation.
- Changes to `ALLOC_REGION`, `PAYLOAD_READ`, `PAYLOAD_WRITE`,
  `SIGNAL_NOTIFY`, `SIGNAL_TEST`, or `SIGNAL_WAIT`.
- Exposing the L2 input window configuration through the L3 API.
- Out-of-order input payload release.
- Fragmented or hole-filled input arena allocators.
- Output-side STOP acknowledgement messages.

## 5. Region Layout

The physical L3-L2 region has one payload range and one counter range. The
queue wrapper divides the payload range into four logical sections:

```text
payload region
├─ input descriptor ring
├─ output descriptor ring
├─ input payload arena
└─ output payload arena
```

The descriptor rings live in the payload region because they are structured
byte metadata. The counter range stores only shared head/tail signals.

The input and output payload arenas are logically separate. This preserves SPSC
ownership:

```text
input arena:
  producer = L3
  consumer = L2

output arena:
  producer = L2
  consumer = L3
```

A shared payload allocator is intentionally out of scope because it would have
two producers and two releasers.

The queue layout is derived, not transmitted as internal offsets. `TaskArgs`
carry the primitive region descriptor followed by four queue parameters:

```text
primitive desc[0..5]
queue_magic_version
depth
input_arena_bytes
output_arena_bytes
```

The queue magic/version belongs to the queue wrapper ABI, not to the primitive
region ABI. It covers the descriptor slot format, opcode values, deterministic
layout function, head/tail reconstruction rules, and STOP/ERROR transport
semantics.

A shared C/C++ layout helper is the source of truth for derived offsets and
sizes. Python may mirror that calculation, but tests must keep the Python
calculation and the C/C++ helper in lockstep. The helper derives:

```text
input_desc_offset
output_desc_offset
input_arena_offset
output_arena_offset
input_desc_tail = 0
input_desc_head = 64
output_desc_tail = 128
output_desc_head = 192
l3_abort_flag = 256
l2_abort_flag = 320
```

Validation rules:

- `depth` must be a power of two and `depth <= 2^30`.
- Queue capacity is `depth` messages, not `depth - 1`.
- Descriptor slot size is fixed at 32 bytes.
- Descriptor rings are 8-byte aligned.
- Payload arena bases are 64-byte aligned.
- `input_arena_bytes` and `output_arena_bytes` must be positive 64-byte
  multiples. They do not need to be powers of two.
- `counter_bytes` must be at least 384.
- `payload_bytes` must contain both descriptor rings and both payload arenas.
- Unsupported `queue_magic_version` on L2 is a fatal queue decode error for
  this region.

The L3 queue creator initializes the four shared head/tail counters and the
two abort flags to zero before submitting the persistent L2 run. Descriptor
slots and payload bytes do not need to be zeroed for correctness.

## 6. Descriptor ABI

Each descriptor slot is 32 bytes and is encoded as four little-endian
`uint64_t` values:

```cpp
struct L3L2QueueDescSlot {
    uint64_t seq;
    uint64_t opcode;
    uint64_t payload_offset;
    uint64_t payload_nbytes;
};
static_assert(sizeof(L3L2QueueDescSlot) == 32);
```

The queue uses 64-byte spacing for shared signal counters, not for descriptor
slots. Each descriptor ring is SPSC, so the base descriptor ABI needs only the
four transport fields above.

`seq` is a full 64-bit infrastructure sequence number used for ring
correctness, wrap detection, diagnostics, and input-window validation. It is
not a user correlation ID. Applications that need request IDs, batch IDs,
partial/final markers, or other correlation should put them in their own
payload header.

`payload_offset` is relative to the primitive region payload base, so L2 can
call `endpoint.payload_read(payload_offset, payload_nbytes, &view)` directly.

Future descriptor extensions should use an ABI version or application payload
headers instead of reserving unused fields in every slot.

## 7. Opcodes

The queue transport defines these opcodes:

```text
0      invalid / never published
DATA   = 1 ordinary application payload message
STOP   = 2 graceful input-side shutdown request, input queue only
ERROR  = 3 ordinary application-level error payload message, either direction
```

`ERROR` is a normal queue message. The queue layer does not interpret its
payload, does not raise a transport exception for it, and does not poison the
queue when it sees one. Applications define whether an `ERROR` payload
correlates with a request, batch, stream, or other application state.

Infrastructure errors are handled through poison state, not by trying to write
an `ERROR` message into a potentially untrusted queue.

`STOP` is valid only on the input queue. The output queue has no STOP message.
L2 shutdown acknowledgement is provided by `Worker.run` drain, not by an
output STOP. Observing STOP on the output queue is invalid published
descriptor state and poisons the queue.

DATA and ERROR may carry zero payload bytes. For any zero-byte message,
`payload_offset` must be zero and the message consumes no payload arena bytes.
STOP must also use `payload_nbytes == 0` and `payload_offset == 0`.

## 8. Descriptor Counters And Derived Payload Cursors

The queue shares only descriptor head/tail values through the primitive layer's
`int32_t` signal counters. Each shared head/tail uses a 64-byte stride:

```text
offset 0:   input_desc_tail       writer=L3
offset 64:  input_desc_head       writer=L2
offset 128: output_desc_tail      writer=L2
offset 192: output_desc_head      writer=L3
offset 256: l3_abort_flag         writer=L3
offset 320: l2_abort_flag         writer=L2
```

`counter_bytes` must be at least 384.

The abort flags are single-writer terminal booleans used to disambiguate
operation timeouts from remote infrastructure abort. They are initialized to
zero and set to one with `NotifyOp.Set` when the owning endpoint enters local
infrastructure poison. They do not carry application `ERROR` semantics, do not
count poison events, and do not reset within a queue lifetime.

Blocking queue operations are not required to poll abort flags on every wait
iteration. When a blocking operation times out, the implementation samples the
peer abort flag. If the peer flag is zero, the timeout remains ordinary
no-progress and does not poison the local queue. If the peer flag is one, the
operation reports remote infrastructure abort and transitions the local handle
to a terminal remote-aborted state. Observing a peer abort flag does not set
the local endpoint's own abort flag.

The shared descriptor counters store the low 32 bits of logical `uint64_t`
head/tail values. These values are monotonic message counts. The primitive
transports these bits through `int32_t` counters. Endpoints reconstruct local
`uint64_t` head/tail values from sampled counter values using signed 32-bit
delta semantics:

```text
delta = int32_t(observed_low32 - local_low32)
valid progress: 0 <= delta <= depth
```

Negative deltas or deltas larger than `depth` are inconsistent shared state.
Queue creation rejects descriptor depths that would make head/tail
reconstruction ambiguous. This is a validation error, not a poison condition.

Descriptor head/tail reconstruction is safe because unobserved descriptor
progress is bounded by the descriptor ring depth. Payload byte cursors are not
shared counters and are not reconstructed from low-32-bit signal values.

Each endpoint maintains the payload cursors it needs as local `uint64_t`
state:

```text
producer local:
  payload_tail
  inferred_payload_head

consumer local:
  payload_head
```

The producer infers reusable payload space by observing `desc_head`
progress and replaying the released descriptors before reusing those descriptor
slots. The consumer maintains its local `payload_head` while releasing
descriptors.
Because payload cursor progress is derived from descriptor FIFO history, payload
arena size is not limited by 32-bit signal counter reconstruction.

Queue correctness is based on reconstructed descriptor head/tail state plus
descriptor replay, not on primitive `GE` / `LT` comparison over the 32-bit
counter value. Blocking queue operations use bounded polling over `SIGNAL_TEST`
snapshots plus local queue-state checks. The timeout belongs to the wrapper
operation. The design does not require primitive `SIGNAL_WAIT` for queue
correctness.

Local queue state may advance only after a matched `SIGNAL_TEST` snapshot. A
failed `SIGNAL_TEST` result does not establish acquire ordering, and its
`observed` value must not drive descriptor head/tail reconstruction, descriptor
replay, or payload release. Implementations should choose a comparison that
matches when the sampled counter has changed, such as `NE` against the local
low-32 value. The protocol does not prescribe a busy-poll, sleep, yield, or
backoff strategy.

If a live endpoint observes counter, head/tail, cursor, or descriptor state that
contradicts the descriptor reconstruction or payload replay rules, that is
inconsistent shared state and poisons the queue.

Descriptor slots carry the full 64-bit per-message `seq`, so message-level
validation does not depend on reconstructing sequence numbers from counters.
Input and output queues have independent sequence spaces. In each direction,
the first published message has `seq = 1`; head/tail counters start at zero and
store the number of messages published or released. A published slot has
`seq = tail_before_publish + 1`.

## 9. Payload Arena

Each direction has a variable-size SPSC byte arena.

Rules:

- `payload_tail` and `payload_head` are logical `uint64_t` byte cursors.
- Actual arena offset is `cursor % arena_bytes`.
- `arena_bytes` is limited by region allocation capacity, addressability, and
  runtime memory budget, not by 32-bit signal counter reconstruction.
- A single message payload must be one contiguous span.
- A single message payload must be `<= arena_bytes`.
- Split payloads across the arena wrap are not supported.
- If remaining bytes at the arena end cannot hold the next payload, the
  producer may insert invisible padding by advancing `payload_tail` to the next
  arena cycle.
- Padding has no descriptor. On release, the consumer compares
  `payload_head % arena_bytes` with the descriptor's arena-relative payload
  offset. If they differ, the only valid base-queue case is wrap padding: the
  descriptor offset is the base offset of this direction's arena and the
  releaser first advances `payload_head` to the next arena cycle. It then
  advances `payload_head` by `payload_nbytes`. Any other mismatch is
  inconsistent shared state and poisons the queue. The same replay rule is used
  by the producer after observing `desc_head` progress, before it reuses
  released descriptor slots.
- Zero-byte messages do not participate in wrap-padding checks and do not
  advance payload cursors.

Backpressure must check both descriptor slots and payload arena bytes. A free
descriptor slot is not enough if the payload arena lacks enough contiguous
space.

Payload validation is direction-local. DATA and ERROR payloads must lie wholly
inside the input arena for input descriptors, and wholly inside the output
arena for output descriptors. Being inside the primitive payload range is not
enough.

## 10. Core Operation Sequence

The queue exposes direction-specific operations. Exact class names may change,
but the operation set and ownership semantics are the transport contract.

L3 owns the input producer and output consumer operations:

```text
input.enqueue(buffer, nbytes, timeout)
input.try_enqueue(buffer, nbytes)
output.dequeue_into(buffer, timeout)
output.try_dequeue_into(buffer)
output.peek(timeout) -> message handle
output.try_peek() -> message handle or no-progress
output.read_into(handle, buffer)
output.release(handle)
request_stop(timeout)
try_request_stop()
free()
```

`dequeue_into` is the convenience path for full-message copy and release.
The `peek` / `read_into` / `release` path is the explicit-ownership path.
`free` releases the L3 queue handle, not the physical region.

L2 owns the input consumer and output producer operations:

```text
input.peek(timeout) -> input handle
input.try_peek() -> input handle or no-progress
input.release(handle)
output.reserve(nbytes, timeout) -> reservation
output.try_reserve(nbytes) -> reservation or no-progress
output.publish(reservation, opcode)
```

The L2 input window extension wraps the input consumer with additional
`complete(handle)` ownership; it does not change the base transport ABI. The
base queue has no output dequeue operation on L2 and no input enqueue operation
on L2.

The producer sequence is:

```text
reserve -> fill/copy payload -> publish descriptor
```

The consumer sequence is:

```text
peek/acquire descriptor -> read/view payload -> release descriptor and payload
```

Convenience APIs are built from the core operation sequence:

```text
enqueue      = reserve + copy + publish
dequeue_into = peek + read + release
```

L3 input enqueue can usually use the convenience path because the input payload
already exists in a host-visible buffer.

L2 output needs the core path because it often must reserve output arena space
before launching AICore work:

```cpp
auto out = output_queue.reserve(output_nbytes, timeout);
Tensor output = make_tensor_external(out.gm_addr, shape, rank, dtype);
// submit AICore work that writes output
// synchronize so output bytes are visible
output_queue.publish(out, L3L2QueueOpcode::DATA);
```

Each queue direction allows at most one outstanding producer reservation.
`publish` accepts only the current outstanding reservation for that direction.
Publishing an unknown, stale, already-published, or cross-queue reservation is
a local ownership contradiction and poisons the queue.

The base queue does not support reservation cancel. A successful reserve must
be published. If filling the reservation fails but the queue remains
trustworthy, the application may publish an ERROR message using that
reservation. If the reservation cannot be safely published, the producer
poisons the queue.

Descriptor publication is ordered. The producer writes payload bytes first,
writes descriptor fields, writes `seq` as the descriptor validity marker after
the other descriptor fields, and then release-publishes the tail counter. The
consumer acquire-observes tail progress before reading the slot, and
accepts the descriptor only when `slot.seq` equals the expected sequence.

Descriptor slots do not need to be cleared before reuse. Sequence validation
distinguishes old and new contents.

Descriptor release is ordered in the opposite direction. The consumer must
finish using the payload, update local release state, and release-publish the
head counter. The producer may replay released descriptors and infer reusable
payload space only after acquire-observing matched head progress.

All blocking operations require finite timeouts. Nonblocking `try_*` variants
return without changing shared state when no descriptor slot, message, or
payload space is available. Timeout under ordinary backpressure does not
poison the queue.

The queue layer returns transport messages to the application:

```text
seq
opcode
payload bytes or payload view
```

The queue layer does not infer application request correlation from queue order
or from transport `seq`.

Queue ownership is per message, not per byte range. Release or complete always
applies to the whole descriptor payload span.

For L3 convenience dequeue, a too-small output buffer is a local validation
failure. The descriptor remains at the queue head, no release is published, and
the caller may retry with a larger child-visible buffer.

## 11. Base L2 Processing Contract

After dequeuing one input message, L2 application code may submit any number
of message-local AICore tasks and use runtime dependencies, manual scopes,
async notify, or other L2 orchestration features.

The base helper and example do not overlap ownership of multiple input
messages. They keep at most one active DATA input message at a time:

```text
peek input
reserve output
submit message-local AICore work
wait or otherwise prove message-local work is safe
publish output
release input
next message
```

L2 must not release an input message until AICore no longer reads that input
payload and any corresponding output has been successfully published.

After an input is released, L2 and any in-flight AICore work must not read its
payload view again.

The queue layer does not understand dtype, shape, stride, or tensor schema. It
returns byte views. Applications build typed tensors with their own protocol
metadata.

## 12. L2 Input Window Extension

The target feature shape includes an L2 input window helper. The helper lets L2
hold multiple DATA input messages concurrently while preserving FIFO-safe input
release. It enables application-defined output cardinality and output order:

- one input may produce no output;
- one input may produce multiple outputs;
- several inputs may produce one output;
- status or progress outputs may be published independently;
- output publish order may differ from input acquire order.

The L3-visible queue API is unchanged by the input window extension. L3 still
observes an input queue and an output queue. L3 receives output messages in
publish order and does not infer input/output correlation from queue order.
Correlation, aggregation, partial/final markers, request IDs, and batch IDs
belong in the application payload header.

`max_l2_inflight` is a local L2 helper policy. It is not part of queue creation
and does not affect region layout:

```cpp
L3L2QueueEndpoint queue(desc, layout);
L3L2InputWindow input_window(
    queue.input(),
    L3L2InputWindowConfig{.max_l2_inflight = 4}
);
```

The helper tracks input handles with these states:

```text
ACQUIRED
  Descriptor has been read. Payload view is available to L2.

COMPLETED
  Application has declared the input payload is no longer needed.

RELEASED
  Helper has advanced the input descriptor and payload cursors past this input.
```

The state transition is:

```text
ACQUIRED -> COMPLETED -> RELEASED
```

The application owns the transition to `COMPLETED`; the helper owns the
transition to `RELEASED`. Completing an input means no future L2 code or
in-flight AICore task will read that input payload, and the payload is no
longer needed to construct future output.

Completion is explicit. The helper must not infer completion from C++ object
destruction or lexical scope exit. A handle that is completed twice, released
twice, or destroyed while still active is a local ownership error.

The helper releases inputs through a FIFO watermark. If inputs 10, 11, and 12
are acquired and inputs 10 and 12 are completed, the helper may release input
10 only. It must not release input 12 until input 11 is also completed. This
keeps the input payload arena monotonic and avoids holes.

Output publish remains application-driven and independent of input handles:

```cpp
auto out = queue.output().reserve(nbytes, timeout);
// fill output directly or submit AICore work that writes out.gm_addr
queue.output().publish(out, L3L2QueueOpcode::DATA);
```

The input window extension does not add an output completion manager. The L2
application owns completion tracking and decides when an output is ready to
publish.

Output reservation and publish remain single-outstanding per direction. The
input window allows multiple active input handles; it does not introduce
multiple concurrent output reservations.

## 13. STOP Semantics

`STOP` is an input queue descriptor message:

```text
seq + opcode=STOP + payload_nbytes=0
```

It follows normal FIFO ordering. STOP is a graceful shutdown request, not
cancel and not an immediate no-more-output marker.

Base helper behavior:

- L2 exits only after processing messages before the STOP.
- L2 releases the STOP descriptor and returns from the persistent run.
- `Worker.run` drain acts as the final acknowledgement.
- No extra STOP ACK counter is required.

Input-window behavior:

- STOP can be acquired while earlier DATA inputs are still active.
- STOP does not take effect ahead of earlier DATA inputs.
- The helper stops acquiring further DATA inputs after STOP is observed.
- Earlier active DATA inputs continue until the application completes them.
- Outputs produced by earlier DATA inputs may still be published while the
  helper drains.
- The helper releases only the FIFO completed prefix.
- Once all earlier DATA inputs are released, the helper releases STOP and the
  persistent L2 run exits.

STOP takes an input descriptor slot but does not count against
`max_l2_inflight`, because `max_l2_inflight` controls only active DATA input
ownership.

STOP is terminal for the input queue. After L3 successfully publishes STOP,
the input queue rejects further DATA, ERROR, or STOP enqueue attempts locally
without poisoning. If L2 has observed STOP and later observes any further
published input descriptor, including a second STOP, that is invalid published
descriptor state and poisons the queue.

STOP does not close the output queue. After publishing STOP, L3 may continue
dequeueing DATA or ERROR messages from the output queue. The transport has no
output-side terminal message and does not automatically know that the
persistent L2 run has returned. Applications that need to know all business
outputs have arrived must define that condition in their payload protocol, for
example with expected counts or final markers.

Publishing STOP and then immediately returning from the L3 orchestration
function is transport-legal. It can still be an application error if L2 needs
to publish final outputs: the output queue may fill and prevent L2 from
finishing, causing `Worker.run` drain to fail or time out.

Convenience APIs may expose:

```text
try_request_stop()
request_stop(timeout)
```

`try_request_stop()` attempts to publish a STOP descriptor to the input queue
and returns immediately if no input descriptor slot is available.

`request_stop(timeout)` performs a bounded wait until a STOP descriptor can be
published. The timeout covers only STOP enqueue/publish. It does not wait for
L2 exit and does not drain outputs. If the timeout expires before STOP is
published, the queue remains live and is not poisoned.

## 14. Queue Lifetime And Cleanup

A queue owns one primitive `L3L2OrchRegion`. Queue cleanup follows the
underlying region cleanup path:

```text
optional request_stop() -> L2 persistent run exits
L3 orchestration function returns
Worker.run drains submitted L2 work
runtime sends FREE_REGION for live L3-L2 regions
queue/region handles expire
```

`request_stop()` and `queue.free()` are different operations. `request_stop()`
is a protocol message that asks L2 to stop acquiring input. `queue.free()` is a
local handle release that rejects later queue use. Neither operation
synchronously releases the physical payload/counter region.

Physical release is deferred until `Worker.run` has drained submitted L2 work.
This keeps region memory live while an in-flight L2 task may still hold the
primitive descriptor or payload views. If the L3 orchestration function exits
with a live queue, runtime cleanup releases it through the same region cleanup
path.

Queue cleanup does not require the output queue to be empty. Once `Worker.run`
has drained and the persistent L2 run has returned, freeing the region is
memory-safe even if L3 left output messages unread. Those unread messages are
discarded with the region. Applications that need every output must dequeue
until their own final-output condition is satisfied before calling
`queue.free()` or returning from the orchestration function.

## 15. Error And Poison

Application-level failure is represented by `opcode=ERROR` and optional
application-defined payload bytes. `ERROR` is allowed in either direction and
may be published during normal processing or while draining after STOP.
Receiving `ERROR` does not poison the queue and does not change STOP
semantics.

Infrastructure poison is a queue/region state, not a descriptor message.

The guiding rule is:

```text
Before shared-state mutation: reject, no poison.
After shared-state mutation or inconsistent shared-state observation: poison.
```

Examples that do not poison:

- `try_enqueue` sees no space.
- `try_request_stop` sees no input descriptor slot.
- Blocking enqueue/dequeue/request-stop times out under ordinary backpressure.
- Payload is larger than the arena before reserve mutates state.
- Queue creation rejects ambiguous descriptor head/tail reconstruction
  parameters.
- User buffer is too small before read copies payload bytes.
- Invalid API arguments are caught before touching shared state.

Examples that poison:

- descriptor sequence mismatch;
- invalid opcode observed in a published descriptor;
- STOP observed on the output queue;
- descriptor payload range outside its arena;
- descriptor head/tail reconstruction or payload replay observes impossible
  shared state;
- payload copy failure after command issue;
- counter notify failure;
- control-service response timeout after command issue;
- L2 endpoint fatal error for this region;
- reservation, publish, or release state becomes self-contradictory.

Ordinary queue operation timeout does not prove remote poison. After a
blocking operation times out, the endpoint samples the peer abort flag. If the
peer flag is still zero, the timeout remains ordinary no-progress and does not
poison the local queue. If the peer flag is one, the endpoint reports remote
infrastructure abort and transitions its local handle to a terminal
remote-aborted state without setting its own abort flag. The peer may also
observe primitive region fatal errors or `Worker.run` drain errors.

Only local infrastructure poison sets the endpoint's own abort flag. Ordinary
timeouts, application `ERROR` messages, pre-mutation validation failures, and
observing the peer's abort flag do not set it.

The L2 input window helper also poisons the queue when local ownership state
becomes contradictory:

- completing an input handle unknown to the helper;
- completing or releasing a handle twice;
- attempting to release a non-contiguous input while earlier inputs remain
  incomplete;
- acquiring DATA after STOP has put the helper into draining;
- observing an acquired input sequence that contradicts the helper window.

The Python queue object mirrors the existing region state model:

```text
LIVE
RELEASED
POISONED(local-infrastructure)
POISONED(remote-aborted)
EXPIRED
```

After poison, reserve, enqueue, peek, read, release, publish, and stop-request
operations reject. Cleanup/free remains idempotent and valid.

L2 C++ helper poison reports a fatal error including the primitive region id,
so existing Host-side parsing can poison the corresponding region.

## 16. Implementation Staging

The feature can be implemented in two review-friendly stages. This staging is
not an API boundary: the base transport should intentionally leave room for
the input window without later ABI or L3 API changes.

```text
Stage 1:
  base SPSC message queue transport
  input and output descriptor rings
  input and output payload arenas
  descriptor head/tail protocol over int32_t signal counters
  single-writer abort flags for timeout disambiguation
  derived uint64_t payload cursors via descriptor replay
  DATA / ERROR / input-only STOP
  one active DATA input in the L2 helper/example

Stage 2:
  L2 input window helper
  max_l2_inflight
  application-driven input complete
  FIFO-safe release of completed input prefix
  flexible output cardinality and out-of-input-order output publish
  FIFO STOP drain with earlier DATA inputs still active
```

Stage 1 intentionally leaves room for Stage 2 through these hook points:

- descriptor `seq` is explicit and 64-bit;
- input release is explicit, not tied to dequeue;
- output reserve and publish are separate;
- each direction has at most one outstanding producer reservation;
- application correlation is kept in payload, so queue transport does not
  assume one input maps to one output;
- L3 queue creation and output ownership/dequeue APIs do not depend on
  `max_l2_inflight`.

Expected implementation locations:

```text
python/simpler/l3_l2_message_queue.py
src/common/platform/include/aicpu/l3_l2_message_queue.h
docs/l3-l2-message-queue.md
examples/a2a3/tensormap_and_ringbuffer/l3_l2_message_queue/
examples/a2a3/tensormap_and_ringbuffer/l3_l2_message_queue_input_window/
```

The exact Python module and public API names may change during implementation,
but the transport contract should remain stable.

## 17. Tests And Examples

Base queue tests should cover:

- layout calculation;
- descriptor slot encoding;
- counter offset assignment;
- queue creation rejecting ambiguous descriptor head/tail reconstruction
  parameters;
- enqueue reserve failure for payload larger than arena;
- backpressure when descriptor ring is full;
- backpressure when payload arena is full;
- arena wrap with invisible padding;
- STOP descriptor handling;
- `try_request_stop` and `request_stop(timeout)` behavior;
- ERROR as a normal application message in either direction;
- L3 ordinary host-buffer enqueue/read through lazy staging;
- L3 primitive-compatible registered Tensor fast paths without staging;
- staging allocation failure before primitive command issue not poisoning the
  queue;
- abort flags distinguishing ordinary timeout from remote infrastructure
  abort;
- local infrastructure poison setting the local abort flag;
- remote-aborted terminal state not setting the local abort flag;
- poison on invalid published descriptor state;
- poison on descriptor head/tail reconstruction or payload replay
  inconsistency;
- no poison on pre-mutation validation failure.

The new example should be parallel to the existing primitive stream example,
not a replacement for it. The primitive stream example should remain as the
minimal demonstration of `docs/l3-l2-orch-comm.md`.

The base queue example should demonstrate:

- `depth > 1`;
- variable-size input and output payloads;
- input and output backpressure;
- L2 persistent loop;
- one input message containing message-local AICore work;
- FIFO STOP shutdown;
- L3 optionally dequeuing output after STOP according to application final
  output rules.

Input window tests and examples should cover:

- `max_l2_inflight > 1`;
- refusing to acquire new DATA input when the input window is full;
- multiple input messages acquired before earlier inputs release;
- application-driven input completion;
- releasing only the FIFO completed prefix;
- one input producing multiple outputs;
- multiple inputs producing one output;
- output publish order differing from input acquire order;
- output correlation stored in the application payload header;
- STOP entering draining while earlier DATA inputs remain active;
- output DATA or ERROR publish during STOP drain;
- local ownership errors poisoning the queue.

Future work beyond the staged implementation is limited to out-of-order input
payload release, fragmented payload arena allocation, abort reason/status
metadata, low-latency abort polling, or concurrent output reservations, if
those become necessary.
