# L3-L2 Message Queue

L3-L2 Message Queue lets an L3 Host Orchestrator exchange ordered messages
with one persistent L2 AICPU Orchestrator task.

The intended use case is repeated in-flight work: L3 enqueues input messages,
L2 consumes them while the L2 task stays alive, L2 publishes output messages,
and L3 dequeues those outputs. The queue is built on top of the lower-level
L3-L2 orchestration communication primitives described in
[l3-l2-orch-comm.md](l3-l2-orch-comm.md). For where L3 and L2 sit in
the runtime stack, see
[hierarchical_level_runtime.md](hierarchical_level_runtime.md).

## 1. API

L3 creates one queue for one chip worker:

```python
queue = orch.create_l3_l2_queue(
    worker_id=0,
    depth=4,
    input_arena_bytes=1 << 20,
    output_arena_bytes=1 << 20,
)
```

The queue owns one underlying `L3L2OrchRegion`. Its payload range is split into
input/output descriptor rings and input/output payload arenas. Its counter
range stores descriptor head/tail signals and abort flags.

L3 passes the primitive region descriptor and queue layout arguments to L2:

```python
l2_args = TaskArgs()
for value in queue.l2_task_arg_scalars():
    l2_args.add_scalar(value)

orch.submit_next_level(l2_handle, l2_args, cfg, worker=0)
```

`l2_task_arg_scalars()` returns:

```text
primitive region descriptor scalars[0..5]
queue_magic_version
depth
input_arena_bytes
output_arena_bytes
payload_bytes
counter_bytes
```

L3 sends input messages through `queue.input`:

```python
host_input = orch.alloc([nbytes], DataType.UINT8)
fill_input(host_input)

queue.input.enqueue(host_input, nbytes=nbytes, timeout=timeout_s)
```

`try_enqueue(buffer, nbytes)` is the non-blocking form. It returns `False`
when the input descriptor ring or payload arena has no space. That result is
ordinary backpressure and does not poison the queue.

L3 receives output messages through `queue.output`:

```python
host_output = orch.alloc([max_output_nbytes], DataType.UINT8)

message = queue.output.peek(timeout=timeout_s)
queue.output.read_into(message, host_output)
queue.output.release(message)
```

The convenience form reads and releases in one operation:

```python
message = queue.output.dequeue_into(host_output, timeout=timeout_s)
```

`try_peek()` and `try_dequeue_into(buffer)` are the non-blocking forms. They
return `None` when no output message is available.

The L3 buffer arguments currently must be runtime-managed tensors returned by
`orch.alloc(...)`. Ordinary Python `bytes`, `bytearray`, and private tensors
are rejected before shared queue state is modified. Zero-byte messages use
`buffer_or_none=None` and `nbytes=0`.

L3 requests graceful shutdown by publishing an input-side `STOP` descriptor:

```python
queue.request_stop(timeout=timeout_s)
queue.free()
```

`try_request_stop()` is the non-blocking form. `queue.free()` releases the L3
queue handle and marks the underlying `L3L2OrchRegion` handle released. It does
not synchronously free device memory; physical cleanup follows the underlying
region lifetime model after submitted L2 work has drained. Small Python wrapper
scratch tensors used for descriptor staging are owned by the queue object and
follow normal Python object lifetime.

On L2, orchestration code receives the primitive descriptor and queue args,
then constructs an endpoint:

```cpp
L3L2OrchRegionDesc desc{/* scalars from TaskArgs */};
L3L2QueueArgs queue_args{
    magic_version,
    depth,
    input_arena_bytes,
    output_arena_bytes,
    payload_bytes,
    counter_bytes,
};

L3L2QueueEndpoint queue(desc, queue_args);
if (queue.error().kind != L3L2QueueErrorKind::NONE) {
    return;
}
```

L2 consumes input messages from `queue.input()` and publishes outputs through
`queue.output()`:

```cpp
while (true) {
    L3L2QueueInputHandle input{};
    if (!queue.input().peek(timeout_ns, &input)) {
        return;
    }

    if (input.opcode == L3L2QueueOpcode::STOP) {
        queue.input().release(input);
        return;
    }

    L3L2QueueOutputReservation output{};
    if (!queue.output().reserve(input.payload_nbytes, timeout_ns, &output)) {
        return;
    }

    launch_aicore(input.payload, output.payload);
    wait_aicore_done();

    queue.output().publish(output, L3L2QueueOpcode::DATA);
    queue.input().release(input);
}
```

`queue.input().try_peek(&input)` and
`queue.output().try_reserve(nbytes, &reservation)` are non-blocking. A `false`
return can mean no progress, timeout, validation failure, or poison; check
`queue.error().kind` to distinguish ordinary no-progress from terminal error.

## 2. Layout

The physical region has one payload range:

```text
payload region
|-- input descriptor ring
|-- output descriptor ring
|-- input payload arena
`-- output payload arena
```

The two payload arenas are separate:

```text
input arena:  producer = L3, consumer = L2
output arena: producer = L2, consumer = L3
```

`depth` is the descriptor-ring capacity in each direction. It must be a power
of two and at most `2^30`. Queue capacity is exactly `depth` messages, not
`depth - 1`.

`input_arena_bytes` and `output_arena_bytes` must be positive 64-byte
multiples. They do not need to be powers of two. A single message payload must
fit as one contiguous span inside its direction's arena. Payloads are not split
across arena wrap.

Python and C++ mirror the same deterministic queue layout calculation:

```text
input_desc_offset
output_desc_offset
input_arena_offset
output_arena_offset
payload_bytes
counter_bytes
```

Python exposes this as `queue.layout`; L2 exposes it as `queue.layout()`.
L3 passes the derived `payload_bytes` and `counter_bytes` to L2. L2 rejects
initialization unless those values match both its local layout calculation and
the primitive region descriptor sizes. Lockstep tests cover representative
layout cases for the mirrored Python and C++ calculations.

## 3. Descriptor ABI

Each descriptor slot is 32 bytes:

```cpp
struct L3L2QueueDescSlot {
    uint64_t seq;
    uint64_t opcode;
    uint64_t payload_offset;
    uint64_t payload_nbytes;
};
```

`seq` is the transport sequence number for ring validation, wrap detection, and
diagnostics. It is not a user request ID. Applications that need request IDs,
batch IDs, final markers, or correlation fields should put them in their own
payload header.

`payload_offset` is relative to the primitive region payload base. The payload
must be wholly inside the matching direction's arena. Zero-byte messages use
`payload_offset == 0` and `payload_nbytes == 0`.

The queue currently defines these opcodes:

| Opcode | Meaning |
| ------ | ------- |
| `DATA` | Ordinary application payload message. |
| `STOP` | Graceful input-side shutdown request. |
| `ERROR` | Ordinary application-level error payload message. |

`STOP` is valid only on the input queue. The output queue has no `STOP`
message; L2 exit is observed through normal `Worker.run` drain.

`ERROR` is a normal queue message. The queue layer does not interpret its
payload and does not poison the queue when an `ERROR` message is received.
Infrastructure failures use poison state instead.

## 4. Signals And Ordering

The queue uses the primitive signal counters as descriptor head/tail values.
Each shared signal is placed on a 64-byte stride:

```text
offset 0:   input_desc_tail       writer=L3
offset 64:  input_desc_head       writer=L2
offset 128: output_desc_tail      writer=L2
offset 192: output_desc_head      writer=L3
offset 256: l3_abort_flag         writer=L3
offset 320: l2_abort_flag         writer=L2
```

Descriptor counters store the low 32 bits of monotonic logical head/tail
values. Each endpoint reconstructs its local 64-bit value from observed
progress. The unobserved progress must be between zero and `depth`; anything
else is inconsistent shared state and poisons the queue.

The producer sequence is:

```text
reserve payload space
write payload bytes
write descriptor fields
write descriptor seq
publish descriptor tail counter
```

The consumer sequence is:

```text
observe descriptor tail progress
read and validate descriptor
use payload bytes or payload view
release descriptor and payload
publish descriptor head counter
```

All Python blocking queue operations require finite positive timeouts; passing
`timeout <= 0` is a caller error and raises `ValueError`. Python `try_*` APIs
are non-blocking and return `False` or `None` for ordinary no-progress.

C++ blocking queue operations take `timeout_ns`; `timeout_ns == 0` is an
immediate timeout probe. They return `false` on no-progress, timeout,
validation failure, or poison. C++ `try_*` APIs are non-blocking and also
return `false` for ordinary no-progress.

Timeout under ordinary backpressure is not poison. After timeout, an endpoint
samples the peer abort flag; if the peer flag is set, the local endpoint
reports remote abort.

## 5. Ownership

Queue ownership is per message.

On L3 output, `peek()` returns a handle that remains active until
`release(handle)`. While a handle is active, repeated `try_peek()` returns the
same handle. The caller may read the payload with `read_into(handle, buffer)`
before releasing it. Releasing the wrong handle is an ownership error and
poisons the queue.

On L2 input, `peek()` returns one active input handle. L2 must not call
`peek()` again before releasing that handle. L2 must not release an input until
all AICore work that reads the input payload has completed.

On L2 output, `reserve()` returns one active output reservation. L2 fills the
reserved payload span, then calls `publish(reservation, opcode)`. Publishing an
unknown, stale, already-published, or cross-queue reservation is an ownership
error and poisons the queue.

The base queue supports at most one active L2 input handle and one active L2
output reservation. It does not provide a multi-input L2 window.

## 6. STOP Semantics

`STOP` is an input descriptor with no payload. It follows normal FIFO ordering:
L2 observes and releases messages before `STOP`, then releases `STOP` and
returns from the persistent run.

After L3 successfully publishes `STOP`, the input queue rejects further input
messages locally without poisoning. L3 may still dequeue output messages that
L2 publishes before returning.

`request_stop(timeout)` waits only until the `STOP` descriptor is published.
It does not wait for L2 exit and does not drain outputs. Applications that need
all outputs must keep dequeuing until their own protocol-level final condition
is satisfied before returning from the L3 orchestration function.

## 7. Error Handling

The queue distinguishes no-progress, application errors, and infrastructure
poison.

No-progress is non-terminal:

- descriptor ring full;
- payload arena full;
- empty output queue;
- blocking operation timeout with no peer abort flag.

Application-level error is represented by `opcode=ERROR`. It is delivered to
the peer as a normal message and does not set an abort flag.

Infrastructure poison is terminal for the local queue handle:

- descriptor sequence mismatch;
- invalid opcode in a published descriptor;
- output-side `STOP`;
- descriptor payload outside its direction's arena;
- impossible counter reconstruction or payload replay;
- payload command failure after shared mutation begins;
- counter notify failure;
- stale or invalid handle/reservation ownership.

When an endpoint enters local infrastructure poison, it sets its own abort flag
for the peer. Observing the peer abort flag reports remote abort but does not
set the local abort flag.

After poison, normal queue operations reject. Cleanup remains valid.

## 8. Platform Support

The message queue uses the existing L3-L2 orchestration communication region,
payload, and counter primitives.

- `a2a3sim`: supported.
- `a5sim`: supported.
- `a2a3` onboard: supported where the underlying L3-L2 communication
  primitives are supported.
- `a5` onboard: follows the underlying L3-L2 communication support status.

Simulation backends preserve the same API, ordering, timeout, and error
semantics as onboard backends.
