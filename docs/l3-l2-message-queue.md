# L3-L2 Message Queue

L3-L2 Message Queue lets an L3 Host Orchestrator exchange ordered messages
with one persistent L2 AICPU Orchestrator task.

The intended use case is repeated in-flight work: L3 enqueues input messages,
L2 consumes them while the L2 task stays alive, L2 publishes output messages,
and L3 dequeues those outputs. The queue is a Region Template bound onto a
payload-and-counter `RegionInstance`. The lower-level orchestration primitives
are documented in [l3-l2-orch-comm.md](l3-l2-orch-comm.md). For where L3 and
L2 sit in the runtime stack, see
[hierarchical-level-runtime.md](hierarchical-level-runtime.md).

There is one current binding: exactly ten little-endian `uint64` scalars. There
is no 12-scalar TaskArgs path and no decoder fallback.

## 1. Create And Bind

L3 creates one queue for one chip worker:

```python
queue = orch.create_worker_chip_queue(
    worker_id=0,
    depth=4,
    input_arena_bytes=1 << 20,
    output_arena_bytes=1 << 20,
)
```

`create_worker_chip_queue` is a compatibility factory. It does not call
`create_worker_chip_region()`. A Region Template coordinator materializes one
`RegionInstance`, binds the SPSC template onto that instance, and projects a
`WorkerChipQueue` for the L3 initiator.

L3 hands the injected peer binding to L2 as TaskArgs scalars starting at
offset 0:

```python
l2_args = TaskArgs()
for value in queue.chip_task_arg_scalars():
    l2_args.add_scalar(value)

orch.submit_next_level(l2_handle, l2_args, cfg, worker=0)
```

`chip_task_arg_scalars()` returns the ten-field `SpscQueueEndpointBinding` in
this order:

```text
magic_version
session_instance_id_bits
transaction_id
payload_base
payload_bytes
counter_base
counter_bytes
depth
input_arena_bytes
output_arena_bytes
```

`magic_version` packs the `SPSQ` magic (`0x53505351`) with the compiled major
and minor wire fields. A decoder that sees any other packed value fails closed
as an unsupported SPSQ version. `transaction_id` is part of the allocation
identity and must be nonzero. `session_instance_id_bits` may be zero. Native
poison diagnostics carry `(session_instance_id_bits, transaction_id)` so a
failure can be correlated to that `RegionInstance`.

`queue.region` is a compatibility escape hatch: a `WorkerChipOrchRegion`
projector over the same instance's PAYLOAD and COUNTER local views. Queue
layout, publication, failure, and lifecycle authority stay on the bound queue.
Independent `create_worker_chip_region()` remains available for primitive
orchestrator communication that is not a queue.

L3 sends input through `queue.input` and receives output through
`queue.output`:

```python
queue.input.enqueue(host_input, nbytes=nbytes, timeout=timeout_s)
message = queue.output.peek(timeout=timeout_s)
queue.output.read_into(message, host_output)
queue.output.release(message)
```

`try_enqueue(buffer, nbytes)` returns `False` for ordinary descriptor-ring or
arena backpressure and does not poison the queue. `try_peek()` and
`try_dequeue_into(buffer)` return `None` when no output is available.
`dequeue_into(buffer, timeout)` peeks, copies, and releases in one call.

Admitted payloads are a registered HOST `Buffer`, or a contiguous host
`bytes` / `bytearray` / `memoryview`. Zero-byte messages use
`buffer_or_none=None` and `nbytes=0`.

L3 requests graceful shutdown with `queue.request_stop(timeout)` or
`try_request_stop()`. `queue.free()` is logical: it releases the L3 queue
handle and marks the projected region handle released. It does not
synchronously free device memory. Physical cleanup follows the underlying
`RegionInstance` lifetime after submitted L2 work has drained.

On L2, orchestration code decodes the ten scalars and constructs an injected
endpoint view:

```cpp
#include "aicpu/region_instance_view.h"
#include "common/region_template.h"

uint64_t scalars[spsc_queue::kSpscQueueEndpointBindingScalarCount];
for (size_t i = 0; i < spsc_queue::kSpscQueueEndpointBindingScalarCount; ++i) {
    scalars[i] = orch_args.scalar(static_cast<int32_t>(i));
}

spsc_queue::SpscQueueEndpointBinding binding{};
if (!spsc_queue::decode_endpoint_binding(
        scalars, spsc_queue::kSpscQueueEndpointBindingScalarCount, &binding)) {
    return;
}

RegionInstanceView view(
    RegionPartLocalSpan{binding.payload_base, binding.payload_bytes},
    RegionPartLocalSpan{binding.counter_base, binding.counter_bytes}
);
spsc_queue::SpscQueueEndpoint<RegionInstanceView> queue(binding, std::move(view), clock);
if (!queue.live()) {
    return;
}
```

The default endpoint allows one active L2 DATA/ERROR input handle at a time.
L2 can opt into a larger input window with a compile-time parameter:

```cpp
spsc_queue::SpscQueueEndpoint<RegionInstanceView, 4> queue(binding, std::move(view), clock);
```

`MaxInflight` is a local construction parameter. It is not part of L3 queue
creation and does not change the shared layout or the ten-scalar binding. The
valid range is `1 <= MaxInflight <= depth`. Invalid combinations report
`BAD_ARGUMENT` without setting the L2 abort flag. STOP does not count against
`MaxInflight`; the endpoint keeps one extra slot so a STOP handle can remain
pending behind earlier DATA/ERROR handles.

## 2. Layout And Descriptor

The physical payload range is split as:

```text
payload region
|-- input descriptor ring
|-- output descriptor ring
|-- input payload arena
`-- output payload arena
```

Input arena: producer = L3, consumer = L2. Output arena: producer = L2,
consumer = L3.

`depth` is the descriptor-ring capacity in each direction. It must be a power
of two and at most `2^30`. Queue capacity is exactly `depth` messages, not
`depth - 1`. `input_arena_bytes` and `output_arena_bytes` must be positive
64-byte multiples. A single message payload must fit as one contiguous span
inside its direction's arena. Payloads are not split across arena wrap.

Python `queue.layout` and C++ `queue.layout()` expose the same mirrored
offsets: descriptor rings, arenas, `payload_bytes`, and `counter_bytes`. L2
rejects construction unless the binding sizes match both the local layout
calculation and the injected view spans.

Each descriptor slot is 32 bytes:

```cpp
struct SpscQueueDescriptor {
    uint64_t seq;
    uint64_t opcode;
    uint64_t payload_offset;
    uint64_t payload_nbytes;
};
```

`seq` is the transport sequence number. It is not a user request ID.
Applications that need request IDs or correlation fields put them in a payload
header. `payload_offset` is relative to the payload base. Zero-byte messages
use `payload_offset == 0` and `payload_nbytes == 0`.

| Opcode | Meaning |
| ------ | ------- |
| `DATA` | Ordinary application payload message. |
| `STOP` | Graceful input-side shutdown request. |
| `ERROR` | Ordinary application-level error payload. |

`STOP` is valid only on the input queue. L2 exit is observed through normal
`Worker.run` drain. `ERROR` is a normal message: the queue does not interpret
its payload and does not poison on receipt. Infrastructure failures use poison
state instead.

## 3. Publication And Timeouts

Shared signals sit on a 64-byte stride:

```text
offset 0:   input_desc_tail        writer=L3
offset 64:  input_desc_head        writer=L2
offset 128: output_desc_tail       writer=L2
offset 192: output_desc_head       writer=L3
offset 256: initiator_abort_flag   writer=L3
offset 320: peer_abort_flag        writer=L2
```

Descriptor counters store the signed low 32 bits of monotonic logical
head/tail values. Each endpoint reconstructs its local 64-bit value from
observed progress. Unobserved progress must be between zero and `depth`;
anything else poisons the queue.

The producer sequence is:

```text
reserve payload space
write payload bytes
make those bytes visible to the peer
write descriptor fields
write descriptor seq
publish descriptor tail counter
```

Payload bytes must be visible to the peer before the producer writes `seq` and
publishes the tail. L2 `RegionInstanceView` payload writes flush the span.
AIV-produced output must be flushed before `publish`. The consumer observes
tail progress, validates the descriptor, uses the payload, then releases and
publishes the head.

Python blocking queue operations require a finite positive timeout;
`timeout <= 0` raises `ValueError`. Python `try_*` APIs are the non-blocking
path and return `False` or `None` for ordinary no-progress.

C++ blocking `peek(timeout_ns, ...)` and `reserve(nbytes, timeout_ns, ...)`
treat `timeout_ns == 0` as a live no-attempt: on a live endpoint they return
`false` without waiting and without attempting the operation. C++ `try_*`
APIs are the non-blocking progress path. A `false` return can mean
ordinary no-progress, validation failure, or poison; check `queue.error().kind`
to distinguish ordinary no-progress from terminal error.

Timeout under ordinary backpressure is not poison. After a positive-timeout
wait expires, an endpoint samples the peer abort flag; if that flag is set,
the local endpoint reports remote abort.

## 4. Ownership, STOP, And Errors

Queue ownership is per message.

On L3 output, `peek()` returns a handle that remains active until
`release(handle)`. While a handle is active, repeated `try_peek()` returns the
same handle. Releasing the wrong handle poisons the queue.

On L2 input, the default endpoint keeps one active DATA/ERROR handle. L2 must
not call `peek()` again before releasing that handle, except that STOP may also
be acquired into the extra STOP slot. With `SpscQueueEndpoint<View, N>`, L2
may hold up to `N` active DATA or ERROR inputs. `release(handle)` is logical
completion; the queue physically advances the shared input head only for the
completed FIFO prefix. If input 2 is released before input 1, input 2 remains
physically owned until input 1 is also released.

On L2 output, `reserve()` returns one active reservation. L2 fills that span,
then calls `publish(reservation, opcode)`. Publishing an unknown, stale,
already-published, or cross-queue reservation poisons the queue. The input
window does not introduce multiple concurrent output reservations.

`STOP` is an input descriptor with no payload. After L3 publishes `STOP`,
further input messages are rejected locally without poisoning. L3 may still
dequeue outputs that L2 publishes before returning. `request_stop(timeout)`
waits only until the `STOP` descriptor is published; it does not wait for L2
exit and does not drain outputs.

With an input window, STOP may be acquired while earlier DATA or ERROR inputs
are still active. After STOP is acquired, the input queue enters draining mode
and does not acquire later DATA or ERROR descriptors. Earlier active inputs
may still produce outputs. STOP is physically released only after all earlier
active inputs are physically released. `queue.input().drained()` returns true
only after STOP has been physically released. If L2 observes a published input
descriptor after STOP, that descriptor poisons the queue with
`INVALID_DESCRIPTOR`.

No-progress is non-terminal: descriptor ring full, payload arena full, empty
output queue, or a blocking timeout with no peer abort flag. Application-level
`ERROR` is a normal message and does not set an abort flag.

Infrastructure poison is terminal for the local queue handle: descriptor
sequence mismatch, invalid opcode, output-side `STOP`, payload outside its
arena, impossible counter reconstruction or payload replay, payload command
failure after shared mutation begins, counter notify failure, or stale handle
ownership. A local poison sets the local abort flag for the peer. Observing
the peer abort flag reports remote abort but does not set the local flag.
After poison, normal queue operations reject. Cleanup remains valid.

## 5. Example And Platform Evidence

The shipped smoke example lives at:

```text
examples/workers/l3/worker_chip_message_queue/
```

It uses `spsc_queue::SpscQueueEndpoint<RegionInstanceView, 4>` and a PTO-ISA
AIV kernel. L3 sends an initial pair of DATA inputs, drains the outputs that
the persistent L2 run publishes for them, then sends another pair of DATA
inputs followed by STOP. L2 acquires multiple inputs before releasing earlier
ones, publishes outputs in a different order from input acquisition, emits
multiple outputs for one input, combines two inputs into one output during
STOP drain, and returns only after `queue.input().drained()`. Application
request IDs in that example are payload-header fields `0, 0, 0, 7`; the
transport `seq` is not used as a request ID.

That example is a smoke path. It is not a formal Queue Acceptance record, and
it does not demonstrate wrap, `ERROR` opcode delivery, or dedicated
HostVmmCopyAccess/cache instrumentation.

Recorded evidence at commit `c87f0bad`, CI run
[33740884796](https://github.com/hw-native-sys/simpler/actions/runs/33740884796):

| Surface | Result |
| ------- | ------ |
| `a2a3sim` example ST | PASS (CI `st-sim-a2a3`; also reproduced locally) |
| `a5sim` example ST | PASS (CI `st-sim-a5`) |
| `a2a3` onboard example ST | PASS (`test_worker_chip_message_queue`, 8.1s, device 0, CI `st-onboard-a2a3`) |
| `a5` onboard example ST | PASS (`test_worker_chip_message_queue`, 7.0s, device 4, CI `st-onboard-a5`) |
| Python / C++ UT | PASS (CI `ut`, `ut-a2a3`, `ut-a5`) |
| Template-level ST under `tests/st/worker/comm_region/templates/queue/` | pending; directory is not in the tree |
| Dedicated wrap / `ERROR` / HostVmmCopyAccess instrumentation | not recorded |

Simulation evidence does not stand in for hardware cache or HostVMM copy
behavior. Hardware rows above are example ST pass/fail only.
