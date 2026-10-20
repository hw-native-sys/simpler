# CPU-NPU Message Queue

CPU-NPU Message Queue lets a CPU orchestration callback exchange ordered
messages with one running NPU orchestration task.

Use this API for request streams, serving loops, or any protocol that benefits
from ordered messages and built-in backpressure. If your application needs a
custom shared-memory layout instead, use
[CPU-NPU Region Communication](CPU-NPU-orch-comm.md).

## What the queue provides

The public `SpscQueue` has two directions:

- `queue.input`: CPU producer to NPU consumer;
- `queue.output`: NPU producer to CPU consumer.

It preserves FIFO transport order in each direction, supports variable-length
payloads and zero-byte messages, and provides cooperative shutdown with
`STOP`.

SPSC means one producer and one consumer per direction. Do not share one queue
between multiple CPU producers or multiple NPU consumers.

## Supported topology

The current API supports one CPU initiator and one same-node AICPU peer. Both
must belong to the current Worker tree.

Create the queue inside its owning orchestration callback:

- an L3 callback can select one of its L2 children;
- an L4 callback can select an L2 descendant through an L3 child;
- an L3 callback running inside an L4 tree can create a queue rooted at its
  own `L3`.

Create the queue before submitting the NPU task that consumes its binding. A
queue cannot be retained across runs, rebound, or connected to a remote node.

## Quick start

The following callback is taken from
`examples/workers/l3/worker_chip_message_queue`. `handle` is the callable
returned by `worker.register(...)`; the payload helpers and constants are
defined in the same example:

```python
def orch(orch_handle, _args, cfg):
    host = ce.at("L3", ce.HOST_CPU)
    device = ce.at("L3/L2[0]", ce.DEVICE_AICPU)
    queue = orch_handle.create_spsc_queue(
        initiator=host,
        peer=device,
        depth=_QUEUE_DEPTH,
        input_arena_bytes=_INPUT_ARENA_BYTES,
        output_arena_bytes=_OUTPUT_ARENA_BYTES,
    )

    task_args = TaskArgs()
    for scalar in queue.peer_task_arg_scalars():
        task_args.add_scalar(int(scalar))
    orch_handle.submit_next_level(handle, task_args, cfg, worker=0)

    payloads = _input_payloads()
    expected_outputs = _expected_outputs()

    for payload in payloads[:2]:
        queue.input.enqueue(payload, nbytes=len(payload), timeout=_TIMEOUT_S)
    _read_expected_outputs(queue, expected_outputs[:3])

    for payload in payloads[2:]:
        queue.input.enqueue(payload, nbytes=len(payload), timeout=_TIMEOUT_S)
    queue.request_stop(timeout=_TIMEOUT_S)
    _read_expected_outputs(queue, expected_outputs[3:])

    queue.free()
```

`peer_task_arg_scalars()` returns the ten opaque scalars expected by the NPU
queue endpoint. Pass them in order and do not edit or reconstruct them.

For an L4 callback, select paths such as `L4` and `L4/L3[0]/L2[0]`, submit to
the corresponding L3 child, and forward the complete scalar list unchanged.

## Configure capacity

`depth` is the number of message descriptors available in each direction. It
must be a power of two and no greater than `2^30`.

`input_arena_bytes` and `output_arena_bytes` reserve payload capacity for the
two directions. Each value must be a positive multiple of 64. A single payload
must fit contiguously in its direction's arena.

Choose the values from the workload:

- increase `depth` when many small messages can be outstanding;
- increase an arena when larger payloads or more payload bytes can be
  outstanding;
- input and output arenas may use different sizes.

Ordinary capacity pressure is not an error. Non-blocking operations report no
progress, while blocking operations wait until their finite timeout expires.

## Send messages from CPU to NPU

Blocking send:

```python
queue.input.enqueue(payload, nbytes=len(payload), timeout=5.0)
```

Non-blocking send:

```python
accepted = queue.input.try_enqueue(payload, nbytes=len(payload))
if not accepted:
    # Retry later or apply application-level backpressure.
    ...
```

Accepted payloads are:

- a registered HOST `Buffer`;
- contiguous `bytes`, `bytearray`, or `memoryview`;
- `None` for a zero-byte message.

For a zero-byte message:

```python
queue.input.enqueue(None, nbytes=0, timeout=5.0)
```

## Receive messages from NPU

Use `peek`, `read_into`, and `release` when the payload size determines the
destination allocation:

```python
message = queue.output.peek(timeout=5.0)
destination = bytearray(message.payload_nbytes)
queue.output.read_into(message, destination)
queue.output.release(message)
```

The message exposes:

- `seq`: transport sequence number;
- `opcode`: `DATA` or `ERROR`;
- `payload_offset`;
- `payload_nbytes`.

`seq` is transport metadata, not an application request ID. Put request IDs
and correlation fields in your payload header.

Always release a message after reading it. Until release, its queue slot and
payload space remain occupied.

If the destination already exists, use the convenience API:

```python
message = queue.output.dequeue_into(destination, timeout=5.0)
```

Non-blocking receive:

```python
message = queue.output.try_peek()
if message is not None:
    ...
```

`ERROR` is an application message, not queue poison. The application defines
its payload format and decides how to handle it.

## Stop cleanly

Request shutdown after all input messages have been published:

```python
queue.request_stop(timeout=5.0)
```

The call publishes `STOP`; it does not wait for the NPU task to exit and does
not drain output messages. Continue receiving any outputs produced while the
NPU side finishes accepted work. The owning `Worker.run(...)` waits for the
submitted task to return.

After `STOP`, further CPU input is rejected. Use
`queue.try_request_stop()` when the caller must not block.

## NPU-side integration

The NPU orchestration entry receives the queue binding in its first ten scalar
arguments. Decode those arguments and create the provided endpoint:

```cpp
#include "aicpu/region_endpoint_diagnostics.h"
#include "aicpu/region_instance_view.h"
#include "common/region_template.h"

uint64_t scalars[spsc_queue::kSpscQueueEndpointBindingScalarCount];
for (size_t i = 0; i < spsc_queue::kSpscQueueEndpointBindingScalarCount; ++i) {
    scalars[i] = orch_args.scalar(static_cast<int32_t>(i));
}

spsc_queue::SpscQueueEndpointBinding binding{};
if (!spsc_queue::decode_endpoint_binding(
        scalars, spsc_queue::kSpscQueueEndpointBindingScalarCount,
        &binding)) {
    return;
}

RegionInstanceView view(
    RegionPartLocalSpan{binding.payload_base, binding.payload_bytes},
    RegionPartLocalSpan{binding.counter_base, binding.counter_bytes});
spsc_queue::MonotonicClock clock{&queue_now_ns};
spsc_queue::SpscQueueEndpoint<RegionInstanceView> queue(
    binding, std::move(view), clock);

if (!queue.live()) {
    region_report_spsc_queue_error(queue.error());
    return;
}
```

The NPU task then loops over `queue.input().peek(...)`, processes each message,
publishes responses through `queue.output()`, and releases each input handle.
On `STOP`, finish earlier accepted work, release the stop handle, verify that
the input is drained, and return.

Check every endpoint operation. If `queue.error().kind` is not `NONE`, call
`region_report_spsc_queue_error(queue.error())` and return.

The complete device-side example is in
`examples/workers/l3/worker_chip_message_queue/kernels/orchestration/`.

## Lifetime

`queue.free()` is idempotent and immediately rejects further user operations.
It does not invalidate memory still used by submitted NPU work. Physical
cleanup happens after that work drains.

If the callback returns without calling `free()`, the run still reclaims the
queue automatically. Do not use a queue or a message handle after `free()` or
after the owning callback has returned.

## Errors

These conditions are ordinary and do not poison the queue:

- full descriptor ring;
- insufficient arena space;
- empty output queue;
- a finite blocking timeout when the peer has not aborted;
- an application `ERROR` message.

Malformed descriptors, invalid opcodes, stale or foreign handles, impossible
queue progress, and failures after shared state mutation poison the queue.
Normal operations then reject, but cleanup remains valid.

A trusted native failure is returned to the owning queue as:

```text
SPSC queue endpoint error session=0x0123456789abcdef transaction=7
```

If the native side cannot establish a trusted allocation identity, the failure
is:

```text
SPSC queue endpoint error identity=untrusted
```

Treat either form as terminal for the current run. Detailed diagnostics remain
in the device log; user code should not parse them.

## Run the example

```bash
pytest examples/workers/l3/worker_chip_message_queue \
  --platform a2a3sim
```

On supported hardware:

```bash
pytest examples/workers/l3/worker_chip_message_queue \
  --platform a2a3 --device <id>
```

The same public API is available on `a2a3sim`, `a5sim`, `a2a3`, and `a5`.
