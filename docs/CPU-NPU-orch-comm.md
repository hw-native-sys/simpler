# CPU-NPU Region Communication

CPU-NPU Region Communication lets a CPU orchestration callback exchange bytes
and synchronization signals with a running NPU orchestration task.

`create_region` is the single entry point for CPU-NPU shared memory. Message
queues, rings, and request/response protocols are application code built on
top of it; [SPSC queue example](#spsc-queue-example) walks through one such
protocol that ships with the repository.

## When to use a region

A region provides:

- one byte-addressed payload area;
- one area containing 32-bit signal counters;
- CPU methods for copying payload bytes;
- CPU methods for publishing, testing, and waiting on counters;
- an opaque descriptor that can be passed to the selected NPU peer.

Typical uses include a persistent NPU task that accepts several rounds of
input, publishes completion for each round, and exits after an application
defined stop message.

### Public surface

`simpler.comm_region` exports exactly these names:

| Name | Purpose |
| ---- | ------- |
| `RegionInstance` | `payload_write`, `payload_read`, `counter`, `descriptor_scalars`, `free` |
| `RegionCounter` | `notify`, `test`, `wait` |
| `NotifyOp` | `Set` and `Add` for `notify` |
| `WaitCmp` | comparison for `test` and `wait` |
| `SignalTestResult` | `matched` and `observed` from `test` |
| `MaterializationRefusal` | creation was refused |
| `RefusalReason` | why creation was refused |

Everything else in that module is internal. An application that stays within
this surface can define its own payload protocol without reaching into
local views, attachments, allocation identity, or the registry.

### Payload and counter are different access contracts

The two spans are not a fixed "data area plus signal area". Each one accepts a
different operation set with different rules:

| Rule | payload | counter |
| ---- | ------- | ------- |
| Operations | `payload_write`, `payload_read` | `notify`, `test`, `wait` |
| Unit | any `nbytes` within the span | exactly 4 bytes |
| Offset | any offset whose range fits | multiple of 4 |
| Value | opaque bytes from a host buffer | one signed 32-bit integer |
| Comparison | none | `WaitCmp` against the observed value |
| Blocking | never | `wait` blocks until its timeout |
| Concurrent writers | the application's own rule | one writer per counter |

So payload moves bounded byte ranges of a size the application chooses, and
counter carries a single comparable value you can publish, poll, or block on.
A counter cannot transfer bytes, and a payload range cannot be waited on.

On the current driver both go through the same host copy backend: payload
transfers are synchronous H2D/D2H copies, and counter notify, test, and wait
are copy-based 4-byte accesses.

## Understand endpoints

An endpoint identifies one participant in the current Worker tree. It combines:

- a logical path, such as `L3`, `L3/L2[0]`, or `L4/L3[0]/L2[0]`;
- a deployment, such as `HOST_CPU` or `DEVICE_AICPU`.

The path answers **which Worker-tree participant**. The deployment answers
**which execution environment at that path**.

An endpoint is not a memory address, device ID, region handle, or persistent
object. User code supplies an `EndpointSelector`; simpler resolves it against
the topology of the current orchestration callback. Do not cache a resolved
endpoint across callbacks or construct internal endpoint records yourself.

### Select one endpoint with `at`

`at(path, deployment)` selects exactly one endpoint:

```python
from simpler import comm_endpoints as ce

host = ce.at("L3", ce.HOST_CPU)
device = ce.at("L3/L2[0]", ce.DEVICE_AICPU)
```

Use `at` when the application knows the exact participant. Region and queue
creation normally use this form because their current public contracts require
one Host endpoint and one Device endpoint.

### Select endpoints under a path with `under`

`under(path, deployment)` expands to matching descendants:

```python
aicpu_endpoints = ce.under("L4/L3[0]", ce.DEVICE_AICPU)
```

Use `under` when an API accepts a set of members. Expansion order is stable,
but the result still has to satisfy that API's topology rules. The current
single-owner CPU-NPU Region shape ultimately requires exactly one CPU consumer
and one AICPU provider, so a selector that expands to several providers is
rejected.

### Assign endpoint roles

`members` declares who participates. `SingleOwner(provider=...)` declares
which member owns and serves the region:

```python
region = orch.create_region(
    members=(host, device),
    topology=ce.SingleOwner(provider=device),
    payload_bytes=payload_bytes,
    counter_bytes=counter_bytes,
)
```

For the supported CPU-NPU shape:

- the Host endpoint is the consumer that calls `payload_write`,
  `payload_read`, and counter methods;
- the Device AICPU endpoint is the provider passed the descriptor scalars;
- the provider selector must identify one of the declared members;
- the two roles must resolve to different endpoints.

Selectors are resolved when `create_region()` runs. A missing path, an empty
`under` expansion, duplicate members, overlapping selectors, or an ambiguous
provider raises `EndpointResolveError` before a region is allocated.

### Choose paths relative to the current callback

Use the registry root owned by the callback:

| Callback creating the region | Host endpoint | Device endpoint |
| ---------------------------- | ------------- | --------------- |
| L3 with its own chip | `at("L3", HOST_CPU)` | `at("L3/L2[0]", DEVICE_AICPU)` |
| L4 reaching a descendant chip | `at("L4", HOST_CPU)` | `at("L4/L3[0]/L2[0]", DEVICE_AICPU)` |
| Child L3 inside an L4 tree | `at("L3", HOST_CPU)` | `at("L3/L2[0]", DEVICE_AICPU)` |

The child L3 case deliberately uses `L3` as its root. It does not reuse the
parent's `L4/L3[...]` path when creating its own region.

## Supported topology

The current API supports one CPU consumer and one same-node AICPU provider.
Both endpoints must belong to the current Worker tree.

Create the region inside the orchestration callback that owns it:

- an L3 callback can select one of its L2 children;
- an L4 callback can select an L2 descendant through an L3 child;
- an L3 callback running as a child of an L4 tree can create a region rooted
  at its own `L3`.

Create the region before submitting work that uses it. A region cannot be
created outside its owning callback, retained across runs, or shared with a
remote node.

Cross-node members, `DEVICE_AICORE` providers, Host providers, several
consumers, and persistence across runs are not supported.

One logical region does not promise the same virtual address at every
endpoint, and it does not promise that every endpoint can load and store the
memory directly. Each side uses its own endpoint-local addresses: the CPU side
uses the `payload_*` and `counter` methods, and the NPU side uses the spans
carried by the descriptor. Pass the descriptor through; do not share a raw
pointer between the two sides.

## Quick start

The following callback is taken from
`examples/workers/l3/worker_chip_orch_comm_stream`. `handle` is the callable
returned by `worker.register(...)`; the constants and small payload helpers
are defined in the same example:

```python
def orch(orch_handle, _args, cfg):
    host = ce.at("L3", ce.HOST_CPU)
    device = ce.at("L3/L2[0]", ce.DEVICE_AICPU)
    region = orch_handle.create_region(
        members=(host, device),
        topology=ce.SingleOwner(provider=device),
        payload_bytes=_PAYLOAD_BYTES,
        counter_bytes=_COUNTER_BYTES,
    )
    data_ready = region.counter(_DATA_READY_COUNTER)
    completion = region.counter(_COMPLETION_COUNTER)
    header = orch_handle.alloc([_HEADER_BYTES], DataType.UINT8)
    host_input = orch_handle.alloc([_NUMEL], DataType.FLOAT32)
    host_output = orch_handle.alloc([_NUMEL], DataType.FLOAT32)

    task_args = TaskArgs()
    for scalar in region.descriptor_scalars():
        task_args.add_scalar(int(scalar))
    task_args.add_scalar(_INPUT_OFFSET)
    task_args.add_scalar(_OUTPUT_OFFSET)
    task_args.add_scalar(_NUMEL)
    task_args.add_scalar(DataType.FLOAT32.value)
    task_args.add_scalar(_NBYTES)
    task_args.add_scalar(_SCALAR)
    task_args.add_scalar(_DATA_READY_COUNTER)
    task_args.add_scalar(_COMPLETION_COUNTER)
    orch_handle.submit_next_level(handle, task_args, cfg, worker=0)

    for seq in range(1, _ROUNDS + 1):
        expected = _fill_input(host_input, seq)
        region.payload_write(_INPUT_OFFSET, host_input, nbytes=_NBYTES)
        _write_header(header, seq, 1)
        region.payload_write(0, header, nbytes=_HEADER.size)
        data_ready.notify(seq, NotifyOp.Set)
        snapshot = completion.test(seq, WaitCmp.GE)
        if not snapshot.matched:
            assert snapshot.observed < seq
            completion.wait(seq, WaitCmp.GE, timeout=5.0)
        region.payload_read(_OUTPUT_OFFSET, host_output, nbytes=_NBYTES)
        _assert_output_matches(host_output, expected)

    stop_seq = _ROUNDS + 1
    _write_header(header, stop_seq, 2)
    region.payload_write(0, header, nbytes=_HEADER.size)
    data_ready.notify(stop_seq, NotifyOp.Set)
    region.free()
```

`descriptor_scalars()` returns eight opaque scalars in the order expected by
the native decoder. Append application arguments after those eight values.
Do not reorder, edit, or reconstruct the descriptor.

For an L4 callback, use paths such as `L4` and `L4/L3[0]/L2[0]`, then submit
to the corresponding L3 child. The complete scalar list must be forwarded
unchanged.

## Define the payload protocol

The application owns the payload layout. Reserve non-overlapping ranges for
headers, inputs, and outputs, and pass the corresponding offsets to the NPU
task as application scalars.

For example:

```text
offset 0                      64-byte command header
offset 64                     input tensor bytes
offset 64 + input_nbytes      output tensor bytes
```

`payload_write(offset, buffer, nbytes)` copies bytes from the CPU buffer into
the region. `payload_read(offset, buffer, nbytes)` copies bytes back.

Buffers must be contiguous and large enough for the requested transfer.
Buffers returned by `orch.alloc(...)` are accepted. Ordinary `bytes`,
`bytearray`, and contiguous memory views may also be used when their
read/write direction is valid.

## Use counters as publish and observe points

Create a counter handle at a byte offset:

```python
ready = region.counter(0)
done = region.counter(64)
```

Publish a value:

```python
ready.notify(sequence, NotifyOp.Set)
ready.notify(delta, NotifyOp.Add)
```

Poll once without blocking:

```python
sample = done.test(sequence, WaitCmp.GE)
if sample.matched:
    print(sample.observed)
```

Wait with a finite timeout:

```python
observed = done.wait(sequence, WaitCmp.GE, timeout=5.0)
```

`WaitCmp` supports `EQ`, `NE`, `GT`, `GE`, `LT`, and `LE`. Prefer `GE` for a
monotonic sequence so the waiter still succeeds if it observes a later value.

`NotifyOp.Add` is a copy-based read-modify-write, not a multi-writer atomic
fetch-add. Keep one writer per counter and one writer per 64-byte counter
cache line. Place counters written by different sides at offsets such as `0`
and `64`.

A failed `test` and an ordinary `wait` timeout do not poison the region.

### What a publish guarantees

The supported path guarantees this ordering in both directions:

1. an endpoint finishes its `payload_write` calls, then calls `notify`;
2. the peer observes that publication through `test` or `wait`;
3. the peer's later `payload_read` sees the payload writes that completed
   before the publication.

The guarantee comes from synchronous copies, call ordering, and the backend's
cache maintenance. It does not imply a shared virtual address or a
general-purpose hardware atomic. Keep the order above: a `notify` issued
before its payload writes complete publishes nothing useful.

## NPU-side integration

The NPU orchestration entry receives the descriptor in its first eight scalar
arguments. Decode it with the provided helper and construct the core view:

```cpp
#include "aicpu/region_instance_view.h"
#include "aicpu/region_endpoint_diagnostics.h"
#include "common/region_task_args.h"

uint64_t scalars[REGION_TASK_SCALAR_COUNT] = {
    orch_args.scalar(0), orch_args.scalar(1),
    orch_args.scalar(2), orch_args.scalar(3),
    orch_args.scalar(4), orch_args.scalar(5),
    orch_args.scalar(6), orch_args.scalar(7),
};

RegionPartLocalSpan payload{};
RegionPartLocalSpan counter{};
uint64_t session_bits = 0;
uint64_t transaction_id = 0;
uint64_t region_id = 0;
RegionTaskArgError decode_error = RegionTaskArgError::OK;

if (!region_decode_task_args(
        scalars, REGION_TASK_SCALAR_COUNT, payload, counter,
        session_bits, transaction_id, region_id, decode_error)) {
    region_report_invalid_task_args(decode_error);
    return;
}

RegionInstanceView region(payload, counter);
```

Use `region.payload()` to obtain validated payload views and
`region.counter()` to notify, test, or wait by offset. Check every operation's
return value. If an operation fails, report it with
`region_report_endpoint_error(...)` and return from the orchestration entry.

The complete native pattern is in
`examples/workers/l3/worker_chip_orch_comm_stream/kernels/orchestration/`.

## Shutdown and lifetime

Define shutdown as part of the application protocol. A common pattern is:

1. write a `STOP` command into the payload header;
2. publish its sequence through the ready counter;
3. let the NPU task finish accepted work and return.

`region.free()` is idempotent and immediately rejects further user access.
It does not invalidate memory still used by submitted NPU work. Physical
cleanup happens after that work drains. If the callback returns without
calling `free()`, the run still reclaims the region automatically.

Do not use a region, a previously acquired counter, or descriptor scalars
after `free()` or after the owning callback has returned.

An application `STOP` and `region.free()` are independent. `STOP` is a message
in your protocol that tells the NPU task to finish; `free()` ends CPU access to
the region. Neither one implies the other, and neither replaces the run's own
drain of submitted work.

## Errors

Argument, range, buffer, and timeout validation errors are reported directly
and do not poison a region when no operation was issued.

An issued data-plane failure makes the region unusable. Native failures with a
trusted allocation identity are returned to the owning region with a short
marker:

```text
Region endpoint error session=0x0123456789abcdef transaction=7
```

If the native side cannot establish a trusted identity, the failure is
reported as:

```text
Region endpoint error identity=untrusted
```

Treat either form as terminal for the current run. Detailed native diagnostics
remain in the device log; user code should not parse those details.

## Run the example

```bash
pytest examples/workers/l3/worker_chip_orch_comm_stream \
  --platform a2a3sim
```

On supported hardware:

```bash
pytest examples/workers/l3/worker_chip_orch_comm_stream \
  --platform a2a3 --device <id>
```

The same public API is available on `a2a3sim`, `a5sim`, `a2a3`, and `a5`.

## SPSC queue example

`examples/workers/l3/worker_chip_message_queue` implements a duplex
single-producer, single-consumer message queue on top of the public region
API. It is a reference application, not part of the simpler runtime: the
layout, the opcodes, the backpressure rule, and the `STOP` handshake all live
in the example. Treat it as a worked example of a protocol you could write
differently, or replace entirely, without changing `python/simpler`.

The example's `create_spsc_queue` helper calls `create_region` and nothing
else:

```python
layout = SpscQueueLayout.create(
    depth=depth,
    input_arena_bytes=input_arena_bytes,
    output_arena_bytes=output_arena_bytes,
)
region = orch.create_region(
    members=(initiator, peer),
    topology=ce.SingleOwner(provider=peer),
    payload_bytes=layout.payload_bytes,
    counter_bytes=layout.counter_bytes,
)
return SpscQueue(region, layout)
```

The layout places two descriptor rings and two payload arenas inside the
payload span, and the head, tail, and abort counters inside the counter span.
Every queue operation is a `payload_write`, `payload_read`, or counter call on
that region; the queue reads no private region attribute.

### Task arguments are 8 + 3 scalars

The NPU side receives eleven scalars:

| Index | Meaning |
| ----- | ------- |
| 0-7 | region descriptor from `descriptor_scalars()`, forwarded unchanged |
| 8 | queue depth |
| 9 | input arena bytes |
| 10 | output arena bytes |

The first eight are decoded by the shared `region_decode_task_args` helper
into a `RegionInstanceView`. The last three are application configuration that
the example's own header validates. Nothing in the first eight is reinterpreted
by the queue.

### Failures split by owner

Region access failures and protocol failures are reported through different
paths. When the underlying view has failed, the example reports it as a region
failure with `region_report_endpoint_error(...)`, so the host attributes it to
the owning region. A bad sequence number, an unknown opcode, or a broken
ownership rule is an application bug and is reported with `rt_report_fatal`.
The runtime does not recognize queue protocol errors.

### Run it

```bash
pytest examples/workers/l3/worker_chip_message_queue --platform a2a3sim
pytest examples/workers/l3/worker_chip_message_queue --platform a2a3 --device <id>
```

See the example's
[README](../examples/workers/l3/worker_chip_message_queue/README.md) for the
request modes, the response ordering, and the device-side loop.
