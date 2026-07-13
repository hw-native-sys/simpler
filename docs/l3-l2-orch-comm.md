# L3-L2 Orchestrator Communication

L3-L2 Orchestrator Communication lets an L3 Host Orchestrator exchange payload
bytes and signal counters with a running L2 AICPU Orchestrator task.

This page documents the low-level region, payload, and counter primitives. For
the ordered SPSC message queue wrapper built on these primitives, see
[l3-l2-message-queue.md](l3-l2-message-queue.md).

The intended use case is in-flight interaction: L3 can write input payload,
publish a data-ready counter, wait for L2/AICore completion, and read output
payload without ending the L2 orchestration task. For where L3 and L2 sit in
the runtime stack, see
[hierarchical_level_runtime.md](hierarchical_level_runtime.md). For dynamic
cross-rank communication domains, see [comm-domain.md](comm-domain.md).

## 1. API

L3 creates a GM communication region for one chip worker:

```python
region = orch.create_l3_l2_region(
    worker_id=0,
    payload_bytes=payload_bytes,
    counter_bytes=128,
)

data_ready = region.counter(0)
completion = region.counter(64)

l2_args = TaskArgs()
for value in region.descriptor_scalars():
    l2_args.add_scalar(value)
l2_args.add_scalar(0)   # data_ready counter offset
l2_args.add_scalar(64)  # completion counter offset

orch.submit_next_level(l2_handle, l2_args, cfg, worker=0)

region.payload_write(input_offset, host_input)
data_ready.notify(seq, NotifyOp.Set)

completion.wait(seq, WaitCmp.GE, timeout=timeout_s)
region.payload_read(output_offset, host_output)

region.free()
```

The L3 handle exposes `descriptor_scalars`, `payload_write`, `payload_read`,
`counter(offset)`, and `free`. Payload operations copy contiguous bytes between
the caller's Host buffer and the region payload range. The steady-state data
path runs in the L3 Host process on both simulation and onboard platforms.
Simulation maps a shared POSIX backing object. Onboard imports the child-owned
GM region with ACL IPC and moves bytes with ACL copy operations. Counter
handles expose `notify`, `test`, and `wait` over address-based `int32_t`
counters.

On L2, orchestration code consumes the descriptor and builds an endpoint:

```cpp
L3L2OrchEndpoint ep(desc);

uint64_t data_ready_addr = 0;
uint64_t completion_addr = 0;
ep.counter_addr(data_ready_offset, data_ready_addr);
ep.counter_addr(completion_offset, completion_addr);

int32_t observed = 0;
bool ok = ep.signal_wait(
              data_ready_addr, seq, L3L2OrchWaitCmp::GE, timeout, observed) &&
          ep.payload_read(input_offset, input_nbytes, input) &&
          ep.payload_read(output_offset, output_nbytes, output);

// The wrapper combines gm_addr/nbytes with task-level dtype and shape.
launch_aicore(input, output);
wait_aicore_done();
ep.signal_notify(completion_addr, seq, L3L2OrchNotifyOp::Set);
```

The L2 endpoint `signal_wait` timeout argument is in nanoseconds. The Python
`counter.wait(..., timeout=timeout_s)` API takes seconds and converts them to
nanoseconds before issuing the operation.

`payload_read` on L2 returns a GM view. It does not copy tensor bytes. Large
outputs should be written directly by AICore into an output view in the region.
L2 `payload_write` is byte-oriented and intended for small metadata or status
payloads, not as the primary large-output path. Payload access is not PTO-ISA
`TLOAD` or `TSTORE`; typed tensor/tile operations remain wrapper or AICore
concerns.

## 2. Region Descriptor

The descriptor is byte-oriented and is encoded as six `uint64_t` scalar slots:

```text
scalar[i + 0] = magic_version
scalar[i + 1] = region_id
scalar[i + 2] = payload_base
scalar[i + 3] = payload_bytes
scalar[i + 4] = counter_base
scalar[i + 5] = counter_bytes
```

| Field | Meaning |
| ----- | ------- |
| `magic_version` | Descriptor ABI marker and version. |
| `region_id` | Region identifier; `0` is reserved as invalid. |
| `payload_base` | Base GM address of the payload byte range. |
| `payload_bytes` | Size of the payload byte range. |
| `counter_base` | Base GM address of the signal counter range. |
| `counter_bytes` | Size of the signal counter range. |

The descriptor deliberately does not contain dtype, shape, stride, tensor rank,
tile layout, stream header layout, queue layout, or semantic counter names.
Wrappers pass those through task arguments or their own protocol fields.

The payload range is:

```text
payload_base .. payload_base + payload_bytes - 1
```

The counter range is:

```text
counter_base .. counter_base + counter_bytes - 1
```

`counter_base` is 64-byte aligned, and `counter_bytes` is a multiple of
`sizeof(int32_t)`. Counter addresses must be 4-byte aligned and inside the
registered counter range. The payload and counter ranges do not overlap.
`payload_base == 0` and `counter_base == 0` are not validity sentinels; only
the range sizes, alignment, overflow, overlap, and reserved `region_id == 0`
checks determine descriptor validity.

## 3. Control Path

The lifecycle control path carries region create/release commands over the
existing worker mailbox. Create returns the unchanged six-scalar L2 descriptor
plus L3 Host-private transport metadata. Release is deferred until submitted L2
work has drained.

Steady-state `payload_write`, `payload_read`, `SIGNAL_NOTIFY`, `SIGNAL_TEST`,
and `SIGNAL_WAIT` run in the L3 Host against parent-private direct-access
metadata. The descriptor still contains L2-side payload and counter addresses
for the L2 AICPU endpoint; L3 Host operations do not dereference descriptor
addresses.

On onboard platforms, region create allocates one child-owned GM range, exports
it with ACL IPC, authorizes the L3 Host PID, and returns the export metadata to
the parent. The parent imports that region and closes the ACL IPC import before
the child frees the physical allocation.

## 4. Signal Counters

Signal primitives are address-based `int32_t` counter operations. The bottom
layer does not assign directions or names to counters. A wrapper can choose
offsets such as `0` for `data_ready` and `64` for `completion`.

L3 counter handles expose:

```python
counter.notify(value, NotifyOp.Set)
counter.notify(delta, NotifyOp.Add)
result = counter.test(cmp_value, WaitCmp.GE)
observed = counter.wait(cmp_value, WaitCmp.GE, timeout=timeout_s)
```

L2 endpoint methods expose the same primitive shape over explicit GM counter
addresses:

```cpp
ep.signal_notify(counter_addr, value, L3L2OrchNotifyOp::Set);
ep.signal_test(counter_addr, cmp_value, L3L2OrchWaitCmp::GE, result);
ep.signal_wait(
    counter_addr, cmp_value, L3L2OrchWaitCmp::GE, timeout, observed);
```

`NotifyOp` supports:

- `Set`: store the operand as the new counter value.
- `Add`: add the operand to the current counter value.

`Add` is a convenience read-modify-write operation, not an atomic operation.
On simulation, L3 counter reads and writes access the parent-side mapping
directly. On onboard, L3 counter reads and writes use 4-byte ACL copy
operations through the imported GM pointer; the parent does not CPU-dereference
imported GM counters. The primitive layer requires only 4-byte counter-address
alignment and does not force 64-byte counter offsets. This leaves room for one
writer to pack a batch of related counters into the same cache line when that
is the right protocol shape.

The ownership invariant is still single-writer per 64-byte cache line: all
counters in one cache line must have the same writer. Counters written by
different L3/L2 agents must not share one cache line. Wrappers should lay out
high-frequency or cross-agent signals on separate 64-byte lines.

`WaitCmp` supports `EQ`, `NE`, `GT`, `GE`, `LT`, and `LE`.
`counter.test(...)` / `signal_test` is a non-blocking snapshot, not a
zero-timeout wait. The primitive reads the current counter once, evaluates the
comparison, and returns immediately. On L3, it returns
`SignalTestResult(matched, observed)`. On L2, `signal_test` fills the endpoint
result with the same `matched` and observed counter fields.

For `test`, `matched` reports whether `observed` satisfies the requested
comparison. A mismatch is a normal polling result: it does not wait, does not
time out, and does not poison the region. Use `wait(..., timeout=positive)`
when the caller needs to block until the comparison matches or the timeout
expires.

At the abstract API level, `signal_notify` is the publish point for writes
ordered before the notify. A matched `signal_test` or `signal_wait` is the
observe point before reading protected data. A failed `signal_test` does not
establish observe semantics.

That contract is not provided by Host-side `atomic_thread_fence` calls alone.
Cross-agent payload and counter visibility comes from successful parent-side
direct transfers, backend DMA ordering, endpoint cache maintenance on onboard
builds, and wrapper-level sequencing that publishes only after prior writes are
complete.

Primitive signal code does not impose sequence monotonicity and does not treat
`observed > cmp_value` as a protocol error. Queue, stream, or channel wrappers
may still store sequence numbers in counters and enforce wrapper-level
protocol rules separately. For monotonic `Add` or sequence counters, wrappers
should normally wait with `GE` or `GT`; `EQ` is appropriate only when the
wrapper guarantees the target value cannot be skipped.

Counter reset is expressed as `notify(0, NotifyOp.Set)`.

## 5. Ordering

A single stream-style round can follow this order:

```text
L3:
  payload_write(input_offset, host_input)
  data_ready.notify(seq, Set)

L2:
  signal_wait(data_ready_addr, seq, GE)
  payload_read(input_offset, input_nbytes, input_view)
  payload_read(output_offset, output_nbytes, output_view)
  submit AICore(input_view, output_view)
  wait for AICore completion
  signal_notify(completion_addr, seq, Set)

L3:
  completion.wait(seq, GE)
  payload_read(output_offset, host_output)
```

The `EQ` versus `GE` choice is a wrapper decision. Prefer `GE`/`GT` for
monotonic sequence or `Add` counters, because `EQ` can miss a target if the
counter steps past it before the waiter observes the value. The primitive layer
only applies the requested comparison.

On onboard builds, correct payload/counter visibility depends on the common
`aicpu/cache_maintenance.h` helpers. The aarch64 path emits data-cache
maintenance instructions; sim and non-aarch64 paths are no-ops. See
[hardware/cache-coherency.md](hardware/cache-coherency.md).

All waits must use finite timeouts. Unbounded waits hide protocol deadlocks.

## 6. Lifetime Model

A region belongs to one L3 orchestration execution. It may be reused by multiple
L2 orchestration runs submitted from that execution, but the handle is invalid
after the L3 orchestration function returns.

The handle has three important user-visible states. A live handle allows
payload, counter, descriptor, and `free` operations. A released handle has seen
`free()` and rejects further payload, counter, or descriptor use. A poisoned
handle means progress is no longer trusted; only cleanup remains valid.

Physical GM release is deferred until submitted L2 work that may hold the
descriptor has drained. This allows a region handle to become released in L3
without freeing memory still referenced by an in-flight L2 task.

If an L3 orchestration execution exits with live regions, runtime cleanup marks
them released, drains submitted work, and then releases the physical resources.

## 7. Host Buffer Requirements

L3 payload buffers must be contiguous and large enough for the requested
transfer. On simulation platforms, the buffer only needs to be accessible from
the L3 Host orchestration code: runtime Host tensors, `bytes`, `bytearray`, and
writable contiguous memoryviews for reads are valid according to the
operation's read/write direction.

On onboard platforms, the L3 Host imports the child-owned region through ACL
IPC and copies directly through ACL runtime primitives. Payload buffers must be
ordinary contiguous L3 Host-accessible byte spans according to the operation's
read/write direction.

Small wrapper metadata is payload too. A header such as `{seq, DATA}` or
`{seq, STOP}` is copied through `payload_write`; any staging needed for a
backend is owned by the primitive payload transfer, not by the queue protocol.

## 8. Error Handling

Primitive `SIGNAL_TEST` mismatch is success with `matched = false` and the
current `observed` counter value; it does not poison the primitive region.
Primitive `SIGNAL_WAIT` timeout reports the last observed counter value and
does not poison the primitive region by itself.

Failures that may corrupt payload or region progress poison the corresponding
L3 Host region:

- DMA failure after a payload command is issued;
- signal notify failure;
- L3 Host transfer failure after operation issue;
- direct L3 Host mapped-region failure after a payload or counter primitive is
  issued;
- L2 endpoint fatal error reported with a valid `region_id`;
- explicit wrapper-level poison in future queue or stream abstractions.

Pre-command validation failures do not poison the region:

- malformed API arguments;
- invalid counter offset rejected by `region.counter(offset)`;
- non-contiguous Host tensor;
- unsupported Host buffer detected before command issue;
- out-of-bounds payload offset detected before command issue;
- descriptor extraction after release or poison.

A poisoned region rejects `payload_write`, `payload_read`, counter operations,
and `descriptor_scalars`. `free` and orchestration cleanup remain valid.

L2 endpoint errors own structured fields inside L2, including `region_id`,
operation, counter address, operand, observed counter, kind, and message. The
current L3 Host poisoning path does not receive structured cross-level fatal
metadata; it depends on the wrapper's canonical fatal text format:
`L3-L2 endpoint error ... region=<id>`. When multiple live regions exist, the
L3 Host poisons only the region parsed from that text.

## 9. Platform Support

- `a2a3sim`: full API and protocol support.
- `a5sim`: full API and protocol support.
- `a2a3` onboard: full API support with direct L3 Host ACL IPC payload and
  counter operations.
- `a5` onboard: full API support with direct L3 Host ACL IPC payload and
  counter operations.

Simulation backends preserve the same API, ordering, timeout, and error
semantics as onboard backends.

## 10. Wrapper Example

The bottom layer does not define stream headers, opcodes, tensor schema, rings,
or stop semantics. A streaming wrapper can reserve payload bytes for that
protocol and reserve counter offsets for its own synchronization names.

For example, a wrapper can reserve payload offset `0..63` as a channel header,
followed by input and output tensor slices. It can use counter offset `0` for
`data_ready` and counter offset `64` for `completion`. One round writes
`{seq, DATA}`, publishes `seq` through `data_ready`, waits until
`completion >= seq`, and reads the output slice. Teardown can write
`{seq + 1, STOP}` and publish it through `data_ready`.

The related example lives in
`examples/workers/l3/l3_l2_orch_comm_stream` and is marked for `a2a3sim`,
`a2a3`, `a5sim`, and `a5`. It creates one region, submits one persistent L2
orchestration task, and drives three DATA
rounds from L3 while the L2 task stays in flight. Each round copies a
`float32[128 * 128]` input slice and a small channel header into the region;
L2 builds input/output GM tensor views from the descriptor, launches an AIV
kernel that adds a scalar to the input, publishes completion, and L3 reads back
and checks the output. The final STOP header lets L2 return, so `Worker.run`
drain acts as the acknowledgement without requiring a third counter.
