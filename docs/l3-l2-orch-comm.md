# L3-L2 Orchestrator Communication

L3-L2 Orchestrator Communication lets an L3 Host Orchestrator exchange tensor
payloads and signals with a running L2 AICPU Orchestrator task.

The intended use case is in-flight interaction: L3 can write input payload,
notify L2, wait for L2/AICore completion, and read output payload without
ending the L2 orchestration task. For where L3 and L2 sit in the runtime stack,
see [hierarchical_level_runtime.md](hierarchical_level_runtime.md). For dynamic
cross-rank communication domains, see [comm-domain.md](comm-domain.md).

## 1. API

L3 creates a GM communication region for one chip worker:

```python
region = orch.create_l3_l2_region(worker_id=0, payload_bytes=nbytes)

l2_args = TaskArgs()
for value in region.descriptor_scalars():
    l2_args.add_scalar(value)

orch.submit_next_level(l2_handle, l2_args, cfg, worker=0)

region.payload_write(input_offset, host_input)
region.notify(seq)

region.wait(seq, timeout=timeout_s)
region.payload_read(output_offset, host_output)

region.free()
```

The L3 handle exposes `descriptor_scalars`, `payload_write`, `payload_read`,
`notify`, `wait`, and `free`. Payload operations copy contiguous bytes between
child-visible Host tensors and the region payload range. Signal operations
publish and wait on monotonically increasing sequence values.

On L2, orchestration code consumes the descriptor and builds an endpoint:

```cpp
L3L2OrchEndpoint ep(desc);

PayloadView input{};
PayloadView output{};

bool ok = ep.wait(seq, timeout) &&
          ep.payload_read(input_offset, input_nbytes, &input) &&
          ep.payload_read(output_offset, output_nbytes, &output);

// The wrapper combines gm_addr/nbytes with task-level dtype and shape.
launch_aicore(input, output);
wait_aicore_done();
ep.notify(seq);
```

`payload_read` on L2 returns a GM view. It does not copy tensor bytes. Large
outputs should be written directly by AICore into an output view in the region.
L2 `payload_write` is byte-oriented and intended for small metadata or status
payloads, not as the primary large-output path.

## 2. Region Descriptor

The descriptor is byte-oriented and is encoded as six `uint64_t` scalar slots:

```text
scalar[i + 0] = magic_version
scalar[i + 1] = region_id
scalar[i + 2] = payload_base
scalar[i + 3] = payload_bytes
scalar[i + 4] = l3_to_l2_signal_base
scalar[i + 5] = l2_to_l3_signal_base
```

| Field | Meaning |
| --- | --- |
| `magic_version` | Descriptor ABI marker and version. |
| `region_id` | Region identifier for diagnostics and region-scoped errors. |
| `payload_base` | Base GM address of the payload byte range. |
| `payload_bytes` | Size of the payload byte range. |
| `l3_to_l2_signal_base` | Signal slot written by L3 and waited on by L2. |
| `l2_to_l3_signal_base` | Signal slot written by L2 and waited on by L3. |

The descriptor deliberately does not contain dtype, shape, stride, tensor rank,
tile layout, stream header layout, or ring layout. Wrappers pass those through
task arguments or their own protocol fields.

The payload range is:

```text
payload_base .. payload_base + payload_bytes - 1
```

Signal slots are separate from the payload range. `payload_read` and
`payload_write` operate only on payload bytes.

## 3. Communication Model

The control path carries descriptors and completion status. Tensor payload bytes
do not flow through the control path. L3 payload operations copy between
child-visible Host tensor storage and Device GM; L2 payload operations expose GM
views that orchestration code can pass to runtime tensor construction helpers.

There are two directional signal slots:

```text
l3_to_l2_signal_base  # L3 notify, L2 wait
l2_to_l3_signal_base  # L2 notify, L3 wait
```

Each signal slot stores one little-endian `uint64_t` sequence number. The
initial value is zero. Valid sequence numbers start at one and increase
monotonically for the region lifetime.

Signal comparison is strict:

```text
notify(seq): store exactly seq
wait(seq):
  current == seq -> success
  current < seq  -> keep waiting until timeout
  current > seq  -> protocol error
```

The current contract supports one outstanding transfer per region. L3 must not
overwrite the input slice for `seq + 1` until L2 has notified completion for
`seq`.

## 4. Ordering

A single round follows this order:

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

`notify(seq)` publishes previously completed payload writes on the same region.
It does not publish writes on other regions or writes issued after the notify.
`wait(seq)` is the acquire point before reading data for that sequence.

All waits must use finite timeouts. Unbounded waits hide protocol deadlocks.

## 5. Lifetime Model

A region belongs to one L3 orchestration execution. It may be reused by multiple
L2 orchestration runs submitted from that execution, but the handle is invalid
after the L3 orchestration function returns.

The handle has three important user-visible states. A live handle allows
payload, signal, descriptor, and `free` operations. A released handle has seen
`free()` and rejects further payload, signal, or descriptor use. A poisoned
handle means protocol progress is no longer trusted; only cleanup remains valid.

Physical GM release is deferred until submitted L2 work that may hold the
descriptor has drained. This allows a region handle to become released in L3
without freeing memory still referenced by an in-flight L2 task.

If an L3 orchestration execution exits with live regions, runtime cleanup marks
them released, drains submitted work, and then releases the physical resources.

## 6. Host Buffer Requirements

L3 payload buffers must be contiguous and visible to the chip child process, so
the child runtime can DMA directly between Host DRAM and Device GM.

Supported sources depend on the active runtime, but the contract is:

- the buffer must be child-visible before the payload command is issued;
- the byte span must be contiguous and large enough for the requested transfer;
- temporary Python bytes, ordinary private tensors, and unmapped post-fork
  buffers are not valid payload sources.

Small wrapper metadata is payload too. A header such as `{seq, DATA}` or
`{seq, STOP}` must be stored in a valid Host buffer and copied through
`payload_write`.

## 7. Error Handling

Timeouts and protocol errors are region-scoped. Failures that may corrupt
protocol progress poison the corresponding Host-side region:

- L3 or L2 signal wait timeout;
- DMA failure after a payload command is issued;
- signal notify failure;
- control-service fatal error after the region is live;
- signal protocol error such as observing `current > seq`.

Pre-command validation failures do not poison the region:

- malformed API arguments;
- non-contiguous Host tensor;
- child-invisible Host buffer detected before command issue;
- out-of-bounds payload offset detected before command issue;
- descriptor extraction after release or poison.

A poisoned region rejects `payload_write`, `payload_read`, `notify`, `wait`, and
`descriptor_scalars`. `free` and orchestration cleanup remain valid.

L2 endpoint errors carry structured metadata including `region_id`, operation,
sequence, kind, and message. When multiple live regions exist, the Host poisons
only the region identified by that endpoint metadata.

## 8. Platform Support

- `a2a3sim`: full API and protocol support.
- `a5sim`: full API and protocol support.
- `a2a3` onboard: full API and protocol support.
- `a5` onboard: symbols are present; region operations fail with a clear
  not-supported error.

Simulation backends preserve the same API, ordering, timeout, and poison
semantics as onboard backends.

## 9. Wrapper Example

The bottom layer does not define stream headers, opcodes, tensor schema, rings,
or stop semantics. A streaming wrapper can reserve payload bytes for that
protocol:

For example, a wrapper can reserve offset `0..63` as a channel header, followed
by input and output tensor slices. One round writes `{seq, DATA}`, publishes
`seq`, waits for L2 completion, and reads the output slice. Teardown can write
`{seq + 1, STOP}` and publish it through the normal L3-to-L2 signal.

The related example lives in
`examples/a2a3/tensormap_and_ringbuffer/l3_l2_orch_comm_stream`. It creates one
region, submits one persistent L2 orchestration task, and drives three DATA
rounds from L3 while the L2 task stays in flight. Each round copies a
`float32[128 * 128]` input slice and a small channel header into the region;
L2 builds input/output GM tensor views from the descriptor, launches an AIV
kernel that adds a scalar to the input, notifies L3, and L3 reads back and
checks the output. The final STOP header lets L2 return, so `Worker.run` drain
acts as the acknowledgement without requiring a third signal slot.
