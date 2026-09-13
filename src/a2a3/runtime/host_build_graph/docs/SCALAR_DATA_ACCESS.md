# Scalar Data Access During Host Graph Construction

`host_build_graph` runs the orchestration function synchronously on the host,
before any AICPU scheduler or AICore kernel starts. `get_tensor_data` and
`set_tensor_data` therefore access the host view the graph's external tensors
were copied in from; they do not interleave host code with device execution.

## Supported Uses

| Tensor state | `get_tensor_data` | `set_tensor_data` |
| ------------ | ----------------- | ----------------- |
| External tensor with no submitted producer | Reads the copied-in host value | Updates the copied-in host value |
| External control/output tensor not referenced by a task | Reads immediately | Writes immediately |
| External tensor a submitted task writes (`OUTPUT`/`INOUT`) | Fails with `INVALID_ARGS` | Fails with `INVALID_ARGS` |
| Output of a submitted task | Fails with `INVALID_ARGS` | Fails with `INVALID_ARGS` |
| Runtime allocation (`alloc_tensors`) | Fails with `INVALID_ARGS` | Fails with `INVALID_ARGS` |
| Tensor with an invalid or stale owner task ID | Fails with `INVALID_ARGS` | Fails with `INVALID_ARGS` |

The supported write changes the data that will be copied to the device. Every
task in the graph observes that final copied-in value; submit order does not turn
the write into a barrier between kernels.

## API

```cpp
uint32_t index[1] = {0};

int32_t value = get_tensor_data<int32_t>(control, 1, index);
set_tensor_data<int32_t>(layout, 1, index, value + 1);
```

Both tensors in this example must be external tensors the host copied in. A
common use is to read an input control value or publish runtime geometry into an
external layout tensor that no submitted task owns.

## Why Device-Produced Values Cannot Be Read Here

The execution order is:

1. The host loads and calls the orchestration shared object.
2. Orchestration builds the entire task graph and returns.
3. The host copies the graph image to device memory.
4. AICPU schedulers boot and dispatch the graph.

A producer submitted in step 1 cannot become `COMPLETED` until step 4, so a read
of its output would see the buffer's pre-run content and a write would be
overwritten by the producer itself. Both accessors therefore reject a tensor
with a producer outright — there is no wait and no timeout.

A runtime allocation is rejected on the same rule. Its creator completes on the
host, but the buffer is uninitialized, lives in the graph heap, and has no
host-view registration.

## No Initial-Value Fill on a Runtime Allocation

`TensorCreateInfo` carries no `set_initial_value` here, unlike its
`tensormap_and_ringbuffer` counterpart. The fill stores to
`ChipTensor::buffer.addr` — a GM-heap device address — which the AICPU
orchestrator can write and the host orchestrator this runtime uses cannot. The
method is absent rather than failing at run time, so an orchestration that asks
for it does not compile against this runtime instead of faulting on device.

To give a runtime allocation a defined starting content — a fixed-size tile
whose producer writes only a prefix, so its consumer reads a known value in the
remainder — have a task write it. Doing it on device also keeps the buffer
correct under Graph Execution, where a value written once while recording would
reach none of the replays: each submission materializes its outputs at addresses
it derives for itself, from a heap block whose prior contents it never reads.

## Producer Rejection

A producer reaches a tensor two ways, and either one rejects the call:

- the tensor names a creator in `owner_task_id` — the task that allocated it,
  whether an ordinary submit or `alloc_tensors`;
- an entry in the TensorMap names a task that wrote a region overlapping this
  one, which is how an external tensor passed as `OUTPUT`/`INOUT` acquires a
  producer.

An invalid or stale owner ID is rejected by the first rule like any other
producer, so a forged ID cannot reach a task-table slot. A rejection latches
`SIMPLER_ERROR_INVALID_ARGS` and names the producer task; the run returns status
`-5`, reads return zero, and writes do not happen.

## Practical Rules

- Use scalar access only on external, host-memory tensors that no submitted task
  produces.
- Use tensor dependencies to order device tasks; do not use host scalar access
  as a device synchronization barrier.
- Pass values needed for graph construction as orchestration inputs or scalars.
- Keep device-produced values on the device or return them after the run.
