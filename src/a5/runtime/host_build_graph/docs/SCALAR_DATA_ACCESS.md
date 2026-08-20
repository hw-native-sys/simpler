# Scalar Data Access During Host Graph Construction

`host_build_graph` runs the orchestration function synchronously on the host,
before any AICPU scheduler or AICore kernel starts. `get_tensor_data` and
`set_tensor_data` therefore access the host view used to stage the graph's
external tensors; they do not interleave host code with device execution.

## Supported Uses

| Tensor state | `get_tensor_data` | `set_tensor_data` |
| ------------ | ----------------- | ----------------- |
| External tensor with no submitted producer | Reads the staged host value | Updates the staged host value |
| External control/output tensor not referenced by a task | Reads immediately | Writes immediately |
| Runtime-created output | Unsupported: no registered host view | Unsupported: no registered host view |
| Tensor owned by a submitted device task | Unsupported during graph construction | Unsupported during graph construction |
| Tensor with an invalid or stale owner task ID | Fails with `INVALID_ARGS` | Fails with `INVALID_ARGS` |

The supported write changes the data that will be copied to the device. Every
task in the graph observes that final staged value; submit order does not turn
the write into a barrier between kernels.

## API

```cpp
uint32_t index[1] = {0};

int32_t value = get_tensor_data<int32_t>(control, 1, index);
set_tensor_data<int32_t>(layout, 1, index, value + 1);
```

Both tensors in this example must be external tensors staged by the host. A
common use is to read an input control value or publish runtime geometry into an
external layout tensor that no submitted task owns.

## Why Device-Produced Values Cannot Be Read Here

The execution order is:

1. The host loads and calls the orchestration shared object.
2. Orchestration builds the entire task graph and returns.
3. The host copies the graph image to device memory.
4. AICPU schedulers boot and dispatch the graph.

A producer submitted in step 1 cannot become `COMPLETED` until step 4. Waiting
for that producer from the orchestration call cannot make progress. The runtime
keeps a timeout as a defensive failure backstop, but it is not a supported
synchronization mechanism.

Runtime-created output buffers also live in the graph heap and have no host-view
registration. Even if their address is nonzero, host orchestration must not
dereference or modify them.

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

## Ownership Validation

Before a wait slot is used, the runtime verifies:

- the task ID is valid and belongs to the single HBG ring;
- the selected ring slot has a bound task descriptor; and
- the descriptor's full task ID matches the tensor's owner/producer ID.

The full-ID check prevents a masked ring-slot lookup from aliasing an unused or
different task. A failure latches `PTO2_ERROR_INVALID_ARGS` and the run returns
status `-5`; reads return zero and writes stop only after that fatal status is
recorded.

## Practical Rules

- Use scalar access only on external, host-staged tensors with no device
  producer or outstanding device consumer.
- Use tensor dependencies to order device tasks; do not use host scalar access
  as a device synchronization barrier.
- Pass values needed for graph construction as orchestration inputs or scalars.
- Keep device-produced values on the device or return them after the run.
