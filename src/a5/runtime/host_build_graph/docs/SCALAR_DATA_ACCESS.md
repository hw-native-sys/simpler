# Scalar Data Access During Host Graph Construction

`host_build_graph` runs the orchestration function synchronously on the host,
before that run's AICPU scheduler or AICore kernels start. `get_tensor_data`
and `set_tensor_data` access ready external inputs through host views or
child-memory scalar copies. An eligible successor may build while its
predecessor executes; the caller owns ordering of conflicting accesses.

## Supported Uses

| Tensor state | `get_tensor_data` | `set_tensor_data` |
| ------------ | ----------------- | ----------------- |
| Ready host-backed `IN` / `INOUT`, with no submitted producer | Reads the caller's host view | Updates the host view and device staging |
| Ready child-memory argument, with no submitted producer | Uses a host mapping or device copy | Uses a host mapping or device copy |
| Host-backed pure `OUT` argument | Fails with `INVALID_ARGS` | Fails with `INVALID_ARGS` |
| External tensor a submitted task writes (`OUTPUT`/`INOUT`) | Fails with `INVALID_ARGS` | Fails with `INVALID_ARGS` |
| Output of a submitted task | Fails with `INVALID_ARGS` | Fails with `INVALID_ARGS` |
| Runtime allocation (`alloc_tensors`) | Fails with `INVALID_ARGS` | Fails with `INVALID_ARGS` |
| Tensor with an invalid or stale owner task ID | Fails with `INVALID_ARGS` | Fails with `INVALID_ARGS` |

A host-backed pure `OUT` has no input copy or registered host view during bind.
For supported inputs, a host write changes the value the graph will consume;
submit order does not turn the write into a barrier between kernels.

## API

```cpp
uint32_t index[1] = {0};

int32_t value = get_tensor_data<int32_t>(control, 1, index);
set_tensor_data<int32_t>(layout, 1, index, value + 1);
```

Both tensors must be ready external inputs with host views or child memory. A
common use is to read an input control value or publish runtime geometry into an
external layout tensor that no submitted task owns.

If a previous run produces the value, finish its handle before submitting the
consumer whose host construction needs it:

```python
producer = worker.submit(producer_callable, producer_args)
producer.result()
consumer = worker.submit(consumer_callable, consumer_args)
consumer.wait()
```

Successful completion must include any required copy-back for host-backed
outputs. For child memory, wait for the device writer and retain the allocation;
there is no host staging copy-back. The same rule protects a predecessor's
readers from a successor's host write. Independent and shared read-only inputs
can prepare early when the backend supports it. Tensor tags do not supply an
implicit cross-run accessor wait.

## Why This Run's Device-Produced Values Cannot Be Read Here

Within one run, the execution order is:

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

- Use scalar access on ready host-backed inputs or child memory that no task in
  the current graph produces. Include aliases when checking for conflicting uses.
- Use tensor dependencies to order device tasks; do not use host scalar access
  as a device synchronization barrier.
- Pass values needed for graph construction as orchestration inputs or scalars.
- Complete the producer run before building a consumer that needs its results.
