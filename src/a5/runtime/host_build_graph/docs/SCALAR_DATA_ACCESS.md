# Scalar Data Access During Host Graph Construction

`host_build_graph` builds the graph synchronously on the host before its device
tasks execute. Both architectures use the shared
[HBG host data access contract](../../../../../docs/hbg-host-data-access.md).

## Supported Uses

Only explicit `HOST/NONE` arguments are eligible: `get_tensor_data` reads
`IN`/`INOUT`, and `set_tensor_data` writes `OUT`/`INOUT`, within the registered
bounds and Buffer grant. `HOST/H2D` becomes device storage at bind; it and
`DEVICE/NONE` are refused by the host scalar accessors. Empty views have no
readable or writable elements. Device values require an explicit copy before
submission when graph construction needs their contents.

## API

```cpp
uint32_t index[1] = {0};
int32_t value = get_tensor_data<int32_t>(host_control, 1, index);
set_tensor_data<int32_t>(host_layout, 1, index, value + 1);
```

The read argument must be a host-only input and the write argument a host-only
output or input/output. Host writes do not update a device counterpart; use
separate arguments for host construction and device execution.

## Why This Run's Device-Produced Values Cannot Be Read Here

The host completes orchestration and uploads the graph before AICPU schedulers
dispatch its device tasks. Their outputs and graph-heap allocations have no
registered host view during construction. Host scalar access cannot wait for
those producers or initialize their device storage.

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

The scalar accessors require an owner-free HOST tensor. A device tensor, an
owner task ID, an incompatible direction or an out-of-bounds access latches
`SIMPLER_ERROR_INVALID_ARGS`. Failed reads return zero and failed writes do
not modify memory; an earlier fatal error remains the reported failure.

## Practical Rules

- Supply explicit `HOST/NONE` controls for graph construction.
- Use device task dependencies for values produced within the graph.
- Complete any earlier producer and the required explicit host copy before
  submitting a construction that consumes those bytes.
- Order conflicting host writes against earlier readers as well.
- See the shared contract for location, transfer and readiness details.
