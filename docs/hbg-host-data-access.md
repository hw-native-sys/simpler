# HBG host data access

`host_build_graph` runs orchestration on the host while building the graph,
before that run's device tasks execute. Scalar reads and writes therefore
require explicit host storage that is ready when construction begins. This
contract applies to application calls as well as scene tests, on both A2/A3
and A5.

## Location and transfer

| Argument at bind | Host orchestration access | Device task access |
| ---------------- | ------------------------- | ------------------ |
| `HOST/NONE` | Reads `IN`/`INOUT`; writes `OUT`/`INOUT`, within the registered view and Buffer grant | Rejected |
| `HOST/H2D` | Rejected after bind materializes the argument as `DEVICE/NONE` | Uses device staging |
| `DEVICE/NONE` | Rejected | Uses the supplied device allocation |

`HOST/NONE` registers host view bounds without allocating device storage.
Strided views are supported; overlapping host views with a writer are refused.
Empty views have no readable or writable elements. `get_tensor_data` and
`set_tensor_data` perform no implicit device mapping, H2D or D2H fallback.
Other invalid location/transfer pairs fail during bind preflight.

When both orchestration and device kernels need a control value, supply
separate host-only and device arguments and prepare both explicitly. A host
write does not update a device counterpart. The
[Buffer memory model](buffer-abi.md#why-tensor-and-chiptensor-are-two-types)
describes the shared location, transfer and direction rules.

## Scalar accessors

```cpp
uint32_t index[1] = {0};
int32_t value = get_tensor_data<int32_t>(host_control, 1, index);
set_tensor_data<int32_t>(host_layout, 1, index, value + 1);
```

`host_control` must be a bound `HOST/NONE` input and `host_layout` a bound
`HOST/NONE` output or input/output. An out-of-bounds access, incompatible
direction, device address or tensor with an owner task ID latches
`SIMPLER_ERROR_INVALID_ARGS`. A failed read returns zero; a failed write does
not modify memory. An earlier fatal error remains the reported failure.

## Readiness across runs

The caller establishes readiness before submitting a run whose host
construction consumes those bytes. Complete the producer and any required
device-to-host copy before submitting the consumer. A device-stream fence alone
does not supply a host copy of a device value. Order conflicting host writes
against earlier readers as well. Shared read-only host inputs can remain
available to independent constructions.

## Values produced by the graph

The current graph has not executed during construction. Its device-produced
values and graph-heap allocations cannot be read or written through these host
accessors; they are device storage without a registered host view. Use device
task dependencies for those values. HBG's `TensorCreateInfo` does not expose
`set_initial_value`; initialize a runtime allocation with a device task.

## Implementation and diagnostics

The shared [scalar accessor implementation](../src/common/host_build_graph/host/runtime_core.cpp)
requires a nonempty, owner-free HOST tensor and checks its registered bounds
and direction through the host tensor access helpers. The bind registers
`HOST/NONE` arguments before device-content copying.

The [`host_view_close` diagnostic](dfx/hbg-bind-phases.md) reports release of
those registrations; it performs no device mapping or copy fallback. Historical
mapped-view and `devcopy` measurements describe an earlier contract and must
not be used to estimate the current access path.
