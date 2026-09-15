# HBG kernel resource declarations

Reference: the supplied `kernel-mode-design.md`, [v9 final decisions](https://icc.gt.tc/vllm-pto?i=2#v9-design).
The final decisions in §0 override historical alternatives, except for the user's subsequent three-stream clarification: caller, dedicated non-hidden AICPU and hidden AICore are distinct.
These are internal host C++ interfaces, not new public workspace-size APIs.
The integrated K1 C entry points and `SimplerKernelInvocationHeader` remain the
shared ABI. The common header has no version or header-size negotiation; packet
offsets use `sizeof(SimplerKernelInvocationHeader)`. HBG's graph header and slot
records have their own format versions. Resource control uses internal C++
methods, without adding a competing public context-control entry point.

## Host build and program upload

`hbg::build_graph` performs Host orchestration against already-staged arguments.
It neither commits device execution regions nor uploads the graph.
`GraphBuild` owns orchestration and Definition records but borrows the SM mirror,
Definition staging and matching runtime workspace. These buffers must remain
leased until the result is no longer used; a second build cannot reuse them
while the first result is still being consumed. Build and upload require
exclusive workspace access. Concurrent read-only queries may share a completed
build, but must not overlap build or upload.

`hbg::upload_for_program_mode` is the explicit program-only allocation and
synchronous H2D boundary. It creates a compact image from the virtual-address
source on each upload. If Definition staging grows and moves, the upload owner
preserves its contents and rebinds the build's staging reference before further
processing, including before a potentially failing copy. Repeated upload and
retry therefore keep a valid source. Program execution still calls build and
upload in sequence and retains its existing resource management.

This borrowed intermediate is consumed by `make_graph_launch_template`, which
produces an independently owned immutable packet. Kernel submission uses that
packet and a prepared working slot, never the program upload path. Device
admission checks the independently registered slot and trusted callable metadata.
The leader restores the packet into that slot before dispatch; see
[execution-slot registration and restoration](host-build-graph-kernel-slot.md).

## Graph requirements and context capacity

After a successful build, call
`hbg::get_graph_resource_requirements(build, layout, requirements)` with the
layout of the intended destination for the same task window and runtime ABI.
Program mode uses its existing layout; kernel mode uses
`make_kernel_graph_layout(build.workspace.task_capacity, layout)`. The query does no device
allocation, address binding or H2D. H1 has already validated and measured the
Definition plan, so the query reads the immutable measurement and does not scan
or mutate the borrowed Definition staging.

`GraphResourceRequirements` is an independent value snapshot for one graph:

| Field | Meaning |
| ----- | ------- |
| `gm_heap_bytes` | Measured graph heap including Graph execution storage, rounded to arena alignment |
| `runtime_arena_bytes` | Device-only prefix, RuntimeContext and compact SM tail; SM is counted once |
| `graph_definition_bytes` | Used retained prefix plus aligned spill objects and framing; distinct Definitions counted once |
| `scheduler_state_bytes` | A5 scheduler allocation upper bound including alignment slack; zero on A2/A3 and A5 Graph fallback |
| `layout` | Architecture, HBG layout ABI, task window and copied-zone identity used to interpret the image offsets |

`required_bytes()` checks the sum of these logical requirements. It is neither
committed HBM telemetry nor context capacity. Public committed-memory reporting
continues to use `committed_device_memory_ctx`; caller tensors, process-pinned
code and CANN packets are outside this graph calculation.

`KernelResourcePlan::create(graphs, count, out)` builds a capacity declaration
for one serialized execution slot. Inputs must belong to the same context's
architecture and runtime layout ABI. Matching byte totals do not make layouts
compatible: the architecture, task window, arena extent and copied-zone bounds
must all match. Invalid or mixed layout identities return `INVALID_ARGUMENT`
without changing an existing plan. It takes the maximum of each compatible
region requirement, then **recomputes** aligned Definition and scheduler offsets
after the runtime/SM capacity. It then reserves a separate aligned
`GraphSlotRegistry` control region, excluded from every restore image.
It does not merge offsets or take the maximum of
already-packed total sizes. The runtime/SM region accommodates one complete
per-invocation image; its internal offsets must be bound from that invocation's
layout during restore, never combined across graphs.

For example, graph A needs runtime 8192 and Definition 512 bytes; graph B needs
runtime 4096 and scheduler 2048 bytes. The combined arena reserves runtime
`[0, 8192)`, Definition `[8192, 8704)`, padding, scheduler `[9216, 11264)`, and registry `[11264, 11520)`.
This is a legal layout for either graph; the larger individual packed total
alone would not describe these combined region capacities.

`plan.admits(graph)` compares every region against capacity without changing the
plan. Exact capacity is accepted; an excess in any region is rejected even if
another region has unused bytes. Queries and plan construction publish outputs
only on success. Overflow returns `CAPACITY_EXCEEDED`; invalid build state or
arguments fail before any resource mutation.

`plan.prepare(context, resource_ops)` passes this layout to
`KernelExecutionState::prepare_resources`. The context owns the actual device
allocations through `KernelDeviceResources`: one heap and one packed runtime
arena. Definition, scheduler and registry destinations are disjoint slices of that arena, so this
path never calls the program allocator's `acquire_graph_definition_block` or
its A5 per-run scheduler allocator. Base alignment and allocation-size overflow
are checked before any device allocation. The common layout validator also
rejects overlapping and out-of-bounds regions.

Preparation is once per execution slot. Repeating prepare with compatible
smaller requirements reuses both the old addresses and old offsets. A larger
requirement is rejected even before freeze; collect the intended capacities
before preparing the slot. Partial allocation failure releases the candidate;
if cleanup itself fails, the context enters `CLOSING`, retains remaining
allocations and rejects dispatch until explicit close retries succeed.

`context.freeze_resources()` is a distinct transition, accepted only after
successful resource preparation. `bind_kernel_resources_for_launch` then checks
ready state, device/context-slot identity, the HBG region schema and each required
size before returning `KernelWorkingBinding`. It does not allocate, free, copy,
clear or replace any buffer. Mutable state must be restored later by the device
from the invocation's immutable source. Caller must serialize binding/enqueue
with close and establish external quiescence before closing.

```cpp
// All known graph requirements have already been collected outside capture.
hbg::KernelResourcePlan plan;
// Check each returned status before proceeding.
hbg::KernelResourcePlan::create(graphs, graph_count, plan);
plan.prepare(context, KernelResourceOps::from_allocator(allocator));
context.freeze_resources();
hbg::prepare_graph_execution_slot(context, device_id, generation, runtime_binary_id, prepare_ops);
// The AICPU stream's FIFO orders these tasks ahead of every later launch.
context.mark_ready_enqueued();

// Resource portion of each launch: no allocator argument is available here.
hbg::KernelWorkingBinding binding;
hbg::bind_kernel_resources_for_launch(context, device_id, generation, graph, binding);
```

The allocator adapter uses the platform `MemoryAllocator`, preserving existing
committed-byte accounting (including alignment slack). It must remain alive
until explicit context close; neither its destructor nor program teardown may
run while captured graphs reference the context. Context close releases only
its own allocations, never caller tensors. External workspace injection remains
deferred. Expanding a captured context requires a new context slot. This
internal slot identity is separate from callable registration: current K1 mints
an `int32_t callable_id` in `prepare_callable` and carries no callable generation
in the 32-byte invocation header.

HBG's public kernel launch remains unsupported. The resource lifecycle and
immutable HBG packet producer are implemented internally, and H4 restores the
pristine packet into the prepared slot before dispatch. H5 exports the dedicated
HBG kernel registration entry and symbol manifest. Public HBG owner integration
is still required before enabling execution. Resource freeze is the internal
`context.freeze_resources()` transition; the HBG owner must connect it to the
preparation lifecycle. The TMR public launch implementation has its own
independent admission path.

## Common contract and stream roles

`plan.pipeline_contract()` projects the complete capacity into existing kinds:

| Kind | Class | Bytes per copy / binding |
| ---- | ----- | ------------------------ |
| `GM_HEAP` | `HOST_PER_RUN` | Heap capacity |
| `RUNTIME_IMAGE` | `HOST_PER_RUN` | Packed runtime/SM + Definition + scheduler + registry capacity, including padding |
| `AICPU_STREAM` | `EXEC_HANDLE` | Zero bytes; dedicated non-hidden AICPU stream |
| `AICORE_STREAM` | `EXEC_HANDLE` | Zero bytes; use context-owned hidden AICore stream |

Depth is one and fixed for the context. HBG omits `GM_SM` because the SM image is
inside the runtime arena. TMR retains its six-resource declaration and scratch
classes; neither runtime uses `TASK_ARGS.bytes_per_copy` as a snapshot-size API.
There are three physical streams. Caller belongs to vLLM Ascend/the framework;
AICPU and AICore each have their own execution stream. The user explicitly
supersedes the v9 decision that placed AICPU work on caller. Event record/wait
must connect both execution branches to caller's entry and exit boundaries.

Validation retains the shared layers: ABI/mode/byte validation, serviceable arena
and stream topology, then runtime-specific resource-set checks.
`is_valid_hbg_kernel_pipeline_contract` checks HBG's four-resource set,
classes and total-size bounds. `bind_kernel_stream_roles` validates the arena
and stream topology and binds three non-null, distinct handles, preserving its
output on error. `KernelLaunchHandles::valid()` also checks these handles at
the shared launch binder. The context retains each
execution stream under `KernelStreamKind`; its creation/destruction callbacks
keep the existing `KernelContextOps` signature. The context event set is
`Start`, `AicoreStart`, `AicoreDone`, `AicpuDone`, and `SerialTail`, chained
caller ⇄ AICPU ⇄ AICore. The shared binder submits AICore before AICPU. HBG's
owner still needs to connect this event protocol to graph registration and
restore.

The no-argument C `get_pipeline_contract()` remains the static program contract
with zero byte fields. The internal TMR-shaped
`build_kernel_pipeline_contract_impl(config, out)` remains `UNSUPPORTED` for HBG:
CallConfig alone cannot determine a graph's sizes. HBG's owner must use the
post-build query and capacity plan; contract consumers are shared, but the HBG
and TMR sizing producers have different inputs. No mutable global size table or
configuration-only fabricated HBG sizes are introduced.

## Immutable graph packet and HostArgs submission

`make_kernel_graph_layout(window, out)` reserves the existing runtime structures
with each configurable scheduler queue sized to `next_power_of_two(max(window,
64))`. The accepted window is 1 through 32768; a kernel graph must contain
strictly fewer outer tasks than its window. Reachable in-graph task populations
must also fit the queues. Overflow returns `CAPACITY_EXCEEDED`; no silent
fallback to the program upload path exists. Program reservations retain their
existing maximum capacities. TensorMap and orchestration scratch are Host-only
in this implementation and are not serialized into a device arena.

Collect graph requirements with this compact layout, aggregate them with
`KernelResourcePlan`, then prepare and freeze the context. The capacity remains
fixed while different graphs use different prefixes of it.
`make_graph_launch_template(build, runtime, context, device_id, slot_generation,
runtime_binary_id, identity, out)` consumes the completed build under its workspace lease and:

1. Checks readiness, graph/window bounds, identity, and frozen context binding.
2. Allocates Host storage for a complete pristine image of every frozen runtime,
   Definition and A5 scheduler region. Unused bytes are zero. No device memory
   is allocated, accessed, cleared or copied by this operation.
3. Emits a clean RuntimeContext with no Host/component pointers. Restacks SM
   from the live Host mirror, preserving self-relative argument references and
   translating virtual heap addresses onto the prepared device heap. External
   tensor device addresses and scalar values retain their meanings.
4. Copies each distinct retained/spilled Graph Definition, including framing,
   into the packet and binds outer Graph tasks to the prepared Definition
   destination. The source graph and staging are not patched.
5. Publishes an owning `GraphLaunchTemplate` only after complete packet validation.
   Failure preserves the previous template. Its source workspace may then be
   reused or released; captured packets still require the device context/storage
   and callable residency to remain alive.

The canonical packet layout is:

```text
SimplerKernelInvocationHeader (sizeof the shared K1 header)
GraphPacketHeader            (192 bytes, HBG format version 2)
GraphImageRegion[]           (32 bytes each)
zero padding to a 64-byte relative offset
inline payload:
  full runtime/SM capacity
  full Definition capacity, if nonzero
  full A5 scheduler capacity, if nonzero
```

The HBG header carries an internal context-slot identity, device ID, runtime
binary identity, per-invocation hashes,
task count/window, runtime/SM offsets and
four destination base/capacity pairs. Region source offsets are relative to the
inline payload; destination offsets are relative to the selected working region.
The existing runtime image types retain their ABI. Their internal runtime
pointers are null and rebuilt after restore; task heap/Definition references
are bound to stable device destinations. Caller tensor contents are not copied.

`validate_graph_packet` uses bounded `memcpy` reads, accepts an unaligned source,
and checks framing before accessing the region table or payload: the HBG format
version and reserved fields, common invocation mode/counts, exact lengths,
overflow, alignment, disjoint destination
ranges, canonical region order and full-capacity coverage. A checksum covers the
common header, HBG binding/identity, descriptors, padding and payload, excluding
only the checksum and the single patched address. It detects accidental
corruption; it cannot replace H3's independent device registry trust check or
the leader image validation and restore described in
[execution-slot registration and restoration](host-build-graph-kernel-slot.md).

[Execution-slot registration and admission](host-build-graph-kernel-slot.md)
defines the prepare-time seal, context-owned AICPU registry and read-only
`admit_graph_packet_for_restore` gate. The registry is not a graph/callable cache
and no launch payload can choose its address. HBG version 1 packets are rejected;
the outer K1 invocation ABI remains unchanged.

`make_graph_host_args` validates the template and produces a fresh writable copy
with one placeholder. Both offsets include the outer common header:
`address_offset = sizeof(SimplerKernelInvocationHeader) + offsetof(GraphPacketHeader, inline_payload_addr)` and
`data_offset = sizeof(SimplerKernelInvocationHeader) + payload_offset`.
`submit_graph_template` passes that copy to
a synchronous HostArgs consumer; only task execution is asynchronous.
`aicpu_loader/host/kernel_graph_launch.h::launch_graph_template` adapts this to
`aclrtLaunchKernelWithHostArgs`, forwarding the supplied dedicated AICPU stream,
function, block count and config, with exactly one `aclrtPlaceHolderInfo`.
Enqueue errors are returned unchanged to the enclosing launch protocol.

The adapter assumes the enclosing protocol has established entry/exit events
and retained the function/context leases. It does not create streams, record
or wait events, implement partial-enqueue recovery, or enable public HBG launch.
The A5 scheduler region is restored from its zeroed template. The common
scheduler queues, mailbox and runtime pointers are rebuilt by the leader restore;
A5 dispatch-specific metadata binding remains with the kernel entry.
Host framing/ownership tests and compilation against the installed CANN header
are not evidence of on-device capture/restore correctness.

## Host tensor-data requirement semantics

The independent orchestration requirements metadata describes generated Host
behavior, not whether a tensor argument exists or carries a device address.
New orchestration libraries may export
`pypto_orchestration_requirements_v1`. Kernel mode requires the symbol and
rejects unknown bits before Host build. Program mode records the metadata when
present but retains its existing behavior for legacy libraries.

| Host orchestration operation | Requires Host tensor-data capability |
| ---------------------------- | ------------------------------------ |
| Inspect shape, dtype, stride or scalar arguments | No |
| Carry device addresses or construct tensor views without dereferencing storage | No |
| Emit device predicate metadata (address, comparison, element size) | No; device evaluates the value |
| Execute `get_tensor_data` / read tensor element values on Host | Yes; kernel mode also requires an explicit Host-copy argument |
| Execute `set_tensor_data` / write tensor values on Host | Yes; unsupported in kernel mode |

H6 encodes arguments as `[device tensors][host-only copies][scalars]`.
`host_copy_tensor_count` names the trailing tensor suffix. The suffix is paired
in order with the equally sized suffix of the device-tensor prefix. A pair must
have identical shape, stride, dtype, start offset and buffer size; only its
address space and address differ. This gives `pa(..., block_table,
block_table_host)` a deterministic wire form without another pointer table.

All main tensors must carry `AddressSpace::DEVICE`. V1 accepts a dense
row-major layout or row-major rows with outer padding: the innermost stride is
one and each outer stride covers the complete next dimension. Broadcast,
transpose and stepped innermost views are rejected before Host build.

Kernel Host build registers only the explicit Host-copy suffix as read-only.
It never maps a device tensor, performs fallback D2H, synchronizes a stream or
allows Host writes. The Host-only copy is build input and cannot enter a task
payload, Graph Definition, predicate operand or device graph packet. The host
packet producer checks this before binding the frozen working slot, and the
AICPU restore path checks again before writing any device destination.

The resulting graph packet contains caller-owned device addresses and the
captured graph structure. Capture builds this immutable packet once. Replay
restores it into the existing fixed-capacity slot and does not re-enter Host
build or read the Host copy. A changed shape, stride family, task topology or
address specialization requires another captured graph.
