# HBG kernel resource declarations

Reference: the supplied `kernel-mode-design.md`, [v9 final decisions](https://icc.gt.tc/vllm-pto#v9-design).
The final decisions in §0 override historical alternatives, except for the user's subsequent three-stream clarification: caller, dedicated non-hidden AICPU and hidden AICore are distinct.
These are internal host C++ interfaces, not new public workspace-size APIs.
K1's current C entry points, context-control POD and 64-byte invocation header
remain unchanged. Shared resource kinds and validation align with
[TMR declarations at 317f5597](https://github.com/Leaf-Salix/simpler/commit/317f55971ce092244c2f370ba17899260092adc6);
that branch's different K1 wire revision must not be substituted here.

## Host build and program upload

`hbg::build_graph` performs Host orchestration against already-staged arguments.
It neither commits device execution regions nor uploads the graph.
`GraphBuild` owns orchestration and Definition records but borrows the SM mirror,
Definition staging and matching runtime workspace. These buffers must remain
leased until the result is no longer used; a second build cannot reuse them
while the first result is still being consumed. Build and upload require
exclusive workspace access. Concurrent read-only queries may share a completed
build, but must not overlap build or upload.

`hbg::upload_program_graph` is the explicit program-only allocation and
synchronous H2D boundary. It creates a compact image from the virtual-address
source on each upload. If Definition staging grows and moves, the upload owner
preserves its contents and rebinds the build's staging reference before further
processing, including before a potentially failing copy. Repeated upload and
retry therefore keep a valid source. Program execution still calls build and
upload in sequence and retains its existing resource management.

## Graph requirements and context capacity

After a successful build, call
`hbg::get_graph_resource_requirements(build, layout, requirements)` with the
layout of the intended destination for the same task window and runtime ABI.
The layout must match the intended runtime destination. The query does no device
allocation, address binding or H2D. It enumerates Host Definition records, so
resource discovery runs outside capture.

`GraphResourceRequirements` is an independent value snapshot for one graph:

| Field | Meaning |
| ----- | ------- |
| `gm_heap_bytes` | Measured graph heap including Graph execution storage, rounded to arena alignment |
| `runtime_arena_bytes` | Device-only prefix, RuntimeContext and compact SM tail; SM is counted once |
| `graph_definition_bytes` | Used retained prefix plus aligned spill objects and framing; distinct Definitions counted once |
| `scheduler_state_bytes` | A5 scheduler allocation upper bound including alignment slack; zero on A2/A3 and A5 Graph fallback |

`required_bytes()` checks the sum of these logical requirements. It is neither
committed HBM telemetry nor context capacity. Public committed-memory reporting
continues to use `committed_device_memory_ctx`; caller tensors, process-pinned
code and CANN packets are outside this graph calculation.

`KernelResourcePlan::create(graphs, count, out)` builds a capacity declaration
for one serialized execution slot. Inputs must belong to the same context's
architecture and runtime layout ABI. It takes the maximum of each compatible
region requirement, then **recomputes** aligned Definition and scheduler offsets
after the runtime/SM capacity. It does not merge offsets or take the maximum of
already-packed total sizes. The runtime/SM region accommodates one complete
per-invocation image; its internal offsets must be bound from that invocation's
layout during restore, never combined across graphs.

For example, graph A needs runtime 8192 and Definition 512 bytes; graph B needs
runtime 4096 and scheduler 2048 bytes. The combined arena reserves runtime
`[0, 8192)`, Definition `[8192, 8704)`, padding, and scheduler `[9216, 11264)`.
This is a legal layout for either graph; the larger individual packed total
alone would not describe these combined region capacities.

`plan.admits(graph)` compares every region against capacity without changing the
plan. Exact capacity is accepted; an excess in any region is rejected even if
another region has unused bytes. Queries and plan construction publish outputs
only on success. Overflow returns `CAPACITY_EXCEEDED`; invalid build state or
arguments fail before any resource mutation.

`plan.prepare(context, resource_ops)` now passes this layout to
`KernelExecutionState::prepare_resources`. The context owns the actual device
allocations through `KernelDeviceResources`: one heap and one packed runtime
arena. Definition and scheduler destinations are slices of that arena, so this
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
ready state, device/generation identity, the HBG region schema and each required
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
// Enqueue device initialization/registration and establish its event ordering.
context.mark_ready_enqueued();
context.freeze_resources();

// Resource portion of each launch: no allocator argument is available here.
hbg::KernelWorkingBinding binding;
hbg::bind_kernel_resources_for_launch(context, device_id, generation, graph, binding);
```

The allocator adapter uses the platform `MemoryAllocator`, preserving existing
committed-byte accounting (including alignment slack). It must remain alive
until explicit context close; neither its destructor nor program teardown may
run while captured graphs reference the context. Context close releases only
its own allocations, never caller tensors. External workspace injection remains
deferred. Expanding a captured context requires a new generation and slot.

The public K1 init/prepare/launch functions remain unsupported stubs: the
resource lifecycle is implemented internally, but the immutable packet producer,
public owner integration, registration and device restore are still required
before enabling execution. K1 context-control FREEZE must delegate to this resource transition
when the public context owner is connected; it must not merely set a flag on
program arena banks. No wire layout or public entry point is added here.

## Common contract and stream roles

`plan.pipeline_contract()` projects the complete capacity into existing kinds:

| Kind | Class | Bytes per copy / binding |
| ---- | ----- | ------------------------ |
| `GM_HEAP` | `HOST_PER_RUN` | Heap capacity |
| `RUNTIME_IMAGE` | `HOST_PER_RUN` | Packed runtime/SM + Definition + scheduler capacity, including inter-region padding |
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
`bind_kernel_stream_roles(contract, caller, aicpu, hidden_aicore, out)` preserves
all three handles. Missing either execution role, invalid classes, null handles
or any pair of aliased streams fails before publication. `KernelContextOps`
receives the stream kind when creating/destroying a handle, allowing the owner
to create a non-hidden AICPU stream and a hidden AICore stream separately.
The context event set includes independent AICPU/AICore completion events.
The actual event enqueue sequence is still part of launch integration.

The no-argument C `get_pipeline_contract()` remains the static program contract
with zero byte fields. The internal TMR-shaped
`build_kernel_pipeline_contract_impl(config, out)` remains `UNSUPPORTED` for HBG:
CallConfig alone cannot determine a graph's sizes. HBG's owner must use the
post-build query and capacity plan; contract consumers are shared, but the HBG
and TMR sizing producers have different inputs. No mutable global size table or
configuration-only fabricated HBG sizes are introduced.

## Host tensor-data requirement semantics

The independent orchestration requirements metadata describes generated Host
behavior, not whether a tensor argument exists or carries a device address.
The Host tensor-data capability bit has these semantics for the gate producer
and consumer:

| Host orchestration operation | Requires Host tensor-data capability |
| ---------------------------- | ------------------------------------ |
| Inspect shape, dtype, stride or scalar arguments | No |
| Carry device addresses or construct tensor views without dereferencing storage | No |
| Emit device predicate metadata (address, comparison, element size) | No; device evaluates the value |
| Execute `get_tensor_data` / read tensor element values on Host | Yes, including reads through a staged Host mirror |
| Execute `set_tensor_data` / write tensor values on Host | Yes, including mirror writes followed by H2D |

The gate must reject that capability for kernel mode before build or execution
resource mutation. Missing metadata and unknown bits also fail closed in kernel
mode. Program mode retains its existing Host accessor behavior and permits old
orchestration libraries without metadata. An explicit future Host-copy argument
ABI must be treated separately; it does not make an arbitrary device tensor
Host-readable. K1's `host_copy_tensor_count` remains zero.

This declaration change does not load or gate requirements symbols and does not
assign a new competing bit number. Producer bit assignments and optional symbol
loading belong to the capability-gate integration with PyPTO.
