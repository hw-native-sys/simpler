# HBG host graph build

These internal C++ interfaces separate Host graph construction from program-mode
resource binding and upload. They introduce no public C entry point or wire ABI.
The Host build boundary is also the input to kernel-mode resource sizing and
serialization; those consumers and kernel launch integration are separate work.

## Construction and ownership

`simpler::hbg::build_graph` consumes already-staged arguments and a
`LeasedWorkspace`: the RuntimeContext, SM mirror, its size/task capacity, and
retained Definition staging. Build is the construction and sizing portion of a
program **bind**; upload completes that bind by assigning device addresses and
transferring the result.

`GraphBuild` owns the orchestration state, Definition records and packing plan.
The workspace is borrowed, and must remain exclusively leased until every
consumer finishes. Neither the workspace nor its RuntimeContext may serve
another build in that interval. GraphBuild is explicitly noncopyable and
nonmovable; a second build in the same object invalidates its previous result.
The type is runtime implementation state, not a stable plugin ABI.

A successful build provides:

- Outer task count, populated SM extents, compact image bytes and heap bytes.
- Definition block bytes and a validated plan for retained and spilled objects.
- Complete queue populations and capacities, including the body of **each**
  Graph invocation, even when several invocations share one Definition.

Definition framing, task-array bounds, execution-storage fit and queue limits
are checked before build reports success. Queue overflow returns the existing
runtime error without acquiring a Definition block, committing device regions,
or copying to device. Host staging can be used or allocated by orchestration;
this boundary does not promise zero Host allocation. Kernel consumers must gate
Host tensor-data access and DFX before construction; this change preserves the
existing program-mode Host tensor and tracing behavior.

`build_complete` is a validity bit, not a device-execution state. Failed builds
cannot be uploaded. A later successful build can reuse the object. Upload does
not consume the build, and success or copy failure leaves a completed build
available for retry while its workspace lease remains held.

## Program-mode upload

`simpler::hbg::upload_for_program_mode` checks build validity and RuntimeContext
identity, then uploads the precomputed Definition plan, commits execution
regions and copies a compact runtime image. It consumes the build's queue
capacities without another sizing pass. Task-count tracing belongs to build;
transfer spans account for every actual upload, including retries.

Retained Definition objects stay in staging; spilled objects are copied into
aligned offsets after the retained prefix. If staging grows, the owner preserves
its contents and updates both the Definition records and LeasedWorkspace view
before any fallible H2D. The plan stores
retained objects by offset, so retries never read the freed staging allocation.
Each runtime image is compacted from the original virtual-heap mirror; repeated
upload can rebind a different device heap without changing that source.

The Definition packing/upload implementation is shared between architectures.
Runtime image wiring and upload remain architecture-specific, including A5
scheduler allocation. Program execution invokes build and upload in sequence.
These upload operations may allocate and synchronously copy device memory;
**kernel capture must not call this program-mode upload path**.

## Kernel resource declaration

`get_graph_resource_requirements` converts one completed `GraphBuild` into a
host-only `GraphResourceRequirements` snapshot. It reads H1's measured heap,
compact runtime image and Definition bytes directly. A5 additionally derives an
upper bound for its optional AICore scheduler state from H1's task count. The
query does not allocate device memory, bind device addresses or perform H2D.
Failure leaves the caller's previous snapshot unchanged.

Each snapshot carries a `RuntimeLayoutKey`. The key records the architecture,
layout ABI, task capacity and copied arena boundaries. Equal byte totals are not
enough to make two images compatible: `GraphCapacityPlan::create` rejects a set
whose keys differ, so an A2/A3 image, an A5 image or an image built for another
task window cannot silently share one prepared slot.

`GraphCapacityPlan` describes capacity; it does not represent committed HBM.
For all allowed graph snapshots it takes the maximum of each independent
region, then builds a new aligned layout in this order:

1. runtime image, including the compact shared-memory tail;
2. Graph Definition payload;
3. optional scheduler state.

The GM heap remains a separate region. Repacking the maxima matters because the
largest runtime image, Definition block and scheduler block may come from
different graphs. Taking only the largest already-packed total would not prove
that all three maxima fit together.

The resulting kernel `PipelineContract` has depth one and declares four
resources: per-run GM heap, per-run runtime slot, a dedicated non-hidden AICPU
stream and a hidden AICore stream. The framework's caller stream is borrowed at
launch and is deliberately not a context-owned resource. `bind_kernel_streams`
requires all three stream handles to be non-null and pairwise distinct; events
recorded and waited across those streams provide ordering in later launch work.

The kernel-context prepare stage must allocate the plan's GM heap and
`runtime_slot_bytes()` once, retain stable bases and freeze those capacities.
Launch may only admit a snapshot that fits the frozen plan and restore its
captured payload into that existing slot. It must not call
`upload_for_program_mode`, grow a region or fall back to allocation. ACLGraph
replay reuses the captured payload and prepared slot; it does not run Host build
again. Actual slot ownership, payload serialization and restore are later
stages, outside this declaration change.
