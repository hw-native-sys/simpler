# L3-L2 Orch Comm Direct Parent Data Path Design

This document designs the child export / L3 import direction for L3-L2
orchestrator communication. The design removes the PR #1015 dedicated
child-side shared-memory request/response data path while preserving the
public primitive and message queue behavior.

The core change is execution ownership. The L2 child process still owns region
allocation, export authorization, physical lifetime, and cleanup. The L3 parent
process owns steady-state payload and counter execution after region creation.

## Goals

- Remove the dedicated L3-L2 orch comm shared-memory request/response service
  introduced by PR #1015.
- Execute `PAYLOAD_WRITE`, `PAYLOAD_READ`, `SIGNAL_NOTIFY`, `SIGNAL_TEST`, and
  `SIGNAL_WAIT` in the L3 parent process.
- Keep GM region ownership child-owned.
- Use existing worker mailbox/generic control for region lifecycle setup and
  teardown.
- Preserve public Python APIs, L2 endpoint ABI, and L2 `TaskArgs` scalar shape.
- Preserve message queue protocol, layout, descriptor format, opcodes,
  STOP/ERROR behavior, capacity/depth semantics, input-window behavior, and
  observable timeout/poison behavior.
- Support `a2a3sim`, `a2a3`, `a5sim`, and `a5`.

## Non-Goals

- Changing region ownership to parent-owned.
- Retaining the PR #1015 child service as a production data-path fallback.
- Adding user-visible environment flags or runtime knobs.
- Setting a numeric latency target.
- Redesigning remote L3 worker/session transport.
- Changing L2 endpoint methods or the six-scalar region descriptor.
- Redesigning the message queue protocol.

## Current State

Today L3 creates a POSIX shared-memory control block and sends its name to the
chip child with `CTRL_L3_L2_ORCH_COMM_INIT`. The child attaches that block and
starts a host-side `L3L2OrchCommService` thread under the child
`DeviceRunner`.

All region commands then flow through that service:

```text
L3 parent process
  L3L2OrchCommClient
  shared-memory request/response block
      |
      v
L2 child process
  L3L2OrchCommService
  DeviceRunner allocation/copy/counter backend
```

The service handles both lifecycle commands and steady-state data-path
commands:

- `ALLOC_REGION`
- `FREE_REGION`
- `PAYLOAD_WRITE`
- `PAYLOAD_READ`
- `SIGNAL_NOTIFY`
- `SIGNAL_TEST`
- `SIGNAL_WAIT`

This adds one child-side command round trip to every payload and counter
operation. It also makes the L2 child a steady-state data-path executor even
though the L3 parent owns the user-facing primitive handle.

## Target Architecture

Split the current service into two responsibilities:

```text
lifecycle control:
  L3 parent -> existing worker mailbox/generic control -> L2 child

steady-state data path:
  L3 parent direct-access backend -> parent-imported child-owned GM
```

After region creation, L3 operations do not send child-side service commands.
The parent handle contains:

- the unchanged L2 descriptor returned by the child;
- parent-private direct-access metadata for payload and counter regions;
- region state used for release, poison, and cleanup.

The child owns allocation and export. The parent owns access execution.

## Address Model

The design uses a two-address model.

The L2 descriptor stays exactly six `uint64_t` scalars:

```text
magic_version
region_id
payload_base
payload_bytes
counter_base
counter_bytes
```

`payload_base` and `counter_base` are L2-consumed child-side GM addresses.
L2 orchestration code passes these scalars to `L3L2OrchEndpoint`, which uses
them to form payload GM views and counter GM addresses.

The parent must not reinterpret those fields as parent-dereferenceable
addresses. On onboard platforms, the parent import may produce different VA or
copy handles from the child GM address. On simulation platforms, the child and
parent may map the same backing object at different host VAs.

Parent access metadata is private to the L3 parent handle. It may include:

- parent payload access VA or DMA endpoint handle;
- parent counter access VA or DMA endpoint handle;
- payload and counter byte sizes;
- backend profile;
- import ids, mmap handles, ACL import handles, or release tokens;
- any platform state needed to close the import before physical free.

This metadata is not exposed through `descriptor_scalars()`, is not passed to
L2, and is not part of the public Python API.

The parent direct-access implementation is owned by host-only parent code. The
Python `L3L2OrchRegion` facade owns live/released/poison policy; native helper
code owns only imported mapping or copy resources and returns enough error
detail for Python to preserve existing poison behavior. The parent path must
not load `host_runtime.so`, run the child runtime, or schedule L2 work.

## Lifecycle Control

Region lifecycle uses the existing worker mailbox/generic control path. The
dedicated `CTRL_L3_L2_ORCH_COMM_INIT` service bootstrap is removed from the
production design.

Introduce internal lifecycle controls equivalent to:

- create region;
- release region;
- shutdown cleanup.

The exact numeric command ids are implementation details. They should share the
same mailbox serialization rules as other child controls: a control command is
mutually exclusive with task dispatch on that worker mailbox.

### Create Region

`orch.create_l3_l2_region(...)` performs these steps:

1. Validate `payload_bytes`, `counter_bytes`, and `worker_id` in the parent.
2. Send a create-region lifecycle control to the target child.
3. The child allocates one child-owned GM communication region and partitions
   it into payload and counter ranges.
4. The child initializes the counter range to zero.
5. The child exports parent-access metadata for the payload and counter ranges.
6. The child returns the unchanged L2 descriptor plus that export metadata.
7. The parent imports or maps the exported access metadata into
   parent-private payload/counter access metadata.
8. The parent constructs `L3L2OrchRegion` with the L2 descriptor and direct
   access metadata.

If parent import fails after child allocation/export succeeds, the parent must
not publish a live region handle. It must close any parent-side import, mmap,
or copy state that was opened before sending release control for the child
allocation. If either close or release cannot be confirmed, the create call
fails with cleanup uncertainty and the region does not become live in user
code.

### Release Region

`region.free()` keeps the existing user-visible release behavior:

- mark the parent handle released;
- reject later payload, counter, and descriptor operations;
- defer physical cleanup until submitted L2 work that may hold the descriptor
  has drained.

The physical cleanup path closes parent imports or mappings and then releases
the child-owned GM allocation through lifecycle control. The child must not free
physical GM while submitted L2 work may still use the descriptor.

Release control is cleanup-oriented and idempotent. Releasing a live region
frees the child-owned allocation; releasing a missing or already-released
`region_id` is a successful no-op, matching the current service behavior.

If an L3 orchestration execution exits with live regions, runtime cleanup marks
them released, drains submitted work, closes parent imports, and releases child
physical resources.

### Shutdown Cleanup

Child shutdown remains the final owner-side cleanup backstop. If parent-side
cleanup is interrupted, the child must release any remaining child-owned
regions during worker shutdown. Parent imported mappings are process-local and
must be closed by the parent when reachable; backend cleanup must tolerate
already-closed or never-imported mappings.

## Onboard Backend

On onboard platforms, the design uses child export plus parent import.

The child allocates one GM communication region with the platform's GM
allocation path, partitions it into payload and counter ranges, authorizes the
L3 parent process to access those ranges, and returns export metadata
sufficient for the parent to materialize direct access. This follows the
ownership pattern already used by onboard communication-domain windows: the
owner allocates and exports access metadata; the peer imports.

The parent direct-access backend owns any parent-side runtime/context setup
needed for import, DMA copy, or CPU access. That setup is internal. It must not
load or run the child runtime, schedule L2 work, or expose a user-visible
configuration knob.

The design does not require a single identical primitive on every onboard
generation. `a2a3` and `a5` may use different ACL import flags, cache
maintenance, or copy APIs as long as they provide the same primitive semantics.
The production per-region path must use explicit parent import close before
child physical free; it must not inherit the CommDomain failure-path model that
relies on device reset/finalize to reclaim IPC resources.

The exact onboard import flags, import-close API, parent device-context binding,
same-device versus cross-device import behavior, and CPU-dereference safety are
platform spike gates. They must be validated for each supported onboard
generation before implementation chooses that platform's production counter
path.

## Simulation Backend

Simulation may use a functionally equivalent mapping instead of onboard IPC.

The sim child should allocate the region from a shared backing object, such as
POSIX shared memory, that both processes can map. POSIX shm naming, collision
avoidance, and cleanup should follow the existing sim CommDomain discipline:
short fixed-width names, per-region uniqueness, child-side ownership, and
shutdown sweep of leftover live regions. The L2 descriptor contains the
child-local mapped addresses. The parent metadata contains the parent-local
mapped addresses.

Only the counter range has an initial-value contract and must be zeroed before
reply publication. Payload bytes are unspecified until written by L3. A sim
implementation may zero more bytes as an implementation detail, but tests and
callers must not depend on payload zero-initialization.

This preserves the same two-address model as onboard and keeps tests honest:
code must not assume descriptor addresses are valid in the parent process.

## Payload Operations

`payload_write(offset, host_buffer, nbytes)` remains a synchronous copy from a
parent-accessible host byte span into the child-owned GM payload range.

`payload_read(offset, host_buffer, nbytes)` remains a synchronous copy from the
child-owned GM payload range into a parent-accessible writable host byte span.

The accepted buffer contract is relaxed from "child-visible runtime-managed
host tensor" to "parent-accessible contiguous byte span that is legal for the
operation." Public method names and argument shape do not change.

Examples:

- `payload_write` may accept a contiguous runtime tensor, `bytes`,
  `bytearray`, or another contiguous buffer-protocol source.
- `payload_read` requires a writable destination such as a runtime tensor,
  `bytearray`, or writable memoryview.
- Non-contiguous sources or destinations remain caller errors unless the
  implementation explicitly copies them into a contiguous parent-side span.

The implementation should use this priority:

1. If the caller-provided parent buffer is directly usable by the platform DMA
   or copy endpoint, transfer directly between that buffer and imported GM.
2. If the buffer is parent-accessible but not directly usable by the DMA or
   copy endpoint, use a parent-owned staging buffer.
3. Never stage merely to make the buffer visible to the L2 child.
4. Never ask the child to execute a steady-state payload command.

Valid reasons for parent-owned staging include DMA registration requirements,
pinning, alignment, contiguity, writability, or lifetime constraints. Staging
is a parent-side DMA adaptation mechanism, not a child-visibility mechanism.

## Counter Operations

Counter operations execute in the L3 parent process against parent direct
access metadata.

Preferred implementation:

- `SIGNAL_NOTIFY` writes the imported counter from the parent.
- `SIGNAL_TEST` loads the imported counter once from the parent.
- `SIGNAL_WAIT` polls the imported counter in the parent until the comparison
  matches or the finite timeout expires.

If a platform import produces a parent CPU-dereferenceable counter mapping,
the backend should use CPU load/store/poll. If a platform can only expose a
copy endpoint, the backend may implement the same operations with parent-side
4-byte copy or DMA operations.

Both implementations are parent data paths. Neither may call a child service.

`NotifyOp.Add` remains a convenience read-modify-write operation. It does not
become an inter-agent atomic operation.

`SIGNAL_TEST` mismatch and `SIGNAL_WAIT` timeout remain ordinary no-progress
results. They do not poison the region by themselves.

## Ordering And Cache Maintenance

The abstract ordering contract is unchanged:

- payload writes ordered before `signal_notify` are published by that notify;
- a matched `signal_test` or `signal_wait` is the observe point before reading
  protected payload;
- a failed `signal_test` does not establish observe semantics;
- all waits use finite timeouts.

The L2 endpoint keeps its existing cache-maintenance behavior and ABI. The
responsibility that moves is the L3-side implementation of equivalent ordering
and visibility. The parent direct-access backend must provide the necessary
ordering around:

- host-to-GM payload transfers;
- GM-to-host payload transfers;
- parent counter load/store or parent 4-byte copy operations;
- publish and observe transitions between payload and counters.

The backend should follow the hardware rules in
[hardware/cache-coherency.md](hardware/cache-coherency.md). The design does
not duplicate per-generation cache opcode rules.

## Error And Poison Semantics

Existing visible error behavior is preserved unless listed here.

Pre-command validation failures remain non-poisoning:

- malformed API arguments;
- invalid counter offset;
- unsupported or non-contiguous host buffer rejected before transfer;
- out-of-bounds payload range;
- descriptor extraction after release or poison.

Failures after a data transfer or counter operation is issued poison the
parent region:

- parent DMA or copy failure;
- failed imported mapping access;
- signal notify access failure;
- lifecycle fatal error after the region is live;
- L2 endpoint fatal error reported with a valid `region_id`.

`SIGNAL_WAIT` timeout reports the last observed value and does not poison the
region by itself. Queue-level ordinary backpressure remains non-poisoning.

The existing endpoint fatal text path remains compatible: if L2 reports an
endpoint error containing a region id, the parent poisons only the matching
live region.

## Message Queue Impact

The message queue remains a wrapper over `L3L2OrchRegion`.

The queue must not define a new protocol, layout, descriptor format, opcode
set, capacity rule, STOP/ERROR behavior, or input-window behavior for this
change. Its observable behavior changes only because its primitive payload and
counter operations now execute in the parent.

The queue wrapper should delegate buffer DMA eligibility and parent-side
staging to the primitive payload implementation. It should not keep a separate
registered scratch-buffer responsibility whose purpose is satisfying the old
child-visible primitive contract.

The queue may still create ordinary small temporary byte objects for
descriptor or header packing. Those temporaries are not child-visible scratch
buffers and do not own data movement policy. If such a temporary cannot be
directly consumed by the platform transfer path, the primitive payload layer
decides whether to stage it.

## Migration Plan

1. Add lifecycle create/release controls over the existing worker mailbox.
2. Add platform direct-access metadata and backend operations for parent
   payload and counter access.
3. Change `L3L2OrchRegion` to store the L2 descriptor plus parent private
   metadata.
4. Move `payload_write`, `payload_read`, and counter operations to the parent
   direct-access backend.
5. Move queue buffer eligibility and staging decisions down to the primitive
   payload layer.
6. Remove the dedicated shared-memory request/response service and its
   bootstrap from production code.
7. Remove tests that assert steady-state child service commands and replace
   them with tests that assert unchanged primitive and queue behavior.
8. Update `docs/l3-l2-orch-comm.md` and `docs/l3-l2-message-queue.md` in the
   same code migration that removes the production child service path.

Temporary bring-up fallbacks may exist only during development. The production
target has no child service data-path fallback.

## Validation

Validation should prove behavior, not a specific latency number.

Required checks:

- Primitive region tests preserve descriptor scalar shape and validation.
- `PAYLOAD_WRITE`, `PAYLOAD_READ`, `SIGNAL_NOTIFY`, `SIGNAL_TEST`, and
  `SIGNAL_WAIT` succeed without a child service command after create.
- `SIGNAL_TEST` mismatch and `SIGNAL_WAIT` timeout remain non-poisoning.
- Payload/counter access failures poison only the affected parent region.
- `region.free()` and orchestration cleanup remain drain-safe.
- Parent operations use parent-private metadata, not descriptor
  `payload_base` or `counter_base`, for parent access.
- L2 endpoint tests continue to pass without ABI changes.
- Message queue tests and examples keep observable behavior, including
  STOP/ERROR and input-window behavior.
- The L3-L2 orch comm stream example and message queue example pass on
  `a2a3sim`, `a2a3`, `a5sim`, and `a5`.

Acceptance is met when the PR #1015 dedicated service path is removed from
production behavior and all steady-state primitive operations execute in the
L3 parent process while existing supported examples and tests keep the same
observable behavior.
