# L3-L2 Orch Comm Direct Parent Data Path Implementation

This document turns the direct-parent-data-path design into an implementation
plan. It is intentionally more concrete than the design note: it names the
controls, wire payloads, parent metadata, backend responsibilities, migration
steps, and validation points.

The implementation keeps the public primitive and message queue APIs unchanged.
It removes the PR #1015 steady-state child service path from production.

## Goals

- Add lifecycle-only region create and release controls on the existing worker
  mailbox.
- Allocate one child-owned GM communication region and partition it into
  payload and counter ranges.
- Export that child-owned region on onboard platforms and import it in the L3
  parent process with public ACL IPC primitives.
- Map an equivalent shared backing object on simulation platforms.
- Store parent direct-access handles only in parent-private metadata.
- Execute `PAYLOAD_WRITE`, `PAYLOAD_READ`, `SIGNAL_NOTIFY`, `SIGNAL_TEST`, and
  `SIGNAL_WAIT` in the L3 parent process.
- Defer physical release until submitted L2 work has drained.
- Preserve public Python APIs, L2 endpoint ABI, L2 `TaskArgs` scalar shape, and
  message queue behavior.

## New Lifecycle Controls

Add typed controls next to the existing worker mailbox controls:

```text
CTRL_L3_L2_REGION_CREATE
CTRL_L3_L2_REGION_RELEASE
```

Concrete numeric ids are assigned in `worker_manager.h` during
implementation. They are not part of the design contract.

The parent-facing helpers should be named by operation role:

```text
control_l3_l2_region_create(worker_id, create_req_shm, create_reply_shm)
control_l3_l2_region_release(worker_id, region_id)
```

Create follows the same shape as `control_alloc_domain`: the mailbox carries
the control id and two POSIX shm tokens, while the request and reply bytes live
in one-shot POSIX shm blocks. Release is smaller than CommDomain release and
uses a mailbox scalar for `region_id`.

The create POSIX shm blocks are lifecycle-control scratch. They are not the
steady state data path and are not kept after the command returns.

### Create Request

The create request shm contains one fixed header:

```text
struct L3L2RegionCreateRequest {
  uint64_t magic_version;
  uint64_t request_bytes;
  uint64_t payload_bytes;
  uint64_t counter_bytes;
  int32_t  parent_pid;
};
```

`parent_pid` is required on onboard platforms so the child can authorize the
L3 parent process with `aclrtIpcMemSetImportPid(...)`. It is `int32_t` to match
the ACL IPC pid-list usage already used by onboard CommDomain IPC.

Parent validation before sending the control:

- `payload_bytes > 0`;
- `counter_bytes > 0`;
- `counter_bytes % 4 == 0`;
- `worker_id` selects an initialized local next-level chip worker.

### Create Reply

The create reply shm contains one fixed reply header plus a fixed-size transport
metadata area:

```text
struct L3L2RegionCreateReply {
  uint64_t desc[6];
  uint32_t access_profile;
  uint32_t reserved;

  // access_profile == onboard_acl_ipc
  int32_t  device_id;
  uint8_t  export_key[ACL_IPC_EXPORT_KEY_BYTES];

  // access_profile == sim_posix_shm
  uint8_t  backing_shm[CTRL_SHM_TOKEN_BYTES];
  uint64_t mapping_bytes;
};
```

`desc` is the unchanged L2 descriptor:

```text
magic_version
region_id
payload_base
payload_bytes
counter_base
counter_bytes
```

`payload_base` and `counter_base` are child-side GM addresses consumed by L2.
The parent must not dereference them.

The descriptor plus the fixed layout rule are the only source of truth for
region layout:

```text
payload_offset = 0
counter_offset = align_up(desc.payload_bytes, 64)
total_bytes    = counter_offset + desc.counter_bytes
```

The parent validates the descriptor before import:

- `payload_bytes > 0`;
- `counter_bytes > 0`;
- `counter_bytes % 4 == 0`;
- no overflow in `align_up` or `counter_offset + counter_bytes`;
- `counter_base == payload_base + counter_offset`;
- `counter_base` is 64-byte aligned.

Any mismatch is a protocol-fatal create failure. The parent must not publish a
region handle and must run create rollback.

`access_profile` identifies the internal transport:

```text
0 = invalid
1 = onboard_acl_ipc
2 = sim_posix_shm
```

The reply layout is fixed across platforms. `access_profile` selects which
transport-specific field group is meaningful. Fields for other transports must
be ignored.

For `onboard_acl_ipc`, the parent uses `device_id` and `export_key` to import
the child allocation. For `sim_posix_shm`, the parent uses `backing_shm` and
`mapping_bytes` to map the shared backing object.

`mapping_bytes` is sim transport metadata, not a layout source of truth. For
`sim_posix_shm`, the parent must validate `mapping_bytes == total_bytes`.
Onboard import size metadata, if a platform spike proves it is needed, must be
transport metadata and must also cover exactly the derived `total_bytes`.

The reply must contain enough information for the parent to construct
parent-private metadata without consulting the child again.

### Release

Release sends only `region_id` through the mailbox scalar area. No release shm
is required. Region ids are unique within one child worker lifetime and are not
reused, so release does not need a separate generation field.

Release is idempotent and cleanup-oriented. If `region_id` is live, the child
frees the allocation and removes it from the live registry. If `region_id` is
missing, already released, or unknown, release returns success as a no-op. This
matches the current L3-L2 service cleanup behavior and lets rollback/cleanup
retry safely.

## Child Region Allocation

The child allocates one physical communication region, not separate
user-visible payload and counter regions.

The layout is:

```text
payload_offset = 0
counter_offset = align_up(payload_bytes, 64)
total_bytes    = counter_offset + counter_bytes
```

The child allocates `total_bytes` with the platform GM allocation path and
initializes the counter range to zero before publishing the reply. Payload
initial bytes are not a public contract; implementations may zero more bytes as
an implementation detail, but tests must not depend on it.
The returned child allocation base must be aligned to at least 64 bytes so the
aligned `counter_offset` also gives a 64-byte-aligned `counter_base`.

The child constructs the L2 descriptor as:

```text
payload_base = child_region_base + payload_offset
counter_base = child_region_base + counter_offset
```

`counter_base` must remain 64-byte aligned. `counter_bytes` remains a multiple
of 4. The primitive layer still validates only 4-byte counter addresses, while
wrappers keep different writers on separate 64-byte cache lines.

The child tracks each live region in a child-owned registry:

```text
region_id
allocation_base
total_bytes
payload_offset / payload_bytes
counter_offset / counter_bytes
export metadata
```

This registry is the owner-side cleanup source of truth. It is swept during
child shutdown/finalize as a backstop. Released or unknown ids are not tracked
separately because release no-ops when the id is missing from the live registry.

## Onboard Backend

On `a2a3` and `a5`, create uses public ACL IPC primitives:

1. Child allocates the single GM communication region.
2. Child calls `aclrtIpcMemGetExportKey(...)` for that region.
3. Child calls `aclrtIpcMemSetImportPid(...)` with the parent PID from the
   create request.
4. Child returns the L2 descriptor and ACL export metadata in the create reply.
5. Parent attaches the required parent-side device context internally.
6. Parent calls `aclrtIpcMemImportByKey(...)`.
7. Parent stores the imported VA or copy handle in private metadata.

The parent direct backend is implemented by private `_task_interface` helpers
in the existing nanobind module, with host-only declarations in
`src/common/platform/include/host/l3_l2_orch_region_access.h`. The Python
`L3L2OrchRegion` facade owns live/released/poison policy. Native helpers own
only imported ACL resources and return structured failures. They must not
`dlopen` `host_runtime.so`, call `l3_l2_orch_comm_init_ctx`, run the child
runtime, schedule L2 work, or expose a user-visible runtime knob.

`a2a3` and `a5` may use different ACL import flags. The implementation may use
the public IPC pattern from onboard CommDomain windows as a reference, but must
not inherit its failure-path cleanup model that relies on device reset/finalize
to reclaim IPC resources. L3-L2 direct access requires explicit per-region
parent import close before child physical free.

The parent direct-access backend must provide:

- H2D copy from a parent byte span to imported GM;
- D2H copy from imported GM to a parent writable byte span;
- exactly one production counter access path selected after platform
  validation;
- import close/release before child physical free.

Before implementation, run platform spikes that answer:

- which device context the parent binds or creates for each target worker;
- whether import is same-device or cross-device for supported topologies;
- the `aclrtIpcMemImportByKey(...)` flag for `a2a3` and `a5`;
- the exact import-close API and imported-handle ownership;
- whether `aclrtIpcMemSetImportPid(...)` must complete before reply
  publication;
- how child `aclrtFree` is sequenced and reported after parent import close;
- whether imported GM can be safely CPU-dereferenced by the parent.

If a platform proves CPU dereference works, implement counter load/store/poll
with CPU access. Otherwise implement counters with parent-side 4-byte copy
operations. Each platform must ship one production counter path. Neither path
calls a child service.

## Simulation Backend

On `a2a3sim` and `a5sim`, create uses a shared backing object rather than ACL
IPC.

The sim child creates a POSIX shm backing object sized to `total_bytes`. Naming
and cleanup should follow the existing sim CommDomain discipline: short
fixed-width shm names, per-region uniqueness, and stale-segment collision
avoidance using process/session identity plus monotonic `region_id`.

The child uses `shm_open(O_CREAT | O_EXCL | O_RDWR)`, `ftruncate(total_bytes)`,
and `mmap`. It initializes the counter range to zero, writes child-local mapped
addresses into the L2 descriptor, and publishes a reply containing the backing
token plus `mapping_bytes == total_bytes`.

The parent import step opens the same backing object and maps it in the parent
process. Parent-private metadata stores the parent-local base, payload pointer,
and counter pointer.

No shared-header ready barrier is needed for this one-child/one-parent mapping:
the create control is synchronous, and the child publishes the reply only after
the backing object is sized, mapped, and initialized. Release order is still
parent `munmap`/close first, then child release control `munmap`/close/unlink.
Child shutdown sweeps any leftover live shm-backed regions as a backstop.

This intentionally preserves the two-address model:

```text
L2 descriptor payload_base/counter_base = child-local addresses
Parent metadata payload/counter access = parent-local addresses
```

Tests should not rely on descriptor addresses being valid in the parent
process.

## Parent Metadata

`L3L2OrchRegion` stores two categories of state:

- the unchanged `L3L2OrchRegionDesc` returned by the child;
- parent-private direct-access metadata.

Suggested parent metadata:

```text
class L3L2ParentRegionAccess:
  worker_id
  region_id
  access_profile
  total_bytes
  payload_offset / payload_bytes
  counter_offset / counter_bytes
  parent_import_base or mapping_base
  parent_payload_access
  parent_counter_access
  import_handle / mmap handle / shm handle / release token
  counter_access_mode
  supports_direct_host_dma
  closed
```

This metadata is not exposed by `descriptor_scalars()`, is not included in
`TaskArgs`, and is not visible to L2.

The lifetime is:

```text
create reply received
parent imports/maps
region handle becomes live
region.free() marks logical release
Worker.run post-drain cleanup closes parent import
release control frees child allocation
```

If parent import fails after child allocation/export succeeds, the parent sends
release control for the partially created region only after closing any
parent-side import, mmap, or copy state that was opened. The rollback order is:

```text
child allocation/export succeeds
parent import/setup begins
parent import/setup fails
parent does not publish a live region handle
parent closes any opened import/mmap/copy state
parent sends release control for the child allocation
```

If parent close or child release cannot be confirmed, the create call fails
with cleanup uncertainty. The region still must not become live in user code.

## Payload Path

`payload_write(offset, host_buffer, nbytes)`:

1. Validate the region is live.
2. Resolve `host_buffer` to a parent-accessible contiguous byte span.
3. Validate `[offset, offset + nbytes)` is inside `payload_bytes`.
4. Transfer bytes from the parent span to
   `parent_payload_access + offset`.
5. Apply backend ordering/cache maintenance required before a later notify.

`payload_read(offset, host_buffer, nbytes)`:

1. Validate the region is live.
2. Resolve `host_buffer` to a parent-accessible writable contiguous byte span.
3. Validate `[offset, offset + nbytes)` is inside `payload_bytes`.
4. Transfer bytes from `parent_payload_access + offset` to the parent span.
5. Apply backend ordering/cache maintenance required after a matched signal.

Transfer selection:

1. Use direct parent DMA/copy when the caller buffer is legal for the backend.
2. Use parent-owned staging only for DMA registration, pinning, alignment,
   contiguity, writability, or lifetime constraints.
3. Never stage only to make the buffer visible to the child.
4. Never issue a child steady-state payload command.

This relaxes primitive payload inputs from child-visible runtime tensors to
parent-accessible contiguous byte spans where legal. Queue wrappers may still
create small temporary bytes for descriptor/header packing; the primitive layer
owns any staging needed to transfer those bytes.

Implementation order may preserve existing runtime `Tensor` payload behavior
first, then broaden accepted parent-contiguous buffer types once the primitive
buffer resolution and staging path exists. Production acceptance still requires
the final direct-parent implementation to preserve existing behavior and to
make any newly accepted buffer types fail before issue, without poisoning, when
they are unsupported, non-contiguous, or not writable for reads.

## Counter Path

Counter handles still validate offsets against the L2 descriptor size:

```text
0 <= offset
offset % 4 == 0
offset + 4 <= counter_bytes
```

The parent access address is derived from parent metadata:

```text
parent_counter_addr = parent_counter_access + offset
```

Do not derive parent access from descriptor `counter_base`.

`SIGNAL_NOTIFY`:

- `Set`: store the operand as `int32_t`.
- `Add`: load current `int32_t`, add operand, store result.
- Apply release/publish ordering before the store becomes visible.

`SIGNAL_TEST`:

- Load one `int32_t`.
- Compare with `EQ`, `NE`, `GT`, `GE`, `LT`, or `LE`.
- On match, apply acquire/observe ordering before protected payload reads.
- Mismatch is ordinary no-progress and does not poison.

`SIGNAL_WAIT`:

- Poll in the parent until match or finite timeout.
- Use the counter access path selected for the active backend.
- Timeout returns the last observed value and does not poison by itself.

Backend counter access is a parent direct-access operation. It must not call a
child service.

`NotifyOp.Add` remains a convenience read-modify-write operation. It does not
become an inter-agent atomic operation.

## Ordering And Cache Maintenance

The abstract contract is unchanged:

- payload writes before `signal_notify` are published by that notify;
- matched `signal_test` or `signal_wait` is the observe point before payload
  reads;
- failed `signal_test` does not establish observe semantics;
- waits use finite timeouts.

The L2 endpoint cache-maintenance behavior and ABI stay unchanged. The L3-side
responsibility moves from the child service to the parent direct-access backend.

Implementation must follow `docs/hardware/cache-coherency.md`:

- on a2a3, AICPU reads of host-DMA-published GM require the existing endpoint
  invalidation behavior;
- on a5, DMA/HBM reads are coherent with AICPU and should not add unnecessary
  invalidates;
- AICore-published data still relies on AICore-side `dcci` before signaling;
- CPU ordering around publish/observe must be explicit in the backend.

Do not add a user-visible environment variable or runtime knob for this.

## Error And Poison Ownership

`L3L2OrchRegion` remains the owner of parent-side live/released/poison state.
The native parent direct-access helpers own only imported mapping/copy
resources and report enough structured failure detail for Python to preserve
the existing visible behavior:

- pre-issue validation failures raise without poisoning;
- parent import, transfer, counter access, or close failures after issue poison
  only the affected parent region;
- `SIGNAL_TEST` mismatch is ordinary no-progress and does not poison;
- `SIGNAL_WAIT` timeout reports the last observed value and does not poison by
  itself;
- L2 endpoint fatal text containing a valid `region_id` still poisons only the
  matching live parent region.

Do not duplicate live/released/poison state in the native helper. If the native
resource handle has its own closed flag, it is only a resource-lifetime guard
and not the public region state.

## Release And Cleanup

`region.free()` performs logical release only:

```text
mark parent handle released
reject future payload/counter/descriptor operations
append region to pending L3-L2 region releases
```

It must not synchronously send `CTRL_L3_L2_REGION_RELEASE`, because a running L2
task may still hold the descriptor and may still occupy the same mailbox.

Post-drain cleanup order:

```text
Worker.run drains submitted L2 work
close parent import/mmap/copy handles
send CTRL_L3_L2_REGION_RELEASE through the mailbox
child frees child-owned physical allocation
expire the parent region object
```

This order means release control naturally runs only after the L2 task that
could use the descriptor has completed. That is a lifetime guarantee, not a
performance problem.

If the L3 orchestration function exits with live regions, cleanup applies the
same sequence after drain. If the parent process fails before it can release,
child shutdown/finalize sweeps remaining child-owned regions.

Rollback order for create failure:

```text
child allocation/export succeeds
parent import/setup begins
parent import/setup fails
parent does not publish a live region handle
parent closes any partially opened import/mmap/copy state
parent sends release control for the region id
child frees physical allocation
```

If parent close fails, report cleanup uncertainty and still avoid publishing a
live handle. If child release fails after parent close, report cleanup
uncertainty and rely on child shutdown/finalize as the owner-side backstop.
Cleanup helpers must tolerate already-closed or never-opened parent mappings.

## Affected Files

Python parent facade:

- `python/simpler/l3_l2_orch_comm.py`
  - replace steady-state `L3L2OrchCommClient` submission in region/counter
    methods with direct parent backend calls;
  - keep `L3L2OrchRegionDesc`, `NotifyOp`, `WaitCmp`, and
    `SignalTestResult`;
  - add parent metadata/access classes;
  - keep live/released/poison policy in Python;
  - broaden buffer resolution to parent-contiguous byte spans.
- `python/simpler/l3_l2_message_queue.py`
  - remove assumptions that queue-owned scratch must be child-visible;
  - keep protocol, layout, counters, opcodes, and observable behavior.
- `python/simpler/worker.py`
  - replace `_ensure_l3_l2_orch_comm` service bootstrap with lifecycle create
    and release helpers;
  - add one-shot shm request/reply encoding and decoding;
  - keep live/pending region lists and run cleanup ordering;
  - remove child-visible host-buffer registration as a primitive requirement.

Python bindings and worker mailbox:

- `python/bindings/task_interface.cpp`
  - add private `_task_interface` helpers for parent import/map, payload
    copy, counter operations, and parent close;
  - implement these helpers in the existing nanobind module rather than a new
    binding source file.
- `python/bindings/worker_bind.h`
  - expose `control_l3_l2_region_create` and
    `control_l3_l2_region_release`.
- `src/common/hierarchical/worker_manager.h`
- `src/common/hierarchical/worker_manager.cpp`
  - add control ids and typed forwarding methods;
  - serialize them with task dispatch through the existing mailbox mutex.

Child process control handlers:

- `python/simpler/worker.py`
  - add `_handle_ctrl_l3_l2_region_create`;
  - add `_handle_ctrl_l3_l2_region_release`;
  - sweep live child-owned regions during shutdown.

Platform backend:

- `src/common/platform/include/host/l3_l2_orch_region_access.h`
  - declare host-only parent direct-access metadata and helper contracts;
  - keep parent import/mmap handles out of
    `src/common/platform/include/common/l3_l2_orch_comm.h`.
- `src/common/platform/include/host/l3_l2_orch_comm_service.h`
- `src/common/platform/shared/host/l3_l2_orch_comm_service.cpp`
  - remove the production persistent service and request/response data path;
  - keep or move reusable validation helpers if needed.
- `src/common/platform/onboard/host/device_runner_base.{h,cpp}`
  - replace service start/stop hooks with region allocate/export/release
    helpers and parent import support as needed.
- `src/common/platform/sim/host/device_runner_base.{h,cpp}`
  - replace service allocation with POSIX shm-backed sim region helpers.
- `src/common/platform/onboard/host/c_api_shared.cpp`
- `src/common/platform/sim/host/c_api_shared.cpp`
  - remove `l3_l2_orch_comm_init_ctx` / shutdown service exports from the
    production ABI.
- `src/a2a3/platform/*/host/CMakeLists.txt`
- `src/a5/platform/*/host/CMakeLists.txt`
  - drop the service source from production builds after replacement.
- `src/common/worker/chip_worker.h`
- `src/common/worker/chip_worker.cpp`
- `src/common/worker/pto_runtime_c_api.h`
  - remove old `l3_l2_orch_comm_init` / shutdown service bootstrap symbols
    from the production path.

Preserved ABI headers:

- `src/common/platform/include/common/l3_l2_orch_comm.h`
  - keep descriptor, validation helpers, notify/wait enums, and scalar count.
- `src/common/platform/include/aicpu/l3_l2_orch_endpoint.h`
  - keep L2 endpoint ABI and behavior.
- `src/common/platform/include/aicpu/l3_l2_message_queue.h`
  - keep queue endpoint ABI and behavior.

Documentation:

- `docs/l3-l2-orch-comm.md`
- `docs/l3-l2-message-queue.md`
  - update in the same migration that removes the production child service
    path so user-facing docs describe implemented behavior.

Tests and examples:

- `tests/ut/py/test_worker/test_l3_l2_orch_comm.py`
- `tests/ut/py/test_worker/test_l3_l2_message_queue.py`
- `tests/ut/py/test_worker/test_host_worker.py`
- `tests/ut/py/test_chip_worker.py`
- `tests/ut/cpp/common/test_l3_l2_orch_comm.cpp`
- `tests/ut/cpp/common/test_l3_l2_orch_endpoint.cpp`
- `tests/ut/cpp/common/test_l3_l2_message_queue.cpp`
- remove or rewrite `tests/ut/cpp/common/test_l3_l2_orch_comm_service.cpp`;
- remove or rewrite service runner tests under `tests/ut/cpp/common` and
  `tests/ut/cpp/hardware` that assert the old child command lane;
- keep examples under
  `examples/workers/l3/l3_l2_orch_comm_stream` and
  `examples/workers/l3/l3_l2_message_queue` behavior-compatible.

## Migration Sequence

Plan the migration as two reviewable PRs. The temporary state after PR1 may
keep the old child service available for onboard or otherwise-unconverted
paths, but it is not the production end state. PR2 removes the production child
service path.

### PR1: Lifecycle + Sim Direct Parent Path

Goal: prove the direct-parent architecture end-to-end on simulation while
leaving onboard on the existing path until the ACL IPC spike is complete.

Scope:

1. Add the create request/reply structs, release scalar control, concrete
   mailbox control ids, bindings, and Python encoding helpers.
2. Implement child create/release handlers with a single-region allocation and
   layout partitioning.
3. Implement sim POSIX shm backing create/import/close/release.
4. Add `src/common/platform/include/host/l3_l2_orch_region_access.h` and
   private `_task_interface` parent helpers in the existing
   `python/bindings/task_interface.cpp`.
5. Add parent metadata and direct sim payload/counter operations.
6. Switch `L3L2OrchRegion` methods to the parent backend for sim.
7. Implement create rollback, post-drain cleanup, and Python-owned poison
   policy for the direct path.
8. Move queue scratch/staging policy down to primitive payload transfer where
   needed for sim behavior.
9. Rewrite or add sim-focused tests away from child steady-state command
   assertions toward behavior and direct-path assertions.

Keep scope controlled in PR1: preserving existing runtime `Tensor` payload
behavior is required, while broader `bytes` / `bytearray` / generic buffer
support may land later in the migration if needed. PR1 must not present the old
child service as the final fallback; it is only a temporary migration state for
paths not yet converted.

### PR2: Onboard Direct Path + Remove Old Service

Goal: complete onboard direct parent access and remove the PR #1015 production
child service path.

Scope:

1. Run and record the onboard ACL IPC spike conclusions in code comments or
   docs where they affect implementation choices.
2. Implement onboard child GM allocate/export/import-pid authorization.
3. Implement onboard parent ACL import, H2D/D2H copy, counter access, and
   import close through the private `_task_interface` helpers.
4. Select one production counter access path per onboard platform.
5. Change cleanup to close parent imports post-drain, then send release
   control.
6. Switch onboard `L3L2OrchRegion` methods to the parent backend.
7. Remove the persistent shared-memory service bootstrap and steady-state
   command structs from production code.
8. Delete the production `l3_l2_orch_comm_init_ctx` / shutdown bootstrap
   symbols and `control_l3_l2_orch_comm_init`.
9. Drop the old service source from production builds and delete or rewrite
   service-specific tests.
10. Update `docs/l3-l2-orch-comm.md` and `docs/l3-l2-message-queue.md` in the
    same PR.
11. Run final sim and onboard validation for the primitive and queue examples.

Temporary bring-up fallback may exist only on a development branch. After PR2,
production code must not keep the PR #1015 child service as a fallback.

## Validation Plan

Unit validation:

- descriptor scalar count remains six;
- descriptor values remain child-side addresses;
- create reply includes parent metadata but `descriptor_scalars()` does not;
- create reply layout is validated from descriptor plus fixed layout rules;
- create failure rolls back child allocation/export;
- `region.free()` is logical and does not synchronously send release;
- post-drain cleanup closes parent import before child physical free;
- shutdown sweeps unreleased child-owned regions;
- sim stores parent and child local address fields separately;
- parent operations never dereference descriptor addresses;
- fake or unit backend coverage injects different descriptor and parent access
  addresses so tests are not dependent on OS-chosen mmap addresses.

Primitive behavior:

- `payload_write` and `payload_read` work with runtime tensors;
- accepted parent-contiguous byte buffers work where backend policy allows;
- invalid, non-contiguous, or out-of-bounds buffers fail before transfer and do
  not poison;
- transfer failure after issue poisons only the affected region;
- `SIGNAL_NOTIFY`, `SIGNAL_TEST`, and `SIGNAL_WAIT` execute without child
  service commands;
- `SIGNAL_TEST` mismatch and `SIGNAL_WAIT` timeout remain non-poisoning;
- `NotifyOp.Add` remains non-atomic read-modify-write convenience.

Queue behavior:

- layout mirror tests still pass;
- input/output queue behavior remains unchanged;
- STOP, ERROR, ordinary backpressure, peer abort, and input-window behavior
  remain unchanged;
- queue no longer owns child-visible registered scratch as a separate data
  movement policy.

Platform examples:

- `examples/workers/l3/l3_l2_orch_comm_stream` passes on `a2a3sim`,
  `a2a3`, `a5sim`, and `a5`;
- `examples/workers/l3/l3_l2_message_queue` passes on `a2a3sim`, `a2a3`,
  `a5sim`, and `a5`.

Onboard tests must follow the repository hardware rules: run through
`task-submit` when touching NPU hardware and run the architecture precheck
before onboard `a2a3` or `a5` commands.

Acceptance is met when production steady-state payload and counter operations
execute in the L3 parent process, lifecycle uses typed mailbox controls, the
old child service data path is removed, and existing supported primitive and
queue behavior is preserved.
