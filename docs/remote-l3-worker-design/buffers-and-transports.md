# Remote L3 Buffers and Transports

This document defines remote buffer lifetime and backend transport contracts
for remote L3 NEXT_LEVEL endpoints.

## Buffer Handles

Remote buffers need an owner, generation, and deferred free point. The parent
tracks user-visible handle state; the remote session runner owns physical
memory and imported mappings.

Parent-side handle:

```text
RemoteBufferHandle:
  endpoint_id
  owner_endpoint_id
  buffer_id
  generation
  import_id
  address_space
  nbytes
  offset
  remote_addr
  rkey_or_token
  ub_ldst_va
  access_flags
  ref_state
  live_slot_refs
```

For an owner allocation, `endpoint_id == owner_endpoint_id` and
`import_id == 0`. For an imported peer mapping, `endpoint_id` is the importer
that can consume the handle, `owner_endpoint_id` is the endpoint that owns the
physical allocation, and `import_id` names the importer-local mapping. The
public Python object stays opaque; it may expose endpoint, owner endpoint, size,
address space, and release state, but not raw transport keys.

`buffer_id` may be reused only with a new `generation`. Stale completions,
imports, releases, or frees whose generation does not match are ignored or
reported as session errors.

## Public Memory API

Remote memory APIs return handles, not bare integer pointers:

```python
buf = w4.remote_malloc(worker=l3_worker_id, nbytes=4096)
w4.remote_copy_to(buf, host_ptr, 4096)

exported = w4.remote_export(buf, offset=0, nbytes=4096, access="read")
peer = w4.remote_import(exported, worker=peer_l3_worker_id)

args = TaskArgs()
args.add_tensor(
    RemoteTensorRef(peer, offset=0, shape=(1024,), dtype=DataType.FLOAT32),
    TensorArgType.INPUT,
)
orch.submit_next_level(l3_handle, args, cfg, worker=peer_l3_worker_id)
orch.drain()

w4.remote_release_import(peer)
w4.remote_free(buf)
```

The binding stores a hidden remote sidecar beside the tensor metadata.
`RemoteTensorRef` is transport metadata, not an extension of the local mailbox
ABI. Local fork/shm endpoints reject it unless the buffer has first been
explicitly imported, staged, or materialized into a local-addressable
`ContinuousTensor`. Remote endpoints reject bare host pointers unless explicit
staging produced a handle.

`remote_export()` returns an opaque session-scoped export descriptor. It does
not expose transport keys such as remote addresses, rkeys/tokens, UB LD/ST
addresses, or transport descriptors, and it does not create a new
user-releasable allocation. `remote_import()` consumes that descriptor on a
target endpoint and returns an imported `RemoteBufferHandle`.
`remote_release_import()` releases the importer-local mapping after all slot
refs on that imported handle have drained. `remote_free()` on the owner handle
releases the owner allocation after all imports and slot refs have drained.

Current status: Python exposes `RemoteBufferHandle` and `RemoteTensorRef`.
`TaskArgs.add_tensor(RemoteTensorRef(...), tag)` stores
`ContinuousTensor.data == 0` in the underlying `TaskArgs` and keeps the remote
descriptor in a same-index sidecar. Public `remote_malloc`, `remote_free`,
`remote_copy_to`, `remote_copy_from`, `remote_export`, `remote_import`, and
`remote_release_import` are implemented for the simulation backend.

## TaskArgs Sidecar Contract

`ContinuousTensor` remains the tensor metadata ABI. Remote transport metadata
is stored in a per-`TaskArgs` sidecar keyed by tensor index.

Python-facing rules:

- `TaskArgs.add_tensor(RemoteTensorRef(...), tag)` appends one normal tensor
  metadata entry and one remote sidecar entry at the same tensor index.
- `RemoteTensorRef` is not converted to a fake integer pointer.
- `RemoteBufferHandle` is opaque to user code. Users may inspect endpoint,
  size, and release state, but not transport keys such as `rkey`.
- Submit validates the remote handle's access flags against the tensor tag:
  `INPUT` requires read, `OUTPUT` and `OUTPUT_EXISTING` require write, and
  `INOUT` and remote `NO_DEP` require read/write. `NO_DEP` skips dependency
  inference only; it does not weaken remote memory access requirements.
- A `TaskArgs` containing any remote sidecar is legal only when the final
  selected endpoint set contains remote endpoints that can consume every
  referenced sidecar.
- Local `submit_next_level()` rejects remote sidecars before slot commit unless
  an explicit import/staging API has converted them into local-addressable
  `ContinuousTensor` values and removed the remote sidecar.
- Remote submit rejects `OUTPUT` tensors with `ContinuousTensor.data == 0`
  unless an explicit remote allocation API has already produced a
  `RemoteTensorRef` sidecar for that tensor.

Parent C++ slot rules:

- `TaskSlotState` owns a copy of the sidecar for the slot lifetime.
- Sidecar length must equal `tensor_count`; entries can be null only for
  metadata-only tensors. `HOST_INLINE` payloads still have sidecar descriptors
  so the frame codec has one validation path for every remote data payload.
- Submit validation captures a live ref on every referenced
  `RemoteBufferHandle` before the slot becomes visible to the Scheduler.
- Validation fails before slot commit when the intersection of callable-eligible
  endpoints and data-eligible endpoints is empty, or when a remote tensor names
  an ineligible endpoint, stale generation, out-of-range offset, or released
  handle.
- Group submit stores one sidecar per group member, aligned with
  `task_args_list[i]`.

The current C++ implementation enforces these submit-time rules for
`submit_next_level()` and group submit. It captures refs for owner-side
and imported simulation handles.

Endpoint rules:

- `LocalMailboxEndpoint` rejects non-empty sidecars. It cannot encode remote
  descriptors into the local fixed-size mailbox, and its child processes expect
  `ContinuousTensor.data` to be a local host/shm pointer or a local child-memory
  pointer.
- `RemoteL3Endpoint` requires a sidecar for every tensor payload that crosses
  the remote protocol, including `HOST_INLINE` payloads.
- Remote TASK frames write `ContinuousTensorWire.data == 0`; parent virtual
  addresses never cross the remote protocol.
- A remote tensor with `child_memory=True` and no sidecar is invalid. Local
  child-memory pointers are meaningful only inside fork/shm topology.
- The remote session runner translates each `RemoteTensorDesc` into a local
  `ContinuousTensor` and fills `data` from its validated local mapping
  immediately before invoking `inner_worker.run()`.

## Remote OUTPUT Allocation Policy

The first implementation does not mirror local HeapRing auto-allocation for
remote outputs. In the local fork/shm path, an `OUTPUT` tensor with
`ContinuousTensor.data == 0` is assigned a parent HeapRing pointer during
submit, and forked children can dereference that shared virtual address. A
remote L3 worker cannot use a parent-host HeapRing pointer.

Remote callers must allocate or import output storage explicitly before submit:

```python
out = w4.remote_malloc(worker=l3_worker_id, nbytes=4096)

args = TaskArgs()
args.add_tensor(
    RemoteTensorRef(out, offset=0, shape=(1024,), dtype=DataType.FLOAT32),
    TensorArgType.OUTPUT,
)
orch.submit_next_level(l3_handle, args, cfg, worker=l3_worker_id)
```

This keeps submit-time validation simple: the slot already carries complete
data eligibility, handle generation, bounds, and lifetime refs before it
becomes visible to the Scheduler.

Future work may add remote output auto-allocation, but only after the runtime
has a well-defined pre-dispatch endpoint selection policy. Auto-allocation must
decide which endpoint owns the output before slot commit, allocate or import the
remote buffer, attach the generated sidecar to the correct tensor index, and
handle group submits where each member may need storage on a different
endpoint. Until those rules exist, remote null `OUTPUT` tensors fail fast.

The implemented path follows this policy: local `OUTPUT` tensors still use the
HeapRing auto-allocation path, while remote OUTPUT tensors with sidecars skip
local HeapRing allocation and register remote dependency keys.

## Required Controls

| Command | Purpose |
| ------- | ------- |
| `ALLOC_REMOTE_BUFFER` | Allocate remote L3 host or chip memory. |
| `FREE_REMOTE_BUFFER` | Mark a handle released; physical free is deferred. |
| `COPY_TO_REMOTE` | Stage host data into a remote buffer. |
| `COPY_FROM_REMOTE` | Pull remote output data back to host. |
| `EXPORT_BUFFER` | Return RDMA key or UB mapping metadata. |
| `IMPORT_BUFFER` | Import a peer buffer into a remote worker. |
| `RELEASE_IMPORT` | Drop an imported peer mapping. |

Control commands are typed remote protocol frames. They are not the local
mailbox `CTRL_*` integers.

## Export/Import Handle Semantics

Owner allocations and imported mappings are separate handle kinds over the
same physical buffer:

```text
owner handle:
  endpoint_id = owner_endpoint_id = owner endpoint
  buffer_id/generation = owner allocation identity
  import_id = 0
  address_space = REMOTE_DEVICE

imported handle:
  endpoint_id = importer endpoint
  owner_endpoint_id = original owner endpoint
  buffer_id/generation = owner allocation identity
  import_id = importer-local mapping id
  address_space = REMOTE_WINDOW or UB_LDST
```

The parent builds TASK descriptors from the handle that will be consumed by
the selected endpoint. For owner handles, the selected endpoint must be the
owner. For imported handles, the selected endpoint must be the importer. In
both cases the descriptor preserves the original owner identity so dependency
tracking sees all views of the same buffer as the same producer/consumer key
when their logical start offset matches.

Import does not clone storage. It creates a peer-visible mapping of an owner
allocation range with a bounded access mask:

- `read` imports can be used for INPUT tensors.
- `write` imports can be used for OUTPUT tensors.
- `readwrite` imports can be used for INOUT or mixed flows.
- Submit validation rejects a tensor whose tag requires an access bit that the
  handle does not carry.

`EXPORT_BUFFER` is sent only to the owner endpoint and returns an opaque
transport descriptor. The parent may keep this descriptor long enough to import
the range into one or more peer endpoints, but Python user code must not depend
on raw `rkey`, HCOMM descriptor, or UB address values.

`IMPORT_BUFFER` is sent to each importer endpoint. A successful reply creates
data eligibility for that endpoint only. It does not make the owner endpoint,
other importers, or local fork/shm endpoints eligible.

`RELEASE_IMPORT` is sent only to the importer endpoint. It tears down the
importer-local mapping and removes that imported handle from future endpoint
eligibility. It does not free the owner allocation and does not release imports
on other endpoints.

If a multi-endpoint import setup partially fails, the parent must roll back
successful imports by sending `RELEASE_IMPORT`. If rollback is uncertain, the
affected endpoint/buffer pair is removed from eligibility for the rest of the
session. The owner allocation remains live until the owner handle is freed and
all confirmed or uncertain imports have been resolved or the session shuts
down.

## Release Policy

- Slot refs are acquired during `submit_next_level()` while walking `TaskArgs`.
- Captured buffers stay live until every capturing slot has reached a terminal
  state and every producer/consumer reference that can expose that buffer has
  reached `CONSUMED` or failed cleanup.
- Explicit buffers used only as INPUT still need slot refs. They have no
  producer slot to protect them.
- Runtime-managed OUTPUT buffers follow the producer slot's terminal cleanup.
- `remote_free(owner_handle)` marks the owner handle released.
  `FREE_REMOTE_BUFFER` physical free runs only when the owner handle has no
  live slot refs and no live imported mappings.
- `remote_release_import(imported_handle)` marks the imported handle released.
  `RELEASE_IMPORT` teardown runs only when that imported handle has no live
  slot refs.
- `FREE_REMOTE_BUFFER` is invalid for imported handles, and `RELEASE_IMPORT`
  is invalid for owner handles.
- If a run fails, the same post-drain cleanup path runs before the next run.
- Session shutdown rejects new work, releases importer mappings first, and
  then frees session-owned owner allocations after completing or failing
  in-flight tasks.

The registry therefore needs both a user-visible release state and a live
slot-ref count. Failed slots still release captured refs through the same
terminal cleanup path as successful slots.

## Dependency Keys

`TensorKey` must grow beyond the current `{ptr, int8 worker}` shape for remote
buffers while preserving the current exact-start lookup semantics. Today, local
dependency tracking keys only on a tensor's start pointer and worker id; shape
and byte length do not participate in lookup. Remote keys follow the same rule:

```text
address_kind
owner_endpoint_id
buffer_id
offset_begin
generation
```

Local fork/shm keys remain a compatibility subset:

```text
host pointer:        (LOCAL_HOST, -1, ptr)
local child memory:  (LOCAL_CHILD, worker_id, ptr)
```

Known limitation: two remote tensors that reference overlapping byte ranges
with different `offset_begin` values do not automatically depend on each other.
For example, a producer writing `[0, 4096)` and a consumer reading
`[1024, 2048)` map to different dependency keys. This matches the current local
`ptr`-based TensorMap behavior, where a subview at `base + offset` is a
different key from `base`.

The first implementation chooses this route to keep remote scheduling behavior
compatible with local fork/shm semantics and to avoid changing TensorMap into a
range index as part of the transport bring-up. `offset_end`/`nbytes` remains in
`RemoteTensorDesc` for bounds checks and for a future range-overlap TensorMap
upgrade, but it is not part of the first dependency key.

## HCOMM Adapter Contract

Remote L3 uses HCOMM for steady-state communication in the first
implementation. The bootstrap socket is only a setup path for session
validation and HCOMM bring-up. Once HCOMM RPC is ready, task metadata,
CONTROL, CONTROL_REPLY, COMPLETION, and SHUTDOWN frames use the HCOMM RPC
adapter; tensor data and remote buffer copies use the HCOMM data adapter.

The endpoint owns the adapter objects. `Orchestrator`, Scheduler, and
`WorkerThread` see only `WorkerEndpoint::run()`, `WorkerEndpoint::control()`,
and logical capability bits from `WorkerEndpoint::caps()`.

Current status: `RemoteL3Endpoint` owns a transport-neutral
`RemoteL3Transport` and the ordered TASK/COMPLETION boundary. The HCOMM RPC
adapter, data adapter, buffer registry, and health lane described below remain
to be implemented behind that transport boundary.

The adapter family provides this logical contract:

```text
BootstrapSocketAdapter:
  open(control_uri)
  exchange hello/capability/HCOMM bootstrap frames
  close after HCOMM_RPC_READY

HcommRpcAdapter:
  submit TASK or CONTROL frame on the ordered command lane
  wait for matching COMPLETION or CONTROL_REPLY
  send SHUTDOWN when no command is in flight

HcommAdapter:
  register/export/import memory
  submit read/write/copy plans
  wait/fence completion
  release registrations and imports
```

Profile-specific buffer rules:

- The simulation profile uses session-local `export_id` and `import_id` tokens.
  It must not expose host process pointers through the public Python API.
- RoCE and HCCS profiles encode HCOMM/RDMA registration metadata in
  `rkey_or_token` and bounded transport descriptors. Import validates the
  profile before publishing a peer mapping.
- UB profiles may populate `ub_ldst_va` only after the mapping and fence rules
  are proven for that platform. Bulk copies still use the HCOMM data adapter
  unless the selected UB profile explicitly supports LD/ST completion ordering.

HCOMM RPC enqueues request frames on the endpoint's ordered command lane.
Data-plane transfers may use RoCE, HCCS, or UB HCOMM profiles, but TASK
doorbells, CONTROL frames, replies, completions, and shutdown state are ordered
by the command lane. Reply frames carry the request sequence they answer. A
task completes only after an explicit remote `COMPLETION` frame; data-copy
completion alone is never a task completion signal. `control()` returns only
after the matching `CONTROL_REPLY` arrives or a timeout/disconnect is converted
into a failed reply. Liveness is handled by an independent health lane or
transport keepalive; it is not queued behind the ordered command lane.

## A2 RoCE HCOMM Profile

- Use HCOMM with `COMM_PROTOCOL_ROCE`.
- Carry command frames and completion records through HCOMM RPC rings and
  notify/fence operations.
- Use a separate health HCOMM lane or transport keepalive for liveness.
- Use registered staging buffers for large callable blobs and bulk data.
- Export buffers as HCOMM memory descriptors plus RoCE-specific channel
  metadata.
- Complete tasks only after an HCOMM RPC `COMPLETION` frame from the session
  runner.
- Bound every wait with a timeout and convert disconnects into endpoint
  failure completions.

## A3 HCCS HCOMM Profile

- Keep the same HCOMM adapter contract as A2.
- Implement memory export/import through the HCCS-capable HCOMM profile.
- Preserve the same command-lane ordering rules: task/control frames are
  observed in sequence order, command frame visible before doorbell, and remote
  writes complete before completion frame.
- Provide health independently from command-lane progress so long-running TASK
  execution does not cause false endpoint failure.
- Reuse the frame codec, HCOMM RPC, and buffer registry tests from the A2 path.

## A5 UB HCOMM Profile

- Export both RDMA metadata and, when available, an LD/ST mapping token.
- Use LD/ST for doorbells and small completion records only when the mapping
  is coherent for the participating hosts.
- Preserve the same per-endpoint command-lane order for TASK, CONTROL,
  CONTROL_REPLY, COMPLETION, and SHUTDOWN frames.
- Keep UB health doorbells or transport health independent from the command
  lane used for state-changing frames.
- Use RDMA for bulk transfers until platform benchmarks justify LD/ST bulk
  copies.
- Add explicit fences around:
  - task payload writes before doorbell;
  - remote output writes before completion;
  - parent completion read before dependent task dispatch.
- Keep RDMA fallback for all UB LD/ST paths.

## Simulation Backend

The simulation backend uses TCP or Unix sockets plus local files/shm for
integration tests. It must exercise:

- framed protocol encode/decode;
- sequence numbers;
- remote callable bootstrap;
- endpoint eligibility validation;
- success and error completions;
- failed dependency poisoning;
- buffer registry ref capture and deferred free;
- timeout handling.

It must not depend on A2/A3/A5 hardware.
