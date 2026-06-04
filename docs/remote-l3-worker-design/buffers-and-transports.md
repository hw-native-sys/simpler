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
  buffer_id
  generation
  address_space
  nbytes
  remote_addr
  rkey_or_token
  ub_ldst_va
  ref_state
  live_slot_refs
```

`buffer_id` may be reused only with a new `generation`. Stale completions,
imports, or frees whose generation does not match are ignored or reported as
session errors.

## Public Memory API

Remote memory APIs return handles, not bare integer pointers:

```python
buf = w4.remote_malloc(worker=l3_worker_id, nbytes=4096)
w4.remote_copy_to(buf, host_ptr, 4096)

args = TaskArgs()
args.add_tensor(
    RemoteTensorRef(buf, offset=0, shape=(1024,), dtype=DataType.FLOAT32),
    TensorArgType.INPUT,
)
```

The binding stores a hidden remote sidecar beside the tensor metadata.
`RemoteTensorRef` is transport metadata, not an extension of the local mailbox
ABI. Local fork/shm endpoints reject it unless the buffer has first been
explicitly imported, staged, or materialized into a local-addressable
`ContinuousTensor`. Remote endpoints reject bare host pointers unless explicit
staging produced a handle.

## TaskArgs Sidecar Contract

`ContinuousTensor` remains the tensor metadata ABI. Remote transport metadata
is stored in a per-`TaskArgs` sidecar keyed by tensor index.

Python-facing rules:

- `TaskArgs.add_tensor(RemoteTensorRef(...), tag)` appends one normal tensor
  metadata entry and one remote sidecar entry at the same tensor index.
- `RemoteTensorRef` is not converted to a fake integer pointer.
- `RemoteBufferHandle` is opaque to user code. Users may inspect endpoint,
  size, and release state, but not transport keys such as `rkey`.
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

Endpoint rules:

- `LocalMailboxEndpoint` rejects non-empty sidecars. It cannot encode remote
  descriptors into the local 4096-byte mailbox, and its child processes expect
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

## Release Policy

- Slot refs are acquired during `submit_next_level()` while walking `TaskArgs`.
- Captured buffers stay live until every capturing slot has reached a terminal
  state and every producer/consumer reference that can expose that buffer has
  reached `CONSUMED` or failed cleanup.
- Explicit buffers used only as INPUT still need slot refs. They have no
  producer slot to protect them.
- Runtime-managed OUTPUT buffers follow the producer slot's terminal cleanup.
- `FREE_REMOTE_BUFFER` and `RELEASE_IMPORT` mark handles released. Physical
  free or import teardown runs only when the released handle has no live slot
  refs.
- If a run fails, the same post-drain cleanup path runs before the next run.
- Session shutdown rejects new work and frees all session-owned buffers after
  completing or failing in-flight tasks.

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
