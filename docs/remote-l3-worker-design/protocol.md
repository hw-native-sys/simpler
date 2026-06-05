# Remote L3 Protocol

This document defines the remote wire protocol used by
`RemoteL3Endpoint`. The local fork/shm path keeps the existing mailbox layout
behind `LocalMailboxEndpoint`.

The bootstrap socket carries only setup and HCOMM bring-up frames. After HCOMM
RPC is ready, steady-state TASK, CONTROL, CONTROL_REPLY, COMPLETION, and
SHUTDOWN frames travel through the HCOMM RPC adapter. Tensor data moves through
the HCOMM data adapter and is referenced from TASK frames by descriptors, not
by host pointers.

## Frames

Remote transport must not reuse the raw 4096-byte mailbox format. Define a
versioned frame header:

```text
FrameHeader:
  magic = "SLR3"
  version = 1
  frame_type =
    HELLO | TASK | CONTROL | CONTROL_REPLY | COMPLETION | HEALTH | SHUTDOWN
  session_id
  endpoint_id
  sequence
  payload_bytes
  flags
```

Rules:

- `magic` and `version` are validated before reading payload fields.
- `session_id` protects against stale frames from prior sessions.
- `endpoint_id` must match the logical NEXT_LEVEL worker selected by the
  parent scheduler.
- For parent-to-runner command requests, `sequence` is monotonic per endpoint
  on the ordered command lane. This includes `TASK`, state-changing `CONTROL`,
  and `SHUTDOWN`.
- `HEALTH` frames travel on an independent health lane, or use an equivalent
  transport-level health signal. They carry their own health sequence and must
  not be queued behind long-running TASK execution.
- Reply frames such as `COMPLETION` and `CONTROL_REPLY` carry the sequence of
  the request they answer. They do not allocate a new command sequence.
- `payload_bytes` is bounds-checked before allocation or decode.
- Unknown frame types, versions, flags, or oversized payloads fail the
  session instead of falling back to best-effort parsing.

## Wire Encoding

Remote frames use canonical field encoding. They do not memcpy C++ POD structs
such as `CallConfig` or `ContinuousTensor` onto the wire. The local fork/shm
mailbox may continue to use the raw POD layout because both endpoints are
fork-related processes running the same binary; remote endpoints must treat the
wire schema below as the compatibility contract.

Encoding rules:

- Multi-byte integers are little-endian.
- Boolean values are encoded as `uint8`, where `0` is false and `1` is true.
- Enum values are encoded as fixed-width integers. Unknown enum values reject
  the frame.
- Strings are `uint32 byte_len` followed by UTF-8 bytes, with per-field maximum
  lengths.
- Bounded byte blobs are `uint32 byte_len` followed by raw bytes, with
  per-field maximum lengths.
- Repeated fields are `uint32 count` followed by that many elements, with
  configured maximum counts.
- Reserved fields must be written as zero and rejected when non-zero unless a
  later protocol version assigns them.

`CallConfigWire v1` encodes the current `CallConfig` fields explicitly:

```text
block_dim: int32
aicpu_thread_num: int32
enable_l2_swimlane: int32
enable_dump_tensor: int32
enable_pmu: int32
enable_dep_gen: int32
enable_scope_stats: int32
output_prefix: string(max=1024)
```

The local codec implementation in `src/common/hierarchical/remote_wire.*`
encodes these fields explicitly and rejects drift through decode-time bounds
checks. The local fork/shm mailbox continues to memcpy the packed `CallConfig`
POD because it is same-binary IPC, not the cross-host protocol.

`ContinuousTensorWire v1` encodes tensor metadata explicitly. In remote TASK
frames, `data` is not a transferable pointer; it is reserved and must be zero.

```text
data: uint64  # reserved in remote TASK frames; must be 0
shapes: uint32[CONTINUOUS_TENSOR_MAX_DIMS]
ndims: uint32
dtype: uint32
child_memory: uint8
reserved: uint8[7]
```

The session runner decodes these wire records into local `CallConfig` and
`ContinuousTensor` values before calling `inner_worker.run()`. For tensors with
a remote descriptor, the runner fills the local `ContinuousTensor.data` from
its buffer/import registry after validating the descriptor. Local ABI structs
therefore remain the in-process execution ABI, not the remote transport ABI.

## HELLO Payload

`HELLO` is a post-prestart command-lane handshake. The session runner sends or
accepts it only after it has read the bootstrap manifest from the daemon
handoff, constructed `Worker(level=3)`, performed an explicit prestart that
forks local chip/sub children, started the inner scheduler, installed bootstrap
registries, initialized the buffer/import registry, and brought up the command
and health lanes. `HELLO` does not carry the bootstrap manifest.

```text
session_id
endpoint_id
protocol_version
comm profile
feature flags
ready_state
```

The parent treats the endpoint as schedulable only after the `HELLO` exchange
confirms `ready_state=READY`, matching `session_id`, matching `endpoint_id`,
and compatible protocol and feature sets.

`READY` is a scheduling barrier. A session that can answer liveness probes but
has not completed prestart reports a non-ready state and must not receive TASK
frames.

## TASK Payload

```text
callable_hash_digest: uint8[32]
config: CallConfigWire v1
args: RemoteTaskArgsWire v1
```

The TASK payload has exactly these top-level fields: callable digest,
`CallConfigWire`, and `RemoteTaskArgsWire`. `RemoteTaskArgsWire` then carries
tensor metadata, remote tensor descriptors, scalars, and optional inline bytes.

TASK frames carry the raw digest from the submitted parent-facing
`CallableHandle.hashid`. Parent-side dependency inference has already consumed
`TaskArgs` tags; tags are Orchestrator input, not the remote L3 wire contract.
A TASK frame always resolves `callable_hash_digest` in the remote TASK
dispatcher registry. The dispatcher is not a Worker: it resolves the digest to
its private orchestration callable slot, materializes the remote arguments, and
then invokes the embedded `inner_worker = Worker(level=3)`. The endpoint uses
the sidecar descriptors captured at submit time to materialize local
`ContinuousTensor` values on the session runner.

The current `RemoteL3Endpoint` implementation builds this payload from
`TaskSlotState`, zeros every tensor metadata `data` field, and submits the
encoded frame through `RemoteL3Transport`. The simulation session runner
resolves the digest in its dispatcher registry, materializes the sidecar
descriptors, and calls `inner_worker.run()`.

`RemoteTaskArgsWire v1` contains:

```text
tensor_count: uint32
scalar_count: uint32
tensor_metadata: ContinuousTensorWire[tensor_count]
remote_desc: OptionalRemoteTensorDescWire[tensor_count]
scalars: uint64[scalar_count]
inline_payload_bytes_len: uint32
inline_payload_bytes: uint8[inline_payload_bytes_len]
```

For each tensor index, exactly one of these is true:

- `remote_desc[i]` is present and names a remote buffer, imported peer buffer,
  UB mapping, or allowed small `HOST_INLINE` payload.
- The tensor is metadata-only and has no data pointer and no remote descriptor.

`tensor_metadata[i].data` must be zero in both cases. Bare host pointers are
rejected for remote endpoints unless an explicit staging API has produced a
remote handle and sidecar descriptor.

`HOST_INLINE` is for small payloads that should travel inside the TASK frame.
It still requires a `RemoteTensorDescWire` with `address_space=HOST_INLINE`;
the descriptor points into `inline_payload_bytes`. Regular and large tensor
data must use `RemoteBufferHandle` / `RemoteTensorDesc` and the remote
data-plane transport, not inline bytes.

## RemoteTensorDesc

```text
RemoteTensorDescWire:
  address_space:
    HOST_INLINE
    REMOTE_DEVICE
    REMOTE_WINDOW
    UB_LDST
  owner_endpoint_id
  buffer_id
  offset
  nbytes
  remote_addr
  rkey_or_token
  generation
  inline_payload_offset
  inline_payload_len
  flags
```

Rules:

- `ContinuousTensor` remains the L2 ABI. The session runner translates
  descriptors into local `ContinuousTensor` values immediately before
  `inner_worker.run()`.
- When a descriptor is present, the incoming `ContinuousTensorWire.data` is
  reserved and must be zero. The session runner derives the executable local
  address only from the validated descriptor and its live buffer/import
  registry.
- Metadata-only tensors also require `ContinuousTensorWire.data == 0`.
- Parent-side dependency keys use a stable logical start-address key derived at
  submit time:
  `(address_kind, owner_endpoint_id, buffer_id, generation, offset)`.
  `nbytes` bounds the descriptor and is kept for validation and future overlap
  detection, but it is not part of the first implementation's TensorMap lookup.
- `offset + nbytes` must fit inside the referenced handle.
- `generation` must match the parent registry entry and the remote daemon's
  live handle entry.
- For `HOST_INLINE`, `inline_payload_offset + inline_payload_len` must fit
  inside `inline_payload_bytes_len`, and `inline_payload_len` must equal
  `nbytes`. Remote handle fields (`owner_endpoint_id`, `buffer_id`,
  `remote_addr`, `rkey_or_token`, `generation`) are reserved and must be zero.
- For non-`HOST_INLINE` descriptors, `inline_payload_len` must be zero.
- A tensor owned by endpoint 3 cannot be submitted to endpoint 5 unless the
  parent has first created an imported peer handle through `IMPORT_BUFFER`.
  The descriptor sent to endpoint 5 still keeps `owner_endpoint_id=3` and the
  original `(buffer_id, generation, offset, nbytes)` identity; endpoint 5
  resolves it through its live importer-local registry entry.
- `child_memory=1` keeps its local meaning: the data is managed by a
  next-level worker. For remote workers, ownership is resolved through the
  remote sidecar, not through a local pointer alone.

## COMPLETION Payload

```text
sequence
error_code
error_message_len
error_message bytes
```

Rules:

- `sequence` must match the request being waited on.
- `error_code=0` means task success.
- Non-zero error means task failure. The parent marks the slot failed/poisoned
  and does not dispatch downstream consumers.
- `error_message` is bounded UTF-8. It should include remote host,
  `endpoint_id`, callable hashid, and `sequence`.
- If health expires or the process exits, the parent fabricates an endpoint
  failure completion for every in-flight sequence.

## CONTROL Payload

```text
control_name
control_version
command-specific bytes
```

Remote control frames use typed names and versioned payloads. Local mailbox
sub-command ids remain local-only and must not leak into the remote protocol.
Every `CONTROL` request produces exactly one `CONTROL_REPLY` frame with the
same `sequence`.

Required remote controls:

- `UNREGISTER_CALLABLE`
- `PREPARE_REGISTER_CALLABLE`
- `COMMIT_REGISTER_CALLABLE`
- `ABORT_REGISTER_CALLABLE`
- `PREPARE_CALLABLE`
- `ALLOC_REMOTE_BUFFER`
- `FREE_REMOTE_BUFFER`
- `COPY_TO_REMOTE`
- `COPY_FROM_REMOTE`
- `EXPORT_BUFFER`
- `IMPORT_BUFFER`
- `RELEASE_IMPORT`

Reserved future controls for Remote CommDomain:

- `COMM_INIT`
- `ALLOC_DOMAIN`
- `RELEASE_DOMAIN`

The first Remote L3 task-dispatch cut rejects the reserved domain controls
with an unsupported-control reply. They become required only when Remote
CommDomain enters scope.

The register-family controls are registry-scope-aware.
`PREPARE_REGISTER_CALLABLE` carries:

```text
target_registry:
  REMOTE_TASK_DISPATCHER
  INNER_L3_WORKER
callable_kind:
  PYTHON_IMPORT
  CHIP_CALLABLE
  PYTHON_SERIALIZED  # optional negotiated extension
callable_hash_digest: uint8[32]
payload_version: uint32
payload bytes
```

Valid target/kind combinations in protocol v1:

| Target registry | Callable kind | Status |
| --------------- | ------------- | ------ |
| `REMOTE_TASK_DISPATCHER` | `PYTHON_IMPORT` | Required baseline. |
| `REMOTE_TASK_DISPATCHER` | `PYTHON_SERIALIZED` | Optional negotiated extension. |
| `REMOTE_TASK_DISPATCHER` | `CHIP_CALLABLE` | Invalid. |
| `INNER_L3_WORKER` | `PYTHON_IMPORT` | Required for inner Python callables. |
| `INNER_L3_WORKER` | `PYTHON_SERIALIZED` | Optional negotiated extension. |
| `INNER_L3_WORKER` | `CHIP_CALLABLE` | Required for inner chip callables. |

Unsupported target/kind combinations are rejected during prepare and must not
create a prepared hashid entry.

Rules:

- `REMOTE_TASK_DISPATCHER` registers callables that can be selected by future
  parent TASK frames. Only Python callable kinds are valid in this parent-facing
  registry.
- `INNER_L3_WORKER` registers callables on the session runner's
  `inner_worker = Worker(level=3)`. It is a remote-internal install target, not
  a parent-facing `CallableHandle` resolver scope. Python callables become
  valid targets for inner `submit_sub` / recursive orchestration;
  `CHIP_CALLABLE` follows the existing prepare/register path for chip workers.
  A successful inner registration never makes the hashid valid for parent TASK
  frames.
- Remote orchestration code resolves committed inner handles through the
  session-runner-local `simpler.remote_l3_session.get_inner_handle(hashid)`
  helper. The returned handle is owned by `inner_worker` and is valid only
  inside that session process.
- `PYTHON_IMPORT` payloads carry a bounded UTF-8 `module:qualname` string.
  The payload parser trims surrounding ASCII whitespace, splits on exactly one
  colon, rejects empty parts, rejects relative module names, and rejects
  `<locals>`. Module and qualname components must be Python identifiers joined
  by dots. `PYTHON_IMPORT` uses callable kind value `3`, extending the PR #891
  values `1 = CHIP_CALLABLE` and `2 = PYTHON_SERIALIZED`.

  The canonical descriptor uses the PR #891 descriptor primitive encoding:
  `uint32` fields are unsigned little-endian, and strings are encoded as
  `uint32 byte_len` followed by UTF-8 bytes. It stores:

  ```text
  REMOTE_PYTHON_IMPORT:
    uint32 descriptor_schema_version = 1
    uint32 callable_kind = 3  # PYTHON_IMPORT
    string module
    string qualname
  ```

  PR866 adds `PYTHON_IMPORT` as a stable callable kind in the PR #891 identity
  model. The parent and endpoint rebuild the same descriptor, compute
  `sha256(descriptor)`, and compare it with `callable_hash_digest` before
  staging the import.
- `PYTHON_SERIALIZED` payloads follow
  [../python-callable-serialization.md](../python-callable-serialization.md).
  They are rejected in remote protocol v1 unless parent and session explicitly
  negotiate serializer version, payload limits, Python ABI/runtime
  compatibility, and dependency/runtime compatibility. Support is optional;
  `PYTHON_IMPORT` remains the required baseline. Rejection happens before any
  staged entry is published.
- A session that does not advertise a requested callable kind rejects the
  control request before installing the hashid.
- The endpoint recomputes the canonical descriptor hashid from the staged
  descriptor/payload and rejects the control if it does not match
  `callable_hash_digest`.
- `CHIP_CALLABLE` is valid only for `INNER_L3_WORKER`. The remote frame never
  embeds a local POSIX shm name from the parent process. The endpoint validates
  the payload against the PR #891 `CHIP_CALLABLE` canonical descriptor before
  installing the hashid.
- `COMMIT_REGISTER_CALLABLE` and `ABORT_REGISTER_CALLABLE` carry
  `target_registry`, `callable_kind`, and `callable_hash_digest`.
- `UNREGISTER_CALLABLE` carries the same `target_registry`, `callable_kind`,
  and `callable_hash_digest` so cleanup is scoped to the intended registry.

`CHIP_CALLABLE` remote payload version 1 uses either inline bytes for the
simulation/small-payload path or a session-local staged blob reference for
large hardware payloads:

```text
RemoteChipCallablePayloadWire v1:
  descriptor_bytes: bytes(max=4096)
  blob_location:
    INLINE_BLOB
    STAGED_BLOB
  blob_size: uint64
  blob_sha256: uint8[32]
  inline_blob: bytes(max=control_payload_max)
  staged_blob_token: bytes(max=1024)
  reserved: uint32 = 0
```

Rules:

- `descriptor_bytes` must decode as the PR #891 `CHIP_CALLABLE` canonical
  descriptor and hash to `callable_hash_digest`.
- The descriptor's `callable_blob_sha256` must equal `blob_sha256`.
- For `INLINE_BLOB`, `inline_blob` is present, `staged_blob_token` is empty,
  `inline_blob.size == blob_size`, and the full CONTROL payload fits within
  the endpoint's negotiated payload limit.
- For `STAGED_BLOB`, `staged_blob_token` names bytes already staged in this
  session by the selected data-plane adapter, `inline_blob` is empty, and the
  staged byte length must equal `blob_size`.
- The endpoint computes SHA-256 over the executable blob bytes before install
  and rejects the prepare if it does not match `blob_sha256`.
- The descriptor's target architecture, platform, runtime, and signature hash
  are validated by the inner worker path before the hashid is committed.
- Staged bytes are owned by the prepare transaction. They are released after
  commit succeeds or after abort/cleanup confirms the hashid was not published.

The current simulation implementation supports `INLINE_BLOB`. `STAGED_BLOB`
is rejected unless a staged-blob adapter has been negotiated for the session.

Bootstrap manifest entries use the same validation model as
`PREPARE_REGISTER_CALLABLE`; the manifest only changes transport of the
already-defined fields, not callable identity semantics:

```text
inner_l3_worker[]:
  hashid: lowercase hex sha256 digest, with or without "sha256:" prefix
  target_registry: "INNER_L3_WORKER"
  kind: "PYTHON_IMPORT" | "CHIP_CALLABLE" | negotiated future kind
  payload_version: uint32 = 1

  # PYTHON_IMPORT
  target: "module:qualname"

  # CHIP_CALLABLE
  payload_hex: hex(RemoteChipCallablePayloadWire v1)
```

The runner rebuilds the register command from each manifest entry, applies the
same descriptor/hash/platform/runtime checks as post-bootstrap controls, and
installs successful entries into `inner_worker` before `HELLO READY`.
Unsupported negotiated extensions, including `PYTHON_SERIALIZED` and
`STAGED_BLOB`, are rejected rather than partially installed.

Multi-endpoint parent registration uses two-phase visibility:

1. The parent sends `PREPARE_REGISTER_CALLABLE` to every selected endpoint.
   Each endpoint validates the descriptor/payload, checks feature gates, stages
   any callable bytes, and records the hashid as prepared but not visible to
   TASK.
2. If every prepare succeeds, the parent sends `COMMIT_REGISTER_CALLABLE` to
   every selected endpoint. A successful commit makes the hashid visible to
   later TASK frames on that endpoint.
3. If any prepare or commit fails, the parent sends `ABORT_REGISTER_CALLABLE`
   to endpoints with prepared or uncertain state, keeps the hashid invisible,
   and marks endpoints failed when their final registry state cannot be proven.

Partial failure outcomes:

- Prepare failure on an endpoint means that endpoint did not publish the
  hashid. The parent aborts endpoints that prepared successfully and does not
  publish the handle.
- Commit failure or timeout means the parent cannot assume all endpoints have
  the same visible state. It keeps the handle unpublished, sends cleanup to any
  endpoint that may have prepared or committed, and removes endpoints with
  uncertain cleanup from eligibility for that endpoint/hashid pair.
- Abort or cleanup failure leaves only the affected endpoint/hashid pair in a
  cleanup-uncertain state. Other endpoint/hashid pairs remain usable if their
  state is confirmed.
- Final unregister failure leaves the endpoint/hashid tombstone in place. The
  same endpoint/hashid pair cannot be re-registered or dispatched until cleanup
  is confirmed or the endpoint is failed for the session.

Final unregister creates a parent-side tombstone for the selected
endpoint/hashid pairs. The hashid remains dispatchable through any other live
handle that still owns a reference. Once the final reference is dropped for an
endpoint, that endpoint/hashid pair cannot be dispatched or published by a new
handle until cleanup is confirmed, or until any non-responsive endpoint has
been removed from eligibility and marked failed. If register rollback cleanup
cannot be confirmed, that endpoint/hashid pair enters a cleanup-uncertain state
and is unusable for the current session.

### Remote Buffer Export and Import Controls

Remote buffer export/import uses owner-stable identity plus importer-local
mapping ids. Public Python handles remain opaque; raw transport keys, HCOMM
descriptors, and UB addresses are protocol/session data, not user-visible
integers.

`EXPORT_BUFFER` is sent to the endpoint that owns the allocation:

```text
ExportBufferRequestWire v1:
  owner_endpoint_id: int32
  buffer_id: uint64
  generation: uint64
  offset: uint64
  nbytes: uint64
  access_flags: uint32  # bit 0 = read, bit 1 = write
  transport_profile: string(max=128)
  reserved: uint32 = 0
```

Rules:

- `owner_endpoint_id` must match the endpoint that receives the control.
- `buffer_id` and `generation` must name a live owner allocation.
- `offset + nbytes` must fit inside that allocation and `nbytes` must be
  non-zero.
- `access_flags` must request at least one supported access bit and no unknown
  bits.
- `transport_profile` must match a profile advertised by the owner endpoint
  for this session.
- The control may create or reuse a transport registration for the same
  `(buffer_id, generation, offset, nbytes, access_flags, transport_profile)`.
  There is no separate public release-export command in v1; the owner endpoint
  releases export registrations when the owner allocation is physically freed
  after all imports and slot refs have drained.

On success, `EXPORT_BUFFER` returns:

```text
ExportBufferResultWire v1:
  owner_endpoint_id: int32
  buffer_id: uint64
  generation: uint64
  address_space: uint32  # REMOTE_WINDOW or UB_LDST
  offset: uint64
  nbytes: uint64
  export_id: uint64
  remote_addr: uint64
  rkey_or_token: uint64
  ub_ldst_va: uint64
  access_flags: uint32
  transport_profile: string(max=128)
  transport_descriptor: bytes(max=4096)
  reserved: uint32 = 0
```

`export_id` is an opaque session-local capability issued by the owner endpoint.
For the simulation profile it may be the only meaningful token. For RoCE/HCCS
it identifies the HCOMM/RDMA memory registration described by
`rkey_or_token` and `transport_descriptor`. For UB it may also carry a
validated `ub_ldst_va`. `REMOTE_DEVICE` is not a valid export result address
space because peer endpoints consume the export through a window or UB mapping,
not by dereferencing the owner's local device pointer.

`IMPORT_BUFFER` is sent to the endpoint that will consume the peer buffer. Its
payload carries the owner-produced export descriptor plus the importer target:

```text
ImportBufferRequestWire v1:
  importer_endpoint_id: int32
  requested_access_flags: uint32
  export_desc: ExportBufferResultWire v1
  reserved: uint32 = 0
```

Rules:

- `importer_endpoint_id` must match the endpoint that receives the control.
- `requested_access_flags` must be a non-empty subset of
  `export_desc.access_flags`.
- The importer must advertise the same `transport_profile` and support
  `export_desc.address_space`.
- The importer validates `owner_endpoint_id`, `buffer_id`, `generation`,
  `offset`, `nbytes`, `export_id`, transport metadata, and access flags before
  publishing a mapping in its local import registry.
- A live import makes the buffer range data-eligible on the importer endpoint.
  Without a live import, non-owner endpoints remain ineligible for descriptors
  that name the owner's allocation.
- The dependency key for an imported handle remains the owner identity
  `(address_kind, owner_endpoint_id, buffer_id, generation, offset)`. The
  importer endpoint id and `import_id` are mapping details, not producer/consumer
  identity. This keeps tasks on different endpoints ordered when they touch the
  same physical remote buffer at the same logical start offset.

On success, `IMPORT_BUFFER` returns:

```text
ImportBufferResultWire v1:
  importer_endpoint_id: int32
  owner_endpoint_id: int32
  buffer_id: uint64
  generation: uint64
  import_id: uint64
  address_space: uint32  # REMOTE_WINDOW or UB_LDST
  offset: uint64
  nbytes: uint64
  remote_addr: uint64
  rkey_or_token: uint64
  ub_ldst_va: uint64
  access_flags: uint32
  transport_profile: string(max=128)
  import_descriptor: bytes(max=4096)
  reserved: uint32 = 0
```

`import_id` is opaque and unique within
`(session_id, importer_endpoint_id)`. The parent stores it inside the imported
`RemoteBufferHandle`; user code does not inspect it. TASK descriptors built
from the imported handle carry the original owner identity and the importer
mapping metadata from `ImportBufferResultWire`. The session runner rejects a
TASK descriptor if the corresponding importer-local mapping is no longer live.

`RELEASE_IMPORT` is sent to the importer endpoint:

```text
ReleaseImportRequestWire v1:
  importer_endpoint_id: int32
  owner_endpoint_id: int32
  buffer_id: uint64
  generation: uint64
  import_id: uint64
  reserved: uint32 = 0
```

Rules:

- `importer_endpoint_id` must match the endpoint that receives the control.
- The request releases only the importer-local mapping. It never frees the
  owner allocation and never invalidates imports on other endpoints.
- The parent sends `RELEASE_IMPORT` only after all slot refs captured from the
  imported handle have drained. If user code requests release earlier, the
  parent marks the imported handle released and defers the control.
- The endpoint may treat release of an already-released matching import as
  idempotent success. A mismatched owner, buffer, generation, or import id is a
  control failure.
- The reply has empty `result_bytes`.

Multi-endpoint import setup is owner-first and rollback-aware:

1. The parent sends `EXPORT_BUFFER` to the owner endpoint.
2. The parent sends `IMPORT_BUFFER` to each selected importer endpoint.
3. If any import fails, the parent sends `RELEASE_IMPORT` to importers that
   succeeded. If cleanup cannot be confirmed, those importer mappings enter an
   import-cleanup-uncertain state and the affected endpoint/buffer pair is
   unusable for new TASK dispatch in the current session.
4. `FREE_REMOTE_BUFFER` on the owner marks the owner handle released but
   defers physical free while exports, imports, or slot refs are live.
5. All three controls are state-changing command-lane controls and serialize
   with TASK on their target endpoint.

## CONTROL_REPLY Payload

```text
sequence
control_name
control_version
error_code
error_message_len
error_message bytes
result_bytes_len
result_bytes
```

Rules:

- `sequence` must match one in-flight `CONTROL` request on the same endpoint.
- `control_name` and `control_version` must match the request being answered.
- `error_code=0` means the control succeeded and `result_bytes` contains the
  command-specific result, if any.
- Non-zero `error_code` means the remote session did not apply the requested
  state change, except for commands whose versioned contract explicitly allows
  best-effort partial cleanup.
- `error_message` is bounded UTF-8. It should include remote host,
  `endpoint_id`, `control_name`, and `sequence`.
- `result_bytes` uses the same canonical encoding rules as other remote
  payloads. For example, `ALLOC_REMOTE_BUFFER` returns a buffer id,
  generation, address space, size, and transport-specific export metadata.
- If health expires or the process exits, the parent fabricates a failed
  `CONTROL_REPLY` for every in-flight control sequence.

Control waits obey the same timeout policy as task waits.

## Ordering

Each endpoint has one ordered command lane. Runtime state-changing requests use
this lane even when their bulk bytes are staged through a separate data plane.
The remote session observes parent-to-runner requests in increasing `sequence`
order, and state changes caused by a request happen-before later requests on
the same endpoint. Reply frames carry the answered request sequence and become
visible only after the corresponding request side effects.

The baseline endpoint admits at most one TASK in flight. State-changing CONTROL
frames serialize with that TASK and are not applied concurrently with
`inner_worker.run()`. This keeps callable registry visibility, buffer release,
import release, and domain lifetime consistent with the local one-mailbox
model.

`HEALTH` frames are not state-changing command-lane requests. The parent uses a
separate health lane, RDMA completion health signal, or transport-native
keepalive so a long-running TASK does not block liveness observation. Health
success does not imply task completion; task completion is only a matching
`COMPLETION` frame.

Health expiry is an endpoint failure for the current session:

- If a TASK or CONTROL is in flight, the parent fabricates the matching failed
  `COMPLETION` or `CONTROL_REPLY` with bounded endpoint-failure text.
- If no command is in flight, the endpoint is removed from the schedulable set
  before new work can be assigned to it.
- Later health frames from the same endpoint do not automatically make it
  schedulable again. Recovery requires creating a new remote session or an
  explicit future reconnect protocol.
- Pending slots whose final eligible set becomes empty after health expiry are
  failed rather than dispatched to an ineligible endpoint.

All transports must provide the following visibility rules:

- Task payload writes are visible before the task doorbell.
- Remote output writes are complete before the completion frame is visible.
- Parent completion reads happen before dependent task dispatch.
- Control request payload writes are visible before the control doorbell.
- Control side effects are visible before the matching `CONTROL_REPLY`.
- Register-family controls and `UNREGISTER_CALLABLE` serialize with TASK
  dispatch on the same endpoint. A successful commit reply happens-before later
  TASK frames that use the hashid. A successful final-unregister reply
  happens-before later same-hashid re-registration or dispatch on the affected
  endpoint.
- Health messages may be observed while a TASK is running, but they must not
  expose or mutate task, callable, buffer, or domain state.

RoCE and HCCS can satisfy this with SEND/RECV ordering plus explicit flush or
completion requirements. UB LD/ST paths require explicit memory fences around
doorbell and completion state transitions.

## Bounds and Fuzz Tests

The frame codec must reject:

- bad magic or unsupported version;
- payload length larger than configured maximum;
- truncated frame headers or payloads;
- tensor/scalar counts that overflow decoded payload size;
- descriptor offsets outside the referenced handle;
- `HOST_INLINE` payload offsets or lengths outside the inline byte arena;
- non-`HOST_INLINE` descriptors with non-zero inline payload lengths;
- non-zero `ContinuousTensorWire.data` in remote TASK frames;
- stale generations;
- unknown control names or control versions;
- completion sequence mismatch;
- control reply sequence or control-name mismatch.
