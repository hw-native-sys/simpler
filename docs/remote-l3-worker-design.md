# Remote L3 Worker Design

This document describes how to extend the L3+ hierarchical runtime so a
parent `Worker` can schedule a remote `Worker(level=3)` as a NEXT_LEVEL
worker. The first target is an L4 parent dispatching to remote L3 workers.
The same contracts can later serve L5/L6.

Detailed protocol, buffer, transport, and rollout notes live in:

- [protocol.md](remote-l3-worker-design/protocol.md)
- [buffers-and-transports.md](remote-l3-worker-design/buffers-and-transports.md)
- [implementation-plan.md](remote-l3-worker-design/implementation-plan.md)
- [pr-split-and-audit-plan.md][split-audit-plan]

[split-audit-plan]: remote-l3-worker-design/pr-split-and-audit-plan.md

Related callable registration and serialization contracts:

- [callable-ipc-dynamic-register.md](callable-ipc-dynamic-register.md)
- [callable-identity-registration.md](callable-identity-registration.md)
- [python-callable-serialization.md](python-callable-serialization.md)

`callable-identity-registration.md` is a prerequisite refinement for this
design: remote L3 callable routing should use `hashid` identities and
target-private execution slots instead of cross-worker integer routing.

The current implementation uses pre-forked local child processes and a
fixed-size (`MAILBOX_SIZE`-byte) shared-memory mailbox. That model depends on copy-on-write callable
registries, identical virtual addresses for `MAP_SHARED` regions, and
parent-visible child PIDs. None of those assumptions holds across hosts.

## Current Implementation Status

The local PR #866 cut has landed the transport-neutral runtime boundary, but
not the production daemon/HCOMM backends.

Implemented:

- `WorkerEndpoint` with `LocalMailboxEndpoint` and `RemoteL3Endpoint`.
  `LocalMailboxEndpoint` keeps the existing fixed-size mailbox local-only.
  `RemoteL3Endpoint` encodes versioned TASK frames through a
  `RemoteL3Transport` interface and waits for matching COMPLETION frames.
- Explicit endpoint outcomes: success, task failure, endpoint failure, and
  skipped group members. Failed producers poison downstream consumers instead
  of completing successfully.
- Stable NEXT_LEVEL `worker_id` metadata shared by local and remote children,
  submit-time validation of the exact target against worker eligibility, and
  directed Scheduler dispatch to that target only.
- C++ remote tensor sidecars, remote-aware `TensorKey` values, and submit-time
  rejection of remote sidecars against local endpoints, bare host pointers, and
  remote null OUTPUT tensors without a sidecar.
- Python `RemoteCallable("module:qualname")`, `PYTHON_IMPORT`, and
  `REMOTE_TASK_DISPATCHER` callable identity. Registration requires an
  explicit `workers=[...]` list naming worker ids returned by
  `add_remote_worker()`.
- Python `RemoteBufferHandle` and `RemoteTensorRef` wrappers. `TaskArgs`
  accepts `RemoteTensorRef` through `add_tensor(...)`, keeps
  `Tensor.data == 0`, and carries the remote descriptor sidecar into C++
  submit.
- Canonical little-endian `remote_wire.{h,cpp}` frame codec with bounds checks
  for TASK payloads, remote tensor descriptors, COMPLETION, CONTROL_REPLY, and
  ordered command-lane sequencing.
- Socket-backed simulation remote sessions via `simpler-remote-worker` and
  `simpler-remote-l3-session`, including `HELLO READY`, TASK/COMPLETION,
  CONTROL/CONTROL_REPLY, SHUTDOWN, and an independent health lane.
- Simulation remote buffer allocation, copy, export, import, release-import,
  imported-handle scheduling eligibility, and deferred owner free.
- Registry-scope-aware remote callable manifest/control install for dispatcher
  `PYTHON_IMPORT`, inner `PYTHON_IMPORT`, and inner inline `CHIP_CALLABLE`.
  Pre-init `ChipCallable` registrations on an L4 worker are serialized into
  each remote session manifest and installed on that L3's L2 children.
- A no-`mpirun` A3 TCP smoke that keeps a Global CommDomain across two L4
  runs, executes peer `TLOAD` from each remote L2, and verifies the reduced
  values before release.
- Two-server hardware validation covers L4-brokered peer `TLOAD`, one L2
  compute followed by cross-machine communication, and two-NPU-per-node
  remote L3 group compute.

Still pending:

- A2 RoCE, A3 HCCS, and A5 UB HCOMM profiles.
- Negotiated `PYTHON_SERIALIZED` remote callable payloads and staged
  `CHIP_CALLABLE` blob adapters.

## Scope

Goals:

- Preserve the Orchestrator/Scheduler DAG model.
- Replace the local mailbox endpoint under `WorkerThread` with a pluggable
  NEXT_LEVEL worker endpoint.
- Support HCOMM-backed remote communication profiles for A2 RoCE, A3 HCCS,
  and A5 UB.
- Carry task dispatch, control commands, completion, error messages, and
  buffer lifetime over the remote endpoint.

Non-goals:

- Rewriting kernel allreduce or PTO-ISA collective kernels.
- Shipping Python closures without an explicit serialization contract.
- Designing general cross-host Python dependency or code distribution for
  arbitrary closures.
- Replacing the local fork/shm path for chip and sub workers.
- Changing the L2 `ChipWorker::run` ABI.
- Exposing `RemoteL3Endpoint`, HCOMM, RDMA, or socket details to
  `Orchestrator`; endpoint selection and materialization stay behind
  `WorkerEndpoint`.
- Redesigning remote `CommDomain` or the device `CommContext` ABI in the first
  Remote L3 task-dispatch cut.

Remote callable registration follows the public callable identity lifecycle
defined by PR #891: `Worker.register()` returns a `CallableHandle`, `hashid` is
the stable cross-process identity, and each target owns a private
`hashid -> local_slot` mapping. A registration becomes visible to the selected
Python-capable endpoint only after the registration control reply succeeds.
Final unregister prevents reuse of endpoint-private execution state until stale
state is cleared or the endpoint is marked failed, and stale callable residue
must not be observable by later TASK frames. The remote design reuses those
lifecycle semantics, but it does not reuse PR #891's local mailbox commands,
POSIX shm names, process-local pointers, or exact serialized payload wire
shape.

The required baseline remote callable descriptor is an import path such as
`pkg.module:orch_fn`. Serialized Python callable payloads follow
[python-callable-serialization.md](python-callable-serialization.md), but they
are a negotiated remote capability, not part of the first required remote L3
baseline. When enabled, the payload travels as a versioned remote CONTROL
payload and must negotiate serializer version, payload limits, Python
ABI/runtime compatibility, and dependency/runtime-environment compatibility.

## Current Seams

Relevant code paths:

- `python/simpler/worker.py`
  - `_start_hierarchical()` forks local child workers.
  - `_child_worker_loop()` runs a nested `Worker` child via shm mailbox.
  - `_run_chip_main_loop()` handles task and control mailbox states.
- `src/common/hierarchical/worker_manager.{h,cpp}`
  - `WorkerThread` owns one local mailbox and blocks until `TASK_DONE`.
  - Control commands share the same mailbox and serialize on `mailbox_mu_`.
  - Errors are reported through `MAILBOX_OFF_ERROR` and
    `MAILBOX_OFF_ERROR_MSG`.
- `src/common/hierarchical/orchestrator.{h,cpp}`
  - `submit_next_level()` stores `TaskArgs`, `CallConfig`,
  `CallableIdentity`, and the required target worker in a parent-side slot.
  - Dependency inference happens before dispatch from tags in `TaskArgs`.
- `src/common/task_interface/task_args.h`
  - Process dispatch writes `[T][S][Tensor x T][uint64 x S]`.
  - Tags are stripped after submit.
- `docs/comm-domain.md`
  - Dynamic communication domains already model deferred release after
    `drain()`.

The Scheduler should not inspect transport details, but it does need enough
metadata to avoid dispatching a task to an endpoint that cannot run it. Remote
tensor identity must be resolved before a slot becomes ready, because
TensorMap dependency inference and buffer-reference capture happen at submit
time.

`Orchestrator` consumes tags, computes dependency keys, and stores endpoint
eligibility metadata. It must not call `RemoteL3Endpoint`, HCOMM, RDMA, or
socket APIs directly. `WorkerThread` and `WorkerEndpoint` own the child
boundary and materialization mechanics.

## Target Architecture

Introduce a communication-neutral `WorkerEndpoint` under `WorkerThread` with
`caps`, `run`, `control`, and `shutdown` operations. `LocalMailboxEndpoint`
wraps the current shm mailbox code without changing wire behavior.
`RemoteL3Endpoint` implements the same interface with a bootstrap socket for
session setup, then HCOMM-backed RPC and data adapters for steady-state
traffic.

`caps()` is part of the endpoint contract because submit-time eligibility must
know whether an endpoint can resolve a callable `hashid` and consume the tensors
in a slot. It is read-only capability metadata, not a transport escape hatch:
the Scheduler sees logical features such as callable kinds, resolver scopes,
memory directions, address spaces, and health state, while `RemoteL3Endpoint`
remains the only layer that knows the selected HCOMM protocol or adapter
handles.

On dispatch, `WorkerThread` builds a task packet from `TaskSlotState`, calls
the endpoint, reports endpoint errors, and notifies the Scheduler with an
explicit success/failure outcome.

Directed ready queues, exact group dispatch, fanin/fanout, and ring release
remain in the common runtime. The first-error-wins policy remains only as the error
reporting policy for choosing which root error `drain()` raises. The important
change is that completion is no longer implicitly success; every endpoint,
including `LocalMailboxEndpoint`, must report an explicit success/failure
outcome.

## Fork-Safe Remote Process Model

The remote runtime must preserve the repository's fork ordering invariant:
all chip/sub child processes are forked before any C++ Scheduler,
`WorkerThread`, transport, or health threads are started.

Use a two-process remote model:

1. `simpler-remote-worker` is a small control daemon. It accepts session
   requests and validates bootstrap manifests on the daemon control channel.
   It never constructs an inner `Worker` and never forks chip/sub children
   after starting transport worker threads.
2. For each accepted session, the daemon starts a fresh
   `simpler-remote-l3-session` runner process, preferably by `exec`.
3. The daemon passes the validated manifest to the runner through a simple
   pre-fork handoff such as an inherited fd, a manifest file path in env, or a
   single-threaded pipe. This handoff is not the remote transport protocol.
4. The session runner reads the manifest before starting transport threads and
   constructs `Worker(level=3)`.
5. The runner then performs an explicit prestart step, `inner_worker.init()`
   for the inner Worker — `init()` is the single startup point: it allocates
   local mailboxes, forks local chip/sub children, registers local endpoints
   with the inner C++ Worker, and starts the inner Scheduler and
   `WorkerThread`s.
6. Only after this local L3 child tree is established does the session runner
   bring up sockets, RDMA queue pairs, health threads, or UB doorbells for task
   traffic.
7. The runner then performs the remote protocol `HELLO`/ready handshake over
   the ordered command lane. `HELLO` confirms session identity, endpoint
   identity, protocol version, comm profile, and negotiated features; it does
   not carry the bootstrap manifest.
8. Session shutdown rejects new frames, completes or fails in-flight tasks,
   drains cleanup, closes the inner Worker, and exits the runner process.

This keeps the local L3 fork/shm implementation intact while preventing a
multi-threaded network daemon from becoming the process that performs the
forks.

`HELLO ready_state=READY` is a scheduling barrier, not just a liveness signal.
The parent must not put a remote endpoint into the schedulable set until the
runner has completed prestart, installed the bootstrap registries, initialized
the buffer/import registry, started the command and health lanes, and confirmed
the negotiated feature set. This mirrors the distributed-system convention that
a worker or actor becomes visible only after runtime and dependency
initialization has completed.

## Worker Identity and Callable Routing

Remote scheduling needs explicit callable resolver scopes and an explicit
mapping from callable identities to eligible NEXT_LEVEL workers. Every submit
also names the exact worker that will execute the task.

Required contracts:

- Every local or remote NEXT_LEVEL child has a stable `worker_id` assigned
  when the Python facade adds that child. Python callers pass the integer
  worker id returned by `add_worker()` or `add_remote_worker()`. The id is
  independent of the C++ worker-thread vector index, so adding later local or
  remote children cannot renumber ids already returned to user code.
- `register(callable, workers=...)` is the single public registration API.
  Local Python/callable objects keep the existing local registration path.
  `RemoteCallable` descriptors use the remote control path and bind the
  returned `CallableHandle.hashid` to one or more remote worker ids.
- PR866 extends the PR #891 callable identity surface with one parent-facing
  remote callable kind, `PYTHON_IMPORT`, and one parent-facing resolver scope,
  `REMOTE_TASK_DISPATCHER`. `RemoteCallable("pkg.module:orch_fn")`
  registration returns a normal `CallableHandle` whose `kind` is
  `PYTHON_IMPORT`, whose resolver scope is `REMOTE_TASK_DISPATCHER`, and whose
  `hashid` is computed from the canonical remote import descriptor.
- The first implementation requires `RemoteCallable` registration to pass an
  explicit non-empty `workers=[...]` list of stable remote worker ids. Future
  releases may replace these ids with named remote pools or placement policies,
  but implicit broadcast to all remote endpoints is not part of the contract.
- Bootstrap manifests are generated by the parent. Users provide remote
  callable descriptors; users do not hand-author raw `hashid -> callable`
  maps.
- Remote callable descriptors have two Python forms:
  - `PYTHON_IMPORT`: a bounded `module:qualname` import path. This is required
    for the remote L3 baseline.
  - `PYTHON_SERIALIZED`: a versioned payload that follows
    [python-callable-serialization.md](python-callable-serialization.md). This
    is rejected in remote protocol v1 unless parent and session explicitly
    negotiate serializer version, payload limits, Python ABI/runtime
    compatibility, and dependency/runtime-environment compatibility.
- The parent computes each remote callable `hashid` from canonical descriptor
  bytes before publication. `PYTHON_IMPORT` uses a remote descriptor schema
  that includes descriptor schema version, callable kind, normalized module
  string, and normalized qualname string; any future environment compatibility
  digest or import policy field must bump the schema version.
  `PYTHON_SERIALIZED` reuses the PR #891 serialized-payload descriptor
  identity. Inner `CHIP_CALLABLE` entries reuse the PR #891 chip descriptor
  identity.
- Remote L3 uses one callable identity scheme: `hashid`. It still has two
  resolver locations because different runtime objects consume the identity:
  - **Remote TASK dispatcher registry**: the parent-facing entry registry in
    the session runner. TASK frames from the L4 parent carry a hash digest; the
    dispatcher resolves it to the remote L3 orchestration callable and then
    invokes `inner_worker.run(...)`.
  - **Inner L3 Worker registry**: remote-internal state owned by
    `inner_worker = Worker(level=3)`. Remote L3 orchestration functions use
    this Worker's own `CallableHandle` values when they call
    `orch.submit_next_level(...)` or `orch.submit_sub(...)`.
- The remote TASK dispatcher is not a Worker and must not reimplement
  Scheduler, Orchestrator, TensorMap, or drain semantics. It is an RPC entry
  adapter that resolves the outer callable, materializes `RemoteTaskArgsWire`,
  calls the embedded `inner_worker`, and wraps completion/error reporting.
- Resolver location is selected by execution context, not by a second identity
  dimension. A remote TASK frame always resolves in the remote TASK dispatcher
  registry; an inner `orch.submit_*` call resolves through the inner Worker's
  normal PR #891 handle validation. The same hashid may appear in both
  registries, but that means the same canonical descriptor was installed in two
  places, not that the two registries share executable slots or eligibility.
- Dynamic Python callable registration follows the public visibility and
  hashid lifecycle semantics from local dynamic registration: registration is
  synchronous per selected worker, future TASK frames may use the hashid only
  after the control reply succeeds, unregister clears stale callable state, and
  TASK/control ordering prevents a TASK from observing a partially registered
  hashid.
- Import-path descriptors are the required remote baseline. Serialized Python
  callable payloads preserve the same hashid lifecycle but remain an optional
  negotiated feature because they require Ray-like environment and serializer
  compatibility checks that local fork/COW registration does not need.
- Multi-endpoint `register(..., workers=[...])` is all-or-nothing by
  default. The parent sends a prepare phase to every selected worker, commits
  hashid only after every prepare succeeds, and exposes the hashid to future
  TASK frames only after every commit reply succeeds. If any worker fails
  prepare or commit, the parent aborts the transaction, keeps the hashid
  invisible, and either rolls back successful workers or marks workers with
  unknown state failed.
- Final multi-endpoint unregister uses a tombstone state for the selected
  worker/hashid pairs. The hashid remains dispatchable through any other live
  handle that still owns a reference. Once the final reference is dropped for
  a worker, that worker/hashid pair is unavailable for dispatch until cleanup
  is confirmed or the worker is removed from eligibility and marked failed.
  Failed-register rollback whose cleanup cannot be confirmed marks that
  worker/hashid pair cleanup-uncertain and unusable for the current session.
- `TaskSlotState` stores the exact target worker id for each slot member.
  Submit-time validation checks that target against the callable's eligible
  worker set — the intersection of workers that can resolve the callable hashid
  and workers that can access every tensor/buffer referenced by the slot.
- `worker=worker_id` is required. Submit-time validation checks that the worker
  id is eligible for that hashid and for the slot's tensor sidecars.
- Group submit requires one distinct worker id per member and validates each
  target independently.
- Mixed local + remote NEXT_LEVEL pools are allowed only when the callable
  hashid is registered on every endpoint that can receive the slot and the
  slot's tensors are materialized in a representation those endpoints can
  consume.
  A callable registered on both local and remote endpoints does not make a
  remote-buffer task eligible for the local endpoint.

Example API shape:

```python
from simpler.worker import RemoteCallable, RemoteWorkerSpec, Worker

w4 = Worker(level=4)

# endpoint host must be a numeric IP (or "localhost"); hostnames are rejected
# because getaddrinfo resolution is unbounded and could pin startup.
l3 = RemoteWorkerSpec(
    endpoint="10.0.0.17:19073",
    platform="a2a3",
    runtime="tensormap_and_ringbuffer",
    device_ids=list(range(16)),
    num_sub_workers=2,
    transport="roce",
)

l3_worker_id = w4.add_remote_worker(l3)
l3_handle = w4.register(
    RemoteCallable("my_pkg.remote_orch:l3_orch"),
    workers=[l3_worker_id],
)
w4.init()
```

`add_worker(local_worker)` still uses fork/shm, but it now returns the local
child's stable NEXT_LEVEL worker id from the same id space as
`add_remote_worker(...)`. Mixed local/remote registration order therefore does
not change worker ids already returned to the caller. Callers should pass the
returned ids back to `worker=` or `workers=[...]`.

Dynamic remote registration uses the same hashid lifecycle whether the
descriptor is installed at bootstrap or through a later control frame. If a
remote L3 orchestration callable refers to inner L3 callables, those are
`CallableHandle` values owned by the embedded inner Worker. The parent may
include inner Worker registry entries in the bootstrap manifest or send
registry-scoped controls to install them, but it does not receive or submit a
parent-facing handle whose resolver scope is `INNER_L3_WORKER`.

Post-bootstrap inner registry controls use the same prepare/commit/abort and
final-unregister visibility rules as dispatcher registration. `CHIP_CALLABLE`
is valid only for `INNER_L3_WORKER`; the remote payload carries bounded inline
blob bytes or a session-local staged blob token, never a parent-local POSIX
shm name. A successful inner install makes the hashid visible only to
`inner_worker = Worker(level=3)` and its chip/sub dispatch paths.
Remote orchestration code that needs a post-bootstrap inner handle resolves
the installed hashid through the session-runner-local
`simpler.remote_l3_session.get_inner_handle(hashid)` helper. That helper
returns a `CallableHandle` owned by the embedded `inner_worker`; it does not
publish an `INNER_L3_WORKER` resolver scope to the parent.

## Remote Worker Session

The parent generates a bootstrap manifest and sends it to the
`simpler-remote-worker` daemon as part of session creation. The daemon validates
it and hands it to the session runner before the runner starts any transport
threads:

```text
session_id
parent_worker_level
remote_worker_level = 3
worker_id
platform, runtime, build flag
device_ids, num_sub_workers, heap_ring_size
callable registry:
  remote TASK dispatcher registry:
    hashid -> remote L3 orch callable descriptor
      descriptor = PYTHON_IMPORT or negotiated PYTHON_SERIALIZED
  inner L3 Worker registry:
    hashid -> ChipCallable register payload, when needed
    hashid -> Python import descriptor, when needed
comm policy: roce | hccs | ub | sim
feature flags
```

The session runner installs the remote TASK dispatcher registry for parent
TASK frames. It installs the inner registry into
`inner_worker = Worker(level=3)` during prestart and before `HELLO READY`.
`INNER_L3_WORKER` is a remote-internal install target in manifests and
controls, not a parent-facing `CallableHandle` resolver scope.
`inner_l3_worker` manifest entries use the same target/kind matrix and
validation rules as `PREPARE_REGISTER_CALLABLE`: `PYTHON_IMPORT` entries carry
`target = "module:qualname"` and inline `CHIP_CALLABLE` entries carry the
versioned remote chip callable payload as `payload_hex`. Remote controls may
add or remove entries in either registry location after bootstrap:

- registering a remote TASK dispatcher Python callable changes what future L4
  TASK frames can dispatch on this remote endpoint;
- registering an inner Python callable changes what already-registered remote
  L3 orch functions can submit to `inner_worker`;
- registering an inner `ChipCallable` follows the existing dynamic callable IPC
  cascade shape, but the remote control payload is a versioned remote frame
  instead of a local POSIX-shm mailbox name.

For a TASK frame, the session runner:

1. Validates the session and sequence number.
2. Decodes `RemoteTaskArgsWire`.
3. Translates remote tensor descriptors into local `Tensor` values.
4. Looks up the L3 orchestration function in the remote TASK dispatcher
   registry by hashid.
5. Calls `inner_worker.run(orch_fn, args, config)`.
6. Sends completion with success or bounded traceback text.

For a CONTROL frame, it forwards the operation to the inner worker or its
buffer registry, then replies with a typed result.

Session execution rules:

- The baseline remote endpoint runs at most one TASK at a time. This matches
  the current one-`WorkerThread`-per-child local scheduling model and keeps
  ordering, buffer lifetime, and callable visibility simple.
- State-changing CONTROL frames such as register, unregister, buffer free,
  copy, export/import, import release, and Global CommDomain transactions
  serialize with TASK execution on the ordered command lane. They are not
  applied concurrently with a running TASK on the same endpoint.
- Bulk data movement may use a separate data plane, but the state change that
  makes staged bytes, callable payloads, or imported handles visible is ordered
  by the command lane.
- Health/liveness does not depend on the command lane making progress. Each
  session has an independent health lane or equivalent transport-level health
  signal so a long-running `inner_worker.run()` does not look like endpoint
  failure merely because queued command-lane frames cannot be serviced.
- Health expiry removes the endpoint from scheduler eligibility for the
  current session. In-flight TASK/CONTROL waits receive fabricated failed
  completions or replies; idle failed endpoints are not automatically re-added
  by later health frames.

## Remote TaskArgs Representation

Keep `Tensor` as the L2 ABI. Do not overload raw pointer values to
carry transport state.

Public Python uses a sidecar representation:

- `RemoteBufferHandle` identifies an allocated or imported remote buffer.
- `RemoteTensorRef(handle, offset, shape, dtype)` is accepted by
  `TaskArgs.add_tensor()` wherever a remote submit is legal.
- The Python/C++ binding stores a normal tensor metadata entry plus a hidden
  sidecar entry at the same tensor index.
- Local endpoints reject remote tensor refs. `RemoteTensorRef` is transport
  metadata, not a local mailbox ABI. A local fork/shm endpoint becomes eligible
  only after the data has been explicitly imported, staged, or materialized into
  a local-addressable `Tensor`.
- Remote endpoints require a sidecar/descriptor for every tensor that carries
  data over the remote protocol, including `HOST_INLINE` tensors. A null
  sidecar is allowed only for metadata-only tensors with no data payload.
  Remote endpoints reject bare host pointers unless an explicit staging API
  produced a remote handle.
- Remote submits reject `OUTPUT` tensors whose `Tensor.data == 0`
  unless the caller has already supplied a `RemoteTensorRef` sidecar. The first
  implementation does not auto-allocate remote outputs during submit.

Parent-side slots therefore store existing `TaskArgs`, an optional
`RemoteTaskArgsView`, eligible worker ids, and captured remote-buffer refs.

`Orchestrator::infer_deps()` builds `TensorKey` from remote handle metadata
when a sidecar exists. The first implementation intentionally preserves the
current exact-start TensorMap semantics: local fork/shm keys use
`(ptr, worker_id)`, while remote keys use
`(address_kind, owner_worker_id, buffer_id, generation, offset)`. The tensor
byte length is bounds-checked by the descriptor but does not participate in
dependency lookup. This means two remote tensors that reference overlapping
byte ranges with different offsets are not automatically ordered, matching the
current local pointer-key behavior.

## Failure Semantics

Remote completion is explicit and sequence-based. Local mailbox completion must
be adapted to the same outcome contract. Failure must not make downstream tasks
run as if the producer succeeded.

Required parent-side behavior:

- `RemoteL3Endpoint::run()` blocks for the matching completion sequence.
- `LocalMailboxEndpoint::run()` maps a non-zero mailbox error to
  `task_failure` instead of reporting a successful completion.
- Non-zero task or endpoint errors become candidates for the worker's first
  reported error.
- The worker still notifies the Scheduler so `drain()` cannot hang.
- The notification carries an outcome: success, task failure, or endpoint
  failure.
- Failed slots transition to a failed/poisoned state rather than successful
  `COMPLETED`.
- Downstream consumers of a failed producer are marked failed/skipped and are
  not dispatched.
- `drain()` waits for bookkeeping and cleanup, then raises the first root
  error with remote host, worker id, hashid, and sequence in the message.

Local mailbox dispatch keeps first-error-wins only for final error reporting.
It must not mark a failed child dispatch as successful `COMPLETED`. The remote
buffer path and the local adapter both use the same poisoned dependency
propagation before dependent tasks are exposed to failed producer outputs.

Every blocking wait must have a configurable timeout. Remote transport must
not copy the current local control path's infinite spin-wait failure mode.

## Buffer Lifecycle

Remote buffers need an owner, generation, and deferred physical free. The
parent owns the visible handle state; the session runner owns remote physical
memory and imported mappings.

The v1 peer-buffer model separates owner allocations from imported mappings.
`EXPORT_BUFFER` runs on the owner worker and returns an opaque
session-scoped descriptor. `IMPORT_BUFFER` runs on the importer worker and
creates an importer-local mapping handle. `RELEASE_IMPORT` tears down only that
importer-local mapping; owner physical free waits until imports and captured
slot refs have drained. Imported handles keep the original owner
`(owner_worker_id, buffer_id, generation, offset)` as the dependency
identity, so owner and peer views of the same logical buffer range serialize
through the same TensorMap key.

See
[buffers-and-transports.md](remote-l3-worker-design/buffers-and-transports.md)
for the handle schema, control commands, release policy, and A2/A3/A5 backend
requirements.

## Protocol

Do not reuse the raw fixed-size mailbox format across hosts. It has no version
field, no sequence number, and assumes shared virtual memory.

Remote endpoints use a versioned frame protocol with `HELLO`, `TASK`,
`CONTROL`, `CONTROL_REPLY`, `COMPLETION`, `HEALTH`, and `SHUTDOWN` frames. The
bootstrap socket carries only session setup and HCOMM bring-up frames. After
HCOMM RPC is ready, steady-state control, task metadata, completions, and
shutdown use the HCOMM-backed RPC adapter; tensor data and remote buffer
operations use the HCOMM data adapter. The local path keeps the existing
mailbox layout behind `LocalMailboxEndpoint`.

Remote frames use canonical little-endian field encoding for `CallConfig`,
`Tensor`, tensor descriptors, strings, counts, and enums; they do not
memcpy local C++ POD structs onto the wire. Each endpoint has one ordered
command lane for runtime state-changing frames, so TASK cannot overtake
registry-changing CONTROL. Liveness uses a separate health lane or equivalent
transport-level signal and is not queued behind user TASK execution.

See [protocol.md](remote-l3-worker-design/protocol.md) for frame layout,
remote tensor descriptors, ordering, and bounds-checking requirements.

## Rollout

The recommended first cut is conservative:

1. Land the endpoint abstraction and local adapter. **Implemented.**
2. Add remote tensor sidecars and endpoint eligibility metadata.
   **Implemented for C++ submit, Python `TaskArgs.add_tensor(RemoteTensorRef(...))`,
   owner buffers, and imported simulation buffers.**
3. Add the versioned frame codec and the independent health-lane contract.
   **Implemented for the socket-backed simulation transport.**
4. Add remote callable registration with all-or-nothing multi-endpoint
   visibility and final-unregister cleanup. **Implemented for dispatcher
   `PYTHON_IMPORT`, inner `PYTHON_IMPORT`, and inner inline
   `CHIP_CALLABLE`.**
5. Add the fork-safe simulation session runner with explicit prestart before
   `HELLO READY`. **Implemented.**
6. Prove local behavior is unchanged and remote sim behavior handles success,
   failure, hashid mapping, timeouts, health, and buffer cleanup.
   **Focused Python remote sim and C++ no-hardware UT coverage is present.**
7. Add A2 RoCE, A3 HCCS, and A5 UB profiles behind the HCOMM adapter layer.
   **Pending.**

See
[implementation-plan.md](remote-l3-worker-design/implementation-plan.md)
for the detailed PR sequence and validation matrix.
