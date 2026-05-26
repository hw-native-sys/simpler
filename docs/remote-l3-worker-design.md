# Remote L3 Worker Design

This document describes how to extend the L3+ hierarchical runtime so a
parent `Worker` can schedule a remote `Worker(level=3)` as a NEXT_LEVEL
worker. The first target is an L4 parent dispatching to remote L3 workers.
The same contracts can later serve L5/L6.

Detailed protocol, buffer, transport, and rollout notes live in:

- [protocol.md](remote-l3-worker-design/protocol.md)
- [buffers-and-transports.md](remote-l3-worker-design/buffers-and-transports.md)
- [implementation-plan.md](remote-l3-worker-design/implementation-plan.md)

The current implementation uses pre-forked local child processes and a
4096-byte shared-memory mailbox. That model depends on copy-on-write callable
registries, identical virtual addresses for `MAP_SHARED` regions, and
parent-visible child PIDs. None of those assumptions holds across hosts.

## Scope

Goals:

- Preserve the Orchestrator/Scheduler DAG model.
- Replace the local mailbox endpoint under `WorkerThread` with a pluggable
  NEXT_LEVEL endpoint.
- Support remote data-plane backends for A2 RoCE, A3 HCCS, and A5 UB.
- Carry task dispatch, control commands, completion, error messages, and
  buffer lifetime over the remote endpoint.

Non-goals:

- Rewriting kernel allreduce or PTO-ISA collective kernels.
- Shipping Python closures without an explicit serialization contract.
- Designing general cross-host Python dependency or code distribution for
  arbitrary closures.
- Replacing the local fork/shm path for chip and sub workers.
- Changing the L2 `ChipWorker::run` ABI.

Remote callable registration follows the same public cid lifecycle defined for
local dynamic Python registration by PR #839: registration becomes visible to
the selected Python-capable endpoint only after the registration control reply
succeeds, unregister permits cid reuse only after stale state is cleared or the
endpoint is marked failed, and stale cid residue must not be observable by
later TASK frames. The remote design reuses those lifecycle semantics, but it
does not reuse PR #839's local mailbox commands, POSIX shm names,
process-local pointers, or exact serialized payload wire shape.

The required baseline remote callable descriptor is an import path such as
`pkg.module:orch_fn`. A serialized Python callable payload produced by the
PR #839 serializer is a negotiated remote capability, not an unconditional
baseline. When enabled, it travels as a versioned remote CONTROL payload and
must negotiate serializer version, payload limits, Python ABI/runtime
compatibility, and dependency/runtime-environment compatibility.

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
  - `submit_next_level()` stores `TaskArgs`, `CallConfig`, callable id, and
    optional worker affinity in a parent-side slot.
  - Dependency inference happens before dispatch from tags in `TaskArgs`.
- `src/common/task_interface/task_args.h`
  - Process dispatch writes `[T][S][ContinuousTensor x T][uint64 x S]`.
  - Tags are stripped after submit.
- `docs/comm-domain.md`
  - Dynamic communication domains already model deferred release after
    `drain()`.

The Scheduler should not inspect transport details, but it does need enough
metadata to avoid dispatching a task to an endpoint that cannot run it. Remote
tensor identity must be resolved before a slot becomes ready, because
TensorMap dependency inference and buffer-reference capture happen at submit
time.

## Target Architecture

Introduce a transport-neutral endpoint under `WorkerThread` with `run`,
`control`, and `shutdown` operations. `LocalMailboxEndpoint` wraps the current
shm mailbox code without changing wire behavior. `RemoteL3Endpoint` implements
the same interface over the framed remote protocol.

On dispatch, `WorkerThread` builds a task packet from `TaskSlotState`, calls
the endpoint, reports endpoint errors, and notifies the Scheduler with an
explicit success/failure outcome.

Ready queues, group dispatch, affinities, fanin/fanout, and ring release remain
in the existing runtime. The first-error-wins policy remains only as the error
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
5. The runner then performs an explicit prestart step equivalent to
   `inner_worker.init()` plus `_start_hierarchical()` for the inner Worker:
   allocate local mailboxes, fork local chip/sub children, register local
   endpoints with the inner C++ Worker, and start the inner Scheduler and
   `WorkerThread`s.
6. Only after this local L3 child tree is established does the session runner
   bring up sockets, RDMA queue pairs, health threads, or UB doorbells for task
   traffic.
7. The runner then performs the remote protocol `HELLO`/ready handshake over
   the ordered command lane. `HELLO` confirms session identity, endpoint
   identity, protocol version, transport kind, and negotiated features; it does
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

## Endpoint Identity and Callable Routing

Remote scheduling needs explicit callable namespaces and an explicit mapping
from callable ids to eligible NEXT_LEVEL endpoints. The current scheduler can
otherwise choose any idle worker, which is only correct when every NEXT_LEVEL
child has the same callable registry.

Required contracts:

- Every local or remote NEXT_LEVEL child has a stable `endpoint_id` equal to
  its logical worker id in `WorkerManager`.
- `register()` continues to register local callables for local fork/shm
  endpoints.
- `register_remote(remote_callable, workers=...)` allocates an outer cid in the
  parent `Worker(level=4)` id space, but binds that cid to one or more remote
  endpoint ids.
- Bootstrap manifests are generated by the parent. Users provide remote
  callable descriptors; users do not hand-author raw `cid -> callable` maps.
- Remote callable descriptors have two Python forms:
  - `PYTHON_IMPORT`: a bounded `module:qualname` import path. This is required
    for the remote L3 baseline.
  - `PYTHON_SERIALIZED`: a versioned payload produced by the PR #839 callable
    serializer. This is allowed only when parent and session negotiate the
    serializer version, payload limits, Python ABI/runtime compatibility, and
    dependency/runtime-environment compatibility.
- Remote L3 uses two independent cid namespaces:
  - **Outer remote cid namespace**: parent-assigned cids carried in L4 TASK
    frames. These cids select the remote L3 orchestration callable.
  - **Inner L3 cid namespace**: cids registered on the session runner's
    `inner_worker = Worker(level=3)`. Remote L3 orch functions use these cids
    when they call `orch.submit_next_level(...)` or `orch.submit_sub(...)`.
- The two namespaces may contain the same integer values, but they are not the
  same registry. A cid from the parent TASK frame must not be assumed to name a
  chip/sub callable inside the inner L3 Worker.
- Dynamic Python callable registration follows the public visibility and cid
  lifecycle semantics from local dynamic registration: registration is
  synchronous per selected endpoint, future TASK frames may use the cid only
  after the control reply succeeds, unregister/cid reuse clears stale callable
  state, and TASK/control ordering prevents a TASK from observing a partially
  registered cid.
- Import-path descriptors are the required remote baseline. Serialized Python
  callable payloads preserve the same cid lifecycle but remain an optional
  negotiated feature because they require Ray-like environment and serializer
  compatibility checks that local fork/COW registration does not need.
- Multi-endpoint `register_remote(..., workers=[...])` is all-or-nothing by
  default. The parent sends a prepare phase to every selected endpoint, commits
  the cid only after every prepare succeeds, and exposes the cid to future TASK
  frames only after every commit reply succeeds. If any endpoint fails prepare
  or commit, the parent aborts the transaction, keeps the cid invisible, and
  either rolls back successful endpoints or marks endpoints with unknown state
  failed.
- Multi-endpoint unregister uses a tombstone state. A cid is unavailable for
  reuse until every selected endpoint confirms unregister cleanup, or until any
  non-responsive endpoint is removed from eligibility and marked failed. This
  prevents stale callable residue from being observed by later TASK frames.
- `TaskSlotState` stores the final eligible endpoint set for the slot. This is
  the intersection of endpoints that can run the `callable_id` and endpoints
  that can access every tensor/buffer referenced by the slot.
- If the user passes `worker=N`, submit-time validation checks that endpoint
  `N` is eligible for that cid and for the slot's tensor sidecars.
- If `worker=-1`, the Scheduler chooses only from idle endpoints in the
  slot's eligible set.
- Group submit validates each affinity independently. Unconstrained group
  members are assigned distinct idle eligible endpoints.
- Mixed local + remote NEXT_LEVEL pools are allowed only when the callable id
  is registered on every endpoint that can receive the slot and the slot's
  tensors are materialized in a representation those endpoints can consume.
  A callable registered on both local and remote endpoints does not make a
  remote-buffer task eligible for the local endpoint.

Example API shape:

```python
from simpler.worker import RemoteCallable, RemoteWorkerSpec, Worker

w4 = Worker(level=4)

l3 = RemoteWorkerSpec(
    endpoint="node17:19073",
    platform="a2a3",
    runtime="tensormap_and_ringbuffer",
    device_ids=list(range(16)),
    num_sub_workers=2,
    transport="roce",
)

l3_worker_id = w4.add_remote_worker(l3)
l3_cid = w4.register_remote(
    RemoteCallable("my_pkg.remote_orch:l3_orch"),
    workers=[l3_worker_id],
)
w4.init()
```

`add_worker(local_worker)` remains unchanged and continues to use fork/shm.

Dynamic remote registration uses the same cid lifecycle whether the descriptor
is installed at bootstrap or through a later control frame. If a callable refers
to inner L3 cids, those cids are values from the inner namespace installed by
the session manifest or by prior remote control registration, not the outer cid
used to dispatch the remote L3 orch callable.

## Remote Worker Session

The parent generates a bootstrap manifest and sends it to the
`simpler-remote-worker` daemon as part of session creation. The daemon validates
it and hands it to the session runner before the runner starts any transport
threads:

```text
session_id
parent_worker_level
remote_worker_level = 3
endpoint_id
platform, runtime, build flag
device_ids, num_sub_workers, heap_ring_size
callable registry:
  outer registry:
    parent-assigned outer cid -> remote L3 orch callable descriptor
      descriptor = PYTHON_IMPORT or negotiated PYTHON_SERIALIZED
  inner L3 registry:
    inner cid -> ChipCallable blob metadata, when needed
    inner cid -> Python sub/orch callable descriptor, when needed
transport kind: roce | hccs | ub | sim
feature flags
```

The session runner installs the outer registry into its remote TASK dispatcher.
It installs the inner registry into `inner_worker = Worker(level=3)` during
prestart and before `HELLO READY`. Remote controls may add or remove entries in
either namespace after bootstrap:

- registering an outer Python callable changes what future L4 TASK frames can
  dispatch on this remote endpoint;
- registering an inner Python callable changes what already-registered remote
  L3 orch functions can submit to `inner_worker`;
- registering an inner `ChipCallable` follows the existing dynamic callable IPC
  cascade shape, but the remote control payload is a versioned remote frame
  instead of a local POSIX-shm mailbox name.

For a TASK frame, the session runner:

1. Validates the session and sequence number.
2. Decodes `RemoteTaskArgs`.
3. Translates remote tensor descriptors into local `ContinuousTensor` values.
4. Looks up the L3 orchestration function in the outer registry by
   parent-assigned outer cid.
5. Calls `inner_worker.run(orch_fn, args, config)`.
6. Sends completion with success or bounded traceback text.

For a CONTROL frame, it forwards the operation to the inner worker or its
buffer registry, then replies with a typed result.

Session execution rules:

- The baseline remote endpoint runs at most one TASK at a time. This matches
  the current one-`WorkerThread`-per-child local scheduling model and keeps
  ordering, buffer lifetime, and cid visibility simple.
- State-changing CONTROL frames such as register, unregister, buffer free,
  import release, comm init, and domain allocation serialize with TASK
  execution on the ordered command lane. They are not applied concurrently with
  a running TASK on the same endpoint.
- Bulk data movement may use a separate data plane, but the state change that
  makes staged bytes, callable payloads, or imported handles visible is ordered
  by the command lane.
- Health/liveness does not depend on the command lane making progress. Each
  session has an independent health lane or equivalent transport-level health
  signal so a long-running `inner_worker.run()` does not look like endpoint
  failure merely because queued command-lane frames cannot be serviced.

## Remote TaskArgs Representation

Keep `ContinuousTensor` as the L2 ABI. Do not overload raw pointer values to
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
  a local-addressable `ContinuousTensor`.
- Remote endpoints require a sidecar/descriptor for every tensor that carries
  data over the remote protocol, including `HOST_INLINE` tensors. A null
  sidecar is allowed only for metadata-only tensors with no data payload.
  Remote endpoints reject bare host pointers unless an explicit staging API
  produced a remote handle.
- Remote submits reject `OUTPUT` tensors whose `ContinuousTensor.data == 0`
  unless the caller has already supplied a `RemoteTensorRef` sidecar. The first
  implementation does not auto-allocate remote outputs during submit.

Parent-side slots therefore store existing `TaskArgs`, an optional
`RemoteTaskArgsView`, eligible endpoint ids, and captured remote-buffer refs.

`Orchestrator::infer_deps()` builds `TensorKey` from remote handle metadata
when a sidecar exists. The first implementation intentionally preserves the
current exact-start TensorMap semantics: local fork/shm keys use
`(ptr, worker)`, while remote keys use
`(address_kind, owner_endpoint_id, buffer_id, generation, offset)`. The tensor
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
  error with remote host, endpoint id, cid, and sequence in the message.

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

See
[buffers-and-transports.md](remote-l3-worker-design/buffers-and-transports.md)
for the handle schema, control commands, release policy, and A2/A3/A5 backend
requirements.

## Protocol

Do not reuse the raw 4096-byte mailbox format across hosts. It has no version
field, no sequence number, and assumes shared virtual memory.

Remote endpoints use a versioned frame protocol with `HELLO`, `TASK`,
`CONTROL`, `CONTROL_REPLY`, `COMPLETION`, `HEALTH`, and `SHUTDOWN` frames. The
local path keeps the existing mailbox layout behind `LocalMailboxEndpoint`.
Remote frames use canonical little-endian field encoding for `CallConfig`,
`ContinuousTensor`, tensor descriptors, strings, counts, and enums; they do not
memcpy local C++ POD structs onto the wire. Each endpoint has one ordered
command lane for runtime state-changing frames, so TASK cannot overtake
registry-changing CONTROL. Liveness uses a separate health lane or equivalent
transport-level signal and is not queued behind user TASK execution.

See [protocol.md](remote-l3-worker-design/protocol.md) for frame layout,
remote tensor descriptors, ordering, and bounds-checking requirements.

## Rollout

The recommended first cut is conservative:

1. Land the endpoint abstraction and local adapter.
2. Add remote tensor sidecars and endpoint eligibility metadata.
3. Add the versioned frame codec and the independent health-lane contract.
4. Add remote callable registration with all-or-nothing multi-endpoint
   visibility and tombstone-based cid reuse.
5. Add the fork-safe simulation session runner with explicit prestart before
   `HELLO READY`.
6. Prove local behavior is unchanged and remote sim behavior handles success,
   failure, cid mapping, timeouts, health, and buffer cleanup.
7. Add hardware transports behind the same protocol.

See
[implementation-plan.md](remote-l3-worker-design/implementation-plan.md)
for the detailed PR sequence and validation matrix.
