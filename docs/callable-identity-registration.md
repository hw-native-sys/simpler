# Callable Identity Registration

This document defines callable identity for local hierarchical workers.
`hashid` is the stable identity across the parent process and its local child
worker processes. Integer execution slots are target-worker internals and are
not exposed through public APIs, task slots, or mailbox task frames.

Related background:

- [python-callable-serialization.md](python-callable-serialization.md)
- [callable-ipc-dynamic-register.md](callable-ipc-dynamic-register.md)

`callable-ipc-dynamic-register.md` is a historical note for the earlier
cid-based IPC design; the hashid identity contract in this document is the
source of truth. This document does not depend on `remote-l3-worker-design.md`.
Remote workers are outside the scope of this document.

## Overview

Hierarchical callable registration needs to handle two cases that a public
integer callable id cannot model safely:

1. A higher-level task may depend on lower-level callables that are only known
   at run time. For example, a parent orchestration callable may depend on a
   child `ChipCallable`.
2. Worker-local execution slots may differ across child workers. A parent or
   orchestration function must not depend on those slot values being inherited
   or consistent.

The local worker design is register-before-dispatch:

```text
bootstrap workers
register callable identities
install target-local digest -> slot mappings
prewarm chip executable state when applicable
dispatch tasks by callable handle
target worker resolves handle hashid to its local slot
```

Runtime demand-fetch is not part of the local worker contract. A child never
pauses task execution to ask its parent for a missing callable binary;
registration must materialize the target-local identity mapping before
dispatch. For chip targets, the local implementation also prewarms executable
state before first use when startup or dynamic register control reaches the
child before dispatch.

The current chip child loop preserves the historical TASK_READY lazy-prepare
safety net: if a digest is already registered in the child identity table but
the explicit prewarm step was missed, the child prepares its private slot in
the dispatch path and logs a warning. This is not a public cid compatibility
path and it does not fetch missing callable bytes from the parent.

### Chip Executable Prewarm

Chip callable registration separates identity installation from executable
materialization. A successful registration installs the target-local
`hashid -> local_slot` mapping. For runtimes that need device-side executable
state before the first dispatch, `prepare_callable()` also prewarms that state
before returning success.

For `tensormap_and_ringbuffer`, prewarm runs a private AICPU entry,
`simpler_aicpu_prewarm_callable`, after the orchestration SO bytes and child
kernel addresses have already been staged. The AICPU prewarm may materialize
the orchestration SO into a temporary file, `dlopen` it, resolve the
entry/config/bind symbols, and populate `orch_so_table_[callable_id]`. It must
stop before any real task execution: it does not call the orchestration entry,
does not configure runtime arguments from user task inputs, does not create a
`PTO2Runtime`, does not enter runtime scopes, and does not submit scheduler or
AICore work. `dlopen` may still run ELF or C++ static constructors; the
contract is that prewarm does not explicitly invoke orchestration logic.

`host_build_graph` does not support AICPU callable prewarm. Its orchestration
SO is resolved on the host during prepare, so there is no AICPU
`orch_so_table_` entry to populate and no AICPU dlopen accounting to update.
The internal prewarm hook for that runtime is a no-op success.

Host-visible cache-hit state is committed only after the device-side SO load
has succeeded. Until that commit happens, the next real run must treat the
callable as load-required rather than advertising an AICPU cache hit. If
prewarm fails, `prepare_callable()` fails and rolls back the callable
registration it just created. If a first real run performs the same load as a
fallback, it follows the same success-before-commit rule. Reusing a
`callable_id` slot after unregister forces the next prewarm or real run to
reload and replace the AICPU table entry for that slot.

`aicpu_dlopen_count` is host telemetry, not proof that the AICPU completed
`dlopen`. Validation must also rely on an AICPU-side signal, such as a log from
the SO-load helper, or on a follow-up real run that reuses a non-empty
`orch_so_table_[callable_id]` with reload disabled.

Introduce `hashid` as the stable callable identity:

```text
hashid = sha256:<64 lowercase hex characters>
```

### Target Namespace

`target_namespace` names the target registry domain that can resolve a
`CallableHandle`. It is scoped to the `Worker` instance that returned the
handle. It is not globally unique, not user-supplied, and not part of
`hashid`.

The namespace answers "which resolver owns this handle?" It is not a raw
worker id, not an affinity selector, and not by itself a platform/runtime
identity. Platform and runtime compatibility for chip callables is encoded in
the `CHIP_CALLABLE` descriptor fields and revalidated by the target.

Local namespace values:

| Namespace | Callable kind | Resolver domain |
| --------- | ------------- | --------------- |
| `LOCAL_CHIP` | `CHIP_CALLABLE` | Chip execution registry. |
| `LOCAL_PYTHON` | `PYTHON_SERIALIZED` | Python dispatch registry. |

`LOCAL_CHIP` is used for an L2 `ChipWorker`, direct L3 chip children created
from `device_ids`, and descendant chip registries reached by the existing
recursive `ChipCallable` cascade. All targets in one `LOCAL_CHIP` install must
accept the descriptor's `target_arch`, `platform`, and `runtime`. If a future
Worker supports heterogeneous chip targets, incompatible groups must use
separate namespaces or fail registration.

`LOCAL_PYTHON` is used for Python-capable child loops owned by this Worker:
SUB children and next-level Worker-child dispatch loops. It updates the
parent-owned Python dispatch registry for those children. It does not populate
an inner Worker's own `_callable_registry`; callables owned by an inner Worker
are registered on that inner Worker separately.

`Worker.register` derives `target_namespace` from the target object and the
Worker topology:

- `ChipCallable` targets produce `LOCAL_CHIP`.
- Python callable targets produce `LOCAL_PYTHON`.

Submit APIs validate that the selected execution path can resolve the handle's
namespace. For example, `submit_next_level` on an L3 Worker with chip children
requires `LOCAL_CHIP`, while `submit_next_level` on an L4 Worker child path
requires a Python dispatch handle. The receiving child mailbox derives its
resolver namespace from the selected child path: chip child loops resolve
against `LOCAL_CHIP`, and SUB children plus next-level Worker-child Python loops
resolve against `LOCAL_PYTHON`. Local task frames carry only the raw `sha256`
digest because each current child path has exactly one resolver domain.

Each target namespace owns a private mapping:

```text
identity_table:
  hashid -> local_slot

slot_table:
  local_slot -> executable callable state
```

Rules:

- `hashid` is valid across the parent and local child workers.
- `local_slot` is valid only inside the target namespace that allocated it.
- Public APIs and task records carry `CallableHandle`, not local slots.
- Local mailbox task frames carry the raw 32-byte digest, not `hashid` strings
  or slots.
- A parent, scheduler, `Orchestrator`, or user orchestration function must not
  know a child worker's `hashid -> local_slot` mapping.
- Registering the same `hashid` with a different descriptor or payload digest
  is an error.
- Public `Worker.register` calls are not deduplicated. Repeated calls may
  return distinct `CallableHandle` objects with the same `hashid`.
- A target may deduplicate executable state by `hashid` internally, but it must
  preserve independent public handle lifetimes with a target-local refcount.

## Canonical Descriptor

`hashid` is computed over canonical descriptor bytes:

```text
hashid = sha256(canonical_descriptor_bytes)
```

The canonical byte stream is versioned and deterministic:

- All integers are unsigned little-endian.
- Strings are UTF-8 encoded as `uint32 byte_len` followed by bytes.
- Byte arrays are encoded as `uint32 byte_len` followed by bytes.
- Lists are encoded as `uint32 count` followed by items in descriptor order.
- Map-like data is not allowed in descriptors.
- Optional fields use `uint8 present` followed by the value when present.
- Enum fields use fixed `uint32` values defined by the descriptor schema.
- Every descriptor starts with `uint32 descriptor_schema_version` and
  `uint32 callable_kind`.

Descriptor enum values:

| Field | Value | Meaning |
| ----- | ----- | ------- |
| `callable_kind` | `1` | `CHIP_CALLABLE` |
| `callable_kind` | `2` | `PYTHON_SERIALIZED` |

The implementation uses internal helpers for descriptor construction and
hashing:

```python
chip_descriptor = build_chip_callable_descriptor(
    target=chip_callable,
    platform=platform,
    runtime=runtime,
)

python_descriptor = build_python_serialized_descriptor(serialized_payload)

chip_hashid = compute_callable_hashid(chip_descriptor)
python_hashid = compute_callable_hashid(python_descriptor)
```

`Worker.register` computes the descriptor and `hashid`. User code does not
provide a hashid override.

Canonical descriptor schemas:

```text
CHIP_CALLABLE:
  descriptor_schema_version
  callable_kind
  target_arch
  platform
  runtime
  callable_blob_sha256: uint8[32]
  signature_schema_hash: uint8[32]

PYTHON_SERIALIZED:
  descriptor_schema_version
  callable_kind
  payload_format_version
  serializer_id
  payload_sha256: uint8[32]
```

For `CHIP_CALLABLE`:

- `target_arch` is the architecture directory selected from the platform,
  such as `a2a3` or `a5`.
- `platform` is the Worker's configured platform id, such as `a2a3sim`,
  `a2a3`, `a5sim`, or `a5`.
- `runtime` is the Worker's configured runtime name.
- `callable_blob_sha256` is computed over the exact contiguous
  `ChipCallable` bytes addressed by `buffer_ptr()` and `buffer_size()`.
- `signature_schema_hash` is the semantic digest of the public chip entry
  signature, defined below.

`CHIP_CALLABLE` identity deliberately excludes build provenance and component
digests from the canonical descriptor:

- `callable_blob_sha256` covers the contiguous `ChipCallable` payload,
  including fixed header fields, the public signature, the orchestration
  binary, and embedded child kernel binaries. Separate `orch_so_sha256` and
  `kernel_binary_sha256[]` values would be redundant for identity. They may be
  recorded as diagnostic or cache metadata outside the canonical hash.
- `compiler_id` and `compiler_version` describe how bytes were produced, not
  the bytes that will execute. Including them would make byte-identical
  callables produce different hashids.
- `runtime_abi_version` is not part of this schema because the repository does
  not define a stable runtime ABI version constant that targets can validate.
  `target_arch`, `platform`, `runtime`, and payload validation are the local
  execution gates.

Any schema change must use a new `descriptor_schema_version` rather than
changing this schema in place.

`signature_schema_hash` is encoded as the raw 32-byte SHA-256 digest in the
descriptor. Logs and diagnostics may render it as `sha256:<64 hex>`. The
digest input uses the same canonical primitive rules as descriptors:

```text
CHIP_SIGNATURE_SCHEMA_V1:
  uint32 signature_schema_version = 1
  uint32 sig_count
  uint32 arg_direction[sig_count]
```

`arg_direction` values are the stable `ArgDirection` enum values:

| Value | Meaning |
| ----- | ------- |
| `0` | `SCALAR` |
| `1` | `IN` |
| `2` | `OUT` |
| `3` | `INOUT` |

The signature hash covers `ChipCallable.signature_[0:sig_count_]` and
`sig_count_` only. It does not cover task-time tensor shapes, data types,
runtime `CallConfig`, child `CoreCallable` signatures, function names, or
executable bytes. Those executable and structural bytes are covered by
`callable_blob_sha256`. Targets must reject a descriptor whose
`signature_schema_hash` does not match the signature decoded from the
`ChipCallable` blob.

`PYTHON_SERIALIZED` covers the existing serialized Python callable route,
including importable functions, lambdas, closures, nested functions, and
callable objects supported by the serializer. Its identity is the serialized
payload identity:

- `payload_format_version` and `serializer_id` identify the existing Python
  callable payload envelope and serializer.
- `payload_sha256` is computed over the exact serializer output bytes. For
  `CLOUDPICKLE`, these are the bytes returned by `cloudpickle.dumps(target)`,
  not the `SPYC` envelope header.

`payload_format_version` is the Python callable wire payload version from
[python-callable-serialization.md](python-callable-serialization.md). The
current value is `1`, matching the `SPYC` header version. Targets validate that
the staged payload header agrees with the descriptor fields before installing
the callable.

`serializer_id` uses the same value space as the `serializer` field in the
`SPYC` Python callable payload header defined by
[python-callable-serialization.md](python-callable-serialization.md):

| Value | Name | Serializer output |
| ----- | ---- | ----------------- |
| `1` | `CLOUDPICKLE` | `cloudpickle.dumps(target)` bytes |

This document does not define a separate serializer registry.

`PYTHON_SERIALIZED` hashids are not semantic Python-code identities. Recreating
an equivalent lambda or closure and serializing it again may produce different
payload bytes and therefore a different hashid. This is acceptable: the handle
identifies the concrete serialized callable payload that was registered.

## Runtime Contracts

Registration returns a callable handle, not an integer child slot:

```python
handle = Worker.register(callable)
```

Registration uses whole-scope install. A successful `Worker.register` means
every active child endpoint in the handle's `target_namespace` for this
`Worker` has installed the callable identity. Registering to a user-selected
worker subset is not part of this contract.
`orch.submit_next_level(..., worker=...)` and
`orch.submit_next_level_group(..., workers=...)` are submit-time affinity
controls; they do not define registration scope. NEXT_LEVEL affinity consumes
stable worker ids. Local Python Worker children and remote L3 workers use the
worker ids returned by `add_worker(...)` / `add_remote_worker(...)`; L3 chip
worker ids are the existing chip worker ids.

`CallableHandle` is the public callable token returned by registration:

```python
CallableHandle(
    hashid: str,
    kind: Literal["CHIP_CALLABLE", "PYTHON_SERIALIZED"],
    target_namespace: str,
)
```

The handle is an opaque parent-side registration object. Its `hashid` is the
stable callable identity used in task frames, but repeated registrations of
the same callable may return distinct handle objects with the same `hashid`.
Unregistering one handle must not invalidate another live handle that shares
the same `hashid`.

The parent Worker tracks live handle ids separately from the target-local
`ref_count`. This keeps public handle lifetime independent from target-side
deduplication: unregistering one live handle consumes exactly that handle and
only releases target-local state when the `ref_count` reaches zero.

Constructing a `CallableHandle` directly from its public fields does not create
a live registration. Submit and unregister paths must validate that the handle
belongs to the `Worker` that returned it and that its public fields still match
the parent-side live registration record.

Submit APIs accept only `CallableHandle`:

```python
matmul = worker.register(chip_callable)
postprocess = worker.register(py_callable)

def parent_orch(orch, args, config):
    orch.submit_next_level(matmul, args, config)
    orch.submit_sub(postprocess, args)
```

They do not accept bare strings or raw callables. Direct string hashids are
registration internals, not submit arguments.

This contract covers the `Worker`, `Orchestrator`, and public `ChipWorker`
wrapper APIs. Manual `callable_id` slots are private implementation details
inside the worker and the underscore `_ChipWorker` binding ABI. They are not
valid `CallableHandle` values, orchestration submit arguments, task slots, or
mailbox task callable references.

Top-level `Worker.run` keeps the current behavior:

- L2 runs a registered `CallableHandle`.
- L3+ runs the raw Python orchestration function in the parent process.

The L3+ orchestration function captures `CallableHandle` values and passes
them to `orch.submit_next_level` or `orch.submit_sub`. Hashid does not add a
new top-level registration requirement for `Worker.run`.

Async worker APIs are level-specific:

```python
# L2 direct chip worker
handle = worker.register(chip_callable)
pending_handle = worker.register_async(chip_callable)
run_handle = worker.run_async(handle, args, config)
unregister_handle = worker.unregister_async(handle)

# L3+ orchestration worker
handle = worker.register(chip_callable)
pending_handle = worker.register_async(chip_callable)
dag_handle = worker.run_async(orch_fn, args, config)
unregister_handle = worker.unregister_async(handle)
```

For L2, `run_async(handle, ...)` submits one chip callable run to the local
chip run lane and returns a run completion handle. Synchronous `run(handle,
...)` submits to the same lane and waits, so sync and async L2 runs share one
ordering queue.

For L3+, `run_async(orch_fn, ...)` is async relative to the caller but runs the
same orchestration DAG body as `run(orch_fn, ...)`. The current DAG scheduler
inside the run remains synchronous: `orch.submit_next_level(...)` uses the
existing ready-task dispatch path, and the DAG run drains before its handle
completes. Synchronous `run(orch_fn, ...)` submits to the same DAG run queue
and waits, so a later sync run cannot overtake an earlier async run.

`register_async(...)` and `unregister_async(...)` are chip-callable APIs in the
current implementation. Passing a Python callable, `RemoteCallable`, or a
non-chip handle raises `TypeError`. Generic `register(...)` remains
synchronous and still supports the existing callable kinds; for `ChipCallable`
it delegates to `register_async(...).wait()`. Generic `unregister(...)`
remains synchronous; for chip handles it delegates to
`unregister_async(...).wait()`.

### Registry Contract

Each target namespace records local identity state. In the current local
mailbox implementation this state is intentionally compact:

```text
identity_table:
  hashid -> local_slot

identity_refs:
  hashid -> ref_count

slot_table:
  local_slot -> {
    hashid,
    callable_kind,
    executable_state,
  }
```

`local_slot` is private to the target process. It may appear in local debug
logs, but it must not appear in public handles, parent-side task slots, or
control replies.

`hashid` itself is content-derived and is not reused for a different callable.
Registering the same hashid with a different descriptor or payload is always a
mismatch error.

Repeated public registrations of the same descriptor increment the target-local
`ref_count` for that `hashid`. Unregistering one handle decrements the
refcount; the target removes local resolution and frees executable state only
when the refcount reaches zero.

Current local slot reuse rule:

- A child resolves `hashid -> local_slot` immediately before execution.
- A run submit holds an in-flight reference to the resolved private slot before
  it enters the run lane.
- Async run/register/unregister controls may overlap a chip child task after
  the child has copied that task's args and published `TASK_RUNNING`.
- Memory and CommDomain controls still wait for an in-flight task dispatch to
  finish before claiming the mailbox.
- Final chip unregister tombstones the identity, rejects new runs through that
  public handle, and releases executable state only when the slot is both
  tombstoned and has no in-flight runs.

This rule prevents stale slot reuse without exposing any extra public field.
A future remote or multi-flight control channel must preserve the same
`INSTALLED` / `TOMBSTONED` / `FAILED` target states, sequence numbers, and
in-flight user draining before it can reuse private slots safely.

### Registration Failure Contract

Synchronous registration remains whole-scope. For a given `target_namespace`,
the scope is every active child endpoint in the current `Worker`'s
corresponding resolver domain at register start. `register_async` for a
`ChipCallable` returns after the async prepare/register request has been
submitted; its `RegisterHandle.wait()` is the whole-scope completion barrier
that either returns the public `CallableHandle` or raises the registration
error.

1. Parent builds the canonical descriptor and computes the `hashid`.
2. Parent allocates an unpublished parent-side registration entry and handle
   id, but does not expose it to user code yet.
3. Parent sends `REGISTER_CALLABLE` to every target in the scope.
4. Target validates descriptor bytes, payload digest, feature gates, and
   namespace.
5. Target installs `hashid -> local_slot`, or increments `ref_count` when the
   same descriptor and payload are already installed.
6. Parent returns the `CallableHandle` only after every target in the scope
   reports success. For `register_async`, this happens from
   `RegisterHandle.wait()`.

If any target fails or times out:

- The parent does not return the handle.
- The parent removes the unpublished handle entry.
- The parent sends cleanup to targets that may have installed the hashid.
- If cleanup cannot be confirmed, that target/hashid pair is marked uncertain
  and must not be used again by the current Worker. The current implementation
  has no recovery path other than Worker restart.
  The local broadcast path returns per-target status; parent-side cleanup sends
  the reverse unregister only to targets that confirmed install or refcount
  increment. A cleanup failure on one of those targets marks the hashid
  uncertain conservatively.

This is failure cleanup with conservative uncertainty handling. The current
implementation treats `_uncertain_hashids` as an intentional unrecoverable
terminal state for the Worker lifetime: a marked digest is blocked until the
Worker restarts. Recovery by successful retry, timeout expiry, or fail-loud
Worker teardown is deferred to a later design.

### Dispatch Contract

Parent-side scheduling assumes the handle's `hashid` is installed on every
active target in its registration scope. Dispatch choices are constrained by
the handle namespace, submit-time affinity, and tensor/buffer accessibility.
Submit-time live validation is a preflight check only. It does not pin the
target identity through later drain or child dispatch. For chip callables, a
run submit pins the resolved slot before enqueueing work. A later
`unregister_async(handle)` removes that public handle from live resolution and
tombstones the slot only when the final public reference is removed. Runs that
already acquired the slot continue to completion; new runs through the
tombstoned handle are rejected. `UnregisterHandle.wait()` is the cleanup
barrier for target-local executable state, not a run result barrier. Call
`RunHandle.wait()` separately when the caller needs run timing or run errors.

Non-chip callable unregister remains synchronous in this PR. Async unregister
for Python callables or remote task dispatcher handles is not part of the
current contract.

Parent-side `TaskSlotState` stores the submitted callable's stable identity:
the 32-byte `sha256` digest plus parent-side scheduling metadata such as
callable kind and target namespace. It never stores a child-local slot, `cid`,
local handle id, or any other integer callable identity. Local mailbox task
frames carry the fixed `sha256` digest.

The target child loop owns the final execution resolve:

```text
TASK(target_namespace, hashid, args)
target namespace identity_table[hashid] -> local_slot
execution engine run(local_slot, args)
```

`ChipWorker` may keep an integer slot API internally, but that integer is an
implementation detail of the target child process.

If a required handle namespace or hashid mapping is missing, fail before
dispatching work to lower-level children. Runtime demand-fetch is not part of
the local worker contract.

Required error codes:

```text
HASHID_FORMAT_INVALID
HASHID_DESCRIPTOR_MISMATCH
CALLABLE_KIND_UNSUPPORTED
LOCAL_SLOT_EXHAUSTED
REGISTER_PARTIAL_FAILURE
REGISTER_CLEANUP_UNCERTAIN
REGISTER_TOMBSTONE_ACTIVE
UNREGISTER_TOMBSTONE_ACTIVE
```

Error messages should include worker id, namespace, `hashid`, and operation.
The current local mailbox has one outstanding operation per endpoint and does
not carry a separate sequence field; a future multi-flight control channel must
add one. Messages must not include user-specific local absolute paths.

### Local Control Contract

Register-family controls carry identity and materialization data, not slots.
The current local IPC encoding is compact because targets can reconstruct the
canonical descriptor from the staged materialization payload and their local
execution context:

```text
callable_hash_digest: uint8[32]
payload:
  staged POSIX shm reference for register
  absent for unregister
```

Rules:

- Register requests never carry a requested slot.
- Register replies never expose target-local slots.
- `target_namespace` and `callable_kind` are determined by the selected worker
  pool and control sub-command, not by a user-provided slot.
- Targets reconstruct or parse the staged payload and recompute the canonical
  descriptor digest locally before install. If a future callable kind cannot
  reconstruct its canonical descriptor from the staged payload and local
  context, its control request must carry descriptor bytes explicitly.
- `REGISTER_CALLABLE` for an already-installed matching hashid increments the
  target-local `ref_count`.
- `REGISTER_CALLABLE` for an already-installed hashid with different
  descriptor or payload digest fails with `HASHID_DESCRIPTOR_MISMATCH`.
- `UNREGISTER_CALLABLE` identifies entries by `target_namespace` and `hashid`
  and decrements the target-local `ref_count`.
- A reply that returns a different `hashid` than requested is invalid.

Local mailbox task callable reference:

```text
MAILBOX_OFF_CALLABLE:
  reserved uint64 = 0

MAILBOX_OFF_ARGS:
  callable_hash_digest: uint8[32]
  task_args_blob: bytes
```

The local mailbox task frame is fixed to `sha256` and carries only the raw
32-byte digest. It does not carry `target_namespace`; the receiving child
mailbox determines the target namespace and resolves the digest in its own
`identity_table`.

This task frame layout is a clean-break wire-format change from the prior
integer callable-id layout where `MAILBOX_OFF_CALLABLE` carried the callable
id. Parent and child processes must use the same mailbox layout version. There
is no compatibility migration path in the current local fork model. The
task-args capacity is computed after the digest prefix, so `task_args_blob`
capacity is reduced by 32 bytes relative to a layout without the prefix.

Unregister uses `hashid` as the target-local primary key:

```text
UNREGISTER_CALLABLE(target_namespace, hashid)
```

Target unregister sequence:

1. Decrement the target-local refcount for `hashid`.
2. If the refcount remains nonzero, keep the mapping installed.
3. If the refcount reaches zero, remove the public digest resolution and mark
   the private slot tombstoned.
4. If no run holds that private slot, call the target-local native unregister
   path immediately and then release the slot for reuse.
5. If any run already holds the private slot, leave executable state installed
   and attach the unregister state to that tombstone.
6. Each run completion decrements the private slot's in-flight count. The
   completion that observes `tombstoned && inflight == 0` performs the native
   unregister/free and completes the unregister state.

This sequence is the concrete unregister form of the target-local slot reuse
rule. Final unregister is non-blocking at submit time for chip callables:
`unregister_async(handle)` returns an `UnregisterHandle` after the tombstone is
submitted, and `UnregisterHandle.wait()` is the cleanup barrier for native
unregister/free. It does not replace `RunHandle.wait()` when the caller needs
the run's timing or error result.

If failed-register cleanup cannot be confirmed, the parent must not dispatch
that hashid to the uncertain target again during the current Worker lifetime.
This protects target-local slot cleanup only; it does not allow the same
hashid to name a different descriptor or payload.

## Implementation Notes

The implementation provides canonical descriptor and hash helpers:

- `build_chip_callable_descriptor` and `build_python_serialized_descriptor`
  produce canonical descriptor bytes for `ChipCallable` and
  `PYTHON_SERIALIZED`.
- `compute_callable_hashid` returns the public `sha256:<hex>` identity.
- Descriptor mismatch and malformed descriptor paths are covered by tests.

The public API is handle-based:

- `Worker.register` returns `CallableHandle`.
- `Worker.register_async` returns `RegisterHandle` for `ChipCallable`
  targets only; non-chip targets raise `TypeError`.
- `Worker.unregister_async` returns `UnregisterHandle` for chip handles only;
  non-chip handles raise `TypeError`.
- `CallableHandle` validation rejects forged, stale, mutated, or
  wrong-namespace handles.
- L2 `Worker.run_async(handle, ...)` submits to the local chip run lane.
- L3+ `Worker.run_async(raw_orch_fn, ...)` submits one whole DAG run to the
  Worker-level DAG run lane.
- L3+ `Worker.run(raw_orch_fn, ...)` behavior is unchanged except that it
  delegates to `run_async(...).wait()` so sync and async DAG runs share one
  ordering queue.
- Integer execution slots remain private to the target child process.

Each target owns identity state:

- The target-local `identity_table` maps `hashid -> local_slot`.
- Target child processes install `hashid -> local_slot` mappings.
- Duplicate public registrations increment target-local refcounts.
- Control replies do not expose slots.

Submit and task records carry identities, not slots:

- Python `Orchestrator` submit APIs accept only `CallableHandle`.
- Nanobind and C++ `Orchestrator` signatures carry digest, callable kind, and
  target namespace.
- `TaskSlotState` stores the 32-byte digest plus scheduling metadata
  instead of integer callable ids.

Local mailbox task frames are hashid-based:

- The local mailbox task payload is prefixed with the 32-byte `sha256` digest.
- The existing `TaskArgs` blob follows the digest prefix.
- Chip child loops resolve `hashid -> local_slot`, copy the args blob, pin the
  private slot, enqueue work on their run lane, and publish `TASK_RUNNING`.
- Sub and Worker-child loops keep the historical synchronous path and publish
  `TASK_DONE` after their Python callable or inner `Worker.run()` returns.
- `ChipWorker.run(local_slot)` remains private to the child process.

Async control overlap is chip-specific:

- `CTRL_REGISTER_ASYNC`, `CTRL_WAIT_REGISTER`, `CTRL_RUN_ASYNC`,
  `CTRL_WAIT_RUN`, `CTRL_UNREGISTER_ASYNC`, and `CTRL_WAIT_UNREGISTER` may
  claim the mailbox while the child is in `TASK_RUNNING`.
- Memory and CommDomain controls still wait for the task to publish
  `TASK_DONE` before claiming the mailbox.
- The control command restores the previous `TASK_RUNNING` state after
  `CONTROL_DONE`, so the parent dispatch path can keep waiting for the
  original task completion.

Register failure cleanup is conservative:

- Handles are not published until every target in scope installed the hashid.
- Failed register removes the unpublished parent handle entry.
- Targets that may have installed the hashid receive reverse cleanup.
- Cleanup uncertainty marks the hashid unavailable for the Worker lifetime.

## Validation

Required tests:

| Test | Expected result |
| ---- | --------------- |
| Stable descriptor hash | Same descriptor bytes produce same hashid. |
| Descriptor mismatch | Rejected with `HASHID_DESCRIPTOR_MISMATCH`. |
| No public slot | Public APIs never return child-local slots. |
| L3 run unchanged | `Worker.run(raw_orch_fn, ...)` still works at L3+. |
| Submit handle only | Submit rejects bare strings and raw callables. |
| Task frame digest | Local mailbox task frames carry the raw 32-byte digest. |
| Private slot resolve | Child loop resolves hashid to private slot. |
| Slot independence | Same hashid runs with different private slots. |
| Duplicate register | Repeated register returns independent handles. |
| Target refcount | Duplicate same-hashid installs share target state safely. |
| Whole-scope register | All active targets in scope installed. |
| Post-start register | Run-time register succeeds after child start. |
| Pre-start register | Startup hashid mappings are visible after ready. |
| Partial register failure | No public handle is returned. |
| Cleanup uncertainty | Unconfirmed cleanup blocks that target/hashid pair. |
| L2 run queue | Sync and async direct chip runs share FIFO ordering. |
| L3 DAG run queue | Sync and async DAG runs share FIFO ordering. |
| Async API type guard | Non-chip async register/unregister inputs raise `TypeError`. |
| Task/control overlap | Chip `TASK_RUNNING` permits async register/run/unregister controls. |
| Deferred unregister | Final unregister tombstones before native free. |
| Unregister cleanup | Hashid resolution stops before final slot cleanup. |
| Unsupported kind | Target rejects unsupported kind before install. |
| Hashid format fuzz | Bad prefix, length, or hex encoding is rejected. |
| No slot consistency | Workers do not need matching private slots. |
