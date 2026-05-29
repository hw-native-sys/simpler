# Callable Identity Registration

This document defines callable identity for local hierarchical workers.
`hashid` is the stable identity across the parent process and its local child
worker processes. Integer execution slots are target-worker internals and are
not exposed through public APIs, task slots, or mailbox task frames.

Related background:

- [python-callable-serialization.md](python-callable-serialization.md)
- [callable-ipc-dynamic-register.md](callable-ipc-dynamic-register.md)

These documents describe existing cid-based materialization paths. This
document defines the hashid identity contract and does not depend on
`remote-l3-worker-design.md`. Remote workers are outside the scope of this
document.

## Overview

Hierarchical callable registration needs to handle two cases that a public
integer callable id cannot model safely:

1. A higher-level task may depend on lower-level callables that are only known
   at run time. For example, a parent orchestration callable may depend on a
   child `ChipCallable`.
2. Worker-local execution slots may differ across child workers. A parent or
   orchestration function must not depend on those slot values being inherited
   or consistent.

The local worker design is materialize-before-dispatch:

```text
bootstrap workers
register callable identities
install target-local executable slots
dispatch tasks by callable handle
target worker resolves handle hashid to its local slot
```

Runtime demand-fetch is not part of the local worker contract. A child never
pauses task execution to ask its parent for a missing callable binary;
registration must materialize executable state before dispatch.

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
requires a Python dispatch handle. The receiving child mailbox determines its
own namespace, so local task frames carry only the raw `sha256` digest.

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
- Local mailbox task frames carry `hashid`, not slots.
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
descriptor = build_callable_descriptor(callable)
hashid = compute_callable_hashid(descriptor)
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
  callable_blob_sha256
  signature_schema_hash

PYTHON_SERIALIZED:
  descriptor_schema_version
  callable_kind
  payload_format_version
  serializer_id
  payload_sha256
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
controls; they do not define registration scope.

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

Top-level `Worker.run` keeps the current behavior:

- L2 runs a registered `CallableHandle`.
- L3+ runs the raw Python orchestration function in the parent process.

The L3+ orchestration function captures `CallableHandle` values and passes
them to `orch.submit_next_level` or `orch.submit_sub`. Hashid does not add a
new top-level registration requirement for `Worker.run`.

### Registry Contract

Each target namespace records identity state:

```text
identity_table:
  hashid -> {
    local_slot,
    callable_kind,
    descriptor_version,
    payload_digest,
    ref_count,
    state,
  }

slot_table:
  local_slot -> {
    hashid,
    callable_kind,
    executable_state,
  }
```

States:

- `INSTALLED`: visible to local hashid resolution and subsequent tasks.
- `TOMBSTONED`: target-local unregister or failed-register cleanup is in
  progress.
  The hashid has been removed from local resolution, and the private slot is
  not reusable until in-flight users have drained.
- `FAILED`: target state is uncertain or cleanup failed.

`local_slot` is private to the target process. It may appear in local debug
logs, but it must not appear in public handles, parent-side task slots, or
control replies.

`hashid` itself is content-derived and is not reused for a different callable.
Registering the same hashid with a different descriptor or payload is always a
mismatch error. Tombstone state protects only target-local cleanup and private
slot reuse; it is not a public hashid reuse guard.

Repeated public registrations of the same descriptor increment the target-local
`ref_count` for that `hashid`. Unregistering one handle decrements the
refcount; the target removes local resolution and frees executable state only
when the refcount reaches zero.

Target-local slot reuse rule:

- A child resolves `hashid -> local_slot` immediately before execution.
- Unregister and failed-register cleanup remove the hashid from resolution
  before cleanup.
- If a task already resolved the hashid, cleanup waits for that task to finish.
- The private slot may be freed or reused only after those users have drained.

This rule prevents stale slot reuse without exposing any extra public field.

### Registration Failure Contract

Registration remains synchronous and whole-scope. For a given
`target_namespace`, the scope is every active child endpoint in the current
`Worker`'s corresponding resolver domain at register start.

1. Parent builds the canonical descriptor and computes the `hashid`.
2. Parent allocates a parent-side `CallableHandle` entry, but does not expose
   it to user code yet.
3. Parent sends `REGISTER_CALLABLE` to every target in the scope.
4. Target validates descriptor bytes, payload digest, feature gates, and
   namespace.
5. Target installs `hashid -> local_slot`, or increments `ref_count` when the
   same descriptor and payload are already installed.
6. Parent returns the `CallableHandle` only after every target in the scope
   reports success.

If any target fails or times out:

- The parent does not return the handle.
- The parent removes the unpublished handle entry.
- The parent sends cleanup to targets that may have installed the hashid.
- If cleanup cannot be confirmed, that target/hashid pair is marked uncertain
  and must not be used again until cleanup is confirmed or the worker restarts.

This is failure cleanup with conservative uncertainty handling. It stays on
the same synchronous install path and does not introduce a separate recovery
API.

### Dispatch Contract

Parent-side scheduling assumes the handle's `hashid` is installed on every
active target in its registration scope. Dispatch choices are constrained by
the handle namespace, submit-time affinity, and tensor/buffer accessibility.

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
UNREGISTER_TOMBSTONE_ACTIVE
```

Error messages should include endpoint id, namespace, `hashid`, operation, and
sequence number. They must not include user-specific local absolute paths.

### Local Control Contract

Register-family controls carry identity and materialization data, not slots:

```text
target_namespace
callable_kind
callable_hash_digest: uint8[32]
descriptor_version
descriptor_len
descriptor bytes
payload_kind:
  INLINE_BYTES
  STAGED_REF
payload_len_or_ref
payload bytes or staged reference
```

Rules:

- Register requests never carry a requested slot.
- Register replies never expose target-local slots.
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

Unregister uses `hashid` as the target-local primary key:

```text
UNREGISTER_CALLABLE(target_namespace, hashid)
```

Target unregister sequence:

1. Decrement the target-local refcount for `hashid`.
2. If the refcount remains nonzero, keep the mapping installed.
3. If the refcount reaches zero, stop new local resolutions from `hashid` to
   private slot.
4. Mark the entry `TOMBSTONED`.
5. Wait until no in-flight task that already resolved this hashid is using the
   private slot.
6. Clear executable state.
7. Release the private slot for reuse.
8. Remove or archive the `hashid` entry.

This sequence is the concrete unregister form of the target-local slot reuse
rule.

If failed-register cleanup cannot be confirmed, the parent must not dispatch
that hashid to the uncertain target again until cleanup is confirmed or the
worker restarts. This protects target-local slot cleanup only; it does not
allow the same hashid to name a different descriptor or payload.

## Implementation Plan

Milestone 1: descriptor and hash helpers.

- Add canonical descriptor builders for `ChipCallable` and
  `PYTHON_SERIALIZED`.
- Add `compute_callable_hashid`.
- Add descriptor mismatch and malformed descriptor tests.

Milestone 2: public handle API.

- Add `CallableHandle` and handle validation.
- Change `Worker.register` to return `CallableHandle`.
- Keep L3+ `Worker.run(raw_orch_fn, ...)` behavior unchanged.
- Keep integer execution slots private to the target child process.

Milestone 3: target identity registry.

- Add per-target `identity_table` and private slot allocation.
- Register `hashid -> local_slot` in target child processes.
- Add target-local refcounts for duplicate public registrations.
- Ensure control replies do not expose slots.

Milestone 4: submit and task slots.

- Change Python `Orchestrator` submit APIs to accept only `CallableHandle`.
- Update nanobind and C++ `Orchestrator` signatures.
- Change `TaskSlotState` to store the 32-byte digest plus scheduling metadata
  instead of integer callable ids.

Milestone 5: local mailbox task hashid.

- Prefix the local mailbox task payload with the 32-byte `sha256` digest.
- Shift the existing `TaskArgs` blob after the digest prefix.
- Resolve `hashid -> local_slot` in the chip/sub child loop.
- Keep `ChipWorker.run(local_slot)` private to the child process.

Milestone 6: register failure cleanup.

- Do not publish a handle until every target in scope installed the hashid.
- On failed register, remove the unpublished parent handle entry.
- Clean up targets that may have installed the hashid.
- Mark target/hashid cleanup uncertainty conservatively.

## Validation

Required tests:

| Test | Expected result |
| ---- | --------------- |
| Stable descriptor hash | Same descriptor bytes produce same hashid. |
| Descriptor mismatch | Same hashid with different descriptor is rejected. |
| No public slot | Public APIs never return child-local slots. |
| L3 run unchanged | `Worker.run(raw_orch_fn, ...)` still works at L3+. |
| Submit handle only | Submit rejects bare strings and raw callables. |
| Task frame hashid | Local mailbox task frames carry hashid. |
| Private slot resolve | Child loop resolves hashid to private slot. |
| Slot independence | Same hashid runs with different private slots. |
| Duplicate register | Repeated register returns independent handles. |
| Target refcount | Duplicate same-hashid installs share target state safely. |
| Whole-scope register | All active targets in scope installed. |
| Post-start register | Run-time register succeeds after child start. |
| Pre-start register | Startup hashid mappings are visible after ready. |
| Partial register failure | No public handle is returned. |
| Cleanup uncertainty | Unconfirmed cleanup blocks that target/hashid pair. |
| Unregister tombstone | Hashid resolution stops before slot cleanup. |
| Unsupported kind | Target rejects unsupported kind before install. |
| Hashid format fuzz | Bad prefix, length, or hex encoding is rejected. |
| No slot consistency | Workers do not need matching private slots. |
