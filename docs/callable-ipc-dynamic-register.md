# Dynamic Callable Registration over IPC

Status: historical design note.

The original version of this document described a cid-based dynamic
registration protocol. That protocol is superseded by
[callable-identity-registration.md](callable-identity-registration.md). Current
implementation and future changes must follow the hashid contract there, not
the old `(cid, shm_name)` control shape.

## Current Local IPC Shape

Dynamic register/unregister is still implemented over the existing child
mailbox control plane, but integer callable slots are target-local internals.
They are never requested by the parent and never returned in public handles or
control replies.

Task dispatch:

```text
MAILBOX_OFF_TASK_CALLABLE_HASH:
  callable_hash_digest: uint8[32]

MAILBOX_OFF_TASK_ARGS_BLOB:
  TaskArgs blob
```

The receiving child resolves the digest through its own local identity table:

```text
hashid digest -> target-local slot -> executable callable state
```

Register control:

```text
MAILBOX_OFF_CALLABLE:
  CTRL_REGISTER

CTRL_OFF_ARG0:
  reserved uint64 = 0

MAILBOX_OFF_ARGS:
  NUL-terminated POSIX shm name

MAILBOX_OFF_CONTROL_CALLABLE_HASH:
  callable_hash_digest: uint8[32]
```

The parent stages the materialization payload in POSIX shared memory. The shm
name format is `simpler-cb-<pid>-<counter>`; it deliberately does not encode a
callable slot.

Unregister control:

```text
MAILBOX_OFF_CALLABLE:
  CTRL_UNREGISTER

MAILBOX_OFF_CONTROL_CALLABLE_HASH:
  callable_hash_digest: uint8[32]
```

## Target Behavior

For a new digest, the child target:

1. Opens the staged shm payload.
2. Reconstructs the payload as either `ChipCallable` or serialized Python
   callable bytes, depending on the control path.
3. Recomputes the canonical descriptor digest and verifies it matches the
   requested `callable_hash_digest`.
4. Allocates a target-local slot.
5. Installs `digest -> slot` and prepares executable state.

For a duplicate matching digest, the child validates the staged payload digest
again and increments the target-local refcount. It does not allocate a new slot
or re-prepare executable state.

Unregister is digest-owned. Each unregister decrements the target-local
refcount; only the final unregister removes digest resolution, clears
executable state, and frees the private slot for reuse.

## L4 Cascade

An L4 parent sends the same digest-owned `CTRL_REGISTER` to each next-level
Worker child. The child loop reconstructs the `ChipCallable` from shm and calls
the inner Worker registration path with the requested digest. The inner Worker
allocates its own local slot and, if already started, broadcasts the same digest
registration to its own children.

No level assumes that a parent slot id matches a child slot id.

## Failure Handling

`Worker.register` publishes a `CallableHandle` only after every active target in
the registration scope reports success. If a register broadcast fails after a
partial install, the parent sends digest-owned unregister cleanup. If cleanup
cannot be confirmed, the parent marks that digest uncertain and rejects later
registration attempts for the same hashid on that Worker. The current local
broadcast API does not report the exact successful-child set, so the parent
uses this conservative uncertain state after any register broadcast failure.

`Worker.unregister` is best effort after a handle has been published. The parent
removes the local handle lifetime and sends digest-owned unregister to active
targets; child errors are reported as warnings while local slots become
available for future registrations according to target-local refcount rules.

## Historical Note

Old revisions of this file described parent-selected numeric callable slots in
control frames and shm names. Those are not valid requirements for the callable
identity implementation. The only stable cross-process callable identity is the
32-byte SHA-256 digest defined by
[callable-identity-registration.md](callable-identity-registration.md).
