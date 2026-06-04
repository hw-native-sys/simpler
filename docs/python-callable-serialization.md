# Python Callable Serialization for L3+ Register

This document describes the current serialized-Python callable payload used by
hierarchical `Worker.register` for Python callables. The public registration
contract is defined in
[callable-identity-registration.md](callable-identity-registration.md):
registration returns a `CallableHandle`, task frames carry the handle hash
digest, and integer execution slots remain target-local internals.

## Payload Envelope

Python callable registration uses a small binary envelope followed by the
serializer bytes:

```text
SPYC payload:
  magic: b"SPYC"
  version: uint8 = 1
  serializer: uint8 = 1  # CLOUDPICKLE
  flags: uint16 = 0
  payload_size: uint64
  payload: bytes
```

The `payload` field is `cloudpickle.dumps(target)`.

Targets validate the envelope before unpickling:

- magic must be `SPYC`;
- version must be supported;
- serializer must be `CLOUDPICKLE`;
- flags must be zero;
- `payload_size` must fit within the staged bytes.

This feature unpickles code/data produced by the same user process. It is not
a security boundary and must not be used for untrusted bytes.

## Hashid Descriptor

`PYTHON_SERIALIZED` callable identity is computed from the serialized payload
identity, not from a semantic Python source-code identity:

```text
PYTHON_SERIALIZED descriptor:
  descriptor_schema_version
  callable_kind = PYTHON_SERIALIZED
  payload_format_version
  serializer_id
  payload_sha256
```

`payload_sha256` is computed over the serializer bytes inside the `SPYC`
envelope, not over the header itself.

Equivalent Python functions may produce different cloudpickle bytes and
therefore different hashids. This is expected: the handle identifies the exact
serialized callable payload registered on that Worker.

## Registration Flow

Before children are started, Python callables are installed into the parent
identity registry and inherited by forked children through the startup
snapshot.

After children have started, the parent:

1. serializes the callable into the `SPYC` payload;
2. computes the `PYTHON_SERIALIZED` descriptor and hash digest;
3. creates an unpublished parent-side registration entry and handle id;
4. stages the `SPYC` payload in POSIX shared memory;
5. broadcasts a Python register control with the staged shm name and digest to
   Python-capable children.

The parent returns the handle only after every target reports success. If any
target fails, the unpublished entry is removed and confirmed installs are
cleaned up by digest-owned unregister control.

The child:

1. reads the digest from the control frame;
2. opens the staged shm payload;
3. validates the envelope and recomputes the descriptor digest;
4. rejects the request with `HASHID_DESCRIPTOR_MISMATCH` if the payload digest
   does not match the requested digest;
5. increments the target-local refcount for an existing matching digest, or
   unpickles and installs the callable into a new private local slot.

## Unregister

Unregister is digest-owned. A child decrements the target-local refcount for
the digest and removes the local mapping only when the final reference is
released. The parent does not send or reuse child slot numbers.

If post-start register fails after a partial install, the parent sends
best-effort digest-owned cleanup. If cleanup cannot be confirmed, that hashid
is marked uncertain on the Worker and later attempts to register the same
hashid are rejected until the Worker is restarted.

## Supported Targets

Python callable serialization applies to Python-capable hierarchical children:

- SUB workers for `orch.submit_sub`;
- next-level Worker child dispatch loops for Python orchestration callables.

It does not populate an inner Worker's own registry for callables that are
owned by that inner Worker. Those must be registered on the inner Worker.
