# A3 Worker-owned Host buffers

Host storage and explicit endpoint mappings belong to the Worker. This is a
Buffer capability extension, independent of communication domains. Buffer, Tensor,
TaskArgs and ImportRegistry remain the common model. Single-node A3 L2/L3 are
supported; extending device communication domains to Host storage is separate work.

## Allocation and mapping

`Worker.create_buffer(nbytes)` retains ordinary POSIX shared memory allocation.
No device registration occurs until an explicit map request. CPU-only buffers,
small controls and existing staging retain their usual allocation behavior.

```python
buffer = worker.create_buffer(4096)
native = buffer.tensor((1024,), DataType.FLOAT32)
mapped = worker.map_view(native, access=AccessMode.READ)  # L2 target defaults to device_location
args = TaskArgs()
args.add_tensor(mapped, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
# Run and retire all accepted uses before cleanup.
worker.unmap_view(mapped)
worker.release_buffer(buffer)
```

`buffer.buffer` exposes native CPU bytes independently of backing kind. Native
memoryviews, including their derived views, retain the CPU mapping and prevent
physical release until dropped. `Buffer.base` is a creator-process CPU address,
not a foreign-process pointer. `Buffer.close()` revokes its handle; the Worker
owns physical cleanup through `release_buffer`.

`Worker.map_view(tensor, target=..., access=...)` handles Host and registered
local DEVICE sources. The returned Tensor preserves the source identity,
placement and geometry while selecting a separate restricted mapping. Native
Host access stays available. An already usable HBM Tensor need not be mapped;
a local DEVICE map reuses its existing device address.

The target registry validates identity/generation, execution context, range,
profile, liveness and grant. Consumer requirements come from callable signatures:
`required access ⊆ mapping grant ⊆ source grant`. There is no domain-specific
fourth grant and no pre-registration of alternative permissions. References are
lookup keys, never permissions: they are backing-scoped, never reused, encode
neither rank nor access, and fail on exhaustion. Mapped parameters require explicit
`TensorTransfer.NONE`; native and mapped arguments can coexist in one invocation.

The A3 platform registers each target process's local shared-memory address with
HostRegister. No Fabric export/import, PID whitelist or Host-target HAL mapping
is required. POSIX shared memory supplies CPU process sharing. The target registry
retains the shared-memory import and device registration until unmap succeeds.
The platform runtime owns device context selection and HostUnregister; the common
bridge retains the runtime module while native cleanup remains owed.

## L3 local endpoints

L3 maps to an existing local next-level worker id, using the same ids as
`orch.submit_next_level(..., worker=...)`. No domain is required:

```python
buffer = worker.create_buffer(nbytes)
native = buffer.tensor(shape, dtype)
mapped = worker.map_view(native, target=0, access=AccessMode.READ)
# Pass mapped in ordinary TaskArgs with transfer=TensorTransfer.NONE.
# Wait for accepted uses, then unmap and release the Buffer.
worker.unmap_view(mapped)
worker.release_buffer(buffer)
```

`target` is a local worker id, not a physical device id or domain rank. L2 keeps
its existing default current-device target. Host and registered local DEVICE
sources use this same entry. Communication domains keep their existing HBM
allocation and release behavior and do not own or borrow Host mappings.
`CommContext` remains 1088 bytes, without a Host field.

## Completion and cleanup

Mapping is neither an H2D copy nor a readiness fence. Finish CPU writes before
starting device readers, and finish readers before modifying the live Host table.
CPU pointer writes require caller synchronization.

Map/unmap controls run before a callback's first submission, or outside callbacks
with conflicting accepted uses drained. `unmap_view` refuses in-flight use before
revocation; domain lifetime has no effect on a Host mapping.

After accepted uses end, unmap first revokes the selected grant. The
last active record releases the endpoint's device registration. Failure remains
CLOSING, rejects new use, and retains cleanup ownership for retry. Old references
never revive. `release_buffer` refuses remaining mappings and releases
the original backing after mappings and accepted uses retire. Worker close reaps
consumers and reports unconfirmed cleanup rather than freeing unsafe storage.

## Qualification

Focused acceptance covers native/mapped same-call consumers, restricted grants,
wrong/stale contexts, unmap failure/retry, retained BF16 row reads, Host/HBM MTE,
PTO SDMA completion and independent endpoint mapping cleanup. Existing WRITE/READWRITE and
application cases retain their manual selection. Do not multiply these into a
runtime/engine/size CI matrix.

Qualification evidence must match this ownership refactor; historical tests at
`ef701aaa2941` cover the earlier domain-coupled implementation. A5 and cross-node access,
Engram table layout and communication/computation overlap remain unqualified.
This work does not close issues #2460 or #2657.
