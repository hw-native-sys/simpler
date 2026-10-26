# A5 same-device Host DDR access

A direct L2 Worker can keep data in Host DDR and give its own A5 device a mapped
Tensor view of that storage. This supports Engram lookups without allocating a
Device replica of the full table. The same mapping also supports Device writes
to Host DDR. TMR and HBG share this interface and mapping implementation.

## Scope and memory model

The Host Buffer belongs to the L2 Worker in the current process. Its initialized
Worker supplies the target device and context. Kernels access the mapped address
through MTE and complete their transfers before returning; this path requires no
communication domain or URMA workspace.

`create_buffer(nbytes)` allocates ordinary POSIX shared Host storage. Mapping
preserves its ownership, native access grant and same-host sharing behavior.
Existing native Tensors retain CPU access and their ordinary H2D staging behavior.
Only the returned mapped Tensor selects direct Device access. The mapped view is
local to the owning Worker and process; sharing the native Buffer does not share
its Device mapping.

L3-to-L2 mapping, other-device access, cross-machine access and asynchronous URMA
completion are outside this design. They require separate registration and
lifetime coordination.

## Interfaces and usage

| Interface | Behavior |
| --------- | -------- |
| `worker.create_buffer(nbytes)` | Allocate Host backing with the existing allocation and sharing semantics. |
| `buffer.tensor(shape, dtype, byte_offset=...)` | Construct a native Tensor view of a region of the Buffer. |
| `buffer.host_view()` | Return a Python memoryview of Host backing under its native grant. |
| `worker.map_view(native)` | Return a new Tensor selecting this Worker's Device mapping. |
| `worker.unmap_view(mapped)` | Revoke the view after its accepted uses complete; unregister backing when the last view is removed. |
| `worker.release_buffer(buffer)` | Release owned backing after accepted uses complete, cleaning up any remaining mapping. |

`map_view` has no `access` or `platform` argument. The source Buffer grant remains
in effect, and callable directions describe the kernel's reads and writes:

| Callable direction | TaskArgs tag for existing storage | Usage |
| ------------------ | --------------------------------- | ----- |
| `IN` | `TensorArgType.INPUT` | Device reads Host DDR, for example an Engram table. |
| `OUT` | `TensorArgType.OUTPUT_EXISTING` | Device writes results directly to Host DDR. |
| `INOUT` | `TensorArgType.INOUT` | Device reads and updates Host DDR. |

These directions participate in dependency tracking; an `IN` declaration is not
hardware write protection. Kernels must honor their declared direction just as
they do for ordinary Device Tensors.

The following example accepts a compiled callable with one Float32 Tensor
parameter declared `INOUT`. Its kernel updates the Tensor through the supplied
Device address and finishes its MTE transfers before returning:

```python
import numpy as np
from simpler.task_interface import CallConfig, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker


def update_host_data(compiled, device_id=0):
    worker = Worker(
        level=2, platform="a5", runtime="tensormap_and_ringbuffer",
        device_id=device_id,
    )
    try:
        worker.init()
        entry = worker.register(compiled)
        buffer = worker.create_buffer(256 * 4)
        with buffer.host_view() as host:
            host[:] = np.arange(256, dtype=np.float32).tobytes()

        native = buffer.tensor((256,), DataType.FLOAT32)
        mapped = worker.map_view(native)
        args = TaskArgs()
        args.add_tensor(mapped, TensorArgType.INOUT)  # Defaults to transfer NONE.
        pending = worker.submit(entry, args, CallConfig())
        pending.wait()

        with buffer.host_view() as host:
            result = np.frombuffer(host, dtype=np.float32).copy()
        worker.unmap_view(mapped)
        worker.release_buffer(buffer)
        return result
    finally:
        worker.close()
```

Select `runtime="host_build_graph"` with a callable compiled for HBG to use the
same API there. Complete kernels and orchestration are in the
[Host DDR examples](../../examples/a5/tensormap_and_ringbuffer/host_ddr_gather/README.md):
`run.py` performs BF16 gather, `run_mx.py` combines a Host MXFP8 payload with Device
scales, and `run_write.py` uses disjoint OUT and INOUT views of one Host Buffer.

## Mapping and submission flow

```text
L2 Host process                             Local A5 device

create_buffer(nbytes)
  -> POSIX Host DDR backing
buffer.tensor(shape, dtype, offset)
  -> native Tensor
worker.map_view(native)
  -> validate ownership and live context
  -> register whole Buffer once:
       aclrtHostRegisterV2(..., ACL_HOST_REG_MAPPED)
       aclrtHostGetDevicePointer(...)
  -> return Tensor with local mapping reference

submit(callable, args)
  -> retain accepted Buffer/view uses
  -> ImportRegistry resolves mapped base
     and preserves Tensor offset/geometry
  -> pass Device address in kernel args ----> MTE reads/writes Host DDR
                                             -> transfers complete
wait() <------------------------------------- task completion
  -> retire accepted uses
  -> CPU can consume Device-written results

unmap_view(mapped)
  -> revoke this view
  -> last view: aclrtHostUnregister(...)
release_buffer(buffer)
  -> release Host backing
```

The physical registration covers the entire Buffer and is reused by subsequent
views in the same context. This registers more memory than a small Tensor view may
need, but does not copy the Buffer into Device memory. A later view reuses that registration while keeping
its own geometry and mapping reference. Once the last view is unmapped, mapping
again requires a new physical registration.

Other native Tensors on the same Buffer retain their original behavior. Although
the backing is physically registered as a whole, the runtime only selects the
Device address for arguments carrying a valid mapping reference. View bounds are
validated by the runtime; this is not per-Tensor hardware memory isolation.

The mapped Tensor retains the Buffer's canonical identity, HOST placement and
view geometry. The consumer ImportRegistry owns the Device address, context and
mapping lifetime. Native arguments resolve by canonical identity; mapped
arguments resolve by identity plus mapping reference. This permits native and
mapped arguments to coexist without confusing their addresses, while alias and
dependency tracking retain the common backing identity.

Mapped arguments use `TensorTransfer.NONE`, selected automatically when omitted.
Explicit staging requests on a mapped view are rejected. No whole-table H2D or
output D2H copy is inserted; Device writes already target Host backing. Use the
native Tensor when ordinary staging is wanted.

## Ordering and lifetime

Finish CPU initialization before submitting Device work. Wait for Device writers
before reading their results on CPU, and wait for all Device uses before modifying
Host backing. Mapping establishes addressability; it does not synchronize
concurrent CPU and Device accesses.

Accepted submissions retain the backing and mapping until completion is proven.
Unmap and release reject outstanding uses. `Worker.close()` drains accepted runs
before releasing their resources. A timeout or failed execution that leaves
completion unknown retains resources rather than assuming the Device has stopped.

Unmapping closes admission before calling the backend. If unregistering fails,
backing and context stay retained for cleanup retry and new uses are rejected.
Stale references, changed geometry, another process or Worker, and unsupported
mapping backends fail explicitly.

Release exported Python memoryviews before releasing the Buffer. Direct
`buffer.shm.close()` or `unlink()` cannot bypass an active registration; use the
Worker/Buffer lifecycle interfaces to clean up mapped backing.
