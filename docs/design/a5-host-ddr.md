# A5 Host DDR views

A Worker can keep data in Host DDR and explicitly map Tensor views for A5 Device
access, without allocating a Device replica of the full table. L2 maps its own
card for synchronous MTE reads and writes on TMR or HBG. L3 prepares all its A5
children for local MTE access and asynchronous URMA reads through nonself peers;
this distributed path requires TMR and at least two child devices.

## Scope and memory model

For an L2 mapping, the Host Buffer belongs to the Worker in the current process. Its initialized
Worker supplies the target device and context. Kernels access the mapped address
through MTE and complete their transfers before returning; this path requires no
communication domain or URMA workspace.

`create_buffer(nbytes)` allocates ordinary POSIX shared Host storage. Mapping
preserves its ownership, native access grant and same-host sharing behavior.
Existing native Tensors retain CPU access and their ordinary H2D staging behavior.
Only the returned mapped Tensor selects direct Device access. An L2 mapped view is
local to the owning Worker and process; sharing the native Buffer does not share
its Device mapping.

An L2 mapped view cannot be forwarded to another Worker. For multiple L2 consumers,
create and map the Buffer at L3 as described below. L4 and cross-machine mapping,
self-rank URMA loopback, automatic table partitioning and multiple independent
Host registration sets per child are not supported.

## Interfaces and usage

| Interface | Behavior |
| --------- | -------- |
| `worker.create_buffer(nbytes)` | Allocate Host backing with the existing allocation and sharing semantics. |
| `buffer.tensor(shape, dtype, byte_offset=...)` | Construct a native Tensor view of a region of the Buffer. |
| `buffer.host_view()` | Return a Python memoryview of Host backing under its native grant. |
| `worker.map_view(native)` | Return a new Tensor selecting local L2 mapping or all-child L3 mapping. |
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
need, but does not copy the Buffer into Device memory. A later view reuses that
registration while keeping its own geometry and mapping reference. Once the last view is unmapped, mapping
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

## L3 shared backing and asynchronous reads

L3 owns one ordinary POSIX Host Buffer. Every child imports the same full backing,
HostRegisters its local CPU address, and obtains its own Device-visible address.
Each child registers that entire mapped range with a dedicated PTO URMA workspace.
There are no equal-sized Device communication windows or table replicas, and no
user-visible partitioning rule. The same offset through any available nonself peer
names the same logical table bytes. Numerical virtual addresses can differ between
processes and devices.

Call `map_view` after `Worker.init()` and outside an orchestration callback, with
no outstanding runs. Setup is collective across all configured children; passing
the mapped Tensor to fewer children does not reduce the prepared membership.
Address-dependent work happens at mapping time. Worker initialization does not
allocate an unused Host URMA workspace.

```text
L3: create_buffer -> native Tensor -> map_view
    | retain backing and attempted-child cleanup obligations
    | bootstrap child communicators
    +--> every L2: import same POSIX backing -> HostRegister -> mapped GM
    | wait for every local mapping to succeed
    +--> every L2: independent HCCL subcommunicator -> PTO Host workspace
    |              -> publish DMA_WORKSPACE_URMA to resident AICPU state
    | wait for every child to accept the exact view
    v
mapped Tensor -> L3 orchestration args -> selected L2 TaskArgs -> AICore
    | local MTE: Tensor's mapped address
    | URMA: Host workspace + nonself peer base + view/row byte offset
    v
URMA completion token retains the actual workspace -> dependent task can execute
```

The Host workspace is returned by `get_dma_workspace(args, DMA_WORKSPACE_URMA)`.
Device communication domains retain their own `comm_workspace<URMA>(ctx)`;
creating a Host mapping does not replace those workspaces. Each asynchronous
event carries the workspace whose queues the scheduler must poll.

For example, an application can register a compiled chip callable, map a table
once, and submit its views to every child:

```python
worker = Worker(
    level=3, platform="a5", runtime="tensormap_and_ringbuffer",
    device_ids=[0, 1], num_sub_workers=0,
)
entry = worker.register(compiled)
try:
    worker.init()
    table = worker.create_buffer(table_bytes)
    with table.host_view() as host:
        host[:] = initial_table_bytes
    native = table.tensor((rows, width), DataType.BFLOAT16)
    mapped = worker.map_view(native)

    def orchestrate(orch, _args, config):
        for rank in range(2):
            args = TaskArgs()
            args.add_tensor(mapped)
            # Add rank-local output and lookup IDs required by the callable.
            args.add_scalar(1 - rank)
            orch.submit_next_level(entry, args, config, worker=rank)

    worker.submit(orchestrate, None, CallConfig()).wait()
    worker.unmap_view(mapped)
    worker.release_buffer(table)
finally:
    worker.close()
```

The complete [L3 examples](../../examples/a5/tensormap_and_ringbuffer/host_ddr_urma/README.md)
include Device outputs, BF16 row lookup, and MXFP8 lookup feeding the existing
matrix consumer with Device scales. Kernels use `UrmaTget` and
`send_request_entry`. Materialization folds the original view offset into the
Tensor address, so kernels derive its byte offset by subtracting the local rank's
registered base from the Tensor's effective address, then add the row offset.
The examples pass the local rank as a scalar and query both bases through PTO.
A peer address is not generally numerically equal to the current card's mapped
Tensor address.

Only one Host backing can be registered per L3 group at a time. Additional views
of that backing reuse the mappings, communicator and workspaces. Views retain
independent references and bounds; revoked references cannot be revived. To map
a different backing, wait for all runs and remove the previous views first.
Native CPU access and ordinary H2D staging remain available through native Tensors.
Mapped Tensors accept IN, OUT and INOUT with the original Buffer grant; the
asynchronous examples perform reads. Concurrent writes to shared table rows need
application-level ordering.

## Distributed lifetime and failures

An L3 accepted run blocks registration changes until its cleanup is published,
even if a kernel accesses the Host workspace without a mapped Tensor argument.
Each L2 also reserves its workspace before native admission under the complete
dispatch identity. Unknown native completion retains that reservation. Host
updates are allowed only after every Device use has drained.

Unmap, Buffer release and Worker close first quiesce every participating child.
Only after every child acknowledges quiescence may release revoke the published
workspace, destroy the Host HCCL subcommunicator, finalize PTO workspace storage,
and unregister the Host mapping. The owning L3 backing is released last. A child
cannot independently unregister exposed Host storage during spontaneous teardown.

Setup and cleanup install ownership records before side effects. An error or lost
reply does not prove that a remote registration disappeared. Unconfirmed cleanup
keeps the backing and transport alive and rejects new use; retry is possible only
where the backend and control endpoint still provide a valid cleanup path. An
unrecoverable device/control failure can therefore retain resources until process
termination. Process termination is not reported as successful DMA completion.
