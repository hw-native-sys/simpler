# A5 local Host DDR access

A direct L2 Worker can use `TensorTransfer.MAP` to let its A5 device access owned
Host DDR through MTE. This supports large Host tables without a full Device
replica and supports Device writes to Host backing. TMR and HBG share the API and
registration implementation. No communication domain or URMA workspace is needed.

## Scope and API

`create_buffer(nbytes)` allocates ordinary POSIX shared Host storage. Tensors keep
their HOST descriptor, canonical identity, native grant and view geometry. Transfer
is an argument-level request, chosen independently on each call:

| Request | Meaning |
| ------- | ------- |
| omitted | HOST defaults to H2D; DEVICE defaults to NONE. |
| `NONE` | Keep native placement without H2D staging; HOST/NONE is explicit. |
| `H2D` | Use ordinary Program-managed staging, with copies governed by callable direction. |
| `MAP` | Prepare/reuse the owning L2 context's mapping and pass its Device address. |

`TaskArgs.add_tensor(..., transfer=TensorTransfer.MAP)` only records intent.
`Worker.submit` and `Worker.prepare_submission` share L2 argument preparation,
which prepares the mapping before passing arguments to the native runtime.
Readonly HOST/NONE control contents are snapshotted at this boundary; MAP entries
remain DEVICE/NONE addresses borrowing the registered Host backing through
completion or proven cancellation. No public mapped Tensor or explicit map/unmap
operation is required.

MAP supports IN, OUT and INOUT, with the usual INPUT, OUTPUT_EXISTING and INOUT
argument tags. Buffer grants remain authoritative. Directions describe kernel use
and dependencies; IN is not hardware write protection.

Direct L2 supports owned POSIX_SHM on A5 TMR and HBG. Same-machine L3 supports
A5 TMR through the coordinated preparation below. Remote access, Host compute
leaves, borrowed L2 contexts, L3 HBG, A3 and capture paths reject MAP.

## Usage

The callable below declares one Float32 INOUT argument and finishes its MTE
transfers before returning:

```python
import numpy as np
from simpler.task_interface import DataType, TaskArgs, TensorArgType, TensorTransfer
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

        tensor = buffer.tensor((256,), DataType.FLOAT32)
        args = TaskArgs()
        args.add_tensor(tensor, TensorArgType.INOUT, transfer=TensorTransfer.MAP)
        worker.submit(entry, args).wait()

        with buffer.host_view() as host:
            result = np.frombuffer(host, dtype=np.float32).copy()
        worker.release_buffer(buffer)
        return result
    finally:
        worker.close()
```

Use `runtime="host_build_graph"` with a callable compiled for HBG to use the same
API. The [Host DDR examples](../../examples/a5/tensormap_and_ringbuffer/host_ddr_gather/README.md)
cover BF16 gather, Host MXFP8 payloads with Device scales, and disjoint OUT/INOUT
views of one Host Buffer.

## Submission and release

```text
create_buffer -> ordinary POSIX Host backing
buffer.tensor -> ordinary HOST Tensor
args.add_tensor(tensor, transfer=MAP)
  -> record the per-argument request

submit
  -> validate grants, ownership, capability and mixed H2D/MAP aliases
  -> retain run-scoped Buffer uses under the owner lock
  -> register whole backing once in this device context
  -> resolve each entry independently
       MAP: mapped Device base + DEVICE/NONE
       native: native base + original address space/transfer
  -> preserve each Tensor's offset and geometry
  -> snapshot readonly HOST/NONE controls; retain MAP addresses and backing
  -> run kernel: MTE reads/writes Host DDR

wait -> prove completion -> release this run's uses
     -> keep cached registration for later calls

release_buffer / Buffer.close
  -> reject outstanding runs and live Python exports
  -> close admission and unregister
  -> close/unlink owned Host backing
```

The whole-Buffer cache is process/context local. Each view and each argument keeps
its own geometry; no per-view mapping ID, counter or separate lifecycle state
machine is added. The registry retains the existing closing flag, native
registration state and run references. Registry publication and first preparation
are serialized, including concurrent first use of the same Buffer. Buffer release
and preparation share the owner lock.

Resolved arguments form a list indexed by argument position, not a dictionary
collapsed by Buffer identity. Thus HOST/NONE control arguments and MAP data
arguments from the same Buffer coexist. H2D/MAP overlap with any callable writer
is rejected before registration, since staging and direct Host writes could race.
Read-only overlap is allowed; disjoint writable views are allowed.

## Ordering and failure behavior

Finish CPU writes before submission. Wait for Device writers before consuming
results on CPU, and wait for all Device uses before modifying Host storage.
Mapping establishes addressability, not CPU/Device synchronization. Device writes
need no D2H copy because they already target Host backing.

Accepted calls retain backing and registration until completion is proven. Failed
direct and queued submissions also release their uses when the native runtime
proves that the per-call context was destroyed and the caller's last use ended.
Host use tracking also covers owned HOST/NONE controls on borrowed contexts.
Framework-owned submissions release these uses after proven completion or
cancellation, before discarding the run record and publishing terminal state.
Device-source cleanup keeps its existing ambiguous-decrement guard; a later Host
cleanup retry must not repeat a successful Device-source release. Unknown
completion retains the Host uses.
A busy release refuses without closing admission. `Worker.close()` drains accepted runs
before releasing resources. Timeout or failed execution with unknown completion
retains resources and prevents context reuse; it does not assume Device access has
stopped. Automatic recovery of an unproven context is not provided.

Before exceptional native owner teardown, unused cached mappings are unregistered
while the device context is still alive. Mappings with outstanding or unproven
uses remain retained; an idle cache cannot block recovery of an unrelated run.

Once unregister starts, admission is closed. Failed unregister keeps backing and
context available for cleanup retry. Preparation rollback preserves its original
error and retains any cleanup debt. Unsupported MAP requests fail explicitly;
they do not silently become H2D.

Release Python memoryviews and all derived NumPy arrays before closing the Buffer.
Export carriers pin the actual Host mapping even when NumPy outlives a temporary
`host_view()`. Raw `buffer.shm.close()` and `unlink()` cannot bypass a registered
Buffer's lifetime checks. Communication-domain mappings reuse export tracking but
retain their own allocation and unlink ownership rules.

## L3 shared tables

An L3 Worker owns complete ordinary Host Buffers. Each local A5 TMR child imports
that same backing and registers its own Device mapping when preparing a child
call with `TensorTransfer.MAP`. This is the same parameter-preparation boundary
at which H2D requests are handled. Outer arguments need no MAP declaration: they
can carry HOST/NONE views, or the callback can construct child args from owned
Buffers directly. Subviews keep their canonical descriptor, offsets and grants.

```text
L3 Host Buffer -> callback submits child args (MAP)
                       |                  |
                  L2-0 import        L2-1 import
                  Device0 map        Device1 map
                       |                  |
                  local MTE          local MTE
```

Only children receiving the backing prepare it. Mapping sends no preparation
control messages and creates no peer relationship, rank communication, MR, URMA
channel or communication workspace. One child is supported; qualification uses
at least two distinct A5 devices and processes. Multiple backings and views
coexist. There is no table-sized HBM replica or payload copy in the control
transport.

Registrations are cached across calls. First use of a new backing does not
require earlier calls to drain; each child establishes its mapping before its
native task can use the address. Child submission resolves each MAP entry to
that child's Device address; native NONE/H2D entries keep their original behavior.
The same writable-alias restrictions as direct L2 apply to each chip callable.
Sharing backing does not order conflicting writers on different devices. CPU
updates require every Device user of that backing to have drained.

Preparation failures follow ordinary child-call failure semantics, as for H2D.
A child mapping failure does not roll back work already dispatched to other
children or establish an all-device readiness barrier. Existing task/group
activation and completion rules still apply. The call reports failure while
accepted tasks and unresolved native cleanup retain their resources.

Before dispatch, the parent retains backing and records every possibly submitted
child, even if submission later fails. Children retain full dispatch identities
for HOST/NONE, H2D and MAP, including uses preceding the first mapping.
Completion or a native proof of safe failed cleanup retires only that invocation.
Unknown completion retains the child use even if the parent receives a task error.
Stale activation also requires the native cleanup proof before retiring Host uses:
a terminal run can return from `abandon()` while cleanup remains unproven.
Release first closes admission and checks all children, then unregisters each local
mapping and closes its CPU import. Backing remains owned until every cleanup reply
is confirmed. A lost reply or failed unmap retains the closed record for idempotent
retry. Partial child mapping remains recorded for coordinated cleanup; unresolved
cleanup retains backing and transport. A mailbox acknowledgement alone never proves device
completion. Spontaneous child exit is not successful coordinated release.

During terminal L3 `Worker.close()`, children with unresolved uses stop native
admission and drain their native owner, including submissions that returned no
Python run handle. If carrier destruction remains unknown, the owned A5 runtime
stops background collectors and requires a successful bounded device synchronize
before releasing Host uses. This does not prove device health, unregister memory,
or reset the context. Failed synchronization keeps the uses, mappings and backing.
Live communication resources or external native references block this recovery.

A poisoned child keeps serving cleanup commands while refusing new tasks. Once
every Host registration has been explicitly unregistered and every CPU import
closed, recovered children finalize their native owner. Only that successful
owner teardown permits later Device Buffer frees to be acknowledged without
calling a dead context. Parent backing and cleanup records remain retained until
all finalization replies arrive; retries do not repeat a completed owner teardown.
Ordinary Buffer release does not initiate terminal owner recovery.

Unconfirmed CCU stop can refuse this recovery and leave `Worker.close()` with
an error. Host backing, cleanup records and child transports remain owned for
retry; this outcome does not promise that chip children have exited. Native
Workers own independent mappings of their local and MPI mailboxes through
scheduler and endpoint destruction, including Python interpreter shutdown.
Closing a Python mailbox view does not invalidate the native endpoint's mapping
or establish that any Device access to Host backing has stopped.

Use the [shared-table examples](../../examples/a5/tensormap_and_ringbuffer/host_ddr_map/README.md)
for BF16 gathers and MXFP8 payloads with Device scales and dependent consumers,
in serial and staged L3 execution. Historical URMA qualification does not qualify
this MAP path.
