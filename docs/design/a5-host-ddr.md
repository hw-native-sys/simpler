# Local Host DDR access

A direct L2 Worker can use `TensorTransfer.MAP` to let its device access owned
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

Direct L2 supports owned POSIX_SHM with both TMR and HBG on A3 and A5. Same-machine L3
supports A3 and A5 TMR through coordinated preparation. Remote access, Host compute
leaves, borrowed L2 contexts, L3 HBG and capture paths reject MAP.

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

An L3 Worker owns complete ordinary Host Buffers. Each local A3 or A5 TMR child imports
that same backing and registers its own Device mapping. Outer `TaskArgs` declare
`TensorTransfer.MAP` before the callback; every child must acknowledge preparation
before the callback can submit any work. Callback-created subviews retain the
canonical descriptor and select MAP per argument. A callback cannot first map
undeclared backing, including backing cached by an earlier call.

```text
L3 Host Buffer -> outer args (MAP) -> prepare all child L2 processes
                                      |                  |
                                 CPU import         CPU import
                                 Device0 map        Device1 map
                                      |                  |
                                 local MTE          local MTE
```

No peer relationship, rank communication, MR, URMA channel or communication
workspace is created by MAP. One child is supported, and multi-device validation
uses distinct devices and processes. Multiple backings and views coexist. There
is no table-sized HBM replica or payload copy in the control transport.

Each invocation declares its mapped backings, even when registrations are cached.
Cache hits issue no preparation messages or physical registrations. First-time
preparation requires previously accepted runs to drain; multiple accepted runs
can overlap when all requested mappings are already prepared. Child submission
resolves each MAP entry to that child's Device address; native NONE/H2D entries
keep their original behavior. The same writable-alias restrictions as direct L2
apply to each chip callable. Sharing backing does not order conflicting writers
on different devices. CPU updates require every Device user to have drained.

The parent retains accepted run uses, and children retain full dispatch identities.
Completion or a native proof of safe failed cleanup retires only that invocation.
Unknown completion retains the child use even if the parent receives a task error.
Stale activation also requires the native cleanup proof before retiring Host uses:
a terminal run can return from `abandon()` while cleanup remains unproven.
Release first closes admission and checks all children, then unregisters each local
mapping and closes its CPU import. Partial preparation rolls back every attempted
child. During Worker close, a responsive, quiesced child can retry a failed unmap.
If control cleanup is unconfirmed and cannot be retried through the child, Worker
close retains the backing while draining and shutting down the children. It can
release that backing after confirmed clean teardown of every child still holding
it. Unproven device completion or abnormal exit leaves the backing retained.

Use the [shared-table examples](../../examples/a5/tensormap_and_ringbuffer/host_ddr_map/README.md)
for BF16 gathers and MXFP8 payloads with Device scales and dependent consumers,
in serial and staged L3 execution. Historical URMA qualification does not qualify
this MAP path.
