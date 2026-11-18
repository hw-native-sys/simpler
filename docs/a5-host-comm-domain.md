# A5 Host-source Device communication domains

An A5 L3 Worker can create a Device communication domain whose per-rank windows
live in Host DDR. Each L2 owns a separate Host allocation. The L3 CPU can fill
those allocations directly. With `workspace_engines=()`, each requesting Device maps the local
and peer Host windows for synchronous MTE reads. Selecting
`workspace_engines=("urma",)` registers the table for asynchronous URMA reads
and leaves every `windowsIn` slot zero, including the local rank. Host UBMEM
access (including local Host DDR) and URMA cannot run concurrently, even on
different addresses.
The Host does not become an additional rank, and the table needs no matching
HBM allocation.

This A5 implementation supports one machine. Synchronous MTE works with
`tensormap_and_ringbuffer`, `host_build_graph`, and `percounter`; URMA requires
`tensormap_and_ringbuffer`. Cross-machine Host windows and automatic
table partitioning are outside its current support. A2/A3 uses the same domain
API with MTE and SDMA; see [A2/A3 Host-source windows](comm-domain.md#a2a3-host-source-windows).

| Host-source path | TMR | HBG | percounter | Workspace |
| ---------------- | --- | --- | ---------- | --------- |
| Synchronous peer MTE, A5 PR and DT | Supported | Supported | Supported | `()` |
| Asynchronous peer URMA | DT qualified; PR unresolved | Unsupported | Unsupported | `("urma",)` |

MTE mappings use `aclrtHostRegisterV2(..., ACL_HOST_REG_MAPPED)` followed by
`aclrtHostGetDevicePointer` in the requesting rank's process. Each address belongs
to the requesting Device; a source rank's Device address cannot be reused by a
peer. With `workspace_engines=()`, no URMA or SDMA workspace or capability query
is involved in this path.
Peer Host writes and `MAPPED | PINNED` registration are outside this contract.

Host-source URMA has passed on Ascend950DT_9581, including CANN 9.3.0 validation.
On the A5 PR PCIe CI machine, Host mapping and segment registration succeeded
but Host-source URMA reported an ACK timeout; this remains an unresolved
[hardware-specific failure](https://github.com/hw-native-sys/simpler/pull/2913#issuecomment-6681762026).
Before entering the URMA collective, all ranks must prepare their local windows.
A positive registration result or UDMA capability does not establish peer
reachability. If CANN reports that UDMA cannot access registered Host memory,
the Host-source URMA allocation fails before entering the collective.

## Create and retain a table

Use the existing domain API inside the L3 orchestration callback. Select the
physical window source with `memory_source`; its default is `AddressSpace.DEVICE`.
The Worker supplies the platform, so domain creation does not take a second
platform argument. An optional `initializer(domain)` fills the Host windows
once during creation. The same orchestration then submits the model on both
its first and subsequent invocations; no separate preparation run is required.
For example, load raw rank-local float32 shard files directly into their windows:

```python
from simpler.task_interface import AddressSpace, CommBufferSpec, DataType

state = {}

def initialize_table(domain):
    offset, nbytes = domain.buffer_range("table")
    for rank in range(2):
        buf = domain.member(rank).buffers["table"]
        with buf.shm.buf[offset : offset + nbytes] as window:
            with open(shard_paths[rank], "rb", buffering=0) as shard:
                position = 0
                while position < nbytes:
                    end = min(position + 8 * 1024 * 1024, nbytes)
                    with window[position:end] as chunk:
                        count = shard.readinto(chunk)
                    if not count:
                        raise ValueError("Engram shard is shorter than its table")
                    position += count

def infer(orch, args, config):
    domain = orch.get_or_allocate_domain(
        name="engram",
        members=[0, 1],
        memory_source=AddressSpace.HOST,
        workspace_engines=("urma",),
        window_size=256 + rows_per_rank * row_bytes,
        buffers=[
            CommBufferSpec("reserved", "uint8", 256, 256),
            CommBufferSpec("table", "float32", rows_per_rank * width,
                           rows_per_rank * row_bytes),
        ],
        retain_after_run=True,
        initializer=initialize_table,
    )
    state["table"] = domain
    submit_model(orch, domain, args, config)

for args in batches:
    worker.run(infer, args, config)
domain = state["table"]
```

`submit_model` is the application's task-submission code. The initializer runs
synchronously on the L3 CPU after all rank windows are mapped and registered,
before the allocation API returns. It receives the whole domain and must finish
all rank writes before returning. Chunked `readinto` avoids a second whole-table
allocation. Release temporary memoryviews and NumPy arrays on both success and
failure; do not store them in application state.

Both `allocate_domain` and `get_or_allocate_domain` accept `initializer`. On A5 it
requires an onboard L3 Host-source domain. The callback only initializes CPU
storage. It must not submit tasks, issue Worker/control operations, recursively
create or look up domains, or release the domain. Allocation and initialization
must precede the run's first task submission.

On an exact configuration reuse hit, `get_or_allocate_domain` skips the callback,
including when the caller supplies a different callback or the existing domain
was created without one. Callback identity is not part of the configuration;
the domain does not retain the callback or its checkpoint closure. Releasing and
recreating the domain invokes the initializer again. To change an existing table,
wait for all users and update its CPU bytes explicitly, or release and recreate it.

An initializer exception fails creation; a partially initialized domain cannot
be reused or submitted. Cleanup retains the existing CPU-export and device-fence
guards. If a leaked CPU view prevents release, the Worker keeps cleanup ownership;
drop that view before retrying cleanup. Initialization is once per newly created
domain instance, not once per process or model name.

As with Device-source domains, sizes must satisfy the domain's alignment
requirements. `CommBufferSpec` slices share the rank's physical allocation.
`buf.shm` is a borrowed Python `SharedMemory` mapping. Its `.buf` exposes the
entire rank window, including page-alignment padding; use `buffer_range(name)`
to select a named slice. All named buffers on a rank share that mapping.
`buffer.tensor(...)` already names the selected slice through
its owning Device's mapping; its task argument uses the ordinary domain-buffer
path, without a per-call MAP request or a whole-table H2D copy.

Host table partitioning is a business decision. The two rank windows can hold
different shards, replicas, or independent tables. The library associates each
window with a rank and does not route table indices automatically.

## Access from a kernel

Pass `domain.device_context(worker_id)` as a scalar argument alongside the local
domain Tensor. Obtain `worker_id` from `domain.member(rank).local_worker_id`;
the domain rank and Worker ID can differ for reordered members. Each Device
receives its own CommContext. For an URMA domain,
request its workspace with `workspace_engines=("urma",)` on both allocation
and context lookup.

```text
L2-owned Host DDR
  +-- L3 CPU mapping -> Buffer.shm.buf
  +-- MTE domain -> local Device mapping -> windowsIn[rankId]
  +-- MTE domain -> peer process mapping -> requesting Device registration -> peer MTE read
  +-- URMA domain -> registration -> per-domain workspace -> peer MR address
```

For a synchronous MTE domain, use `ctx.windowsIn[peer] + offset` as the source
address for MTE2, and MTE3 to write ordinary local HBM. The `peer` index follows
the domain's member order, including reordered members and subsets. The local
Tensor address minus `ctx.windowsIn[ctx.rankId]` gives the named buffer's offset.
The kernel must complete its loads and stores before returning. Normal task
dependencies then allow consumers to read the result; no deferred completion is
used and the AIV remains occupied during the transfer. Only the local
`windowsOut` entry is published; peer Host writeback is not supported.

For an URMA domain, pass the byte offset returned by
`domain.buffer_range("table")[0]` from L3 to the kernel as a scalar. Resolve
the source through the peer MR in the URMA workspace.

```cpp
auto *workspace = reinterpret_cast<__gm__ uint8_t *>(
    comm_workspace<CommWorkspaceKind::URMA>(*ctx).addr);
auto *peer_source = pto2::urma_backend::peer_mr_ptr<float>(workspace, peer, offset);
send_request_entry(async, UrmaTget(local_destination, peer_source_tensor, workspace, peer));
```

The destination and source in `UrmaTget` are PTO GlobalTensors. The runnable
example constructs them for each requested row on a non-self peer. Kernel return alone does not prove URMA completion: the
runtime's deferred-completion path holds the producer's output dependencies
until the transfer completes.

The selected MTE mappings or URMA workspace belong to this communication domain. They do not replace the
workspaces of Device-source domains, and they do not use a separate global
Host workspace slot. No Host registration is performed during Worker init.

## Reuse and release

Host-source and Device-source domains have the same lifetime:

```text
first inference run: create domain -> map/register -> initialize -> submit reads -> wait
later inference runs: get retained domain -> submit reads -> wait for completion
CPU: update table only after all readers complete
shutdown: release domain -> wait for users -> unregister/unmap peers -> release own backing
```

`get_or_allocate_domain` checks the requested configuration and returns the
existing handle without clearing its data. `get_domain` looks up an existing
domain and registers the current run's use. Cached domain Tensors also participate
in the existing use tracking. A successful run retains domains marked
`retain_after_run=True`; a failed run closes its used domains under the existing
failure policy.

Call `domain.release()` after the final use, or let `Worker.close()` release it.
Do not wrap a retained domain in `with` unless leaving that scope should release
it. CPU memoryviews and arrays must not be used to race Device readers or writers;
`.shm.buf` is an addressable CPU view, not a synchronization operation. L2 owns
creation and unlink; L3 only borrows the mapping and does not register Python
resource-tracker ownership. Use `domain.release()`, not `buf.close()` or
`buf.shm.unlink()`.

Host windows use the owning Worker's `sp-<token>-h-<pgid>-...` namespace. Worker
teardown also unlinks orphan windows left by an abruptly terminated L2,
including allocations that never returned a reply. Other Worker trees are
outside that namespace and are not swept.

Each `.shm.buf` access returns a managed, zero-copy memoryview. NumPy arrays
created with either `np.ndarray(buffer=...)` or `np.frombuffer(...)`, including
their slices, keep the borrow alive even after the original memoryview is
released. Drop those arrays and release memoryviews before releasing the domain
or closing its Worker. Live exports cause `BufferError` before the domain is
marked released or any of its native windows are freed. Dropping the last view
allows the operation to be retried; it does not transfer allocation ownership
from L2 to L3. Each requester owns its imported registration and mapping but
never unlinks a peer's backing. Failed initialization retains partially acquired
resources until all rank initialization requests have joined and rollback can
close them. A successful release waits for device use before unregistering.

Accepted release prevents new CPU exports while it waits for Device use to
finish. After physical release, `.shm.buf` is `None`. Automatic run cleanup
uses the same export check. A cleanup refusal reports the run error and keeps
the domain owned by the Worker; drop the views and close the Worker to clean up.

```python
import numpy as np

buf = domain.member(0).buffers["table"]
array = np.ndarray((rows_per_rank, width), dtype=np.float32,
                   buffer=buf.shm.buf, offset=offset)
rows = array[1:3]
# domain.release() would raise BufferError while array or rows is alive.
del array, rows
domain.release()
```

## Runnable example

[`host_mte_domain`](../examples/a5/host_mte_domain/README.md) reads non-self Host
windows into ordinary local HBM with a dependent consumer, reversed member order,
independent domains, retained reuse and CPU updates. Its runtime selection covers
TMR, HBG and percounter. Prepare its compiled artifacts before taking a device
allocation, then use its prepared execution mode on both PR and DT.

[`host_comm_domain`](../examples/a5/tensormap_and_ringbuffer/host_comm_domain/README.md)
initializes distinct rank tables from L3 and gathers boundary and repeated rows
from the peer through URMA. The same L3 orchestration creates and initializes
the Host domain on its first invocation and reuses it on later invocations, with
an assertion that the initializer ran exactly once. URMA writes the gathered rows into HBM. A dependent consumer checks the URMA
deferred-completion contract. It reuses the table across runs, updates the Host bytes
between completed runs, and keeps a Device-source URMA domain active after the
Host-source domain is released.
The one-AIV and four-AIV variants both run in serial and staged modes. In the
four-AIV variant, resident blocks write disjoint rows while fetching from the
same peer through the domain's URMA workspace; the consumer waits for every
block's deferred completion.
