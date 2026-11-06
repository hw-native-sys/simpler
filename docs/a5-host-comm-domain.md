# A5 Host-source Device communication domains

An A5 L3 Worker can create a Device communication domain whose per-rank windows
live in Host DDR. Each L2 owns a separate Host allocation. The L3 CPU can fill
those allocations directly, each Device can read its own mapped window with MTE,
and URMA can asynchronously fetch another rank's window into local Device memory.
The Host does not become an additional rank, and the table needs no matching
HBM allocation.

This implementation supports one machine, A5, and the
`tensormap_and_ringbuffer` runtime. Cross-machine Host windows, A2/A3 support,
and automatic table partitioning are outside this interface's current support.

## Create and retain a table

Use the existing domain API inside the L3 orchestration callback. Select the
physical window source with `memory_source`; its default is `AddressSpace.DEVICE`.
The Worker supplies the platform, so domain creation does not take a second
platform argument.

```python
from simpler.task_interface import AddressSpace, CommBufferSpec, DataType

state = {}

def prepare(orch, args, config):
    state["table"] = orch.get_or_allocate_domain(
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
    )

worker.run(prepare)
domain = state["table"]
offset, nbytes = domain.buffer_range("table")
for rank in range(2):
    buf = domain.member(rank).buffers["table"]
    with buf.shm.buf[offset : offset + nbytes] as memory:
        memory[:] = table_shards[rank].tobytes()
```

As with Device-source domains, sizes must satisfy the domain's alignment
requirements. `CommBufferSpec` slices share the rank's physical allocation.
`buf.shm` is a borrowed Python `SharedMemory` mapping. Its `.buf` exposes the
entire rank window, including page-alignment padding; use `buffer_range(name)`
to select a named slice. All named buffers on a rank share that mapping.
`buffer.tensor(...)` already names the selected slice through
its owning Device's mapping; its task argument uses the ordinary domain-buffer
path, without `map_view` or a whole-table H2D copy.

Host table partitioning is a business decision. The two rank windows can hold
different shards, replicas, or independent tables. The library associates each
window with a rank and does not route table indices automatically.

## Access from a kernel

Pass `domain.device_context(rank, workspace_engines=("urma",))` as a scalar
argument alongside the local domain Tensor. Each Device receives its own
CommContext.

```text
L2-owned Host DDR
  +-- L3 CPU mapping -> Buffer.shm.buf
  +-- local Device mapping -> local Tensor / windowsIn[rankId]
  +-- URMA registration -> per-domain workspace -> peer MR address
```

For local MTE access, use the Tensor's Device address or
`ctx.windowsIn[ctx.rankId] + offset`. A Host-source context does not advertise
ordinary peer MTE mappings. URMA peer MR addresses must be obtained through the
URMA workspace, not read from `windowsIn[peer]`.

The same byte offset identifies the named slice on each rank:

```cpp
auto *workspace = reinterpret_cast<__gm__ uint8_t *>(
    comm_workspace<CommWorkspaceKind::URMA>(*ctx).addr);
auto offset = local_tensor_address - ctx->windowsIn[ctx->rankId];
auto *peer_source = pto2::urma_backend::peer_mr_ptr<float>(workspace, peer, offset);
send_request_entry(async, UrmaTget(local_destination, peer_source_tensor, workspace, peer));
```

The destination and source in `UrmaTget` are PTO GlobalTensors. The runnable
example constructs them for each requested row. Use a non-self peer for URMA;
local reads use MTE. Kernel return alone does not prove URMA completion: the
runtime's deferred-completion path holds the producer's output dependencies
until the transfer completes.

URMA resources belong to this communication domain. They do not replace the
workspaces of Device-source domains, and they do not use a separate global
Host workspace slot. No Host registration is performed during Worker init.

## Reuse and release

Host-source and Device-source domains have the same lifetime:

```text
preparation run: create domain -> complete mappings and registration -> retain
CPU: fill rank windows
inference runs: get retained domain -> submit reads -> wait for completion
CPU: update table only after all readers complete
shutdown: release domain -> retire communication and Device mappings -> release backing
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

Host windows use the owning Worker's `sp-<token>-h-...` namespace. Worker
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
from L2 to L3.

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

[`host_comm_domain`](../examples/a5/tensormap_and_ringbuffer/host_comm_domain/README.md)
initializes distinct rank tables from L3 and gathers boundary and repeated rows
using local MTE and non-self URMA. Local MTE writes into a Host output slice,
which L3 reads directly after completion. A dependent consumer checks the URMA
deferred-completion contract. It reuses the table across runs, updates the Host bytes
between completed runs, and keeps a Device-source URMA domain active after the
Host-source domain is released.
The one-AIV and four-AIV variants both run in serial and staged modes. In the
four-AIV variant, resident blocks write disjoint rows while fetching from the
same peer through the domain's URMA workspace; the consumer waits for every
block's deferred completion.
