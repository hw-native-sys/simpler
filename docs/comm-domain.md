# Communication Domains — Dynamic Allocation

A **communication domain** provides symmetric Device-addressable windows shared by a
subset of ranks, used for cross-rank reads/writes (collectives, SDMA, notify
protocols). Domains are allocated **dynamically from inside the orchestration
function** via `orch.allocate_domain(...)` — there is no init-time / static
declaration path.

For where the Orchestrator sits among the engine components see
[hierarchical-level-runtime.md](hierarchical-level-runtime.md); for the DAG
submission internals see [orchestrator.md](orchestrator.md).

---

## 1. API

```python
with orch.allocate_domain(
    name="default",                       # local label (peers need not agree)
    workers=[0, 1],                        # subset of the Worker's device_ids indices
    window_size=4096,                      # per-rank symmetric window, bytes
    buffers=[                              # named slices carved from the window
        CommBufferSpec(name="scratch", dtype="float32", count=1024, nbytes=4096),
    ],
) as handle:
    for chip_idx in handle.workers:
        domain = handle[chip_idx]          # -> ChipDomainContext
        ...
        orch.submit_next_level(chip_handle, args, cfg, worker=chip_idx)
```

Buffer sizes and `window_size` are validated on the orch thread **before** any
chip-side allocation. Each `CommBufferSpec.nbytes` is a physical allocation size
and must be a positive multiple of **64 bytes**; `window_size` must also be a
multiple of 64 covering their sum, so consecutive rank windows stay aligned.
Invalid requests raise `ValueError` without registering a backend allocation.
The caller (including pypto codegen) supplies the padding. Simpler places buffers
consecutively in declaration order and never rounds their sizes or offsets.
For example, a one-element `int32` signal uses `count=1, nbytes=64`; its tensor
view still has shape `(1,)`. This separates named buffers at cache-line boundaries;
concurrent signal slots inside one buffer still require a kernel-defined safe layout.

### Unified creation and reuse

L3 and L4 use `allocate_domain(members=...)`, `get_or_allocate_domain(...)`,
`get_domain(name)` and `release_domain(handle)`, returning the same
`CommDomainHandle`. L3 members are logical chip worker ids, such as `[1, 0]`;
L4 members are `(node_worker_id, chip_worker_id)` pairs. Order defines domain
rank. `member(rank).buffers[name]` provides the same logical Buffer and `.tensor(...)`
interface at both levels. `local_context(rank)` requires a mapping on the current node.
Legacy `handle[chip_id]` keeps chip-id semantics.
For L3 member metadata, `node_worker_id=-1` denotes the creator's own node.

Handle attributes and context/buffer mappings are read-only metadata by contract;
reassigning them is unsupported. Configure domains through allocation arguments
and end ownership through `release()`. Buffer contents remain writable.

```python
# In each run that uses the retained domain:
domain = orch.get_or_allocate_domain(
    name="tp", members=members, window_size=4096,
    buffers=[CommBufferSpec("payload", "float32", 1024, 4096)],
    retain_after_run=True,
)
payload = domain.member(0).buffers["payload"]
tensor = payload.tensor([1024], DataType.FLOAT32)
```

`allocate_domain` rejects a live name. `get_or_allocate_domain` returns the
same handle only if ordered members and target identities, requested window
size, every ordered buffer definition field, retention, and communication
topology match an immutable snapshot. A conflict reports the differing field
before device effects. A hit does not allocate, initialize or clear memory.
`get_domain(name)` performs creator-local lookup and raises `KeyError` when
missing. `allocate_global_domain` and `release_global_domain` are compatibility
wrappers; `workers=` remains the legacy L3 allocation spelling.

Domain creation and configuration reuse occur in an orchestration callback.
L3 creation and staging precede its first submission; a reuse hit performs no device control and remains valid afterwards.
A successful run preserves domains created with `retain_after_run=True`.
Later runs can submit cached Buffers/Tensors directly. Submission and ordinary copies enroll
the current run; release waits for it and its descendants, while a failed run closes its used domains.
Use `with` only when the scope should release the shared handle: context exit
releases a reused handle too. After release, callers must not submit new uses
of previously obtained Buffers; lookup, submission and local resolution reject revoked references.
Physical release still waits for accepted tasks and their children to complete.
After run completion, retained handles can be inspected, copied through a
matching Orchestrator and released serially outside a callback; successful release
reclaims the backend resources synchronously and repeated release is a no-op.
Only the creating Worker owns release. Lookup, creation, publication and release share
the callback serializer; concurrent matching requests share one transaction and conflicts fail.
`Worker.close()` attempts to free remaining owned domains after its existing lease/fence drain.
If device completion or reclamation is unconfirmed, the domain remains released
but not freed; its cleanup obligation stays tracked and is reported, while
independent worker-tree cleanup continues.
`Worker.live_domains` includes both backends.

`copy_to_domain` and `copy_from_domain` on Device-source domains use domain rank and the same named-buffer
bounds checks. They synchronously stage data without issuing an implicit run/DAG
fence. L3 staging precedes task submission and reuses ordinary VMM Buffer
copies and must stay inside the declared slices, including when `buffer=None`.
L4 retains the existing bounded whole-window copy path. `buffer_range(name)`
returns the slice's window offset and byte size. `requested_window_size` preserves the request;
`mapping_size` reports the backend's actual mapping extent. Complete buffer definitions remain available.

For Host-source domains, use `Buffer.shm.buf` and `domain.buffer_range(name)` for CPU access.
Domain copy helpers and `Worker.copy_to` / `Worker.copy_from` reject the mapped device address.

L4 Buffer/Tensor views reach their owning chip through local or remote L3s, preserving canonical
identity without exposing remote device addresses. Pass an immutable `CommDomainRef` (creator instance,
domain id, generation) across nodes; `get_domain(domain.ref)` borrows an established domain using the
same handle type, with `release()` and `with` rejected. Pass ref/rank explicitly for kernel context
and resolve it through `local_context(rank)`. Rebuilding a name creates new domain and Buffer identities.
Scalar-only ref/context arguments require `get_domain(name_or_ref)` in the owner's current callback
to enroll that run. A child lookup alone does not enroll the owner's run:

```python
# Owner callback, before forwarding ref/rank as control scalars:
domain = orch.get_domain("tp")
ref, rank = domain.ref, 0
# Child callback, after decoding those scalars:
context = child_orch.get_domain(ref).local_context(rank)
```

Failed or unknown release keeps cleanup debt visible and `freed=False` while transport/native teardown
continues; child exit alone does not prove device reclamation.

### `ChipDomainContext` (one per participating chip, via `handle[chip_idx]`)

| Field | Meaning |
| ----- | ------- |
| `name` | the domain's local label |
| `domain_rank` | this chip's dense rank within the subset (`workers.index(chip_idx)`) |
| `domain_size` | number of ranks in the subset |
| `device_ctx` | pointer to the device-side `CommContext` (pass as a kernel scalar) |
| `local_window_base` | base device address of this rank's window |
| `actual_window_size` | window size actually allocated |
| `buffers` | `{buffer_name: Buffer}` for each `CommBufferSpec` (device `VMM_WINDOW`, owned by this chip) |

For Device-source domains, `device_ctx` exposes the local and imported peer
window bases. For Host-source domains, peer access follows the platform rules
below. Use `buffers[name].tensor(shapes, dtype)` to pass a local slice to a task.

### A5 Host-source windows

L3 A5 `tensormap_and_ringbuffer` domains accept `memory_source=AddressSpace.HOST`.
The default remains `AddressSpace.DEVICE`. Each participating L2 owns independent
Host shared memory, maps it to its Device, and optionally registers an URMA
workspace with `workspace_engines=("urma",)`. Other Host workspace engines and
L4 Host domains are rejected before allocation.

The L3 creator accesses the rank's entire Host window through
`domain.member(rank).buffers[name].shm.buf` and selects a named slice with
`domain.buffer_range(name)`. Its `.tensor(...)` already addresses the named
slice through the owning Device's mapping. L2 owns allocation and unlink;
L3 borrows the `SharedMemory` mapping until domain release. CPU access is explicit and
requires ordering against Device tasks. Peer reads use the domain URMA workspace;
only the local `windowsIn`/`windowsOut` entry is populated. Host and Device domains
share the same retention, reuse checks, failure policy, and release fences.

See [A5 Host-source domains](a5-host-comm-domain.md) for initialization, kernel
addressing, and the retained Engram-table example.

### A2/A3 Host-source windows

Onboard L3 `tensormap_and_ringbuffer` domains use the same Host-source API.
As on A5, L2 creates and unlinks its rank's POSIX backing and L3 borrows the CPU
mapping. Worker teardown reclaims orphan windows in the startup namespace after
the relevant L2 children or enclosing process group have exited.
Local reads use MTE. For peer SDMA, initialize the Worker with `enable_sdma=True`
and request `workspace_engines=("sdma",)` to map the peer windows. The kernel
gets the source address from `CommContext` and the Worker workspace from
`get_dma_workspace(args, DMA_WORKSPACE_SDMA)`. CPU access and release use the
same borrowed views and fences as on A5.

Both MTE and SDMA require the installed stack to register POSIX mappings.
The CANN 9.0 [`aclrtHostRegister` documentation](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/runtimeapi/aclcppdevg_03_1804.html)
specifies runtime-allocated page-locked memory on Linux 5.10 and earlier.
A3 onboard MTE and SDMA passed with CANN 9.0.0 on HCE kernel `5.10.0-182.0.0.95.r1941_123.hce2.aarch64`.

Pass `initializer(domain)` to `allocate_domain` or `get_or_allocate_domain` to
fill the rank windows synchronously before submitting tasks in the same
orchestration. The callback runs once per new domain instance and is skipped
on retained reuse. It follows the shared Host initializer restrictions and
rollback rules described in [A5 Host-source domains](a5-host-comm-domain.md).

### Global CommDomain across local and remote L3 nodes

An L4 worker can build the same `CommContext` shape across any combination of
forked local L3 workers (`add_worker`) and TCP-connected L3 workers
(`add_remote_worker`) without `mpirun`:

```python
with orch.allocate_domain(
    name="tp",
    members=[(node0_worker_id, 0), (node1_worker_id, 0)],
    window_size=4096,
    buffers=[CommBufferSpec("payload", "uint8", 4096, 4096)],
) as domain:
    ...
```

Each member is `(l3_worker_id, local_l2_worker_id)`. The order defines dense
domain ranks. A remote node reads `comm_profile` and `global_device_ranks`
from `RemoteWorkerSpec`; a local L3 reads the same fields from its `Worker`
configuration. All participating nodes must use the same profile.

An MPI-launched group registered with `add_mpirun_worker_group` uses the same
member contract. Registration returns configuration ids and publishes the group
atomically; it does not establish readiness. The parent writes the manifest
before launch, and every MPI rank must publish READY before native attach.
The `MpiL3GroupSpec.hosts` order defines node ranks. Global CommDomain members
must include the complete returned group; their order still defines dense
domain ranks. A full MPI group exchanges descriptors rank-side over an MPI
collective, so the L4 import fanout is skipped; a partial group falls back to
the L4 descriptor broker above.

Global CommDomain capability follows the backend that the node actually
loads: a platform ending in `sim` supports the `sim` profile, and a real
`a2a3` platform supports `a3-fabric-v1`. Real A5 and any other
platform/profile combination currently reject allocation before `PREPARE`.
Each local or remote L3 repeats the same check during `COMM_INIT`, so an
unsupported backend never advertises a usable descriptor capability.

The control flow is:

1. L4 sends `COMM_INIT` with cluster, node, global-device, and domain-rank
   identities.
2. Each L3 asks its participating L2 children to create a local window and
   export a transport descriptor.
3. L4 validates and assembles one complete rank-ordered descriptor table.
4. L4 returns that table to every L3, which forwards it to each L2 for import.
5. L4 commits only after all imports succeed. Any earlier failure sends
   `ABORT` and releases every prepared local window.

`GlobalDomainCommand` is the transaction wire for all four phases:
`PREPARE_EXPORT`, `IMPORT`, `COMMIT`, and `ABORT`. Runtime data movement does
not resend it: `COPY_TO_DOMAIN`/`COPY_FROM_DOMAIN` use
`GlobalDomainCopyCommand`, and teardown uses `GlobalDomainReleaseCommand`.

### MPI session identity and availability

A Direct MPI controller associates one job-wide `session_id` with its static
rank/worker topology and an owner-local `session_epoch`, initially one. The
nonzero session identity occupies the full uint64 range and is shared by all
executor ranks. All executor endpoints must attach successfully before that
generation becomes available. Startup rollback and Worker close retire the
session before native teardown.

The endpoint registry stores an immutable `MpiSessionView` for each executor's
host and device endpoints. It identifies the owner path, job session, epoch,
rank and worker, together with snapshot availability. Records must match the
registry's owner nonce and directory version. `session_epoch` versions the MPI
session view; `registry_epoch` versions the owner's endpoint directory. A
directory change alone does not advance the session epoch.

Registry lookup, planning and region materialization observe a coherent native
hub snapshot containing closure, failure and unavailable routes. A newly
observed change advances the session epoch and invalidates the cached registry.
Repeated observations do not advance it again. An observation captured for a
different session object or epoch cannot overwrite the current state. Hub
failure or closure makes the session unavailable; a cancelled route is refused
when selected. Native errors can still terminate the entire MPI world.

Planning and materialization recheck selected session identities and directory
versions at their validation boundaries. Old snapshots must be refreshed after
a change; an old session epoch is refused even if the directory version matches.
A standalone resolver checks snapshot availability; live validation belongs to
the owning Worker. Nested MPI configuration describes endpoints but does not
establish their availability.

Direct MPI topology accepts an optional positive finite `health_timeout_s`.
When absent or `null`, it uses `session_timeout_s`. The periodic
`heartbeat_interval_s` must be shorter than the effective health timeout. The
controller checks each initialized, uncancelled route's health independently of
pending commands, including idle periods. Command deadlines keep their own
timeout; receiving heartbeats does not extend them. Changing the health timeout
does not change startup or close wait budgets.

Native hub failure, route cancellation and closure publish transport state and
wake waiting requests independently of Python epoch observation. Region checks
do not add a transport notification path or a heartbeat loop. Task and control
admission retain their lifecycle and backend checks.

After a valid delegated `ALLOCATED` reply, region materialization binds the
release identity before rechecking session availability. A registry or session
refusal aborts materialization through that release edge; refusal alone does
not latch a fatal delegated-session error. Invalid allocation replies are
fatal; incomplete release records an unreclaimable-resource error and prevents
further work.

Session invalidation does not establish that processes, device contexts or
remote resources have been released. The epoch is an owner-local validation
boundary; transport and domain commands carry their own identities. It does
not implement same-session reconnection or cross-generation wire validation.

### Attachment axis

`GlobalDomainMember` remains the rank table: one entry and one exported window
per device rank. A host is not inserted into that table and never consumes an
HCCL rank. The version-2 `GlobalDomainCommand` carries a flattened attachment
matrix alongside `members`. Each receiving L3 node contributes one row, and
each row has one entry for every member in dense rank order. Thus a domain with
three ranks on two nodes carries six attachment records; a receiver selects
the row whose `node_worker_id` is its own. MPI and non-MPI control paths use
the same complete command bytes.

The first wire form describes host consumers for the whole rank window (all
named buffer slices); it does not create a host mapping or lease during
`PREPARE`. Attachments are immutable for the domain lifetime and are serialized
only on `PREPARE_EXPORT`; `IMPORT` and `COMMIT` carry the descriptor table but
an empty attachment field, and the receiving L3 reuses the row saved by
`PREPARE_EXPORT`. `ABORT` carries only the transaction identity needed for
rollback. `address_space`, `role`, and the adapter fields reuse the endpoint
planner's existing vocabulary. The planner supplies an adapter when the
relation is currently supported; an unsupported relation remains an explicit
host-consumer record with no adapter instead of being mistaken for a usable
mapping. Host access still needs the separate host-window access primitive
before a consumer can dereference a device window.

The descriptor reports the backend's actual mapped size. A3 Fabric may align
the requested size to its VMM granularity; buffer carving and bounds checks
therefore use the returned mapping size. Handles and device pointers never
cross the public Python API. Remote orchestration code calls
`orch.get_global_domain(domain_id)` to obtain only its committed L3-local
contexts.

`copy_to_global_domain` and `copy_from_global_domain` provide bounded
control-plane staging and smoke checks. Normal communication still runs in
L2 kernels through the imported `CommContext`.

By default a live CommDomain at either level is swept after the current `Worker.run`
drains. Set `retain_after_run=True` when a communication kernel writes results
into the window and a second run must inspect them. The later run should
call `domain.release()` after copying the results; `Worker.close()` is the
final safety net.

Repository CI exercises the complete transaction, rollback, and mixed
local/remote paths with the `sim` backend. It also exercises the real
`a3-fabric-v1` backend: the two-machine `st-network1-onboard-a2a3` job runs
`global_tload_mixed_l3` and `compute_then_tload_mixed_l3`, whose default
profile is `a3-fabric-v1` on real A3 devices. Those two examples are the
in-repository harness for the Fabric path; a run that needs to know whether
Fabric was covered should read that job rather than infer it from the
simulation checks. The job now drives their `test_*.py` wrappers through
`network1-run-pytest` rather than calling `run_parent.sh` directly.

---

## 2. Lifetime model

The handle is a context manager. Its lifecycle has **two distinct states**:

- **`released`** — set the moment `release()` is called (or the `with` block
  exits). Further indexing (`handle[i]`) raises. This is the *user-visible*
  state: "do not hand this domain to any new `submit_*`."
- **`freed`** — the backend `comm_release_domain_windows` has actually run and
  the device memory is gone.

Inside an orchestration callback, `submit_next_level()` only *enqueues* DAG work;
`Worker.run()` does not wait for completion until the orch function returns.
If `release()` freed memory immediately on `with`-exit, a still-queued task that
captured the domain's `device_ctx` / `buffers` would read freed memory. So
**release inside the callback is deferred**: `release()` flips `released` and
queues the backend free; the real free runs after the run fence, when every
task that could reference the window has completed.

When a retained domain is released serially after its run has completed,
the backend free is synchronous: successful `release()` returns with `freed=True`,
including release at context-manager exit. Repeated release is a no-op.
Use `handle.released` to guard against accidental reuse; use `handle.freed`
to check physical teardown.

Cleanup is **failure-safe**: even if a chip task fails and the run wait
re-raises, `Worker.run` still attempts pending releases and sweeps any live
domains the orch fn forgot to release (LIFO). Failed or unconfirmed releases
remain tracked and prevent a successor run from using unresolved device state;
neither a failed wait nor child-process exit proves physical reclamation.

---

## 3. Lazy base communicator (created once, cached)

`Worker.init()` does **no** comm work. The first `allocate_domain(...)` lazily
fires `CTRL_COMM_INIT` to every chip in parallel, which runs the base HCCL
`comm_init` (RootInfo handshake + membership). This base communicator is
**cached** (`_comm_base_ready`), and the native chip runtime caches the
communicator. Each chip child's serve loop records the handle returned by a
successful `CTRL_COMM_INIT` and passes it explicitly to allocation/release
controls on its native chip binding. Failed initialization leaves the loop's
previous handle unchanged. Native
finalization retains ownership of communicator teardown.

Consequently, when a `Worker` runs multiple times, or `allocate_domain` is
called many times:

- the **base communicator is created once** and reused — it is *not* rebuilt
  per `run` or per domain;
- only the **per-domain windows** are allocated (and freed after the run fence) on each
  `allocate_domain` / `run`. Each allocation gets a fresh `allocation_id` so
  concurrent or sequential domains never collide on IPC handshake / barrier
  names.

---

## 4. Backends

Both backends present the same `ChipDomainContext`; they differ only in how the
symmetric window is realized:

| Aspect | Sim | HCCL (onboard) |
| ------ | --- | -------------- |
| Window memory | POSIX shm + `ftruncate`, mmap'd per rank | a2a3: Fabric V2 handle exchange (`ACL_MEM_SHARE_HANDLE_TYPE_FABRIC`), falling back to VMM + shareable-handle IPC where Fabric is unsupported. a5: VMM shareable handles only. Cross-card P2P via `aclrtDeviceEnablePeerAccess` on both |
| Subset barrier | shm-header atomic, `allocation_id`-scoped | file barriers, `allocation_id`-scoped |
| Window init | window zeroed before the subset barrier (`memset`) | window zeroed before the handle is announced (`aclrtMemset`) |
| Async-DMA workspace | opt-in per Worker (`enable_sdma`, provisions inert 16 KiB scratch without STARS streams) | a2a3: opt-in per Worker (`enable_sdma`); a5: domain-selected SDMA/URMA/RDMA; RDMA retains its optional build gate |

The window is zero-initialized on both backends so scratch/signal protocols see
a known starting state (matching the historical static-path contract).

The wipe happens before the window becomes reachable by any peer — before the
shareable handle is announced on HCCL, before the `ready_count` barrier on sim.
A peer that clears the subset barrier can return, launch its kernel and store a
barrier signal into this rank's window immediately; a wipe issued after that
point can erase a signal the owner has not yet waited on, and the owner then
waits on it forever. The rank skew that opens that window grows with host load,
so the resulting hang shows up only under a loaded box.

On a2a3, async-DMA resources are a Worker-level opt-in, not a
communication-domain property. Construct the Worker with `enable_sdma=True` and
the runtime provisions the SDMA workspace once at init, latches its address into
the resident `KernelArgs`, and injects it into every run's kernel
`GlobalContext` (`get_dma_workspace`). A Worker without `enable_sdma` creates no
SDMA streams and its kernels read a zero workspace address. The platform calls
`SdmaWorkspaceManager::GetWorkspaceSize`, allocates device memory with
`aclrtMalloc`, and binds it with `Init(workspaceAddr, workspaceSize)`. Ordinary
Worker finalize destroys the manager's streams before freeing the allocation;
fatal device teardown retains both with the abandoned provider.

**`enable_sdma` is a defect quarantine, not a capability switch, and it has an
exit condition.** Selecting an engine already belongs to the kernel, which reads
whichever addresses were injected; the runtime would otherwise provision every
engine the platform supports. SDMA is the exception because
`SdmaWorkspaceManager::Init(workspaceAddr, workspaceSize)` creates 48 CP-process
STARS streams while binding the external workspace. A usable SDMA workspace
therefore holds those streams, and a Worker holding them gets a
single device-reset attempt instead of three after an AICore fault. Defaulting it
on would therefore put every ordinary Worker in that population, which is exactly
the regression [the investigation](investigations/2026-07-a2a3-sdma-fault-teardown.md)
records. The flag disappears once CANN bounds the final CP-process stream release,
or once pto-isa supports binding a usable workspace without creating the streams.

Provisioning also warms the SDMA control path once, in the same call: a
vector-only AICore ELF (`sdma_warmup_kernel.o`, staged per arch under
`build/lib/<arch>/sdma_warmup/`) walks every channel so the first
`TPREFETCH_ASYNC` of a run does not pay the cold STARS submit-queue publication
(~92 µs per channel, ~4.4 ms of init for all 48). Either way the Worker comes
up; the two ways of not having the ELF differ only in what says so. An arch that
carries no `sdma_warmup_kernel.cpp` builds none by design and reports nothing —
a5 is that case, so its first `TPREFETCH_ASYNC` still pays the cold path. An arch
that carries the source but staged no object is a build or staging regression,
and is warned about twice: by the runtime builder and again at Worker init. A
warmup whose device launch or sync *fails* is not in this category at all — it
fails Worker init, because the card it faulted on must not reach the first run.

On a2a3, SDMA uses the Worker-owned workspace for both Device- and Host-source
domains. Because an SDMA-enabled Worker's 48 STARS
streams sit in the device fault/sync domain, a fault on that Worker slows its
teardown; keep SDMA workloads on their own Worker (and, in CI, their own task)
so ordinary workloads are unaffected — see
[docs/investigations/2026-07-a2a3-sdma-fault-teardown.md](investigations/2026-07-a2a3-sdma-fault-teardown.md)
and issue #1425. `enable_sdma` is honored by **both** a2a3 onboard runtimes: the
PTO-SDMA provider is compiled into every a2a3 onboard `host_runtime.so`, so the
gate is the platform, not the runtime. Both schedulers copy the resident
workspace addresses into every core's `GlobalContext`; kernels read them with
`get_dma_workspace(args, kind)`. Invalid kinds and unprovisioned engines return
`nullptr`. The HBG [prefetch demo](../examples/a2a3/host_build_graph/prefetch_async_demo/)
reuses the TMR prefetch kernel to exercise address injection and event completion.
A2/A3 simulation provisions inert scratch for address-injection tests without creating
SDMA streams; its prefetch is a no-op. A5 onboard and provider-disabled builds fail Worker
init fast when `enable_sdma` is set. A5 provisions
its communication-context SDMA workspace only when a domain explicitly requests it;
this is separate from
the Worker-level workspace mechanism controlled by `enable_sdma`.

On a5, SDMA and URMA providers are always compiled. RDMA remains optional through
`SIMPLER_ENABLE_PTO_RDMA_WORKSPACE=ON`, defaulting to OFF. Compilation alone creates
no streams or domain resources. `allocate_domain` and `get_or_allocate_domain`
create synchronous mapped-window communication by default, without initializing
any asynchronous provider. `workspace_engines=("sdma", "urma")` or `("rdma",)`
adds the requested asynchronous resources. An omitted parameter and an empty
sequence have identical meaning and can reuse the same retained domain.
The same parameter also accepts a mapping from engine names to immutable
`SdmaConfig`, `UrmaConfig` or `RdmaConfig` objects, available from `simpler`:

```python
from simpler import SdmaConfig, UrmaConfig

handle = orch.allocate_domain(
    name="exchange",
    members=[0, 1],
    window_size=window_bytes,
    workspace_engines={
        "sdma": SdmaConfig(),
        "urma": UrmaConfig(jetties_per_aiv=2),
    },
)
```

A name sequence uses default configurations, so `("urma",)` and
`{"urma": UrmaConfig()}` are equivalent. Mapping values must match their engine
names. Reusing a retained domain requires both the same engine set and the same
configuration; changing queue settings requires a new domain.
The native `comm_alloc_domain_windows` takes the corresponding bitmask and optional
`CommWorkspaceConfig`;
zero requests synchronous mappings only. Explicit asynchronous engine selection
is supported on L3 A5. On L3 A2/A3 Host domains, selecting SDMA enables peer
mappings for the Worker-owned workspace. Each participating device must report
`ACL_RT_HOST_MEM_MAP_SUPPORTED` for `ACL_RT_HAC_TYPE_SDMA` through
`aclrtHostMemMapCapabilities`; unsupported or failed queries abort allocation
through the existing domain rollback path. Host registration or AIV access alone
does not establish SDMA support. Simulation and L4
domains support synchronous mappings, with no domain-owned asynchronous engines.

Domain-owned workspace descriptors are stored in `CommContext.workspaces`,
indexed by `CommWorkspaceKind::{SDMA, URMA, RDMA}`. Kernels use
`comm_workspace<CommWorkspaceKind::URMA>(*ctx)` (or the corresponding engine),
whose template argument is checked at compile time. Host publication rejects
out-of-range enum values before indexing. An absent backend has zero address
and size.

Before submitting a URMA task, obtain its context through the checked host API:

```python
args.add_scalar(handle.device_context(rank, workspace_engines=("urma",)))
```

The required engines must have been selected when allocating that live domain.
For a mixed task use `workspace_engines=("sdma", "urma")`. A missing or different
engine, a released domain, an unknown worker or a null context raises before
submission. Scalar-only context use also retains the domain for the current run.
The engine set records successful native initialization; it is not inferred from
an environment variable. Native allocation rejects a null URMA workspace address
or zero size before publishing the context, and propagates initialization failures
as host errors. Callers must treat handle/context metadata as read-only and obtain
the checked context in every orchestration run rather than caching its raw address.

URMA requests require A5 TMR L3 domains. Device domains take an ordered subset of
the Worker's chips. For example, chip 1 has domain rank 0 in `[1, 2]` and domain
rank 1 in `[2, 1]`. The HCCL child communicator and the window and peer MR tables
use that same domain-rank order. Overlapping Device domains can stay live and be
used in successive runs, with each domain owning its communicator and URMA
workspace. Host-source URMA domains require ranks in dense prefix order.

Single-rank requests reach the provider, and initialization failures are
reported on the host. The currently pinned PTO URMA manager rejects single-rank
initialization. The successful
[single-device CANN Hcomm loopback probe](https://github.com/hw-native-sys/simpler/pull/2714#issuecomment-6491020508)
uses two endpoints and has not been integrated into that provider; the provider
restriction is not a hardware limitation. HBG URMA completion and L4 async
domains are unsupported.

URMA domains use `PER_PEER`: each resident AIV owns a separate queue group
for each remote peer. `UrmaConfig(jetties_per_aiv=1, sq_depth=0)` preserves one
jetty per AIV and the provider's default SQ depth. Increasing the jetty count
is opt-in per domain. `sq_depth` is optional: zero keeps the provider default;
an explicit value must be a positive power of two below 2**31 and supported by
the provider. CQ capacity is provider-selected and checked independently.
`sq_depth` is forwarded as a provider configuration request; it is not a promise
of an identical number of hardware SQ entries. The runtime uses the returned
SQ/CQ capacities for admission, and resource comparisons must use those actual
capacities rather than the requested value.

The provider supplies the device AIV count `A`. The pinned ISA permits at most
8 jetties per AIV and 256 queue rows per peer, so `A * jetties_per_aiv <= 256`.
Configuration is checked before window allocation. With `R` ranks and `J`
jetties per AIV, workspace tables have `R * A * J` rows, including the self-rank
rows; remote send queue groups account for `(R - 1) * A * J` rows. SQ/CQ payload
storage scales with their actual depths. Workspace descriptor bytes exclude
HCCL-owned queue storage and must not be reported as total communication memory.
Different AIVs never share a producer queue.

Each GET/PUT request is split into chunks of at most 256 MiB. Chunk `i` uses
jetty `i % J` within that sender's group; its absolute row is
`peer * rowsPerPeer + urma_sender_id * J + i % J`. Every request restarts at
jetty zero, including consecutive small requests. There is no persistent
round-robin cursor and no implicit ordering across different queues. Requests
that require ordering must establish completion dependencies.

Before posting, the runtime checks the entire request against each queue's
actual chunk count and the task's remaining completion capacity. Each chunk
uses one completion slot (64 slots per task, shared with other deferred work).
All participating queue targets must complete before dependent tasks start.
See the [multi-jetty acceptance tests](../tests/st/a5/tensormap_and_ringbuffer/urma_multi_jetty/README.md)
for real queue-counter checks, boundary cases and the timing comparison.

`get_async_ctx(args)` carries `urma_sender_id`, the scheduler's contiguous
resident AIV ordinal, including distinct identities for both lanes of a MIX
cluster. It remains the same across tasks and both dispatch buffers. It is
neither a hardware core ID nor a task-local block index. AIC and unbound
contexts have no valid sender and cannot submit URMA requests.

The URMA adapter constructs its session directly from this identity and the
chosen workspace; it does not use CCE block IDs. It checks the workspace
layout, peer and queue bounds before posting. The same adapter binds AIV
requests to Host or Device URMA workspaces without changing peer MR addressing.
ISA-specific session activation is isolated in the adapter's session helper
and must be checked when updating the ISA pin.

The A5 common request layer (`src/a5/runtime/urma/urma_request.h`) handles
session binding, queue preflight, chunking and conversion of ISA events into
workspace/queue-row/BB/CQE completion targets. Its runtime adapter supplies the
sender identity, checks and reserves completion capacity before the first post,
registers each actual target, finishes the reservation on success or partial
failure, and records the first error. TMR converts these targets into its
existing completion tokens and publishes the slab at the kernel request entry.
Reservation failure posts nothing; partial posting retains the accepted targets.
The common request layer neither consumes CQEs nor advances queue tails.

`capacity_available` is checked both before reservation and before target
registration. After `reserve` succeeds, it must count unused reserved slots as
available to this request, even if reservation reduced the runtime's free pool.
A successful reservation must make valid target registration reliable. If
registration can still fail, the runtime must retain resources used by posted
requests and enter its failure-recovery path. `finish` ends the reservation;
it does not establish network completion or permit freeing resources still
used by submitted work.

The queue's submitted and completed counters persist across task switches;
the AICPU completion poller is the only tail publisher. An AIV returning does
not complete its network requests: dependencies and buffer lifetimes remain
protected by the existing deferred-completion protocol. Requests that exceed
the available queue or completion-slab capacity fail before posting.

The legacy `device_ctx` field remains a raw address for synchronous consumers;
it does not declare or validate an asynchronous engine requirement. No compiled
capability query or `enable_urma` initialization switch is needed.

The context is simpler's own 1088-byte host/device ABI. Rank metadata starts
at offset zero; there is no legacy workspace pair or PTO-compatible prefix.
PTO session builders receive the chosen workspace address explicitly and do
not parse this context.

All runtime libraries, bindings, communication-context producers and compiled
kernels must come from the same simpler build and be updated together. This
follows the same-build contract used by `BufferDescriptor` and
`SimplerKernelInvocationHeader`. There is no per-context ABI version check,
layout negotiation or support for mixing old and new runtime artifacts, including
on global-domain paths. Rebuild/reinstall the complete artifact set after a
layout or C API change. The global-domain descriptor wire version is a separate
protocol field and does not establish build compatibility.

SDMA domains reference the shared CommHandle-owned manager; URMA/RDMA managers
belong to individual dynamic domains. URMA/RDMA deferred-completion kernels
remain specific to `tensormap_and_ringbuffer`; workspace storage does not add
completion backends to `host_build_graph`. Domain cleanup does not release another
domain's workspace or the shared SDMA resource.

Each A5 URMA domain owns an HCCL subcommunicator, including its registrations
and channels. The L2 host process keeps that owner from the first initialization
step through rollback or release. After the run fence, release checks every
active queue's BB and CQE completion counters, waits for peer quiescence, destroys
the domain communicator, and then releases its workspace and windows. A healthy
domain is reclaimed without waiting for the base communicator or Worker to close.
The additional communicator costs initialization time and HCCL resources per live
domain; synchronous and SDMA-only domains do not create it.

If any allocation participant fails before the domain is published, the parent
joins the allocation requests and locally aborts every reported A2/A3 or A5 owner,
including failed participants. Failed initialization keeps exported physical
memory alive until peers finish importing; its ownership flag does not publish
a usable domain. This rollback does not use a normal release collective.
The A2/A3 Fabric-to-IPC fallback can discard its temporary Fabric window only
before publishing the export; failed initialization after publication keeps the
window owned until abort.
Allocation IDs cannot be reused during an A5 communicator's lifetime. Used-ID
bookkeeping remains until communicator close; device windows and communication
resources do not. Python allocates fresh IDs automatically.

On A5, an unconfirmed drain, handshake or communication destruction is reported as an
error. The affected owner, its stream and device environment remain retained,
new work is refused, and later close does not blindly retry uncertain handles.
This is a terminal failure of that Worker, not successful resource reclamation.

The workspace coverage is split by platform and execution layer:

| Platform / layer | Supported behavior | Coverage |
| ---------------- | ------------------ | -------- |
| A5 host | SDMA and URMA plus optional RDMA in one build; RDMA uses a separate domain | RDMA on/off builds, domain isolation, invalid URMA descriptors, initialization failure rollback and finalization in `tests/ut/py/test_comm_workspaces.py` |
| A5 TMR | SDMA/URMA in one domain and one Worker | Ordered mixed DAG and same-kernel disjoint transfers in `urma_deferred_completion_demo` |
| A5 completion scheduler | A task waits for every registered backend condition | Controlled SDMA-first / URMA-first completion and failure in `test_urma_completion_scheduler.cpp` |
| A2/A3 | Worker-owned SDMA and optional peer mappings for L3 TMR Host domains | SDMA and prefetch examples, and `test_a3_host_source_domain.py` for Host MTE and peer SDMA |
| Simulation | Communication windows and zero backend descriptors | Existing domain/window tests; explicit nonempty engine selections are rejected |

The A5 mixed hardware examples use the full two-rank communicator. They do not
qualify arbitrary URMA subset communicators or multiple producers sharing a
peer queue. RDMA transfer examples exercise their own low-address domains;
joint compilation alone does not qualify same-domain RDMA/SDMA/URMA requests.

RDMA selection uses the HNS1825 backend. The RDMA integration only supports `orch.allocate_domain()`
dynamic domains; it does not support `comm_derive_context()`, sim platforms,
a2a3, or `host_build_graph`. RDMA kernels derive remote WQE addresses from the
RDMA workspace peer MR base plus a local-window offset, not from
`CommContext::windowsIn[peer]` — the symmetric window is a low-GM-segment
`aclrtMalloc` buffer the HNS1825 NIC can DMA to, and peers are reached over
RoCE by rkey + VA.

**RDMA-only domain (caller precondition).** An RDMA domain's
allocation path intentionally skips peer import and mapping: `windowsIn[peer]`
stays zero for every remote rank, and there is no locally addressable mapped VA
for a peer's window. A peer MR address (`rkey` + VA) is not a substitute — only
the NIC can dereference it. Consequently, a kernel that computes
`windowsIn[peer] + offset` and issues a synchronous `TPUT`/`TGET` against that
address cannot run on an RDMA domain unchanged; it must use the asynchronous
RDMA path (`TGET_ASYNC`/`TPUT_ASYNC<DmaEngine::RDMA>`), which carries the peer
MR base and rkey instead of a local VA. Switching a workload from SDMA/URMA to
RDMA is therefore a kernel contract change, not a configuration flip.

**Single-producer submission (caller precondition).** Multiple AICores (AIVs)
submitting through the same workspace and the same peer/QP read and write the
same SQ head and WQE slot with no multi-producer reservation — the posting path
reads the SQ head, fills the WQE at that slot, and advances the head. Concurrent
submissions to one peer/QP can therefore overwrite or lose a request. Callers
must serialize submissions to a given peer/QP; the `rdma_deferred_completion_demo`
relies on its inter-kernel marker dependencies to order the submissions it makes
to each peer. This is a documented constraint, not a supported multi-producer
contract.

**Completion consumption.** Deferred completions are consumed by the AICPU
scheduler poller, which advances the shared CQ consumer index and mirrors it
into the CQ tail and the SQ tail record (matching the pinned PTO
`RingCqDoorbell`). The AICore post path reads that same SQ tail to decide when
to drain the CQ before the SQ fills (`head - tail >= depth - threshold`).
Because both the poller and the AICore-side drain advance the same monotonic
index, a submission that triggers an AICore-side drain consumes only the CQEs
still outstanding at that point; the AICPU poller never re-consumes them.

---

## 5. Staging host data into a window

To preload host data (rather than have a kernel write the window), use
`orch.copy_to`:

```python
orch.copy_to(handle[chip_idx].buffers["input"], src_buffer)
```

`copy_to(dst_handle, src)` is **synchronous** (control-mailbox round-trip +
synchronous `rtMemcpy` H2D): when it returns, the bytes are in that rank's
window. `dst` is the window's `VMM_WINDOW` Buffer; `src` is a host `Buffer`. Both ends cross
the fork as buffer descriptors, and the chip child resolves each one through the same
map-once-by-identity path a task argument takes, then DMAs straight out of the
backing — no intermediate copy, and no host address ever leaves this process:

```python
src_buffer = worker.create_buffer(nbytes)
torch.frombuffer(src_buffer.shm.buf, dtype=torch.float32, count=n).copy_(src_tensor)
orch.copy_to(handle[rank].buffers["input_window"], src_buffer)
```

**Cross-rank ordering:** when a kernel reads a *peer's* staged window, stage
**all** ranks' windows before submitting any kernel — `copy_to` is synchronous
but `submit_next_level` is async, so interleaving stage/submit per rank lets one
rank's producer run before another rank has finished staging:

```python
with orch.allocate_domain(...) as handle:
    for chip_idx in handle.workers:                       # stage all first
        orch.copy_to(handle[chip_idx].buffers["input"], tensor)
    for chip_idx in handle.workers:                       # then submit
        orch.submit_next_level(chip_handle, args, cfg, worker=chip_idx)
```

---

## 6. Host tensor visibility for `worker.run`

A host tensor named in a task arg is dereferenced by the local L3 child, so its
backing must reach that child. Under the
Buffer ABI an arg is a `Tensor` carrying a self-describing descriptor; the
child materializes it lazily on first receipt (map-once, keyed by canonical
identity) — there is no eager broadcast and no host-pointer rewrite. Two sources
are legal:

| Source | How | Why it works |
| ------ | --- | ------------ |
| **fork-inherited (fork children only)** | `tensor.share_memory_()` **before `Worker.init()`**, named with `worker.make_tensor_arg(t, shapes, dtype)` (FORK_SHM) | the child inherits the MAP_SHARED page at the fork; it resolves to that same VA |
| **worker-allocated shared memory** | `worker.create_buffer(nbytes)` after the children exist, named with `handle.tensor(shapes, dtype)` (POSIX_SHM) | the child maps the shm by identity on first receipt, **zero-copy** |

After direct A5 L2 device initialization has been attempted, a top-level A5/TMR
L3 Worker starts its chip children in new interpreters. Use `create_buffer` for
inputs on this path and for host tensors created after `Worker.init()`:

```python
worker = Worker(level=3, ...); worker.register(chip); worker.init()   # starts the chip workers

buf_h = worker.create_buffer(tokens * hidden_size * 4)   # POSIX shm
buf_o = worker.create_buffer(batch * vocab * 4)
try:
    hidden = torch.frombuffer(buf_h.shm.buf, dtype=torch.float32, count=tokens * hidden_size)
    out    = torch.frombuffer(buf_o.shm.buf, dtype=torch.float32, count=batch * vocab)
    for step in batches:
        fill(hidden)
        # name each buffer as a ref in the task args; the child maps it once
        worker.run(orch, ...)                   # no per-run copy — child reads/writes the same pages
        use(out)
    del hidden, out                             # drop views before close
finally:
    buf_h.close()
    buf_o.close()
```

**Create once, reuse many runs.** The first ref over a `create_buffer` handle
maps its shm into the child (map-once, cached by identity); later runs reuse the
mapping, so there is **no per-run copy**. Build the tensor over `handle.shm.buf`
(buffer protocol — `torch.frombuffer` / `np.frombuffer`) once and reuse it.
simpler stays framework-free — torch/numpy appear only on the caller's side.

### Contract / limits

- **Zero-copy is a live shared medium.** The buffer's pages are shared with the
  child; during a `run` the child is reading/writing them, so the parent must
  not read or write the buffer until `run` returns (same contract as a
  fork-inherited `.share_memory_()` tensor). In-run cross-task ordering (a
  producer task's output read by a consumer task) is enforced by the runtime's
  dependency inference, keyed on the ref's canonical identity — no host-side copy.
- **`orch.copy_to` is the low-level device path.** It moves host bytes into a
  domain window (`copy_to(dst_handle, src)`, §5); both ends are `Buffer`s, so the
  child resolves them exactly as it resolves a task argument.
- **Fork-inherited anonymous memory is copy-on-write, hence stale.** Even a
  tensor the child legitimately inherited is only useful as a *live* input if it
  is MAP_SHARED: anonymous (non-`share_memory_`) pages are COW, so writes the
  parent makes *after* fork do not reach the child. A live input must be
  file-backed (`.share_memory_()` before `init()`) or a `create_buffer` one.

---

## 7. Examples

- [`tests/st/worker/collectives/allreduce/`](../tests/st/worker/collectives/allreduce/) — single domain, PTO-ISA remote
  reads over the window (allreduce scene tests with multiple algorithm modes).
- `examples/workers/l3/domain_rank_map/` — two domains, domain-local ranks,
  missing-domain `KeyError`, per-domain allreduce.
- `examples/workers/l3/dual_domain_overlap/` — overlapping domains where one
  worker participates in both.
- `examples/a2a3/tensormap_and_ringbuffer/sdma_async_completion_demo/` — host
  staging via `copy_to` + cross-rank `SdmaTget`; the SDMA workspace is
  provisioned by constructing the `Worker` with `enable_sdma=True`.
