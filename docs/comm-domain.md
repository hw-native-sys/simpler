# Communication Domains — Dynamic Allocation

A **communication domain** is a symmetric device-memory window shared by a
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

`window_size` is validated on the orch thread **before** any chip-side
allocation: if `sum(b.nbytes) > window_size`, `allocate_domain` raises
`ValueError` immediately and no backend allocation is registered.

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

Kernels read peer windows through `device_ctx` (which holds every rank's
window base, local + imported peer); `buffers[name]` is the local slice —
name it in a task arg with `buffers[name].tensor(shapes, dtype)`.

### Global CommDomain across local and remote L3 nodes

An L4 worker can build the same `CommContext` shape across any combination of
forked local L3 workers (`add_worker`) and TCP-connected L3 workers
(`add_remote_worker`) without `mpirun`:

```python
with orch.allocate_global_domain(
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
member contract. Rank 0 writes the group manifest before launch, every MPI rank
must publish READY before the L4 parent exposes the returned worker ids, and
the `MpiL3GroupSpec.hosts` order defines node ranks. Global CommDomain members
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

By default a live Global CommDomain is swept after the current `Worker.run`
drains. Set `retain_after_run=True` when a communication kernel writes results
into the window and a second L4 run must inspect them. The later run should
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
  the device memory is gone. This happens **after** the owning run's completion
  fence, never inside the `with` block.

This split exists because `submit_next_level()` only *enqueues* DAG work;
`Worker.run()` does not wait for completion until the orch function returns.
If `release()` freed memory immediately on `with`-exit, a still-queued task that
captured the domain's `device_ctx` / `buffers` would read freed memory. So
**release is deferred**: `release()` flips `released` and queues the backend
free; the real free runs after the run fence, when every task that could
reference the window has completed.

Mental model: like `with open(f) as fh: ...` — the user-visible close is
lexical (end of block), the physical teardown is managed for you. Use
`handle.released` to guard against accidental reuse; use `handle.freed` only if
you must assert physical teardown.

Cleanup is **failure-safe**: even if a chip task fails and the run wait
re-raises, `Worker.run` still executes pending releases and sweeps any live
domains the orch fn forgot to release (LIFO), so a failed run cannot strand
backend allocations into the next run.

---

## 3. Lazy base communicator (created once, cached)

`Worker.init()` does **no** comm work. The first `allocate_domain(...)` lazily
fires `CTRL_COMM_INIT` to every chip in parallel, which runs the base HCCL
`comm_init` (RootInfo handshake + membership). This base communicator is
**cached** (`_comm_base_ready`), and `ChipWorker.comm_init` itself caches the
handle.

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
| Async-DMA workspace | opt-in per Worker (`enable_sdma`, provisions inert 16 KiB scratch without STARS streams) | a2a3: opt-in per Worker (`enable_sdma`); a5: SDMA by default, URMA as an opt-in alternative |

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
SDMA streams and its kernels read a zero workspace address. The workspace is
released at Worker finalize by ordinary stream/manager teardown.

**`enable_sdma` is a defect quarantine, not a capability switch, and it has an
exit condition.** Selecting an engine already belongs to the kernel, which reads
whichever addresses were injected; the runtime would otherwise provision every
engine the platform supports. SDMA is the exception because
`SdmaWorkspaceManager::Init()` is indivisible — the 16 KB workspace *is* the
descriptor table for 48 CP-process STARS streams, so there is no way to have the
address without holding the streams — and a Worker holding those streams gets a
single device-reset attempt instead of three after an AICore fault. Defaulting it
on would therefore put every ordinary Worker in that population, which is exactly
the regression [the investigation](investigations/2026-07-a2a3-sdma-fault-teardown.md)
records. The flag disappears once CANN bounds the final CP-process stream release,
or once pto-isa offers an `Init()` that separates the workspace from the streams.

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

Communication-domain allocation does not create SDMA streams or carry the
workspace through `CommContext`. Because an SDMA-enabled Worker's 48 STARS
streams sit in the device fault/sync domain, a fault on that Worker slows its
teardown; keep SDMA workloads on their own Worker (and, in CI, their own task)
so ordinary workloads are unaffected — see
[docs/investigations/2026-07-a2a3-sdma-fault-teardown.md](investigations/2026-07-a2a3-sdma-fault-teardown.md)
and issue #1425. `enable_sdma` is honored by **both** a2a3 onboard runtimes: the
PTO-SDMA provider is compiled into every a2a3 onboard `host_runtime.so`, so the
gate is the platform, not the runtime. Only `tensormap_and_ringbuffer` is
exercised with it, so host-build-graph's path from a provisioned address to a
kernel's `get_dma_workspace` is unverified rather than closed. Simulation, a5,
and provider-disabled builds fail Worker init fast when it is set. A5 provisions
its communication-context SDMA workspace by default; this is separate from
the Worker-level workspace mechanism controlled by `enable_sdma`.

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

A host tensor named in a task arg is ultimately dereferenced from a forked local
L3 child, not the parent, so its backing must reach that child. Under the
Buffer ABI an arg is a `Tensor` carrying a self-describing descriptor; the
child materializes it lazily on first receipt (map-once, keyed by canonical
identity) — there is no eager broadcast and no host-pointer rewrite. Two sources
are legal:

| Source | How | Why it works |
| ------ | --- | ------------ |
| **fork-inherited** | `tensor.share_memory_()` **before `Worker.init()`**, named with `worker.make_tensor_arg(t, shapes, dtype)` (FORK_SHM) | the child inherits the MAP_SHARED page at the fork; it resolves to that same VA |
| **worker-allocated post-fork** | `worker.create_buffer(nbytes)` after the children exist, named with `handle.tensor(shapes, dtype)` (POSIX_SHM) | the child maps the shm by identity on first receipt, **zero-copy** |

The local L3 children are forked eagerly in `Worker.init()`. A host tensor
created after that — the natural dynamic-shape serving pattern — reaches the
children by naming a `create_buffer` handle as a `Tensor`:

```python
worker = Worker(level=3, ...); worker.register(chip); worker.init()   # forks the chips

buf_h = worker.create_buffer(tokens * hidden_size * 4)   # POSIX shm, post-fork
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
