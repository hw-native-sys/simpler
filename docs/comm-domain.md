# Communication Domains — Dynamic Allocation

A **communication domain** is a symmetric device-memory window shared by a
subset of ranks, used for cross-rank reads/writes (collectives, SDMA, notify
protocols). Domains are allocated **dynamically from inside the orchestration
function** via `orch.allocate_domain(...)` — there is no init-time / static
declaration path.

For where the Orchestrator sits among the engine components see
[hierarchical_level_runtime.md](hierarchical_level_runtime.md); for the DAG
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
| `buffer_ptrs` | `{buffer_name: device_ptr}` for each `CommBufferSpec` |

Kernels read peer windows through `device_ctx` (which holds every rank's
window base, local + imported peer); `buffer_ptrs[name]` is the local slice.

---

## 2. Lifetime model

The handle is a context manager. Its lifecycle has **two distinct states**:

- **`released`** — set the moment `release()` is called (or the `with` block
  exits). Further indexing (`handle[i]`) raises. This is the *user-visible*
  state: "do not hand this domain to any new `submit_*`."
- **`freed`** — the backend `comm_release_domain_windows` has actually run and
  the device memory is gone. This happens **after** `Worker.run` drains the
  DAG, never inside the `with` block.

This split exists because `submit_next_level()` only *enqueues* DAG work;
`Worker.run()` does not drain until the orch function returns. If `release()`
freed memory immediately on `with`-exit, a still-queued task that captured the
domain's `device_ctx` / `buffer_ptrs` would read freed memory. So **release is
deferred**: `release()` flips `released` and queues the backend free; the real
free runs after drain, when every task that could reference the window has
completed.

Mental model: like `with open(f) as fh: ...` — the user-visible close is
lexical (end of block), the physical teardown is managed for you. Use
`handle.released` to guard against accidental reuse; use `handle.freed` only if
you must assert physical teardown.

Cleanup is **drain-safe**: even if a chip task fails and `drain()` re-raises,
`Worker.run` still executes the pending releases and sweeps any live domains the
orch fn forgot to release (LIFO), so a failed run cannot strand backend
allocations into the next run.

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
- only the **per-domain windows** are allocated (and freed after drain) on each
  `allocate_domain` / `run`. Each allocation gets a fresh `allocation_id` so
  concurrent or sequential domains never collide on IPC handshake / barrier
  names.

---

## 4. Backends

Both backends present the same `ChipDomainContext`; they differ only in how the
symmetric window is realized:

| Aspect | Sim | HCCL (onboard) |
| ------ | --- | -------------- |
| Window memory | POSIX shm + `ftruncate`, mmap'd per rank | VMM physical allocation + shareable-handle import; peer access via `aclrtDeviceEnablePeerAccess` |
| Subset barrier | shm-header atomic, `allocation_id`-scoped | file barriers, `allocation_id`-scoped |
| Window init | window zeroed after handshake (`memset`) | window zeroed after handshake (`aclrtMemset`) |
| Async-DMA workspace | n/a | a2a3: opt-in per Worker (`enable_sdma`); a5: optional communication overlay, gated off by default |

The window is zero-initialized on both backends so scratch/signal protocols see
a known starting state (matching the historical static-path contract).

On a2a3, async-DMA resources are a Worker-level opt-in, not a
communication-domain property. Construct the Worker with `enable_sdma=True` and
the runtime provisions the SDMA workspace once at init, latches its address into
the resident `KernelArgs`, and injects it into every run's kernel
`GlobalContext` (`get_dma_workspace`). A Worker without `enable_sdma` creates no
SDMA streams and its kernels read a zero workspace address. The workspace is
released at Worker finalize by ordinary stream/manager teardown.
Communication-domain allocation does not create SDMA streams or carry the
workspace through `CommContext`. Because an SDMA-enabled Worker's 48 STARS
streams sit in the device fault/sync domain, a fault on that Worker slows its
teardown; keep SDMA workloads on their own Worker (and, in CI, their own task)
so ordinary workloads are unaffected — see
[docs/investigations/2026-07-a2a3-sdma-fault-teardown.md](investigations/2026-07-a2a3-sdma-fault-teardown.md)
and issue #1425. `enable_sdma` is currently honored only by the a2a3 onboard
`tensormap_and_ringbuffer` runtime; host-build-graph, simulation, a5, and
provider-disabled builds fail Worker init fast when it is set. The a5
communication overlay remains isolated behind its default-off gate; see
[a5-sdma-overlay.md](a5-sdma-overlay.md).

---

## 5. Staging host data into a window

To preload host data (rather than have a kernel write the window), use
`orch.copy_to`:

```python
orch.copy_to(chip_idx, dst=handle[chip_idx].buffer_ptrs["input"], src=tensor.data_ptr(), size=n)
```

`copy_to` is **synchronous** (control-mailbox round-trip + synchronous
`rtMemcpy` H2D): when it returns, the bytes are in that rank's window. `src`
must be device-visible from the forked chip child — e.g. a `torch` tensor moved
to shared memory with `.share_memory_()` **before** `Worker.init()` forks the
chips.

**Cross-rank ordering:** when a kernel reads a *peer's* staged window, stage
**all** ranks' windows before submitting any kernel — `copy_to` is synchronous
but `submit_next_level` is async, so interleaving stage/submit per rank lets one
rank's producer run before another rank has finished staging:

```python
with orch.allocate_domain(...) as handle:
    for chip_idx in handle.workers:                       # stage all first
        orch.copy_to(chip_idx, dst=handle[chip_idx].buffer_ptrs["input"], src=..., size=n)
    for chip_idx in handle.workers:                       # then submit
        orch.submit_next_level(chip_handle, args, cfg, worker=chip_idx)
```

---

## 6. Host tensor visibility for `worker.run`

A host tensor passed to `worker.run(...)` / `orch.submit_next_level(...)` /
`orch.submit_sub(...)` is ultimately dereferenced from a forked local L3 child,
not the parent, so its memory must be backed by pages mapped into that child.
Fork-inherited MAP_SHARED mappings retain their virtual address, while post-fork
worker-allocated buffers may map at a different address and have their pointers
rewritten before decoding. Two sources are legal:

| Source | How | Why it works |
| ------ | --- | ------------ |
| **fork-inherited** | `tensor.share_memory_()` **before `Worker.init()`** (before the local L3 children are forked) | the child inherits the MAP_SHARED page at fork |
| **worker-allocated post-fork** | `worker.create_host_buffer(nbytes)` after the children exist | born-shared memory attached into every local child, **zero-copy** |

The local L3 children are forked eagerly in `Worker.init()`. A host tensor
created after that — the natural dynamic-shape serving pattern — is invisible to
the children unless it lives in a `create_host_buffer` buffer:

```python
worker = Worker(level=3, ...); worker.register(chip); worker.init()   # forks the chips

buf_h = worker.create_host_buffer(tokens * hidden_size * 4)   # born-shared, post-fork
buf_o = worker.create_host_buffer(batch * vocab * 4)
try:
    hidden = torch.frombuffer(buf_h.buffer, dtype=torch.float32).view(tokens, hidden_size)
    out    = torch.frombuffer(buf_o.buffer, dtype=torch.float32).view(batch, vocab)
    for step in batches:
        fill(hidden); worker.run(orch, ...)     # no per-run copy — child reads/writes the same pages
        use(out)
    del hidden, out                             # drop views before free
finally:
    worker.free_host_buffer(buf_h)
    worker.free_host_buffer(buf_o)
```

**Create once, reuse many runs.** `create_host_buffer` maps a shm into each
child and keeps it mapped; the child reads and writes the same physical pages
the parent sees, so there is **no per-run copy**. Build the tensor over
`buf.buffer` (buffer protocol — `torch.frombuffer` / `np.frombuffer`) once and
reuse it; a sub-view (slice) that fits inside the buffer is resolved
automatically. simpler stays framework-free — torch/numpy appear only on the
caller's side.

**Unregistered post-fork tensors are forwarded unvalidated.** A tensor that is
neither fork-inherited nor from `create_host_buffer` is passed through as-is:
the fork-inherited case is the common legitimate one, so it must keep working.
An anonymous post-fork tensor forwarded this way reads stale/unmapped memory in
the child — allocate it with `create_host_buffer` instead.

### Contract / limits

- **Zero-copy is a live shared medium.** The buffer's pages are shared with the
  child; during a `run` the child is reading/writing them, so the parent must
  not read or write the buffer until `run` returns (same contract as a
  fork-inherited `.share_memory_()` tensor). In-run cross-task ordering (a
  producer task's output read by a consumer task) is still enforced by the
  runtime's OverlapMap, keyed on the host address — no host-side copy involved.
- **Shape varies within `nbytes`.** A tensor built over the buffer may take any
  shape whose bytes fit; a view that runs past the buffer raises before dispatch
  (`overruns its host buffer`). To grow beyond `nbytes`, free and re-create a
  larger buffer. Do not free a buffer while a `run` using it is in flight, and
  drop every tensor/`memoryview` over `buffer.buffer` before `free_host_buffer`
  (or `close()`) so the shm can be released promptly.
- **`orch.copy_to` is the unmanaged low-level path.** `create_host_buffer`
  covers the `run` / `submit_next_level` host-tensor args. The explicit
  `orch.copy_to(src=tensor.data_ptr())` staging path (§5) is *not* validated —
  its `src` must be fork-inherited (`.share_memory_()` before `init()`)
  or a `create_host_buffer` buffer.
- **Fork-inherited anonymous memory is copy-on-write, hence stale.** Even a
  tensor the child legitimately inherited is only useful as a *live* input if it
  is MAP_SHARED: anonymous (non-`share_memory_`) pages are COW, so writes the
  parent makes *after* fork do not reach the child. A live input must be
  file-backed (`.share_memory_()` before `init()`) or a `create_host_buffer` one.

---

## 7. Examples

- [`tests/st/worker/collectives/allreduce/`](../tests/st/worker/collectives/allreduce/) — single domain, PTO-ISA remote
  reads over the window (allreduce scene tests with multiple algorithm modes).
- `examples/workers/l3/domain_rank_map/` — two domains, domain-local ranks,
  missing-domain `KeyError`, per-domain allreduce.
- `examples/workers/l3/dual_domain_overlap/` — overlapping domains where one
  worker participates in both.
- `examples/a2a3/tensormap_and_ringbuffer/sdma_async_completion_demo/` — host
  staging via `copy_to` + cross-rank `SdmaTget`; its producer `CoreCallable`
  declares the SDMA workspace requirement.
