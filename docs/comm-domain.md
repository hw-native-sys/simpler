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
| Window memory | POSIX shm + `ftruncate`, mmap'd per rank | `aclrtMalloc` + `aclrtIpcMem*` import; peer access via `aclrtDeviceEnablePeerAccess` |
| Subset barrier | shm-header atomic, `allocation_id`-scoped | file barriers, `allocation_id`-scoped |
| Window init | window zeroed after handshake (`memset`) | window zeroed after handshake (`aclrtMemset`) |
| SDMA workspace | n/a | provisioned once per handle (`ensure_sdma_workspace`); inherited into each domain `CommContext` |

The window is zero-initialized on both backends so scratch/signal protocols see
a known starting state (matching the historical static-path contract).

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

A host tensor passed to `worker.run(...)` / `orch.submit_next_level(...)` is
ultimately dereferenced from the forked chip child, not the parent. For the
design rationale behind the choices below (copy vs zero-copy, explicit
registration, the procfs/inode classification, and the alternatives that were
deferred) see [`host-buffer-registration-design.md`](host-buffer-registration-design.md).
Either the
tensor's own storage must already be reachable there (fork-inherited), or
`worker.register_host_buffer(...)` must provide a child-visible shm mirror — in
the registered case the tensor storage itself need not be child-visible. Three
sources are legal:

| Source | How | Why it works |
| ------ | --- | ------------ |
| **fork-inherited** | `tensor.share_memory_()` **before the chip children are forked** (i.e. before the first `Worker.run()`) | the child inherits the MAP_SHARED page at fork |
| **registered post-fork** | `worker.register_host_buffer(tensor)` after the chips exist | maps a shm into every child for the buffer's lifetime |
| anything else | — | raises an actionable error before dispatch |

The chip children are forked lazily on the **first** `run()`. A host tensor
created after that — the natural dynamic-shape serving pattern — is invisible to
the children unless registered:

```python
worker = Worker(level=3, ...); worker.register(chip); worker.init()
worker.run(orch0, ...)                          # forks the chips

hidden = torch.empty((tokens, hidden_size)).share_memory_()   # created post-fork
out    = torch.empty((batch, vocab)).share_memory_()
h_hidden = worker.register_host_buffer(hidden)  # map into every chip child
h_out    = worker.register_host_buffer(out)
try:
    for batch in batches:
        fill(hidden); worker.run(orch, ...)     # H2D copy-in, D2H copy-out per run
        use(out)
finally:
    worker.unregister_host_buffer(h_hidden)
    worker.unregister_host_buffer(h_out)
```

**Register once, reuse many runs.** Registration maps a shm into each child and
keeps it mapped; every `run` mirrors the tensor through it (H2D copy-in before
the task, D2H copy-out after the run drains). Register the buffer — or a
fixed-size superset you sub-slice — once and reuse it; re-registering per run
pays a map/unmap broadcast each time. A sub-view (slice) of a registered buffer
is resolved automatically.

**Error path.** An unregistered post-fork host tensor raises before any dispatch:

> Host tensor 0x… is not visible to the L3 chip child (created after fork, not
> registered). Call worker.register_host_buffer(tensor) before run(), or allocate
> it with .share_memory_() before init().

### Scope / limits (v1)

- **memcpy, not zero-copy.** A registered buffer is a *separate* shm; each run
  copies `tensor → shm` (in) and `shm → tensor` (out). For a large hot-path
  tensor this is a double copy. True zero-copy (mapping the tensor's own storage)
  is a later optimization.
- **`orch.copy_to` is the unmanaged low-level path.** Registration covers the
  `run` / `submit_next_level` host-tensor args (the post-fork host-tensor
  scenario). The
  explicit `orch.copy_to(src=tensor.data_ptr())` staging path (§5) is *not*
  translated or validated by `register_host_buffer` — its `src` must still be
  fork-inherited (`.share_memory_()` before the chip children are forked, i.e.
  before the first `run()`).
- **Anonymous post-fork heap.** A *large* post-fork `torch.empty` (its own mmap)
  is correctly rejected. A *small* non-shared tensor the allocator sub-slices out
  of a fork-time heap arena can slip past the check (anonymous, inside a fork
  range) and read stale data in the child — always `share_memory_` or register
  host tensors used for chip dispatch.
- **Non-procfs platforms (e.g. macOS).** The reachability check reads
  `/proc/self/maps`; where it is unavailable the fork snapshot is empty, so an
  unregistered host tensor cannot be classified and is **passed through
  unvalidated** (rather than rejected) — a fork-inherited tensor is the common
  legitimate case and must keep working. The first such pass-through emits a
  one-time `UserWarning`; the caller is then responsible for ensuring the tensor
  is fork-inherited or registered. Error path C above is enforced only where
  procfs exists (Linux, including onboard).
- **No in-run produce-then-consume of the same registered buffer.** Copy-in
  (`tensor → shm`) runs per `submit_next_level`, while tasks may already be in
  flight before `drain()`. If one task writes a registered buffer and a later
  dependent task in the *same* `run` reads it, the consumer's copy-in can overwrite
  the producer's result with the stale parent tensor. Use a registered buffer as a
  run input *or* a run output, not as an intermediate handed between tasks within
  one run; chain results through device buffers instead.
- **Fork-inherited anonymous memory is copy-on-write, hence stale.** Even a tensor
  the child legitimately inherited is only useful as a *live* input if it is
  MAP_SHARED: anonymous (non-`share_memory_`) pages are COW, so writes the parent
  makes *after* fork do not reach the child. A live input must be file-backed
  (`.share_memory_()` before `init()`) or registered.

---

## 7. Examples

- `examples/workers/l3/allreduce_distributed/` — single domain, PTO-ISA remote
  reads over the window.
- `examples/workers/l3/domain_rank_map/` — two domains, domain-local ranks,
  missing-domain `KeyError`, per-domain allreduce.
- `examples/workers/l3/dual_domain_overlap/` — overlapping domains where one
  worker participates in both.
- `examples/a2a3/tensormap_and_ringbuffer/sdma_async_completion_demo/` — host
  staging via `copy_to` + cross-rank `SdmaTget` (needs the SDMA workspace).
