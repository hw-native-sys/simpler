# Dynamic Callable Registration over IPC

This document specifies a design for **registering and unregistering
`ChipCallable` objects after `Worker.init()` has forked the child
processes**, using an IPC control-plane that complements the existing
mailbox / ringbuffer / handshake data-plane.

The current `Worker.register()` API rejects all calls after `init()` at
L3+ ([python/simpler/worker.py:647-651](../python/simpler/worker.py#L647-L651)).
The rejection exists because chip-children and sub-workers inherit
`_callable_registry` purely through fork-time copy-on-write — any
post-fork mutation in the parent is invisible to the child. This design
removes that constraint for `ChipCallable` only.

It is a **design document**, not an implementation. Concrete code lands
in subsequent PRs.

---

## 1. Context and motivation

### Real drivers

- **JIT-compiled callables**: kernels produced mid-run by a compiler
  cannot be registered at init time.
- **Plugin operator libraries** loaded at runtime.
- **cid reuse via `unregister`**: `MAX_REGISTERED_CALLABLE_IDS = 64`
  ([src/common/task_interface/callable_protocol.h](../src/common/task_interface/callable_protocol.h))
  is exhausted quickly by long-running JIT workloads. Without a
  dynamic-unregister path, the only way to free a slot is to recycle
  the worker pool. Bundling `unregister` into this design is the only
  way to make dynamic register practically usable beyond 64
  registrations.

### Rejected pseudo-drivers

- **Lazy startup cost**. `_CTRL_PREPARE` already lets a parent warm a
  cid post-init using the CoW-inherited registry entry. Dynamic
  register does not help here.
- **Memory peak at fork**. `ChipCallable` bytes live in the parent's
  heap and are CoW-shared; they do not multiply across children.

### Non-goals

- Lifting `MAILBOX_SIZE = 4096` ([src/common/hierarchical/worker_manager.h](../src/common/hierarchical/worker_manager.h)).
- Lifting `MAX_REGISTERED_CALLABLE_IDS = 64`.
- Hot-swap of a cid in place (re-register the same cid with new bytes).
- Dynamic register of Python `OrchFn` / `SubFn` — only `ChipCallable`
  is in scope. Calls with other target types raise `NotImplementedError`.
- Cross-chip shared device GM for the orch SO.

---

## 2. Current constraints

What any design must respect:

1. `_callable_registry` is a Python `dict` keyed by `cid`. Parent writes
   after fork are invisible to children
   ([python/simpler/worker.py:628-671](../python/simpler/worker.py#L628-L671)).
2. `MAX_REGISTERED_CALLABLE_IDS = 64`. The AICPU keeps a fixed-size
   `orch_so_table_[64]`; post-fork register must enforce `cid < 64` at
   the parent and re-check at the child.
3. A `ChipCallable` is contiguous bytes — header + signature + storage
   with embedded orch SO + child kernel binaries — typically hundreds
   of KB to a few MB. It cannot fit in the 4 KB mailbox; a side-band
   transport is required.
4. `_sub_worker_loop` ([python/simpler/worker.py:226-255](../python/simpler/worker.py#L226-L255))
   and `_child_worker_loop` currently handle only `TASK_READY` /
   `SHUTDOWN`; they do not service `CONTROL_REQUEST`.
5. `prepare_callable` is expensive: it copies the orch SO + kernel
   binaries to device GM (`rtMemcpy`), records kernel addresses, and on
   the `host_build_graph` variant also `dlopen`s the host orch SO.
   Eager prepare at register time matches the cost shape of init-time
   pre-register; lazy prepare would defer this to the first TASK_READY.
6. Register must serialize against TASK_READY on the same mailbox.
   Both `dispatch_process` and `WorkerThread::control_register` hold
   `mailbox_mu_`, so the two paths are guaranteed serial within C++.

---

## 3. Wire format and binding prerequisites

### Sub-command constants

New constants extend the existing `_CTRL_MALLOC=0` / `FREE=1` /
`COPY_TO=2` / `COPY_FROM=3` / `PREPARE=4` series:

```text
_CTRL_REGISTER   = 5
_CTRL_UNREGISTER = 6
```

### Mailbox layout (state == CONTROL_REQUEST)

CTRL_REGISTER:

| Offset | Field | Notes |
| ------ | ----- | ----- |
| `OFF_CALLABLE` | sub_cmd = 5 | uint64 |
| `CTRL_OFF_ARG0` | cid | low 32 bits; high bits reserved |
| `OFF_ARGS[0..]` | NUL-terminated shm name | ≤ 32 bytes |

CTRL_UNREGISTER:

| Offset | Field | Notes |
| ------ | ----- | ----- |
| `OFF_CALLABLE` | sub_cmd = 6 | uint64 |
| `CTRL_OFF_ARG0` | cid | low 32 bits |

`MAILBOX_SIZE` is unchanged. No `total_size` field: the child reads
`SharedMemory(name=...).size` directly. No `hash` field: POSIX shm is
strongly consistent between parent and child; a checksum only catches
defects in this code itself.

### Shm naming convention

`simpler-cb-<pid>-<cid>-<monotonic_counter>`. The pid prefix prevents
collisions when multiple simpler instances run on the same host.

### Binding prerequisite

The child needs to call into C++ with a raw pointer to the shm-mapped
bytes. The C++ entrypoint already exists:

- `ChipWorker::prepare_callable(int32_t, const void *)`
  ([src/common/worker/chip_worker.h:56](../src/common/worker/chip_worker.h#L56))
- `prepare_callable(DeviceContextHandle, int32_t, const void *)`
  ([src/common/worker/pto_runtime_c_api.h:148](../src/common/worker/pto_runtime_c_api.h#L148))

The original nanobind binding took a `PyChipCallable` object.
A companion `prepare_callable_from_blob(cid, blob_ptr)` overload
([python/bindings/task_interface.cpp](../python/bindings/task_interface.cpp))
follows the existing precedent set by `run_prepared_from_blob` —
"Python passes a raw mailbox pointer, C++ casts to `const void *`".
The overload takes two arguments only; no `size` parameter, because
`prepare_callable` reads the total size from the ChipCallable header
itself. No C++ changes are required.

---

## 4. Sequence flow

### Successful register (single level)

1. Parent `Worker.register(target)` acquires `_registry_lock`,
   allocates `cid` via `_allocate_cid` (smallest unused integer in
   `[0, MAX_REGISTERED_CALLABLE_IDS)`), inserts
   `_callable_registry[cid] = target`. The lock is held through the
   broadcast so a concurrent register cannot allocate the same cid
   if this broadcast fails and pops the entry.
2. Parent calls `_Worker.broadcast_register_all(cid, blob_ptr,
   blob_size)` (nanobind). The GIL is released for the C++ call.
3. C++ `WorkerManager::broadcast_register_all` creates a POSIX shm
   under name `simpler-cb-<pid>-<cid>-<counter>`, `mmap`s it, and
   `memcpy`s `blob_size` bytes from `blob_ptr`.
4. C++ spawns one `std::thread` per `next_level_threads_` entry; each
   calls `WorkerThread::control_register(cid, shm_name)`, which holds
   that `WorkerThread::mailbox_mu_` (serialising against any in-flight
   `dispatch_process` on the same mailbox), writes
   `(CTRL_REGISTER, cid, shm_name)` into the mailbox, and spin-polls
   `CONTROL_DONE`.
5. Each chip child opens the shm by name, calls
   `cw._impl.prepare_callable_from_blob(cid, addr)`, closes its mmap,
   writes `OFF_ERROR = 0`, sets `CONTROL_DONE`.
6. C++ joins all per-thread workers, `munmap`s and `shm_unlink`s the
   staging region, then throws if any child failed.

The child does **not** copy the bytes into its own bytearray.
`prepare_callable` performs the H2D copy into device GM internally, so
once that call returns the child no longer needs the source bytes. The
mmap is held only for the duration of the call.

The child does **not** insert anything into its own
`_callable_registry`. Eager prepare records the cid in the `prepared`
set, and `_ensure_prepared` short-circuits on `cid in prepared` before
it ever consults the registry.

### Successful register (L4 cascade)

`_child_worker_loop` ([python/simpler/worker.py](../python/simpler/worker.py))
gains a `CONTROL_REQUEST` branch. On CTRL_REGISTER it:

1. Decodes `(cid, shm_name)` from its own mailbox.
2. Opens the shm by name, reconstructs the `ChipCallable` via
   `ChipCallable.from_bytes`, and calls
   `inner_worker._register_at(cid, callable_obj)`.
3. The inner `Worker._register_at` records the cid in its own
   `_callable_registry` and calls
   `_Worker.broadcast_register_all` recursively for its own
   `next_level_threads_` (which fans out to a fresh shm at the inner
   level — the L4 shm is **not** reused by the inner broadcast).
4. ACKs the L4 parent only after the recursive broadcast completes.

The L4 parent's shm is unlinked when its own `broadcast_register_all`
returns — after every L3 child has ACKed, which in turn happens after
every L3's own broadcast has fully drained.

### Failure path

If any child reports a non-zero `OFF_ERROR`, the parent immediately
pops the cid from `_callable_registry`, unlinks the shm, and raises
`RuntimeError`. There is no reverse rollback to children that already
ACKed successfully — see section 5 for the rationale.

### Unregister

Symmetric, smaller payload. CTRL_UNREGISTER carries only the cid;
each child calls `cw.unregister_callable(cid)`
([python/bindings/task_interface.cpp](../python/bindings/task_interface.cpp)).
That API releases the per-cid share of the device orch SO buffer.
Kernel binaries remain resident until `finalize`. Errors during
unregister are best-effort: C++
`WorkerManager::broadcast_unregister_all` returns a per-child error
list (empty on full success); the parent warns to stderr and removes
the registry entry unconditionally so the cid slot becomes reusable.

The chip-child `_CTRL_REGISTER` handler is **self-healing on cid reuse**:
before calling `prepare_callable_from_blob`, if the child's local
`prepared` set still contains the cid, it defensively re-issues
`unregister_callable` to clear any residual host-side
`prepared_callables_` / `aicpu_seen_callable_ids_` left by a previous
`_CTRL_UNREGISTER` whose Python handler failed before reaching
`prepared.discard`. This makes the parent's best-effort "cid slot is
reusable" contract hold even when a subset of chip children reported
errors on the prior unregister. The C++ boundary
(`DeviceRunner::register_prepared_callable`) keeps its fail-fast
duplicate-registration check; the self-heal lives entirely at the IPC
layer, and is zero-cost on the happy path (the `cid in prepared` gate
skips it when there is no residue to clean).

---

## 5. Failure modes and concurrency

### Failure modes

| Trigger | Handling |
| ------- | -------- |
| Child cannot open shm or `prepare_callable` raises | Child writes `OFF_ERROR=1` + message; parent raises `RuntimeError` |
| Some children succeed before failure | Parent unlinks shm and raises; **no reverse rollback** |
| `cid >= 64` | Rejected at parent before broadcast; child re-checks defensively |
| Child process crashes mid-CONTROL | Parent's spin-poll on `CONTROL_DONE` never returns — same hang as existing `_CTRL_MALLOC` / `_CTRL_PREPARE`. Out of scope for this design |
| Prior CTRL_UNREGISTER raised in child, parent popped registry | Next CTRL_REGISTER for the same cid self-heals at the child: re-issues `unregister_callable` before `prepare_callable_from_blob`. AICPU `orch_so_table_[cid]` is dlclose'd + reloaded on the next dispatch via `aicpu_seen_callable_ids_.erase` |

The "no reverse rollback" decision is deliberate. A best-effort
CTRL_UNREGISTER broadcast to the successful children would itself have
a failure path, doubling the surface area of the error handler.
Partial state left in successful children — an entry in the AICPU
`orch_so_table_` that will never receive a TASK_READY for this cid —
is inert garbage and is overwritten when the cid is reused.

The self-heal path has a compound failure mode worth noting: if
`cw.unregister_callable` (the defensive cleanup) raises *and* the
follow-up `prepare_callable_from_blob` also raises, the child publishes
`OFF_ERROR=1` and the parent pops the registry, but C++-side
`prepared_callables_` / AICPU `orch_so_table_` may stay populated. That
residue is the same flavour of inert garbage discussed above: the next
attempt to register the same cid re-enters self-heal (because the
child's `prepared` set still holds it) and gives `unregister_callable`
another chance; if the cid is never reused, the entry sits until
`DeviceRunner::finalize` clears the slot at process exit.

### Concurrency

- `Worker.register` / `Worker.unregister` are driven through C++
  `_Worker.broadcast_register_all` / `_Worker.broadcast_unregister_all`
  ([python/bindings/worker_bind.h](../python/bindings/worker_bind.h)).
  Each per-WorkerThread `control_register` call holds that
  WorkerThread's `mailbox_mu_` for the round-trip, so the broadcast
  serializes against any in-flight `dispatch_process` on that mailbox.
  **No Python-side quiescent guard is needed**: register issued during
  `Worker.run()` blocks until the in-flight TASK acknowledges on that
  mailbox, then claims the lock — same shape as `control_malloc` /
  `control_free`, which have always been safe to call mid-run.
- A narrow `self._registry_lock` (`threading.Lock`) protects only the
  `_callable_registry` dict mutation against concurrent register /
  unregister calls from multiple Python threads. The lock is released
  before the C++ broadcast so the spin-poll wait does not serialize
  Python-side registration.
- Same-level broadcast runs in parallel inside C++ via
  `std::thread` fan-out across `next_level_threads_`. Per-WorkerThread
  `mailbox_mu_` is independent, so N `control_register` calls proceed
  concurrently — latency is `1 × prepare_cost` instead of
  `N × prepare_cost`.
- `Worker.register()` is synchronous and blocking. When it returns,
  every child has completed prepare. Users may submit TASK_READY for
  the new cid on the very next line.

---

## 6. Observability and coupling

### Observability conventions

- Child-side error messages use the format
  `register cid=<N> chip=<id>: <reason>`. The C++ broadcast helper
  strips its own `control_register failed on child:` wrapper and
  re-throws as `Worker.register(cid=<N>) failed on next_level <i>:
  <child msg>` so the operator sees a single, layered context line.
- Partial state after a failed register is a known blind spot. There
  is no query API to list which children registered which cids. Users
  must treat a failed register as "the cid is gone" and retry with a
  fresh registration.
- `aicpu_dlopen_count` and `host_dlopen_count` on `ChipWorker`
  ([src/common/worker/chip_worker.h](../src/common/worker/chip_worker.h))
  continue to reflect dlopen counts under dynamic register and cid
  reuse. This is a correctness constraint on the implementation.

### Coupling to existing task flow

- **L2 is unaffected.** L2 is in-process; `prepare_callable` does not
  traverse the mailbox. CTRL_REGISTER only fires at L3+.
- **The lazy `_ensure_prepared` path is decoupled.** Dynamic
  registrations always go through eager prepare and land in
  `prepared`, so the lazy fallback (which would consult the registry)
  is never reached. Future work to support lazy dynamic register would
  require the child registry to hold bytes, not just a sentinel.
- **`dummy_task` is orthogonal.** It does not traverse callable
  dispatch and shares no state with this design.
- **Pre-fork register remains the recommended path.** Dynamic register
  does not replace it. Use pre-fork when the full callable set is
  known at init; use dynamic register for JIT, plugins, or callables
  generated inside a training step. Each dynamic register pays full
  broadcast + per-child prepare cost.

For the broader callable / task data flow, see
[task-flow.md](task-flow.md) and
[hierarchical_level_runtime.md](hierarchical_level_runtime.md).

---

## 7. Open questions

- How to detect and recover from a child crash that strands the parent
  spin-polling on `CONTROL_DONE`. This problem is shared with all
  existing `_CTRL_*` operations and is not solved here.
- Whether `unregister` should wait for in-flight TASK_READYs to the
  same cid to drain before proceeding. This design assumes the user is
  responsible for not unregistering a cid with outstanding work.
- Path to raising `MAX_REGISTERED_CALLABLE_IDS` beyond 64. Requires
  AICPU-side changes to `orch_so_table_` and is out of scope here.

---

## 8. Out of scope

- Raising `MAILBOX_SIZE` past 4 KB.
- Raising `MAX_REGISTERED_CALLABLE_IDS` past 64.
- Cross-chip shared device GM for the orch SO.
- Hot-swap of a cid (re-register the same cid with new bytes).
- Switching the IPC backend off `multiprocessing.shared_memory`.
- A C++-side equivalent of `Worker.register` — the registry lives in
  Python; the C++ side only dispatches.
- Persistent cross-restart cache of prepared callables.
