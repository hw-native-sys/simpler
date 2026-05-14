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
   The Python `_chip_control` path and the C++ scheduler both go through
   `mailbox_mu_`; the implementation must verify the two paths share
   the same mutex.

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

1. Parent `Worker.register(target)` acquires `_register_lock`,
   allocates `cid`, inserts `_callable_registry[cid] = target`.
2. Parent creates `SharedMemory(create=True, size=target.buffer_size)`
   under the naming convention, memcpys the `ChipCallable` bytes in.
3. Parent dispatches CTRL_REGISTER **in parallel** to every chip
   mailbox, then waits for each `CONTROL_DONE`.
4. Each chip child opens the shm by name, reads
   `addr = ctypes.addressof(shm.buf)`, calls
   `cw._impl.prepare_callable_from_blob(cid, addr, shm.size)`, closes
   its mmap, writes `OFF_ERROR = 0`, sets `CONTROL_DONE`.
5. Parent unlinks the shm after all children ACK.

The child does **not** copy the bytes into its own bytearray.
`prepare_callable` performs the H2D copy into device GM internally, so
once that call returns the child no longer needs the source bytes. The
mmap is held only for the duration of the call.

The child does **not** insert anything into its own
`_callable_registry`. Eager prepare records the cid in the `prepared`
set, and `_ensure_prepared` short-circuits on `cid in prepared` before
it ever consults the registry.

### Successful register (L4 cascade)

`_child_worker_loop` ([python/simpler/worker.py:526-561](../python/simpler/worker.py#L526-L561))
gains a `CONTROL_REQUEST` branch. On CTRL_REGISTER it:

1. Decodes `(cid, shm_name)` from its own mailbox.
2. Calls `inner_worker._post_init_register(cid, shm_name)`, which is a
   new internal entrypoint that mirrors `Worker.register`'s post-init
   path but operates from inside the L3 process — broadcasting through
   L3's own chip and sub mailboxes using the **same shm name**.
3. ACKs the L4 parent only after the recursive broadcast completes.

The shm is created once at the top and reused throughout the cascade.
The top-level parent unlinks it only after the leaves' ACKs propagate
all the way back.

### Failure path

If any child reports a non-zero `OFF_ERROR`, the parent immediately
pops the cid from `_callable_registry`, unlinks the shm, and raises
`RuntimeError`. There is no reverse rollback to children that already
ACKed successfully — see section 5 for the rationale.

### Unregister

Symmetric, smaller payload. CTRL_UNREGISTER carries only the cid;
each child calls `cw.unregister_callable(cid)`
([python/bindings/task_interface.cpp:678-687](../python/bindings/task_interface.cpp#L678-L687)).
That API releases the per-cid share of the device orch SO buffer.
Kernel binaries remain resident until `finalize`. Errors during
unregister are warned-and-continued; the parent removes the registry
entry unconditionally so the cid slot becomes reusable.

---

## 5. Failure modes and concurrency

### Failure modes

| Trigger | Handling |
| ------- | -------- |
| Child cannot open shm or `prepare_callable` raises | Child writes `OFF_ERROR=1` + message; parent raises `RuntimeError` |
| Some children succeed before failure | Parent unlinks shm and raises; **no reverse rollback** |
| `cid >= 64` | Rejected at parent before broadcast; child re-checks defensively |
| Child process crashes mid-CONTROL | Parent's spin-poll on `CONTROL_DONE` never returns — same hang as existing `_CTRL_MALLOC` / `_CTRL_PREPARE`. Out of scope for this design |

The "no reverse rollback" decision is deliberate. A best-effort
CTRL_UNREGISTER broadcast to the successful children would itself have
a failure path, doubling the surface area of the error handler.
Partial state left in successful children — an entry in the AICPU
`orch_so_table_` that will never receive a TASK_READY for this cid —
is inert garbage and is overwritten when the cid is reused.

### Concurrency

- A `self._register_lock` (`threading.Lock`) serializes register calls
  across parent threads. It also gates `Worker.run` entry/exit, which
  increments and decrements a `_orch_in_flight` counter.
- **Register is quiescent-state-only.** The Python `_chip_control` path
  is **unlocked** with respect to the C++ scheduler's `mailbox_mu_` —
  the two paths do not share a mutex. Safety relies on the scheduler
  being provably idle outside `Worker.run()`: no orch fn → no submits
  → no `dispatch_process`. The parent enforces this by checking
  `_orch_in_flight == 0` under `_register_lock` and raising
  `RuntimeError` if a register races a run. This matches the existing
  Python-only model of `_CTRL_PREPARE`, which is also issued only at
  init-time.
- Same-level broadcast runs in parallel. Parent writes CTRL_REQUEST
  concurrently to every chip mailbox and waits for all ACKs. Latency
  drops from `N × prepare_cost` to `1 × prepare_cost`. Parallel
  broadcast does not complicate error handling: any single child
  failure still raises, and there is no rollback to coordinate.
- `Worker.register()` is synchronous and blocking. When it returns,
  every child has completed prepare. Users may submit TASK_READY for
  the new cid on the very next line.

---

## 6. Observability and coupling

### Observability conventions

- Child-side error messages use the format
  `register cid=<N> chip=<id>: <reason>`. The parent prepends
  `device_id` when raising.
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
- Lifting the quiescent-state constraint to support register during
  `Worker.run()` would require a C++-side `WorkerThread::control_register`
  method (mirror `control_malloc`) that acquires `mailbox_mu_` and
  serializes against `dispatch_process`. No production use case
  demands this today; deferred.

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
