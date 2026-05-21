# Python Callable Serialization for L3+ Register

This document specifies a design for registering Python callables after an
L3+ `Worker` has already initialized, and in the common case after child
processes have already started.

The design is separate from
[callable-ipc-dynamic-register.md](callable-ipc-dynamic-register.md). That
document covers `ChipCallable` binary registration for chip children. This
document covers Python callables consumed by SUB workers and by higher-level
Worker-child dispatch loops.

It is a design document, not an implementation.

---

## 1. Context

Every task submitted through the hierarchical runtime carries a `callable_id`.
For L3+ Python execution paths, that id is resolved in a Python registry:

| Submit path | Recipient | Registry entry |
| ----------- | --------- | -------------- |
| `orch.submit_sub(cid, ...)` | SUB child | Python sub callable |
| L4+ `submit_next_level(cid, ...)` | Worker child | Python orch callable |

Today, these entries must be registered before fork. The child process sees
the parent's `_callable_registry` only through fork-time copy-on-write. Any
parent-side mutation after fork is invisible to the already-running child.

`ChipCallable` post-init registration already uses a control-plane plus
side-band shm payload because binary callables can be copied and prepared in
chip children. Python callables need the same high-level shape, but the
payload is serialized Python code/data and the recipients are Python-capable
children, not chip children.

### Goals

- Allow `Worker.register(py_callable)` after `Worker.init()` at level >= 3.
- Make the returned `cid` usable when `register()` returns.
- Preserve the current registration behavior before children start.
- Reuse the existing mailbox control-plane and per-mailbox serialization
  against in-flight dispatch.
- Support unregister and cid reuse.
- Keep the API synchronous and deterministic from the caller's perspective.

### Non-goals

- Dynamic registration of `ChipCallable`; that is covered by the binary
  callable design.
- Cross-host or cross-Python-version serialization.
- Detecting or recovering from child process crashes while a mailbox control
  request is in flight. This is a shared control-plane liveness limitation,
  not specific to Python callable registration.
- Loading untrusted serialized bytes safely. This feature unpickles code from
  the same user process and is not a security boundary.
- Automatically registering callables inside arbitrary descendant Workers.
  A `Worker.register()` call updates the registry owned by that Worker and
  the already-started children that consume that registry.
- Changing `MAX_REGISTERED_CALLABLE_IDS`.

---

## 2. Public Contract

`Worker.register(target)` keeps one cid space for both `ChipCallable` and
Python callables. The target type selects the dynamic-register route.

- L2 `ChipCallable`: existing prepare path.
- L2 Python callable: invalid target.
- L3+ before this Worker has started child processes: store the target in the
  parent registry; future children will inherit the registry when they start.
  This preserves the pre-`init()` behavior and extends it to the post-`init()`,
  before-first-`run()` window, where no child process has been forked yet.
- L3+ after this Worker has started child processes: existing binary IPC for
  `ChipCallable`, and the new serialized Python IPC path for Python callables.

The post-start Python path is synchronous:

1. The parent allocates a cid and stores `target` in its registry.
2. The parent serializes `target`.
3. The parent broadcasts the payload to every Python-capable child that may
   resolve this Worker's registry.
4. Each child deserializes the payload and updates its local registry.
5. `register()` returns only after every required child has acknowledged.

The parent must not submit a newly registered cid until `register()` returns.
The runtime does not attempt to make a cid visible before the synchronous
broadcast completes.

### Recipients

The parent routes Python callable registration to Python-capable children:

- SUB child processes of the same Worker.
- L4+ next-level Worker-child dispatch loops, because they resolve the
  parent's registered orch functions before calling `inner_worker.run(...)`.

L3 chip children are not recipients for Python callable payloads. They can
only consume prepared `ChipCallable` ids.

Because `Worker.register()` does not currently take a "sub" versus
"next-level orch" kind, the simplest compatible policy is to broadcast to all
Python-capable child groups owned by this Worker. Extra registry entries are
inert if a cid is never submitted to that worker type.

This preserves the current public API: `Worker.register(target)` does not gain
an explicit target-kind parameter. Submit-time APIs continue to decide how the
cid is interpreted.

If no Python-capable child exists after children start, registering a Python
callable should fail with a clear `RuntimeError`. Keeping a cid that no child
can ever resolve is more confusing than rejecting it.

### Callable Shape

The runtime does not validate function signatures at register time. Existing
dispatch-time behavior remains:

- SUB callables are invoked as `fn(args)`.
- Worker-child orchestration callables are invoked through
  `inner_worker.run(orch_fn, args, cfg)`, so they must match the usual
  orchestration shape.

Signature errors surface from the child execution path and are reported
through the mailbox error field, as they are today.

---

## 3. Serialization

The payload must fit outside the 4 KB mailbox, so Python callables use a
side-band POSIX shm exactly like dynamic `ChipCallable` registration. The
mailbox carries only a shm name and cid.

### Serializer Policy

Dynamic Python callable registration uses `cloudpickle`.

`cloudpickle` is a runtime dependency of the `simpler` package, not only a test
dependency, because child processes deserialize user callables during normal
`Worker.register()` operation.

Registration before children start already allows lambdas and closures because
the startup path copies the registry directly. A dynamic feature that rejects
these common shapes would be surprising and would make several existing L3/L4
test patterns impossible to move to dynamic registration.

Stdlib `pickle` is not used for this path because it serializes most
functions by module/name reference and is therefore limited to importable
top-level functions and callable classes. It is a useful mental model for the
trust boundary, but it is not the runtime format.

### Callable Shape and Closure Semantics

Post-start registration supports callable shapes that `cloudpickle` can
serialize and the child can deserialize in the same Python environment:

- importable top-level functions;
- lambdas and nested functions whose captured values are serializable;
- callable class instances whose instance state is serializable.

This is not identical to registration before children start. Startup children
inherit a snapshot of the parent's address space, so a closure may appear to
work because the child inherited the captured object at startup. Post-start
registration sends serialized bytes to an already-running child, so captured
objects are copied or reconstructed through `cloudpickle`.

Callables should not rely on captured process-local resources being equivalent
to fork inheritance. Examples include locks, events, open files, sockets,
`SharedMemory.buf` memoryviews, mmap views, `Worker` or `ChipWorker` instances,
nanobind/C++ handles, and device-pointer wrappers. Prefer capturing stable
identifiers that the child can reopen or reconstruct, such as a shared-memory
name instead of a live `SharedMemory.buf` object.

### Payload Format

The parent serializes the callable into an in-memory byte blob. The C++
broadcast binding creates the side-band POSIX shm, copies that blob into it,
fan-outs the shm name to children, and unlinks the shm after all child
round-trips have completed. Python does not create or unlink the broadcast shm.

The Python binding must accept a Python buffer object, preferably `bytes`, not
only a raw integer pointer. The binding copies the buffer into the staging shm
before releasing any reference to the Python object. This avoids depending on a
temporary `cloudpickle.dumps(...)` result staying alive while nanobind has
released the GIL and C++ worker threads are fanning out the control command.

The shm bytes are exactly the result of `cloudpickle.dumps(target)`. There is no
custom payload header in the first implementation. The control command already
identifies the operation as Python callable registration, and this feature is a
trusted local transport rather than a cross-version or untrusted wire protocol.
Malformed or incompatible bytes fail through `cloudpickle.loads(...)` and are
reported through the normal mailbox error field.

### Child Deserialization

Each recipient child:

1. Opens the shm by name.
2. Copies the shm contents into `bytes`.
3. Verifies that `cid` is in `[0, MAX_REGISTERED_CALLABLE_IDS)`.
4. Deserializes the callable with `cloudpickle`.
5. Verifies that the result is callable.
6. Installs it into the child's local registry under the requested cid.
7. Closes the shm and acknowledges `CONTROL_DONE`.

For cid reuse after partial unregister failures, Python registration should
overwrite `registry[cid]` in the child. The parent only allocates free cids
from its own registry, so an existing child entry at the same cid is residue
from a prior best-effort failure and should be replaced.

Because Python callables and `ChipCallable` objects share one cid space, the
same cleanup rule also applies when a cid is reused across target types. A
post-start `ChipCallable` registration must clear any stale Python dispatch
entry for the same cid from Python-capable children before the cid is reported
usable. Otherwise a failed Python unregister followed by `ChipCallable` reuse
could leave a Worker-child dispatch loop resolving the old Python callable.

---

## 4. Control Plane

Add new control subcommands rather than overloading the existing
`CTRL_REGISTER` used for `ChipCallable`:

```text
CTRL_PY_REGISTER   = 10
CTRL_PY_UNREGISTER = 11
```

The mailbox layout for `CTRL_PY_REGISTER` mirrors binary register:

| Offset | Field | Notes |
| ------ | ----- | ----- |
| `OFF_CALLABLE` | sub_cmd = `CTRL_PY_REGISTER` | uint64 |
| `CTRL_OFF_ARG0` | cid | low 32 bits |
| `OFF_ARGS[0..]` | NUL-terminated shm name | fixed-width slot |

`CTRL_PY_UNREGISTER` carries only the cid in `CTRL_OFF_ARG0`.

### Parent-Side Flow

`Worker.register(target)` gains a Python-callable dynamic route:

1. Hold `_registry_lock`.
2. Reject non-callable Python targets.
3. If `_py_register_active_run` is true, raise `RuntimeError`.
4. If this Worker has not started child processes, allocate the smallest free
   cid, insert `self._callable_registry[cid] = target`, and return the cid;
   future children will inherit the registry when they start.
5. If no configured Python-capable child group exists, raise `RuntimeError`.
6. Allocate the smallest free cid.
7. Insert `self._callable_registry[cid] = target`.
8. Serialize the target into a bytes blob with `cloudpickle.dumps(...)`.
9. Broadcast `CTRL_PY_REGISTER` to required Python-capable worker groups.
10. On any failure, pop the parent registry entry and raise.
11. Return cid on success.

The "configured Python-capable child group" check uses the Worker's own
configuration, not child-process state:

- `num_sub_workers > 0` means SUB children will consume this registry.
- `len(_next_level_workers) > 0` means Worker children will consume this
  registry.

This check applies only after child processes have started. Before children
start, including after `init()` but before the first `run()`, registration uses
the parent-registry path and does not reject unused Python callables.

If no free cid exists in `[0, MAX_REGISTERED_CALLABLE_IDS)`, register raises
`RuntimeError` before mutating the parent registry or broadcasting to children.
The caller can recover by unregistering unused callables and retrying.

`_py_register_active_run` is initialized to false in `Worker.__init__`.
`Worker.run()` and Python callable register/unregister use `_registry_lock` as
their gate:

1. Before starting L3+ children or entering the orchestration body,
   `Worker.run()` performs the active-state transition as one critical section:

   ```python
   with self._registry_lock:
       if self._py_register_active_run:
           raise RuntimeError("Worker.run() is already active")
       self._py_register_active_run = True
   ```

   On the first run, this happens before `_start_hierarchical()` takes the
   startup registry snapshot, so a concurrent `register()` cannot return
   through the startup path after children have already missed that registry
   entry.
2. Python callable register/unregister hold `_registry_lock` for the full
   parent-side mutation and broadcast. A concurrent `Worker.run()` blocks until
   the broadcast has completed, then marks the run active.
3. A Python callable register/unregister that starts after `Worker.run()` has
   marked the run active acquires `_registry_lock`, observes the active flag,
   raises `RuntimeError`, and leaves the registry unchanged.
4. `Worker.run()` clears `_py_register_active_run` under `_registry_lock` in a
   `finally` block after drain and cleanup.

This makes dynamic Python registration deterministic: it is supported after
children start between `run()` calls, but rejected while a run is actively
submitting or draining tasks.

This requires a generic C++ binding that can broadcast a control command to a
selected worker pool:

```python
_Worker.broadcast_control_all(worker_type, sub_cmd, cid, payload=None)
```

`worker_type` selects `SUB` versus `NEXT_LEVEL`; `sub_cmd` is
`CTRL_PY_REGISTER` or `CTRL_PY_UNREGISTER`. For register, `payload` is the
`cloudpickle`-serialized callable, passed as a Python buffer object. For
unregister, `payload` is absent. The binding owns shm creation, copying,
fan-out, and unlink when a payload is present, matching
`broadcast_register_all` for binary callables while avoiding four
near-identical Python-specific bindings.

The binding's error contract is subcommand-specific:

- `CTRL_PY_REGISTER` raises on any child error so the parent can pop the newly
  allocated cid before reporting failure.
- `CTRL_PY_UNREGISTER` returns per-child error messages for best-effort
  cleanup; the parent warns and releases its cid slot even if some children
  failed.

The existing `mailbox_mu_` must be held for each child round trip, just like
binary register. This serializes Python register/unregister against
`TASK_READY` dispatch on the same child.

Every child `CONTROL_REQUEST` handler, including existing chip-child handlers,
must reject unknown subcommands by writing `OFF_ERROR` and publishing
`CONTROL_DONE`. A misrouted Python control command must fail visibly, not ACK
as a successful no-op.

### Parent-Side Unregister

`Worker.unregister(cid)` uses the registered target type to select the
unregister route:

1. Hold `_registry_lock`.
2. Raise `KeyError` if `cid` is absent from the parent registry.
3. If the target is a Python callable and `_py_register_active_run` is true,
   raise `RuntimeError` before mutation.
4. If the Worker has not started child processes yet, pop the parent entry and
   return. Future children will inherit the already-removed registry.
5. For a post-start `ChipCallable`, keep the existing binary unregister path.
6. If the target is a Python callable and this Worker has started child
   processes, broadcast `CTRL_PY_UNREGISTER` to every Python-capable child
   group configured for this Worker, regardless of when the callable was
   originally registered.
7. Warn on per-child unregister errors, but pop the parent registry entry
   unconditionally so the cid slot becomes reusable.

Python callable unregister never cascades into `inner_worker.unregister(...)`.
For L4+ Worker children it removes only the parent-owned dispatch registry entry
inside `_child_worker_loop`, matching the `CTRL_PY_REGISTER` ownership rule.

Unregister is still best-effort, but reuse must self-heal. Before any
post-start `ChipCallable` registration for a cid that may have previously held a
Python callable, the parent must clear that cid from all Python-capable child
registries owned by the same Worker. This can reuse `CTRL_PY_UNREGISTER` as an
idempotent "clear Python dispatch entry" command. If the clear step fails during
registration, the new registration fails, the parent pops the newly allocated
cid, and no reverse rollback is attempted.

### SUB Child Handler

`_sub_worker_loop` currently handles `TASK_READY` and `SHUTDOWN`. It gains a
`CONTROL_REQUEST` branch:

- `CTRL_PY_REGISTER`: deserialize the callable and store `registry[cid] = fn`.
- `CTRL_PY_UNREGISTER`: `registry.pop(cid, None)`.
- Any unknown control subcommand: write `OFF_ERROR`, publish `CONTROL_DONE`,
  and leave the registry unchanged.

The loop is single-threaded, and parent-side `mailbox_mu_` serializes control
commands against task dispatch, so no child-side lock is required.

### Worker-Child Handler

`_child_worker_loop` already has a `CONTROL_REQUEST` branch for binary
callable cascade. It gains Python subcommands with different semantics:

- `CTRL_PY_REGISTER`: deserialize and store into the `registry` dict passed
  to `_child_worker_loop`.
- `CTRL_PY_UNREGISTER`: remove from that same `registry`.
- Existing binary `CTRL_REGISTER`: before cascading the `ChipCallable` into
  `inner_worker._register_at(...)`, remove `registry[cid]` from the
  Worker-child dispatch registry. This self-heals stale Python callable residue
  when a cid is reused as a `ChipCallable`.
- Any unknown control subcommand: write `OFF_ERROR`, publish `CONTROL_DONE`,
  and leave both the parent-owned dispatch registry and `inner_worker`
  unchanged.

This registry is the dispatch registry used when the parent submits a cid to
the Worker child. It is distinct from `inner_worker._callable_registry`.
Updating it makes a dynamically registered parent orch function visible to
the already-started Worker child.

The Python callable is not automatically cascaded into
`inner_worker._callable_registry`. Registering callables owned by an inner
Worker remains a separate operation on that Worker. This keeps cid ownership
local and avoids unexpected collisions with entries the inner Worker already
owns.

---

## 5. Failure Modes and Tests

### Failure Semantics

| Trigger | Handling |
| ------- | -------- |
| `cloudpickle` unavailable | Import fails at parent register time |
| Serializer cannot encode target | Parent pops cid and raises before IPC |
| Post-start no Python child group | Parent raises before cid allocation |
| cid space exhausted | Parent raises before parent mutation |
| Active `Worker.run()` register | Parent raises before cid allocation |
| Active `Worker.run()` unregister | Parent raises before parent mutation |
| Child cannot open shm | Child writes `OFF_ERROR`; parent raises |
| Child receives invalid cid | Child writes `OFF_ERROR`; parent raises |
| Child deserialization fails | Child writes `OFF_ERROR`; parent raises |
| Result is not callable | Child writes `OFF_ERROR`; parent raises |
| Unknown control subcommand | Child writes `OFF_ERROR`; parent raises |
| Some children succeed before another fails | Parent raises; no rollback |
| Unregister fails on some children | Parent warns and pops its registry |
| Cross-type cid reuse | New register clears or overwrites child residue |
| Child crashes during control | Parent may hang waiting for `CONTROL_DONE` |

No reverse rollback is attempted after partial register success. A successful
child may retain a registry entry for a cid the parent reports as failed.
Future cid reuse must overwrite it for Python registration, or clear it before
`ChipCallable` registration, matching the best-effort unregister contract.

If a child process crashes or stops polling its mailbox during
`CONTROL_REQUEST`, the parent may wait indefinitely for `CONTROL_DONE`. This is
the same liveness failure mode as existing mailbox control operations such as
`CTRL_PREPARE`, `CTRL_MALLOC`, and binary dynamic register. Adding timeout,
child liveness detection, and hierarchical recovery is out of scope for this
feature and should be handled as a broader control-plane reliability change.

### Concurrency

- Parent registry mutation stays under `_registry_lock`.
- The first `Worker.run()` marks `_py_register_active_run` before children
  startup, preventing a callable from being inserted after the startup registry
  snapshot but before the caller observes children as started.
- Each child mailbox round trip stays under `mailbox_mu_`.
- `register()` is synchronous. A caller that races `register()` and
  `Worker.run()` from different Python threads must still wait for
  `register()` to return before submitting the new cid.
- Child registry mutation is serialized by the mailbox state machine.
- The first implementation requires a quiescent Worker: dynamic Python
  callable registration while `Worker.run()` is actively executing is rejected
  with a clear error. Post-start registration between `run()` calls is the
  supported target.

### Test Plan

Keep the first implementation's tests focused on behavior and ownership, not on
format evolution:

- Unit test `cloudpickle` round trip for the supported callable shapes.
- Unit test that closures over serializable Python values work, and that
  specific known-unpickleable captures fail before cid visibility.
- Unit test that child-side deserialize and execute failures are reported
  through the normal mailbox error path.
- Unit test that Python register before children start uses the startup
  registry path and performs no control broadcast.
- Unit test that first-run startup is serialized against Python register, so a
  racing register cannot miss the startup registry snapshot.
- Unit test that post-start Python register rejects Workers with no SUB workers
  and no next-level Worker children.
- Unit test selected-pool routing: `worker_type=SUB` reaches only
  `sub_threads_`, and `worker_type=NEXT_LEVEL` reaches only
  `next_level_threads_`.
- L3 integration test: start an L3 Worker with SUB workers, run once to start
  children, dynamically register a Python sub callable, then
  `submit_sub(cid, ...)`.
- L4 integration test: start an L4 Worker with an L3 child, run once to start
  children, dynamically register an L3 orchestration callable on the L4 parent,
  then `submit_next_level(cid, ...)`.
- Unregister test: once children have started, Python callable unregister
  broadcasts `CTRL_PY_UNREGISTER`, pops the parent registry, and allows cid
  reuse regardless of whether the callable was registered before or after
  children started.
- Cross-type reuse test: stale Python dispatch residue from a failed
  best-effort unregister is cleared when the same cid is reused for a
  `ChipCallable`.
- Failure test: unsupported or non-serializable callable raises and releases the
  parent cid slot.

## Related

- [task-flow.md](task-flow.md) explains how `Callable`, `TaskArgs`, and
  `CallConfig` move through L3+ dispatch.
- [worker-manager.md](worker-manager.md) explains WorkerThread mailbox
  dispatch and forked Python child loops.
- [callable-ipc-dynamic-register.md](callable-ipc-dynamic-register.md)
  covers dynamic binary `ChipCallable` registration.
