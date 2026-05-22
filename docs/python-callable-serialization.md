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

- Dynamic registration of `ChipCallable`; that protocol is covered by the
  binary callable design. This document only adds the Python-residue cleanup
  hook that the existing `ChipCallable` register implementation needs when a
  shared cid is reused across target types.
- Cross-host or cross-Python-version serialization.
- Recovering a child process that crashes or wedges while a mailbox control
  request is in flight. This design specifies timeout reporting for Python
  callable broadcasts, but rebuilding the child process tree is broader
  control-plane reliability work.
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

1. The parent serializes `target`.
2. The parent allocates a cid and stores `target` in its registry.
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

This design assumes child processes are forked from the same Python process and
therefore share the same Python major/minor version, installed package set, and
`cloudpickle` runtime. If a future startup mode uses `spawn` or independently
provisioned interpreters, dynamic Python callable registration is supported only
when the child environment is version-compatible with the parent and can import
the callable's dependencies.

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
before it releases the Python object reference or fans out worker threads. The
binding must not retain a raw pointer into the Python buffer after returning or
after releasing the GIL for control fan-out.

The shm starts with a minimal Python-callable payload header followed by the
exact bytes returned by `cloudpickle.dumps(target)`:

| Field | Size | Value |
| ----- | ---- | ----- |
| magic | 4 bytes | `SPYC` |
| version | 1 byte | `1` |
| serializer | 1 byte | `1` for `cloudpickle` |
| flags | 2 bytes | reserved, must be zero |
| payload_size | 8 bytes | little-endian unsigned byte count |

The first implementation accepts only `(magic="SPYC", version=1,
serializer=1, flags=0)`. Unknown magic, version, serializer, non-zero flags,
size mismatch, malformed bytes, or incompatible pickle data fail through the
normal mailbox error field.

### Child Deserialization

Each recipient child:

1. Opens the shm by name.
2. Validates the payload header.
3. Copies the payload region into `bytes`.
4. Verifies that `cid` is in `[0, MAX_REGISTERED_CALLABLE_IDS)`.
5. Deserializes the callable with `cloudpickle.loads(payload_bytes)`.
6. Verifies that the result is callable.
7. Installs it into the child's local registry under the requested cid.
8. Closes the shm and acknowledges `CONTROL_DONE`.

The child intentionally copies the payload region before deserializing it.
This avoids coupling `cloudpickle.loads(...)` to the lifetime rules of an
active `SharedMemory.buf` memoryview and keeps shm close/unlink behavior simple.

For cid reuse after partial unregister failures, Python registration should
overwrite `registry[cid]` in the child. The parent only allocates free cids
from its own registry, so an existing child entry at the same cid is residue
from a prior best-effort failure and should be replaced.

Because Python callables and `ChipCallable` objects share one cid space, the
same cleanup rule also applies when a cid is reused across target types. A
post-start `ChipCallable` registration performed after this feature lands must
clear any stale Python dispatch entry for the same cid from Python-capable
children owned by the same Worker before the cid is reported usable. This is a
v2 integration hook on the existing `ChipCallable` register implementation, not
a new binary payload protocol. Otherwise a failed Python unregister followed by
`ChipCallable` reuse could leave a Worker-child dispatch loop resolving the old
Python callable.

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

1. Reject non-callable Python targets.
2. If the first hierarchical startup is in progress, wait for that startup to
   either complete or fail without holding `_registry_lock`. A registration
   must not return through the startup path after the fork-time registry
   snapshot has already been taken.
3. If this Worker has not started child processes, hold `_registry_lock`,
   allocate the smallest free cid, insert
   `self._callable_registry[cid] = target`, and return the cid; future children
   will inherit the registry when they start.
4. If no configured Python-capable child group exists, raise `RuntimeError`.
5. Serialize the target into a bytes blob with `cloudpickle.dumps(...)`.
6. Hold `_registry_lock`, allocate the smallest free cid, insert
   `self._callable_registry[cid] = target`, and release `_registry_lock`.
7. Broadcast `CTRL_PY_REGISTER` to required Python-capable worker groups.
8. On any failure, reacquire `_registry_lock`, remove the parent registry entry
   if it still points at this target, and raise.
9. Return cid on success.

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

The startup race is handled by a one-time hierarchical startup state, not by a
run-wide quiescent guard:

- `_hierarchical_start_state` is protected by a dedicated
  `_hierarchical_start_mu` / `_hierarchical_start_cv`, separate from
  `_registry_lock`.
- Startup begins as `not_started`, moves to `starting` before
  `_start_hierarchical()` takes the registry snapshot, and moves to `started`
  only after child mailboxes are registered with the C++ Worker.
- A Python callable register/unregister that observes `starting` waits on a
  condition variable without holding `_registry_lock`. After startup succeeds,
  it uses the post-start control path; after startup fails, it raises.
- `_start_hierarchical()` snapshots `self._callable_registry` while holding
  `_registry_lock`, then forks children from that immutable snapshot. It must
  not hold `_registry_lock` across `os.fork()`.

Once children have started, dynamic Python registration is allowed while
`Worker.run()` is actively submitting or draining tasks. The operation is still
synchronous: the caller must wait for `register()` to return before submitting
the new cid. Per-child `mailbox_mu_` serialization orders each
`CTRL_PY_REGISTER` / `CTRL_PY_UNREGISTER` round trip against any in-flight
`TASK_READY` on that same child mailbox.

`_registry_lock` protects parent-side cid allocation and registry mutation
only. It is not held while waiting for child ACKs from
`broadcast_control_all`.

This requires a generic C++ binding that can broadcast a control command to a
selected worker pool:

```python
_Worker.broadcast_control_all(worker_type, sub_cmd, cid, payload=None,
                              timeout_s=None)
```

`worker_type` selects `SUB` versus `NEXT_LEVEL`; `sub_cmd` is
`CTRL_PY_REGISTER` or `CTRL_PY_UNREGISTER`. For register, `payload` is the
`cloudpickle`-serialized callable, passed as a Python buffer object. For
unregister, `payload` is absent. The binding owns shm creation, copying,
fan-out, and unlink when a payload is present, matching
`broadcast_register_all` for binary callables while avoiding four
near-identical Python-specific bindings.

For a selected worker pool, fan-out is parallel: C++ starts one worker thread
per target child, each round trip holds that child's `mailbox_mu_`, and the
binding waits for every child to publish `CONTROL_DONE` before returning the
per-child results. Latency is bounded by the slowest child round trip, not by
the sum of all child round trips.

`timeout_s` is optional. When set, each child round trip that does not publish
`CONTROL_DONE` before the deadline returns a failed result with a timeout error
message. The timeout does not repair the wedged child or reclaim a mailbox
that is still owned by a stuck control command; it only bounds the caller's
wait and makes the partial failure visible to Python policy code.

The binding always returns structured per-child results. It does not switch
between "raise" and "return errors" based on `sub_cmd`. Python decides whether
those results are strict or best-effort:

```text
ControlResult(worker_type, worker_index, ok, error_message)
```

- `Worker.register()` treats any failed `CTRL_PY_REGISTER` result as strict:
  it removes the new parent registry entry and raises.
- `Worker.unregister()` treats failed `CTRL_PY_UNREGISTER` results as
  best-effort: it warns, then releases its parent cid slot after the broadcast
  has returned.
- The cross-type reuse hook treats failed Python-residue cleanup as strict: it
  fails the `ChipCallable` registration before starting binary
  `CTRL_REGISTER`.

The existing `mailbox_mu_` must be held for each child round trip, just like
binary register. This serializes Python register/unregister against
`TASK_READY` dispatch on the same child.

Every child `CONTROL_REQUEST` handler, including existing chip-child handlers,
must reject unknown subcommands by writing `OFF_ERROR` and publishing
`CONTROL_DONE`. A misrouted Python control command must fail visibly, not ACK
as a successful no-op.

### Cross-Type Reuse Hook

The existing post-start `ChipCallable` register path keeps its binary payload
protocol, but gains one v2 hook before reporting a reused cid as usable:

1. After allocating a cid for a `ChipCallable`, check whether that cid may have
   held a Python callable in this Worker lifetime.
2. If so, broadcast `CTRL_PY_UNREGISTER` to every Python-capable child group
   owned by this Worker as an idempotent clear operation.
3. If that clear operation reports any child error, fail the `ChipCallable`
   registration, remove the new parent registry entry, and do not start the
   binary `CTRL_REGISTER` broadcast.
4. If the clear succeeds, continue through the existing binary
   `broadcast_register_all` path.

This hook is needed only for Python-capable child registries. Chip children
continue to rely on the existing binary self-heal before
`prepare_callable_from_blob`.

### Parent-Side Unregister

`Worker.unregister(cid)` uses the registered target type to select the
unregister route:

1. If the first hierarchical startup is in progress, wait for it to complete
   without holding `_registry_lock`.
2. Hold `_registry_lock`.
3. Raise `KeyError` if `cid` is absent from the parent registry or already has
   an unregister in progress.
4. If the Worker has not started child processes yet, pop the parent entry and
   return. Future children will inherit the already-removed registry.
5. Mark `cid` as pending unregister, then release `_registry_lock`. A pending
   cid remains unavailable for reuse until the broadcast finishes.
6. For a post-start `ChipCallable`, keep the existing binary unregister path.
7. If the target is a Python callable and this Worker has started child
   processes, broadcast `CTRL_PY_UNREGISTER` to every Python-capable child
   group configured for this Worker, regardless of when the callable was
   originally registered.
8. Warn on per-child unregister errors. Reacquire `_registry_lock`, pop the
   parent registry entry unconditionally, clear the pending marker, and make
   the cid slot reusable.

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

Registry ownership in a Worker-child process is:

- Parent `CTRL_PY_REGISTER`: mutates the parent dispatch `registry`, read by
  `_child_worker_loop`; does not cascade.
- Parent `CTRL_PY_UNREGISTER`: removes from the parent dispatch `registry`;
  does not cascade.
- Parent binary `CTRL_REGISTER`: mutates `inner_worker._callable_registry` and
  cascades through the inner Worker's own register route.
- Parent binary `CTRL_UNREGISTER`: mutates `inner_worker._callable_registry`
  and cascades through the inner Worker's own unregister route.
- Inner Worker register/unregister: mutates `inner_worker._callable_registry`
  and is owned by the inner Worker.

The parent dispatch registry and `inner_worker._callable_registry` may contain
the same numeric cid for different owners. A parent Python unregister must not
call `inner_worker.unregister(cid)`, because that could delete a callable that
belongs to the inner Worker. Cross-type cleanup before parent `ChipCallable`
reuse clears stale Python entries from SUB registries and from Worker-child
parent dispatch registries. It does not clear
`inner_worker._callable_registry`; the binary register then cascades into
`inner_worker` through the normal binary route.

---

## 5. Failure Modes and Tests

### Failure Semantics

| Trigger | Handling |
| ------- | -------- |
| `cloudpickle` unavailable | Import fails at parent register time |
| Serializer cannot encode target | Parent raises before cid allocation |
| Post-start no Python child group | Parent raises before cid allocation |
| cid space exhausted | Parent raises before parent mutation |
| Startup race | Wait, then use post-start route |
| Duplicate unregister for same cid | Raise before second broadcast |
| Child cannot open shm | Child writes `OFF_ERROR`; parent raises |
| Child receives invalid cid | Child writes `OFF_ERROR`; parent raises |
| Child deserialization fails | Child writes `OFF_ERROR`; parent raises |
| Result is not callable | Child writes `OFF_ERROR`; parent raises |
| Unknown control subcommand | Child writes `OFF_ERROR`; parent raises |
| Some children succeed before another fails | Parent raises; no rollback |
| Unregister fails on some children | Parent warns and pops its registry |
| Cross-type cid reuse | New register clears or overwrites child residue |
| Child `cloudpickle.loads` times out | Failed child result |
| Child crashes during control | Timeout result, or hang if unset |

No reverse rollback is attempted after partial register success. A successful
child may retain a registry entry for a cid the parent reports as failed.
Future cid reuse must overwrite it for Python registration, or clear it before
`ChipCallable` registration, matching the best-effort unregister contract.

Python deserialization has a larger liveness surface than binary callable
prepare. `cloudpickle.loads(...)` may import modules or run user-defined object
reconstruction hooks, and that code can block, spin, or wedge the child before
it writes `CONTROL_DONE`. For Python callable broadcasts, callers should pass a
finite `timeout_s` so `broadcast_control_all` can return a failed per-child
result instead of waiting forever. Timeout does not make the child healthy; it
only lets `Worker.register()` fail visibly and lets best-effort cleanup report
which child did not respond. Child liveness detection, process replacement, and
hierarchical recovery remain out of scope for this feature.

### Concurrency

- Parent registry mutation stays under `_registry_lock`.
- The first `Worker.run()` marks hierarchical startup as `starting` before
  taking the startup registry snapshot. Concurrent register/unregister callers
  wait for startup to finish, then use the correct post-start route.
- `_registry_lock` is released before any broadcast waits for child ACKs. The
  parent registry entry, plus the pending-unregister marker for unregister,
  keeps the cid unavailable for reuse while the IPC operation is in flight.
- Each child mailbox round trip stays under `mailbox_mu_`, so post-start Python
  register/unregister can run during `Worker.run()` and will serialize against
  `TASK_READY` on each recipient mailbox.
- `register()` is synchronous. A caller that races `register()` and
  `Worker.run()` from different Python threads must still wait for
  `register()` to return before submitting the new cid.
- Child registry mutation is serialized by the mailbox state machine.
- `unregister()` is synchronous from the caller's perspective. The user remains
  responsible for not unregistering a cid with outstanding submitted work.

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
- Unit test that post-start Python register during an active `Worker.run()`
  succeeds after the relevant child mailbox reaches a safe control point.
- Unit test that unregister keeps the cid unavailable for reuse until its
  broadcast has completed, even though `_registry_lock` is not held across the
  broadcast.
- Unit test that post-start Python register rejects Workers with no SUB workers
  and no next-level Worker children.
- Unit test selected-pool routing: `worker_type=SUB` reaches only
  `sub_threads_`, and `worker_type=NEXT_LEVEL` reaches only
  `next_level_threads_`.
- Unit test that `broadcast_control_all` returns the same structured
  per-child result shape for register and unregister commands.
- Unit test that `broadcast_control_all(timeout_s=...)` reports a timed-out
  child as a failed per-child result without blocking indefinitely.
- L3 integration test: start an L3 Worker with SUB workers, run once to start
  children, dynamically register a Python sub callable, then
  `submit_sub(cid, ...)`.
- L4 integration test: start an L4 Worker with an L3 child, run once to start
  children, dynamically register an L3 orchestration callable on the L4 parent,
  then `submit_next_level(cid, ...)`.
- Unregister test: once children have started, Python callable unregister
  broadcasts `CTRL_PY_UNREGISTER`, pops the parent registry, and allows cid
  reuse only after `unregister()` returns, regardless of whether the callable
  was registered before or after children started.
- Cross-type reuse test: stale Python dispatch residue from a failed
  best-effort unregister is cleared when the same cid is reused for a
  `ChipCallable`.
- Failure test: unsupported or non-serializable callable raises without
  consuming a parent cid slot.

## Related

- [task-flow.md](task-flow.md) explains how `Callable`, `TaskArgs`, and
  `CallConfig` move through L3+ dispatch.
- [worker-manager.md](worker-manager.md) explains WorkerThread mailbox
  dispatch and forked Python child loops.
- [callable-ipc-dynamic-register.md](callable-ipc-dynamic-register.md)
  covers dynamic binary `ChipCallable` registration.
