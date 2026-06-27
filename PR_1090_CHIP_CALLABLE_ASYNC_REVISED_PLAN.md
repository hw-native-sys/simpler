# PR 1090 Revised Plan

## 1. L3 DAG Run Async

Goal:

```python
dag_h = l3_worker.run_async(orch_fn, args, config)
timing = dag_h.wait()
```

This API is async relative to the caller. The DAG itself still executes with
the current synchronous `Worker.run(orch_fn)` semantics:

```text
orch_fn calls orch.submit_next_level(...)
current scheduler dispatches READY tasks
current WorkerThread waits task completion
current drain waits the whole DAG
```

Do not convert `orch.submit_next_level(...)` to child `RUN_ASYNC` in this step.
That is a separate lower-layer scheduler/completion redesign.

Use one per-worker DAG run queue so sync and async DAG runs share ordering:

```python
@dataclass
class DagRunState:
    cv: Condition
    completed: bool = False
    result: RunTiming | None = None
    error: BaseException | None = None


@dataclass
class DagRunRequest:
    orch_fn: Callable
    args: Any
    config: CallConfig
    state: DagRunState
```

Worker initialization:

```python
def _start_dag_run_lane(self):
    if self._dag_run_thread is not None:
        return

    self._dag_run_queue = Queue()
    self._dag_run_thread = Thread(
        target=self._dag_run_thread_loop,
        name="simpler-l3-dag-run",
        daemon=True,
    )
    self._dag_run_thread.start()
```

L3 public async DAG run:

```python
def run_async(self, orch_fn, args=None, config=None) -> DagRunHandle:
    assert self.level >= 3
    assert self._initialized

    state = DagRunState()
    req = DagRunRequest(
        orch_fn=orch_fn,
        args=args,
        config=copy_call_config(config),
        state=state,
    )

    self._dag_run_queue.put(req)
    return DagRunHandle(state)
```

L3 public sync DAG run:

```python
def run(self, orch_fn, args=None, config=None) -> RunTiming:
    assert self.level >= 3

    return self.run_async(orch_fn, args, config).wait()
```

The DAG lane calls the existing synchronous DAG implementation:

```python
def _dag_run_thread_loop(self):
    while True:
        req = self._dag_run_queue.get()
        if req is None:
            break

        try:
            timing = self._run_dag_sync_impl(
                req.orch_fn,
                req.args,
                req.config,
            )
            complete_success(req.state, timing)
        except BaseException as exc:
            complete_error(req.state, exc)
```

Move the current L3 body of `Worker.run(...)` into `_run_dag_sync_impl(...)`:

```python
def _run_dag_sync_impl(self, orch_fn, args, config) -> RunTiming:
    self._start_hierarchical()
    self._orch._clear_error()
    self._orch._scope_begin()

    t_start = time.perf_counter_ns()
    try:
        orch_fn(self._orch, args, config)
    finally:
        self._orch._scope_end()
        self._orch._drain()
        self._execute_pending_domain_releases()
        self._release_all_live_domains()

    return RunTiming(time.perf_counter_ns() - t_start, 0)
```

Handle wait:

```python
class DagRunHandle:
    @property
    def completed(self) -> bool:
        return self._state.completed

    def wait(self) -> RunTiming:
        with self._state.cv:
            while not self._state.completed:
                self._state.cv.wait()

            if self._state.error is not None:
                raise self._state.error

            return self._state.result
```

Ordering requirement:

```python
h1 = worker.run_async(orch_fn_1)
t2 = worker.run(orch_fn_2)

# Required:
# orch_fn_1 completes before orch_fn_2 starts.
```

## 2. Public API Shape

Use level-specific public semantics.

L2 direct worker APIs:

```python
h = l2.register(chip_callable)
rh = l2.register_async(chip_callable)

run_h = l2.run_async(h, args)
timing = run_h.wait()

timing = l2.run(h, args)

unreg_h = l2.unregister_async(h)
unreg_h.wait()

l2.unregister(h)
```

L3 worker APIs:

```python
h = l3.register(chip_callable)
rh = l3.register_async(chip_callable)

dag_h = l3.run_async(orch_fn)
timing = dag_h.wait()

timing = l3.run(orch_fn)

unreg_h = l3.unregister_async(h)
unreg_h.wait()

l3.unregister(h)
```

Do not expose an L3 public direct chip-run API:

```python
# Not part of the public L3 API.
l3.run_chip_callable_async(...)
l3.run_chip_callable_sync(...)
```

L3 execution should go through `run(orch_fn)` or `run_async(orch_fn)`.

PR scope rule:

```text
register_async(target):
  if target is not ChipCallable:
    raise TypeError

unregister_async(handle):
  if handle is not a ChipCallable handle:
    raise TypeError
```

Async register/unregister for Python callables or `RemoteCallable` is a future
extension, not part of this PR.

`register_async(...)` and `unregister_async(...)` support `ChipCallable`
targets/handles only. Non-chip callable async registration or unregister must
fail explicitly:

```python
def register_async(self, target, *, workers=None):
    if not isinstance(target, ChipCallable):
        raise TypeError("Worker.register_async only supports ChipCallable")

    return self._register_chip_async(target, workers=workers)


def unregister_async(self, handle):
    if not is_chip_callable_handle(handle):
        raise TypeError("Worker.unregister_async only supports ChipCallable")

    return self._unregister_chip_async(handle)
```

Generic `register(...)` remains synchronous and supports existing callable
kinds. For a `ChipCallable`, it delegates to `register_async(...).wait()`:

```python
def register(self, target, *, workers=None):
    if isinstance(target, ChipCallable):
        return self.register_async(
            target,
            workers=workers,
        ).wait()

    return self._register_non_chip_sync(target, workers=workers)
```

Async register implementation:

```python
def _register_chip_async(
    self,
    target: ChipCallable,
    *,
    workers: list[int] | None = None,
) -> RegisterHandle:
    if not isinstance(target, ChipCallable):
        raise TypeError("expected ChipCallable")

    reg = build_callable_registration(self, target, workers=workers)

    with self._registry_lock:
        handle, is_new = self._install_registration_locked(reg)

    if not self._initialized:
        return completed_register_handle(handle)

    if self.level == 2:
        return self._l2_submit_register_async(handle, target, is_new=is_new)

    return self._l3_submit_register_async(handle, target, is_new=is_new)
```

L2 async run:

```python
def run_async(
    self,
    handle: CallableHandle,
    args=None,
    config=None,
) -> RunHandle:
    assert self._initialized
    assert self.level == 2

    return self._l2_submit_run_async(handle, args, config)
```

L2 sync run:

```python
def _run_l2_sync(self, handle, args=None, config=None):
    return self.run_async(handle, args, config).wait()
```

Generic `unregister(...)` remains synchronous and delegates for chip callables:

```python
def unregister(self, handle):
    if is_chip_callable_handle(handle):
        return self.unregister_async(handle).wait()

    return self._unregister_non_chip_sync(handle)
```

## 3. L2 Direct Worker Lanes

L2 direct usage should have the same lane shape as an L3 chip child:

```text
run lane:
  serial chip runs

register lane:
  async chip prepare/register
```

L2 state:

```python
class Worker:
    _l2_run_queue: Queue[LocalRunRequest]
    _l2_register_queue: Queue[LocalRegisterRequest]

    _l2_run_thread: Thread
    _l2_register_thread: Thread

    _slot_inflight: dict[int, int]
    _slot_tombstoned: set[int]
    _slot_pending_unregister: dict[int, UnregisterState]
```

Start lanes during L2 init:

```python
def _init_l2(self):
    self._chip_worker = ChipWorker(...)
    self._chip_worker.init(...)

    self._start_l2_lanes()
    self._initialized = True


def _start_l2_lanes(self):
    self._l2_run_queue = Queue()
    self._l2_register_queue = Queue()

    self._l2_run_thread = Thread(target=self._l2_run_loop, daemon=True)
    self._l2_register_thread = Thread(
        target=self._l2_register_loop,
        daemon=True,
    )

    self._l2_run_thread.start()
    self._l2_register_thread.start()
```

L2 async register submit:

```python
def _l2_submit_register_async(self, handle, target, *, is_new):
    if not is_new:
        return completed_register_handle(handle)

    state = RegisterState()

    callable_bytes = bytes_from_chip_callable(target)
    with self._registry_lock:
        slot_id = self._identity_registry[handle.digest].slot_id

    self._l2_register_queue.put(LocalRegisterRequest(
        slot_id=slot_id,
        digest=handle.digest,
        callable_bytes=callable_bytes,
        state=state,
    ))

    return RegisterHandle(state, result=handle)
```

L2 register lane:

```python
def _l2_register_loop(self):
    while True:
        req = self._l2_register_queue.get()
        if req is None:
            break

        try:
            callable_obj = ChipCallable.from_bytes(req.callable_bytes)
            validate_digest(callable_obj, req.digest)

            self._chip_worker._prepare_callable_at_slot(
                req.slot_id,
                callable_obj,
            )

            complete_success(req.state)
        except BaseException as exc:
            rollback_parent_registration(req.digest)
            complete_error(req.state, exc)
```

L2 async run submit:

```python
def _l2_submit_run_async(self, handle, args, config):
    with self._registry_lock:
        slot = self._resolve_handle_locked(
            handle,
            expected_namespace="LOCAL_CHIP",
        )

        if slot.slot_id in self._slot_tombstoned:
            raise KeyError("callable handle is pending unregister")

        self._slot_inflight[slot.slot_id] += 1

    run_state = RunState()
    self._l2_run_queue.put(LocalRunRequest(
        slot_id=slot.slot_id,
        args=copy_run_args(args),
        config=copy_call_config(config),
        state=run_state,
    ))

    return RunHandle(run_state)
```

L2 run lane:

```python
def _l2_run_loop(self):
    while True:
        req = self._l2_run_queue.get()
        if req is None:
            break

        try:
            timing = self._chip_worker._run_slot(
                req.slot_id,
                req.args,
                req.config,
            )
            complete_success(req.state, timing)
        except BaseException as exc:
            complete_error(req.state, exc)
        finally:
            self._release_slot_inflight(req.slot_id)
```

L2 sync APIs use the same queues:

```python
def register(self, target):
    if self.level == 2 and isinstance(target, ChipCallable):
        return self.register_async(target).wait()


def run(self, handle, args=None, config=None):
    if self.level == 2:
        return self.run_async(handle, args, config).wait()
```

## 4. Nonblocking Tombstone And Deferred Free

Unregister submit must be nonblocking:

```python
unreg_h = worker.unregister_async(handle)
```

Return means:

```text
public handle has been tombstoned
new runs using that handle are rejected
native unregister/free may still be pending
unreg_h.wait() waits for actual native cleanup
```

Free condition:

```text
free when tombstoned(slot_id) and inflight(slot_id) == 0
never free merely because inflight becomes zero
```

Shared state:

```python
@dataclass
class SlotLifetime:
    slot_id: int
    digest: bytes
    ref_count: int
    tombstoned: bool = False
    inflight: int = 0
    unregister_state: UnregisterState | None = None
```

Run submit holds an in-flight reference before enqueue:

```python
def hold_slot_for_run(handle) -> tuple[int, bytes]:
    with registry_lock:
        slot = resolve_live_chip_handle(handle)

        if slot.tombstoned:
            raise KeyError("callable handle is pending unregister")

        slot.inflight += 1
        return slot.slot_id, slot.digest
```

Run completion releases that reference:

```python
def release_slot_after_run(slot_id):
    cleanup_state = None
    cleanup_digest = b""

    with registry_lock:
        slot = slot_by_id[slot_id]
        slot.inflight -= 1

        if slot.inflight == 0 and slot.tombstoned:
            cleanup_state = slot.unregister_state
            cleanup_digest = slot.digest

    if cleanup_state is not None:
        native_unregister_and_finish(slot_id, cleanup_digest, cleanup_state)
```

Async unregister:

```python
def unregister_async(handle) -> UnregisterHandle:
    cleanup_slot_id = -1
    cleanup_digest = b""
    cleanup_state = None

    with registry_lock:
        slot = resolve_live_chip_handle(handle)

        remove_public_handle(handle)
        slot.ref_count -= 1

        state = UnregisterState()

        if slot.ref_count > 0:
            complete_success(state)
            return UnregisterHandle(state)

        slot.tombstoned = True
        slot.unregister_state = state
        pending_unregister_cids.add(slot.slot_id)

        if slot.inflight == 0:
            cleanup_slot_id = slot.slot_id
            cleanup_digest = slot.digest
            cleanup_state = state

    if cleanup_state is not None:
        native_unregister_and_finish(
            cleanup_slot_id,
            cleanup_digest,
            cleanup_state,
        )

    return UnregisterHandle(state)
```

Local native cleanup:

```python
def native_unregister_and_finish(slot_id, digest, state):
    try:
        chip_worker._unregister_slot(slot_id)
        complete_success(state)
    except BaseException as exc:
        mark_cleanup_uncertain(digest)
        complete_error(state, exc)
    finally:
        with registry_lock:
            callable_registry.pop(slot_id, None)
            identity_registry.pop(digest, None)
            pending_unregister_cids.discard(slot_id)
            slot_lifetime.pop(slot_id, None)
```

L3 parent remote cleanup:

```python
def _l3_submit_unregister_async(slot) -> UnregisterHandle:
    parent_state = slot.unregister_state
    remote_handles = []

    for child in chip_children:
        remote_handles.append(
            child.control_unregister_async(slot.digest)
        )

    wait_thread = Thread(
        target=_wait_remote_unregisters,
        args=(slot, remote_handles, parent_state),
        daemon=True,
    )
    wait_thread.start()

    return UnregisterHandle(parent_state)
```

L3 child unregister submit:

```python
def handle_unregister_async(digest):
    state = UnregisterState()
    cleanup_cid = None

    with registry_cv:
        cid = identity_table.get(digest)
        if cid is None:
            complete_success(state)
            return make_remote_handle(state)

        remove_identity_mapping(digest)
        tombstoned_cids.add(cid)

        if inflight_cids.get(cid, 0) == 0:
            cleanup_cid = cid
        else:
            deferred_unregister[cid] = state

    if cleanup_cid is not None:
        child_native_unregister_and_finish(cleanup_cid, digest, state)

    return make_remote_handle(state)
```

L3 child run submit stores `cid`, not only `digest`:

```python
def handle_run_async(digest, args_blob, config):
    state = RunState()

    with registry_cv:
        cid = identity_table.get(digest)
        if cid is None or cid in tombstoned_cids:
            complete_error(state, KeyError("callable not live"))
            return make_remote_handle(state)

        inflight_cids[cid] += 1

    run_queue.put(ChildRunRequest(
        cid=cid,
        args_blob=copy(args_blob),
        config=config,
        state=state,
    ))

    return make_remote_handle(state)
```

L3 child run completion triggers deferred cleanup:

```python
def child_release_inflight(cid):
    cleanup_digest = b""
    cleanup_state = None

    with registry_cv:
        inflight_cids[cid] -= 1
        if inflight_cids[cid] == 0:
            del inflight_cids[cid]

            if cid in tombstoned_cids:
                state = deferred_unregister.pop(cid, None)
                if state is not None:
                    cleanup_digest = digest_by_cid[cid]
                    cleanup_state = state

    if cleanup_state is not None:
        child_native_unregister_and_finish(cid, cleanup_digest, cleanup_state)
```

L3 child native cleanup:

```python
def child_native_unregister_and_finish(cid, digest, state):
    try:
        chip_worker._unregister_slot(cid)
        complete_success(state)
    except BaseException as exc:
        complete_error(state, exc)
    finally:
        with registry_cv:
            registry.pop(cid, None)
            identity_table.pop(digest, None)
            identity_refs.pop(digest, None)
            prepared.discard(cid)
            tombstoned_cids.discard(cid)
            digest_by_cid.pop(cid, None)
```

Unregister wait is a resource cleanup barrier:

```python
run_h = l2.run_async(handle, args, config)
unreg_h = l2.unregister_async(handle)

unreg_h.wait()
```

`unreg_h.wait()` guarantees native unregister/free completed after all accepted
runs that held the slot stopped using it. It does not return run timing or
rethrow the run error. Call `run_h.wait()` when the caller needs the run result:

```python
timing = run_h.wait()
unreg_h.wait()
```
