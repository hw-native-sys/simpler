# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Owner-side ``Worker.create_buffer``: the child-topology gate and the identity it mints.

``_create_buffer_locked`` reads ``level``, the three child-shm lists, the buffer registry,
and the Worker's identity allocator, so these run against a Worker built with ``__new__`` —
no fork, no device, no ``init()``.
The gate is the point: an L3+ backing is resolved by a forked child mapping the shm by name, and
**every** kind of forked child can do that, so counting only chip and sub children refuses an L4
whose children are local L3 Workers.
"""

from __future__ import annotations

import threading
import time

import pytest
from simpler.buffer import (
    AddressSpace,
    BackendKind,
    BufferIdentityExhaustedError,
    LocalEndpointBufferIdentityAllocator,
    mint_owner_instance_id,
)
from simpler.worker import Worker, _Lifecycle, _NoBufferConsumerError, _SharedExclusiveLock


def _bare_worker(level: int, *, chip: int = 0, sub: int = 0, next_level: int = 0) -> Worker:
    w = Worker.__new__(Worker)
    w.level = level
    w._chip_shms = [object()] * chip
    w._sub_shms = [object()] * sub
    w._next_level_shms = [object()] * next_level
    w._registry_lock = threading.Lock()
    w._owner_instance_id = mint_owner_instance_id()
    w._buffer_identity_allocator = LocalEndpointBufferIdentityAllocator(w._owner_instance_id)
    w._buffers = {}
    w._direct_host_domains = {}
    w._hierarchical_start_mu = threading.Lock()
    w._hierarchical_start_cv = threading.Condition(w._hierarchical_start_mu)
    w._accepted_run_handles = set()
    w._abandoned_run_handles = []
    # Run admission is shared/exclusive now, not a plain Lock: production code calls
    # `.exclusive()` / `.shared()` on it, so the stand-in has to be the real type.
    w._submit_mu = _SharedExclusiveLock()
    w._chip_run_touched_identities = {}
    w._chip_import_registry = None
    w._reexport_by_source = {}
    w._worker = None
    return w


def _drain(w: Worker) -> None:
    for buffer in list(w._buffers.values()):
        buffer.close()
    w._buffers.clear()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chip": 1},
        {"sub": 1},
        # An L4 whose only children are local L3 Workers: they are forked processes that map a
        # POSIX_SHM backing by name exactly as a chip child does, so the buffer has a consumer.
        {"next_level": 1},
    ],
    ids=["chip", "sub", "next_level"],
)
def test_any_forked_child_admits_a_buffer(kwargs):
    w = _bare_worker(4, **kwargs)
    try:
        buffer = w._create_buffer_locked(64)
        assert buffer.nbytes == 64
        assert buffer.backend_kind == BackendKind.POSIX_SHM
        assert buffer.address_space == AddressSpace.HOST
    finally:
        _drain(w)


def test_childless_l3_plus_is_refused():
    w = _bare_worker(3)
    with pytest.raises(RuntimeError, match="at least one forked"):
        w._create_buffer_locked(64)
    assert not w._buffers


def test_the_childless_refusal_carries_its_own_type():
    # The remote L3 runner catches exactly this refusal to supply a session-scoped backing instead,
    # so it has to stay distinguishable from every other way create_buffer fails.
    w = _bare_worker(3)
    with pytest.raises(_NoBufferConsumerError):
        w._create_buffer_locked(64)


def test_l2_leaf_needs_no_child():
    # An L2 leaf materializes in-process, so it has nothing to hand the backing to and needs no child.
    w = _bare_worker(2)
    try:
        assert w._create_buffer_locked(64).nbytes == 64
    finally:
        _drain(w)


def test_rejects_nonpositive_size():
    w = _bare_worker(2)
    for nbytes in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            w._create_buffer_locked(nbytes)
    assert not w._buffers


def test_buffer_ids_are_unique_within_one_incarnation():
    w = _bare_worker(2)
    try:
        ids = [w._create_buffer_locked(32).identity.buffer_id for _ in range(4)]
        assert len(set(ids)) == 4
        assert all(b.identity.generation == 1 for b in w._buffers.values())
        # One incarnation, one nonce: every buffer this Worker owns shares it.
        nonces = {b.identity.owner_instance_id for b in w._buffers.values()}
        assert len(nonces) == 1
    finally:
        _drain(w)


def test_release_all_buffers_unlinks_and_empties_the_registry():
    w = _bare_worker(2)
    w._create_buffer_locked(32)
    w._create_buffer_locked(32)
    w._release_all_buffers()
    assert not w._buffers


def test_release_buffer_drops_only_its_own_entry():
    w = _bare_worker(2)
    try:
        keep = w._create_buffer_locked(32)
        drop = w._create_buffer_locked(32)
        w.release_buffer(drop)
        assert drop.shm is None
        assert list(w._buffers.values()) == [keep]
        # A second release of the same buffer is a no-op, not a KeyError on the registry.
        w.release_buffer(drop)
        assert list(w._buffers.values()) == [keep]
    finally:
        _drain(w)


def test_release_buffer_keeps_the_entry_when_close_fails():
    # Same discipline as _release_all_buffers: a close that fails leaves the entry behind so the
    # leak is still reported at close() instead of being dropped here.
    w = _bare_worker(2)
    bad = w._create_buffer_locked(32)
    bad_id = next(k for k, v in w._buffers.items() if v is bad)
    real_close = bad.close

    def boom():
        raise OSError("close failed")

    bad.close = boom  # type: ignore[method-assign]
    with pytest.raises(OSError, match="close failed"):
        w.release_buffer(bad)
    assert bad_id in w._buffers
    bad.close = real_close  # type: ignore[method-assign]
    _drain(w)


def test_release_all_buffers_reports_the_failure_and_keeps_the_entry():
    # Per-buffer best-effort: a failing close must neither strand the others nor be swallowed, and
    # its registry entry stays so the cleanup journal can retry it.
    w = _bare_worker(2)
    ok = w._create_buffer_locked(32)
    bad = w._create_buffer_locked(32)
    bad_id = next(k for k, v in w._buffers.items() if v is bad)

    def boom():
        raise OSError("close failed")

    bad.close = boom  # type: ignore[method-assign]
    with pytest.raises(OSError, match="close failed"):
        w._release_all_buffers()
    assert bad_id in w._buffers  # retryable
    assert ok.shm is None  # the healthy one was still released
    w._buffers.clear()
    bad.shm.close()  # type: ignore[union-attr]
    bad.shm.unlink()  # type: ignore[union-attr]


def _patch_level2_init(monkeypatch, worker: Worker) -> None:
    monkeypatch.setattr(worker, "_init_level2", lambda: None)


def test_pre_init_burn_retains_constructor_allocator(monkeypatch):
    worker = Worker(level=2)
    nonce = bytes(worker._owner_instance_id)
    allocator = worker._buffer_identity_allocator
    first = worker._burn_buffer_identity()
    assert bytes(first.owner_instance_id) == nonce
    assert int(first.buffer_id) == 1
    _patch_level2_init(monkeypatch, worker)
    worker.init()
    try:
        assert bytes(worker._owner_instance_id) == nonce
        assert worker._buffer_identity_allocator is allocator
        second = worker._burn_buffer_identity()
        assert bytes(second.owner_instance_id) == nonce
        assert int(second.buffer_id) == 2
    finally:
        worker.close()


def test_root_init_without_pre_init_burn_keeps_the_constructor_allocator(monkeypatch):
    worker = Worker(level=2)
    constructor = bytes(worker._owner_instance_id)
    allocator = worker._buffer_identity_allocator
    _patch_level2_init(monkeypatch, worker)
    worker.init()
    try:
        assert bytes(worker._owner_instance_id) == constructor
        assert worker._buffer_identity_allocator is allocator
        identity = worker._burn_buffer_identity()
        assert bytes(identity.owner_instance_id) == constructor
        assert int(identity.buffer_id) == 1
        assert int(identity.generation) == 1
    finally:
        worker.close()


def test_init_and_pre_init_burn_do_not_mix_nonces(monkeypatch):
    worker = Worker(level=2)
    constructor = bytes(worker._owner_instance_id)
    _patch_level2_init(monkeypatch, worker)
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []
    burned = []

    def burn() -> None:
        try:
            barrier.wait()
            burned.append(worker._burn_buffer_identity())
        except BaseException as exc:  # noqa: BLE001 -- the test asserts the collected failure
            errors.append(exc)

    def start() -> None:
        try:
            barrier.wait()
            worker.init()
        except BaseException as exc:  # noqa: BLE001 -- the test asserts the collected failure
            errors.append(exc)

    threads = [threading.Thread(target=burn), threading.Thread(target=start)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    try:
        assert errors == []
        identity = burned[0]
        assert bytes(identity.owner_instance_id) == constructor
        nxt = worker._burn_buffer_identity()
        assert bytes(nxt.owner_instance_id) == constructor
        assert int(nxt.buffer_id) == int(identity.buffer_id) + 1
        assert int(nxt.buffer_id) != int(identity.buffer_id)
    finally:
        if worker._lifecycle is _Lifecycle.READY:
            worker.close()


def test_add_worker_child_keeps_constructor_identity_across_init():
    parent = Worker(level=4)
    child = Worker(level=3)
    try:
        child_nonce = bytes(child._owner_instance_id)
        allocator = child._buffer_identity_allocator
        first = child._burn_buffer_identity()
        assert bytes(first.owner_instance_id) == child_nonce
        assert int(first.buffer_id) == 1
        parent.add_worker(child)
        assert parent._next_level_workers == [child]
        assert bytes(parent._owner_instance_id) != child_nonce
        child._init_hierarchical = lambda: None  # type: ignore[method-assign]
        child._start_hierarchical = lambda: None  # type: ignore[method-assign]
        child.init(_startup_deadline=time.monotonic() + 30)
        assert bytes(child._owner_instance_id) == child_nonce
        assert child._buffer_identity_allocator is allocator
        second = child._burn_buffer_identity()
        assert bytes(second.owner_instance_id) == child_nonce
        assert int(second.buffer_id) == 2
    finally:
        for worker in (child, parent):
            if worker._lifecycle is _Lifecycle.READY:
                try:
                    worker.close()
                except BaseException:  # noqa: BLE001 -- no-op init has no native tree to tear down
                    worker._lifecycle = _Lifecycle.NEW
            elif worker._lifecycle is not _Lifecycle.NEW:
                worker._lifecycle = _Lifecycle.NEW


class _CountingMalloc:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def malloc(self, worker_id: int, nbytes: int) -> int:
        self.calls += 1
        if self.fail:
            raise RuntimeError("native malloc failed")
        return 0x1000 + int(worker_id)


def _ready_l3() -> Worker:
    worker = Worker(level=3, num_sub_workers=0, platform="a2a3sim", runtime="tensormap_and_ringbuffer")
    worker._lifecycle = _Lifecycle.READY
    worker._chip_shms = [object()]
    return worker


def test_alloc_child_tensor_exhaustion_does_not_malloc():
    worker = _ready_l3()
    native = _CountingMalloc()
    worker._worker = native  # type: ignore[assignment]
    worker._buffer_identity_allocator._next_buffer_id = 1 << 64
    with pytest.raises(BufferIdentityExhaustedError):
        worker.alloc_child_tensor(0, (4,), 0)
    assert native.calls == 0
    assert len(worker._child_alloc) == 0


def test_alloc_child_tensor_malloc_failure_consumes_the_burned_id():
    worker = _ready_l3()
    native = _CountingMalloc(fail=True)
    worker._worker = native  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="native malloc failed"):
        worker.alloc_child_tensor(0, (4,), 0)
    assert native.calls == 1
    assert len(worker._child_alloc) == 0
    nxt = worker._burn_buffer_identity()
    assert int(nxt.buffer_id) == 2
