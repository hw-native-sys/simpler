# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``Worker.free``'s in-flight refusal, and the lock it must not take twice.

The refusal scans the same run sets ``release_buffer`` does, and ``_submit_mu`` is what keeps that
scan from landing mid-callback with a half-populated touched set. But an ``orch.free`` inside a
graph callback reaches it on a thread that *already* holds that lock for the whole callback, and it
is not reentrant: taking it again is a self-deadlock, with the submitter alive and ``close``
waiting on it forever.

So the scan runs without re-acquiring when this thread is inside one of this Worker's own
callbacks, where the serialization the lock provides already holds — no other callback can be
running to observe a partial set. From any other thread it is taken as before.

``_refuse_free_while_in_flight`` reads only the run sets, the two locks and the L2 touched map, so
these run against a Worker built with ``__new__`` — no fork, no device, no ``init()``. A watchdog
thread bounds every call, because the defect being guarded is a hang rather than a wrong answer.
"""

from __future__ import annotations

import threading
from typing import cast

import pytest
from simpler.buffer import AddressSpace, BackendKind, mint_owner_instance_id, wrap_device_malloc
from simpler.orchestrator import _callback_run
from simpler.worker import RunHandle, Worker, _RunResources, _SharedExclusiveLock

_CALL_BUDGET_S = 5.0


def _bare_worker() -> Worker:
    w = Worker.__new__(Worker)
    w.level = 3
    w._registry_lock = threading.Lock()
    w._submit_mu = _SharedExclusiveLock()
    w._hierarchical_start_mu = threading.Lock()
    w._hierarchical_start_cv = threading.Condition(w._hierarchical_start_mu)
    w._accepted_run_handles = set()
    w._abandoned_run_handles = []
    w._chip_run_touched_identities = {}
    w._owner_instance_id = mint_owner_instance_id()
    return w


def _device_handle(worker: Worker, buffer_id: int = 7):
    return wrap_device_malloc(0x4000, 256, worker._owner_instance_id, buffer_id, "L3", owner_worker_id=0)


def _call_within_budget(fn) -> BaseException | None:
    """Run ``fn`` on a helper thread and fail the test if it does not return in time.

    The thread is left daemonic on timeout rather than joined: the defect this guards is a
    permanently blocked call, so there is nothing to wait for and the assertion is the result.
    """
    outcome: dict[str, BaseException] = {}
    done = threading.Event()

    def body() -> None:
        try:
            fn()
        except BaseException as error:  # noqa: BLE001 -- handed back to the test thread
            outcome["error"] = error
        finally:
            done.set()

    thread = threading.Thread(target=body, daemon=True)
    thread.start()
    assert done.wait(_CALL_BUDGET_S), "the in-flight refusal never returned; it re-took a lock this thread holds"
    return outcome.get("error")


def test_a_free_inside_a_graph_callback_does_not_retake_the_submit_lock():
    """The R4 shape: the callback's own thread holds ``_submit_mu`` and frees a buffer."""
    worker = _bare_worker()
    handle = _device_handle(worker)

    def inside_callback() -> None:
        # Exactly the nesting `_submit_l3_locked` establishes: the lock held for the whole
        # callback, and the callback frame marking this thread as the one running it.
        with worker._submit_mu.exclusive(), _callback_run(11, worker):
            worker._refuse_free_while_in_flight(handle)

    assert _call_within_budget(inside_callback) is None


def test_a_free_inside_a_callback_still_refuses_a_buffer_another_run_dispatched():
    """Skipping the lock does not skip the check: the scan itself is unchanged."""
    worker = _bare_worker()
    handle = _device_handle(worker)
    other = _StubHandle({handle.identity})
    worker._accepted_run_handles.add(cast(RunHandle, other))

    def inside_callback() -> None:
        with worker._submit_mu.exclusive(), _callback_run(11, worker):
            worker._refuse_free_while_in_flight(handle)

    error = _call_within_budget(inside_callback)
    assert isinstance(error, RuntimeError)
    assert "still referenced by an in-flight run" in str(error)


def test_a_free_from_another_thread_takes_the_submit_lock():
    """Outside a callback the lock is still taken, so the scan cannot run mid-callback."""
    worker = _bare_worker()
    handle = _device_handle(worker)
    entered = threading.Event()
    release = threading.Event()

    def hold_the_lock() -> None:
        with worker._submit_mu.exclusive():
            entered.set()
            release.wait(_CALL_BUDGET_S)

    holder = threading.Thread(target=hold_the_lock, daemon=True)
    holder.start()
    assert entered.wait(_CALL_BUDGET_S)

    finished = threading.Event()

    def from_another_thread() -> None:
        worker._refuse_free_while_in_flight(handle)
        finished.set()

    waiter = threading.Thread(target=from_another_thread, daemon=True)
    waiter.start()
    # It must be waiting on the lock, which is the property that keeps the scan out of a
    # concurrent callback's window.
    assert not finished.wait(0.5)
    release.set()
    assert finished.wait(_CALL_BUDGET_S), "the refusal never completed after the lock was released"
    holder.join(timeout=_CALL_BUDGET_S)


def test_a_callback_frame_for_another_worker_does_not_waive_the_lock():
    """The waiver is this Worker's own callback, not any callback on the thread."""
    worker = _bare_worker()
    other_worker = _bare_worker()
    handle = _device_handle(worker)
    entered = threading.Event()
    release = threading.Event()

    def hold_the_lock() -> None:
        with worker._submit_mu.exclusive():
            entered.set()
            release.wait(_CALL_BUDGET_S)

    holder = threading.Thread(target=hold_the_lock, daemon=True)
    holder.start()
    assert entered.wait(_CALL_BUDGET_S)

    finished = threading.Event()

    def in_another_workers_callback() -> None:
        with _callback_run(11, other_worker):
            worker._refuse_free_while_in_flight(handle)
        finished.set()

    waiter = threading.Thread(target=in_another_workers_callback, daemon=True)
    waiter.start()
    assert not finished.wait(0.5), "a frame belonging to another Worker waived this Worker's lock"
    release.set()
    assert finished.wait(_CALL_BUDGET_S)
    holder.join(timeout=_CALL_BUDGET_S)


class _StubHandle:
    """A run handle carrying only what the refusal reads: its touched set and cleanup flag."""

    def __init__(self, touched, *, cleanup_published: bool = False) -> None:
        self._resources = _RunResources()
        self._resources.touched_identities.update(touched)
        self._cleanup_published = cleanup_published


def test_an_l2_run_still_refuses_from_inside_a_callback():
    """The L2 scan takes its own lock and is unaffected by the waiver."""
    worker = _bare_worker()
    handle = _device_handle(worker)
    worker._chip_run_touched_identities[1] = {handle.identity}

    def inside_callback() -> None:
        with worker._submit_mu.exclusive(), _callback_run(11, worker):
            worker._refuse_free_while_in_flight(handle)

    error = _call_within_budget(inside_callback)
    assert isinstance(error, RuntimeError)
    assert "in-flight L2 run" in str(error)


@pytest.mark.parametrize("cleanup_published", [False, True])
def test_a_retired_run_does_not_refuse(cleanup_published):
    """A run past its ordered cleanup no longer holds the allocation."""
    worker = _bare_worker()
    handle = _device_handle(worker)
    worker._accepted_run_handles.add(
        cast(RunHandle, _StubHandle({handle.identity}, cleanup_published=cleanup_published))
    )

    def inside_callback() -> None:
        with worker._submit_mu.exclusive(), _callback_run(11, worker):
            worker._refuse_free_while_in_flight(handle)

    error = _call_within_budget(inside_callback)
    if cleanup_published:
        assert error is None
    else:
        assert isinstance(error, RuntimeError)


def test_an_unrelated_identity_is_not_refused():
    """The scan is by identity, so another allocation's reference does not block this one."""
    worker = _bare_worker()
    handle = _device_handle(worker, buffer_id=7)
    unrelated = _device_handle(worker, buffer_id=8)
    worker._accepted_run_handles.add(cast(RunHandle, _StubHandle({unrelated.identity})))
    assert handle.identity != unrelated.identity
    assert handle.backend_kind is BackendKind.DEVICE_MALLOC
    assert handle.address_space is AddressSpace.DEVICE

    def inside_callback() -> None:
        with worker._submit_mu.exclusive(), _callback_run(11, worker):
            worker._refuse_free_while_in_flight(handle)

    assert _call_within_budget(inside_callback) is None
