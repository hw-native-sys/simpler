# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Admission fencing for register / unregister, and the one-way shutdown word.

``Worker.close()`` publishes CLOSED and then drains ``_active_ops`` before it
touches the tree, so every API that mutates the registry or drives a child must
hold an ``_operation_lease`` for its WHOLE transaction — publication included.
A registry mutation made outside the lease is invisible to that drain, which is
how a registration lands on a worker whose teardown is already running.

Termination has the mirror-image problem on the wire: the mailbox state word has
three writers, and the ``CONTROL_DONE`` a child publishes for an in-flight
control command overwrites a concurrent ``SHUTDOWN`` store, leaving the child
polling a mailbox whose request has been erased. The sticky ``_OFF_SHUTDOWN``
word is written only by a terminating parent and never cleared, so it survives
that overwrite.

Every test here is device-free. Live-tree races use an L3 worker with one SUB
child; lifecycle crossover cases use inert L2/L3/L4 workers. A module-wide hard
timeout turns any reintroduced unbounded wait into a prompt failure.
"""

from __future__ import annotations

import contextlib
import os
import signal
import threading
import time
from multiprocessing.shared_memory import SharedMemory

import pytest
import simpler.worker as worker_mod
from _task_interface import ChipCallable  # pyright: ignore[reportMissingImports]
from simpler.worker import RemoteCallable, RemoteWorkerSpec, Worker

from ._harness import SIM_PLATFORM, SIM_RUNTIME, TEST_WALL_BUDGET_S, ObservedCondition, install_fake_chip

# Comfortably above every wait these tests actually take, well under any hang.
_TEST_WALL_BUDGET_S = TEST_WALL_BUDGET_S
_DEADLINE_AFTER_WATCHDOG_S = 2 * _TEST_WALL_BUDGET_S


@contextlib.contextmanager
def _hard_timeout(seconds: float):
    """Bound the whole test, including Worker init/close and fixture teardown."""

    def _handler(_signum, _frame):
        raise TimeoutError("admission-fence test exceeded its hard wall budget")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


@pytest.fixture(autouse=True)
def _test_wall_timeout():
    with _hard_timeout(_TEST_WALL_BUDGET_S):
        yield


def _chip_callable(func_name: str = "admission_fence") -> ChipCallable:
    return ChipCallable.build(signature=[], func_name=func_name, binary=b"\x00", children=[])


@pytest.fixture
def ready_worker(monkeypatch):
    """A READY L3 worker with one SUB child and a fake chip — no NPU required."""
    install_fake_chip(monkeypatch)
    w = Worker(
        level=3,
        device_ids=[0],
        platform=SIM_PLATFORM,
        runtime=SIM_RUNTIME,
        num_sub_workers=1,
        startup_timeout_s=_DEADLINE_AFTER_WATCHDOG_S,
    )
    w.init()
    try:
        yield w
    finally:
        with contextlib.suppress(BaseException):  # a test may have closed it already
            w.close()


class _ParkedTransaction:
    """Park a worker method mid-transaction so close() can be raced against it.

    ``entered`` fires once the patched method has run the real implementation;
    the method then blocks until ``release`` is set (bounded), which is the
    window a close() must not be able to slip through.
    """

    def __init__(self, obj, name: str):
        self.entered = threading.Event()
        self.completed = threading.Event()
        self.release = threading.Event()
        self.lease_held: bool | None = None
        self._obj = obj
        self._name = name
        self._original = getattr(obj, name)

    def __enter__(self):
        def patched(*args, **kwargs):
            result = self._original(*args, **kwargs)
            self.lease_held = threading.get_ident() in self._obj._lease_depth
            self.entered.set()
            self.release.wait(timeout=_TEST_WALL_BUDGET_S)
            self.completed.set()
            return result

        setattr(self._obj, self._name, patched)
        return self

    def __exit__(self, *exc_info):
        self.release.set()
        try:
            delattr(self._obj, self._name)
        except AttributeError:
            pass
        return False


def _close_while_parked(worker: Worker, parked: _ParkedTransaction, monkeypatch) -> None:
    close_waiting = threading.Event()
    monkeypatch.setattr(
        worker,
        "_hierarchical_start_cv",
        ObservedCondition(worker._hierarchical_start_cv, wait_entered=close_waiting),
    )

    close_wait_observed: list[tuple[bool, bool]] = []

    def release_after_close_waits():
        try:
            observed = close_waiting.wait(5.0)
            close_wait_observed.append((observed, worker._active_ops > 0))
        finally:
            parked.release.set()

    releaser = threading.Thread(target=release_after_close_waits)
    releaser.start()
    try:
        worker.close()
    finally:
        parked.release.set()
        releaser.join(_TEST_WALL_BUDGET_S)
    assert not releaser.is_alive()
    assert close_wait_observed == [(True, True)], "close never waited for an admitted operation lease"
    assert parked.completed.is_set(), "close returned before the parked transaction completed"


def _run_in_thread(fn):
    box: dict[str, BaseException] = {}

    def target():
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 -- reported to the test thread
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, box


class TestRegisterAdmissionFence:
    """register() must publish inside the lease, not before it."""

    def test_publication_holds_the_lease(self, ready_worker):
        with _ParkedTransaction(ready_worker, "_install_registration_locked") as parked:
            parked.release.set()
            handle = ready_worker.register(_chip_callable())
        assert parked.lease_held is True
        ready_worker.unregister(handle)

    def test_close_drains_a_register_in_flight(self, ready_worker, monkeypatch):
        # Parks at the broadcast, not at publication: publication runs under
        # ``_registry_lock``, which close()'s registry detach also takes, so a
        # park there would delay close() even with no lease at all.
        with _ParkedTransaction(ready_worker, "_post_init_register") as parked:
            register_thread, register_box = _run_in_thread(lambda: ready_worker.register(_chip_callable()))
            assert parked.entered.wait(timeout=_TEST_WALL_BUDGET_S), "register never reached its broadcast"
            _close_while_parked(ready_worker, parked, monkeypatch)
            register_thread.join(timeout=_TEST_WALL_BUDGET_S)

        assert not register_thread.is_alive()
        assert "error" not in register_box, f"register() failed: {register_box.get('error')}"
        # Nothing survives a terminal close: the registration either completed
        # inside the drained lease and was detached with the rest of the
        # registry, or never landed.
        assert ready_worker._identity_registry == {}
        assert ready_worker._live_handles == {}

    def test_l2_ready_registration_cannot_fall_back_to_new_after_close(self, monkeypatch):
        worker = Worker(level=2)
        worker._lifecycle = worker_mod._Lifecycle.READY
        original_wait = worker._wait_out_init_locked

        def close_after_ready_check(api: str) -> None:
            original_wait(api)
            worker._lifecycle = worker_mod._Lifecycle.CLOSED

        monkeypatch.setattr(worker, "_wait_out_init_locked", close_after_ready_check)
        try:
            with pytest.raises(RuntimeError):
                worker.register(_chip_callable("l2_close_crossover"))
            assert worker._identity_registry == {}
            assert worker._live_handles == {}
        finally:
            worker.close()

    def test_l2_pre_start_register_unregister_stays_registry_only(self):
        worker = Worker(level=2)
        try:
            handle = worker.register(_chip_callable("l2_pre_start"))
            worker.unregister(handle)
            assert worker._identity_registry == {}
            assert worker._live_handles == {}
        finally:
            worker.close()


class TestUnregisterAdmissionFence:
    """Every READY unregister path holds one lease across mutation and cleanup."""

    def test_broadcast_holds_the_lease(self, ready_worker):
        handle = ready_worker.register(_chip_callable())
        with _ParkedTransaction(ready_worker, "_broadcast_unregister") as parked:
            parked.release.set()
            ready_worker.unregister(handle)
        assert parked.lease_held is True

    def test_close_cannot_slip_into_the_unregister_broadcast(self, ready_worker, monkeypatch):
        handle = ready_worker.register(_chip_callable())
        with _ParkedTransaction(ready_worker, "_broadcast_unregister") as parked:
            unregister_thread, unregister_box = _run_in_thread(lambda: ready_worker.unregister(handle))
            assert parked.entered.wait(timeout=_TEST_WALL_BUDGET_S), "unregister never reached its broadcast"
            _close_while_parked(ready_worker, parked, monkeypatch)
            unregister_thread.join(timeout=_TEST_WALL_BUDGET_S)

        assert not unregister_thread.is_alive()
        assert "error" not in unregister_box, f"unregister() failed: {unregister_box.get('error')}"
        assert ready_worker._identity_registry == {}
        assert ready_worker._live_handles == {}

    def test_ready_unregister_cannot_fall_back_to_registry_only_after_close(self, monkeypatch):
        worker = Worker(level=3, num_sub_workers=0)
        handle = worker.register(_chip_callable("unregister_close_crossover"))
        worker._lifecycle = worker_mod._Lifecycle.READY
        original_wait = worker._wait_out_init_locked

        def close_after_ready_check(api: str) -> None:
            original_wait(api)
            worker._lifecycle = worker_mod._Lifecycle.CLOSED

        monkeypatch.setattr(worker, "_wait_out_init_locked", close_after_ready_check)
        try:
            with pytest.raises(RuntimeError):
                worker.unregister(handle)
            assert handle._handle_id in worker._live_handles
            assert handle.digest in worker._identity_registry
        finally:
            worker.close()

    def test_remote_unregister_holds_the_lease(self):
        class RemoteWorker:
            def remote_unregister(self, worker_id, *_args):
                return type("RemoteResult", (), {"ok": True, "worker_id": worker_id})()

        worker = Worker(level=4, num_sub_workers=0)
        worker_id = worker.add_remote_worker(RemoteWorkerSpec(endpoint="127.0.0.1:19073", platform="a2a3sim"))
        handle = worker.register(RemoteCallable("builtins:len"), workers=[worker_id])
        worker._lifecycle = worker_mod._Lifecycle.READY
        worker._worker = RemoteWorker()  # type: ignore[assignment]
        try:
            with _ParkedTransaction(worker, "_unregister_remote_handle") as parked:
                parked.release.set()
                worker.unregister(handle)
            assert parked.lease_held is True
        finally:
            worker._worker = None
            worker.close()

    def test_remote_pre_start_unregister_stays_registry_only(self):
        worker = Worker(level=4, num_sub_workers=0)
        worker_id = worker.add_remote_worker(RemoteWorkerSpec(endpoint="127.0.0.1:19073", platform="a2a3sim"))
        try:
            handle = worker.register(RemoteCallable("builtins:len"), workers=[worker_id])
            worker.unregister(handle)
            assert worker._identity_registry == {}
            assert worker._live_handles == {}
        finally:
            worker.close()


# ---------------------------------------------------------------------------
# One-way shutdown word
# ---------------------------------------------------------------------------

_GATE_CHILD_IN_CONTROL = 0
_GATE_PARENT_RELEASED = 1


def _serve_until_shutdown(mailbox: SharedMemory, gate: SharedMemory) -> None:
    """Child half of the shutdown race: a serve loop whose control handler is
    held open long enough for the parent to request shutdown underneath it."""
    mailbox_buf = mailbox.buf
    gate_buf = gate.buf
    assert mailbox_buf is not None
    assert gate_buf is not None
    state_addr = worker_mod._buffer_field_addr(mailbox_buf, worker_mod._OFF_STATE)

    def handle_task(_task_buf):
        return 0, ""

    def handle_control(_sub_cmd):
        gate_buf[_GATE_CHILD_IN_CONTROL] = 1
        deadline = time.monotonic() + _TEST_WALL_BUDGET_S
        while not gate_buf[_GATE_PARENT_RELEASED] and time.monotonic() < deadline:
            pass
        return 0, ""

    worker_mod._run_mailbox_loop(
        mailbox_buf,
        state_addr,
        handle_task=handle_task,
        handle_control=handle_control,
    )


def _wait_for_exit(pid: int, budget_s: float) -> int | None:
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        reaped, status = os.waitpid(pid, os.WNOHANG)
        if reaped == pid:
            return status
        time.sleep(0.01)
    return None


class TestOneWayShutdown:
    def test_shutdown_survives_an_in_flight_control_command(self):
        """A shutdown requested while a control command is in flight must still
        end the child, even though the CONTROL_DONE it publishes overwrites the
        SHUTDOWN state word."""
        mailbox = SharedMemory(create=True, size=worker_mod.mailbox_size(worker_mod._DEFAULT_PIPELINE_DEPTH_REQUEST))
        gate = SharedMemory(create=True, size=8)
        mailbox_buf = mailbox.buf
        gate_buf = gate.buf
        assert mailbox_buf is not None
        assert gate_buf is not None
        pid = None
        try:
            pid = os.fork()
            if pid == 0:  # pragma: no cover -- child process
                try:
                    _serve_until_shutdown(mailbox, gate)
                finally:
                    os._exit(0)

            state_addr = worker_mod._buffer_field_addr(mailbox_buf, worker_mod._OFF_STATE)
            worker_mod._mailbox_store_i32(state_addr, worker_mod._CONTROL_REQUEST)

            deadline = time.monotonic() + _TEST_WALL_BUDGET_S
            while not gate_buf[_GATE_CHILD_IN_CONTROL] and time.monotonic() < deadline:
                time.sleep(0.01)
            assert gate_buf[_GATE_CHILD_IN_CONTROL], "child never entered its control handler"

            # The request lands while the control command is still in flight, so
            # the child's CONTROL_DONE store comes after it.
            worker_mod._request_child_shutdown(mailbox_buf)
            gate_buf[_GATE_PARENT_RELEASED] = 1

            status = _wait_for_exit(pid, _TEST_WALL_BUDGET_S)
            assert status is not None, "child did not exit after SHUTDOWN raced an in-flight control command"
            assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
            pid = None
            assert worker_mod._mailbox_load_i32(state_addr) == worker_mod._CONTROL_DONE, (
                "expected the child's CONTROL_DONE to have overwritten the SHUTDOWN state word"
            )
        finally:
            if pid is not None:
                with contextlib.suppress(ProcessLookupError, ChildProcessError):
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
            mailbox_buf.release()
            gate_buf.release()
            mailbox.close()
            mailbox.unlink()
            gate.close()
            gate.unlink()
