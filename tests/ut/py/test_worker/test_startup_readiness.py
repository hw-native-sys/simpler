# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Recursive worker-startup readiness protocol.

A hierarchical Worker's startup is a strong READY boundary: every child
process (sub, chip, and next-level) must either publish INIT_READY after its
own init succeeds, or publish INIT_FAILED with a bounded error. The parent
waits for each child with a deadline and a ``waitpid(WNOHANG)`` liveness
check, so a child that crashes, exits, or hangs during init surfaces as a
prompt ``RuntimeError`` instead of an unbounded parent spin (the #1003 / #980
hang). On failure the parent rolls the whole startup epoch back: children that
reached their serve loop are closed gracefully so they unlink their own nested
shms, the rest are SIGKILLed, and every child is reaped.

Most tests inject failures at the L4 -> L3 (next-level) edge, which needs no
NPU device: the child runs ``inner_worker.init()`` before entering its serve
loop. The chip (L2) edge shares the same parent-side barrier; its device-free
failure path is covered by ``TestChipStartupFailure`` with a faked
``ChipWorker`` on the ``a2a3sim`` platform (no silicon).

Every failure test is wrapped in a hard SIGALRM timeout so a protocol
regression that reintroduces an unbounded spin fails the suite promptly
instead of hanging CI.
"""

import os
import struct
import threading
import time
from multiprocessing.shared_memory import SharedMemory

import pytest
from simpler.task_interface import CallConfig, TaskArgs
from simpler.worker import RemoteCallable, RemoteWorkerSpec, RunHandle, Worker

from ._harness import (
    CHIP_INIT_FAILURE,
    TEST_WALL_BUDGET_S,
    chip_callable,
    fake_chip_l3,
    hard_timeout,
    install_fake_chip,
    requires_sim_binaries,
)

_TEST_WALL_BUDGET_S = TEST_WALL_BUDGET_S
_hard_timeout = hard_timeout


def _raiser(exc: BaseException):
    """A stand-in whose call always raises ``exc``."""

    def _fn(*_a, **_k):
        raise exc

    return _fn


def _run_catch(fn):
    """Run ``fn`` in a thread body, returning None on success or the exception."""
    try:
        fn()
        return None
    except BaseException as e:  # noqa: BLE001
        return e


def _make_shared_counter():
    shm = SharedMemory(create=True, size=4)
    buf = shm.buf
    assert buf is not None
    struct.pack_into("i", buf, 0, 0)
    return shm, buf


def _read_counter(buf) -> int:
    return struct.unpack_from("i", buf, 0)[0]


def _increment_counter(buf) -> None:
    v = struct.unpack_from("i", buf, 0)[0]
    struct.pack_into("i", buf, 0, v + 1)


# Injected inner-worker init failures. Defined at module scope so the forked
# next-level child inherits them (copy-on-write) and calls the replacement in
# place of the real Worker.init.


def _init_raises(*_a, **_k):
    raise RuntimeError("injected inner init failure")


def _init_slow_raises(*_a, **_k):
    # Delay so a healthy sibling reliably reaches READY before this one fails,
    # exercising the graceful-close rollback path deterministically.
    time.sleep(0.5)
    raise RuntimeError("injected slow inner init failure")


def _init_hard_exits(*_a, **_k):
    os._exit(42)


def _init_hangs(*_a, **_k):
    time.sleep(3600)


def _l3_child(sub_fn=None, num_sub_workers=1):
    l3 = Worker(level=3, num_sub_workers=num_sub_workers)
    l3.register(sub_fn if sub_fn is not None else (lambda args: None))
    return l3


def _trivial_orch(orch, args, config):
    return None


class TestNextLevelStartupFailure:
    def test_inner_init_failure_raises_bounded_error(self):
        """A next-level child whose init raises surfaces its error at startup."""
        l3 = _l3_child()
        l3.init = _init_raises  # noqa: SLF001 -- test injection inherited across fork

        w4 = Worker(level=4, num_sub_workers=0, startup_timeout_s=10.0)
        w4.register(_trivial_orch)
        w4.add_worker(l3)
        start = time.monotonic()
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                with pytest.raises(RuntimeError, match="injected inner init failure"):
                    w4.init()
            assert time.monotonic() - start < _TEST_WALL_BUDGET_S
        finally:
            w4.close()

    def test_inner_exit_before_ready_raises(self):
        """A child that exits during init (before READY) is detected via waitpid."""
        l3 = _l3_child()
        l3.init = _init_hard_exits  # noqa: SLF001

        w4 = Worker(level=4, num_sub_workers=0, startup_timeout_s=10.0)
        w4.register(_trivial_orch)
        w4.add_worker(l3)
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                with pytest.raises(RuntimeError, match="exited during init"):
                    w4.init()
        finally:
            w4.close()

    def test_startup_deadline_fires_on_hung_child(self):
        """A child that hangs in init trips the startup deadline, not an infinite spin."""
        l3 = _l3_child()
        l3.init = _init_hangs  # noqa: SLF001

        w4 = Worker(level=4, num_sub_workers=0, startup_timeout_s=1.5)
        w4.register(_trivial_orch)
        w4.add_worker(l3)
        start = time.monotonic()
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                with pytest.raises(RuntimeError, match="deadline"):
                    w4.init()
            elapsed = time.monotonic() - start
            assert 1.5 <= elapsed < _TEST_WALL_BUDGET_S
        finally:
            w4.close()

    def test_failed_startup_reaps_children_no_leak(self, monkeypatch):
        """After a startup failure the forked children are killed and reaped."""
        l3 = _l3_child()
        l3.init = _init_hangs  # noqa: SLF001

        captured: dict[str, list[int]] = {}
        orig_abort = Worker._abort_hierarchical

        def spy_abort(self, deadline=None):
            captured["pids"] = list(self._chip_pids) + list(self._sub_pids) + list(self._next_level_pids)
            return orig_abort(self, deadline=deadline)

        monkeypatch.setattr(Worker, "_abort_hierarchical", spy_abort)

        w4 = Worker(level=4, num_sub_workers=0, startup_timeout_s=1.0)
        w4.register(_trivial_orch)
        w4.add_worker(l3)
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                with pytest.raises(RuntimeError):
                    w4.init()

            assert "pids" in captured and captured["pids"], "rollback did not run"
            for pid in captured["pids"]:
                with pytest.raises(ChildProcessError):
                    os.waitpid(pid, os.WNOHANG)

            # Rollback clears the process/mailbox bookkeeping so a later close()
            # is a clean no-op.
            assert w4._next_level_pids == []
            assert w4._next_level_shms == []
            assert w4._worker is None
        finally:
            w4.close()

    def test_ready_sibling_closed_gracefully_on_sibling_failure(self):
        """A child that reached READY is closed gracefully (not SIGKILLed) when a sibling fails.

        The healthy L3 owns a nested sub-worker mailbox shm only it can unlink;
        graceful SHUTDOWN lets it clean up. The failing L3 is delayed so the
        healthy one is reliably READY first.
        """
        good = _l3_child(num_sub_workers=1)
        bad = _l3_child(num_sub_workers=1)
        bad.init = _init_slow_raises  # noqa: SLF001

        w4 = Worker(level=4, num_sub_workers=0, startup_timeout_s=20.0)
        w4.register(_trivial_orch)
        w4.add_worker(good)
        w4.add_worker(bad)
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                with pytest.raises(RuntimeError, match="slow inner init failure"):
                    w4.init()

            report = w4._last_rollback
            assert report is not None
            # The healthy sibling reached its serve loop and was closed
            # gracefully (SHUTDOWN + reaped), not SIGKILLed.
            assert len(report["graceful"]) >= 1
            assert set(report["graceful"]).isdisjoint(report["killed"])
            assert w4._next_level_pids == []
        finally:
            w4.close()

    def test_second_child_failure_reaps_first(self):
        """When one of several next-level children fails, all are torn down."""
        good = _l3_child()
        bad = _l3_child()
        bad.init = _init_raises  # noqa: SLF001

        w4 = Worker(level=4, num_sub_workers=0, startup_timeout_s=10.0)
        w4.register(_trivial_orch)
        w4.add_worker(good)
        w4.add_worker(bad)
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                with pytest.raises(RuntimeError, match="injected inner init failure"):
                    w4.init()
            assert w4._next_level_pids == []
        finally:
            w4.close()


class TestSubStartupFailure:
    def test_sub_child_exit_before_ready_raises(self, monkeypatch):
        """A sub child that dies before entering its loop aborts startup.

        Injects a failure into the child's identity-table build (the sub's only
        fallible pre-loop step) so the sub exits before publishing READY; the
        parent's sub readiness barrier must catch it rather than silently
        succeeding and hanging a later submit_sub.
        """
        import simpler.worker as worker_mod  # noqa: PLC0415

        def _boom(*_a, **_k):
            os._exit(7)

        monkeypatch.setattr(worker_mod, "_make_local_identity_tables", _boom)

        w3 = Worker(level=3, num_sub_workers=1, startup_timeout_s=10.0)
        w3.register(lambda args: None)
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                with pytest.raises(RuntimeError, match="sub worker .* exited during init"):
                    w3.init()
        finally:
            w3.close()


class TestStartupConfigValidation:
    def test_nonpositive_timeout_rejected(self):
        with pytest.raises(ValueError, match="startup_timeout_s"):
            Worker(level=4, num_sub_workers=0, startup_timeout_s=0)

    def test_nonfinite_timeout_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            Worker(level=4, num_sub_workers=0, startup_timeout_s=float("inf"))
        with pytest.raises(ValueError, match="finite"):
            Worker(level=4, num_sub_workers=0, startup_timeout_s=float("nan"))


class TestReadyBarrierHappyPath:
    """The barrier passes a healthy tree through and dispatch still works.

    These verify the next-level readiness barrier does not break a healthy
    startup and that tasks dispatch afterwards. init() is eager and recursive, so
    a next-level child's INIT_READY means its whole subtree (its own sub / chip
    children) came up during the parent's init(), not on first run.
    """

    def test_l4_l3_tree_comes_up_and_runs(self):
        counter_shm, counter_buf = _make_shared_counter()
        w4 = None
        try:
            l3 = Worker(level=3, num_sub_workers=1)
            l3_sub = l3.register(lambda args: _increment_counter(counter_buf))

            def l3_orch(orch, args, config):
                orch.submit_sub(l3_sub)

            w4 = Worker(level=4, num_sub_workers=0, startup_timeout_s=30.0)
            l3_handle = w4.register(l3_orch)
            l3_worker_id = w4.add_worker(l3)
            w4.init()

            def l4_orch(orch, args, config):
                orch.submit_next_level(l3_handle, TaskArgs(), CallConfig(), worker=l3_worker_id)

            w4.run(l4_orch)
            assert _read_counter(counter_buf) == 1
        finally:
            if w4 is not None:
                w4.close()
            counter_shm.close()
            counter_shm.unlink()

    def test_multiple_l3_children_all_ready(self):
        """Two next-level children both pass the barrier and dispatch.

        Each child increments its OWN counter (the counter is a non-atomic RMW
        that would race if the two children shared one).
        """
        a_shm, a_buf = _make_shared_counter()
        b_shm, b_buf = _make_shared_counter()
        w4 = None
        try:
            l3a = Worker(level=3, num_sub_workers=1)
            a_sub = l3a.register(lambda args: _increment_counter(a_buf))

            def l3a_orch(orch, args, config):
                orch.submit_sub(a_sub)

            l3b = Worker(level=3, num_sub_workers=1)
            b_sub = l3b.register(lambda args: _increment_counter(b_buf))

            def l3b_orch(orch, args, config):
                orch.submit_sub(b_sub)

            w4 = Worker(level=4, num_sub_workers=0, startup_timeout_s=30.0)
            ha = w4.register(l3a_orch)
            hb = w4.register(l3b_orch)
            l3a_worker_id = w4.add_worker(l3a)
            l3b_worker_id = w4.add_worker(l3b)
            w4.init()

            def l4_orch(orch, args, config):
                orch.submit_next_level(ha, TaskArgs(), CallConfig(), worker=l3a_worker_id)
                orch.submit_next_level(hb, TaskArgs(), CallConfig(), worker=l3b_worker_id)

            w4.run(l4_orch)
            assert _read_counter(a_buf) == 1
            assert _read_counter(b_buf) == 1
        finally:
            if w4 is not None:
                w4.close()
            a_shm.close()
            a_shm.unlink()
            b_shm.close()
            b_shm.unlink()


class TestEagerInitContract:
    """init() is the single, eager startup point for level >= 3.

    Children come up during init(); run() / the other former lazy triggers no
    longer start the hierarchy, and init() without any run() closes cleanly.
    """

    def test_l3_sub_children_ready_after_init_no_run(self):
        hw = Worker(level=3, num_sub_workers=2)
        hw.register(lambda args: None)
        hw.init()
        try:
            assert hw._hierarchical_started is True
            assert len(hw._sub_pids) == 2
            # Every sub child is forked and alive (READY) before any run().
            for pid in hw._sub_pids:
                assert os.waitpid(pid, os.WNOHANG) == (0, 0)
        finally:
            hw.close()

    def test_init_then_close_without_run(self):
        hw = Worker(level=3, num_sub_workers=1)
        hw.register(lambda args: None)
        hw.init()
        hw.close()
        assert hw._sub_pids == []
        assert hw._worker is None

    def test_l4_l3_sub_subtree_ready_after_init_no_run(self):
        l3 = _l3_child(num_sub_workers=1)
        w4 = Worker(level=4, num_sub_workers=0, startup_timeout_s=30.0)
        w4.register(_trivial_orch)
        w4.add_worker(l3)
        w4.init()
        try:
            # The L4's direct L3 child is READY, which — because init() is
            # recursive — means the L3 already forked and readied its own sub
            # grandchild before publishing INIT_READY.
            assert w4._hierarchical_started is True
            assert len(w4._next_level_pids) == 1
            assert os.waitpid(w4._next_level_pids[0], os.WNOHANG) == (0, 0)
        finally:
            w4.close()

    def test_run_does_not_start_hierarchy(self, monkeypatch):
        counter_shm, counter_buf = _make_shared_counter()
        hw = Worker(level=3, num_sub_workers=1)
        sub = hw.register(lambda args: _increment_counter(counter_buf))
        hw.init()

        orig = Worker._start_hierarchical
        calls = {"n": 0}

        def spy(self):
            calls["n"] += 1
            return orig(self)

        monkeypatch.setattr(Worker, "_start_hierarchical", spy)
        try:

            def orch(o, a, c):
                o.submit_sub(sub)

            hw.run(orch)
            assert calls["n"] == 0
            assert _read_counter(counter_buf) == 1
        finally:
            hw.close()
            counter_shm.close()
            counter_shm.unlink()


class TestSubtreeCancellation:
    """§4.6 cancellation domain: a mid-init subtree is deterministically reaped.

    A next-level child is a process-group leader whose forked descendants
    inherit the group, so the startup root's rollback reaps the whole subtree
    (the child plus its grandchildren) with killpg rather than leaking orphans
    to the multiprocessing resource_tracker.
    """

    def test_stuck_midinit_subtree_killpg_reaps_grandchild(self, monkeypatch):
        import simpler.worker as worker_mod  # noqa: PLC0415

        # Shrink the cooperative-cleanup grace so the stuck survivor hits the
        # killpg backstop quickly.
        monkeypatch.setattr(worker_mod, "_ROLLBACK_GRACEFUL_TIMEOUT_S", 1.0)

        gshm = SharedMemory(create=True, size=8)
        gbuf = gshm.buf
        assert gbuf is not None
        struct.pack_into("q", gbuf, 0, 0)

        def _fork_grandchild_then_ignore_cancel(*_a, _gbuf=gbuf, **_k):
            pid = os.fork()
            if pid == 0:
                # Grandchild: inherits the L3 child's process group (no setpgid).
                while True:
                    try:
                        time.sleep(3600)
                    except BaseException:  # noqa: BLE001, PERF203
                        pass
            struct.pack_into("q", _gbuf, 0, pid)
            # Swallow the cooperative cancel so this subtree becomes a stuck
            # survivor only killpg can reap.
            while True:
                try:
                    time.sleep(3600)
                except BaseException:  # noqa: BLE001, PERF203
                    pass

        l3 = _l3_child()
        l3.init = _fork_grandchild_then_ignore_cancel  # noqa: SLF001

        w4 = Worker(level=4, num_sub_workers=0, startup_timeout_s=1.0)
        w4.register(_trivial_orch)
        w4.add_worker(l3)
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                with pytest.raises(RuntimeError, match="deadline"):
                    w4.init()

            gpid = struct.unpack_from("q", gbuf, 0)[0]
            assert gpid > 0, "grandchild was never forked"
            # killpg reaped the whole process group: the grandchild is gone.
            deadline = time.monotonic() + 5.0
            alive = True
            while time.monotonic() < deadline:
                try:
                    os.kill(gpid, 0)
                except ProcessLookupError:
                    alive = False
                    break
                time.sleep(0.05)
            assert not alive, "grandchild survived — killpg backstop did not reap the subtree"
        finally:
            w4.close()
            gshm.close()
            gshm.unlink()


class TestApiLinearizationDuringInit:
    """§4.4/§5.2: init() is one atomic epoch; concurrent API calls linearize.

    Each test pauses init() inside _start_hierarchical (state == "starting") and
    drives a second API call to observe the INITIALIZING behavior.
    """

    @staticmethod
    def _pause_start(entered, release):
        orig = Worker._start_hierarchical

        def slow(self):
            entered.set()
            if not release.wait(timeout=10.0):
                raise TimeoutError("start not released")
            return orig(self)

        return slow

    def test_register_blocks_during_initializing_then_completes(self, monkeypatch):
        entered = threading.Event()
        release = threading.Event()
        monkeypatch.setattr(Worker, "_start_hierarchical", self._pause_start(entered, release))

        w = Worker(level=3, num_sub_workers=1, startup_timeout_s=30.0)
        w.register(lambda args: None)
        init_err: list = []
        proceed = threading.Event()

        def owner_body():
            init_err.append(_run_catch(w.init))
            # A READY worker must be closed on its init-owner thread.
            proceed.wait(10.0)
            _run_catch(w.close)

        it = threading.Thread(target=owner_body)
        it.start()
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                assert entered.wait(3.0)
                reg_out: list[object] = []
                reg_started = threading.Event()

                def do_reg():
                    reg_started.set()
                    reg_out.append(w.register(lambda args: None))

                rt = threading.Thread(target=do_reg)
                rt.start()
                assert reg_started.wait(3.0)
                time.sleep(0.3)
                assert reg_out == [], "register must block while INITIALIZING"
                release.set()
                rt.join(10.0)  # register completes post-READY (implies init done)
                assert init_err == [None]
                assert len(reg_out) == 1
        finally:
            release.set()
            proceed.set()  # owner thread closes the READY worker
            it.join(5.0)

    def test_init_failure_wakes_register_waiter_with_startup_error(self, monkeypatch):
        entered = threading.Event()
        release = threading.Event()

        def boom(_self):
            entered.set()
            release.wait(timeout=10.0)
            raise RuntimeError("injected start failure")

        monkeypatch.setattr(Worker, "_start_hierarchical", boom)

        w = Worker(level=3, num_sub_workers=1, startup_timeout_s=30.0)
        w.register(lambda args: None)
        init_err: list = []
        it = threading.Thread(target=lambda: init_err.append(_run_catch(w.init)))
        it.start()
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                assert entered.wait(3.0)
                reg_err: list = []
                reg_started = threading.Event()

                def do_reg():
                    reg_started.set()
                    reg_err.append(_run_catch(lambda: w.register(lambda args: None)))

                rt = threading.Thread(target=do_reg)
                rt.start()
                assert reg_started.wait(3.0)
                time.sleep(0.2)
                release.set()
                it.join(10.0)
                rt.join(10.0)
                assert any("injected start failure" in str(e) for e in init_err)
                assert any(e is not None and "startup failed" in str(e) for e in reg_err)
        finally:
            release.set()
            it.join(5.0)
            w.close()

    def test_add_worker_rejected_during_initializing(self, monkeypatch):
        entered = threading.Event()
        release = threading.Event()
        monkeypatch.setattr(Worker, "_start_hierarchical", self._pause_start(entered, release))

        w4 = Worker(level=4, num_sub_workers=0, startup_timeout_s=30.0)
        w4.register(_trivial_orch)
        w4.add_worker(_l3_child())
        proceed = threading.Event()

        def owner_body():
            _run_catch(w4.init)
            proceed.wait(10.0)
            _run_catch(w4.close)  # owner thread closes the READY tree

        it = threading.Thread(target=owner_body)
        it.start()
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                assert entered.wait(3.0)
                with pytest.raises(RuntimeError, match="before init"):
                    w4.add_worker(_l3_child())
                release.set()
        finally:
            release.set()
            proceed.set()
            it.join(10.0)

    def test_close_during_initializing_cancels_init(self, monkeypatch):
        # close() on a non-owner thread during INITIALIZING cooperatively
        # cancels the in-progress init and reaches CLOSED. Run close() in
        # a thread so release.set() can fire while close() awaits init unwind.
        import simpler.worker as worker_mod  # noqa: PLC0415

        entered = threading.Event()
        release = threading.Event()
        monkeypatch.setattr(Worker, "_start_hierarchical", self._pause_start(entered, release))

        w = Worker(level=3, num_sub_workers=1, startup_timeout_s=30.0)
        w.register(lambda args: None)

        def owner_body():
            _run_catch(w.init)

        it = threading.Thread(target=owner_body)
        it.start()
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                assert entered.wait(3.0)

                close_result: list = []
                ct = threading.Thread(target=lambda: close_result.append(_run_catch(w.close)))
                ct.start()
                # Wait for cancel token to latch, then release the init thread.
                while not w._cancel_token:
                    time.sleep(0.001)
                release.set()
                ct.join(10.0)
                it.join(10.0)
                assert close_result == [None]
                assert w._lifecycle is worker_mod._Lifecycle.CLOSED
        finally:
            release.set()
            it.join(10.0)


# Module-level (picklable-by-reference) probe for the callable-__del__-reenters-
# close regression: the callable must not hold a Worker ref (register pickles it),
# so it reaches the worker via this module global at __del__ time only.
_REENTRY_STATE: dict = {"worker": None, "count": 0}


class _ReentryProbe:
    """A registered callable whose __del__ reenters Worker.close()."""

    def __call__(self, args):  # pragma: no cover - never dispatched in this test
        return None

    def __del__(self):
        _REENTRY_STATE["count"] += 1
        worker = _REENTRY_STATE["worker"]
        if worker is not None:
            try:
                worker.close()
            except BaseException:  # noqa: BLE001
                pass


class TestLevel2Lifecycle:
    """An L2 worker's init()/close() must not deadlock on the level>=3-only
    epoch state machine.

    Regression: init() left the L2 worker's lifecycle state at "starting" (only
    level>=3 committed "started"), so close()'s wait-out-"starting" hung forever
    — which timed out the first L2 test of every sim / onboard suite.
    """

    def _make_l2(self, monkeypatch):
        import simpler_setup.runtime_builder as rb_mod  # noqa: PLC0415

        class _FakeBuilder:
            def __init__(self, *_a, **_k):
                pass

            def get_binaries(self, *_a, **_k):
                return object()

        # Mock the device/runtime layer so this stays device-free (no sim
        # binaries required) — the regression is purely the lifecycle state.
        install_fake_chip(monkeypatch)
        monkeypatch.setattr(rb_mod, "RuntimeBuilder", _FakeBuilder)
        return Worker(level=2, device_id=0, platform="a2a3sim", runtime="tensormap_and_ringbuffer")

    def test_l2_init_then_close_does_not_hang(self, monkeypatch):
        w = self._make_l2(monkeypatch)
        with _hard_timeout(_TEST_WALL_BUDGET_S):
            w.init()
            assert w._initialized is True
            w.close()
            assert w._initialized is False
            # A second close is a clean no-op (does not re-block on the epoch cv).
            w.close()

    def test_l2_submit_returns_live_handle(self, monkeypatch):
        from simpler.task_interface import ChipCallable  # noqa: PLC0415

        w = self._make_l2(monkeypatch)
        callable = ChipCallable.build(signature=[], func_name="x", binary=b"\x00", children=[])
        callable_handle = w.register(callable)
        with _hard_timeout(_TEST_WALL_BUDGET_S):
            w.init()
            run_handle = w.submit(callable_handle)
            assert isinstance(run_handle, RunHandle)
            # The handle is backed by a live lane run rather than born terminal:
            # a fake chip reaches its fence immediately, so `done` says nothing
            # here, but the run must be registered and waiting must reach it.
            assert w._chip_run_for(run_handle._run_id) is not None
            assert run_handle.wait() is None
            # Waiting retires the lane entry; the handle stays terminal and
            # waiting again is idempotent rather than a second submit.
            assert w._chip_run_for(run_handle._run_id) is None
            assert run_handle.done
            assert run_handle.wait() is None
            w.close()

    def test_l2_run_equals_submit_then_wait(self, monkeypatch):
        """``run`` is the blocking composition of ``submit`` and ``wait``."""
        from simpler.task_interface import ChipCallable  # noqa: PLC0415

        w = self._make_l2(monkeypatch)
        callable = ChipCallable.build(signature=[], func_name="x", binary=b"\x00", children=[])
        callable_handle = w.register(callable)
        with _hard_timeout(_TEST_WALL_BUDGET_S):
            w.init()
            impl = w._chip_worker._impl
            w.run(callable_handle)
            w.submit(callable_handle).wait()
            # Both forms admit through the same lane entry point, so the direct
            # path has one authority rather than a blocking bypass beside it.
            assert len(impl.submitted) == 2
            assert all(run.wait_count >= 1 for _cid, run in impl.submitted)
            w.close()

    def test_l2_unwaited_handle_is_drained_by_close(self, monkeypatch):
        """A handle the caller drops still owns device work until close drains it."""
        from simpler.task_interface import ChipCallable  # noqa: PLC0415

        w = self._make_l2(monkeypatch)
        callable = ChipCallable.build(signature=[], func_name="x", binary=b"\x00", children=[])
        callable_handle = w.register(callable)
        with _hard_timeout(_TEST_WALL_BUDGET_S):
            w.init()
            impl = w._chip_worker._impl
            w.submit(callable_handle)  # deliberately never waited
            assert not impl.lane_closed
            w.close()
            # close() drives the lane's own drain, where a failure is reported
            # through the cleanup journal instead of being swallowed by the
            # lane destructor — and it does so while the device is still up,
            # since draining after finalize would wait on a device that is gone.
            assert impl.lane_closed
            assert impl.lane_closed_before_finalize is True

    def test_l2_close_reports_lane_failure_only_when_undelivered(self, monkeypatch):
        """Close re-raises the lane's poison only if no wait already delivered it.

        The lane rethrows its poison on every close. A run whose handle was
        waited on has already raised that error at the wait, so repeating it at
        close would turn a failure the caller handled into an unhandled one —
        which is what the onboard fault-injection scene tests do: they assert
        ``run()`` raises and then close in a ``finally``.
        """
        from simpler.task_interface import ChipCallable  # noqa: PLC0415

        callable = ChipCallable.build(signature=[], func_name="x", binary=b"\x00", children=[])
        boom = RuntimeError("chip run lane is poisoned")

        # Waited: the error reached the caller at wait(), so close is silent.
        w = self._make_l2(monkeypatch)
        handle = w.register(callable)
        with _hard_timeout(_TEST_WALL_BUDGET_S):
            w.init()
            impl = w._chip_worker._impl
            monkeypatch.setattr(impl, "_close_chip_run_lane", _raiser(boom))
            w.submit(handle).wait()
            assert not w._chip_runs
            w.close()

        # Never waited: nobody has been told, so close is the first report.
        w2 = self._make_l2(monkeypatch)
        handle2 = w2.register(callable)
        with _hard_timeout(_TEST_WALL_BUDGET_S):
            w2.init()
            impl2 = w2._chip_worker._impl
            monkeypatch.setattr(impl2, "_close_chip_run_lane", _raiser(boom))
            w2.submit(handle2)
            assert w2._chip_runs
            with pytest.raises(RuntimeError, match="poisoned"):
                w2.close()

    def test_l2_close_without_init_is_noop(self, monkeypatch):
        w = self._make_l2(monkeypatch)
        with _hard_timeout(_TEST_WALL_BUDGET_S):
            w.close()

    def test_close_terminal_residual_raises_and_replays(self, monkeypatch):
        # If teardown runs but leaves a resource un-reclaimed, close() must NOT
        # return success: it synthesizes a terminal error naming the leak, and a
        # later close() replays the same result (teardown is never re-driven).
        import simpler.worker as worker_mod  # noqa: PLC0415

        w = self._make_l2(monkeypatch)  # owner = main
        with _hard_timeout(_TEST_WALL_BUDGET_S):
            w.init()
            # A teardown that reclaims nothing (the chip stays live).
            monkeypatch.setattr(Worker, "_teardown_ready_tree", lambda self: None)
            err = _run_catch(w.close)
            assert isinstance(err, RuntimeError)
            assert "un-reclaimed" in str(err)
            assert w._lifecycle is worker_mod._Lifecycle.CLOSED
            assert w._chip_worker is not None  # leaked — terminal, not retried
            # Terminal: a later close() replays the same error, never re-drives.
            err2 = _run_catch(w.close)
            assert isinstance(err2, RuntimeError)
            assert "un-reclaimed" in str(err2)
            assert w._chip_worker is not None

    def test_close_of_registered_new_worker_releases_registry(self):
        # A NEW worker (never init'd) with pre-registered callables has no native
        # tree, but close() must still release its callable/identity/handle
        # registries — not keep the callable refs forever.
        import simpler.worker as worker_mod  # noqa: PLC0415

        w = Worker(level=3, num_sub_workers=1)
        w.register(lambda args: None)
        assert w._identity_registry and w._callable_registry and w._live_handles
        with _hard_timeout(_TEST_WALL_BUDGET_S):
            w.close()
        assert w._lifecycle is worker_mod._Lifecycle.CLOSED
        assert w._callable_registry == {}
        assert w._identity_registry == {}
        assert w._live_handles == {}

    def test_close_tolerates_callable_del_reentry(self):
        # A registered callable whose __del__ reenters close() must not deadlock:
        # the registry refs are released AFTER the attempt is completed and
        # OUTSIDE _registry_lock, so the reentrant close() resolves against the
        # done attempt instead of waiting on itself.
        import simpler.worker as worker_mod  # noqa: PLC0415

        w = Worker(level=3, num_sub_workers=1)
        _REENTRY_STATE["worker"] = w
        _REENTRY_STATE["count"] = 0
        try:
            w.register(_ReentryProbe())  # only the registry holds the callable
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                w.close()  # dropping the registry ref runs __del__ -> reentrant close()
            assert w._lifecycle is worker_mod._Lifecycle.CLOSED
            assert _REENTRY_STATE["count"] >= 1  # __del__ actually fired during close()
        finally:
            _REENTRY_STATE["worker"] = None

    def test_close_detach_interrupt_folds_into_single_result(self, monkeypatch):
        # A BaseException during the registry detach must fold into the ONE
        # attempt result, never leaving one close() seeing success and another
        # the error. Teardown succeeds here (result would be None); the injected
        # detach interrupt makes the whole close terminally fail CONSISTENTLY,
        # the attempt is still completed (no strand), and a later close() replays
        # that same error — never a spurious success.
        import simpler.worker as worker_mod  # noqa: PLC0415
        from simpler.task_interface import ChipCallable  # noqa: PLC0415

        def _catch(fn):
            try:
                fn()
                return None
            except BaseException as e:  # noqa: BLE001
                return e

        w = self._make_l2(monkeypatch)  # owner = main
        with _hard_timeout(_TEST_WALL_BUDGET_S):
            w.init()
            w.register(ChipCallable.build(signature=[], func_name="x", binary=b"\x00", children=[]))
            assert w._identity_registry  # registry populated
            # Teardown "succeeds" (chip gone, no residual) WITHOUT touching the
            # registry lock, so the injected interrupt fires only at the detach.
            monkeypatch.setattr(Worker, "_teardown_ready_tree", lambda self: setattr(self, "_chip_worker", None))

            class _KIOnEnter:
                def __enter__(self):
                    raise KeyboardInterrupt

                def __exit__(self, *_a):
                    return False

            w._registry_lock = _KIOnEnter()  # the detach acquire raises

            r1 = _catch(w.close)
            assert isinstance(r1, KeyboardInterrupt)  # the detach interrupt surfaced
            assert w._close_completion is not None and w._close_completion.done  # not stranded
            assert isinstance(w._close_completion.error, KeyboardInterrupt)  # folded → one result
            assert w._lifecycle is worker_mod._Lifecycle.CLOSED
            # No fork: a later close() replays the SAME terminal error, not success.
            w._registry_lock = threading.Lock()
            assert isinstance(_catch(w.close), KeyboardInterrupt)

    def test_close_done_set_before_notify_lock_no_strand(self, monkeypatch):
        # attempt.done must be set BEFORE acquiring the notify CV, so a
        # BaseException during that (interruptible, possibly-blocking) acquire
        # cannot leave the attempt at done=False. The KI is injected on the
        # completion's CV acquire BY ORDER (the 2nd acquire of a clean close),
        # NOT by observing done — so if the code regressed to set done *inside*
        # that block, done would stay False and this test fails (strand assert +
        # a hanging second close caught by the hard timeout).
        import simpler.worker as worker_mod  # noqa: PLC0415

        w = self._make_l2(monkeypatch)  # owner = main
        with _hard_timeout(_TEST_WALL_BUDGET_S):
            w.init()
            real_cv = w._hierarchical_start_cv
            state = {"count": 0, "injected": False}

            class _KIOnCompletionAcquire:
                # A clean close acquires the CV twice: (1) claim/drain block,
                # (2) the completion's notify. Raise on #2 regardless of `done`.
                def __enter__(self):
                    state["count"] += 1
                    if state["count"] == 2:
                        state["injected"] = True
                        raise KeyboardInterrupt
                    return real_cv.__enter__()

                def __exit__(self, *a):
                    return real_cv.__exit__(*a)

                def __getattr__(self, name):
                    return getattr(real_cv, name)

            w._hierarchical_start_cv = _KIOnCompletionAcquire()
            r1 = _run_catch(w.close)
            assert state["injected"] is True  # the KI was actually injected
            assert isinstance(r1, KeyboardInterrupt)  # the first close() surfaced it
            # Regression catch: done was published BEFORE the interrupted acquire.
            assert w._close_completion is not None and w._close_completion.done
            assert w._lifecycle is worker_mod._Lifecycle.CLOSED
            # The second close() resolves against the saved completion, no hang
            # (teardown succeeded, so it returns cleanly rather than replaying).
            w._hierarchical_start_cv = real_cv
            assert _run_catch(w.close) is None

    def test_close_outcome_retries_an_interrupt_before_publication(self, monkeypatch):
        import simpler.worker as worker_mod  # noqa: PLC0415

        interrupt = SystemExit("before close outcome publication")
        entered = threading.Event()
        release = threading.Event()
        attempt_type = worker_mod._CloseAttempt

        class _InterruptBeforeOutcome(attempt_type):
            __slots__ = ("_armed",)

            def __init__(self):
                super().__init__()
                object.__setattr__(self, "_armed", True)

            def publish(self, error, incomplete):
                if self._armed:
                    object.__setattr__(self, "_armed", False)
                    entered.set()
                    assert release.wait(10.0)
                    raise interrupt
                return super().publish(error, incomplete)

        monkeypatch.setattr(worker_mod, "_CloseAttempt", _InterruptBeforeOutcome)
        w = self._make_l2(monkeypatch)
        owner_result: list = []
        joiner_result: list = []

        def owner():
            w.init()
            owner_result.append(_run_catch(w.close))

        owner_thread = threading.Thread(target=owner)
        joiner_thread = threading.Thread(target=lambda: joiner_result.append(_run_catch(w.close)))
        owner_thread.start()
        try:
            assert entered.wait(5.0)
            joiner_thread.start()
            time.sleep(0.1)
            assert joiner_thread.is_alive()
            release.set()
            owner_thread.join(5.0)
            joiner_thread.join(5.0)

            assert not owner_thread.is_alive()
            assert not joiner_thread.is_alive()
            assert owner_result == [interrupt]
            assert joiner_result == [interrupt]
            assert w._close_completion is not None and w._close_completion.done
            assert w._close_completion.error is interrupt
            assert _run_catch(w.close) is interrupt
        finally:
            release.set()
            owner_thread.join(5.0)
            if joiner_thread.ident is not None:
                joiner_thread.join(5.0)

    def test_close_outcome_survives_an_interrupt_after_publication(self, monkeypatch):
        import simpler.worker as worker_mod  # noqa: PLC0415

        interrupt = SystemExit("after close outcome publication")
        entered = threading.Event()
        release = threading.Event()
        attempt_type = worker_mod._CloseAttempt

        class _InterruptAfterOutcome(attempt_type):
            __slots__ = ("_armed",)

            def __init__(self):
                super().__init__()
                object.__setattr__(self, "_armed", True)

            def __setattr__(self, name, value):
                if name == "_outcome" and value is not None and self._armed:
                    super().__setattr__(name, value)
                    object.__setattr__(self, "_armed", False)
                    entered.set()
                    assert release.wait(10.0)
                    raise interrupt
                return super().__setattr__(name, value)

        monkeypatch.setattr(worker_mod, "_CloseAttempt", _InterruptAfterOutcome)
        w = self._make_l2(monkeypatch)
        owner_result: list = []
        joiner_result: list = []

        def owner():
            w.init()
            owner_result.append(_run_catch(w.close))

        owner_thread = threading.Thread(target=owner)
        joiner_thread = threading.Thread(target=lambda: joiner_result.append(_run_catch(w.close)))
        owner_thread.start()
        try:
            assert entered.wait(5.0)
            assert w._close_completion is not None and w._close_completion.done
            joiner_thread.start()
            joiner_thread.join(5.0)
            assert not joiner_thread.is_alive()
            assert joiner_result == [None]

            release.set()
            owner_thread.join(5.0)
            assert not owner_thread.is_alive()
            assert owner_result == [interrupt]
            assert w._close_completion.error is None
            assert _run_catch(w.close) is None
        finally:
            release.set()
            owner_thread.join(5.0)
            if joiner_thread.ident is not None:
                joiner_thread.join(5.0)

    def test_reap_deadline_starts_after_shutdown_broadcast(self, monkeypatch):
        # The child-reap grace must be measured from when SHUTDOWN is broadcast,
        # not from teardown entry: a slow pre-child cleanup step must not consume
        # it and reduce the reap to a single poll.
        import types  # noqa: PLC0415

        import simpler.worker as worker_mod  # noqa: PLC0415

        monkeypatch.setattr(worker_mod, "_ROLLBACK_GRACEFUL_TIMEOUT_S", 1.0)
        w = Worker(level=3, num_sub_workers=0)
        w._worker = types.SimpleNamespace(close=lambda: None)  # look "started" for the L3 branch

        # A slow pre-child cleanup step (runs before the SHUTDOWN broadcast).
        monkeypatch.setattr(Worker, "_release_all_buffers", lambda self: time.sleep(0.6))
        captured: dict = {}

        def capture_reap(groups, deadline):
            captured["remaining"] = deadline - time.monotonic()

        monkeypatch.setattr(Worker, "_reap_child_groups", staticmethod(capture_reap))
        with _hard_timeout(_TEST_WALL_BUDGET_S):
            w._teardown_ready_tree()
        # The reap got ~full grace (1.0s), not 1.0 - 0.6 left over from a deadline
        # fixed at teardown entry.
        assert captured["remaining"] > 0.7

    def test_reap_child_groups_stuck_child_no_starvation(self, monkeypatch):
        # A stuck child in one group must not starve the reap of healthy children
        # in another. With every group SHUTDOWN up front, the interleaved reap
        # polls all groups each round, so healthy children that take a few polls
        # to exit are still reaped while one group's child is wedged; only the
        # wedged child remains a (reported) survivor.
        import simpler.worker as worker_mod  # noqa: PLC0415

        class _FakeShm:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

            def unlink(self):
                pass

        stuck_pid = 90001
        polls: dict[int, int] = {}

        def fake_waitpid(pid, _flags):
            polls[pid] = polls.get(pid, 0) + 1
            if pid == stuck_pid:
                return (0, 0)  # never exits
            if polls[pid] < 3:
                return (0, 0)  # healthy: needs a few polls to exit after SHUTDOWN
            return (pid, 0)  # reaped, clean exit

        monkeypatch.setattr(worker_mod.os, "waitpid", fake_waitpid)
        sub_shms, sub_pids = [_FakeShm()], [stuck_pid]
        chip_shms, chip_pids = [_FakeShm()], [90002]
        next_shms, next_pids = [_FakeShm()], [90003]
        groups = [(sub_shms, sub_pids), (chip_shms, chip_pids), (next_shms, next_pids)]
        deadline = time.monotonic() + 1.0
        with _hard_timeout(_TEST_WALL_BUDGET_S):
            err = _run_catch(lambda: Worker._reap_child_groups(groups, deadline))  # type: ignore[arg-type]
        # The wedged child is a reported survivor, kept in its group...
        assert isinstance(err, TimeoutError) and str(stuck_pid) in str(err)
        assert sub_pids == [stuck_pid] and len(sub_shms) == 1
        # ...but the healthy children were reaped, freed, and removed.
        assert chip_pids == [] and chip_shms == [] and next_pids == [] and next_shms == []

    def test_non_owner_close_of_ready_raises(self, monkeypatch):
        # A READY worker holds same-thread-only native objects, so a close() from
        # a thread other than the init owner is rejected before touching the
        # lifecycle; the owner can still close cleanly.
        w = self._make_l2(monkeypatch)  # init owner = this (main) thread
        with _hard_timeout(_TEST_WALL_BUDGET_S):
            w.init()
            err: list = []
            t = threading.Thread(target=lambda: err.append(_run_catch(w.close)))
            t.start()
            t.join(10.0)
            assert isinstance(err[0], RuntimeError)
            assert "thread that init" in str(err[0])
            w.close()  # owner closes cleanly

    def test_concurrent_close_owner_plus_joiner(self, monkeypatch):
        # Once the owner has published CLOSED and is mid-teardown, any thread may
        # join the in-flight attempt and observe the same completion — no second
        # (double-finalize) teardown.
        import simpler.worker as worker_mod  # noqa: PLC0415

        entered = threading.Event()
        release = threading.Event()
        orig_teardown = Worker._teardown_ready_tree

        def paused_teardown(self):
            entered.set()
            assert release.wait(10.0)
            return orig_teardown(self)

        monkeypatch.setattr(Worker, "_teardown_ready_tree", paused_teardown)
        w = self._make_l2(monkeypatch)
        results: dict = {}

        def owner():
            w.init()  # owner thread claims the epoch
            results["owner"] = _run_catch(w.close)

        ot = threading.Thread(target=owner)
        ot.start()
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                assert entered.wait(3.0)  # owner published CLOSED, teardown paused
                assert w._lifecycle is worker_mod._Lifecycle.CLOSED
                joiner: list = []
                jt = threading.Thread(target=lambda: joiner.append(_run_catch(w.close)))
                jt.start()
                time.sleep(0.2)  # joiner parks on the in-flight attempt
                assert w._close_completion is not None and not w._close_completion.done
                release.set()
                ot.join(10.0)
                jt.join(10.0)
                assert results["owner"] is None
                assert joiner == [None]
                assert w._lifecycle is worker_mod._Lifecycle.CLOSED
        finally:
            release.set()
            ot.join(5.0)

    def test_close_drains_in_flight_operation(self, monkeypatch):
        # An operation admitted before close() holds a lease; close() (the owner)
        # blocks in its drain until the in-flight op finishes, then tears down.
        import simpler.worker as worker_mod  # noqa: PLC0415

        entered = threading.Event()
        release = threading.Event()

        def paused_run(self, *_a, **_k):
            entered.set()
            assert release.wait(10.0)
            return RunHandle._completed(self)

        monkeypatch.setattr(Worker, "_submit_locked", paused_run)
        w = self._make_l2(monkeypatch)  # owner = this (main) thread
        with _hard_timeout(_TEST_WALL_BUDGET_S):
            w.init()
            rt = threading.Thread(target=lambda: w.run(lambda *_a: None))
            rt.start()
            try:
                assert entered.wait(3.0)  # run() admitted, holding a lease
                assert w._active_ops == 1
                releaser = threading.Thread(target=lambda: (time.sleep(0.5), release.set()))
                releaser.start()
                t0 = time.monotonic()
                w.close()  # owner close: drains the lease before teardown
                assert time.monotonic() - t0 >= 0.4
                assert w._lifecycle is worker_mod._Lifecycle.CLOSED
                assert w._active_ops == 0
                releaser.join(5.0)
            finally:
                release.set()
                rt.join(5.0)

    def test_reentrant_close_from_operation_rejected(self, monkeypatch):
        # close() called from inside a leased operation (e.g. an orch fn) would
        # drain its own never-releasing lease; it must be rejected outright.
        result: dict = {}

        def reentrant_run(self, *_a, **_k):
            result["close_err"] = _run_catch(self.close)
            return RunHandle._completed(self)

        monkeypatch.setattr(Worker, "_submit_locked", reentrant_run)
        w = self._make_l2(monkeypatch)  # owner = main
        with _hard_timeout(_TEST_WALL_BUDGET_S):
            w.init()
            w.run(lambda *_a: None)  # inside, calls w.close() -> rejected
            assert isinstance(result["close_err"], RuntimeError)
            assert "within a run" in str(result["close_err"])
            w.close()  # after the op returns, close succeeds

    def test_close_timeout_defers_teardown_and_retry_completes(self, monkeypatch):
        # If an admitted operation outlives the drain budget, close() must NOT
        # tear down the live tree: it publishes CLOSED (admission fenced) but
        # leaves teardown UN-attempted (attempt INCOMPLETE) and the native object
        # intact, reporting TimeoutError. Because teardown never ran, this is the
        # one retryable close() path: a later close() drives it once the op ends.
        import simpler.worker as worker_mod  # noqa: PLC0415

        monkeypatch.setattr(worker_mod, "_ROLLBACK_GRACEFUL_TIMEOUT_S", 0.5)
        entered = threading.Event()
        release = threading.Event()

        def paused_run(self, *_a, **_k):
            entered.set()
            assert release.wait(10.0)
            return RunHandle._completed(self)

        monkeypatch.setattr(Worker, "_submit_locked", paused_run)
        w = self._make_l2(monkeypatch)  # owner = main
        with _hard_timeout(_TEST_WALL_BUDGET_S):
            w.init()
            rt = threading.Thread(target=lambda: _run_catch(lambda: w.run(lambda *_a: None)))
            rt.start()
            try:
                assert entered.wait(3.0)
                assert w._active_ops == 1
                err = _run_catch(w.close)  # owner drains, times out at 0.5s
                assert isinstance(err, TimeoutError)
                assert w._lifecycle is worker_mod._Lifecycle.CLOSED  # admission fenced
                assert w._close_completion is not None and w._close_completion.incomplete
                assert w._chip_worker is not None  # native object NOT torn down
            finally:
                release.set()
                rt.join(5.0)
            w.close()  # op drained -> teardown runs once, to completion
            assert w._lifecycle is worker_mod._Lifecycle.CLOSED
            assert w._chip_worker is None
            assert w._close_completion is not None and not w._close_completion.incomplete

    def test_close_retry_still_drains_before_teardown(self, monkeypatch):
        # Regression: a retry close() while the op is STILL in flight must drain
        # again (teardown is still un-attempted), never tear down under a live op.
        import simpler.worker as worker_mod  # noqa: PLC0415

        monkeypatch.setattr(worker_mod, "_ROLLBACK_GRACEFUL_TIMEOUT_S", 0.4)
        entered = threading.Event()
        release = threading.Event()

        def paused_run(self, *_a, **_k):
            entered.set()
            assert release.wait(10.0)
            return RunHandle._completed(self)

        monkeypatch.setattr(Worker, "_submit_locked", paused_run)
        w = self._make_l2(monkeypatch)  # owner = main
        with _hard_timeout(_TEST_WALL_BUDGET_S):
            w.init()
            rt = threading.Thread(target=lambda: _run_catch(lambda: w.run(lambda *_a: None)))
            rt.start()
            try:
                assert entered.wait(3.0)
                # First close: times out, CLOSED + incomplete, native intact.
                assert isinstance(_run_catch(w.close), TimeoutError)
                assert w._chip_worker is not None
                # Retry WHILE the op is still running: must drain again and time
                # out — must NOT tear down the still-in-use device.
                assert isinstance(_run_catch(w.close), TimeoutError)
                assert w._chip_worker is not None
                assert w._active_ops == 1
            finally:
                release.set()
                rt.join(5.0)
            w.close()  # op drained -> teardown completes
            assert w._chip_worker is None

    def test_l2_register_after_close_rejected(self, monkeypatch):
        from simpler.task_interface import ChipCallable  # noqa: PLC0415

        w = self._make_l2(monkeypatch)
        with _hard_timeout(_TEST_WALL_BUDGET_S):
            w.init()
            w.close()
            cc = ChipCallable.build(signature=[], func_name="x", binary=b"\x00", children=[])
            with pytest.raises(RuntimeError, match="closed"):
                w.register(cc)

    def _make_l2_with_chip(self, monkeypatch, chip_cls):
        import simpler.worker as worker_mod  # noqa: PLC0415

        import simpler_setup.runtime_builder as rb_mod  # noqa: PLC0415

        class _FakeBuilder:
            def __init__(self, *_a, **_k):
                pass

            def get_binaries(self, *_a, **_k):
                return object()

        monkeypatch.setattr(worker_mod, "ChipWorker", chip_cls)
        monkeypatch.setattr(rb_mod, "RuntimeBuilder", _FakeBuilder)
        return Worker(level=2, device_id=0, platform="a2a3sim", runtime="tensormap_and_ringbuffer")

    def test_two_concurrent_l2_init_serialize_on_epoch(self, monkeypatch):
        # Two concurrent init() calls must serialize on the lifecycle epoch: the
        # first claims INITIALIZING and builds the one ChipWorker; the second is
        # rejected while the first holds the epoch — never a second _chip_worker.
        entered = threading.Event()
        release = threading.Event()
        build_count = {"n": 0}

        class _PausingChip:
            def __init__(self):
                build_count["n"] += 1

            def init(self, *_a, **_k):
                entered.set()
                assert release.wait(10.0)

            def _register_callable_at_slot(self, *_a, **_k):  # pragma: no cover
                pass

            def finalize(self):
                pass

        w = self._make_l2_with_chip(monkeypatch, _PausingChip)
        errs: list = []
        proceed = threading.Event()
        state: dict = {}

        def owner_body():
            errs.append(_run_catch(w.init))
            state["initialized"] = w._initialized
            state["build"] = build_count["n"]
            proceed.wait(10.0)
            _run_catch(w.close)  # the winning (owner) thread closes

        t1 = threading.Thread(target=owner_body)
        t1.start()
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                assert entered.wait(3.0)
                # The second init sees INITIALIZING and rejects immediately.
                err2 = _run_catch(w.init)
                assert isinstance(err2, RuntimeError)
                assert "in progress" in str(err2)
                release.set()
                # Wait for the owner to finish init (it then parks on `proceed`).
                deadline = time.monotonic() + 10.0
                while "initialized" not in state and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert errs == [None]
                assert state["initialized"] is True
                assert state["build"] == 1
        finally:
            release.set()
            proceed.set()
            t1.join(5.0)

    def test_l2_close_during_initializing_cancels_init(self, monkeypatch):
        # close() on a non-owner thread during an L2 INITIALIZING cooperatively
        # cancels the in-progress ChipWorker.init. The device is finalized by
        # init's cleanup, and close reaches CLOSED. Run close() in a thread so
        # release.set() fires while close() awaits init unwind.
        import simpler.worker as worker_mod  # noqa: PLC0415

        entered = threading.Event()
        release = threading.Event()
        finalized = {"n": 0}

        class _PausingChip:
            def init(self, *_a, **_k):
                entered.set()
                assert release.wait(10.0)

            def _register_callable_at_slot(self, *_a, **_k):  # pragma: no cover
                pass

            def finalize(self):
                finalized["n"] += 1

        w = self._make_l2_with_chip(monkeypatch, _PausingChip)
        init_err: list = []

        def owner_body():
            init_err.append(_run_catch(w.init))

        t1 = threading.Thread(target=owner_body)
        t1.start()
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                assert entered.wait(3.0)

                close_result: list = []
                ct = threading.Thread(target=lambda: close_result.append(_run_catch(w.close)))
                ct.start()
                # Wait for cancel token to latch, then release init.
                while not w._cancel_token:
                    time.sleep(0.001)
                release.set()
                ct.join(10.0)
                t1.join(10.0)
                assert close_result == [None]
                assert w._lifecycle is worker_mod._Lifecycle.CLOSED
                assert finalized["n"] == 1  # device finalized by init cleanup
        finally:
            release.set()
            t1.join(5.0)


@requires_sim_binaries
class TestChipStartupFailure:
    """Chip (L2) startup failure — device-free via a faked ChipWorker on a2a3sim.

    Constructing an L3 with ``device_ids`` only reads the prebuilt runtime
    binaries; the forked chip child instantiates ``worker.ChipWorker``, which
    the test replaces so no silicon is required. The failure trips before
    ``dw.init()``, so the sim runtime is never actually driven. Exercises the
    same parent-side readiness barrier as the next-level edge (the #1003 spin at
    the former ``while ... != INIT_DONE`` was on this chip path).
    """

    def test_chip_init_failure_raises_bounded(self, monkeypatch):
        # chip-only L3; the chip forks from device_ids
        with fake_chip_l3(monkeypatch, script="raises", init=False, startup_timeout_s=10.0) as l3:
            with pytest.raises(RuntimeError, match=CHIP_INIT_FAILURE):
                l3.init()

    def test_chip_init_hang_trips_deadline(self, monkeypatch):
        start = time.monotonic()
        with fake_chip_l3(monkeypatch, script="hangs", init=False, startup_timeout_s=1.5) as l3:
            with pytest.raises(RuntimeError, match="deadline"):
                l3.init()
            assert 1.5 <= time.monotonic() - start < _TEST_WALL_BUDGET_S


class TestEligibleTargetPrecheck:
    """A childless L3 that accepted a callable must fail at init() — before any
    startup resource — rather than come up READY yet inert (a callable with no
    process to run on)."""

    def test_childless_l3_with_callable_rejected_at_init(self):
        w = Worker(level=3, num_sub_workers=0)
        w.register(lambda args: None)
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                with pytest.raises(RuntimeError, match="no eligible dispatch target"):
                    w.init()
                # The rejection is pre-resource: nothing was forked, and the
                # epoch never left NEW, so the worker is still constructible-away.
                assert w._sub_pids == []
        finally:
            w.close()

    def test_childless_l3_without_callable_inits(self):
        # No pre-registered callable: a childless L3 is a valid (if inert) host,
        # e.g. targets are registered later once children exist. It must init.
        w = Worker(level=3, num_sub_workers=0)
        with _hard_timeout(_TEST_WALL_BUDGET_S):
            w.init()
            assert w._initialized is True
            w.close()

    def test_sub_backed_l3_with_callable_inits(self):
        w = Worker(level=3, num_sub_workers=1)
        w.register(lambda args: None)
        with _hard_timeout(_TEST_WALL_BUDGET_S):
            w.init()
            assert w._initialized is True
            w.close()

    def test_registered_python_on_chip_only_l3_rejected(self):
        # A registered LOCAL_PYTHON callable is resolved only by a SUB/next-level
        # child, never a chip — so a chip-only L3 (device_ids, no sub/next) has no
        # eligible target for it. Rejected at init before any chip is forked
        # (device-free: the check runs before _start_hierarchical).
        w = Worker(level=3, device_ids=[0])
        w.register(lambda args: None)
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                with pytest.raises(RuntimeError, match="LOCAL_PYTHON callable .* has no eligible"):
                    w.init()
        finally:
            w.close()

    def test_sub_backed_l3_python_callable_inits(self):
        # A LOCAL_PYTHON callable with a SUB resolver is eligible.
        w = Worker(level=3, num_sub_workers=1)
        w.register(lambda args: None)
        with _hard_timeout(_TEST_WALL_BUDGET_S):
            w.init()
            assert w._initialized is True
            w.close()

    @requires_sim_binaries
    def test_chip_backed_l3_with_chip_callable_inits(self, monkeypatch):
        # LOCAL_CHIP positive: the chip child resolves the ChipCallable and
        # prepares it before publishing INIT_READY, so the tree comes up READY.
        with fake_chip_l3(monkeypatch, init=False) as w:
            w.register(chip_callable())
            w.init()
            assert w._initialized is True

    def test_chipless_l3_with_chip_callable_rejected_at_init(self):
        # LOCAL_CHIP negative: a ChipCallable is installed only into chip child
        # loops, so a sub-backed but chipless L3 has no resolver for it.
        w = Worker(level=3, num_sub_workers=1)
        w.register(chip_callable())
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                with pytest.raises(
                    RuntimeError, match=r"LOCAL_CHIP callable .* \(needs a chip device \(device_ids\)\)"
                ):
                    w.init()
                assert w._chip_pids == []
        finally:
            w.close()

    def test_post_init_chip_callable_on_chipless_l3_rejected(self):
        # The same LOCAL_CHIP rule, one epoch later: an L3 that came up without
        # a chip child cannot resolve a ChipCallable handed to it post-init
        # either, so register() must reject it instead of returning an inert
        # handle.
        w = Worker(level=3, num_sub_workers=1)
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                w.init()
                with pytest.raises(ValueError, match=r"\(needs a chip device \(device_ids\)\)"):
                    w.register(chip_callable())
        finally:
            w.close()

    def test_remote_callable_naming_its_own_remote_passes_init_gate(self):
        # REMOTE_TASK_DISPATCHER positive: the callable names a worker id that
        # add_remote_worker returned, so the gate init() runs over the startup
        # snapshot accepts it. Driven directly rather than through init(), which
        # would additionally attach a live remote L3 session.
        w = Worker(level=4, num_sub_workers=0)
        worker_id = w.add_remote_worker(RemoteWorkerSpec(endpoint="127.0.0.1:19073", platform="a2a3sim"))
        handle = w.register(RemoteCallable("pkg.remote:orch"), workers=[worker_id])
        try:
            assert w._resolve_handle(handle).target_namespace == "REMOTE_TASK_DISPATCHER"
            w._validate_eligible_targets()
        finally:
            w.close()

    def test_remote_callable_naming_unknown_worker_is_rejected(self):
        # REMOTE_TASK_DISPATCHER negative: naming an id that add_remote_worker
        # never returned is refused at register, i.e. before the registration
        # can reach the startup snapshot that _validate_eligible_targets scans.
        w = Worker(level=4, num_sub_workers=0)
        worker_id = w.add_remote_worker(RemoteWorkerSpec(endpoint="127.0.0.1:19073", platform="a2a3sim"))
        try:
            with pytest.raises(ValueError, match="workers must name remote worker ids"):
                w.register(RemoteCallable("pkg.remote:orch"), workers=[worker_id + 1])
        finally:
            w.close()

    def test_remote_need_is_the_unmet_named_workers(self):
        # The init-time rule behind both cases above: a REMOTE_TASK_DISPATCHER
        # registration is eligible exactly when every id it names is a live
        # remote worker. Driven directly because register() refuses to build the
        # ineligible registration in the first place.
        w = Worker(level=4, num_sub_workers=0)
        worker_id = w.add_remote_worker(RemoteWorkerSpec(endpoint="127.0.0.1:19073", platform="a2a3sim"))
        try:
            assert w._eligible_target_need("REMOTE_TASK_DISPATCHER", (worker_id,)) is None
            assert "add_remote_worker" in w._eligible_target_need("REMOTE_TASK_DISPATCHER", (worker_id + 1,))
        finally:
            w.close()


class TestTerminalStateContract:
    """CLOSED is terminal: no later API reopens the epoch."""

    def test_init_after_close_is_rejected(self):
        w = Worker(level=3, num_sub_workers=1)
        with _hard_timeout(_TEST_WALL_BUDGET_S):
            w.init()
            w.close()
            import simpler.worker as worker_mod  # noqa: PLC0415

            assert w._lifecycle is worker_mod._Lifecycle.CLOSED
            with pytest.raises(RuntimeError, match="closed"):
                w.init()

    def test_register_after_close_is_rejected(self):
        w = Worker(level=3, num_sub_workers=1)
        with _hard_timeout(_TEST_WALL_BUDGET_S):
            w.init()
            w.close()
            with pytest.raises(RuntimeError, match="closed"):
                w.register(lambda args: None)

    def test_double_close_is_idempotent(self):
        w = Worker(level=3, num_sub_workers=1)
        with _hard_timeout(_TEST_WALL_BUDGET_S):
            w.init()
            w.close()
            w.close()

    def test_init_after_close_claim_is_rejected(self):
        # close() publishes CLOSED atomically at claim (before teardown finishes);
        # a concurrent init() observing CLOSED must be rejected, never reviving
        # the epoch mid-teardown.
        w = Worker(level=3, num_sub_workers=1)
        with _hard_timeout(_TEST_WALL_BUDGET_S):
            w.init()
            w.close()  # -> CLOSED
            with pytest.raises(RuntimeError, match="closed"):
                w.init()

    def test_add_worker_rejects_non_new_child(self):
        # add_worker requires a pristine NEW child (init happens in the forked
        # child process); an already-started/closed child is rejected.
        child = Worker(level=3, num_sub_workers=0)
        with _hard_timeout(_TEST_WALL_BUDGET_S):
            child.init()  # childless L3 comes up READY device-free
            try:
                parent = Worker(level=4, num_sub_workers=0)
                with pytest.raises(RuntimeError, match="must be NEW"):
                    parent.add_worker(child)
            finally:
                child.close()

    def test_add_worker_freezes_child_before_topology_publication(self):
        parent = Worker(level=4, num_sub_workers=0)
        child = Worker(level=3, num_sub_workers=0)
        parent.add_worker(child)

        with child._hierarchical_start_cv:
            assert child._topology_parent is parent
        with pytest.raises(RuntimeError, match="attached as a child"):
            child.init()

    def test_add_worker_rejects_self_attachment(self):
        worker = Worker(level=4, num_sub_workers=0)
        with pytest.raises(ValueError, match="cannot add a Worker to itself"):
            worker.add_worker(worker)

    def test_add_worker_rejects_already_attached_child(self):
        first_parent = Worker(level=4, num_sub_workers=0)
        second_parent = Worker(level=4, num_sub_workers=0)
        child = Worker(level=3, num_sub_workers=0)
        first_parent.add_worker(child)
        with pytest.raises(RuntimeError, match="already attached to another parent"):
            second_parent.add_worker(child)

    def test_add_worker_rejects_attached_parent(self):
        root = Worker(level=5, num_sub_workers=0)
        middle = Worker(level=4, num_sub_workers=0)
        leaf = Worker(level=3, num_sub_workers=0)
        root.add_worker(middle)
        with pytest.raises(RuntimeError, match="already attached as a child"):
            middle.add_worker(leaf)


class TestFailureSurfacing:
    """Every waiter on a failed init observes the same original cause, and a
    BaseException in start unwinds through the same rollback."""

    @staticmethod
    def _fail_start(entered, release, exc):
        def boom(self):
            entered.set()
            assert release.wait(10.0)
            raise exc

        return boom

    def test_all_waiters_get_same_original_failure(self, monkeypatch):
        entered = threading.Event()
        release = threading.Event()
        original = RuntimeError("distinctive start failure")
        monkeypatch.setattr(Worker, "_start_hierarchical", self._fail_start(entered, release, original))

        w = Worker(level=3, num_sub_workers=1, startup_timeout_s=30.0)
        init_err: list = []
        it = threading.Thread(target=lambda: init_err.append(_run_catch(w.init)))
        it.start()
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                assert entered.wait(3.0)
                reg_errs: list = []
                started = threading.Event()
                n = 3

                def do_reg():
                    started.set()
                    reg_errs.append(_run_catch(lambda: w.register(lambda args: None)))

                threads = [threading.Thread(target=do_reg) for _ in range(n)]
                for t in threads:
                    t.start()
                assert started.wait(3.0)
                time.sleep(0.3)  # let the waiters park on the epoch condition
                release.set()
                for t in threads:
                    t.join(10.0)
                it.join(10.0)
                # init surfaced the original; every parked register raised a
                # RuntimeError chained from the SAME original exception object.
                assert init_err[0] is original
                assert len(reg_errs) == n
                for err in reg_errs:
                    assert isinstance(err, RuntimeError)
                    assert err.__cause__ is original
        finally:
            release.set()
            it.join(5.0)
            w.close()

    def test_keyboardinterrupt_before_ready_rolls_back(self, monkeypatch):
        def boom(self):
            raise KeyboardInterrupt()

        monkeypatch.setattr(Worker, "_start_hierarchical", boom)
        w = Worker(level=3, num_sub_workers=1)
        import simpler.worker as worker_mod  # noqa: PLC0415

        with _hard_timeout(_TEST_WALL_BUDGET_S):
            with pytest.raises(KeyboardInterrupt):
                w.init()
            # BaseException funnels into the same rollback: FAILED, no residual.
            assert w._lifecycle is worker_mod._Lifecycle.FAILED
            assert w._sub_pids == []
            w.close()
