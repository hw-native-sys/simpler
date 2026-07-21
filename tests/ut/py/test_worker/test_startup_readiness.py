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
import signal
import struct
import threading
import time
from contextlib import contextmanager
from multiprocessing.shared_memory import SharedMemory

import pytest
from simpler.task_interface import CallConfig, TaskArgs
from simpler.worker import Worker

# Hard wall budget for a single failure scenario — comfortably above the
# injected startup_timeout_s values below, well under any real hang.
_TEST_WALL_BUDGET_S = 30.0


@contextmanager
def _hard_timeout(seconds: float, msg: str = "startup did not return within the hard test budget"):
    """Abort the body with TimeoutError if it runs longer than ``seconds``.

    A backstop against a regression that reintroduces an unbounded startup
    spin: instead of hanging CI, the test fails with TimeoutError. Uses SIGALRM
    (pytest runs on the main thread), which interrupts the barrier's
    ``time.sleep`` poll.
    """

    def _handler(_signum, _frame):
        raise TimeoutError(msg)

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


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
            w4.add_worker(l3)
            w4.init()

            def l4_orch(orch, args, config):
                orch.submit_next_level(l3_handle, TaskArgs(), CallConfig())

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
            w4.add_worker(l3a)
            w4.add_worker(l3b)
            w4.init()

            def l4_orch(orch, args, config):
                orch.submit_next_level(ha, TaskArgs(), CallConfig())
                orch.submit_next_level(hb, TaskArgs(), CallConfig())

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

    def test_close_during_initializing_fails_fast(self, monkeypatch):
        # close() does not cancel an in-progress init: it fails fast so the
        # INITIALIZING epoch is never torn down under its owner. The owner
        # completes init and closes the READY tree itself.
        import simpler.worker as worker_mod  # noqa: PLC0415

        entered = threading.Event()
        release = threading.Event()
        monkeypatch.setattr(Worker, "_start_hierarchical", self._pause_start(entered, release))

        w = Worker(level=3, num_sub_workers=1, startup_timeout_s=30.0)
        w.register(lambda args: None)
        proceed = threading.Event()

        def owner_body():
            _run_catch(w.init)
            proceed.wait(10.0)
            _run_catch(w.close)  # owner closes the READY tree

        it = threading.Thread(target=owner_body)
        it.start()
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                assert entered.wait(3.0)
                # A concurrent close() while INITIALIZING is rejected outright.
                with pytest.raises(RuntimeError, match="in progress"):
                    w.close()
                assert w._lifecycle is worker_mod._Lifecycle.INITIALIZING
                release.set()  # let init finish -> READY
        finally:
            release.set()
            proceed.set()
            it.join(10.0)


class _FakeChipOk:
    """Stand-in for a ChipWorker whose init succeeds — no NPU touched."""

    def init(self, *_a, **_k):
        pass

    def _register_callable_at_slot(self, *_a, **_k):  # pragma: no cover
        pass

    def finalize(self):
        pass


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
        import simpler.worker as worker_mod  # noqa: PLC0415

        import simpler_setup.runtime_builder as rb_mod  # noqa: PLC0415

        class _FakeBuilder:
            def __init__(self, *_a, **_k):
                pass

            def get_binaries(self, *_a, **_k):
                return object()

        # Mock the device/runtime layer so this stays device-free (no sim
        # binaries required) — the regression is purely the lifecycle state.
        monkeypatch.setattr(worker_mod, "ChipWorker", _FakeChipOk)
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
        monkeypatch.setattr(Worker, "_release_all_host_buffers", lambda self: time.sleep(0.6))
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

        monkeypatch.setattr(Worker, "_run_locked", paused_run)
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

        monkeypatch.setattr(Worker, "_run_locked", reentrant_run)
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

        monkeypatch.setattr(Worker, "_run_locked", paused_run)
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

        monkeypatch.setattr(Worker, "_run_locked", paused_run)
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

    def test_l2_close_during_initializing_fails_fast(self, monkeypatch):
        # close() cannot cancel an in-progress L2 init: it fails fast, the owner
        # finishes init to READY, and the owner then finalizes the device
        # same-thread. No cross-thread finalize, no resurrected epoch.
        import simpler.worker as worker_mod  # noqa: PLC0415

        entered = threading.Event()
        release = threading.Event()
        finalized = {"n": 0}
        proceed = threading.Event()

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
            proceed.wait(10.0)
            _run_catch(w.close)  # owner finalizes the device same-thread

        t1 = threading.Thread(target=owner_body)
        t1.start()
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                assert entered.wait(3.0)
                with pytest.raises(RuntimeError, match="in progress"):
                    w.close()
                assert w._lifecycle is worker_mod._Lifecycle.INITIALIZING
                release.set()  # init completes -> READY
                proceed.set()  # owner then closes
                t1.join(10.0)
                assert init_err[0] is None
                assert w._lifecycle is worker_mod._Lifecycle.CLOSED
                assert finalized["n"] == 1  # device finalized once, same-thread
        finally:
            release.set()
            proceed.set()
            t1.join(5.0)


class _FakeChipRaises:
    """Stand-in for ChipWorker whose init raises — no NPU touched."""

    def init(self, *_a, **_k):
        raise RuntimeError("injected chip init failure")

    def finalize(self):  # pragma: no cover - never reached (init raises)
        pass


class _FakeChipHangs:
    def init(self, *_a, **_k):
        time.sleep(3600)

    def finalize(self):  # pragma: no cover
        pass


def _sim_binaries_available() -> bool:
    try:
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        RuntimeBuilder("a2a3sim").get_binaries("tensormap_and_ringbuffer")
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _sim_binaries_available(), reason="a2a3sim runtime binaries not built")
class TestChipStartupFailure:
    """Chip (L2) startup failure — device-free via a faked ChipWorker on a2a3sim.

    Constructing an L3 with ``device_ids`` only reads the prebuilt runtime
    binaries; the forked chip child instantiates ``worker.ChipWorker``, which
    the test replaces so no silicon is required. The failure trips before
    ``dw.init()``, so the sim runtime is never actually driven. Exercises the
    same parent-side readiness barrier as the next-level edge (the #1003 spin at
    the former ``while ... != INIT_DONE`` was on this chip path).
    """

    def _make_l3(self, timeout_s):
        return Worker(
            level=3,
            device_ids=[0],
            platform="a2a3sim",
            runtime="tensormap_and_ringbuffer",
            num_sub_workers=0,
            startup_timeout_s=timeout_s,
        )

    def test_chip_init_failure_raises_bounded(self, monkeypatch):
        import simpler.worker as worker_mod  # noqa: PLC0415

        monkeypatch.setattr(worker_mod, "ChipWorker", _FakeChipRaises)
        l3 = self._make_l3(timeout_s=10.0)  # chip-only; the chip forks from device_ids
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                with pytest.raises(RuntimeError, match="injected chip init failure"):
                    l3.init()
        finally:
            l3.close()

    def test_chip_init_hang_trips_deadline(self, monkeypatch):
        import simpler.worker as worker_mod  # noqa: PLC0415

        monkeypatch.setattr(worker_mod, "ChipWorker", _FakeChipHangs)
        l3 = self._make_l3(timeout_s=1.5)  # chip-only; the chip forks from device_ids
        start = time.monotonic()
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                with pytest.raises(RuntimeError, match="deadline"):
                    l3.init()
            assert 1.5 <= time.monotonic() - start < _TEST_WALL_BUDGET_S
        finally:
            l3.close()


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
