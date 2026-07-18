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
        w4.init()
        start = time.monotonic()
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                with pytest.raises(RuntimeError, match="injected inner init failure"):
                    w4.run(_trivial_orch)
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
        w4.init()
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                with pytest.raises(RuntimeError, match="exited during init"):
                    w4.run(_trivial_orch)
        finally:
            w4.close()

    def test_startup_deadline_fires_on_hung_child(self):
        """A child that hangs in init trips the startup deadline, not an infinite spin."""
        l3 = _l3_child()
        l3.init = _init_hangs  # noqa: SLF001

        w4 = Worker(level=4, num_sub_workers=0, startup_timeout_s=1.5)
        w4.register(_trivial_orch)
        w4.add_worker(l3)
        w4.init()
        start = time.monotonic()
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                with pytest.raises(RuntimeError, match="deadline"):
                    w4.run(_trivial_orch)
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

        def spy_abort(self):
            captured["pids"] = list(self._chip_pids) + list(self._sub_pids) + list(self._next_level_pids)
            return orig_abort(self)

        monkeypatch.setattr(Worker, "_abort_hierarchical", spy_abort)

        w4 = Worker(level=4, num_sub_workers=0, startup_timeout_s=1.0)
        w4.register(_trivial_orch)
        w4.add_worker(l3)
        w4.init()
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                with pytest.raises(RuntimeError):
                    w4.run(_trivial_orch)

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
        w4.init()
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                with pytest.raises(RuntimeError, match="slow inner init failure"):
                    w4.run(_trivial_orch)

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
        w4.init()
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                with pytest.raises(RuntimeError, match="injected inner init failure"):
                    w4.run(_trivial_orch)
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
        w3.init()
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                with pytest.raises(RuntimeError, match="sub worker .* exited during init"):
                    w3.run(_trivial_orch)
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
    startup and that tasks dispatch afterwards. They do NOT assert full L4->L3->L2
    eager readiness: L3 descendants are still started lazily on first inner
    run(), so INIT_READY here means the L3's own init succeeded.
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
        l3 = self._make_l3(timeout_s=10.0)
        l3.register(lambda args: None)
        l3.init()
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                with pytest.raises(RuntimeError, match="injected chip init failure"):
                    l3.run(lambda orch, args, config: None)
        finally:
            l3.close()

    def test_chip_init_hang_trips_deadline(self, monkeypatch):
        import simpler.worker as worker_mod  # noqa: PLC0415

        monkeypatch.setattr(worker_mod, "ChipWorker", _FakeChipHangs)
        l3 = self._make_l3(timeout_s=1.5)
        l3.register(lambda args: None)
        l3.init()
        start = time.monotonic()
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                with pytest.raises(RuntimeError, match="deadline"):
                    l3.run(lambda orch, args, config: None)
            assert 1.5 <= time.monotonic() - start < _TEST_WALL_BUDGET_S
        finally:
            l3.close()
