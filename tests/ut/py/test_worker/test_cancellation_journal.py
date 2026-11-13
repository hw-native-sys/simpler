# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""P0.2-c cooperative init cancellation and retryable resource cleanup journal."""

import contextlib
import os
import threading
import time
import uuid
from types import SimpleNamespace

import pytest
import simpler.worker as worker_mod
from simpler.worker import CleanupJournal, Worker, _journal_child_survivors, _Lifecycle, _shm_name

from ._harness import TEST_WALL_BUDGET_S, TickingClock, hard_timeout

_TEST_WALL_BUDGET_S = TEST_WALL_BUDGET_S
_DEADLINE_AFTER_WATCHDOG_S = 2 * _TEST_WALL_BUDGET_S
_hard_timeout = hard_timeout


def _run_catch(fn):
    try:
        fn()
        return None
    except BaseException as e:
        return e


def _init_hangs(*_a, **_k):
    time.sleep(3600)


def _init_raises(*_a, **_k):
    raise RuntimeError("injected inner init failure")


def _l3_child(sub_fn=None, num_sub_workers=1):
    l3 = Worker(level=3, num_sub_workers=num_sub_workers)
    l3.register(sub_fn if sub_fn is not None else (lambda args: None))
    return l3


def _trivial_orch(orch, args, config):
    return None


class TestCleanupJournal:
    def test_empty_drive_returns_none(self):
        j = CleanupJournal()
        assert j.empty
        assert j.drive() is None

    def test_success_removes_entry(self):
        j = CleanupJournal()
        called = []
        j.add("child", "test", lambda: called.append(1))
        assert not j.empty
        assert j.drive() is None
        assert j.empty
        assert called == [1]

    def test_failure_keeps_entry(self):
        j = CleanupJournal()
        j.add("child", "test", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        err = j.drive()
        assert err is not None
        assert "boom" in str(err)
        assert not j.empty

    def test_failed_entry_retryable(self):
        j = CleanupJournal()
        attempts = []

        def fail_then_ok():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("first attempt fails")

        j.add("child", "test", fail_then_ok)
        assert j.drive() is not None
        assert not j.empty
        assert j.drive() is None
        assert j.empty
        assert attempts == [1, 1]

    def test_all_entries_attempted_on_failure(self):
        j = CleanupJournal()
        called = []
        j.add("child", "a", lambda: called.append("ok"))
        j.add("child", "b", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        j.add("child", "c", lambda: called.append("also_ok"))
        err = j.drive()
        assert err is not None
        assert called == ["ok", "also_ok"]
        assert not j.empty
        assert len(j) == 1

    def test_errors_accumulated(self):
        j = CleanupJournal()
        j.add("child", "a", lambda: (_ for _ in ()).throw(ValueError("a")))
        j.add("child", "b", lambda: (_ for _ in ()).throw(TypeError("b")))
        j.drive()
        assert len(j.errors) == 2

    def test_extend(self):
        j = CleanupJournal()
        called = []
        j.extend([("child", "a", lambda: called.append("a")), ("child", "b", lambda: called.append("b"))])
        j.drive()
        assert j.empty
        assert called == ["a", "b"]


class TestCloseDuringInitializing:
    def test_close_cancels_init_reaches_closed(self, monkeypatch):
        entered = threading.Event()
        release = threading.Event()
        cancel_latched = threading.Event()
        orig = Worker._start_hierarchical

        def paused_start(self):
            entered.set()
            assert release.wait(10.0)
            return orig(self)

        monkeypatch.setattr(Worker, "_start_hierarchical", paused_start)
        w = Worker(level=3, num_sub_workers=1, startup_timeout_s=_DEADLINE_AFTER_WATCHDOG_S)
        w.register(lambda args: None)
        real_monotonic = worker_mod._monotonic

        def observe_cancel():
            if w._cancel_token:
                cancel_latched.set()
            return real_monotonic()

        monkeypatch.setattr(worker_mod, "_monotonic", observe_cancel)

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
                assert cancel_latched.wait(3.0)
                release.set()
                ct.join(10.0)
                it.join(10.0)
                assert close_result == [None]
                assert w._lifecycle is worker_mod._Lifecycle.CLOSED
        finally:
            release.set()
            it.join(10.0)

    def test_uncooperative_init_bounds_the_close_wait(self, monkeypatch):
        """An init blocked past every cooperative point makes close() raise on
        its unwind deadline instead of blocking forever."""
        entered = threading.Event()
        release = threading.Event()
        orig = Worker._start_hierarchical

        def paused_start(self):
            entered.set()
            assert release.wait(30.0)
            return orig(self)

        monkeypatch.setattr(Worker, "_start_hierarchical", paused_start)
        monkeypatch.setattr(worker_mod, "_CLOSE_CANCEL_UNWIND_TIMEOUT_S", 0.5)
        clock = TickingClock()
        monkeypatch.setattr(worker_mod, "_monotonic", clock.monotonic)
        w = Worker(level=3, num_sub_workers=1, startup_timeout_s=_DEADLINE_AFTER_WATCHDOG_S)
        w.register(lambda args: None)

        def owner_body():
            _run_catch(w.init)

        it = threading.Thread(target=owner_body)
        it.start()
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                assert entered.wait(3.0)
                err = _run_catch(w.close)
                assert isinstance(err, RuntimeError)
                assert "did not unwind" in str(err)
                assert w._lifecycle is worker_mod._Lifecycle.INITIALIZING
        finally:
            release.set()
            it.join(10.0)

    def test_same_thread_close_during_init_rejected(self, monkeypatch):
        entered = threading.Event()
        release = threading.Event()
        orig = Worker._start_hierarchical

        def paused_start(self):
            entered.set()
            err = _run_catch(self.close)
            assert isinstance(err, RuntimeError)
            assert "init-owner" in str(err)
            assert release.wait(10.0)
            return orig(self)

        monkeypatch.setattr(Worker, "_start_hierarchical", paused_start)
        w = Worker(level=3, num_sub_workers=1, startup_timeout_s=_DEADLINE_AFTER_WATCHDOG_S)
        w.register(lambda args: None)
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                release.set()
                w.init()
                assert w._lifecycle is worker_mod._Lifecycle.READY
        finally:
            w.close()


class TestClosedAbsorption:
    def test_closed_absorbing(self):
        w = Worker(level=3, num_sub_workers=1)
        w.register(lambda args: None)
        with w._hierarchical_start_cv:
            w._lifecycle = _Lifecycle.INITIALIZING
        with w._hierarchical_start_cv:
            w._lifecycle = _Lifecycle.CLOSED
        with w._hierarchical_start_cv:
            if w._lifecycle is _Lifecycle.INITIALIZING:
                w._lifecycle = _Lifecycle.FAILED
        assert w._lifecycle is worker_mod._Lifecycle.CLOSED
        w.close()


class TestJournalInAbortHierarchical:
    class _FakeShm:
        def __init__(self):
            self.buf = bytearray(worker_mod.mailbox_size(worker_mod._DEFAULT_PIPELINE_DEPTH_REQUEST))
            self.close_count = 0
            self.unlink_count = 0

        def close(self):
            self.close_count += 1

        def unlink(self):
            self.unlink_count += 1

    def test_unreaped_sigkill_survivor_is_journaled_with_live_shm(self, monkeypatch):
        """Signal delivery is not reap: abort must retain the pid/shm pair."""
        w = Worker(level=3, num_sub_workers=0)
        shm = self._FakeShm()
        w._chip_pids = [12345]
        w._chip_shms = [shm]
        monkeypatch.setattr(os, "kill", lambda *_args: None)
        monkeypatch.setattr(os, "killpg", lambda *_args: None)
        monkeypatch.setattr(os, "waitpid", lambda *_args: (0, 0))

        w._abort_hierarchical(deadline=time.monotonic() - 1.0)

        assert not w._cleanup_journal.empty
        assert shm.close_count == 0
        assert shm.unlink_count == 0

    def test_journal_reclaims_child_already_reaped_elsewhere(self, monkeypatch):
        """ECHILD proves the pid is gone and permits release of its mailbox."""
        journal = CleanupJournal()
        shm = self._FakeShm()
        monkeypatch.setattr(os, "waitpid", lambda *_args: (_ for _ in ()).throw(ChildProcessError()))
        _journal_child_survivors(journal, [], [], [shm], [12345], [], [], set())

        assert journal.drive() is None
        assert journal.empty
        assert shm.close_count == 1
        assert shm.unlink_count == 1

    def test_journal_persists_after_close_failure(self):
        """A journal entry added before close() persists through a failed
        close() and is retried by a second close()."""
        w = Worker(level=3, num_sub_workers=1)
        w.register(lambda args: None)
        attempts = []

        def fail_then_ok():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("transient journal failure")

        w._cleanup_journal.add("child", "test", fail_then_ok)
        # First close: journal drive fails, entry stays.
        err1 = _run_catch(w.close)
        assert err1 is not None
        assert "transient journal failure" in str(err1)
        assert not w._cleanup_journal.empty
        assert w._lifecycle is worker_mod._Lifecycle.CLOSED
        # The attempt is incomplete — journal still has entries.
        assert w._close_completion is not None
        assert w._close_completion.incomplete
        assert w._close_completion.done
        # Second close: retries the journal, succeeds.
        err2 = _run_catch(w.close)
        assert err2 is None
        assert w._cleanup_journal.empty
        assert attempts == [1, 1]

    def test_fully_reaped_abort_leaves_empty_journal(self, monkeypatch):
        l3 = _l3_child()
        l3.init = _init_raises
        w4 = Worker(level=4, num_sub_workers=0, startup_timeout_s=_DEADLINE_AFTER_WATCHDOG_S)
        w4.register(_trivial_orch)
        w4.add_worker(l3)
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                with pytest.raises(RuntimeError, match="injected inner init failure"):
                    w4.init()
            assert w4._cleanup_journal.empty
            assert w4._next_level_pids == []
        finally:
            w4.close()


class TestJournalRetryOnClose:
    def test_domain_retirement_error_survives_failed_child_exit(self, monkeypatch):
        worker = Worker(level=3, num_sub_workers=0)
        domain = worker_mod.CommDomainHandle(
            name="rejected", workers=(), contexts={}, allocation_id=7, _release_fn=lambda _handle: None
        )
        domain._released = True
        worker._domain_free_results[7] = RuntimeError("backend reclamation failed")
        worker._defer_domain_retirement(domain)
        child_error = RuntimeError("child teardown: pid 123 exited with code 1")

        def reap(_deadline):
            raise child_error

        monkeypatch.setattr(worker, "_reclaim_child_groups", reap)
        try:
            with pytest.raises(RuntimeError, match="window reclamation is unconfirmed") as failure:
                worker._teardown_ready_tree()
            assert failure.value.__cause__ is child_error
            assert "child teardown: pid 123 exited with code 1" in str(failure.value)
            assert not domain.freed
            assert [(kind, identity) for kind, identity, _ in worker._cleanup_journal._entries] == [
                ("domain-retirement", "True:7")
            ]
            monkeypatch.setattr(worker, "_reclaim_child_groups", lambda _deadline: None)
            with pytest.raises(RuntimeError, match="window reclamation is unconfirmed"):
                worker._teardown_ready_tree()
            assert not domain.freed
        finally:
            monkeypatch.setattr(worker, "_reclaim_child_groups", lambda _deadline: None)
            worker._domain_free_results[7] = None
            worker.close()

    def test_child_journal_driven_after_shutdown_phase(self, monkeypatch):
        """A retained child cannot fence the SHUTDOWN that lets it converge."""
        w = Worker(level=3, num_sub_workers=1)
        w.register(lambda args: None)
        events = []
        w._cleanup_journal.add("child", "test", lambda: events.append("journal"))
        monkeypatch.setattr(w, "_broadcast_child_shutdown", lambda _shms: events.append("shutdown"))
        w.close()
        assert events[-1] == "journal"
        assert "shutdown" in events[:-1]
        assert w._cleanup_journal.empty

    def test_journal_retry_on_close(self):
        """A journal entry that fails on first close stays in the journal
        and is retried by a second close()."""
        w = Worker(level=3, num_sub_workers=1)
        w.register(lambda args: None)
        attempts = []

        def fail_then_ok():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("transient failure")

        w._cleanup_journal.add("child", "test", fail_then_ok)
        # First close: journal drive fails, entry stays, error surfaces.
        err1 = _run_catch(w.close)
        assert err1 is not None
        assert "transient failure" in str(err1)
        assert not w._cleanup_journal.empty
        assert w._lifecycle is worker_mod._Lifecycle.CLOSED
        # Second close: retries the journal entry, succeeds.
        err2 = _run_catch(w.close)
        assert err2 is None
        assert w._cleanup_journal.empty
        assert attempts == [1, 1]

    @pytest.mark.parametrize("startup_abort", [False, True])
    @pytest.mark.parametrize(
        ("method_name", "kind"),
        [
            ("_sweep_region_instances", "region"),
            ("_release_all_live_domains", "domain"),
            ("_flush_pending_remote_frees", "remote"),
            ("_release_all_buffers", "buffer"),
        ],
    )
    def test_both_teardown_paths_retry_each_resource_class(self, monkeypatch, startup_abort, method_name, kind):
        w = Worker(level=3, num_sub_workers=0)
        attempts = []

        def fail_then_ok():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError(f"injected {kind} cleanup failure")

        monkeypatch.setattr(w, method_name, fail_then_ok)
        with pytest.raises(RuntimeError, match=f"injected {kind}"):
            w._teardown_worker_tree(startup_abort=startup_abort)
        assert len(w._cleanup_journal) == 1

        w._teardown_worker_tree(startup_abort=startup_abort)
        assert w._cleanup_journal.empty
        assert attempts == [1, 1]


class TestShmNames:
    def test_shm_name_with_token(self):
        assert _shm_name("abc123456789", "sub-0") == "sp-abc12345-sub-0"

    def test_shm_name_without_token(self):
        assert _shm_name("", "sub-0") is None

    def test_root_assigns_unique_tokens_to_the_whole_nested_tree(self):
        l3a = Worker(level=3, num_sub_workers=0)
        l3b = Worker(level=3, num_sub_workers=0)
        l4 = Worker(level=4, num_sub_workers=0)
        l4.add_worker(l3a)
        l4.add_worker(l3b)
        l5 = Worker(level=5, num_sub_workers=0)
        l5.add_worker(l4)

        l5._assign_shm_namespace()

        tokens = {l5._shm_token, l4._shm_token, l3a._shm_token, l3b._shm_token}
        assert len(tokens) == 4
        assert l5._shm_tree_tokens == tokens
        assert l4._shm_tree_tokens == {l4._shm_token, l3a._shm_token, l3b._shm_token}

    def test_root_namespace_scan_reclaims_nested_only(self):
        if not os.path.isdir("/dev/shm"):
            pytest.skip("namespace scanning requires Linux /dev/shm")
        nested = Worker(level=3, num_sub_workers=0)
        root = Worker(level=4, num_sub_workers=0)
        root.add_worker(nested)
        root._assign_shm_namespace()
        nested_name = _shm_name(nested._shm_token, "sub-99")
        foreign_name = _shm_name(uuid.uuid4().hex, "sub-99")
        nested_shm = worker_mod.SharedMemory(create=True, size=8, name=nested_name)
        foreign_shm = worker_mod.SharedMemory(create=True, size=8, name=foreign_name)
        nested_shm.close()
        foreign_shm.close()
        try:
            root._unlink_shm_namespace(set())
            with pytest.raises(FileNotFoundError):
                worker_mod.SharedMemory(name=nested_name)
            reopened = worker_mod.SharedMemory(name=foreign_name)
            reopened.close()
        finally:
            with contextlib.suppress(FileNotFoundError):
                nested_shm.unlink()
            with contextlib.suppress(FileNotFoundError):
                foreign_shm.unlink()

    @pytest.mark.parametrize("kind", ["a3", "h"])
    def test_host_backing_sweep_waits_for_the_creator_group(self, monkeypatch, kind):
        posixshmem = pytest.importorskip("_posixshmem")
        nested = Worker(level=3, num_sub_workers=0)
        root = Worker(level=4, num_sub_workers=0)
        root.add_worker(nested)
        root._assign_shm_namespace()
        own_group = os.getpgrp()
        child_group = own_group + 1
        backing = _shm_name(nested._shm_token, f"{kind}-{child_group:x}-backing")
        local = _shm_name(root._shm_token, f"{kind}-{own_group:x}-local")
        foreign = _shm_name(uuid.uuid4().hex, f"{kind}-{child_group:x}-foreign")
        invalid = _shm_name(nested._shm_token, f"{kind}-0-invalid")
        names = [backing, local, foreign, invalid]
        unlinked = []
        probe_error = None

        def probe(group, sig):
            assert (group, sig) == (child_group, 0)
            if probe_error is not None:
                raise probe_error

        with monkeypatch.context() as patch:
            patch.setattr(os.path, "isdir", lambda path: path == "/dev/shm")
            patch.setattr(os, "scandir", lambda _path: [SimpleNamespace(name=name) for name in names])
            patch.setattr(os, "killpg", probe)
            patch.setattr(posixshmem, "shm_unlink", unlinked.append)
            with pytest.raises(RuntimeError, match="still alive"):
                root._reclaim_child_groups(time.monotonic())
            assert unlinked == [] and not root._cleanup_journal.empty

            probe_error = PermissionError("group visibility denied")
            assert root._cleanup_journal.drive_kinds({"shm"}) is probe_error
            assert unlinked == [] and not root._cleanup_journal.empty

            probe_error = ProcessLookupError()
            assert root._cleanup_journal.drive_kinds({"shm"}) is None
            assert root._cleanup_journal.empty
            assert unlinked == [f"/{backing}"]

    @pytest.mark.parametrize("kind", ["a3", "h"])
    def test_local_host_backing_sweep_waits_for_l2_and_retries_unlink(self, monkeypatch, kind):
        posixshmem = pytest.importorskip("_posixshmem")
        worker = Worker(level=3, num_sub_workers=0)
        worker._assign_shm_namespace()
        backing = _shm_name(worker._shm_token, f"{kind}-{os.getpgrp():x}-backing")
        calls = []
        child_alive = True
        unlink_fails = True

        def reap_l2():
            if child_alive:
                raise RuntimeError("L2 still alive")

        def unlink(name):
            calls.append(name)
            if unlink_fails:
                raise PermissionError("injected Host unlink failure")

        monkeypatch.setattr(os.path, "isdir", lambda _path: True)
        monkeypatch.setattr(os, "scandir", lambda _path: [SimpleNamespace(name=backing)])
        monkeypatch.setattr(posixshmem, "shm_unlink", unlink)
        worker._cleanup_journal.add("child", "chip survivor", reap_l2)
        with pytest.raises(RuntimeError, match="unconfirmed L2 child exit"):
            worker._reclaim_child_groups(time.monotonic())
        assert calls == []

        child_alive = False
        assert worker._cleanup_journal.drive_kinds({"child"}) is None
        error = worker._cleanup_journal.drive_kinds({"shm"})
        assert isinstance(error, PermissionError)
        assert not worker._cleanup_journal.empty

        unlink_fails = False
        assert worker._cleanup_journal.drive_kinds({"shm"}) is None
        assert worker._cleanup_journal.empty
        assert calls == [f"/{backing}"] * 2

    @pytest.mark.parametrize("nested", [False, True], ids=["local", "nested"])
    def test_comm_marker_sweep_waits_for_l2_and_retries_unlink(self, monkeypatch, tmp_path, nested):
        l3 = Worker(level=3, num_sub_workers=0)
        owner = Worker(level=4, num_sub_workers=0) if nested else l3
        if nested:
            owner.add_worker(l3)
        owner._assign_shm_namespace()
        monkeypatch.setattr(worker_mod, "_COMM_ROOTINFO_DIR", str(tmp_path))

        group = os.getpgrp() + int(nested)
        with monkeypatch.context() as patch:
            patch.setattr(os, "getpgrp", lambda: group)
            rootinfo = os.path.basename(l3._comm_plan_rootinfo_path())
        marker = tmp_path / f"{rootinfo}.destroy_failed.1"
        other_rootinfo = f"pto_multi_comm_{uuid.uuid4().hex}_{group:x}.bin"
        unrelated = tmp_path / f"{other_rootinfo}.destroy_released.1.2"
        unrelated_ready = tmp_path / f"barrier_{other_rootinfo}_alloc_1_host_ready_1_1.ready"
        unrelated_pending = tmp_path / f"barrier_{other_rootinfo}_alloc_2_host_ready_1_1.ready.tmp"
        blocked = tmp_path / f"{rootinfo}.destroy_released.1.1"
        blocked.mkdir()
        release = tmp_path / f"{rootinfo}.destroy_released.1.2"
        ready = tmp_path / f"barrier_{rootinfo}_alloc_1_host_ready_1_1.ready"
        pending = tmp_path / f"barrier_{rootinfo}_alloc_2_host_ready_1_1.ready.tmp"
        failed = tmp_path / f"barrier_{rootinfo}_alloc_1_host_failed_1_0.ready"
        owned = (marker, release, ready, pending, failed)
        for path in (*owned, unrelated, unrelated_ready, unrelated_pending):
            path.touch()

        child_alive = True

        def reap_l2():
            if child_alive:
                raise RuntimeError("L2 still alive")

        def probe_group(pgid, sig):
            assert (pgid, sig) == (group, 0)
            if not child_alive:
                raise ProcessLookupError

        if nested:
            # The L3 leader is already reaped. Its L2 descendants still own the group.
            monkeypatch.setattr(os, "killpg", probe_group)
        else:
            owner._cleanup_journal.add("child", "chip survivor", reap_l2)
        with pytest.raises(RuntimeError, match="(still alive|unconfirmed L2 child exit)"):
            owner._reclaim_child_groups(time.monotonic())
        assert all(path.exists() for path in owned)

        child_alive = False
        assert owner._cleanup_journal.drive_kinds({"child"}) is None
        unlink = os.unlink

        def fail_unlink(path, *args, **kwargs):
            if os.fspath(path) == str(failed):
                raise PermissionError("injected marker unlink failure")
            unlink(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(os, "unlink", fail_unlink)
            assert isinstance(owner._cleanup_journal.drive_kinds({"shm"}), PermissionError)
            assert failed.exists()

        assert owner._cleanup_journal.drive_kinds({"shm"}) is None
        assert owner._cleanup_journal.empty
        assert all(not path.exists() for path in owned)
        assert unrelated.exists() and unrelated_ready.exists() and unrelated_pending.exists() and blocked.is_dir()


class TestNewWorkerClose:
    def test_new_worker_close_journal_empty(self):
        w = Worker(level=3, num_sub_workers=1)
        w.close()
        assert w._cleanup_journal.empty
        assert w._lifecycle is worker_mod._Lifecycle.CLOSED
