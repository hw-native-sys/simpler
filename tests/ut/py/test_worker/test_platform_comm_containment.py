# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU process containment for the hardware communication test's actual parent."""

from __future__ import annotations

import ctypes
import io
import multiprocessing as mp
import os
import queue
import signal
import sys
from multiprocessing import popen_fork
from types import ModuleType, SimpleNamespace

import pytest

from . import test_platform_comm as comm_test


class _RankProcessCase:
    def __init__(self, monkeypatch, tmp_path):
        self.ctx = mp.get_context("fork")
        self.results = self.ctx.Queue()
        self.entered = self.ctx.Queue()
        self.entered_counts = self.ctx.Array("i", 2)
        self.read_fd, self.write_fd = os.pipe()
        self.processes = []
        self.unpublished = []
        self.os_waitpid, self.os_kill = os.waitpid, os.kill
        self.queue_error = queue.Empty("controlled rank result failure")
        self.fail_queue = True
        self.report = False
        self.reported_error = False
        self.hold = True
        self.start_failure = None
        self.start_error = RuntimeError("controlled second start failure")
        self.exit_code = 0
        self.ignore_term = False
        self.terminate_error = False
        self.close_error = False
        self.reap_error = False
        self.actions = []
        self.get_timeouts = []
        self.path = tmp_path / "rootinfo.bin"
        self.path.write_text("owned rootinfo")
        builder = ModuleType("simpler_setup.runtime_builder")
        monkeypatch.setattr(
            builder, "RuntimeBuilder", lambda **kwargs: SimpleNamespace(get_binaries=lambda *a, **k: {}), raising=False
        )
        monkeypatch.setitem(sys.modules, builder.__name__, builder)
        monkeypatch.setattr(comm_test, "_rank_entry", self.rank_entry)
        monkeypatch.setattr(comm_test, "mp", SimpleNamespace(get_context=lambda _: self))

    def Queue(self):  # noqa: N802 -- matches multiprocessing context
        return self

    def Pipe(self):  # noqa: N802 -- matches multiprocessing context
        return self.ctx.Pipe()

    def Process(self, **kwargs):  # noqa: N802 -- matches multiprocessing context
        process = self.ctx.Process(**kwargs)
        self.processes.append(process)
        return _ProcessControl(self, process, len(self.processes) - 1)

    def get(self, timeout):
        self.get_timeouts.append(timeout)
        if self.fail_queue:
            assert sorted(self.entered.get(timeout=5) for _ in range(2)) == [0, 1]
            raise self.queue_error
        return self.results.get(timeout=min(timeout, 5))

    def put(self, result):
        self.results.put(result)

    def rank_entry(self, rank, nranks, device_id, bins, rootinfo_path, result_queue):
        with self.entered_counts.get_lock():
            self.entered_counts[rank] += 1
        if self.ignore_term:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        self.entered.put(rank)
        if self.report:
            result_queue.put(
                {
                    "rank": rank,
                    "ok": not self.reported_error,
                    "barrier_ok": True,
                    "stage": "memcpy",
                    "error": "controlled rank failure",
                }
            )
            # Exit-code tests must deliver the report before os._exit.
            self.results.close()
            self.results.join_thread()
        if self.hold:
            os.read(self.read_fd, 1)
        if rank == 0 and self.exit_code:
            os._exit(self.exit_code)

    def run(self):
        comm_test.test_two_rank_comm_lifecycle("a5", [5, 6], self.path.parent)

    def close(self):
        self.results.close()
        if self.close_error:
            raise RuntimeError("controlled queue close failure")

    def join_thread(self):
        return self.results.join_thread()

    def reap(self):
        # The unpublished Popen is tracked independently of the tested parent.
        reap_records = []
        for popen in self.unpublished:
            try:
                pid, status = self.os_waitpid(popen.pid, os.WNOHANG)
                if pid == 0:
                    self.os_kill(popen.pid, signal.SIGKILL)
                    pid, status = self.os_waitpid(popen.pid, 0)
                assert pid == popen.pid
                reap_records.append((popen.pid, "independent-reap", os.waitstatus_to_exitcode(status)))
            except ChildProcessError:
                reap_records.append((popen.pid, "already-reaped"))
            popen.close()
        # Only the probe's own children are affected, including on original RED.
        for process in self.processes:
            if process.pid is not None and process.is_alive():
                process.kill()
        for process in self.processes:
            if process.pid is not None:
                process.join(timeout=5)
                assert not process.is_alive()
                reap_records.append((process.pid, "process-reaped", process.exitcode))
            process.close()
        print(f"[comm-containment independent-teardown] {reap_records!r}", file=sys.stderr, flush=True)
        os.close(self.read_fd)
        os.close(self.write_fd)
        self.entered.close()
        self.entered.join_thread()
        self.results.close()
        self.results.join_thread()
        self.path.unlink(missing_ok=True)


class _ProcessControl:
    def __init__(self, case, process, rank):
        self.case = case
        self.process = process
        self.rank = rank
        self.killed = False

    def __getattr__(self, name):
        return getattr(self.process, name)

    def start(self):
        if self.rank == 1 and self.case.start_failure:
            assert self.case.entered.get(timeout=5) == 0
            if self.case.start_failure == "after":
                self.process.start()
            if self.case.start_failure == "unpublished":
                original_launch = popen_fork.Popen._launch

                def interrupted_launch(popen, process):
                    original_launch(popen, process)
                    self.case.unpublished.append(popen)
                    assert process.pid is None
                    raise self.case.start_error

                with pytest.MonkeyPatch.context() as patch:
                    patch.setattr(popen_fork.Popen, "_launch", interrupted_launch)
                    self.process.start()
            raise self.case.start_error
        self.process.start()

    def join(self, timeout):
        if self.case.reap_error and self.rank == 0 and self.killed:
            self.process.join(timeout=5)
            raise RuntimeError("controlled reap proof failure")
        self.case.actions.append(("join", self.rank, timeout))
        if timeout == 60 and self.case.hold:
            # The report is delivered but this child cannot exit until its
            # owned pipe writer closes. No elapsed window establishes that.
            self.process.join(timeout=0)
        elif self.case.hold and not self.killed and (self.case.ignore_term or self.case.terminate_error):
            self.process.join(timeout=0)
        else:
            self.process.join(timeout=min(timeout, 5))

    def terminate(self):
        self.case.actions.append(("terminate", self.rank))
        if self.case.terminate_error and self.rank == 0:
            raise RuntimeError("controlled terminate failure")
        self.process.terminate()

    def kill(self):
        self.case.actions.append(("kill", self.rank))
        self.killed = True
        self.process.kill()


@pytest.fixture
def rank_case(monkeypatch, tmp_path):
    case = _RankProcessCase(monkeypatch, tmp_path)
    try:
        yield case
    finally:
        case.reap()


def test_queue_failure_reaps_both_started_ranks_and_preserves_error(rank_case):
    with pytest.raises(queue.Empty) as caught:
        rank_case.run()
    assert caught.value is rank_case.queue_error
    assert all(not process.is_alive() for process in rank_case.processes)
    assert not rank_case.path.exists()


@pytest.mark.parametrize("when", ("before", "after"))
def test_partial_start_reaps_every_child_and_preserves_start_error(rank_case, when):
    rank_case.start_failure = when
    with pytest.raises(RuntimeError) as caught:
        rank_case.run()
    assert caught.value is rank_case.start_error
    assert all(not process.is_alive() for process in rank_case.processes)
    if when == "before":
        assert rank_case.path.read_text() == "owned rootinfo"
    else:
        assert not rank_case.path.exists()


@pytest.mark.parametrize("error_kind", (RuntimeError, KeyboardInterrupt))
def test_startup_interrupted_before_popen_publication_reaps_exact_os_child(rank_case, error_kind):
    rank_case.start_failure = "unpublished"
    rank_case.start_error = error_kind("controlled fork-before-publication interruption")
    with pytest.raises(error_kind) as caught:
        rank_case.run()
    assert caught.value is rank_case.start_error
    assert len(rank_case.unpublished) == 1
    assert rank_case.processes[1].pid is None
    with pytest.raises(ChildProcessError):
        os.waitpid(rank_case.unpublished[0].pid, os.WNOHANG)
    assert rank_case.processes[0].exitcode is not None
    assert list(rank_case.entered_counts) == [1, 0]
    assert not rank_case.path.exists()


def test_unpublished_pid_without_parenthood_proof_keeps_rootinfo(rank_case, monkeypatch, capsys):
    rank_case.start_failure = "unpublished"
    waitpid, kill = os.waitpid, os.kill
    signalled = []

    def refuse_unpublished(pid, options):
        if any(popen.pid == pid for popen in rank_case.unpublished):
            raise ChildProcessError("controlled missing child ownership proof")
        return waitpid(pid, options)

    def note_kill(pid, sig):
        signalled.append(pid)
        return kill(pid, sig)

    monkeypatch.setattr(os, "waitpid", refuse_unpublished)
    monkeypatch.setattr(os, "kill", note_kill)
    with pytest.raises(RuntimeError) as caught:
        rank_case.run()
    assert caught.value is rank_case.start_error
    assert len(rank_case.unpublished) == 1
    assert rank_case.unpublished[0].pid not in signalled
    assert rank_case.path.read_text() == "owned rootinfo"
    assert "controlled missing child ownership proof" in capsys.readouterr().err
    assert list(rank_case.entered_counts) == [1, 0]


def test_fixture_and_parent_use_only_the_fixture_rootinfo(rank_case, tmp_path):
    assert rank_case.path == tmp_path / "rootinfo.bin"
    with pytest.raises(queue.Empty):
        rank_case.run()
    assert not rank_case.path.exists()


def test_successful_reports_do_not_hide_a_child_that_cannot_exit(rank_case):
    rank_case.fail_queue = False
    rank_case.report = True
    with pytest.raises(RuntimeError, match="reported a result but did not exit"):
        rank_case.run()
    assert all(not process.is_alive() for process in rank_case.processes)
    assert not rank_case.path.exists()
    assert rank_case.get_timeouts == [180, 180]
    assert sum(action == ("join", 0, 60) for action in rank_case.actions) == 1
    assert sum(action == ("join", 1, 60) for action in rank_case.actions) == 1


def test_successful_reports_do_not_hide_a_nonzero_process_exit(rank_case):
    rank_case.fail_queue = False
    rank_case.report = True
    rank_case.hold = False
    rank_case.exit_code = 17
    with pytest.raises(RuntimeError, match="exited with code 17"):
        rank_case.run()
    assert all(not process.is_alive() for process in rank_case.processes)
    assert not rank_case.path.exists()


def test_normal_success_reaps_both_ranks(rank_case):
    rank_case.fail_queue = False
    rank_case.report = True
    rank_case.hold = False
    rank_case.run()
    assert [process.exitcode for process in rank_case.processes] == [0, 0]
    assert not rank_case.path.exists()
    assert rank_case.get_timeouts == [180, 180]
    assert not any(action[0] in ("terminate", "kill") for action in rank_case.actions)


def test_cleanup_signals_all_ranks_before_waiting_and_kills_survivors(rank_case):
    rank_case.ignore_term = True
    with pytest.raises(queue.Empty) as caught:
        rank_case.run()
    assert caught.value is rank_case.queue_error
    assert rank_case.actions[:2] == [("terminate", 0), ("terminate", 1)]
    assert sorted(action[1] for action in rank_case.actions if action[0] == "kill") == [0, 1]
    assert [process.exitcode for process in rank_case.processes] == [-signal.SIGKILL, -signal.SIGKILL]
    assert not rank_case.path.exists()


def test_cleanup_error_keeps_original_failure_and_handles_other_rank(rank_case, capsys):
    rank_case.terminate_error = True
    with pytest.raises(queue.Empty) as caught:
        rank_case.run()
    assert caught.value is rank_case.queue_error
    assert rank_case.actions[:2] == [("terminate", 0), ("terminate", 1)]
    assert all(not process.is_alive() for process in rank_case.processes)
    assert "controlled terminate failure" in capsys.readouterr().err
    assert not rank_case.path.exists()


def test_cleanup_failure_turns_an_otherwise_successful_run_into_failure(rank_case):
    rank_case.fail_queue = False
    rank_case.report = True
    rank_case.hold = False
    rank_case.close_error = True
    with pytest.raises(RuntimeError, match="controlled queue close failure"):
        rank_case.run()
    assert all(not process.is_alive() for process in rank_case.processes)


def test_unconfirmed_reap_keeps_rootinfo_and_original_failure(rank_case, capsys):
    rank_case.ignore_term = True
    rank_case.reap_error = True
    with pytest.raises(queue.Empty) as caught:
        rank_case.run()
    assert caught.value is rank_case.queue_error
    assert all(not process.is_alive() for process in rank_case.processes)
    assert rank_case.path.read_text() == "owned rootinfo"
    assert "controlled reap proof failure" in capsys.readouterr().err


@pytest.mark.parametrize("fail_copy", (False, True))
def test_rank_entry_flushes_native_boundaries_without_changing_result(monkeypatch, capsys, fail_copy):
    class RecordingStderr(io.StringIO):
        def __init__(self):
            super().__init__()
            self.flushed = ""

        def flush(self):
            self.flushed = self.getvalue()
            super().flush()

    stderr = RecordingStderr()
    monkeypatch.setattr(sys, "stderr", stderr)

    class Worker:
        def init(self, device_id, bins):
            pass

        def comm_init(self, rank, nranks, path):
            return 12

        def comm_alloc_windows(self, comm, size):
            return 0x1000

        def comm_get_local_window_base(self, comm):
            return 0x2000

        def comm_get_window_size(self, comm):
            return 4096

        def copy_from(self, destination, source, size):
            assert stderr.flushed.endswith("phase=memcpy rootinfo=owned-rootinfo] before\n")
            if fail_copy:
                raise RuntimeError("controlled native copy failure")
            context = comm_test._CommContext()
            context.rankId = 0
            context.rankNum = 2
            context.winSize = 4096
            context.windowsIn[0] = 0x2000
            context.windowsIn[1] = 0x3000
            ctypes.memmove(destination, ctypes.addressof(context), size)

        def comm_barrier(self, comm):
            pass

        def comm_destroy(self, comm):
            pass

        def finalize(self):
            pass

    interface = ModuleType("simpler.task_interface")
    monkeypatch.setattr(interface, "ChipWorker", Worker, raising=False)
    monkeypatch.setitem(sys.modules, interface.__name__, interface)
    results = mp.get_context("fork").Queue()
    try:
        comm_test._rank_entry(0, 2, 5, {}, "owned-rootinfo", results)
        result = results.get(timeout=5)
    finally:
        results.close()
        results.join_thread()
    lines = stderr.getvalue().splitlines()
    phases = ["create", "init", "comm_init", "alloc", "query_base", "query_size", "memcpy"]
    if not fail_copy:
        phases += ["barrier", "destroy", "finalize"]
    assert len(lines) == 2 * len(phases)
    for i, phase in enumerate(phases):
        prefix = f"[comm-lifecycle pid={os.getpid()} rank=0 device=5 phase={phase} rootinfo=owned-rootinfo]"
        assert lines[2 * i] == prefix + " before"
        expected = " failed" if fail_copy and phase == "memcpy" else " after"
        assert lines[2 * i + 1] == prefix + expected
    assert result["ok"] is not fail_copy
    if fail_copy:
        assert result["stage"] == "memcpy"
        assert "controlled native copy failure" in result["error"]
    else:
        assert result["stage"] == "finalize"
        assert result["barrier_ok"] is True
        assert result["peer_windows"] == [0x2000, 0x3000]


def test_rank_failure_is_not_replaced_by_storage_cleanup_failure(rank_case, capsys):
    rank_case.fail_queue = False
    rank_case.report = True
    rank_case.hold = False
    rank_case.reported_error = True
    rank_case.close_error = True
    with pytest.raises(pytest.fail.Exception, match="controlled rank failure"):
        rank_case.run()
    assert all(not process.is_alive() for process in rank_case.processes)
    assert "controlled queue close failure" in capsys.readouterr().err
    assert not rank_case.path.exists()
