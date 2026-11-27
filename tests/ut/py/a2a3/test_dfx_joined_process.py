# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real process/shared-memory ownership, without a device or calibrated waits."""

import ctypes
import json
import os
import signal
import socket
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.st.a2a3.tensormap_and_ringbuffer.dfx_joined_launch import process


def _failed_owner(record):
    memory = SharedMemory(create=True, size=128)
    assert memory.buf is not None
    memory.buf[0] = 37
    Path(record).write_text(json.dumps({"pid": os.getpid(), "name": memory.name}))
    # Model a terminal result escaping while an explicit-close resource is live.
    raise RuntimeError("terminal task failure")


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="Linux process and POSIX-shm ownership")
def test_terminal_error_reclaims_the_resource_owner_and_preserves_original_failure(tmp_path):
    record = tmp_path / "owner.json"
    try:
        with pytest.raises(RuntimeError, match="terminal task failure"):
            process.run_isolated(_failed_owner, (str(record),), tmp_path)
        owner = json.loads(record.read_text())
        assert not (Path("/dev/shm") / owner["name"]).exists(), "failed scenario left its backing live"
        assert owner["pid"] != os.getpid(), "failure still owns resources in the pytest process"
        with pytest.raises(ProcessLookupError):
            os.kill(owner["pid"], 0)
    finally:
        if record.exists():
            name = json.loads(record.read_text())["name"]
            if (Path("/dev/shm") / name).exists():
                memory = SharedMemory(name=name)
                memory.unlink()
                memory.close()


def _hung_executor_after_release(socket_path, record):
    memory = SharedMemory(create=True, size=128)
    descendant = os.fork()
    if descendant == 0:
        while True:
            signal.pause()
    entered = threading.Event()

    def stuck_submit():
        entered.set()
        threading.Event().wait()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(stuck_submit)
        assert entered.wait(5), "submission never reached its explicit blocking point"
        try:
            future.result(timeout=0)
        finally:
            # This is the released-gate point. The submitter remains stuck,
            # so exiting the actual executor will wait forever without isolation.
            Path(record).write_text(json.dumps({"pid": os.getpid(), "descendant": descendant, "name": memory.name}))
            with socket.socket(socket.AF_UNIX) as channel:
                channel.connect(socket_path)
                channel.sendall(b"released")


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="Linux process and POSIX-shm ownership")
def test_timeout_after_release_stops_a_stuck_executor_and_its_descendant(tmp_path, monkeypatch):
    record = tmp_path / "owner.json"
    socket_path = str(tmp_path / "stage.sock")
    polls = 0

    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(socket_path)
        listener.listen(1)
        listener.settimeout(5)

        def deadline_after_release(*_args):
            nonlocal polls
            polls += 1
            assert polls == 1, "the explicit scenario deadline was ignored"
            with listener.accept()[0] as channel:
                assert channel.recv(16) == b"released"
            # Expire only after the real callback confirms submission is stuck
            # past release. No machine-speed window establishes this scenario.
            return [], [], []

        monkeypatch.setattr(process.select, "select", deadline_after_release)
        with pytest.raises(TimeoutError, match="gate scenario.*timeout"):
            process.run_isolated(_hung_executor_after_release, (socket_path, str(record)), tmp_path)

    owner = json.loads(record.read_text())
    assert not Path("/proc", str(owner["pid"])).exists(), "scenario owner was not reaped"
    descendant = Path("/proc", str(owner["descendant"]), "stat")
    try:
        state = descendant.read_bytes().rsplit(b") ", 1)[1].split()[0]
    except FileNotFoundError:
        state = b"X"
    assert state in (b"Z", b"X"), f"descendant {owner['descendant']} remains live: {state!r}"
    assert not (Path("/dev/shm") / owner["name"]).exists()


def _successful_owner(record):
    memory = SharedMemory(create=True, size=128)
    Path(record).write_text(json.dumps({"pid": os.getpid(), "name": memory.name}))


def _non_utf8_owner(record):
    # PR_SET_NAME accepts bytes; decoding comm is unnecessary to establish PGID.
    assert ctypes.CDLL(None).prctl(15, ctypes.c_char_p(b"gate-\xff) name"), 0, 0, 0) == 0
    _successful_owner(record)


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="Linux process and POSIX-shm ownership")
def test_cleanup_accepts_a_non_utf8_process_name(tmp_path):
    record = tmp_path / "owner.json"
    process.run_isolated(_non_utf8_owner, (str(record),), tmp_path)
    owner = json.loads(record.read_text())
    assert not Path("/proc", str(owner["pid"])).exists()
    assert not (Path("/dev/shm") / owner["name"]).exists()


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="Linux process and POSIX-shm ownership")
def test_relative_scenario_directory_outside_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    record = tmp_path / "artifacts" / "owner.json"
    process.run_isolated(_successful_owner, (str(record),), "artifacts")
    owner = json.loads(record.read_text())
    assert not Path("/proc", str(owner["pid"])).exists()
    assert not (Path("/dev/shm") / owner["name"]).exists()


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="Linux process and POSIX-shm ownership")
@pytest.mark.parametrize("scenario_fails", [False, True])
def test_owner_wait_failure_does_not_replace_scenario_failure(tmp_path, monkeypatch, scenario_fails):
    original_wait = process.subprocess.Popen.wait

    def failed_wait(child, *args, **kwargs):
        original_wait(child, *args, **kwargs)
        # The actual owner has exited and been reaped before this OS-boundary fault.
        raise OSError("injected owner wait failure")

    monkeypatch.setattr(process.subprocess.Popen, "wait", failed_wait)
    record = tmp_path / "owner.json"
    expected_type = RuntimeError if scenario_fails else OSError
    expected_text = "terminal task failure" if scenario_fails else "injected owner wait failure"
    with pytest.raises(expected_type, match=expected_text) as failure:
        process.run_isolated(_failed_owner if scenario_fails else _successful_owner, (str(record),), tmp_path)
    if scenario_fails:
        assert any("injected owner wait failure" in note for note in getattr(failure.value, "__notes__", []))
    owner = json.loads(record.read_text())
    assert not Path("/proc", str(owner["pid"])).exists()
    assert not (Path("/dev/shm") / owner["name"]).exists()


def _failure_with_cleanup_note():
    from simpler.worker import attach_exception_note  # noqa: PLC0415

    error = RuntimeError("terminal task failure")
    attach_exception_note(error, "retained mapping: cleanup also failed")
    raise error


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="Linux process and POSIX-shm ownership")
def test_cleanup_note_is_visible_after_crossing_process_boundary(tmp_path):
    with pytest.raises(RuntimeError, match="terminal task failure") as failure:
        process.run_isolated(_failure_with_cleanup_note, (), tmp_path)
    assert "retained mapping: cleanup also failed" in str(failure.value)


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="Linux process and POSIX-shm ownership")
def test_success_reclaims_only_its_own_namespace(tmp_path, monkeypatch):
    foreign_token, own_token = uuid.uuid4().hex, uuid.uuid4().hex
    foreign = SharedMemory(name=f"sp-{foreign_token[:8]}-foreign", create=True, size=128)
    assert foreign.buf is not None
    foreign.buf[0] = 91
    candidates = iter((foreign_token, own_token))
    monkeypatch.setattr(process.uuid, "uuid4", lambda: SimpleNamespace(hex=next(candidates)))
    try:
        record = tmp_path / "owner.json"
        process.run_isolated(_successful_owner, (str(record),), tmp_path)
        owner = json.loads(record.read_text())
        assert not (Path("/dev/shm") / owner["name"]).exists()
        assert foreign.buf[0] == 91
        assert (Path("/dev/shm") / foreign.name).exists(), "cleanup removed another owner's namespace"
    finally:
        foreign.unlink()
        foreign.close()
