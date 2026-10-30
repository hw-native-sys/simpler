# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Kernel lease evidence for local communication generation reclamation."""

import fcntl
import os
import subprocess
import sys
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import pytest
from simpler.comm_generation import CommGeneration


def test_active_and_surviving_rank_leases_are_preserved(tmp_path):
    root = str(tmp_path / f"simpler-comm-{os.getuid()}")
    first = CommGeneration(root)
    generation = Path(first.rootinfo_path).parent
    marker = generation / "barrier_rootinfo.bin_alloc_1_release_1_0.ready"
    marker.write_text("1")
    rank_fd = os.open(generation / ".lease", os.O_RDONLY)
    try:
        fcntl.flock(rank_fd, fcntl.LOCK_SH)
        second = CommGeneration(root)
        assert marker.exists()
        first.close()
        assert marker.exists()
        second.close()
        assert marker.exists()
    finally:
        os.close(rank_fd)
    third = CommGeneration(root)
    assert not generation.exists()
    third.close()


def test_crashed_owner_is_reclaimed_on_next_generation(tmp_path):
    root = str(tmp_path / f"simpler-comm-{os.getuid()}")
    program = """
import os, sys
from pathlib import Path
from simpler.comm_generation import CommGeneration
owner = CommGeneration(sys.argv[1])
Path(owner.rootinfo_path).write_text('incomplete generation')
print(owner.rootinfo_path, flush=True)
os._exit(23)
"""
    process = subprocess.run([sys.executable, "-c", program, root], capture_output=True, text=True, check=False)
    assert process.returncode == 23
    stale = Path(process.stdout.strip())
    assert stale.exists()
    replacement = CommGeneration(root)
    assert not stale.parent.exists()
    replacement.close()


def test_reclamation_error_blocks_admission_and_preserves_unknown_entry(tmp_path):
    root = str(tmp_path / f"simpler-comm-{os.getuid()}")
    owner = CommGeneration(root)
    generation = Path(owner.rootinfo_path).parent
    (generation / "unexpected-directory").mkdir()
    with pytest.raises(RuntimeError, match="unexpected entry"):
        owner.close()
    with pytest.raises(RuntimeError, match="unexpected entry"):
        CommGeneration(root)
    assert (generation / "unexpected-directory").exists()


def test_private_registry_required(tmp_path):
    root = tmp_path / f"simpler-comm-{os.getuid()}"
    root.mkdir(mode=0o755)
    with pytest.raises(RuntimeError, match="private owned directory"):
        CommGeneration(str(root))


def test_generation_capacity_refuses_more_active_leases(tmp_path, monkeypatch):
    import simpler.comm_generation as generation_module  # noqa: PLC0415

    monkeypatch.setattr(generation_module, "_MAX_GENERATIONS", 2)
    root = str(tmp_path / f"simpler-comm-{os.getuid()}")
    owners = [CommGeneration(root), CommGeneration(root)]
    try:
        with pytest.raises(RuntimeError, match="capacity reached"):
            CommGeneration(root)
    finally:
        for owner in owners:
            owner.close()


@pytest.mark.parametrize("operation", ["alloc", "release"])
def test_worker_stops_admission_after_any_rank_failure(monkeypatch, operation):
    from types import SimpleNamespace  # noqa: PLC0415
    from unittest.mock import Mock  # noqa: PLC0415

    import simpler.worker as worker_module  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    worker = Worker(level=3, platform="a2a3", runtime="tensormap_and_ringbuffer", device_ids=[0])
    worker._worker = SimpleNamespace(
        control_release_domain=Mock(side_effect=RuntimeError("rank failure")),
        control_alloc_domain=Mock(side_effect=RuntimeError("rank failure")),
    )
    request = SharedMemory(create=True, size=8)
    try:
        with pytest.raises(RuntimeError, match=f"{operation}_domain") as failed:
            worker._dispatch_control_domain(
                workers=(0,),
                request_shms={0: request},
                reply_shms={0: request} if operation == "alloc" else None,
                op=operation,
                allocation_id=7,
            )
        assert worker._comm_domain_error is failed.value
        monkeypatch.setattr(worker_module, "_validate_domain_allocation", lambda *args: None)
        bootstrap = Mock()
        monkeypatch.setattr(worker, "_ensure_comm_base", bootstrap)
        with pytest.raises(RuntimeError, match="admission stopped") as admission:
            worker._allocate_domain(name="next", workers=(0,), window_size=4096, buffers=[])
        assert admission.value.__cause__ is failed.value
        bootstrap.assert_not_called()
    finally:
        request.close()
        request.unlink()
        worker._worker = None


def test_invalid_generation_layout_is_rejected_before_creation(tmp_path):
    root = tmp_path / "registry"
    with pytest.raises(ValueError, match="root must be named"):
        CommGeneration(str(root))
    assert not root.exists()


def test_partial_init_generation_failure_remains_retryable(monkeypatch):
    from unittest.mock import Mock  # noqa: PLC0415

    from simpler.worker import Worker  # noqa: PLC0415

    worker = Worker(level=3, platform="a2a3", runtime="tensormap_and_ringbuffer", device_ids=[0])
    close_generation = Mock(side_effect=[OSError("unlink denied"), None])
    worker._comm_generation = Mock(close=close_generation)
    teardown = worker._teardown_worker_tree
    monkeypatch.setattr(worker, "_teardown_worker_tree", Mock(side_effect=RuntimeError("startup failure")))
    worker._cleanup_partial_init()
    assert worker._has_live_resources()
    assert not worker._cleanup_journal.empty
    monkeypatch.setattr(worker, "_teardown_worker_tree", teardown)
    worker.close()
    assert close_generation.call_count == 2
    assert worker._comm_generation is None
    assert worker._cleanup_journal.empty


def test_bootstrap_failure_stops_domain_admission(monkeypatch):
    from types import SimpleNamespace  # noqa: PLC0415
    from unittest.mock import Mock  # noqa: PLC0415

    import simpler.worker as worker_module  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    worker = Worker(level=3, platform="a2a3", runtime="tensormap_and_ringbuffer", device_ids=[0])
    bootstrap = Mock(side_effect=RuntimeError("bootstrap failed"))
    worker._worker = SimpleNamespace(control_comm_init=bootstrap)
    monkeypatch.setattr(worker, "_comm_plan_rootinfo_path", lambda: "/tmp/test-bootstrap.bin")
    monkeypatch.setattr(worker_module, "_validate_domain_allocation", lambda *args: None)
    try:
        with pytest.raises(RuntimeError, match="_ensure_comm_base") as failed:
            worker._allocate_domain(name="first", workers=(0,), window_size=4096, buffers=[])
        assert worker._comm_domain_error is failed.value
        with pytest.raises(RuntimeError, match="admission stopped"):
            worker._allocate_domain(name="next", workers=(0,), window_size=4096, buffers=[])
        bootstrap.assert_called_once()
    finally:
        worker._worker = None


def test_unrelated_fork_child_does_not_retain_parent_generation(tmp_path):
    owner = CommGeneration(str(tmp_path / f"simpler-comm-{os.getuid()}"))
    directory = Path(owner.rootinfo_path).parent
    read_fd, write_fd = os.pipe()
    ready_read, ready_write = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(write_fd)
        os.close(ready_read)
        os.write(ready_write, b"1")
        os.close(ready_write)
        os.read(read_fd, 1)
        os._exit(0)
    os.close(read_fd)
    os.close(ready_write)
    assert os.read(ready_read, 1) == b"1"
    os.close(ready_read)
    try:
        owner.close()
        assert not directory.exists()
    finally:
        os.close(write_fd)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
