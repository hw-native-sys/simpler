# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Clean-interpreter chip bootstrap over the real simulator task loop."""

import os
import signal
from time import monotonic

import pytest
from simpler.chip_process import _pack_config, _unpack_config, launch_chip
from simpler.task_interface import CallConfig
from simpler.worker import Worker

from ._harness import chip_callable


def test_bootstrap_config_keeps_per_ring_sizes_and_diagnostics():
    config = CallConfig()
    config.aicpu_thread_num = 2
    config.enable_chip_swimlane = 3
    config.enable_scope_stats = True
    config.output_prefix = "capture"
    config.runtime_env.ring_task_window = [4, 8, 16, 32]
    config.runtime_env.ring_heap = [1024, 2048, 4096, 8192]
    config.runtime_env.ring_dep_pool = [4, 8, 12, 16]
    assert _pack_config(_unpack_config(_pack_config(config))) == _pack_config(config)
    assert _unpack_config(_pack_config(None)) is None


@pytest.mark.parametrize("runtime", ["host_build_graph", "tensormap_and_ringbuffer"])
def test_clean_interpreter_reaches_ready_and_is_reaped(monkeypatch, runtime):
    monkeypatch.setattr(Worker, "_launch_chip_process", launch_chip)
    worker = Worker(
        level=3,
        platform="a5sim",
        runtime=runtime,
        device_ids=[0],
        num_sub_workers=0,
        heap_ring_size=4096,
        startup_timeout_s=30,
    )
    try:
        worker.init()
        pid = worker._chip_pids[0]
        os.kill(pid, 0)
    finally:
        worker.close()
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)


def test_bootstrap_bad_digest_rolls_back_before_ready(monkeypatch):
    def corrupt(worker, index, device_id, snapshots, log_level, requested_depth, deadline):
        snapshots = [
            (bytes(32), target, count, kind, namespace) for _digest, target, count, kind, namespace in snapshots
        ]
        return launch_chip(worker, index, device_id, snapshots, log_level, requested_depth, deadline)

    monkeypatch.setattr(Worker, "_launch_chip_process", corrupt)
    worker = Worker(
        level=3,
        platform="a5sim",
        runtime="tensormap_and_ringbuffer",
        device_ids=[0],
        num_sub_workers=0,
        heap_ring_size=4096,
        startup_timeout_s=30,
    )
    worker.register(chip_callable())
    try:
        with pytest.raises(RuntimeError, match="HASHID_DESCRIPTOR_MISMATCH"):
            worker.init()
    finally:
        worker.close()
    assert not worker._chip_pids


def test_expired_bootstrap_rejects_without_initializing_chip(monkeypatch):
    def expired(worker, index, device_id, snapshots, log_level, requested_depth, deadline):
        return launch_chip(worker, index, device_id, snapshots, log_level, requested_depth, monotonic() - 1)

    monkeypatch.setattr(Worker, "_launch_chip_process", expired)
    worker = Worker(
        level=3,
        platform="a5sim",
        runtime="tensormap_and_ringbuffer",
        device_ids=[0],
        num_sub_workers=0,
        heap_ring_size=0,
        startup_timeout_s=30,
    )
    try:
        with pytest.raises(RuntimeError, match="deadline|exited during init"):
            worker.init()
    finally:
        worker.close()
    assert not worker._chip_pids


def test_chip_killed_after_ready_is_reaped_by_existing_close(monkeypatch):
    monkeypatch.setattr(Worker, "_launch_chip_process", launch_chip)
    worker = Worker(
        level=3,
        platform="a5sim",
        runtime="tensormap_and_ringbuffer",
        device_ids=[0],
        num_sub_workers=0,
        heap_ring_size=4096,
        startup_timeout_s=30,
    )
    worker.init()
    pid = worker._chip_pids[0]
    os.kill(pid, signal.SIGKILL)
    with pytest.raises(RuntimeError, match="killed by signal 9"):
        worker.close()
    assert not worker._chip_pids
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)


def test_mailbox_loop_keeps_bootstrap_parent_identity():
    from simpler import worker as module  # noqa: PLC0415

    storage = bytearray(module.mailbox_size(1))
    view = memoryview(storage)
    try:
        module._run_mailbox_loop(
            view,
            module._buffer_field_addr(view, module._OFF_STATE),
            handle_task=lambda *_args: pytest.fail("unexpected task"),
            handle_control=lambda *_args: pytest.fail("unexpected control"),
            expected_parent_pid=os.getppid() + 1,
        )
    finally:
        view.release()


@pytest.mark.parametrize("platform", ["a2a3", "a5"])
def test_onboard_default_uses_clean_chip_launcher(monkeypatch, platform):
    from simpler import chip_process  # noqa: PLC0415

    calls = []
    monkeypatch.setattr(chip_process, "launch_chip", lambda *args: calls.append(args) or 123)
    monkeypatch.setattr(os, "fork", lambda: pytest.fail("onboard chip inherited the parent image"))
    worker = Worker(level=3, platform=platform, runtime="tensormap_and_ringbuffer", device_ids=[0], num_sub_workers=0)
    deadline = monotonic() + 30
    assert worker._launch_chip_process(0, 0, [], 25, 1, deadline) == 123
    assert calls == [(worker, 0, 0, [], 25, 1, deadline)]
