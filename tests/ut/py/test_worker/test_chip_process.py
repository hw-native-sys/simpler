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
import sys
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path
from time import monotonic
from unittest.mock import Mock

import pytest
from simpler import chip_process
from simpler.chip_process import _pack_config, _unpack_config, launch_chip
from simpler.task_interface import CallConfig
from simpler.worker import Worker

from ._harness import TEST_WALL_BUDGET_S, chip_callable, hard_timeout


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


def test_spawn_preserves_only_the_requested_descriptors(tmp_path):
    with (tmp_path / "expected").open("w+b") as expected, (tmp_path / "extra").open("w+b") as extra:
        expected.write(b"heap")
        expected.seek(0)
        os.set_inheritable(extra.fileno(), True)
        with ExitStack() as cleanup:
            actions = chip_process._spawn_file_actions({expected.fileno()}, cleanup)
            script = (
                f"import os; assert os.read({expected.fileno()}, 4) == b'heap'; "
                f"assert not os.path.exists('/dev/fd/{extra.fileno()}')"
            )
            pid = os.posix_spawn(sys.executable, [sys.executable, "-c", script], os.environ, file_actions=actions)
        assert os.waitpid(pid, 0) == (pid, 0)


def test_exec_child_uses_parent_import_order(monkeypatch, tmp_path):
    real_spawn = os.posix_spawn

    def launch_without_site(*args):
        def spawn(path, command, env, **kwargs):
            # Disable the editable-install finder in the child.
            return real_spawn(path, [command[0], "-S", *command[1:]], env, **kwargs)

        with monkeypatch.context() as scoped:
            scoped.setattr(chip_process.os, "posix_spawn", spawn)
            return launch_chip(*args)

    shadow = tmp_path / "simpler"
    shadow.mkdir()
    (shadow / "__init__.py").write_text('raise RuntimeError("wrong simpler package")\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.syspath_prepend(str(Path(chip_process.__file__).resolve().parents[1]))
    monkeypatch.setattr(Worker, "_uses_exec_chip", lambda _worker: True)
    monkeypatch.setattr(Worker, "_launch_chip_process", launch_without_site)
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
        worker.init()
    finally:
        worker.close()


@pytest.mark.parametrize("child_bootstrap", [True, False])
def test_exec_startup_failure_reaps_created_chip(monkeypatch, child_bootstrap):
    real_spawn = os.posix_spawn
    real_tempfile = chip_process.tempfile.TemporaryFile
    spawned = []

    def spawn(path, command, env, **kwargs):
        if not child_bootstrap:
            command = [path, "-c", "import signal; signal.pause()"]
        pid = real_spawn(path, command, env, **kwargs)
        spawned.append(pid)
        return pid

    @contextmanager
    def failing_tempfile():
        with real_tempfile() as stream:
            yield stream
        raise OSError("injected startup cleanup failure")

    def launch_with_failure(*args):
        with monkeypatch.context() as scoped:
            scoped.setattr(chip_process.os, "posix_spawn", spawn)
            scoped.setattr(chip_process.tempfile, "TemporaryFile", failing_tempfile)
            return launch_chip(*args)

    monkeypatch.setattr(Worker, "_launch_chip_process", launch_with_failure)
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
        with pytest.raises(OSError, match="injected startup cleanup failure"):
            worker.init()
        assert len(spawned) == 1
        with pytest.raises(ChildProcessError):
            os.waitpid(spawned[0], os.WNOHANG)
    finally:
        worker.close()
        for pid in spawned:
            with suppress(ChildProcessError, ProcessLookupError):
                if os.waitpid(pid, os.WNOHANG)[0] == 0:
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)


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
    monkeypatch.setattr(Worker, "_uses_exec_chip", lambda _worker: True)
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


@hard_timeout(TEST_WALL_BUDGET_S)
def test_exec_chip_loop_keeps_bootstrap_parent_identity(monkeypatch):
    from simpler import worker as module  # noqa: PLC0415

    actual_parent = os.getppid()
    get_parent = Mock(side_effect=[actual_parent])
    monkeypatch.setattr(module.os, "getppid", get_parent)
    monkeypatch.setattr(module, "_PARENT_LIVENESS_POLL_INTERVAL", 1)
    chip = Mock(spec_set=["_close_chip_run_lane"])
    regions = Mock()
    regions.sweep.return_value = ()
    with memoryview(bytearray(module.mailbox_size(1))) as view:
        module._run_chip_main_loop(
            chip,
            view,
            module._buffer_field_addr(view, 0),
            module._buffer_field_addr(view, module._OFF_STATE),
            0,
            {},
            {},
            {},
            module.mint_owner_instance_id(),
            chip_platform="a5sim",
            provider_region_store=regions,
            fork_inherited=False,
            expected_parent_pid=actual_parent + 1,
        )
    get_parent.assert_called_once_with()
    chip._close_chip_run_lane.assert_called_once_with()
    regions.sweep.assert_called_once_with()


@pytest.mark.parametrize(
    ("platform", "runtime", "exec_chip"),
    [
        ("a5", "tensormap_and_ringbuffer", True),
        ("a5", "host_build_graph", False),
        ("a2a3", "tensormap_and_ringbuffer", False),
        ("a5sim", "tensormap_and_ringbuffer", False),
    ],
)
def test_default_chip_launcher_scope(monkeypatch, platform, runtime, exec_chip):
    from simpler import chip_process  # noqa: PLC0415

    monkeypatch.setattr(chip_process, "launch_chip", lambda *_: 123)
    monkeypatch.setattr(os, "fork", lambda: 456)
    worker = Worker(level=3, platform=platform, runtime=runtime, device_ids=[0], num_sub_workers=0)
    import simpler.worker as worker_module  # noqa: PLC0415

    worker._is_startup_root = True
    monkeypatch.setattr(worker_module, "_a5_l2_initialized", False)
    assert not worker._uses_exec_chip()
    monkeypatch.setattr(worker_module, "_a5_l2_initialized", True)
    worker._exec_chip = worker._uses_exec_chip()
    assert worker._exec_chip is exec_chip
    assert worker._launch_chip_process(0, 0, [], 25, 1, monotonic() + 30) == (123 if exec_chip else 456)
    worker._is_startup_root = False
    assert not worker._uses_exec_chip()


def test_invalid_direct_a5_device_keeps_l3_fork(monkeypatch):
    import simpler.worker as worker_module  # noqa: PLC0415

    monkeypatch.setattr(worker_module, "_a5_l2_initialized", False)
    monkeypatch.setattr("simpler_setup.runtime_builder.RuntimeBuilder", lambda _platform: Mock())
    runtime = "tensormap_and_ringbuffer"
    direct = Worker(level=2, platform="a5", runtime=runtime, device_id=-1)
    try:
        with pytest.raises(RuntimeError, match="non-negative device_id"):
            direct.init()
    finally:
        direct.close()

    hierarchical = Worker(level=3, platform="a5", runtime=runtime, device_ids=[0])
    assert not hierarchical._uses_exec_chip()
