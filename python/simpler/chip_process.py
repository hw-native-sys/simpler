# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Start a chip owner in a clean interpreter; the Worker retains waitpid ownership."""

from __future__ import annotations

import contextlib
import ctypes
import os
import pickle
import sys
import tempfile
from dataclasses import asdict
from multiprocessing import Pipe, resource_tracker
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from time import monotonic

_CONFIG_FIELDS = (
    "aicpu_thread_num",
    "enable_chip_swimlane",
    "enable_dump_args",
    "enable_pmu",
    "enable_dep_gen",
    "enable_scope_stats",
    "output_prefix",
)
_RUNTIME_FIELDS = ("ring_task_window", "ring_heap", "ring_dep_pool")


def _pack_config(config):
    if config is None:
        return None
    config.validate()
    return (
        {name: getattr(config, name) for name in _CONFIG_FIELDS},
        {name: list(getattr(config.runtime_env, name)) for name in _RUNTIME_FIELDS},
    )


def _unpack_config(packed):
    if packed is None:
        return None
    from .task_interface import CallConfig  # noqa: PLC0415

    config = CallConfig()
    fields, runtime_fields = packed
    for name, value in fields.items():
        setattr(config, name, value)
    for name, value in runtime_fields.items():
        setattr(config.runtime_env, name, value)
    config.validate()
    return config


def _spawn_file_actions(pass_fds: set[int], cleanup: contextlib.ExitStack) -> list[tuple[int, ...]]:
    # Some platforms leave CLOEXEC set for dup2(fd, fd). Use distinct
    # source fds to pass the required descriptors at their original numbers.
    actions: list[tuple[int, ...]] = []
    for fd in sorted(pass_fds):
        source = os.dup(fd)
        cleanup.callback(os.close, source)
        actions.append((os.POSIX_SPAWN_DUP2, source, fd))
    fd_dir = Path("/proc/self/fd")
    if not fd_dir.is_dir():
        fd_dir = Path("/dev/fd")
    for entry in fd_dir.iterdir():
        if not entry.name.isdecimal():
            continue
        fd = int(entry.name)
        if fd > 2 and fd not in pass_fds:
            # The directory snapshot can include descriptors already closed.
            with contextlib.suppress(OSError):
                if os.get_inheritable(fd):
                    actions.append((os.POSIX_SPAWN_CLOSE, fd))
    return actions


def launch_chip(worker, index, device_id, snapshots, log_level, requested_depth, deadline) -> int:
    """Pass owned bootstrap data and heap fds; never create a second child reaper."""
    from .comm_endpoints import DEVICE_AICORE, DEVICE_AICPU  # noqa: PLC0415
    from .task_interface import _bind_host_log_session_directory  # noqa: PLC0415

    entries = [
        (digest, ctypes.string_at(target.buffer_ptr(), target.buffer_size()), count)
        for digest, target, count, kind, namespace in snapshots
        if kind == "CHIP_CALLABLE" and namespace == "LOCAL_CHIP"
    ]
    data = dict(
        mailbox=worker._chip_shms[index].name,
        bins=asdict(worker._l3_bins),
        device_id=device_id,
        registry=entries,
        owner=worker._owner_instance_id,
        aicpu=worker._device_endpoint_identities[(index, DEVICE_AICPU)][0],
        aicore=worker._device_endpoint_identities[(index, DEVICE_AICORE)][0],
        platform=str(worker._config["platform"]),
        runtime=str(worker._config["runtime"]),
        prewarm_config=_pack_config(worker._prewarm_config),
        enable_sdma=bool(worker._config.get("enable_sdma", False)),
        log_level=log_level,
        host_log_directory=_bind_host_log_session_directory(),
        chip_rank=index,
        launch_depth=requested_depth if worker._automatic_admission_requested else worker._launch_depth,
        collect_across_runs=bool(worker._config.get("collect_across_runs", False)),
        pipeline_depth_request=requested_depth,
        automatic_admission=worker._automatic_admission_requested,
        shared_heap_files=worker._shared_heap_files,
        parent_pid=os.getpid(),
        group_leader=worker._is_startup_root,
        deadline=deadline,
    )
    tracker_fd = resource_tracker.getfd()
    assert tracker_fd is not None
    heap_fds = {fd for _owner, _pid, fds in worker._shared_heap_files for fd in fds}
    parent, child = Pipe()
    with parent, child:
        with tempfile.TemporaryFile() as bootstrap, contextlib.ExitStack() as cleanup:
            pickle.dump(data, bootstrap, protocol=pickle.HIGHEST_PROTOCOL)
            bootstrap.seek(0)
            fds = bootstrap.fileno(), tracker_fd, child.fileno()
            command = [sys.executable, "-m", "simpler.chip_process", *(str(fd) for fd in fds)]
            actions = _spawn_file_actions(heap_fds | set(fds), cleanup)
            pid = os.posix_spawn(sys.executable, command, os.environ, file_actions=actions)
            worker._record_chip_pid(pid)
        # An unacknowledged child exits without opening its mailbox or touching CANN.
        parent.send_bytes(b"1")
    return pid


def _main(bootstrap_fd: int, tracker_fd: int, startup_fd: int) -> None:
    with Connection(startup_fd) as startup:
        try:
            command = startup.recv_bytes(1)
            if command != b"1":
                return
        except (EOFError, OSError):
            return
    # Match multiprocessing.spawn: all descendants share the creator's tracker.
    resource_tracker._resource_tracker._fd = tracker_fd
    with os.fdopen(bootstrap_fd, "rb") as bootstrap:
        data = pickle.load(bootstrap)
    if data["group_leader"]:
        os.setpgid(0, 0)
    from simpler_setup.runtime_builder import RuntimeBinaries  # noqa: PLC0415

    from . import worker as module  # noqa: PLC0415
    from .task_interface import ChipCallable, _native_set_host_log_directory  # noqa: PLC0415

    mailbox = SharedMemory(name=data["mailbox"])
    buf = mailbox.buf
    assert buf is not None
    try:
        if not _native_set_host_log_directory(data["host_log_directory"]):
            raise RuntimeError("chip bootstrap: cannot bind parent host log directory")
        if os.getppid() != data["parent_pid"] or monotonic() >= data["deadline"]:
            raise RuntimeError("chip bootstrap: parent exited or startup deadline exceeded")
        entries = []
        for digest, payload, count in data["registry"]:
            target = ChipCallable.from_bytes(payload)
            module._validate_chip_payload_digest(
                target,
                digest,
                platform=data["platform"],
                runtime=data["runtime"],
                context="chip bootstrap",
            )
            entries.append((digest, target, count, "CHIP_CALLABLE", "LOCAL_CHIP"))
        tables = module._make_local_identity_tables(
            entries,
            callable_kind="CHIP_CALLABLE",
            target_namespace="LOCAL_CHIP",
        )
        module._chip_process_loop(
            buf,
            RuntimeBinaries(**data["bins"]),
            data["device_id"],
            *tables,
            data["owner"],
            data["aicpu"],
            data["aicore"],
            log_level=data["log_level"],
            platform=data["platform"],
            runtime=data["runtime"],
            prewarm_config=_unpack_config(data["prewarm_config"]),
            enable_sdma=data["enable_sdma"],
            chip_rank=data["chip_rank"],
            launch_depth=data["launch_depth"],
            collect_across_runs=data["collect_across_runs"],
            pipeline_depth_request=data["pipeline_depth_request"],
            automatic_admission=data["automatic_admission"],
            fork_inherited=False,
            expected_parent_pid=data["parent_pid"],
            shared_heap_files=data["shared_heap_files"],
        )
    except BaseException as exc:
        import traceback  # noqa: PLC0415

        traceback.print_exc()
        if module._mailbox_load_i32(module._buffer_field_addr(buf, module._OFF_STATE)) == module._IDLE:
            module._write_error(buf, 1, module._format_exc("chip bootstrap", exc))
            module._mailbox_store_i32(module._buffer_field_addr(buf, module._OFF_STATE), module._INIT_FAILED)
        module._exit_after_host_log_flush(1)
    finally:
        mailbox.close()
    module._exit_after_host_log_flush(0)


if __name__ == "__main__":
    _main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
