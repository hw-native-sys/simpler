# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Test-owned process boundary for gate resources and uncancellable native calls."""

import contextlib
import ctypes
import importlib
import itertools
import json
import os
import select
import signal
import subprocess
import sys
import time
import traceback
import uuid
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import cloudpickle

_MODULE = "tests.st.a2a3.tensormap_and_ringbuffer.dfx_joined_launch.process"
_scope_token = None
_owner_roots = []


def retain_until_owner_exit(resource):
    """Prevent Python/native finalizers from racing unconfirmed device uses."""
    if _scope_token is not None:
        _owner_roots.append(resource)


def attach_cleanup_failure(primary, cleanup):
    """Keep the original exception and record cleanup diagnostics without replacing its cause."""
    from simpler.worker import attach_exception_note  # noqa: PLC0415 -- only an error path needs the Worker module

    details = "".join(traceback.format_exception(type(cleanup), cleanup, cleanup.__traceback__, chain=False))
    attach_exception_note(primary, f"Resource cleanup also failed:\n{details}")


def _reserve_scope():
    while True:
        token = uuid.uuid4().hex
        prefix = f"sp-{token[:8]}-"
        if any(path.name.startswith(prefix) for path in Path("/dev/shm").iterdir()):
            continue
        marker = Path("/dev/shm") / (prefix + "gate-owner")
        try:
            descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        try:
            os.write(descriptor, b"\0")
        finally:
            os.close(descriptor)
        # Existing Worker namespace allocation also scans sp-<prefix>- names.
        # Reserve the prefix before this owner's worker can create any mailbox.
        # Use a plain reservation file: Worker.close may already unlink it, so
        # the parent's resource tracker must not also claim ownership of it.
        return token, marker


def _live_group(pgid):
    members = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_bytes().rsplit(b") ", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if int(fields[2]) == pgid and fields[0] not in (b"Z", b"X"):
            members.append(int(path.parent.name))
    return members


def _reclaim_owner(child, token):
    # The leader is deliberately unreaped until cleanup ends, so its PID/PGID
    # cannot be reused by another user's process while these names are retired.
    deadline = time.monotonic() + 5.0
    while True:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        live = _live_group(child.pid)
        if not live:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"gate scenario owner exit is unconfirmed; retain resources for {live}")
        time.sleep(min(0.01, remaining))  # Backoff only; actual process exit is the proof.
    prefixes = (f"dg-{token[:16]}-", f"sp-{token[:8]}-", f"simpler-cb-{child.pid}-")
    for path in Path("/dev/shm").iterdir():
        if path.name.startswith(prefixes):
            # Attaching here registers this exact remaining name before unlink,
            # including backings whose original resource tracker died with its owner.
            with contextlib.suppress(FileNotFoundError):
                memory = SharedMemory(name=path.name)
                try:
                    memory.unlink()
                finally:
                    memory.close()
    child.wait(timeout=max(0.01, deadline - time.monotonic()))


def run_isolated(function, args, directory, *, timeout=60.0, environment=None):
    """Run one scenario with a private process group and resource namespace.

    Device workers must be created with device_worker(), which keeps their
    children in this owner's group. No caller-owned Worker or mapping crosses
    the process boundary. An error never frees a backing while a device child
    might still access it: the full group exits before named memory is unlinked.
    """
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    token, marker = _reserve_scope()
    payload = directory / f"gate-owner-{token}.pickle"
    result = directory / f"gate-owner-{token}.json"
    read_fd = write_fd = None
    child = None
    try:
        payload.write_bytes(cloudpickle.dumps((function, args)))
        read_fd, write_fd = os.pipe()
        child = subprocess.Popen(
            [sys.executable, "-m", _MODULE, str(payload), str(result), str(write_fd), token],
            cwd=Path(__file__).resolve().parents[5],
            pass_fds=(write_fd,),
            start_new_session=True,
            env=environment,
        )
        os.close(write_fd)
        write_fd = None
        if not select.select([read_fd], [], [], timeout)[0]:
            raise TimeoutError(f"gate scenario exceeded its {timeout:g}s timeout")
        if os.read(read_fd, 1) != b"D":
            raise RuntimeError("gate scenario owner exited before reporting its result")
        outcome = json.loads(result.read_text())
        if not outcome["ok"]:
            details = outcome["traceback"]
            for note in outcome.get("notes", []):
                if note not in details:
                    details += "\n" + note
            raise RuntimeError(f"{outcome['type']}: {outcome['message']}\n{details}")
    finally:
        primary = sys.exc_info()[1]
        try:
            if child is not None:
                _reclaim_owner(child, token)
        except BaseException as cleanup:  # noqa: BLE001 -- preserve a pending scenario failure
            if primary is None:
                raise
            attach_cleanup_failure(primary, cleanup)
        finally:
            if read_fd is not None:
                os.close(read_fd)
            if write_fd is not None:
                os.close(write_fd)
            payload.unlink(missing_ok=True)
            if child is None:
                marker.unlink(missing_ok=True)


@contextlib.contextmanager
def device_worker(payload, device_id, root):
    """A local L3 worker whose owner is the enclosing isolated scenario."""
    from _task_interface import _set_host_log_directory  # noqa: PLC0415
    from simpler import Worker  # noqa: PLC0415
    from simpler.task_interface import ChipCallable  # noqa: PLC0415

    assert _scope_token is not None, "device_worker must run inside run_isolated"
    host = Path(root) / "host"
    host.mkdir(parents=True, exist_ok=True)
    _set_host_log_directory(str(host))
    worker = Worker(
        level=3,
        device_ids=[device_id],
        num_sub_workers=0,
        platform="a2a3",
        runtime="tensormap_and_ringbuffer",
        launch_depth=2,
        collect_across_runs=True,
    )
    retain_until_owner_exit(worker)
    # Pass an enclosing owner's deadline: the existing nested-startup protocol
    # then keeps every chip child in this scenario's process group.
    worker._shm_token = _scope_token  # noqa: SLF001 -- private test-owned startup namespace
    worker._shm_tree_tokens = {_scope_token}  # noqa: SLF001
    handle = worker.register(ChipCallable.from_bytes(payload))
    worker.init(_startup_deadline=time.monotonic() + 30.0)  # noqa: SLF001
    assert all(os.getpgid(pid) == os.getpgrp() for pid in worker._chip_pids)  # noqa: SLF001
    try:
        yield worker, handle
    except BaseException:
        # Preserve the primary failure and every potentially live mapping.
        # The supervisor stops all device users before reclaiming this owner.
        raise
    else:
        worker.close()


def run_native_scene(prepared_worker, handle, directory):
    """Transfer compiled code, never a live Worker/context, into the scenario owner."""
    directory = Path(directory).resolve()
    target = prepared_worker._resolve_handle(handle, expected_namespace="LOCAL_CHIP").target  # noqa: SLF001
    payload = ctypes.string_at(int(target.buffer_ptr()), int(target.buffer_size()))
    device_id = prepared_worker._config["device_ids"][0]  # noqa: SLF001 -- actual allocated physical device
    from .notify import build_notify_helper  # noqa: PLC0415

    library = build_notify_helper()
    prepared_worker.close()
    environment = dict(os.environ)
    prior = environment.get("LD_PRELOAD", "")
    environment["LD_PRELOAD"] = str(library) + (":" + prior if prior else "")
    run_isolated(_native_scene, (payload, device_id, str(directory), str(library)), directory, environment=environment)


def _native_scene(payload, device_id, root, library):
    from .notify import NativeNotifications  # noqa: PLC0415

    module = importlib.import_module("tests.st.a2a3.tensormap_and_ringbuffer.test_dfx_joined_launch_attribution")
    root = Path(root)
    controller = NativeNotifications([root / "agreeing/run0", root / "agreeing/run1", root / "mixed/run0"], library)
    with controller.observe(), device_worker(payload, device_id, root) as (worker, handle):
        worker._dfx_notify = controller  # noqa: SLF001 -- owned by this test scene only
        controller.pids = set(worker._chip_pids)  # noqa: SLF001
        cls = module.TestDfxJoinedLaunchAttribution
        cls._st_chip_handles = {
            "vector": handle,
            "vector_sig": cls.CALLABLE["callables"][0]["orchestration"]["signature"],
        }
        case = cls()
        case._notify = controller  # noqa: SLF001
        case._run_scenarios("a2a3", worker, root)  # noqa: SLF001
        controller.close()


def _entry(payload, result, write_fd, token):
    global _scope_token  # noqa: PLW0603 -- one disposable owner per interpreter
    _scope_token = token
    original_init = SharedMemory.__init__
    sequence = itertools.count()

    def owned_init(self, name=None, create=False, size=0):
        if create and name is None:
            name = f"dg-{token[:16]}-{next(sequence):x}"
        original_init(self, name=name, create=create, size=size)

    SharedMemory.__init__ = owned_init
    try:
        function, args = cloudpickle.loads(Path(payload).read_bytes())
        function(*args)
    except BaseException as error:  # noqa: BLE001 -- report before whole-owner recovery
        outcome = {
            "ok": False,
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "notes": getattr(error, "__notes__", []),
        }
    else:
        outcome = {"ok": True}
    temporary = Path(result).with_suffix(".tmp")
    temporary.write_text(json.dumps(outcome))
    temporary.replace(result)
    sys.stdout.flush()
    sys.stderr.flush()
    os.write(write_fd, b"D")
    # Keep the owner PID reserved, including when executor atexit would hang.
    # The supervisor reclaims the entire group on both success and failure.
    while True:
        signal.pause()


if __name__ == "__main__":
    sys.modules[_MODULE] = sys.modules[__name__]
    try:
        _entry(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
    except BaseException:  # noqa: BLE001 -- never wait on interpreter finalizers after a report failure
        traceback.print_exc()
        os._exit(70)
