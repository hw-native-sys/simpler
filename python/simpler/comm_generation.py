# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Local POSIX communication generations with process-lifetime leases."""

from __future__ import annotations

import contextlib
import fcntl
import os
import re
import stat
import tempfile
import threading
import uuid
import weakref
from collections.abc import Iterator

_GENERATION_NAME = re.compile(r"gen-[0-9a-f]{32}\Z")
_MAX_GENERATIONS = 64
_PARENT_LEASES: weakref.WeakSet[CommGeneration] = weakref.WeakSet()
_FORK_LOCK = threading.RLock()


def _after_fork_child() -> None:
    for owner in _PARENT_LEASES:
        if owner._fd >= 0:
            os.close(owner._fd)
            owner._fd = -1
    _FORK_LOCK.release()


os.register_at_fork(before=_FORK_LOCK.acquire, after_in_parent=_FORK_LOCK.release, after_in_child=_after_fork_child)


def _open_regular(path: str, flags: int) -> int:
    fd = os.open(path, flags | os.O_NOFOLLOW, 0o600)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
        os.close(fd)
        raise RuntimeError(f"invalid communication lease: {path}")
    return fd


@contextlib.contextmanager
def _registry_lock(root: str) -> Iterator[None]:
    try:
        os.mkdir(root, 0o700)
    except FileExistsError:
        pass
    info = os.lstat(root)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise RuntimeError(f"communication registry must be a private owned directory: {root}")
    fd = _open_regular(os.path.join(root, ".registry"), os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _reclaim(root: str) -> int:
    live = 0
    with os.scandir(root) as entries:
        for entry in entries:
            if not _GENERATION_NAME.fullmatch(entry.name):
                continue
            if not entry.is_dir(follow_symlinks=False):
                raise RuntimeError(f"invalid communication generation: {entry.path}")
            lease = os.path.join(entry.path, ".lease")
            try:
                fd = _open_regular(lease, os.O_RDONLY)
            except FileNotFoundError:
                # A crash between mkdir and lease creation leaves an empty generation.
                os.rmdir(entry.path)
                continue
            try:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    live += 1
                    continue
                with os.scandir(entry.path) as files:
                    for marker in files:
                        if marker.name == ".lease":
                            continue
                        if not marker.is_file(follow_symlinks=False):
                            raise RuntimeError(f"unexpected entry in communication generation: {marker.path}")
                        os.unlink(marker.path)
                os.unlink(lease)
                os.rmdir(entry.path)
            finally:
                os.close(fd)
    return live


class CommGeneration:
    """Parent lease; native ranks hold their own lease until communicator destruction.

    An arriving rank must verify the lease path's inode after acquiring SH.
    An EX reclaimer can then unlink safely while a late opener is blocked.
    The registry lock serializes generation creation and reclamation.
    """

    def __init__(self, root: str | None = None) -> None:
        self._fd = -1
        self._root = root or os.path.join(tempfile.gettempdir(), f"simpler-comm-{os.getuid()}")
        if os.path.basename(os.path.normpath(self._root)) != f"simpler-comm-{os.getuid()}":
            raise ValueError("communication generation root must be named simpler-comm-<uid>")
        with _FORK_LOCK, _registry_lock(self._root):
            if _reclaim(self._root) >= _MAX_GENERATIONS:
                raise RuntimeError("communication generation capacity reached; active leases retained")
            directory = os.path.join(self._root, "gen-" + uuid.uuid4().hex)
            os.mkdir(directory, 0o700)
            lease = os.path.join(directory, ".lease")
            fd = _open_regular(lease, os.O_RDWR | os.O_CREAT | os.O_EXCL)
            try:
                fcntl.flock(fd, fcntl.LOCK_SH)
            except BaseException:
                os.close(fd)
                raise
            self._fd = fd
            _PARENT_LEASES.add(self)
            self.rootinfo_path = os.path.join(directory, "rootinfo.bin")

    def close(self) -> None:
        with _FORK_LOCK:
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1
            with _registry_lock(self._root):
                _reclaim(self._root)

    def __del__(self) -> None:
        with _FORK_LOCK:
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1
