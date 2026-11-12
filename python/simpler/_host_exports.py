# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host shared-memory exports retained through NumPy and memoryview derivatives."""

from __future__ import annotations

import ctypes
import os
import threading
import weakref
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager, ExitStack, nullcontext
from multiprocessing.shared_memory import SharedMemory
from typing import Any


class HostSharedMemory(SharedMemory):
    """SharedMemory whose exported carriers pin the mmap until their final release."""

    def __init__(self, *args: Any, **kwargs: Any):
        self._export_pid = os.getpid()
        self._export_lock = threading.RLock()
        self._export_count = 0
        self._closing = False
        self._release_guard: Callable[[], AbstractContextManager[None]] = nullcontext
        super().__init__(*args, **kwargs)

    def _check_pid(self) -> None:
        if os.getpid() != self._export_pid:
            raise ValueError("Host mapping belongs to another process")

    @property
    def buf(self) -> memoryview | None:
        """Export bytes while retaining the mapping through all carrier derivatives."""
        self._check_pid()
        with self._export_lock:
            if self._mmap is None:
                return None
            if self._closing:
                raise BufferError("Host mapping is closing; new CPU exports are forbidden")
            # NumPy can keep a temporary memoryview's base instead of the view.
            # The carrier retains a real mmap export until every derivative dies.
            backing = memoryview(self._mmap)
            try:
                address = ctypes.addressof(ctypes.c_ubyte.from_buffer(backing))
                carrier = (ctypes.c_ubyte * self.size).from_address(address)
            except BaseException:
                backing.release()
                raise
            finalizer = weakref.finalize(carrier, self._release_export, backing)
            finalizer.atexit = False
            self._export_count += 1
            return memoryview(carrier).cast("B")

    def _release_export(self, backing: memoryview) -> None:
        if os.getpid() != self._export_pid:
            backing.release()
            return
        with self._export_lock:
            backing.release()
            self._export_count -= 1

    def seal(self) -> None:
        """Reject live exports before permanently closing new export admission."""
        seal_host_mappings((self,))

    def stop_exports(self) -> None:
        """Close export admission while retaining any existing views."""
        self._check_pid()
        with self._export_lock:
            self._closing = True

    def close(self) -> None:
        if not hasattr(self, "_export_pid"):
            return
        self._check_pid()
        with self._release_guard(), self._export_lock:
            self.seal()
            super().close()

    def unlink(self) -> None:
        self._check_pid()
        with self._release_guard(), self._export_lock:
            self.seal()
            super().unlink()


def seal_host_mappings(mappings: Iterable[HostSharedMemory], *, seal: bool = True) -> None:
    """Reject live exports and optionally close admission for the whole set."""
    ordered = sorted({id(mapping): mapping for mapping in mappings}.values(), key=id)
    for mapping in ordered:
        mapping._check_pid()
    with ExitStack() as locks:
        for mapping in ordered:
            locks.enter_context(mapping._export_lock)
        for mapping in ordered:
            if mapping._export_count:
                raise BufferError(
                    f"Host mapping {mapping.name!r} has live CPU exports; "
                    "drop arrays and release memoryviews before closing or releasing storage"
                )
        if seal:
            for mapping in ordered:
                mapping._closing = True
