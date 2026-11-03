# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Non-owning shared-memory mappings; the creator controls backing lifetime."""

from __future__ import annotations

import _posixshmem
import ctypes
import mmap
import os


class SharedMemoryAttachment:
    """A local mapping without resource-tracker registration or unlink ownership."""

    def __init__(self, name: str):
        self.name = name.lstrip("/")
        fd = _posixshmem.shm_open("/" + self.name, os.O_RDWR, mode=0o600)
        try:
            self._map(fd)
        finally:
            os.close(fd)

    @classmethod
    def from_fd(cls, fd: int) -> SharedMemoryAttachment:
        attachment = cls.__new__(cls)
        attachment.name = f"fd:{fd}"
        attachment._map(fd)
        return attachment

    def _map(self, fd: int) -> None:
        self.size = os.fstat(fd).st_size
        self._mapping = mmap.mmap(fd, self.size)
        self._buffer: memoryview | None = memoryview(self._mapping)
        self.address = ctypes.addressof(ctypes.c_char.from_buffer(self._mapping))

    @property
    def buf(self) -> memoryview | None:
        return self._buffer

    def close(self) -> None:
        if self._buffer is not None:
            self._buffer.release()
            self._buffer = None
        self._mapping.close()

    def __del__(self) -> None:
        if hasattr(self, "_mapping"):
            try:
                self.close()
            except (BufferError, OSError):
                pass
