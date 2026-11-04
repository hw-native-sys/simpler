# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Endpoint-owned registration of an existing POSIX Host backing."""

import ctypes

from .comm_endpoints import AdapterProfile


class HostRegistration:
    attachment_profile = AdapterProfile.HOST_REGISTER_MAP

    def __init__(self, *, platform, runtime_path):
        self.platform = platform
        self.runtime_path = runtime_path
        self.native = None
        self.closing = False
        self._base = 0
        self.shm = None

    @property
    def base(self):
        if self.closing or not self._base:
            raise ValueError("Host registration is not live")
        return self._base

    def register(self, device, imported):
        self.shm = imported.shm
        view = self.shm.buf
        address = ctypes.addressof(ctypes.c_char.from_buffer(view))
        offset = imported.base - address
        if self.platform == "a2a3":
            from _task_interface import _HostRegistration  # noqa: PLC0415

            self.native = _HostRegistration(str(self.runtime_path))
            self.native.register(device, address, self.shm.size)
            self._base = self.native.base + offset
        elif self.platform == "a2a3sim":
            self._base = imported.base
        else:
            raise ValueError("Host registration requires A3 or its simulator")

    def release(self):
        self.closing = True
        if self.native is not None:
            self.native.release()
        self._base = 0
        self.shm = None
