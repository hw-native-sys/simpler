# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Owned test-control storage independent of the whole-run admission fence."""

import ctypes
import struct
import time

from simpler.task_interface import DataType

from simpler_setup.scene_test import _temporary_env


def device_gate_environment():
    """Gate expiry precedes the scheduler, op and stream failure backstops."""
    return _temporary_env(
        {
            "SIMPLER_SCHEDULER_TIMEOUT_MS": "15000",
            "SIMPLER_OP_EXECUTE_TIMEOUT_US": "20000000",
            "SIMPLER_STREAM_SYNC_TIMEOUT_MS": "25000",
        }
    )


class DeviceGate:
    """Two cache lines: device publishes entered, host publishes release.

    The native mailbox copies address only this test-owned control allocation.
    Ordinary Worker copies take the admission fence being tested and could not
    release an active predecessor. Both backings stay alive until its run ends.
    """

    def __init__(self, worker, rank=0):
        self.worker = worker
        self.rank = rank
        self.device = (
            worker.malloc(256) if worker.level == 2 else worker.alloc_child_tensor(rank, (32,), DataType.INT64)
        )
        self.host = None
        try:
            self.host = worker.create_buffer(256)
            self.host.shm.buf[:256] = bytes(256)
            worker.copy_to(self.device, self.host, nbytes=256)
        except BaseException:
            worker.free(self.device)
            if self.host is not None:
                self.host.close()
            raise

    @property
    def address(self):
        return self.device.base

    def _copy(self, *, to_device, offset):
        worker = self.worker
        if worker.level == 2:
            host_address = ctypes.addressof(ctypes.c_char.from_buffer(self.host.shm.buf)) + offset
            if to_device:
                worker._chip_worker.copy_to(self.device.base + offset, host_address, 8)
            else:
                worker._chip_worker.copy_from(host_address, self.device.base + offset, 8)
        elif to_device:
            worker._worker.copy_to(self.rank, self.device.to_descriptor(), self.host.to_descriptor(), 8, offset, offset)
        else:
            worker._worker.copy_from(
                self.rank, self.host.to_descriptor(), self.device.to_descriptor(), 8, offset, offset
            )

    def wait_entered(self, timeout=10.0):
        deadline = time.monotonic() + timeout
        while True:
            self._copy(to_device=False, offset=0)
            state = struct.unpack_from("q", self.host.shm.buf, 0)[0]
            assert state != -1, "the device gate expired before release"
            if state == 1:
                return
            assert time.monotonic() < deadline, "the device did not acknowledge entering its gate"

    def release(self):
        struct.pack_into("q", self.host.shm.buf, 128, 1)
        self._copy(to_device=True, offset=128)

    def close(self):
        self.worker.free(self.device)
        self.host.close()
