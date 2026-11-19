# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent HBM control words for a device-observed release handshake."""

import struct
import time
from contextlib import suppress

from simpler.task_interface import DataType


class DeviceGate:
    """Device publishes entered at byte 0; the CPU publishes release at byte 128.

    Native mailbox copies address only these control buffers. Public Worker
    copies take the whole-run admission fence and cannot release an active run.
    Both buffers remain allocated until that run completes.
    """

    def __init__(self, worker, worker_id):
        self.worker = worker
        self.worker_id = worker_id
        self.device = worker.alloc_child_tensor(worker_id, (32,), DataType.INT64)
        self.host = None
        try:
            self.host = worker.create_buffer(256)
            with self.host.shm.buf[:] as memory:
                memory[:] = bytes(256)
            worker.copy_to(self.device, self.host, nbytes=256)
        except BaseException:
            worker.free(self.device)
            if self.host is not None:
                self.host.close()
            raise

    @property
    def address(self):
        return self.device.base

    def wait_entered(self, timeout=10.0):
        assert self.host is not None and self.host.shm is not None and self.host.shm.buf is not None
        deadline = time.monotonic() + timeout
        while True:
            self.worker._worker.copy_from(  # noqa: SLF001 -- independent test-control buffer, outside run admission
                self.worker_id, self.host.to_descriptor(), self.device.to_descriptor(), 8, 0, 0
            )
            with self.host.shm.buf[:] as memory:
                state = struct.unpack_from("q", memory, 0)[0]
            if state == -1:
                raise AssertionError("the MTE gate expired before release")
            if state == 1:
                return
            if time.monotonic() >= deadline:
                raise AssertionError("the device did not acknowledge the MTE gate")

    def release(self):
        assert self.host is not None and self.host.shm is not None and self.host.shm.buf is not None
        with self.host.shm.buf[:] as memory:
            struct.pack_into("q", memory, 128, 1)
        self.worker._worker.copy_to(  # noqa: SLF001 -- independent test-control buffer, outside run admission
            self.worker_id, self.device.to_descriptor(), self.host.to_descriptor(), 8, 128, 128
        )

    def close(self):
        assert self.host is not None
        self.worker.free(self.device)
        self.host.close()


def release_domain_during_reads(worker, domain, start_run):
    gates = []
    pending = None
    try:
        for worker_id in (0, 1):
            gates.append(DeviceGate(worker, worker_id))
        pending = start_run(gates)
        for gate in gates:
            gate.wait_entered()
        assert not pending.done
        domain.release()
        assert domain.released and not domain.freed
        for gate in gates:
            gate.release()
        pending.wait()
    finally:
        for gate in gates:
            with suppress(Exception):
                gate.release()
        if pending is not None:
            with suppress(Exception):
                pending.wait()
        if pending is None or pending.done:
            for gate in gates:
                gate.close()
