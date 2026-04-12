# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import ctypes
import os
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "examples" / "scripts"))

from code_runner import CodeRunner, create_code_runner  # noqa: E402
from task_interface import (  # noqa: E402
    DIST_CHIP_BOOTSTRAP_MAILBOX_SIZE,
    ChipBootstrapMailboxState,
    DistChipBootstrapChannel,
)
from worker import Worker  # noqa: E402


def _mailbox_addr(shm: SharedMemory) -> int:
    assert shm.buf is not None
    return ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))


class TestDistChipBootstrapChannel:
    def test_success_roundtrip(self):
        shm = SharedMemory(create=True, size=DIST_CHIP_BOOTSTRAP_MAILBOX_SIZE)
        try:
            ch = DistChipBootstrapChannel(_mailbox_addr(shm), 2)
            ch.reset()
            ch.write_success(11, 22, 33, [44, 55])

            assert ch.state == ChipBootstrapMailboxState.SUCCESS
            assert ch.error_code == 0
            assert ch.device_ctx == 11
            assert ch.local_window_base == 22
            assert ch.actual_window_size == 33
            assert ch.buffer_ptrs == [44, 55]
        finally:
            shm.close()
            shm.unlink()

    def test_error_roundtrip(self):
        shm = SharedMemory(create=True, size=DIST_CHIP_BOOTSTRAP_MAILBOX_SIZE)
        try:
            ch = DistChipBootstrapChannel(_mailbox_addr(shm), 1)
            ch.reset()
            ch.write_error(7, "bootstrap failed")

            assert ch.state == ChipBootstrapMailboxState.ERROR
            assert ch.error_code == 7
            assert "bootstrap failed" in ch.error_message
        finally:
            shm.close()
            shm.unlink()


class TestDistributedWorkerApi:
    def test_create_code_runner_returns_unified_code_runner_for_distributed(self):
        kernels_dir = ROOT / "examples" / "a2a3" / "tensormap_and_ringbuffer" / "async_notify_demo" / "kernels"
        golden_path = ROOT / "examples" / "a2a3" / "tensormap_and_ringbuffer" / "async_notify_demo" / "golden.py"

        old_platform = os.environ.get("PTO_PLATFORM")
        os.environ["PTO_PLATFORM"] = "a2a3"
        try:
            runner = create_code_runner(
                kernels_dir=str(kernels_dir),
                golden_path=str(golden_path),
                platform="a2a3",
                nranks=2,
                device_ids=[0, 1],
            )
        finally:
            if old_platform is None:
                os.environ.pop("PTO_PLATFORM", None)
            else:
                os.environ["PTO_PLATFORM"] = old_platform

        assert isinstance(runner, CodeRunner)
        assert runner._is_distributed is True
        assert runner.nranks == 2
        assert runner.device_ids == [0, 1]
