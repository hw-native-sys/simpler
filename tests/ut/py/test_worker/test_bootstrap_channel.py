# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for ChipBootstrapChannel (per-chip bootstrap mailbox).

All tests run without hardware — pure shared-memory / in-process.
"""

import ctypes
import os
from multiprocessing.shared_memory import SharedMemory

import pytest
from _task_interface import (  # pyright: ignore[reportMissingImports]
    CHIP_BOOTSTRAP_MAILBOX_SIZE,
    ChipBootstrapChannel,
    ChipBootstrapMailboxState,
)


def _mailbox_addr(shm: SharedMemory) -> int:
    """Return the raw memory address of a SharedMemory buffer."""
    buf = shm.buf
    assert buf is not None
    return ctypes.addressof(ctypes.c_char.from_buffer(buf))


class TestBootstrapChannel:
    def test_fresh_channel_state_idle(self):
        """Freshly reset channel reads as IDLE."""
        shm = SharedMemory(create=True, size=CHIP_BOOTSTRAP_MAILBOX_SIZE)
        try:
            ch = ChipBootstrapChannel(_mailbox_addr(shm), max_buffer_count=376)
            # buf is zeroed by SharedMemory — state at offset 0 is 0 (IDLE)
            assert ch.state == ChipBootstrapMailboxState.IDLE
        finally:
            shm.close()
            shm.unlink()

    def test_write_success_fields(self):
        """write_success stores all fields and parent reads them back."""
        shm = SharedMemory(create=True, size=CHIP_BOOTSTRAP_MAILBOX_SIZE)
        try:
            ch = ChipBootstrapChannel(_mailbox_addr(shm), max_buffer_count=376)
            ch.reset()
            ch.write_success(
                device_ctx=0xDEADBEEFCAFE1234,
                local_window_base=0xAABBCCDD00112233,
                actual_window_size=65536,
                buffer_ptrs=[0x1000, 0x2000, 0x3000],
            )
            assert ch.state == ChipBootstrapMailboxState.SUCCESS
            assert ch.device_ctx == 0xDEADBEEFCAFE1234
            assert ch.local_window_base == 0xAABBCCDD00112233
            assert ch.actual_window_size == 65536
            assert ch.buffer_ptrs == [0x1000, 0x2000, 0x3000]
        finally:
            shm.close()
            shm.unlink()

    def test_write_error_fields(self):
        """write_error stores error_code and message."""
        shm = SharedMemory(create=True, size=CHIP_BOOTSTRAP_MAILBOX_SIZE)
        try:
            ch = ChipBootstrapChannel(_mailbox_addr(shm), max_buffer_count=376)
            ch.reset()
            ch.write_error(42, "device not found")
            assert ch.state == ChipBootstrapMailboxState.ERROR
            assert ch.error_code == 42
            assert ch.error_message == "device not found"
        finally:
            shm.close()
            shm.unlink()

    def test_state_machine_reset(self):
        """write_success -> SUCCESS, reset -> IDLE."""
        shm = SharedMemory(create=True, size=CHIP_BOOTSTRAP_MAILBOX_SIZE)
        try:
            ch = ChipBootstrapChannel(_mailbox_addr(shm), max_buffer_count=376)
            ch.reset()
            ch.write_success(0, 0, 0, [])
            assert ch.state == ChipBootstrapMailboxState.SUCCESS
            ch.reset()
            assert ch.state == ChipBootstrapMailboxState.IDLE
        finally:
            shm.close()
            shm.unlink()

    def test_cross_process_fork(self):
        """Parent allocates shm, forks, child writes, parent reads after fork."""
        shm = SharedMemory(create=True, size=CHIP_BOOTSTRAP_MAILBOX_SIZE)
        try:
            addr = _mailbox_addr(shm)
            pid = os.fork()
            if pid == 0:
                # Child: wrap same shm and write success.
                ch = ChipBootstrapChannel(addr, max_buffer_count=376)
                ch.reset()
                ch.write_success(
                    device_ctx=0x1111222233334444,
                    local_window_base=0x5555666677778888,
                    actual_window_size=128,
                    buffer_ptrs=[0xA, 0xB, 0xC, 0xD],
                )
                os._exit(0)
            else:
                # Parent: poll until SUCCESS.
                ch = ChipBootstrapChannel(addr, max_buffer_count=376)
                while ch.state == ChipBootstrapMailboxState.IDLE:
                    pass
                assert ch.state == ChipBootstrapMailboxState.SUCCESS
                assert ch.device_ctx == 0x1111222233334444
                assert ch.local_window_base == 0x5555666677778888
                assert ch.actual_window_size == 128
                assert ch.buffer_ptrs == [0xA, 0xB, 0xC, 0xD]
                os.waitpid(pid, 0)
        finally:
            shm.close()
            shm.unlink()

    def test_buffer_ptrs_overflow(self):
        """write_success with too many ptrs throws."""
        shm = SharedMemory(create=True, size=CHIP_BOOTSTRAP_MAILBOX_SIZE)
        try:
            ch = ChipBootstrapChannel(_mailbox_addr(shm), max_buffer_count=376)
            ch.reset()
            too_many = list(range(377))
            with pytest.raises(ValueError, match="buffer_ptrs exceeds max_buffer_count"):
                ch.write_success(0, 0, 0, too_many)
        finally:
            shm.close()
            shm.unlink()

    def test_error_message_truncation(self):
        """write_error with >1024 byte message truncates to 1023 + null."""
        shm = SharedMemory(create=True, size=CHIP_BOOTSTRAP_MAILBOX_SIZE)
        try:
            ch = ChipBootstrapChannel(_mailbox_addr(shm), max_buffer_count=376)
            ch.reset()
            long_msg = "x" * 2000
            ch.write_error(-1, long_msg)
            assert ch.state == ChipBootstrapMailboxState.ERROR
            msg = ch.error_message
            assert len(msg) == 1023
            assert msg == "x" * 1023
        finally:
            shm.close()
            shm.unlink()
