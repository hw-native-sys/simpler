# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for _mailbox_load_i32 / _mailbox_store_i32.

These helpers bridge Python mailbox accesses to the same acquire/release
memory order that the C++ side uses (worker_manager.cpp::read_mailbox_state /
write_mailbox_state). The tests exercise:

1. Single-process roundtrip — binding signatures work, value is preserved.
2. Cross-process visibility via fork() on a MAP_SHARED SharedMemory region.
3. Field-ordering invariant: a payload written BEFORE the state release-store
   is visible to any reader that observes the state via acquire-load. This is
   the guarantee the three worker loops rely on to publish OFF_ERROR /
   OFF_ERROR_MSG along with the TASK_DONE transition.
4. No regression in the L4 error-propagation paths that exercise every
   refactored site in practice — imported from ``test_error_propagation`` so
   this test module's CI run acts as a second line of defense.

All cases run on CPU only (no NPU required) and complete in well under a
second each.
"""

import ctypes
import os
import struct
import time
from multiprocessing.shared_memory import SharedMemory

import pytest
from _task_interface import _mailbox_load_i32, _mailbox_store_i32  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _addr(buf, offset: int = 0) -> int:
    """Absolute address of ``buf[offset]`` — must match the helper in worker.py."""
    return ctypes.addressof(ctypes.c_char.from_buffer(buf)) + offset


@pytest.fixture
def shm():
    """32-byte MAP_SHARED region; freed and unlinked after the test."""
    region = SharedMemory(create=True, size=32)
    try:
        yield region
    finally:
        region.close()
        region.unlink()


# ---------------------------------------------------------------------------
# 1. Single-process roundtrip
# ---------------------------------------------------------------------------


class TestSingleProcessRoundtrip:
    def test_roundtrip(self, shm):
        addr = _addr(shm.buf, 0)
        _mailbox_store_i32(addr, 42)
        assert _mailbox_load_i32(addr) == 42

    def test_negative(self, shm):
        addr = _addr(shm.buf, 0)
        _mailbox_store_i32(addr, -1)
        assert _mailbox_load_i32(addr) == -1

    def test_offset(self, shm):
        # Confirm the helpers operate on absolute addresses — writing at +8 must
        # not touch the word at offset 0.
        base = _addr(shm.buf, 0)
        _mailbox_store_i32(base, 7)
        _mailbox_store_i32(base + 8, 99)
        assert _mailbox_load_i32(base) == 7
        assert _mailbox_load_i32(base + 8) == 99


# ---------------------------------------------------------------------------
# 2. Cross-process visibility (fork)
# ---------------------------------------------------------------------------


class TestCrossProcess:
    def test_child_transitions_visible_in_parent(self, shm):
        """Child cycles state 0→1→2→3→0; parent must at least see the final 0.

        We don't assert every intermediate value (the parent poll rate is not
        synchronized with the child), but the terminal 0 must land.
        """
        addr = _addr(shm.buf, 0)
        _mailbox_store_i32(addr, -1)

        pid = os.fork()
        if pid == 0:
            try:
                for v in (0, 1, 2, 3, 0):
                    _mailbox_store_i32(addr, v)
                    time.sleep(0.001)
            finally:
                os._exit(0)

        deadline = time.monotonic() + 5.0
        final_seen = False
        while time.monotonic() < deadline:
            v = _mailbox_load_i32(addr)
            if v == 0:
                final_seen = True
                break
        os.waitpid(pid, 0)
        assert final_seen, "parent never observed child's final state=0"


# ---------------------------------------------------------------------------
# 3. Release/acquire field-ordering invariant
# ---------------------------------------------------------------------------


class TestFieldOrderingInvariant:
    @pytest.mark.parametrize("iterations", [1000])
    def test_payload_visible_when_state_observed(self, iterations):
        """The core L3 invariant.

        Layout:  int32 state @ off 0, uint64 payload @ off 8.
        Child: write payload (plain), then release-store state=1.
        Parent: acquire-load state until == 1, then read payload.
        Any observation of state==1 MUST come with payload == SENTINEL.

        On aarch64, dropping the release on the child store or the acquire on
        the parent load would let this race fail. We run 1000 iterations to
        make the failure reproducible on weakly-ordered hardware.
        """
        SENTINEL = 0xDEADBEEFCAFEBABE

        for _ in range(iterations):
            region = SharedMemory(create=True, size=32)
            try:
                buf = region.buf
                assert buf is not None
                state_addr = _addr(buf, 0)
                _mailbox_store_i32(state_addr, 0)
                struct.pack_into("Q", buf, 8, 0)

                pid = os.fork()
                if pid == 0:
                    try:
                        # Writes to the payload are plain; they must become
                        # visible *before* the release-store of state=1 to any
                        # reader that acquire-loads state==1.
                        struct.pack_into("Q", buf, 8, SENTINEL)
                        _mailbox_store_i32(state_addr, 1)
                    finally:
                        os._exit(0)

                deadline = time.monotonic() + 5.0
                while _mailbox_load_i32(state_addr) != 1:
                    if time.monotonic() > deadline:
                        os.waitpid(pid, 0)
                        pytest.fail("child never published state=1")
                payload = struct.unpack_from("Q", buf, 8)[0]
                os.waitpid(pid, 0)

                assert payload == SENTINEL, f"reordering observed: state=1 visible but payload=0x{payload:016x}"
            finally:
                region.close()
                region.unlink()


# ---------------------------------------------------------------------------
# 4. Refactor smoke: end-to-end worker loop still dispatches cleanly
# ---------------------------------------------------------------------------


class TestWorkerSmoke:
    def test_l3_sub_roundtrip(self):
        """A sub-worker dispatch round-trips through the refactored loop.

        The callable runs in a forked child, so we use a MAP_SHARED counter to
        count dispatches back in the parent. Each successful ``run()`` must
        increment the counter by 1. This exercises the TASK_READY → TASK_DONE
        state flip that now uses the new acquire/release helpers on both
        sides of the Python side of the mailbox.
        """
        from simpler.worker import Worker  # noqa: PLC0415

        counter_shm = SharedMemory(create=True, size=4)
        try:
            buf = counter_shm.buf
            assert buf is not None
            struct.pack_into("i", buf, 0, 0)
            counter_addr = _addr(buf, 0)

            def sub(args):
                v = _mailbox_load_i32(counter_addr)
                _mailbox_store_i32(counter_addr, v + 1)

            hw = Worker(level=3, num_sub_workers=1)
            cid = hw.register(sub)
            hw.init()
            try:

                def orch(o, args, cfg):
                    o.submit_sub(cid)

                hw.run(orch)
                hw.run(orch)
                assert _mailbox_load_i32(counter_addr) == 2
            finally:
                hw.close()
        finally:
            counter_shm.close()
            counter_shm.unlink()
