# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for L4 error propagation from child workers up to Worker.run().

Covers the three failure paths that the mailbox OFF_ERROR / OFF_ERROR_MSG
channel has to carry:

1. SubWorker callable raises   →  Worker.run(orch) re-raises with original
                                  Python exception type + message in the text.
2. Scope mid-failure           →  Nth submit sees has_error, next Worker.run
                                  is not wedged (error state cleared).
3. L4 → L3 → SubWorker chain   →  Exception raised in bottom sub surfaces at
                                  L4 Worker.run() with an identifiable chain
                                  (sub_worker / child_worker prefixes).

All cases run on sub-workers only — no NPU required.
"""

import struct
import time
from multiprocessing.shared_memory import SharedMemory

import pytest
from simpler.task_interface import ChipCallConfig, TaskArgs
from simpler.worker import Worker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shared_counter():
    shm = SharedMemory(create=True, size=4)
    buf = shm.buf
    assert buf is not None
    struct.pack_into("i", buf, 0, 0)
    return shm, buf


def _read_counter(buf) -> int:
    return struct.unpack_from("i", buf, 0)[0]


def _increment_counter(buf) -> None:
    v = struct.unpack_from("i", buf, 0)[0]
    struct.pack_into("i", buf, 0, v + 1)


class SentinelError(ValueError):
    pass


# ---------------------------------------------------------------------------
# 1. SubWorker callable raises — original type+message preserved
# ---------------------------------------------------------------------------


class TestSubWorkerException:
    def test_sub_callable_raises_surfaces_at_run(self):
        def boom(args):
            raise SentinelError("boom-from-sub")

        hw = Worker(level=3, num_sub_workers=1)
        cid = hw.register(boom)
        hw.init()
        try:

            def orch(o, args, cfg):
                o.submit_sub(cid)

            with pytest.raises(RuntimeError) as info:
                hw.run(orch)

            text = str(info.value)
            assert "sub_worker" in text
            assert "SentinelError" in text
            assert "boom-from-sub" in text
        finally:
            hw.close()

    def test_sub_callable_missing_id_surfaces(self):
        hw = Worker(level=3, num_sub_workers=1)
        hw.init()
        try:

            def orch(o, args, cfg):
                o.submit_sub(42)

            with pytest.raises(RuntimeError) as info:
                hw.run(orch)
            assert "not registered" in str(info.value)
        finally:
            hw.close()


# ---------------------------------------------------------------------------
# 2. Scope mid-failure — drain rethrows once, next run clean
# ---------------------------------------------------------------------------


class TestScopeMidFailure:
    def test_failure_does_not_wedge_worker(self):
        """After a run() fails, the next run() using a clean orch succeeds.

        Register two callables — one that always raises, one that increments
        a shared counter. Fire the failing one first (should raise) then the
        succeeding one (should run to completion). Proves ``_clear_error`` at
        the top of ``Worker.run`` resets the error slot so the next run is
        not permanently poisoned.

        Two callables rather than a shared-state toggle: the sub-worker is
        a forked child that inherits closure state copy-on-write; updates
        the parent makes to a Python dict are invisible on the child side.
        """
        counter_shm, counter_buf = _make_shared_counter()

        try:

            def boom(args):
                raise SentinelError("first run failure")

            hw = Worker(level=3, num_sub_workers=1)
            fail_cid = hw.register(boom)
            ok_cid = hw.register(lambda args: _increment_counter(counter_buf))
            hw.init()
            try:

                def failing_orch(o, args, cfg):
                    o.submit_sub(fail_cid)

                def ok_orch(o, args, cfg):
                    o.submit_sub(ok_cid)

                with pytest.raises(RuntimeError):
                    hw.run(failing_orch)

                hw.run(ok_orch)
                assert _read_counter(counter_buf) == 1

                hw.run(ok_orch)
                assert _read_counter(counter_buf) == 2
            finally:
                hw.close()
        finally:
            counter_shm.close()
            counter_shm.unlink()

    def test_subsequent_submit_raises_after_failure(self):
        """A second submit_sub after a failed first one must not swallow the error.

        With one sub worker we submit two tasks sequentially in the orch fn;
        the first one fails, so the second submit sees has_error and re-raises
        immediately (fail-fast). The exception reaching Worker.run is the
        original child failure — not some generic "orchestrator poisoned"
        wrapper.
        """

        def boom(args):
            raise SentinelError("fail-fast")

        hw = Worker(level=3, num_sub_workers=1)
        cid = hw.register(boom)
        hw.init()
        try:

            def orch(o, args, cfg):
                o.submit_sub(cid)
                # Give the child enough wall-clock time to run and fail
                # before issuing the second submit, so the fail-fast check
                # in submit_impl has something to trip on.
                time.sleep(1.0)
                o.submit_sub(cid)

            with pytest.raises(RuntimeError) as info:
                hw.run(orch)
            assert "SentinelError" in str(info.value)
            assert "fail-fast" in str(info.value)
        finally:
            hw.close()


# ---------------------------------------------------------------------------
# 3. L4 → L3 → SubWorker chain — identifiable propagation
# ---------------------------------------------------------------------------


class TestL4ChainedFailure:
    def test_bottom_sub_failure_surfaces_at_l4(self):
        """A SentinelError in the innermost sub callable must reach the L4
        caller's run() with enough context to identify both layers.

        The error first bubbles from the SubWorker through the parent's
        dispatch_process as `std::runtime_error("sub_worker: ... SentinelError: ...")`.
        Inside the L3 orch fn this reaches _drain which rethrows. The L3
        child process catches that in _child_worker_loop and rewrites it
        as `child_worker level=3: RuntimeError: <original>`. The L4 parent's
        dispatch_process rethrows again. The final string visible at the L4
        caller contains both prefixes.
        """

        def boom_sub(args):
            raise SentinelError("l3-sub-boom")

        l3 = Worker(level=3, num_sub_workers=1)
        l3_sub_cid = l3.register(boom_sub)

        def l3_orch(orch, args, config):
            orch.submit_sub(l3_sub_cid)

        w4 = Worker(level=4, num_sub_workers=0)
        l3_cid = w4.register(l3_orch)
        w4.add_worker(l3)
        w4.init()
        try:

            def l4_orch(orch, args, config):
                orch.submit_next_level(l3_cid, TaskArgs(), ChipCallConfig())

            with pytest.raises(RuntimeError) as info:
                w4.run(l4_orch)
            text = str(info.value)
            # Two layers of prefix should both be visible in the message chain.
            assert "child_worker" in text
            assert "sub_worker" in text
            assert "SentinelError" in text
            assert "l3-sub-boom" in text
        finally:
            w4.close()
