# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for Worker (Python L3 wrapper over DistWorker).

Tests use SubWorker (fork/shm) as the only worker type — no NPU device required.
Each test verifies a distinct aspect of the L3 scheduling pipeline.
"""

import struct
import time as _time
from multiprocessing.shared_memory import SharedMemory

import pytest
from simpler.task_interface import WorkerPayload, WorkerType
from simpler.worker import Task, Worker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shared_counter():
    """Allocate a 4-byte shared counter accessible from forked subprocesses."""
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


# ---------------------------------------------------------------------------
# Test: lifecycle (init / close without submitting any tasks)
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_init_close_no_workers(self):
        hw = Worker(level=3, num_sub_workers=0)
        hw.init()
        hw.close()

    def test_init_close_with_sub_workers(self):
        hw = Worker(level=3, num_sub_workers=2)
        hw.init()
        hw.close()

    def test_context_manager(self):
        with Worker(level=3, num_sub_workers=1) as hw:
            hw.register(lambda: None)
        # close() called by __exit__, no exception

    def test_register_after_init_raises(self):
        hw = Worker(level=3, num_sub_workers=0)
        hw.init()
        with pytest.raises(RuntimeError, match="before init"):
            hw.register(lambda: None)
        hw.close()


# ---------------------------------------------------------------------------
# Test: single independent SUB task executes and completes
# ---------------------------------------------------------------------------


class TestSingleSubTask:
    def test_sub_task_executes(self):
        counter_shm, counter_buf = _make_shared_counter()

        try:
            hw = Worker(level=3, num_sub_workers=1)
            cid = hw.register(lambda: _increment_counter(counter_buf))
            hw.init()

            def orch(hw, _args):
                p = WorkerPayload()
                p.worker_type = WorkerType.SUB
                p.callable_id = cid
                hw.submit(WorkerType.SUB, p)

            hw.run(Task(orch=orch))
            hw.close()

            assert _read_counter(counter_buf) == 1
        finally:
            counter_shm.close()
            counter_shm.unlink()

    def test_sub_task_runs_multiple_times(self):
        counter_shm, counter_buf = _make_shared_counter()

        try:
            hw = Worker(level=3, num_sub_workers=1)
            cid = hw.register(lambda: _increment_counter(counter_buf))
            hw.init()

            def orch(hw, _args):
                for _ in range(3):
                    p = WorkerPayload()
                    p.worker_type = WorkerType.SUB
                    p.callable_id = cid
                    hw.submit(WorkerType.SUB, p)

            hw.run(Task(orch=orch))
            hw.close()

            assert _read_counter(counter_buf) == 3
        finally:
            counter_shm.close()
            counter_shm.unlink()


# ---------------------------------------------------------------------------
# Test: multiple SUB workers execute in parallel
# ---------------------------------------------------------------------------


class TestParallelSubWorkers:
    def test_parallel_wall_time(self):
        """Three workers each sleeping 0.2s should finish in <0.54s (not 0.6s)."""
        n = 3
        sleep_s = 0.2
        counters = [SharedMemory(create=True, size=4) for _ in range(n)]
        for c in counters:
            assert c.buf is not None
            struct.pack_into("i", c.buf, 0, 0)

        hw = Worker(level=3, num_sub_workers=n)
        cids = []
        for i in range(n):
            buf = counters[i].buf
            assert buf is not None

            def make_fn(b):
                def fn():
                    _time.sleep(sleep_s)
                    struct.pack_into("i", b, 0, 1)

                return fn

            cids.append(hw.register(make_fn(buf)))
        hw.init()

        def orch(hw, _args):
            for i in range(n):
                p = WorkerPayload()
                p.worker_type = WorkerType.SUB
                p.callable_id = cids[i]
                hw.submit(WorkerType.SUB, p)

        start = _time.monotonic()
        hw.run(Task(orch=orch))
        elapsed = _time.monotonic() - start
        hw.close()

        for c in counters:
            assert c.buf is not None
            assert struct.unpack_from("i", c.buf, 0)[0] == 1
            c.close()
            c.unlink()

        assert elapsed < sleep_s * n * 0.9, (
            f"Expected parallel wall time < {sleep_s * n * 0.9:.2f}s, got {elapsed:.2f}s"
        )


# ---------------------------------------------------------------------------
# Test: output allocation — outputs are accessible after execute()
# ---------------------------------------------------------------------------


class TestOutputAllocation:
    def test_output_buffer_allocated(self):
        hw = Worker(level=3, num_sub_workers=0)
        hw.init()

        def orch(hw, _args):
            p = WorkerPayload()
            # no workers — submit with empty workers list isn't useful here;
            # instead verify that submit() allocates output buffers correctly
            # by using a SUB worker that immediately signals done
            p.worker_type = WorkerType.NEXT_LEVEL  # no NEXT_LEVEL workers — task stays RUNNING
            # For output allocation test, just verify DistSubmitResult has outputs
            # We re-init with sub workers for a real execution test
            pass

        hw.close()

        # Re-test with actual SUB worker + output allocation
        hw2 = Worker(level=3, num_sub_workers=1)
        counter_shm, counter_buf = _make_shared_counter()

        try:
            cid = hw2.register(lambda: _increment_counter(counter_buf))
            hw2.init()

            captured = []

            def orch2(hw, _args):
                p = WorkerPayload()
                p.worker_type = WorkerType.SUB
                p.callable_id = cid
                result = hw.submit(WorkerType.SUB, p, outputs=[64, 128])
                captured.append(result)

            hw2.run(Task(orch=orch2))

            assert len(captured) == 1
            r = captured[0]
            assert r.task_slot >= 0
            assert len(r.outputs) == 2
            assert r.outputs[0].size == 64
            assert r.outputs[1].size == 128
            assert r.outputs[0].ptr != 0
            assert r.outputs[1].ptr != 0
            assert _read_counter(counter_buf) == 1

        finally:
            hw2.close()
            counter_shm.close()
            counter_shm.unlink()


# ---------------------------------------------------------------------------
# Test: scope management
# ---------------------------------------------------------------------------


class TestScope:
    def test_scope_begin_end(self):
        counter_shm, counter_buf = _make_shared_counter()

        try:
            hw = Worker(level=3, num_sub_workers=1)
            cid = hw.register(lambda: _increment_counter(counter_buf))
            hw.init()

            def orch(hw, _args):
                with hw.scope():
                    p = WorkerPayload()
                    p.worker_type = WorkerType.SUB
                    p.callable_id = cid
                    hw.submit(WorkerType.SUB, p)

            hw.run(Task(orch=orch))
            hw.close()

            assert _read_counter(counter_buf) == 1
        finally:
            counter_shm.close()
            counter_shm.unlink()
