# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Multi-worker parallel tests — validates thread isolation introduced in PR 2-3.

DeviceRunner is now thread_local so each ChipWorker thread gets its own instance.
These tests verify that multiple concurrent DistWorker / Worker instances
execute correctly and in parallel without interference.

No NPU device required; SubWorker (fork/shm) is used as the execution backend.
"""

import struct
import threading
import time
from multiprocessing.shared_memory import SharedMemory

from simpler.worker import Task, Worker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _alloc_counter() -> SharedMemory:
    shm = SharedMemory(create=True, size=4)
    assert shm.buf is not None
    struct.pack_into("i", shm.buf, 0, 0)
    return shm


def _read(shm: SharedMemory) -> int:
    assert shm.buf is not None
    return struct.unpack_from("i", shm.buf, 0)[0]


def _inc(buf) -> None:
    v = struct.unpack_from("i", buf, 0)[0]
    struct.pack_into("i", buf, 0, v + 1)


# ---------------------------------------------------------------------------
# Two independent Workers run concurrently
# ---------------------------------------------------------------------------


class TestTwoWorkersParallel:
    """Simulates the multi-device scenario where each Worker manages one device.

    Without thread_local DeviceRunner, two ChipWorker threads sharing a single
    DeviceRunner instance would interfere.  With thread_local, each thread owns
    its own instance and executes independently.
    """

    def test_two_workers_correct_results(self):
        """Each Worker's tasks execute exactly once and in the right worker."""
        counters = [_alloc_counter() for _ in range(2)]
        workers = []

        try:
            for i in range(2):
                buf = counters[i].buf
                assert buf is not None
                hw = Worker(level=3, num_sub_workers=1)
                cid = hw.register(lambda b=buf: _inc(b))
                hw.init()
                workers.append((hw, cid))

            # Submit and execute on both workers (sequential execute, but independent)
            for hw, cid in workers:

                def make_orch(c):
                    def orch(o, _args):
                        o.submit_sub(c)

                    return orch

                hw.run(Task(orch=make_orch(cid)))

            # Each counter must be incremented exactly once
            assert _read(counters[0]) == 1
            assert _read(counters[1]) == 1
            # No cross-contamination
            assert _read(counters[0]) != _read(counters[1]) + 1

        finally:
            for hw, _ in workers:
                hw.close()
            for c in counters:
                c.close()
                c.unlink()

    def test_two_workers_wall_time(self):
        """Two workers with 0.2s tasks should finish in ~0.2s, not 0.4s."""
        sleep_s = 0.2
        counters = [_alloc_counter() for _ in range(2)]
        workers = []
        threads = []

        try:
            for i in range(2):
                buf = counters[i].buf
                assert buf is not None
                hw = Worker(level=3, num_sub_workers=1)

                def make_fn(b, d):
                    def fn():
                        time.sleep(d)
                        _inc(b)

                    return fn

                cid = hw.register(make_fn(buf, sleep_s))
                hw.init()
                workers.append((hw, cid))

            start = time.monotonic()

            def run(hw, cid):
                def orch(o, _args):
                    o.submit_sub(cid)

                hw.run(Task(orch=orch))

            for hw, cid in workers:
                t = threading.Thread(target=run, args=(hw, cid))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            elapsed = time.monotonic() - start

            for c in counters:
                assert _read(c) == 1

            assert elapsed < sleep_s * 2 * 0.9, (
                f"Expected ~{sleep_s}s wall time, got {elapsed:.2f}s (serial would be {sleep_s * 2:.2f}s)"
            )

        finally:
            for hw, _ in workers:
                hw.close()
            for c in counters:
                c.close()
                c.unlink()


# ---------------------------------------------------------------------------
# Many tasks across two workers — no resource leak
# ---------------------------------------------------------------------------


class TestManyTasksNoLeak:
    def test_many_tasks_complete(self):
        """20 sequential tasks through 1 SubWorker — tests ring slot wrap-around."""
        n_tasks = 20
        counter = _alloc_counter()

        try:
            # Single SubWorker: tasks run sequentially, no counter race
            hw = Worker(level=3, num_sub_workers=1)
            buf = counter.buf
            assert buf is not None
            cid = hw.register(lambda: _inc(buf))
            hw.init()

            def orch(o, _args):
                for _ in range(n_tasks):
                    o.submit_sub(cid)

            hw.run(Task(orch=orch))
            hw.close()

            assert _read(counter) == n_tasks

        finally:
            counter.close()
            counter.unlink()

    def test_many_tasks_two_workers_all_complete(self):
        """20 tasks across 2 SubWorkers — each task has a dedicated counter (no shared-counter race)."""
        n_tasks = 20
        counters = [_alloc_counter() for _ in range(n_tasks)]

        try:
            hw = Worker(level=3, num_sub_workers=2)
            cids = []
            for i in range(n_tasks):
                buf = counters[i].buf
                cids.append(hw.register(lambda b=buf: _inc(b)))
            hw.init()

            def orch(o, _args):
                for i in range(n_tasks):
                    o.submit_sub(cids[i])

            hw.run(Task(orch=orch))
            hw.close()

            # Every task's dedicated counter must be exactly 1
            for i, c in enumerate(counters):
                assert _read(c) == 1, f"task {i} counter is {_read(c)}, expected 1"

        finally:
            for c in counters:
                c.close()
                c.unlink()
