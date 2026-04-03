# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""POC: HostSubWorker via fork + shared memory.

Verifies the full communication path:
  1. mmap(MAP_SHARED) is bidirectional after fork
  2. torch.share_memory_() tensor is accessible (zero-copy) in forked child
  3. Callable registry is accessible in forked child (no pickle needed)
  4. Mailbox state machine: IDLE → TASK_READY → TASK_DONE cycles correctly
  5. Multiple workers execute pure-Python in parallel (wall time < serial)
  6. C++ threading (via Python threading module) after fork is safe
"""

import os
import struct
import threading
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Callable

import torch

# ---------------------------------------------------------------------------
# Mailbox layout (256 bytes per worker, fits in 4 cache lines)
# ---------------------------------------------------------------------------
#   offset  0  int32  state         IDLE=0, TASK_READY=1, TASK_DONE=2, SHUTDOWN=3
#   offset  4  int32  callable_id
#   offset  8  int64  result_int    worker writes a simple int result for the POC
#   offset 16  int32  error_code    0 = ok
# ---------------------------------------------------------------------------

MAILBOX_SIZE = 256
IDLE = 0
TASK_READY = 1
TASK_DONE = 2
SHUTDOWN = 3

_STATE_OFF = 0
_CID_OFF = 4
_RESULT_OFF = 8
_ERR_OFF = 16


def _mb_read_state(buf) -> int:
    return struct.unpack_from("i", buf, _STATE_OFF)[0]


def _mb_write(buf, state: int, cid: int = 0) -> None:
    struct.pack_into("i", buf, _CID_OFF, cid)
    # write state last so worker sees consistent mailbox
    struct.pack_into("i", buf, _STATE_OFF, state)


def _mb_write_result(buf, result: int, error: int = 0) -> None:
    struct.pack_into("q", buf, _RESULT_OFF, result)
    struct.pack_into("i", buf, _ERR_OFF, error)
    struct.pack_into("i", buf, _STATE_OFF, TASK_DONE)


def _mb_read_result(buf) -> tuple[int, int]:
    result = struct.unpack_from("q", buf, _RESULT_OFF)[0]
    error = struct.unpack_from("i", buf, _ERR_OFF)[0]
    return result, error


# ---------------------------------------------------------------------------
# Worker process main loop
# ---------------------------------------------------------------------------


def _worker_loop(buf, registry: dict) -> None:
    """Runs in forked child process. buf is a SharedMemory.buf memoryview."""
    while True:
        state = _mb_read_state(buf)

        if state == TASK_READY:
            cid = struct.unpack_from("i", buf, _CID_OFF)[0]
            fn = registry.get(cid)
            if fn is None:
                _mb_write_result(buf, 0, error=1)
                continue
            try:
                result = fn()
                _mb_write_result(buf, result, error=0)
            except Exception:  # noqa: BLE001
                _mb_write_result(buf, 0, error=2)

        elif state == SHUTDOWN:
            break
        # tight spin (same as L2 AICPU spin-wait — no yield)


# ---------------------------------------------------------------------------
# Minimal HostSubWorker pool
# ---------------------------------------------------------------------------


class _SubWorkerPool:
    """
    Fork-based worker pool.  Must be constructed before any threads are started.
    callable_registry maps int → () -> int for this POC.
    """

    def __init__(self, num_workers: int, registry: dict[int, Callable]):
        self._num_workers = num_workers
        self._shms: list[SharedMemory] = []
        self._pids: list[int] = []

        for _ in range(num_workers):
            shm = SharedMemory(create=True, size=MAILBOX_SIZE)
            assert shm.buf is not None
            struct.pack_into("i", shm.buf, _STATE_OFF, IDLE)
            self._shms.append(shm)

        # fork after all mailboxes are allocated — single-threaded here
        for i in range(num_workers):
            pid = os.fork()
            if pid == 0:
                # child: only run this worker's loop then exit
                buf = self._shms[i].buf
                assert buf is not None
                _worker_loop(buf, registry)
                os._exit(0)  # skip pytest atexit handlers
            else:
                self._pids.append(pid)

    def dispatch(self, worker_idx: int, callable_id: int) -> None:
        buf = self._shms[worker_idx].buf
        assert buf is not None
        _mb_write(buf, TASK_READY, cid=callable_id)

    def wait(self, worker_idx: int, timeout: float = 5.0) -> tuple[int, int]:
        buf = self._shms[worker_idx].buf
        assert buf is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if _mb_read_state(buf) == TASK_DONE:
                result, err = _mb_read_result(buf)
                _mb_write(buf, IDLE)
                return result, err
        raise TimeoutError(f"worker {worker_idx} did not complete within {timeout}s")

    def shutdown(self) -> None:
        for shm in self._shms:
            buf = shm.buf
            assert buf is not None
            _mb_write(buf, SHUTDOWN)
        for pid in self._pids:
            os.waitpid(pid, 0)
        for shm in self._shms:
            shm.close()
            shm.unlink()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestMapSharedAfterFork:
    """Case 1 — SharedMemory is bidirectional after fork."""

    def test_parent_writes_child_reads(self):
        shm = SharedMemory(create=True, size=64)
        buf = shm.buf
        assert buf is not None
        struct.pack_into("i", buf, 0, 0)
        struct.pack_into("i", buf, 4, 0)

        pid = os.fork()
        if pid == 0:
            # child: spin until parent writes 42
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if struct.unpack_from("i", buf, 0)[0] == 42:
                    struct.pack_into("i", buf, 4, 99)  # ack
                    os._exit(0)
            os._exit(1)

        # parent: write 42, wait for ack
        struct.pack_into("i", buf, 0, 42)
        deadline = time.monotonic() + 2.0
        ack = 0
        while time.monotonic() < deadline:
            ack = struct.unpack_from("i", buf, 4)[0]
            if ack == 99:
                break

        _, status = os.waitpid(pid, 0)
        shm.close()
        shm.unlink()
        assert os.WEXITSTATUS(status) == 0, "child exited with error"
        assert ack == 99, "child did not write ack into shared memory"


class TestTorchShareMemoryAfterFork:
    """Case 2 — torch.share_memory_() tensor is zero-copy accessible in child."""

    def test_child_reads_and_mutates_shared_tensor(self):
        t = torch.tensor([10.0, 20.0, 30.0])
        t.share_memory_()
        assert t.is_shared()

        # shared flag: child signals completion here
        shm = SharedMemory(create=True, size=16)
        shm_buf = shm.buf
        assert shm_buf is not None
        struct.pack_into("i", shm_buf, 0, 0)  # done flag

        pid = os.fork()
        if pid == 0:
            # child sees same physical pages — read and mutate
            val = t[0].item()
            if val != 10.0:
                os._exit(2)
            t[0] = 99.0
            struct.pack_into("i", shm_buf, 0, 1)  # done
            os._exit(0)

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if struct.unpack_from("i", shm_buf, 0)[0] == 1:
                break

        _, status = os.waitpid(pid, 0)
        shm.close()
        shm.unlink()
        assert os.WEXITSTATUS(status) == 0, f"child exit {os.WEXITSTATUS(status)}"
        # parent sees child's mutation — same physical page
        assert t[0].item() == 99.0, f"expected 99.0, got {t[0].item()}"


class TestCallableRegistryAfterFork:
    """Case 3 — callable registry is accessible in child without pickle."""

    def test_child_calls_registered_fn(self):
        registry = {0: lambda: 1234}

        shm = SharedMemory(create=True, size=16)
        shm_buf = shm.buf
        assert shm_buf is not None
        struct.pack_into("q", shm_buf, 0, 0)

        pid = os.fork()
        if pid == 0:
            fn = registry[0]
            result = fn()
            struct.pack_into("q", shm_buf, 0, result)
            os._exit(0)

        _, status = os.waitpid(pid, 0)
        result = struct.unpack_from("q", shm_buf, 0)[0]
        shm.close()
        shm.unlink()
        assert os.WEXITSTATUS(status) == 0
        assert result == 1234


class TestMailboxStateMachine:
    """Case 4 — mailbox state machine: IDLE → TASK_READY → TASK_DONE, multiple rounds."""

    def test_multiple_rounds(self):
        registry = {0: lambda: 42, 1: lambda: 99}
        pool = _SubWorkerPool(num_workers=1, registry=registry)

        try:
            for cid, expected in [(0, 42), (1, 99), (0, 42)]:
                pool.dispatch(0, cid)
                result, err = pool.wait(0)
                assert err == 0
                assert result == expected
        finally:
            pool.shutdown()


class TestParallelExecution:
    """Case 5 — multiple workers execute pure-Python in parallel.

    Each task sleeps for 0.2s (pure Python, holds GIL).  With N workers we
    expect wall time close to 0.2s rather than N * 0.2s.
    """

    def _make_sleep_fn(self, duration: float) -> Callable[[], int]:
        def fn():
            time.sleep(duration)
            return int(duration * 1000)

        return fn

    def test_parallel_wall_time(self):
        n_workers = 3
        sleep_sec = 0.2
        registry = {i: self._make_sleep_fn(sleep_sec) for i in range(n_workers)}
        pool = _SubWorkerPool(num_workers=n_workers, registry=registry)

        try:
            start = time.monotonic()
            for i in range(n_workers):
                pool.dispatch(i, i)
            for i in range(n_workers):
                result, err = pool.wait(i, timeout=5.0)
                assert err == 0
                assert result == int(sleep_sec * 1000)
            elapsed = time.monotonic() - start

            serial_time = n_workers * sleep_sec
            assert elapsed < serial_time * 0.7, (
                f"expected parallel wall time < {serial_time * 0.7:.2f}s "
                f"(serial would be {serial_time:.2f}s), got {elapsed:.2f}s"
            )
        finally:
            pool.shutdown()


class TestThreadingAfterFork:
    """Case 6 — starting Python threads after fork does not deadlock."""

    def test_thread_starts_cleanly(self):
        # fork first (simulating HostWorker.__init__ order)
        shm = SharedMemory(create=True, size=8)
        shm_buf = shm.buf
        assert shm_buf is not None
        struct.pack_into("i", shm_buf, 0, 0)
        pid = os.fork()
        if pid == 0:
            struct.pack_into("i", shm_buf, 0, 1)
            os._exit(0)
        os.waitpid(pid, 0)
        assert struct.unpack_from("i", shm_buf, 0)[0] == 1
        shm.close()
        shm.unlink()

        # now start a thread in the parent (simulating Scheduler/ChipWorker threads)
        results = []

        def thread_fn():
            results.append(threading.get_ident())

        t = threading.Thread(target=thread_fn)
        t.start()
        t.join(timeout=2.0)
        assert not t.is_alive(), "thread did not finish"
        assert len(results) == 1
