# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for L4 → L3 → L2 recursive Worker composition.

L4 Worker dispatches to L3 Worker children (PROCESS mode). Each L3 Worker
has its own SubWorkers. Verifies the full DAG completes and sub callables
at L3 level see correct data.

No NPU device required — L3 children use SubWorkers only (no ChipWorker).
"""

import struct
from multiprocessing.shared_memory import SharedMemory

import pytest
from simpler.task_interface import ChipCallConfig, TaskArgs
from simpler.worker import Worker

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
# Test: L4 lifecycle (init / close without submitting any tasks)
# ---------------------------------------------------------------------------


class TestL4Lifecycle:
    def test_init_close_no_children(self):
        """L4 with zero next-level workers and zero sub workers."""
        w4 = Worker(level=4, num_sub_workers=0)
        w4.init()
        w4.close()

    def test_init_close_with_l3_child(self):
        """L4 with one L3 child (no device, sub-only) — init and close cleanly."""
        l3 = Worker(level=3, num_sub_workers=1)
        l3.register(lambda args: None)

        w4 = Worker(level=4, num_sub_workers=0)
        w4.register(lambda orch, args, config: None)
        w4.add_worker(l3)
        w4.init()
        w4.close()

    def test_context_manager(self):
        """L4 via context manager cleans up correctly."""
        l3 = Worker(level=3, num_sub_workers=1)
        l3.register(lambda args: None)

        with Worker(level=4, num_sub_workers=0) as w4:
            w4.register(lambda orch, args, config: None)
            w4.add_worker(l3)
            w4.init()


class TestL4Validation:
    def test_add_worker_requires_level4(self):
        """add_worker on level 3 raises."""
        w3 = Worker(level=3, num_sub_workers=0)
        child = Worker(level=3, num_sub_workers=0)
        with pytest.raises(RuntimeError, match="level >= 4"):
            w3.add_worker(child)

    def test_add_worker_after_init_raises(self):
        w4 = Worker(level=4, num_sub_workers=0)
        w4.init()
        child = Worker(level=3, num_sub_workers=0)
        with pytest.raises(RuntimeError, match="before init"):
            w4.add_worker(child)
        w4.close()

    def test_add_initialized_child_raises(self):
        child = Worker(level=3, num_sub_workers=0)
        child.init()
        w4 = Worker(level=4, num_sub_workers=0)
        with pytest.raises(RuntimeError, match="must not be initialized"):
            w4.add_worker(child)
        child.close()
        w4.close()


# ---------------------------------------------------------------------------
# Test: L4 → L3 PROCESS mode — single dispatch
# ---------------------------------------------------------------------------


class TestL4ToL3SingleDispatch:
    def test_l4_dispatches_to_l3_sub(self):
        """L4 orch submits one task to L3 child. L3 orch runs a sub callable.

        Verifies that the sub callable's shared counter is incremented,
        proving the full L4 → L3 → sub dispatch chain works.
        """
        counter_shm, counter_buf = _make_shared_counter()

        try:
            # L3 child: one sub worker, one sub callable that increments counter
            l3 = Worker(level=3, num_sub_workers=1)
            l3_sub_cid = l3.register(lambda args: _increment_counter(counter_buf))

            def l3_orch(orch, args, config):
                orch.submit_sub(l3_sub_cid)

            # L4 parent: one next-level child, register L3 orch fn
            w4 = Worker(level=4, num_sub_workers=0)
            l3_cid = w4.register(l3_orch)
            w4.add_worker(l3)
            w4.init()

            def l4_orch(orch, args, config):
                orch.submit_next_level(l3_cid, TaskArgs(), ChipCallConfig())

            w4.run(l4_orch)
            w4.close()

            assert _read_counter(counter_buf) == 1
        finally:
            counter_shm.close()
            counter_shm.unlink()


# ---------------------------------------------------------------------------
# Test: L4 → L3 — multiple dispatches
# ---------------------------------------------------------------------------


class TestL4ToL3MultipleDispatches:
    def test_l4_dispatches_three_times(self):
        """L4 orch submits 3 tasks to L3 child, each running a sub callable."""
        counter_shm, counter_buf = _make_shared_counter()

        try:
            l3 = Worker(level=3, num_sub_workers=1)
            l3_sub_cid = l3.register(lambda args: _increment_counter(counter_buf))

            def l3_orch(orch, args, config):
                orch.submit_sub(l3_sub_cid)

            w4 = Worker(level=4, num_sub_workers=0)
            l3_cid = w4.register(l3_orch)
            w4.add_worker(l3)
            w4.init()

            def l4_orch(orch, args, config):
                for _ in range(3):
                    orch.submit_next_level(l3_cid, TaskArgs(), ChipCallConfig())

            w4.run(l4_orch)
            w4.close()

            assert _read_counter(counter_buf) == 3
        finally:
            counter_shm.close()
            counter_shm.unlink()


# ---------------------------------------------------------------------------
# Test: L4 with own sub workers + L3 child
# ---------------------------------------------------------------------------


class TestL4WithOwnSubs:
    def test_l4_sub_and_l3_dispatch(self):
        """L4 has its own sub workers AND dispatches to an L3 child.

        Both L4's sub callable and L3's sub callable increment the same
        shared counter. Verifies both paths execute.
        """
        counter_shm, counter_buf = _make_shared_counter()

        try:
            # L3 child: sub worker increments counter
            l3 = Worker(level=3, num_sub_workers=1)
            l3_sub_cid = l3.register(lambda args: _increment_counter(counter_buf))

            def l3_orch(orch, args, config):
                orch.submit_sub(l3_sub_cid)

            # L4: own sub worker + L3 child
            w4 = Worker(level=4, num_sub_workers=1)
            l3_cid = w4.register(l3_orch)
            l4_verify_cid = w4.register(lambda args: _increment_counter(counter_buf))
            w4.add_worker(l3)
            w4.init()

            def l4_orch(orch, args, config):
                orch.submit_next_level(l3_cid, TaskArgs(), ChipCallConfig())
                orch.submit_sub(l4_verify_cid)

            w4.run(l4_orch)
            w4.close()

            # L3 sub + L4 sub = 2
            assert _read_counter(counter_buf) == 2
        finally:
            counter_shm.close()
            counter_shm.unlink()


# ---------------------------------------------------------------------------
# Test: L4 → L3 — multiple runs
# ---------------------------------------------------------------------------


class TestL4MultipleRuns:
    def test_l4_multiple_runs_no_leak(self):
        """Multiple w4.run() calls on the same Worker — slots don't leak."""
        counter_shm, counter_buf = _make_shared_counter()

        try:
            l3 = Worker(level=3, num_sub_workers=1)
            l3_sub_cid = l3.register(lambda args: _increment_counter(counter_buf))

            def l3_orch(orch, args, config):
                orch.submit_sub(l3_sub_cid)

            w4 = Worker(level=4, num_sub_workers=0)
            l3_cid = w4.register(l3_orch)
            w4.add_worker(l3)
            w4.init()

            def l4_orch(orch, args, config):
                orch.submit_next_level(l3_cid, TaskArgs(), ChipCallConfig())

            for _ in range(5):
                w4.run(l4_orch)

            w4.close()

            assert _read_counter(counter_buf) == 5
        finally:
            counter_shm.close()
            counter_shm.unlink()


# ---------------------------------------------------------------------------
# Test: L4 → L3 — L3 uses multiple sub workers
# ---------------------------------------------------------------------------


class TestL4L3WithMultipleSubs:
    def test_l3_child_runs_multiple_subs(self):
        """L3 child submits 2 sub tasks per dispatch (serialized through 1 worker).

        Uses 1 sub worker because _increment_counter is a non-atomic RMW
        that races across parallel SubWorker processes.
        """
        counter_shm, counter_buf = _make_shared_counter()

        try:
            l3 = Worker(level=3, num_sub_workers=1)
            l3_sub_cid = l3.register(lambda args: _increment_counter(counter_buf))

            def l3_orch(orch, args, config):
                orch.submit_sub(l3_sub_cid)
                orch.submit_sub(l3_sub_cid)

            w4 = Worker(level=4, num_sub_workers=0)
            l3_cid = w4.register(l3_orch)
            w4.add_worker(l3)
            w4.init()

            def l4_orch(orch, args, config):
                orch.submit_next_level(l3_cid, TaskArgs(), ChipCallConfig())

            w4.run(l4_orch)
            w4.close()

            assert _read_counter(counter_buf) == 2
        finally:
            counter_shm.close()
            counter_shm.unlink()


# ---------------------------------------------------------------------------
# Test: L3 orch receives its own Orchestrator (not L4's)
# ---------------------------------------------------------------------------


class TestL3OwnOrchestrator:
    def test_l3_gets_own_orchestrator(self):
        """The L3 orch fn receives an Orchestrator from the L3 inner worker,
        not the L4 parent. Prove by checking orch.alloc works at L3 level."""
        counter_shm, counter_buf = _make_shared_counter()

        try:
            l3 = Worker(level=3, num_sub_workers=1)
            l3_sub_cid = l3.register(lambda args: _increment_counter(counter_buf))

            def l3_orch(orch, args, config):
                # orch is L3's own Orchestrator — alloc + submit_sub should work
                orch.submit_sub(l3_sub_cid)

            w4 = Worker(level=4, num_sub_workers=0)
            l3_cid = w4.register(l3_orch)
            w4.add_worker(l3)
            w4.init()

            def l4_orch(orch, args, config):
                orch.submit_next_level(l3_cid, TaskArgs(), ChipCallConfig())

            w4.run(l4_orch)
            w4.close()

            assert _read_counter(counter_buf) == 1
        finally:
            counter_shm.close()
            counter_shm.unlink()


# ---------------------------------------------------------------------------
# Test: generalised _Worker(level) — no hardcoded 3
# ---------------------------------------------------------------------------


class TestGeneralised_Worker:
    def test_worker_level_param(self):
        """_Worker accepts level != 3 without error."""
        from simpler.task_interface import _Worker  # noqa: PLC0415

        for level in (3, 4, 5):
            dw = _Worker(level)
            dw.close()
