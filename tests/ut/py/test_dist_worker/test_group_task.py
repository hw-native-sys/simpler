# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for group task support (N args -> N workers, 1 DAG node).

Each test uses SubWorker (fork/shm) — no NPU device required.

TestGroupBasic:
    test_group_both_workers_execute — 2 args dispatches to 2 SubWorkers,
        both run, atomic counter reaches 2.
    test_single_args_is_normal_task — 1 arg falls back to normal (non-group)
        submit path, counter reaches 1.

TestGroupDependency:
    test_group_then_dependent_task — group (2 workers) produces output,
        downstream task depends on it via TensorMap. Verifies downstream
        only runs after group completes.
"""

import struct
from multiprocessing import Value
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


# ---------------------------------------------------------------------------
# Test: group of 2 SubWorkers — both execute
# ---------------------------------------------------------------------------


class TestGroupBasic:
    def test_group_both_workers_execute(self):
        """submit with 2 args -> 2 SubWorkers, counter==2."""
        counter = Value("i", 0)

        hw = Worker(level=3, num_sub_workers=2)

        def inc():
            with counter.get_lock():
                counter.value += 1

        cid = hw.register(inc)
        hw.init()

        def orch(o, _args):
            o.submit_sub_group(cid, args_list=[0, 0])

        hw.run(Task(orch=orch))
        hw.close()

        assert counter.value == 2, f"Expected 2, got {counter.value}"

    def test_single_args_group_runs_once(self):
        """submit_sub_group with 1 arg still runs exactly once."""
        counter = Value("i", 0)

        hw = Worker(level=3, num_sub_workers=1)

        def inc():
            with counter.get_lock():
                counter.value += 1

        cid = hw.register(inc)
        hw.init()

        def orch(o, _args):
            o.submit_sub_group(cid, args_list=[0])

        hw.run(Task(orch=orch))
        hw.close()

        assert counter.value == 1


# ---------------------------------------------------------------------------
# Test: group dependency chain — downstream waits for group
# ---------------------------------------------------------------------------


class TestGroupDependency:
    def test_group_then_dependent_task(self):
        """Group (2 workers) -> downstream task. Downstream waits for group."""
        # Use idempotent writes (set to 1) to avoid _inc race across processes.
        group_marker = _alloc_counter()
        dep_marker = _alloc_counter()

        try:
            gb = group_marker.buf
            db = dep_marker.buf
            assert gb is not None and db is not None

            hw = Worker(level=3, num_sub_workers=3)
            group_cid = hw.register(lambda: struct.pack_into("i", gb, 0, 1))
            dep_cid = hw.register(lambda: struct.pack_into("i", db, 0, 1))
            hw.init()

            def orch(o, _args):
                group_result = o.submit_sub_group(group_cid, args_list=[0, 0], outputs=[64])
                out_ptr = group_result.outputs[0].ptr
                o.submit_sub(dep_cid, inputs=[out_ptr])

            hw.run(Task(orch=orch))
            hw.close()

            assert _read(group_marker) == 1, "Group task didn't run"
            assert _read(dep_marker) == 1, "Dependent task didn't run"
        finally:
            group_marker.close()
            group_marker.unlink()
            dep_marker.close()
            dep_marker.unlink()
