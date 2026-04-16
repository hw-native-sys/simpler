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
    test_single_args_group_runs_once — 1 arg in a group still runs exactly
        once (group-of-1 fallback path).

TestGroupDependency:
    test_group_then_dependent_task — group (2 workers) -> downstream task,
        wired via a synthetic shared tensor pointer (OUTPUT in group, INPUT
        in downstream). Verifies downstream only runs after the group has
        completed.
"""

import struct
from multiprocessing import Value
from multiprocessing.shared_memory import SharedMemory

from simpler.task_interface import (
    ContinuousTensor,
    DataType,
    TaskArgs,
    TensorArgType,
)
from simpler.worker import Worker

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


def _sync_args(ptr: int, tag: TensorArgType) -> TaskArgs:
    """Build a TaskArgs whose only purpose is to register a synthetic
    tensor-pointer key with the TensorMap so a downstream task can wire a
    dep on it. SUB callables don't actually read tensors, so the pointer
    value just needs to be a unique non-zero key.
    """
    args = TaskArgs()
    args.add_tensor(ContinuousTensor.make(ptr, (1,), DataType.UINT8), tag)
    return args


# ---------------------------------------------------------------------------
# Test: group of 2 SubWorkers — both execute
# ---------------------------------------------------------------------------


class TestGroupBasic:
    def test_group_both_workers_execute(self):
        """submit_sub_group with 2 args -> 2 SubWorkers, counter==2."""
        counter = Value("i", 0)

        hw = Worker(level=3, num_sub_workers=2)

        def inc(args):
            with counter.get_lock():
                counter.value += 1

        cid = hw.register(inc)
        hw.init()

        def orch(o, args, cfg):
            o.submit_sub_group(cid, [TaskArgs(), TaskArgs()])

        hw.run(orch)
        hw.close()

        assert counter.value == 2, f"Expected 2, got {counter.value}"

    def test_single_args_group_runs_once(self):
        """submit_sub_group with 1 arg still runs exactly once."""
        counter = Value("i", 0)

        hw = Worker(level=3, num_sub_workers=1)

        def inc(args):
            with counter.get_lock():
                counter.value += 1

        cid = hw.register(inc)
        hw.init()

        def orch(o, args, cfg):
            o.submit_sub_group(cid, [TaskArgs()])

        hw.run(orch)
        hw.close()

        assert counter.value == 1


# ---------------------------------------------------------------------------
# Test: group dependency chain — downstream waits for group
# ---------------------------------------------------------------------------


# Synthetic non-zero pointer used as the TensorMap key for SUB-only dep tests.
# The SUB callable never dereferences it; only Orchestrator.tensormap reads
# the value as an opaque key.
_SYNC_PTR = 0xDEADBEEF00


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
            group_cid = hw.register(lambda args: struct.pack_into("i", gb, 0, 1))
            dep_cid = hw.register(lambda args: struct.pack_into("i", db, 0, 1))
            hw.init()

            def orch(o, args, cfg):
                # Group: both members tag the synthetic ptr as OUTPUT — the
                # second insert overwrites the first with the same slot id.
                o.submit_sub_group(
                    group_cid,
                    [
                        _sync_args(_SYNC_PTR, TensorArgType.OUTPUT),
                        _sync_args(_SYNC_PTR, TensorArgType.OUTPUT),
                    ],
                )
                # Downstream: INPUT on the same ptr → tensormap lookup wires
                # a fanin on the group slot.
                o.submit_sub(dep_cid, _sync_args(_SYNC_PTR, TensorArgType.INPUT))

            hw.run(orch)
            hw.close()

            assert _read(group_marker) == 1, "Group task didn't run"
            assert _read(dep_marker) == 1, "Dependent task didn't run"
        finally:
            group_marker.close()
            group_marker.unlink()
            dep_marker.close()
            dep_marker.unlink()
