# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for Worker (Python L3 wrapper over _Worker).

Tests use SubWorker (fork/shm) as the only worker type — no NPU device required.
Each test verifies a distinct aspect of the L3 scheduling pipeline.
"""

import struct
from multiprocessing.shared_memory import SharedMemory

import pytest
from simpler.task_interface import DataType, TaskArgs, TensorArgType
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
            hw.register(lambda args: None)
        # close() called by __exit__, no exception

    def test_register_after_init_raises(self):
        hw = Worker(level=3, num_sub_workers=0)
        hw.init()
        with pytest.raises(RuntimeError, match="before init"):
            hw.register(lambda args: None)
        hw.close()


# ---------------------------------------------------------------------------
# Test: single independent SUB task executes and completes
# ---------------------------------------------------------------------------


class TestSingleSubTask:
    def test_sub_task_executes(self):
        counter_shm, counter_buf = _make_shared_counter()

        try:
            hw = Worker(level=3, num_sub_workers=1)
            cid = hw.register(lambda args: _increment_counter(counter_buf))
            hw.init()

            def orch(o, args, cfg):
                o.submit_sub(cid)

            hw.run(orch)
            hw.close()

            assert _read_counter(counter_buf) == 1
        finally:
            counter_shm.close()
            counter_shm.unlink()

    def test_sub_task_runs_multiple_times(self):
        counter_shm, counter_buf = _make_shared_counter()

        try:
            hw = Worker(level=3, num_sub_workers=1)
            cid = hw.register(lambda args: _increment_counter(counter_buf))
            hw.init()

            def orch(o, args, cfg):
                for _ in range(3):
                    o.submit_sub(cid)

            hw.run(orch)
            hw.close()

            assert _read_counter(counter_buf) == 3
        finally:
            counter_shm.close()
            counter_shm.unlink()


# ---------------------------------------------------------------------------
# Test: multiple SUB workers execute in parallel
# ---------------------------------------------------------------------------


class TestParallelSubWorkers:
    # test_parallel_wall_time was dropped: wall-clock timing assertions on
    # shared CI runners (macOS in particular) are too flaky — scheduling
    # jitter routinely pushes observed elapsed past a 0.9-factor-of-serial
    # threshold. Parallel SubWorker execution is still covered via
    # test_many_tasks_two_workers_all_complete (all tasks run) and the
    # scheduler's dispatch tests in tests/ut/cpp.
    pass


# ---------------------------------------------------------------------------
# Test: SubmitResult shape — just {slot_id}; no outputs[] anymore.
# Output buffers are user-provided tensors tagged OUTPUT in the TaskArgs.
# ---------------------------------------------------------------------------


class TestSubmitResult:
    def test_submit_returns_slot_id_only(self):
        counter_shm, counter_buf = _make_shared_counter()
        try:
            hw = Worker(level=3, num_sub_workers=1)
            cid = hw.register(lambda args: _increment_counter(counter_buf))
            hw.init()

            captured = []

            def orch(o, args, cfg):
                result = o.submit_sub(cid)
                captured.append(result)

            hw.run(orch)
            hw.close()

            assert len(captured) == 1
            r = captured[0]
            assert r.task_slot >= 0
            # Note: SubmitResult no longer carries outputs[]; downstream consumers
            # reference output tensors by their own data pointers (which the
            # Orchestrator finds in the TensorMap).
            assert _read_counter(counter_buf) == 1
        finally:
            counter_shm.close()
            counter_shm.unlink()


# ---------------------------------------------------------------------------
# Test: scope management (owned by Worker.run; user doesn't see scope_begin/end)
# ---------------------------------------------------------------------------


class TestScope:
    def test_scope_managed_by_run(self):
        counter_shm, counter_buf = _make_shared_counter()

        try:
            hw = Worker(level=3, num_sub_workers=1)
            cid = hw.register(lambda args: _increment_counter(counter_buf))
            hw.init()

            def orch(o, args, cfg):
                o.submit_sub(cid)

            hw.run(orch)
            hw.close()

            assert _read_counter(counter_buf) == 1
        finally:
            counter_shm.close()
            counter_shm.unlink()

    def test_user_nested_scope_runs_to_completion(self):
        """User opens a nested scope with ``with orch.scope():``; all tasks run."""
        counter_shm, counter_buf = _make_shared_counter()
        try:
            # Use one sub worker so the increments serialize — _increment_counter
            # is a non-atomic RMW and races across parallel SubWorker processes.
            hw = Worker(level=3, num_sub_workers=1)
            cid = hw.register(lambda args: _increment_counter(counter_buf))
            hw.init()

            def orch(o, args, cfg):
                with o.scope():
                    o.submit_sub(cid)
                    o.submit_sub(cid)
                o.submit_sub(cid)  # back on outer-scope ring

            hw.run(orch)
            hw.close()

            assert _read_counter(counter_buf) == 3
        finally:
            counter_shm.close()
            counter_shm.unlink()

    def test_user_nested_scope_binding_is_exposed(self):
        """The scope context manager and raw scope_begin / scope_end are bound."""
        from simpler.task_interface import _Orchestrator  # noqa: PLC0415

        # Binding carries the new accessors.
        assert hasattr(_Orchestrator, "scope_begin")
        assert hasattr(_Orchestrator, "scope_end")

        hw = Worker(level=3, num_sub_workers=1)
        hw.register(lambda args: None)
        hw.init()

        def orch(o, args, cfg):
            # Raw calls — match L2's pto2_scope_begin / pto2_scope_end.
            o.scope_begin()
            o.scope_end()
            # Context-manager form.
            with o.scope():
                pass
            # Mixed with submits.
            with o.scope():
                inner = o.alloc((32,), DataType.FLOAT32)
                assert inner.data != 0

        hw.run(orch)
        hw.close()

    def test_user_nested_scope_three_deep(self):
        """Three levels of nested scopes drain cleanly (no leaked refs)."""
        counter_shm, counter_buf = _make_shared_counter()
        try:
            hw = Worker(level=3, num_sub_workers=1)
            cid = hw.register(lambda args: _increment_counter(counter_buf))
            hw.init()

            def orch(o, args, cfg):
                o.submit_sub(cid)  # outer scope (ring 0)
                with o.scope():
                    o.submit_sub(cid)  # ring 1
                    with o.scope():
                        o.submit_sub(cid)  # ring 2
                        with o.scope():
                            o.submit_sub(cid)  # ring 3
                            with o.scope():
                                o.submit_sub(cid)  # clamps to ring 3

            hw.run(orch)
            hw.close()
            assert _read_counter(counter_buf) == 5
        finally:
            counter_shm.close()
            counter_shm.unlink()


# ---------------------------------------------------------------------------
# Test: orch.alloc — runtime-managed intermediate buffer lifecycle
# ---------------------------------------------------------------------------


class TestOrchAlloc:
    def test_alloc_returns_valid_tensor(self):
        """alloc returns a ContinuousTensor whose data ptr is non-zero and writeable."""
        captured = []

        hw = Worker(level=3, num_sub_workers=1)
        cid = hw.register(lambda args: None)  # sub callable doesn't actually read
        hw.init()

        def orch(o, args, cfg):
            inter = o.alloc((64,), DataType.FLOAT32)
            captured.append((inter.data, inter.ndims, inter.shapes[0]))

            # Tag as OUTPUT in some submit so the synthetic alloc slot has a
            # downstream consumer (otherwise scope_end consumes alone — still fine).
            sub_args = TaskArgs()
            sub_args.add_tensor(inter, TensorArgType.INPUT)
            o.submit_sub(cid, sub_args)

        hw.run(orch)
        hw.close()

        assert len(captured) == 1
        data_ptr, ndims, shape0 = captured[0]
        assert data_ptr != 0
        assert ndims == 1
        assert shape0 == 64

    def test_alloc_dep_wires_via_tensormap(self):
        """INOUT producer -> alloc'd ptr -> INPUT consumer wires the dep."""
        marker_shm, marker_buf = _make_shared_counter()

        try:
            hw = Worker(level=3, num_sub_workers=2)
            producer_cid = hw.register(lambda args: _increment_counter(marker_buf))
            consumer_cid = hw.register(lambda args: _increment_counter(marker_buf))
            hw.init()

            def orch(o, args, cfg):
                inter = o.alloc((128,), DataType.FLOAT32)

                # Producer writes into the alloc'd slab and must depend on
                # the alloc-slot (the creator) so the slab is not reclaimed
                # while the producer is still writing. That lifetime link
                # goes through INOUT — matching L2, only INPUT and INOUT
                # do TensorMap.lookup. Plain OUTPUT / OUTPUT_EXISTING are
                # pure inserts and would leave no dep on the alloc slot.
                p_args = TaskArgs()
                p_args.add_tensor(inter, TensorArgType.INOUT)
                o.submit_sub(producer_cid, p_args)

                # Consumer tags inter as INPUT — tensormap.lookup finds the
                # producer slot, dep wired automatically.
                c_args = TaskArgs()
                c_args.add_tensor(inter, TensorArgType.INPUT)
                o.submit_sub(consumer_cid, c_args)

            hw.run(orch)
            hw.close()

            # Both ran (we don't assert order strictly — relies on dep enforcement
            # which we'd need a write-then-read assert to verify; counter==2 at
            # least confirms both fired and no deadlock).
            assert _read_counter(marker_buf) == 2
        finally:
            marker_shm.close()
            marker_shm.unlink()

    def test_alloc_unused_freed_at_scope_end(self):
        """alloc that's never tagged still consumes cleanly via scope ref."""
        hw = Worker(level=3, num_sub_workers=0)
        hw.init()

        def orch(o, args, cfg):
            o.alloc((16,), DataType.UINT8)
            o.alloc((32,), DataType.FLOAT32)
            # No submits using these — synthetic slots' fanout_total = 1 (scope only)
            # scope_end's release_ref alone hits the threshold (sim self + scope = 2 = total + 1).

        hw.run(orch)
        hw.close()
        # If munmap leaks or the slot doesn't reach CONSUMED, drain hangs above.

    def test_alloc_across_runs_does_not_leak(self):
        """Repeated runs each alloc + use; slots must be released between runs."""
        marker_shm, marker_buf = _make_shared_counter()

        try:
            hw = Worker(level=3, num_sub_workers=1)
            cid = hw.register(lambda args: _increment_counter(marker_buf))
            hw.init()

            def orch(o, args, cfg):
                inter = o.alloc((64,), DataType.FLOAT32)
                args = TaskArgs()
                args.add_tensor(inter, TensorArgType.INPUT)
                o.submit_sub(cid, args)

            for _ in range(8):
                hw.run(orch)

            hw.close()
            assert _read_counter(marker_buf) == 8
        finally:
            marker_shm.close()
            marker_shm.unlink()


# ---------------------------------------------------------------------------
# Test: sub callable receives args blob correctly
# ---------------------------------------------------------------------------


class TestSubCallableArgs:
    def test_sub_callable_receives_tensor_metadata(self):
        """Sub callable receives TaskArgs with correct tensor count and shape."""
        from simpler.task_interface import ContinuousTensor  # noqa: PLC0415

        result_shm, result_buf = _make_shared_counter()
        try:
            hw = Worker(level=3, num_sub_workers=1)

            def check_args(args):
                # Verify args decoded correctly: 1 tensor, shape (4,), FLOAT32
                if args.tensor_count() == 1 and args.scalar_count() == 0:
                    t = args.tensor(0)
                    if t.ndims == 1 and t.shapes[0] == 4:
                        _increment_counter(result_buf)

            cid = hw.register(check_args)
            hw.init()

            # Use a synthetic non-zero pointer — sub callable only checks metadata,
            # doesn't dereference the pointer.
            ct = ContinuousTensor.make(0xCAFE0000, (4,), DataType.FLOAT32)

            def orch(o, args, cfg):
                sub_args = TaskArgs()
                sub_args.add_tensor(ct, TensorArgType.INPUT)
                o.submit_sub(cid, sub_args)

            hw.run(orch)
            hw.close()

            assert _read_counter(result_buf) == 1, "Sub callable did not receive correct args"
        finally:
            result_shm.close()
            result_shm.unlink()

    def test_sub_callable_receives_scalar(self):
        """Sub callable receives TaskArgs with a scalar value."""
        result_shm, result_buf = _make_shared_counter()
        try:
            hw = Worker(level=3, num_sub_workers=1)

            def check_scalar(args):
                if args.scalar_count() == 1 and args.scalar(0) == 42:
                    _increment_counter(result_buf)

            cid = hw.register(check_scalar)
            hw.init()

            def orch(o, args, cfg):
                sub_args = TaskArgs()
                sub_args.add_scalar(42)
                o.submit_sub(cid, sub_args)

            hw.run(orch)
            hw.close()

            assert _read_counter(result_buf) == 1, "Sub callable did not receive correct scalar"
        finally:
            result_shm.close()
            result_shm.unlink()

    def test_sub_callable_empty_args(self):
        """Sub callable receives empty TaskArgs when no args submitted."""
        result_shm, result_buf = _make_shared_counter()
        try:
            hw = Worker(level=3, num_sub_workers=1)

            def check_empty(args):
                if args.tensor_count() == 0 and args.scalar_count() == 0:
                    _increment_counter(result_buf)

            cid = hw.register(check_empty)
            hw.init()

            def orch(o, args, cfg):
                o.submit_sub(cid)

            hw.run(orch)
            hw.close()

            assert _read_counter(result_buf) == 1, "Sub callable did not receive empty args"
        finally:
            result_shm.close()
            result_shm.unlink()
