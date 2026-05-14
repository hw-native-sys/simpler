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
import threading
from multiprocessing.shared_memory import SharedMemory

import pytest
from _task_interface import MAX_REGISTERED_CALLABLE_IDS  # pyright: ignore[reportMissingImports]
from simpler.task_interface import ChipCallable, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker, _make_callable_shm_name

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

    def test_register_python_fn_after_init_raises(self):
        # Post-init register of a non-ChipCallable (lambda / sub fn) is
        # rejected because Python callables cannot cross the fork boundary.
        # ChipCallable is the only post-init target — see the next test.
        hw = Worker(level=3, num_sub_workers=0)
        hw.init()
        with pytest.raises(NotImplementedError, match="only ChipCallable is supported post-init"):
            hw.register(lambda args: None)
        hw.close()

    def test_register_chip_callable_after_init_no_chips_succeeds(self):
        # With no chip children (device_ids unset), the dynamic register
        # broadcast loop iterates over zero mailboxes — exercises the
        # facade path (lock, _post_init_register, shm create/unlink,
        # registry insertion) end-to-end without needing an NPU.
        hw = Worker(level=3, num_sub_workers=0)
        hw.init()
        try:
            callable_obj = ChipCallable.build(signature=[], func_name="x", binary=b"\x00", children=[])
            cid = hw.register(callable_obj)
            assert isinstance(cid, int)
            assert cid >= 0
        finally:
            hw.close()

    def test_register_chip_callable_at_cid_overflow_raises(self):
        # cid budget is enforced under the new dynamic-register path too:
        # pre-fill registry with lambdas pre-init, init, then attempt one
        # post-init ChipCallable register and observe the existing
        # MAX_REGISTERED_CALLABLE_IDS RuntimeError.
        hw = Worker(level=3, num_sub_workers=0)
        try:
            for _ in range(MAX_REGISTERED_CALLABLE_IDS):
                hw.register(lambda args: None)
            hw.init()
            callable_obj = ChipCallable.build(signature=[], func_name="x", binary=b"\x00", children=[])
            with pytest.raises(RuntimeError, match="MAX_REGISTERED_CALLABLE_IDS"):
                hw.register(callable_obj)
        finally:
            hw.close()

    def test_register_chip_callable_during_run_raises(self):
        # The quiescent-state guard: a register call that races a Worker.run
        # must surface a clear RuntimeError instead of deadlocking on a
        # CONTROL_REQUEST that overlaps an in-flight TASK_READY. The orch fn
        # signals once it is in flight (which is also after Worker.run() has
        # bumped _orch_in_flight under _register_lock); the side thread
        # waits for that signal before attempting register.
        hw = Worker(level=3, num_sub_workers=1)
        hw.register(lambda args: None)
        hw.init()
        try:
            run_started = threading.Event()
            release = threading.Event()
            register_err: list[BaseException] = []

            def orch(orch_handle, _args, _cfg):
                run_started.set()
                release.wait(timeout=5.0)

            def try_register():
                # Wait until orch fn is in flight (guarantees _orch_in_flight
                # was bumped to 1 before our register attempts to acquire
                # _register_lock).
                assert run_started.wait(timeout=5.0), "orch never started"
                try:
                    callable_obj = ChipCallable.build(signature=[], func_name="x", binary=b"\x00", children=[])
                    hw.register(callable_obj)
                except BaseException as e:  # noqa: BLE001
                    register_err.append(e)
                finally:
                    release.set()

            t = threading.Thread(target=try_register)
            t.start()
            try:
                hw.run(orch)
            finally:
                t.join(timeout=5.0)
            assert register_err, "expected register to raise during run"
            assert isinstance(register_err[0], RuntimeError)
            assert "Worker.run() is executing" in str(register_err[0])
        finally:
            hw.close()

    def test_register_chip_callable_broadcast_failure_rolls_back(self, monkeypatch):
        # Forces _post_init_register down the CTRL_REGISTER broadcast path
        # (hierarchical_started=True + a fake chip mailbox), then makes the
        # broadcast helper raise. Verifies the parent rolls back cleanly:
        # registry entry popped, shm unlinked, RuntimeError surfaced. This
        # is the only test that exercises the failure path end-to-end on
        # the facade — the ST suite cannot stage a deterministic
        # prepare-time fault in the chip child.
        import simpler.worker  # noqa: PLC0415

        hw = Worker(level=3, num_sub_workers=0)
        hw.register(lambda args: None)
        hw.init()
        try:
            # Splice a placeholder mailbox so the broadcast loop iterates
            # once; the monkeypatched helper never reads it.
            fake_mailbox = SharedMemory(create=True, size=4096)
            hw._chip_shms.append(fake_mailbox)
            hw._hierarchical_started = True
            try:
                captured_shm_names: list[str] = []
                real_make = simpler.worker._make_callable_shm_name

                def capture_shm_name(pid, cid):
                    name = real_make(pid, cid)
                    captured_shm_names.append(name)
                    return name

                monkeypatch.setattr(simpler.worker, "_make_callable_shm_name", capture_shm_name)

                def fail_broadcast(self, worker_id, cid, shm_name):
                    raise RuntimeError(f"chip {worker_id}: register cid={cid} chip=0: simulated")

                monkeypatch.setattr(Worker, "_chip_control_register", fail_broadcast)

                callable_obj = ChipCallable.build(signature=[], func_name="x", binary=b"\x00", children=[])
                registry_size_before = len(hw._callable_registry)
                with pytest.raises(RuntimeError, match=r"failed; 1 of 1 chips reported errors"):
                    hw.register(callable_obj)
                # cid was popped — registry size is unchanged.
                assert len(hw._callable_registry) == registry_size_before
                # The shm the parent staged for the broadcast was unlinked.
                assert len(captured_shm_names) == 1
                with pytest.raises(FileNotFoundError):
                    SharedMemory(name=captured_shm_names[0])
            finally:
                hw._chip_shms.pop()
                fake_mailbox.close()
                fake_mailbox.unlink()
        finally:
            hw.close()

    def test_register_overflow_raises(self):
        # The AICPU side reserves a fixed-size orch_so_table_[MAX_REGISTERED_CALLABLE_IDS];
        # Worker.register must surface the bound at register-time, not later when
        # DeviceRunner::register_prepared_callable rejects the cid.
        hw = Worker(level=3, num_sub_workers=0)
        try:
            for _ in range(MAX_REGISTERED_CALLABLE_IDS):
                hw.register(lambda args: None)
            with pytest.raises(RuntimeError, match="MAX_REGISTERED_CALLABLE_IDS"):
                hw.register(lambda args: None)
        finally:
            # init() was never called; close() is still safe (idempotent
            # against an uninitialised Worker).
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


# ---------------------------------------------------------------------------
# Test: _CTRL_REGISTER shm name helper
# ---------------------------------------------------------------------------


class TestCallableShmName:
    def test_make_callable_shm_name_unique(self):
        # Consecutive calls with the same (pid, cid) must produce different
        # names — the monotonic counter is the disambiguator that lets
        # multiple registers coexist without POSIX shm name collisions.
        a = _make_callable_shm_name(1234, 7)
        b = _make_callable_shm_name(1234, 7)
        assert a != b
        # Both must fit in the on-wire field with a NUL terminator (32 B).
        assert len(a.encode("utf-8")) + 1 <= 32
        assert len(b.encode("utf-8")) + 1 <= 32
        assert a.startswith("simpler-cb-1234-7-")
        assert b.startswith("simpler-cb-1234-7-")
