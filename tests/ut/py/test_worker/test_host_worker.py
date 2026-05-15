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
        # With no chip children (device_ids unset), the C++ broadcast is a
        # no-op (next_level_threads_ is empty) — exercises the facade path
        # (registry lock, cid allocation, broadcast call, return) end-to-end
        # without needing an NPU.
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

    def test_unregister_unknown_cid_raises(self):
        # Symmetric to register: unregister must fail loud if the caller
        # confuses cid for an unrelated integer or unregisters twice.
        hw = Worker(level=3, num_sub_workers=0)
        hw.init()
        try:
            with pytest.raises(KeyError, match="cid=999 not registered"):
                hw.unregister(999)
        finally:
            hw.close()

    def test_unregister_chip_callable_after_init_no_chips_succeeds(self):
        # With zero chip mailboxes the C++ broadcast is a no-op, so the
        # facade path (registry lock, broadcast, registry pop) is exercised
        # end-to-end without an NPU. Also verifies cid reuse — unregistering
        # frees the slot and the next register reuses the same cid via
        # `_allocate_cid` (smallest-unused-integer).
        hw = Worker(level=3, num_sub_workers=0)
        hw.init()
        try:
            callable_obj = ChipCallable.build(signature=[], func_name="x", binary=b"\x00", children=[])
            cid_a = hw.register(callable_obj)
            assert cid_a in hw._callable_registry
            hw.unregister(cid_a)
            assert cid_a not in hw._callable_registry
            cid_b = hw.register(callable_obj)
            assert cid_b == cid_a, "smallest-unused-cid policy should reuse the freed slot"
        finally:
            hw.close()

    def test_unregister_middle_cid_reuses_hole(self):
        # `_allocate_cid` must fill the smallest hole, not append at
        # len(registry). The bug it guards against: register 0/1/2,
        # unregister 1, next register would silently overwrite the
        # existing cid=2 under a `len(registry)` policy.
        hw = Worker(level=3, num_sub_workers=0)
        hw.init()
        try:
            cb = ChipCallable.build(signature=[], func_name="x", binary=b"\x00", children=[])
            cid0 = hw.register(cb)
            cid1 = hw.register(cb)
            cid2 = hw.register(cb)
            assert (cid0, cid1, cid2) == (0, 1, 2)
            hw.unregister(cid1)
            cid_reused = hw.register(cb)
            assert cid_reused == 1, "hole at cid=1 should be reused before appending"
            # cid=2 entry must still be the original callable, not silently overwritten.
            assert hw._callable_registry[cid2] is cb
            # Next register fills cid=3 since 0..2 are all occupied.
            cid_next = hw.register(cb)
            assert cid_next == 3
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
# Test: _CTRL_REGISTER self-heal on cid reuse
# ---------------------------------------------------------------------------


class TestChipMainLoopRegisterSelfHeal:
    """Direct white-box tests on the _run_chip_main_loop self-heal branch.

    Drives the loop in a background thread with a MagicMock ChipWorker and
    a real shm mailbox. Each test simulates the parent by writing a control
    command, waiting for the child to publish _CONTROL_DONE, resetting the
    state to _IDLE, and finally writing _SHUTDOWN. This exercises the
    actual state-machine code path including the self-heal block; injecting
    `prepared = {cid}` directly is not possible because the set is a local
    in the loop function — the seed comes from a real prior CTRL_REGISTER.
    """

    @staticmethod
    def _build_mailbox():
        from simpler.task_interface import MAILBOX_SIZE  # noqa: PLC0415
        from simpler.worker import _IDLE, _OFF_STATE, _buffer_field_addr, _mailbox_store_i32  # noqa: PLC0415

        shm = SharedMemory(create=True, size=MAILBOX_SIZE)
        buf = shm.buf
        assert buf is not None
        # Loop reads the state field via a raw address (atomic_int32 in C++),
        # so we hand it the absolute address and let it cast back inside.
        state_addr = _buffer_field_addr(buf, _OFF_STATE)
        _mailbox_store_i32(state_addr, _IDLE)
        # `mailbox_addr` is only consumed by the TASK_READY branch, which we
        # never reach in these tests; passing 0 keeps the harness lean.
        return shm, buf, state_addr

    @staticmethod
    def _send_ctrl_register(buf, state_addr, cid: int, shm_name: str):
        """Stage a CTRL_REGISTER request and flip the state to CONTROL_REQUEST."""
        from simpler.worker import (  # noqa: PLC0415
            _CONTROL_REQUEST,
            _CTRL_OFF_ARG0,
            _CTRL_REGISTER,
            _CTRL_SHM_NAME_BYTES,
            _OFF_ARGS,
            _OFF_CALLABLE,
            _mailbox_store_i32,
        )

        struct.pack_into("Q", buf, _OFF_CALLABLE, _CTRL_REGISTER)
        struct.pack_into("Q", buf, _CTRL_OFF_ARG0, cid)
        encoded = shm_name.encode("utf-8")
        assert len(encoded) + 1 <= _CTRL_SHM_NAME_BYTES
        buf[_OFF_ARGS : _OFF_ARGS + len(encoded)] = encoded
        buf[_OFF_ARGS + len(encoded) : _OFF_ARGS + _CTRL_SHM_NAME_BYTES] = b"\x00" * (
            _CTRL_SHM_NAME_BYTES - len(encoded)
        )
        _mailbox_store_i32(state_addr, _CONTROL_REQUEST)

    @staticmethod
    def _wait_for_done_and_reset(buf, state_addr, timeout: float = 5.0):
        """Block until the loop publishes _CONTROL_DONE, then read the error
        code and reset the mailbox to _IDLE so the next round can start."""
        import time  # noqa: PLC0415

        from simpler.worker import (  # noqa: PLC0415
            _CONTROL_DONE,
            _IDLE,
            _OFF_ERROR,
            _mailbox_load_i32,
            _mailbox_store_i32,
        )

        deadline = time.monotonic() + timeout
        while _mailbox_load_i32(state_addr) != _CONTROL_DONE:
            if time.monotonic() > deadline:
                raise TimeoutError("loop did not publish CONTROL_DONE")
            time.sleep(0.001)
        err_code = struct.unpack_from("i", buf, _OFF_ERROR)[0]
        _mailbox_store_i32(state_addr, _IDLE)
        return err_code

    @staticmethod
    def _shutdown(state_addr):
        from simpler.worker import _SHUTDOWN, _mailbox_store_i32  # noqa: PLC0415

        _mailbox_store_i32(state_addr, _SHUTDOWN)

    @staticmethod
    def _spawn_loop(cw, buf, state_addr):
        from simpler.worker import _run_chip_main_loop  # noqa: PLC0415

        t = threading.Thread(
            target=_run_chip_main_loop,
            args=(cw, buf, 0, state_addr, 0, {}),
            daemon=True,
        )
        t.start()
        return t

    def test_no_self_heal_when_prepared_clean(self):
        # First CTRL_REGISTER on a fresh loop: `prepared` starts empty, so the
        # self-heal branch must be skipped — no extra unregister_callable call.
        # Locks in the zero-cost happy path.
        from unittest.mock import MagicMock  # noqa: PLC0415

        cw = MagicMock()
        cw.unregister_callable = MagicMock()
        cw._impl.prepare_callable_from_blob = MagicMock()

        payload_shm = SharedMemory(create=True, size=64)
        shm, buf, state_addr = self._build_mailbox()
        try:
            t = self._spawn_loop(cw, buf, state_addr)
            try:
                self._send_ctrl_register(buf, state_addr, cid=7, shm_name=payload_shm.name)
                err = self._wait_for_done_and_reset(buf, state_addr)
                assert err == 0
                # Critical assertion: no self-heal cleanup on a fresh slot.
                assert cw.unregister_callable.call_count == 0
                assert cw._impl.prepare_callable_from_blob.call_count == 1
            finally:
                self._shutdown(state_addr)
                t.join(timeout=2.0)
        finally:
            shm.close()
            shm.unlink()
            payload_shm.close()
            payload_shm.unlink()

    def test_self_heal_triggers_on_repeat_register(self):
        # Second CTRL_REGISTER on the same cid: after the first round
        # `prepared` holds 7, so the loop must self-heal — call
        # unregister_callable to clear host-side residue before re-preparing.
        # This is the scenario a best-effort unregister failure leaves behind.
        from unittest.mock import MagicMock  # noqa: PLC0415

        cw = MagicMock()
        cw.unregister_callable = MagicMock()
        cw._impl.prepare_callable_from_blob = MagicMock()

        payload_shm = SharedMemory(create=True, size=64)
        shm, buf, state_addr = self._build_mailbox()
        try:
            t = self._spawn_loop(cw, buf, state_addr)
            try:
                # Round 1: seed `prepared = {7}`.
                self._send_ctrl_register(buf, state_addr, cid=7, shm_name=payload_shm.name)
                assert self._wait_for_done_and_reset(buf, state_addr) == 0
                assert cw.unregister_callable.call_count == 0
                # Round 2: cid=7 already in `prepared` -> self-heal fires.
                self._send_ctrl_register(buf, state_addr, cid=7, shm_name=payload_shm.name)
                assert self._wait_for_done_and_reset(buf, state_addr) == 0
                # Self-heal called unregister_callable exactly once, then
                # prepare_callable_from_blob ran on the cleaned slot.
                assert cw.unregister_callable.call_count == 1
                cw.unregister_callable.assert_called_with(7)
                assert cw._impl.prepare_callable_from_blob.call_count == 2
            finally:
                self._shutdown(state_addr)
                t.join(timeout=2.0)
        finally:
            shm.close()
            shm.unlink()
            payload_shm.close()
            payload_shm.unlink()

    def test_self_heal_tolerates_unregister_exception(self):
        # The self-heal try/except must swallow exceptions from
        # unregister_callable so a flaky cleanup does not block the new
        # registration. The follow-on prepare_callable_from_blob still runs
        # and the mailbox publishes a clean (code=0) CONTROL_DONE.
        from unittest.mock import MagicMock  # noqa: PLC0415

        cw = MagicMock()
        # First call: succeeds (seed phase has no self-heal invocation).
        # Second call (self-heal): raises — must be swallowed.
        cw.unregister_callable = MagicMock(side_effect=[RuntimeError("simulated")])
        cw._impl.prepare_callable_from_blob = MagicMock()

        payload_shm = SharedMemory(create=True, size=64)
        shm, buf, state_addr = self._build_mailbox()
        try:
            t = self._spawn_loop(cw, buf, state_addr)
            try:
                # Round 1: no self-heal, prepared seeded with {7}.
                self._send_ctrl_register(buf, state_addr, cid=7, shm_name=payload_shm.name)
                assert self._wait_for_done_and_reset(buf, state_addr) == 0
                # Round 2: self-heal fires; unregister_callable raises but is
                # caught; prepare_callable_from_blob still runs.
                self._send_ctrl_register(buf, state_addr, cid=7, shm_name=payload_shm.name)
                err = self._wait_for_done_and_reset(buf, state_addr)
                assert err == 0, "self-heal exception leaked into mailbox error code"
                assert cw.unregister_callable.call_count == 1
                assert cw._impl.prepare_callable_from_blob.call_count == 2
            finally:
                self._shutdown(state_addr)
                t.join(timeout=2.0)
        finally:
            shm.close()
            shm.unlink()
            payload_shm.close()
            payload_shm.unlink()
