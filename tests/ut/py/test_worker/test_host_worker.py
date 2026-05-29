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
from simpler.callable_identity import CallableHandle
from simpler.task_interface import (
    MAILBOX_SIZE,
    ChipCallable,
    DataType,
    TaskArgs,
    TensorArgType,
    WorkerType,
    _Worker,
)
from simpler.worker import (
    _CONTROL_REQUEST,
    _CTRL_PY_REGISTER,
    _CTRL_PY_UNREGISTER,
    _IDLE,
    _OFF_STATE,
    Worker,
    _buffer_field_addr,
    _mailbox_addr,
    _mailbox_load_i32,
    _mailbox_store_i32,
    _pack_py_callable_payload,
)

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


def _add_counter(buf, delta: int) -> None:
    v = struct.unpack_from("i", buf, 0)[0]
    struct.pack_into("i", buf, 0, v + delta)


def _set_flag(buf, offset: int, value: int) -> None:
    struct.pack_into("i", buf, offset, value)


def _get_flag(buf, offset: int) -> int:
    return struct.unpack_from("i", buf, offset)[0]


def _roundtrip_py_callable_payload(target):
    from simpler.worker import _load_py_callable_from_shm, _pack_py_callable_payload  # noqa: PLC0415

    payload = _pack_py_callable_payload(target)
    shm = SharedMemory(create=True, size=len(payload))
    try:
        assert shm.buf is not None
        shm.buf[: len(payload)] = payload
        return _load_py_callable_from_shm(shm.name)
    finally:
        shm.close()
        shm.unlink()


def _test_handle(cid: int, digest: bytes, *, kind: str = "PYTHON_SERIALIZED", namespace: str = "LOCAL_PYTHON"):
    return CallableHandle(
        hashid="sha256:" + digest.hex(),
        kind=kind,
        target_namespace=namespace,
        _slot_id=cid,
        _digest=digest,
        _handle_id=10_000 + cid,
    )


def _unique_py_callable(index: int):
    def fn(args, _index=index):
        return _index

    return fn


def _unique_chip_callable(index: int):
    return ChipCallable.build(signature=[], func_name=f"x{index}", binary=bytes([index & 0xFF]), children=[])


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

    def test_l2_rejects_python_callable(self):
        hw = Worker(level=2, device_id=0, platform="a2a3sim", runtime="tensormap_and_ringbuffer")
        with pytest.raises(TypeError, match="level 2 only supports ChipCallable"):
            hw.register(lambda args: None)

    def test_register_python_fn_after_init_before_start_succeeds(self):
        # init() allocates mailboxes but does not fork children. Python
        # callables registered in this window still land in the startup
        # snapshot consumed by the first run().
        hw = Worker(level=3, num_sub_workers=0)
        hw.init()
        try:
            cid = hw.register(lambda args: None)
            assert cid in hw._callable_registry
        finally:
            hw.close()

    def test_register_python_fn_after_init_before_start_does_not_broadcast(self):
        class BroadcastTrap:
            def broadcast_control_all(self, *args, **kwargs):
                raise AssertionError("pre-start Python register must not broadcast")

        hw = Worker(level=3, num_sub_workers=1)
        hw.init()
        real_worker = hw._worker
        try:
            hw._worker = BroadcastTrap()
            cid = hw.register(lambda args: None)
            assert cid in hw._callable_registry
        finally:
            hw._worker = real_worker
            hw.close()

    def test_register_python_fn_after_start_no_python_children_raises(self):
        hw = Worker(level=3, num_sub_workers=0)
        hw.init()
        try:
            hw.run(lambda orch, args, cfg: None)
            with pytest.raises(RuntimeError, match="no Python-capable child"):
                hw.register(lambda args: None)
        finally:
            hw.close()

    def test_register_waits_for_first_startup_then_uses_post_start_path(self):
        hw = Worker(level=3, num_sub_workers=1)
        hw.init()
        try:
            with hw._hierarchical_start_cv:
                hw._hierarchical_start_state = "starting"

            observed = {}

            def fake_post_start_register(target):
                observed["target"] = target
                observed["state"] = hw._hierarchical_start_state
                observed["hierarchical_started"] = hw._hierarchical_started
                return 7

            hw._post_start_register_python = fake_post_start_register
            result: list[int] = []
            errors: list[BaseException] = []
            wait_entered = threading.Event()
            original_wait = hw._hierarchical_start_cv.wait

            def wait_with_signal(timeout=None):
                wait_entered.set()
                return original_wait(timeout)

            hw._hierarchical_start_cv.wait = wait_with_signal

            def do_register():
                try:
                    result.append(hw.register(lambda args: None))
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            t = threading.Thread(target=do_register)
            t.start()
            assert wait_entered.wait(timeout=2.0)
            with hw._hierarchical_start_cv:
                hw._hierarchical_started = True
                hw._hierarchical_start_state = "started"
                hw._hierarchical_start_cv.notify_all()
            t.join(timeout=2.0)

            assert not t.is_alive()
            assert errors == []
            assert result == [7]
            assert observed["state"] == "started"
            assert observed["hierarchical_started"] is True
        finally:
            if "original_wait" in locals():
                hw._hierarchical_start_cv.wait = original_wait
            hw.close()

    def test_register_blocks_startup_snapshot_from_not_started_window(self):
        hw = Worker(level=3, num_sub_workers=0)
        hw.init()

        real_registry_lock = hw._registry_lock
        register_waiting = threading.Event()
        release_register = threading.Event()
        startup_snapshot_attempted = threading.Event()
        result: list[int] = []
        errors: list[BaseException] = []

        class BlockingRegistryLock:
            def __enter__(self):
                thread_name = threading.current_thread().name
                if thread_name == "register-thread":
                    register_waiting.set()
                    if not release_register.wait(timeout=2.0):
                        raise TimeoutError("test timed out waiting to release register")
                elif thread_name == "startup-thread":
                    startup_snapshot_attempted.set()
                return real_registry_lock.__enter__()

            def __exit__(self, exc_type, exc, tb):
                return real_registry_lock.__exit__(exc_type, exc, tb)

            def locked(self):
                return real_registry_lock.locked()

        hw._registry_lock = BlockingRegistryLock()

        def do_register():
            try:
                result.append(hw.register(lambda args: None))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def do_startup():
            try:
                hw._start_hierarchical()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        register_thread = threading.Thread(target=do_register, name="register-thread")
        startup_thread = threading.Thread(target=do_startup, name="startup-thread")
        try:
            register_thread.start()
            assert register_waiting.wait(timeout=2.0)

            startup_thread.start()
            assert not startup_snapshot_attempted.wait(timeout=0.2)

            release_register.set()
            register_thread.join(timeout=2.0)
            startup_thread.join(timeout=2.0)

            assert not register_thread.is_alive()
            assert not startup_thread.is_alive()
            assert errors == []
            assert result == [0]
            assert startup_snapshot_attempted.is_set()
            assert hw._hierarchical_start_state == "started"
        finally:
            release_register.set()
            register_thread.join(timeout=2.0)
            startup_thread.join(timeout=2.0)
            hw._registry_lock = real_registry_lock
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
            handle = hw.register(callable_obj)
            assert isinstance(handle, CallableHandle)
            assert handle._slot_id >= 0
        finally:
            hw.close()

    def test_register_chip_callable_at_cid_overflow_raises(self):
        # cid budget is enforced under the new dynamic-register path too:
        # pre-fill registry with lambdas pre-init, init, then attempt one
        # post-init ChipCallable register and observe the existing
        # MAX_REGISTERED_CALLABLE_IDS RuntimeError.
        hw = Worker(level=3, num_sub_workers=0)
        try:
            for i in range(MAX_REGISTERED_CALLABLE_IDS):
                hw.register(_unique_py_callable(i))
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
            assert cid_b._slot_id == cid_a._slot_id, "smallest-unused-cid policy should reuse the freed slot"
        finally:
            hw.close()

    def test_register_chip_callable_broadcast_runs_without_registry_lock(self):
        hw = Worker(level=3, num_sub_workers=0)
        hw._initialized = True
        hw._hierarchical_started = True
        callable_obj = ChipCallable.build(signature=[], func_name="x", binary=b"\x00", children=[])
        observed = {}

        def fake_post_init_register(cid, target, digest):
            observed["cid"] = cid
            observed["target"] = target
            observed["digest"] = digest
            observed["locked"] = hw._registry_lock.locked()

        hw._post_init_register = fake_post_init_register

        cid = hw.register(callable_obj)

        assert observed == {"cid": cid._slot_id, "target": callable_obj, "digest": cid.digest, "locked": False}
        assert hw._callable_registry[cid] is callable_obj

    def test_register_at_broadcast_runs_without_registry_lock(self):
        hw = Worker(level=3, num_sub_workers=0)
        hw._initialized = True
        callable_obj = ChipCallable.build(signature=[], func_name="x", binary=b"\x00", children=[])
        observed = {}

        def fake_post_init_register(cid, target, digest):
            observed["cid"] = cid
            observed["target"] = target
            observed["digest"] = digest
            observed["locked"] = hw._registry_lock.locked()

        hw._post_init_register = fake_post_init_register

        hw._register_at(7, callable_obj)

        assert observed == {
            "cid": 7,
            "target": callable_obj,
            "digest": next(iter(hw._identity_registry)),
            "locked": False,
        }
        assert hw._callable_registry[7] is callable_obj

    def test_python_control_broadcast_passes_default_timeout(self):
        from simpler.worker import _CTRL_PY_UNREGISTER, _PY_CONTROL_TIMEOUT_S  # noqa: PLC0415

        class FakeControlWorker:
            def __init__(self):
                self.calls = []

            def broadcast_control_all(self, worker_type, sub_cmd, cid, payload, digest=None, timeout_s=None):
                self.calls.append((worker_type, sub_cmd, cid, payload, digest, timeout_s))
                return []

        fake = FakeControlWorker()
        hw = Worker(level=3, num_sub_workers=1)
        hw._worker = fake

        errors = hw._broadcast_py_control([WorkerType.SUB], _CTRL_PY_UNREGISTER, 3, strict=False)

        assert errors == []
        assert fake.calls == [(WorkerType.SUB, _CTRL_PY_UNREGISTER, 3, None, None, _PY_CONTROL_TIMEOUT_S)]

    def test_cloudpickle_payload_roundtrip_supported_callable_shapes(self):
        class AddValue:
            def __init__(self, value):
                self.value = value

            def __call__(self, arg):
                return arg + self.value

        scale = 3

        def nested(arg):
            return arg * scale

        cases = [
            (lambda arg: arg + 1, 4, 5),
            (nested, 4, 12),
            (AddValue(7), 4, 11),
        ]
        for target, arg, expected in cases:
            loaded = _roundtrip_py_callable_payload(target)
            assert callable(loaded)
            assert loaded(arg) == expected

    def test_python_unregister_child_failure_warns_pops_and_allows_reuse(self, capsys):
        from simpler.worker import _CTRL_PY_REGISTER, _CTRL_PY_UNREGISTER  # noqa: PLC0415

        hw = Worker(level=3, num_sub_workers=1)
        cid = hw.register(lambda args: None)
        hw._initialized = True
        hw._hierarchical_started = True
        calls = []

        def fake_broadcast(worker_types, sub_cmd, broadcast_cid, *, digest=None, payload=None, strict):
            calls.append((list(worker_types), sub_cmd, broadcast_cid, digest, strict))
            if sub_cmd == _CTRL_PY_UNREGISTER:
                return ["SUB[0]: injected unregister failure"]
            if sub_cmd == _CTRL_PY_REGISTER:
                return []
            raise AssertionError(f"unexpected sub_cmd={sub_cmd}")

        hw._broadcast_py_control = fake_broadcast

        hw.unregister(cid)

        captured = capsys.readouterr()
        assert "Python children reported errors" in captured.err
        assert "injected unregister failure" in captured.err
        assert cid not in hw._callable_registry
        assert cid not in hw._pending_unregister_cids

        reused = hw.register(lambda args: None)
        assert reused._slot_id == cid._slot_id
        assert calls[0] == ([WorkerType.SUB], _CTRL_PY_UNREGISTER, cid._slot_id, cid.digest, False)
        assert calls[1][0:3] == ([WorkerType.SUB], _CTRL_PY_REGISTER, cid._slot_id)
        assert calls[1][4] is True

    def test_pending_unregister_cid_is_not_reused_until_broadcast_returns(self):
        from simpler.worker import _CTRL_PY_REGISTER, _CTRL_PY_UNREGISTER  # noqa: PLC0415

        hw = Worker(level=3, num_sub_workers=1)
        cid = hw.register(lambda args: None)
        hw._initialized = True
        hw._hierarchical_started = True

        broadcast_started = threading.Event()
        release_broadcast = threading.Event()
        errors: list[BaseException] = []

        def fake_broadcast(worker_types, sub_cmd, broadcast_cid, *, digest=None, payload=None, strict):
            if sub_cmd == _CTRL_PY_UNREGISTER:
                broadcast_started.set()
                assert release_broadcast.wait(timeout=2.0)
            elif sub_cmd == _CTRL_PY_REGISTER:
                return []
            else:
                raise AssertionError(f"unexpected sub_cmd={sub_cmd}")
            return []

        hw._broadcast_py_control = fake_broadcast

        def do_unregister():
            try:
                hw.unregister(cid)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        t = threading.Thread(target=do_unregister)
        t.start()
        assert broadcast_started.wait(timeout=2.0)

        cid_during_unregister = hw.register(lambda args: None)
        assert cid_during_unregister != cid
        assert cid in hw._pending_unregister_cids

        release_broadcast.set()
        t.join(timeout=2.0)
        assert not t.is_alive()
        assert errors == []

        cid_after_unregister = hw.register(lambda args: None)
        assert cid_after_unregister._slot_id == cid._slot_id

    def test_register_python_sub_callable_after_start_succeeds(self):
        counter_shm, counter_buf = _make_shared_counter()
        try:
            hw = Worker(level=3, num_sub_workers=1)
            bootstrap_cid = hw.register(lambda args: None)
            hw.init()

            def bootstrap(orch, args, cfg):
                orch.submit_sub(bootstrap_cid)

            hw.run(bootstrap)
            counter_name = counter_shm.name

            def dynamic_sub(args):
                shm = SharedMemory(name=counter_name)
                try:
                    _increment_counter(shm.buf)
                finally:
                    shm.close()

            dynamic_cid = hw.register(dynamic_sub)

            def run_dynamic(orch, args, cfg):
                orch.submit_sub(dynamic_cid)

            hw.run(run_dynamic)
            hw.close()

            assert _read_counter(counter_buf) == 1
        finally:
            counter_shm.close()
            counter_shm.unlink()

    def test_post_start_python_register_waits_for_active_sub_mailbox(self):
        import time  # noqa: PLC0415

        control_shm = SharedMemory(create=True, size=8)
        counter_shm, counter_buf = _make_shared_counter()
        hw = Worker(level=3, num_sub_workers=1)
        run_errors: list[BaseException] = []
        register_errors: list[BaseException] = []
        dynamic_cids: list[int] = []
        run_thread = None
        register_thread = None
        try:
            assert control_shm.buf is not None
            _set_flag(control_shm.buf, 0, 0)  # started
            _set_flag(control_shm.buf, 4, 0)  # release
            control_name = control_shm.name
            counter_name = counter_shm.name

            def blocking_sub(args):
                import time as child_time  # noqa: PLC0415

                shm = SharedMemory(name=control_name)
                try:
                    _set_flag(shm.buf, 0, 1)
                    while _get_flag(shm.buf, 4) == 0:
                        child_time.sleep(0.001)
                finally:
                    shm.close()

            blocking_cid = hw.register(blocking_sub)
            hw.init()

            def run_blocking():
                try:
                    hw.run(lambda orch, args, cfg: orch.submit_sub(blocking_cid))
                except BaseException as exc:  # noqa: BLE001
                    run_errors.append(exc)

            run_thread = threading.Thread(target=run_blocking)
            run_thread.start()

            deadline = time.monotonic() + 2.0
            while _get_flag(control_shm.buf, 0) == 0 and time.monotonic() < deadline:
                time.sleep(0.001)
            assert _get_flag(control_shm.buf, 0) == 1

            def dynamic_sub(args):
                shm = SharedMemory(name=counter_name)
                try:
                    _increment_counter(shm.buf)
                finally:
                    shm.close()

            def do_register():
                try:
                    dynamic_cids.append(hw.register(dynamic_sub))
                except BaseException as exc:  # noqa: BLE001
                    register_errors.append(exc)

            register_thread = threading.Thread(target=do_register)
            register_thread.start()
            register_thread.join(timeout=0.05)
            assert register_thread.is_alive()

            _set_flag(control_shm.buf, 4, 1)
            run_thread.join(timeout=2.0)
            register_thread.join(timeout=2.0)

            assert not run_thread.is_alive()
            assert not register_thread.is_alive()
            assert run_errors == []
            assert register_errors == []
            assert len(dynamic_cids) == 1

            hw.run(lambda orch, args, cfg: orch.submit_sub(dynamic_cids[0]))
            assert _read_counter(counter_buf) == 1
        finally:
            if control_shm.buf is not None:
                _set_flag(control_shm.buf, 4, 1)
            if run_thread is not None:
                run_thread.join(timeout=2.0)
            if register_thread is not None:
                register_thread.join(timeout=2.0)
            hw.close()
            control_shm.close()
            control_shm.unlink()
            counter_shm.close()
            counter_shm.unlink()

    def test_post_start_unregister_pre_start_python_callable_removes_child_entry(self):
        counter_shm, counter_buf = _make_shared_counter()
        try:
            hw = Worker(level=3, num_sub_workers=1)
            cid = hw.register(lambda args: _increment_counter(counter_buf))
            hw.init()

            hw.run(lambda orch, args, cfg: orch.submit_sub(cid))
            assert _read_counter(counter_buf) == 1

            hw.unregister(cid)
            assert cid not in hw._callable_registry
            with pytest.raises(RuntimeError, match="not registered"):
                hw.run(lambda orch, args, cfg: orch.submit_sub(cid))

            counter_name = counter_shm.name

            def replacement(args):
                shm = SharedMemory(name=counter_name)
                try:
                    _add_counter(shm.buf, 10)
                finally:
                    shm.close()

            reused = hw.register(replacement)
            assert reused._slot_id == cid._slot_id
            hw.run(lambda orch, args, cfg: orch.submit_sub(reused))
            hw.close()

            assert _read_counter(counter_buf) == 11
        finally:
            counter_shm.close()
            counter_shm.unlink()

    def test_post_start_unregister_post_start_python_callable_removes_child_entry(self):
        counter_shm, counter_buf = _make_shared_counter()
        try:
            hw = Worker(level=3, num_sub_workers=1)
            bootstrap_cid = hw.register(lambda args: None)
            hw.init()
            hw.run(lambda orch, args, cfg: orch.submit_sub(bootstrap_cid))

            counter_name = counter_shm.name

            def dynamic(args):
                shm = SharedMemory(name=counter_name)
                try:
                    _increment_counter(shm.buf)
                finally:
                    shm.close()

            cid = hw.register(dynamic)
            hw.run(lambda orch, args, cfg: orch.submit_sub(cid))
            assert _read_counter(counter_buf) == 1

            hw.unregister(cid)
            assert cid not in hw._callable_registry
            with pytest.raises(RuntimeError, match="not registered"):
                hw.run(lambda orch, args, cfg: orch.submit_sub(cid))

            reused = hw.register(dynamic)
            assert reused._slot_id == cid._slot_id
            hw.run(lambda orch, args, cfg: orch.submit_sub(reused))
            hw.close()

            assert _read_counter(counter_buf) == 2
        finally:
            counter_shm.close()
            counter_shm.unlink()

    def test_post_start_dynamic_python_callable_execute_failure_propagates(self):
        hw = Worker(level=3, num_sub_workers=1)
        bootstrap_cid = hw.register(lambda args: None)
        hw.init()
        try:
            hw.run(lambda orch, args, cfg: orch.submit_sub(bootstrap_cid))

            def boom(args):
                raise RuntimeError("dynamic callable boom")

            cid = hw.register(boom)
            with pytest.raises(RuntimeError, match="dynamic callable boom"):
                hw.run(lambda orch, args, cfg: orch.submit_sub(cid))
        finally:
            hw.close()

    def test_broadcast_control_all_accepts_memoryview_payload(self):
        counter_shm, counter_buf = _make_shared_counter()
        try:
            hw = Worker(level=3, num_sub_workers=1)
            bootstrap_cid = hw.register(lambda args: None)
            hw.init()

            def bootstrap(orch, args, cfg):
                orch.submit_sub(bootstrap_cid)

            hw.run(bootstrap)
            counter_name = counter_shm.name

            def dynamic_sub(args):
                shm = SharedMemory(name=counter_name)
                try:
                    _increment_counter(shm.buf)
                finally:
                    shm.close()

            cid = 5
            worker_impl = hw._worker
            assert worker_impl is not None
            digest = bytes([5]) * 32
            results = worker_impl.broadcast_control_all(
                WorkerType.SUB,
                _CTRL_PY_REGISTER,
                cid,
                memoryview(_pack_py_callable_payload(dynamic_sub)),
                digest,
            )
            assert len(results) == 1
            assert results[0].ok
            handle = _test_handle(cid, digest)

            def run_dynamic(orch, args, cfg):
                orch.submit_sub(handle)

            hw.run(run_dynamic)
            hw.close()

            assert _read_counter(counter_buf) == 1
        finally:
            counter_shm.close()
            counter_shm.unlink()

    def test_broadcast_control_all_reports_malformed_payload(self):
        hw = Worker(level=3, num_sub_workers=1)
        bootstrap_cid = hw.register(lambda args: None)
        hw.init()
        try:
            hw.run(lambda orch, args, cfg: orch.submit_sub(bootstrap_cid))
            worker_impl = hw._worker
            assert worker_impl is not None
            results = worker_impl.broadcast_control_all(WorkerType.SUB, _CTRL_PY_REGISTER, 5, b"bad")
            assert len(results) == 1
            assert not results[0].ok
            assert "payload" in results[0].error_message
        finally:
            hw.close()

    def test_broadcast_control_all_empty_payload_raises_before_fanout(self):
        hw = Worker(level=3, num_sub_workers=1)
        bootstrap_cid = hw.register(lambda args: None)
        hw.init()
        try:
            hw.run(lambda orch, args, cfg: orch.submit_sub(bootstrap_cid))
            worker_impl = hw._worker
            assert worker_impl is not None
            with pytest.raises(RuntimeError, match="payload pointer and size"):
                worker_impl.broadcast_control_all(WorkerType.SUB, _CTRL_PY_REGISTER, 5, b"")
        finally:
            hw.close()

    def test_broadcast_control_all_timeout_reports_failed_child(self):
        shm = SharedMemory(create=True, size=MAILBOX_SIZE)
        dw = _Worker(3)
        try:
            assert shm.buf is not None
            _mailbox_store_i32(_buffer_field_addr(shm.buf, _OFF_STATE), _IDLE)
            dw.add_sub_worker(_mailbox_addr(shm))
            dw.init()
            results = dw.broadcast_control_all(
                WorkerType.SUB,
                _CTRL_PY_UNREGISTER,
                0,
                None,
                timeout_s=0.001,
            )
            assert len(results) == 1
            assert not results[0].ok
            assert "timed out" in results[0].error_message
        finally:
            dw.close()
            shm.close()
            shm.unlink()

    def test_broadcast_control_all_selected_pool_routing(self):
        def make_mailbox():
            shm = SharedMemory(create=True, size=MAILBOX_SIZE)
            assert shm.buf is not None
            _mailbox_store_i32(_buffer_field_addr(shm.buf, _OFF_STATE), _IDLE)
            return shm

        for selected_type, selected_kind in (
            (WorkerType.SUB, "SUB"),
            (WorkerType.NEXT_LEVEL, "NEXT_LEVEL"),
        ):
            sub_shm = make_mailbox()
            next_shm = make_mailbox()
            dw = _Worker(3)
            try:
                dw.add_sub_worker(_mailbox_addr(sub_shm))
                dw.add_next_level_worker(_mailbox_addr(next_shm))
                dw.init()
                results = dw.broadcast_control_all(
                    selected_type,
                    _CTRL_PY_UNREGISTER,
                    0,
                    None,
                    timeout_s=0.001,
                )
                assert len(results) == 1
                assert results[0].worker_type == selected_kind
                sub_state = _mailbox_load_i32(_buffer_field_addr(sub_shm.buf, _OFF_STATE))
                next_state = _mailbox_load_i32(_buffer_field_addr(next_shm.buf, _OFF_STATE))
                if selected_type == WorkerType.SUB:
                    assert sub_state == _CONTROL_REQUEST
                    assert next_state == _IDLE
                else:
                    assert sub_state == _IDLE
                    assert next_state == _CONTROL_REQUEST
            finally:
                dw.close()
                sub_shm.close()
                sub_shm.unlink()
                next_shm.close()
                next_shm.unlink()

    def test_broadcast_control_all_result_shape_for_register_and_unregister(self):
        hw = Worker(level=3, num_sub_workers=1)
        bootstrap_cid = hw.register(lambda args: None)
        hw.init()
        try:
            hw.run(lambda orch, args, cfg: orch.submit_sub(bootstrap_cid))
            worker_impl = hw._worker
            assert worker_impl is not None
            register_results = worker_impl.broadcast_control_all(
                WorkerType.SUB,
                _CTRL_PY_REGISTER,
                5,
                b"bad",
            )
            unregister_results = worker_impl.broadcast_control_all(
                WorkerType.SUB,
                _CTRL_PY_UNREGISTER,
                bootstrap_cid._slot_id,
                None,
                bootstrap_cid.digest,
            )

            for result in (register_results[0], unregister_results[0]):
                assert isinstance(result.worker_type, str)
                assert isinstance(result.worker_index, int)
                assert isinstance(result.ok, bool)
                assert isinstance(result.error_message, str)
            assert not register_results[0].ok
            assert unregister_results[0].ok
        finally:
            hw.close()

    def test_nonserializable_dynamic_python_callable_does_not_consume_cid(self):
        lock = threading.Lock()
        hw = Worker(level=3, num_sub_workers=1)
        bootstrap_cid = hw.register(lambda args: None)
        hw.init()
        try:
            hw.run(lambda orch, args, cfg: orch.submit_sub(bootstrap_cid))
            before = dict(hw._callable_registry)

            def captures_lock(args):
                lock.acquire(False)

            with pytest.raises(TypeError, match="lock"):
                hw.register(captures_lock)
            assert hw._callable_registry == before
        finally:
            hw.close()

    def test_chip_register_reuse_clears_seen_python_cid_before_binary_register(self):
        from simpler.worker import _CTRL_PY_UNREGISTER  # noqa: PLC0415

        calls = []

        class FakeWorker:
            def broadcast_register_all(self, cid, blob_ptr, blob_size, digest):
                calls.append(("binary_register", cid, blob_size, digest))

        hw = Worker(level=3, num_sub_workers=1)
        hw._initialized = True
        hw._hierarchical_started = True
        hw._worker = FakeWorker()
        hw._py_callable_cids_seen.add(0)

        def fake_py_control(worker_types, sub_cmd, cid, *, digest=None, payload=None, strict):
            calls.append(("py_clear", list(worker_types), sub_cmd, cid, digest, strict))
            return []

        hw._broadcast_py_control = fake_py_control
        callable_obj = ChipCallable.build(signature=[], func_name="x", binary=b"\x00", children=[])

        cid = hw.register(callable_obj)

        assert cid == 0
        assert calls[0][0:4] == ("py_clear", [WorkerType.SUB], _CTRL_PY_UNREGISTER, 0)
        assert calls[0][5] is True
        assert calls[1][0] == "binary_register"
        assert 0 not in hw._py_callable_cids_seen

        hw._callable_registry.pop(0)
        hw._identity_registry.pop(cid.digest, None)
        calls.clear()

        cid = hw.register(ChipCallable.build(signature=[], func_name="y", binary=b"\x00", children=[]))

        assert cid == 0
        assert len(calls) == 1
        assert calls[0][0:2] == ("binary_register", 0)

    def test_chip_register_reuse_fails_before_binary_register_when_python_clear_fails(self):
        calls = []

        class FakeWorker:
            def broadcast_register_all(self, cid, blob_ptr, blob_size, digest):
                calls.append(("binary_register", cid))

        hw = Worker(level=3, num_sub_workers=1)
        hw._initialized = True
        hw._hierarchical_started = True
        hw._worker = FakeWorker()
        hw._py_callable_cids_seen.add(0)

        def fake_py_control(worker_types, sub_cmd, cid, *, digest=None, payload=None, strict):
            calls.append(("py_clear", cid, strict))
            raise RuntimeError("clear failed")

        hw._broadcast_py_control = fake_py_control
        callable_obj = ChipCallable.build(signature=[], func_name="x", binary=b"\x00", children=[])

        with pytest.raises(RuntimeError, match="clear failed"):
            hw.register(callable_obj)

        assert calls == [("py_clear", 0, True)]
        assert hw._callable_registry == {}

    def test_unregister_middle_cid_reuses_hole(self):
        # `_allocate_cid` must fill the smallest hole, not append at
        # len(registry). The bug it guards against: register 0/1/2,
        # unregister 1, next register would silently overwrite the
        # existing cid=2 under a `len(registry)` policy.
        hw = Worker(level=3, num_sub_workers=0)
        hw.init()
        try:
            cb0 = _unique_chip_callable(0)
            cb1 = _unique_chip_callable(1)
            cb2 = _unique_chip_callable(2)
            cb3 = _unique_chip_callable(3)
            cid0 = hw.register(cb0)
            cid1 = hw.register(cb1)
            cid2 = hw.register(cb2)
            assert (cid0, cid1, cid2) == (0, 1, 2)
            hw.unregister(cid1)
            cid_reused = hw.register(cb3)
            assert cid_reused == 1, "hole at cid=1 should be reused before appending"
            # cid=2 entry must still be the original callable, not silently overwritten.
            assert hw._callable_registry[cid2] is cb2
            # Next register fills cid=3 since 0..2 are all occupied.
            cid_next = hw.register(_unique_chip_callable(4))
            assert cid_next == 3
        finally:
            hw.close()

    def test_register_overflow_raises(self):
        # The AICPU side reserves a fixed-size orch_so_table_[MAX_REGISTERED_CALLABLE_IDS];
        # Worker.register must surface the bound at register-time, not later when
        # DeviceRunner::register_prepared_callable rejects the cid.
        hw = Worker(level=3, num_sub_workers=0)
        try:
            for i in range(MAX_REGISTERED_CALLABLE_IDS):
                hw.register(_unique_py_callable(i))
            with pytest.raises(RuntimeError, match="MAX_REGISTERED_CALLABLE_IDS"):
                hw.register(_unique_py_callable(MAX_REGISTERED_CALLABLE_IDS))
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
    def _send_ctrl_register(buf, state_addr, cid: int, shm_name: str, digest: bytes = b"\x07" * 32):
        """Stage a CTRL_REGISTER request and flip the state to CONTROL_REQUEST."""
        from simpler.worker import (  # noqa: PLC0415
            _CONTROL_REQUEST,
            _CTRL_OFF_ARG0,
            _CTRL_REGISTER,
            _CTRL_SHM_NAME_BYTES,
            _OFF_ARGS,
            _OFF_CALLABLE,
            _OFF_CONTROL_CALLABLE_HASH,
            _mailbox_store_i32,
        )

        struct.pack_into("Q", buf, _OFF_CALLABLE, _CTRL_REGISTER)
        struct.pack_into("Q", buf, _CTRL_OFF_ARG0, cid)
        assert len(digest) == 32
        buf[_OFF_CONTROL_CALLABLE_HASH : _OFF_CONTROL_CALLABLE_HASH + len(digest)] = digest
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
            args=(cw, buf, 0, state_addr, 0, {}, {}),
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
