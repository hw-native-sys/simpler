# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Direct L2 submission validates the same Buffer identities and argument grants as L3."""

import os
import signal
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
from simpler.buffer import AccessMode, Buffer, mint_owner_instance_id, wrap_device_malloc, wrap_fork_inherited
from simpler.task_interface import CallConfig, DataType, TaskArgs, Tensor, TensorArgType
from simpler.worker import Worker, _Lifecycle


@pytest.fixture
def l2():
    worker = Worker(level=2)
    calls = []
    frees = []
    impl = SimpleNamespace(_submit_chip_run_direct=lambda cid, args, cfg: calls.append(args) or object())
    worker._chip_worker = SimpleNamespace(_impl=impl, free=frees.append)
    worker._lifecycle = _Lifecycle.READY
    yield worker, calls, frees
    worker._chip_runs.clear()
    worker._chip_run_touched_identities.clear()
    worker._close_chip_import_registry()
    worker._lifecycle = _Lifecycle.CLOSED


def register_buffer(worker, buffer_id=1, *, access=AccessMode.READWRITE, owner_worker_id=0, base=0x4000):
    buffer = wrap_device_malloc(
        base, 64, worker._owner_instance_id, buffer_id, access=access, owner_worker_id=owner_worker_id
    )
    with worker._child_prov_lock:
        worker._record_device_alloc(buffer)
    return buffer


def arguments(buffer, *, tag=TensorArgType.INPUT):
    args = TaskArgs()
    args.add_tensor(buffer.tensor((4,), DataType.FLOAT32), tag)
    return args


@pytest.mark.parametrize(
    "case", ["revoked", "cached_revoked", "generation", "wrong_chip", "foreign_owner", "changed_extent"]
)
def test_invalid_device_identity_is_rejected_before_mapping_or_launch(l2, monkeypatch, case):
    worker, calls, _ = l2
    buffer = register_buffer(worker, owner_worker_id=1 if case == "wrong_chip" else 0)
    if case == "cached_revoked":
        worker._materialize_l2_args(arguments(buffer))
    if case in ("revoked", "cached_revoked"):
        with worker._child_prov_lock:
            worker._drop_device_alloc(buffer.identity)
    elif case == "generation":
        buffer = wrap_device_malloc(0x4000, 64, worker._owner_instance_id, 1, generation=2)
    elif case == "foreign_owner":
        buffer = wrap_device_malloc(0x4000, 64, mint_owner_instance_id(), 1)
    elif case == "changed_extent":
        buffer = replace(buffer, nbytes=128)
    mapped = []
    original = worker._materialize_l2_args
    monkeypatch.setattr(worker, "_materialize_l2_args", lambda args: mapped.append(True) or original(args))
    with pytest.raises(ValueError, match="not a live|does not match"):
        worker._submit_l2_locked(3, arguments(buffer), CallConfig())
    assert not mapped and not calls
    assert not worker._chip_run_touched_identities


def test_mutated_output_tag_is_rechecked_before_any_mapping(l2, monkeypatch):
    worker, calls, _ = l2
    args = arguments(wrap_fork_inherited(1, 64, worker._owner_instance_id, 99, access=AccessMode.READ))
    read_only = register_buffer(worker, access=AccessMode.READ)
    args.add_tensor(read_only.tensor((4,), DataType.FLOAT32))
    args.set_tag(1, TensorArgType.OUTPUT_EXISTING)
    mapped = []
    original = worker._materialize_l2_args
    monkeypatch.setattr(worker, "_materialize_l2_args", lambda args: mapped.append(True) or original(args))
    with pytest.raises(ValueError, match="does not grant"):
        worker._submit_l2_locked(3, args, CallConfig())
    assert not mapped and not calls


def test_mutating_the_callers_args_does_not_change_the_accepted_binding(l2, monkeypatch):
    worker, calls, _ = l2
    first = register_buffer(worker)
    second = register_buffer(worker, 2, base=0x8000)
    args = arguments(first)
    original = worker._materialize_l2_args

    def mutate_then_materialize(accepted):
        args.clear()
        args.add_tensor(second.tensor((4,), DataType.FLOAT32))
        return original(accepted)

    monkeypatch.setattr(worker, "_materialize_l2_args", mutate_then_materialize)
    handle = worker._submit_l2_locked(3, args, CallConfig())
    assert calls[0].tensor(0).data == first.base
    assert worker._chip_run_touched_identities[handle._run_id] == {first.identity}


def test_free_rechecks_after_a_submission_wins_the_reservation(l2, monkeypatch):
    worker, calls, frees = l2
    buffer = register_buffer(worker)
    checked = threading.Event()
    resume = threading.Event()
    errors = []
    original = worker._refuse_free_while_in_flight

    def pause_after_fast_check(handle):
        original(handle)
        checked.set()
        assert resume.wait(5)

    monkeypatch.setattr(worker, "_refuse_free_while_in_flight", pause_after_fast_check)

    def free():
        try:
            worker.free(buffer)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=free, daemon=True)
    thread.start()
    try:
        assert checked.wait(5)
        worker._submit_l2_locked(3, arguments(buffer), CallConfig())
    finally:
        resume.set()
        thread.join(5)
    assert not thread.is_alive()
    assert len(calls) == 1 and not frees
    assert len(errors) == 1 and "in-flight L2" in str(errors[0])
    assert worker._child_alloc.get(buffer.identity) is not None


def test_binding_failure_releases_the_reservation(l2, monkeypatch):
    worker, calls, frees = l2
    buffer = register_buffer(worker)

    def fail(_args):
        raise ValueError("cannot import")

    monkeypatch.setattr(worker, "_materialize_l2_args", fail)
    with pytest.raises(ValueError, match="cannot import"):
        worker._submit_l2_locked(3, arguments(buffer), CallConfig())
    assert not calls and not worker._chip_run_touched_identities
    worker.free(buffer)
    assert frees == [buffer.base]


def test_accepted_device_buffer_stays_live_through_finalization(l2):
    worker, calls, frees = l2
    buffer = register_buffer(worker)
    handle = worker._submit_l2_locked(3, arguments(buffer), CallConfig())
    assert calls[0].tensor(0).data == buffer.base
    with pytest.raises(RuntimeError, match="in-flight L2"):
        worker.free(buffer)
    worker._finalize_run_handle(handle, handle._run_id, None)
    worker.free(buffer)
    assert frees == [buffer.base]


@pytest.mark.parametrize("second_offset, rejected", [(8, True), (16, False)])
def test_direct_l2_uses_the_shared_writable_overlap_rule(l2, second_offset, rejected):
    worker, calls, _ = l2
    buffer = register_buffer(worker)
    args = arguments(buffer, tag=TensorArgType.OUTPUT_EXISTING)
    args.add_tensor(buffer.tensor((4,), DataType.FLOAT32, byte_offset=second_offset), TensorArgType.OUTPUT_EXISTING)
    if rejected:
        with pytest.raises(ValueError, match="overlapping bytes"):
            worker._submit_l2_locked(3, args, CallConfig())
        assert not calls
    else:
        worker._submit_l2_locked(3, args, CallConfig())
        assert calls[0].tensor(1).data == buffer.base + second_offset


def test_source_wrap_is_not_a_worker_registration(l2):
    worker, calls, frees = l2
    backing = Buffer.wrap(address=0x8000, nbytes=64, location=worker.device_location)
    assert worker._child_alloc.get(backing.identity) is None
    args = TaskArgs()
    args.add_tensor(Tensor(backing, shapes=(4,), dtype=DataType.FLOAT32))
    handle = worker._submit_l2_locked(3, args, CallConfig())
    assert calls[0].tensor(0).data == backing.base
    with pytest.raises(RuntimeError, match="in-flight"):
        backing.close()
    with pytest.raises(ValueError, match="borrowed"):
        worker.free(backing)
    worker._finalize_run_handle(handle, handle._run_id, None)
    backing.close()
    assert not frees
    with pytest.raises(ValueError, match="released|not a live"):
        worker._submit_l2_locked(3, args, CallConfig())
    assert len(calls) == 1


def test_source_wrap_context_and_descriptor_checks_precede_materialization(l2, monkeypatch):
    worker, calls, _ = l2
    backing = Buffer.wrap(address=0x8000, nbytes=64, location=worker.device_location)
    original = arguments(backing)
    changed = replace(backing, nbytes=128)
    mapped = []
    monkeypatch.setattr(worker, "_materialize_l2_args", lambda args: mapped.append(args))
    with pytest.raises(ValueError, match="descriptor"):
        worker._submit_l2_locked(3, arguments(changed), CallConfig())
    assert not mapped and not calls
    backing.close()
    with pytest.raises(ValueError):
        worker._submit_l2_locked(3, original, CallConfig())
    assert not mapped and not calls


def test_source_wrap_reservation_precedes_mapping_and_unwinds_on_bind_failure(l2, monkeypatch):
    worker, calls, _ = l2
    backing = Buffer.wrap(address=0x8000, nbytes=64, location=worker.device_location)

    def fail(args):
        with pytest.raises(RuntimeError, match="in-flight"):
            backing.close()
        raise ValueError("cannot import")

    monkeypatch.setattr(worker, "_materialize_l2_args", fail)
    with pytest.raises(ValueError, match="cannot import"):
        worker._submit_l2_locked(3, arguments(backing), CallConfig())
    backing.close()
    assert not calls


def test_source_wrap_unknown_native_completion_retains_source(l2, monkeypatch):
    worker, _, _ = l2
    backing = Buffer.wrap(address=0x8000, nbytes=64, location=worker.device_location)

    def fail(*args):
        raise RuntimeError("unproven native completion")

    monkeypatch.setattr(worker._chip_worker._impl, "_submit_chip_run_direct", fail)
    with pytest.raises(RuntimeError, match="unproven"):
        worker._submit_l2_locked(3, arguments(backing), CallConfig())
    with pytest.raises(RuntimeError, match="in-flight"):
        backing.close()


@pytest.mark.parametrize("address,nbytes", [(True, 64), (0, 64), (1, 0), (1.2, 64), ((1 << 64) - 8, 64)])
def test_source_wrap_rejects_invalid_ranges(l2, address, nbytes):
    worker, calls, _ = l2
    with pytest.raises((ValueError, TypeError)):
        Buffer.wrap(address=address, nbytes=nbytes, location=worker.device_location)
    assert not calls


def test_source_wrap_rejects_other_context_and_closed_incarnation(l2):
    worker, calls, _ = l2
    location = worker.device_location
    backing = Buffer.wrap(address=0x8000, nbytes=64, location=location)
    other = Worker(level=2)
    other._lifecycle = _Lifecycle.READY
    other._chip_worker = worker._chip_worker
    try:
        with pytest.raises(ValueError, match="not a live"):
            other._submit_l2_locked(3, arguments(backing), CallConfig())
        assert not calls
        worker._lifecycle = _Lifecycle.CLOSED
        with pytest.raises(ValueError, match="live device context"):
            Buffer.wrap(address=0x9000, nbytes=64, location=location)
        with pytest.raises(ValueError, match="live device context"):
            backing.to_descriptor()
    finally:
        other._lifecycle = _Lifecycle.CLOSED
        other._chip_worker = None
        worker._lifecycle = _Lifecycle.READY
        backing.close()


def test_source_wrap_refuses_inherited_process_location(l2, monkeypatch):
    import simpler.buffer as buffer_module  # noqa: PLC0415

    worker, _, _ = l2
    location = worker.device_location
    monkeypatch.setattr(buffer_module.os, "getpid", lambda: -1)
    with pytest.raises(ValueError, match="this process"):
        Buffer.wrap(address=0x8000, nbytes=64, location=location)


def test_source_wrap_overlap_and_address_reuse_keep_distinct_identities(l2):
    worker, calls, _ = l2
    location = worker.device_location
    first = Buffer.wrap(address=0x8000, nbytes=64, location=location)
    old_args = arguments(first)
    with pytest.raises(ValueError, match="overlaps"):
        Buffer.wrap(address=0x8010, nbytes=64, location=location)
    handle = worker._submit_l2_locked(3, old_args, CallConfig())
    worker._finalize_run_handle(handle, handle._run_id, None)
    first.close()
    second = Buffer.wrap(address=0x8000, nbytes=64, location=location)
    assert first.identity != second.identity
    handle = worker._submit_l2_locked(3, arguments(second), CallConfig())
    worker._finalize_run_handle(handle, handle._run_id, None)
    with pytest.raises(ValueError, match="not a live"):
        worker._submit_l2_locked(3, old_args, CallConfig())
    second.close()
    assert len(calls) == 2


def test_source_cannot_alias_owned_allocation_and_failed_attach_unwinds(l2):
    worker, calls, _ = l2
    owned = register_buffer(worker)
    source = Buffer.wrap(address=owned.base, nbytes=owned.nbytes, location=worker.device_location)
    with pytest.raises(ValueError, match="overlaps"):
        worker._submit_l2_locked(3, arguments(source), CallConfig())
    source.close()
    assert not calls and not worker._source_attachments


def test_source_grant_failure_precedes_reading_host_bytes(l2):
    worker, calls, _ = l2
    source = Buffer.wrap(address=0x8000, nbytes=64, location=worker.device_location, access=AccessMode.READ)
    args = arguments(wrap_fork_inherited(1, 64, worker._owner_instance_id, 99, access=AccessMode.READ))
    args.add_tensor(Tensor(source, shapes=(4,), dtype=DataType.FLOAT32))
    args.set_tag(1, TensorArgType.OUTPUT_EXISTING)
    with pytest.raises(ValueError, match="does not grant"):
        worker._submit_l2_locked(3, args, CallConfig())
    source.close()
    assert not calls


def test_source_close_is_fenced_while_another_thread_materializes(l2, monkeypatch):
    worker, calls, _ = l2
    source = Buffer.wrap(address=0x8000, nbytes=64, location=worker.device_location)
    entered = threading.Event()
    resume = threading.Event()
    handles = []
    errors = []
    original = worker._materialize_l2_args

    def paused(args):
        entered.set()
        assert resume.wait(5)
        return original(args)

    def submit():
        try:
            handles.append(worker._submit_l2_locked(3, arguments(source), CallConfig()))
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(worker, "_materialize_l2_args", paused)
    thread = threading.Thread(target=submit, daemon=True)
    thread.start()
    try:
        assert entered.wait(5)
        with pytest.raises(RuntimeError, match="in-flight"):
            source.close()
    finally:
        resume.set()
        thread.join(5)
    assert not thread.is_alive() and not errors and len(calls) == 1
    handle = handles[0]
    worker._finalize_run_handle(handle, handle._run_id, None)
    source.close()


def test_source_run_error_does_not_prove_completion(l2):
    worker, _, _ = l2
    source = Buffer.wrap(address=0x8000, nbytes=64, location=worker.device_location)
    handle = worker._submit_l2_locked(3, arguments(source), CallConfig())
    worker._finalize_run_handle(handle, handle._run_id, RuntimeError("device fault"))
    with pytest.raises(RuntimeError, match="unproven"):
        source.close()


@pytest.mark.parametrize("failure", ["submit", "wait"])
def test_native_error_invalidates_source_context_token(l2, monkeypatch, failure):
    worker, _, _ = l2
    location = worker.device_location
    source = Buffer.wrap(address=0x8000, nbytes=64, location=location)
    if failure == "submit":

        def fail(*args):
            raise RuntimeError("native failure")

        monkeypatch.setattr(worker._chip_worker._impl, "_submit_chip_run_direct", fail)
        with pytest.raises(RuntimeError, match="native failure"):
            worker._submit_l2_locked(3, arguments(source), CallConfig())
    else:
        handle = worker._submit_l2_locked(3, arguments(source), CallConfig())
        worker._finalize_run_handle(handle, handle._run_id, RuntimeError("native failure"))
    with pytest.raises(ValueError, match="live device context"):
        Buffer.wrap(address=0x9000, nbytes=64, location=location)


def test_source_attach_interruption_rolls_back_registration_and_use(l2, monkeypatch):
    worker, calls, _ = l2
    source = Buffer.wrap(address=0x8000, nbytes=64, location=worker.device_location)
    record = worker._record_device_alloc

    def fail(buffer):
        record(buffer)
        raise RuntimeError("interrupted registration")

    monkeypatch.setattr(worker, "_record_device_alloc", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        worker._submit_l2_locked(3, arguments(source), CallConfig())
    assert worker._child_alloc.get(source.identity) is None
    assert not worker._source_attachments and not calls
    source.close()


def test_worker_close_drains_an_unawaited_source_use(l2):
    worker, _, _ = l2
    source = Buffer.wrap(address=0x8000, nbytes=64, location=worker.device_location)
    worker._submit_l2_locked(3, arguments(source), CallConfig())
    worker._chip_worker._impl._close_chip_run_lane = lambda: None
    worker._chip_worker._impl.workspace_report = lambda: ("disabled", {})
    worker._chip_worker.finalize = lambda: None
    worker.close()
    source.close()
    assert source.closed


def test_cached_source_closed_before_reservation_cannot_submit(l2, monkeypatch):
    worker, calls, _ = l2
    backing = Buffer.wrap(address=0x8000, nbytes=64, location=worker.device_location)
    args = arguments(backing)
    handle = worker._submit_l2_locked(3, args, CallConfig())
    worker._finalize_run_handle(handle, handle._run_id, None)
    source = worker._device_buffer_source
    original = source.live

    def close_after_cache_check(identity):
        was_live = original(identity)
        backing.close()
        return was_live

    monkeypatch.setattr(source, "live", close_after_cache_check)
    with pytest.raises(ValueError, match="released|not a live"):
        worker._submit_l2_locked(3, args, CallConfig())
    assert len(calls) == 1


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_source_inherited_with_locked_mutex_rejects_before_locking(l2):
    worker, _, _ = l2
    location = worker.device_location
    backing = Buffer.wrap(address=0x8000, nbytes=64, location=location)
    with location._source._lock:
        pid = os.fork()
        if pid == 0:
            signal.alarm(3)
            operations = [
                lambda: Buffer.wrap(address=0x9000, nbytes=64, location=location),
                backing.to_descriptor,
                backing.close,
            ]
            for operation in operations:
                try:
                    operation()
                except ValueError:
                    continue
                except BaseException:
                    os._exit(3)
                os._exit(2)
            os._exit(0)
        _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, status
    backing.close()
