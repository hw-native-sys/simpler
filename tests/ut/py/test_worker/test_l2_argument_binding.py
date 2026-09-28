# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Direct L2 submission validates the same Buffer identities and argument grants as L3."""

import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
from simpler.buffer import AccessMode, mint_owner_instance_id, wrap_device_malloc, wrap_fork_inherited
from simpler.task_interface import CallConfig, DataType, TaskArgs, TensorArgType
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
