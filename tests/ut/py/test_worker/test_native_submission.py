# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public preparation/retirement tests with a recording native transport."""

import gc
import threading
import weakref

import pytest
from simpler import worker as worker_module
from simpler.task_interface import RunPreparationRefused

from . import test_l2_close_drain

chip_fixture = test_l2_close_drain.chip


@pytest.fixture
def prepared(chip_fixture, monkeypatch):
    chip = chip_fixture
    pending = []

    class Pending:
        def __init__(self, cid, args, config):
            self.values = cid, args, config
            self.ready = threading.Event()
            self.entered_wait = threading.Event()
            self.run = None
            self.error = None
            self.retired = False
            self.recovery_required = True

        def submit(self):
            assert not self.ready.is_set()
            self.run = chip.impl._submit_chip_run_direct(*self.values)
            self.ready.set()

        def cancel(self):
            if self.ready.is_set():
                return False
            self.error = RunPreparationRefused("cancelled")
            self.ready.set()
            return True

        def done(self):
            if not self.ready.is_set():
                return False
            if self.error is not None:
                return True
            assert self.run is not None
            return self.run.done()

        def wait(self, timeout=-1):
            self.entered_wait.set()
            if not self.ready.wait(None if timeout < 0 else timeout):
                return False
            if self.error:
                raise self.error
            assert self.run is not None
            return self.run.wait(timeout)

        @property
        def _owner_recovery_required(self):
            return self.recovery_required

        def _retire(self):
            self.retired = True

        def capsule(self):
            return self

    def prepare(cid, values, config, *, caller_stream):
        assert caller_stream == 123
        result = Pending(cid, values.args, config)
        pending.append(result)
        return result

    monkeypatch.setattr(chip.impl, "_validate_caller_stream", lambda stream: None, raising=False)
    monkeypatch.setattr(chip.impl, "_prepare_chip_submission", prepare, raising=False)
    yield chip, pending
    # Recording transport only: settle all outstanding fixtures without device work.
    for item in pending:
        item.cancel()
    for handle in tuple(worker_module._native_submission_roots):
        if handle._worker is chip.worker:
            try:
                handle.wait()
            except Exception:
                pass
            worker_module._native_submission_roots.pop(handle, None)


def test_prepare_is_not_submit_and_unregister_refusal_preserves_handle(prepared):
    chip, pending = prepared
    handle = chip.worker.prepare_submission(chip.callable, caller_stream=123)
    assert not chip.runs and not handle.done
    assert handle.native_handle is pending[0]
    with pytest.raises(RuntimeError, match="unretired L2 use"):
        chip.worker.unregister(chip.callable)
    assert handle.cancel()
    assert pending[0].retired
    assert handle not in worker_module._native_submission_roots
    chip.worker.unregister(chip.callable)


def test_cancel_wakes_wait_and_retires_owners(prepared):
    chip, pending = prepared

    class Owner:
        pass

    owner = Owner()
    weak_owner = weakref.ref(owner)
    handle = chip.worker.prepare_submission(chip.callable, caller_stream=123, keepalive=(owner,))
    del owner
    gc.collect()
    assert weak_owner() is not None
    assert handle.cancel()
    assert not handle.cancel()
    assert not chip.runs
    assert pending[0].retired
    gc.collect()
    assert weak_owner() is None
    with pytest.raises(RunPreparationRefused):
        handle.wait()


def test_dropped_python_handle_is_rooted_until_close(prepared):
    chip, pending = prepared
    handle = chip.worker.prepare_submission(chip.callable, caller_stream=123)
    weak_handle = weakref.ref(handle)
    del handle
    gc.collect()
    assert weak_handle() is not None
    pending[0].submit()
    chip.worker.close()
    gc.collect()
    assert weak_handle() is None
    assert pending[0].retired
    assert chip.impl.finalized


def test_close_waits_for_queued_submission_before_teardown(prepared):
    chip, pending = prepared
    handle = chip.worker.prepare_submission(chip.callable, caller_stream=123)

    def callback():
        assert pending[0].entered_wait.wait(5)
        assert not chip.impl.finalized
        pending[0].submit()

    thread = threading.Thread(target=callback)
    thread.start()
    chip.worker.close()
    thread.join(5)
    assert not thread.is_alive()
    assert handle.done and pending[0].retired
    assert chip.impl.finalized


def test_unsubmitted_token_close_times_out_without_teardown(prepared, monkeypatch):
    chip, pending = prepared
    handle = chip.worker.prepare_submission(chip.callable, caller_stream=123)
    monkeypatch.setattr(worker_module, "_ROLLBACK_GRACEFUL_TIMEOUT_S", 0.01)
    with pytest.raises(TimeoutError):
        chip.worker.close()
    assert not chip.impl.finalized
    assert handle in worker_module._native_submission_roots
    assert handle.cancel()
    chip.worker.close()
    assert chip.impl.finalized


def test_unknown_completion_retains_external_owners(prepared):
    chip, pending = prepared
    handle = chip.worker.prepare_submission(chip.callable, caller_stream=123, keepalive=(object(),))
    pending[0].submit()
    chip.state.error = RuntimeError("completion unknown")
    with pytest.raises(RuntimeError, match="completion unknown"):
        handle.wait()
    assert handle in worker_module._native_submission_roots
    assert not pending[0].retired


def test_submit_then_cancel_does_not_revoke_native_run(prepared):
    chip, pending = prepared
    handle = chip.worker.prepare_submission(chip.callable, caller_stream=123)
    pending[0].submit()
    assert not handle.cancel()
    handle.wait()
    assert pending[0].retired
    assert handle not in worker_module._native_submission_roots


@pytest.mark.parametrize("entry", ["prepare_submission", "submit"])
@pytest.mark.parametrize("replace_callable", [False, True])
def test_unregister_during_preparation_refuses_original_handle_without_source_leak(
    prepared, monkeypatch, replace_callable, entry
):
    from simpler.buffer import Buffer  # noqa: PLC0415
    from simpler.task_interface import ChipCallable  # noqa: PLC0415

    from .test_l2_argument_binding import arguments  # noqa: PLC0415

    chip, pending = prepared
    worker = chip.worker
    backing = Buffer.wrap(address=0x8000, nbytes=64, location=worker.device_location)
    original_slot = worker._resolve_handle(chip.callable).slot_id
    resolved, resume = threading.Event(), threading.Event()
    handles, errors = [], []
    original_prepare = worker._prepare_l2_args

    def pause_before_reservation(*args, **kwargs):
        resolved.set()
        assert resume.wait(5)
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(worker, "_prepare_l2_args", pause_before_reservation)

    def prepare():
        try:
            if entry == "submit":
                handles.append(worker.submit(chip.callable, arguments(backing)))
            else:
                handles.append(worker.prepare_submission(chip.callable, arguments(backing), caller_stream=123))
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=prepare, daemon=True)
    thread.start()
    try:
        assert resolved.wait(5)
        worker.unregister(chip.callable)
        if replace_callable:
            replacement = worker.register(
                ChipCallable.build(signature=[], func_name="replacement", binary=b"\x01", children=[])
            )
            assert replacement.digest != chip.callable.digest
            assert worker._resolve_handle(replacement).slot_id == original_slot
        resume.set()
        thread.join(5)
        assert not thread.is_alive()
        assert not handles, "an invalid original handle must never bind to a reused callable slot"
        assert not pending and not chip.runs
        assert not worker._chip_source_uses and not worker._chip_run_callable_ids
        assert worker._device_buffer_source._records[backing.identity].uses == 0
        backing.close()
        assert backing.closed
        assert len(errors) == 1 and isinstance(errors[0], KeyError)
    finally:
        resume.set()
        thread.join(5)
        for handle in handles:
            if isinstance(handle, worker_module.PreparedSubmission):
                handle.cancel()
            else:
                handle.wait()
        # The transport is fake; clean up a failed regression run's leaked use.
        if not backing.closed:
            source = worker._device_buffer_source
            while source._records[backing.identity].uses:
                source.release((backing.identity,))
            backing.close()


def test_safe_retirement_after_cancel(prepared):
    chip, _ = prepared
    ticket = chip.worker.prepare_submission(chip.callable, caller_stream=123)
    assert not ticket.safe_to_release
    assert ticket.cancel()
    assert ticket.safe_to_release
    with pytest.raises(RunPreparationRefused):
        ticket.wait()
    assert ticket.safe_to_release


def test_safe_retirement_after_success(prepared):
    chip, pending = prepared
    ticket = chip.worker.prepare_submission(chip.callable, caller_stream=123)
    pending[0].submit()
    assert not ticket.safe_to_release
    ticket.wait()
    assert ticket.safe_to_release


def test_timeout_is_not_retirement(prepared):
    chip, _ = prepared
    ticket = chip.worker.prepare_submission(chip.callable, caller_stream=123)
    with pytest.raises(TimeoutError):
        ticket.wait(timeout=0)
    assert not ticket.safe_to_release
    assert ticket.cancel()
    assert ticket.safe_to_release


@pytest.mark.parametrize("recovery_required", [False, True])
def test_failure_retirement_uses_native_cleanup_proof(prepared, recovery_required):
    chip, pending = prepared

    class Owner:
        pass

    owner = Owner()
    weak_owner = weakref.ref(owner)
    ticket = chip.worker.prepare_submission(chip.callable, caller_stream=123, keepalive=(owner,))
    del owner
    pending[0].submit()
    pending[0].recovery_required = recovery_required
    error = RuntimeError("original execution failure")
    chip.state.error = error
    with pytest.raises(RuntimeError, match="original execution failure") as caught:
        ticket.wait()
    assert caught.value is error
    assert ticket.safe_to_release is (not recovery_required)
    assert (ticket in worker_module._native_submission_roots) is recovery_required
    assert pending[0].retired is (not recovery_required)
    gc.collect()
    assert (weak_owner() is not None) is recovery_required
    with pytest.raises(RuntimeError) as again:
        ticket.wait()
    assert again.value is error


def test_ordinary_submit_unregister_is_retryable_after_wait(chip_fixture):
    chip = chip_fixture
    handle = chip.worker.submit(chip.callable)
    with pytest.raises(RuntimeError, match="unretired L2 use"):
        chip.worker.unregister(chip.callable)
    handle.wait()
    chip.worker.unregister(chip.callable)


@pytest.mark.parametrize("recovery_required", [False, True])
def test_ordinary_failure_retires_callable_use_only_with_cleanup_proof(chip_fixture, recovery_required):
    chip = chip_fixture
    error = RuntimeError("device execution failed")
    chip.state.error = error
    handle = chip.worker.submit(chip.callable)
    run_id = handle._run_id
    chip.runs[0]._owner_recovery_required = recovery_required
    chip.runs[0].lane_poisoned = False
    with pytest.raises(RuntimeError) as raised:
        handle.wait()
    assert raised.value is error
    if recovery_required:
        assert chip.worker._source_context_failed
        assert run_id in chip.worker._chip_run_callable_ids
        assert run_id in chip.worker._chip_run_touched_identities
        with pytest.raises(RuntimeError, match="unretired L2 use"):
            chip.worker.unregister(chip.callable)
    else:
        assert not chip.worker._source_context_failed
        assert not chip.worker._chip_run_callable_ids
        assert not chip.worker._chip_run_touched_identities
        chip.worker.unregister(chip.callable)
    chip.state.error = None
    chip.live.clear()


def test_failed_enqueue_after_submit_keeps_ticket(prepared):
    chip, pending = prepared
    ticket = chip.worker.prepare_submission(chip.callable, caller_stream=123)
    pending[0].submit()
    assert not ticket.cancel()
    assert not ticket.safe_to_release
    assert ticket in worker_module._native_submission_roots
    ticket.wait()
    assert ticket.safe_to_release


def test_context_refusal_preserves_ticket_for_retry(prepared):
    from simpler.task_interface import ExecutionContextRefused  # noqa: PLC0415

    chip, pending = prepared
    ticket = chip.worker.prepare_submission(chip.callable, caller_stream=123)
    pending[0].submit()
    chip.state.error = ExecutionContextRefused("wrong context")
    with pytest.raises(ExecutionContextRefused):
        ticket.wait()
    assert not ticket.safe_to_release
    assert ticket in worker_module._native_submission_roots
    chip.state.error = None
    ticket.wait()
    assert ticket.safe_to_release


def test_unknown_completion_retires_only_after_successful_owner_close(prepared, monkeypatch):
    chip, pending = prepared
    ticket = chip.worker.prepare_submission(chip.callable, caller_stream=123, keepalive=(object(),))
    pending[0].submit()
    chip.state.error = RuntimeError("unknown completion")
    with pytest.raises(RuntimeError, match="unknown completion"):
        ticket.wait()
    assert not ticket.safe_to_release
    original_finalize = chip.impl.finalize
    attempts = []

    def finalize():
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("owner recovery refused")
        chip.live.clear()
        original_finalize()

    monkeypatch.setattr(chip.impl, "finalize", finalize)
    with pytest.raises(RuntimeError, match="owner recovery refused"):
        chip.worker.close()
    assert not ticket.safe_to_release
    assert ticket in worker_module._native_submission_roots
    chip.worker.close()
    assert ticket.safe_to_release
    assert pending[0].retired
    with pytest.raises(RuntimeError, match="unknown completion"):
        ticket.wait()


def test_cancel_retains_owners_when_retirement_fails(prepared, monkeypatch):
    chip, pending = prepared
    ticket = chip.worker.prepare_submission(chip.callable, caller_stream=123, keepalive=(object(),))
    release = chip.worker._release_source_uses

    def fail_release(run_id):
        raise RuntimeError("injected retirement failure")

    monkeypatch.setattr(chip.worker, "_release_source_uses", fail_release)
    assert ticket.cancel()
    assert not ticket.cancel()
    assert not ticket.safe_to_release
    assert not pending[0].retired
    assert ticket in worker_module._native_submission_roots
    with pytest.raises(RunPreparationRefused, match="cancelled"):
        ticket.wait()
    monkeypatch.setattr(chip.worker, "_release_source_uses", release)
    chip.worker.close()
    assert ticket.safe_to_release
    assert pending[0].retired


@pytest.mark.parametrize("stage", ["before_publication", "after_chip_run_publication"])
def test_publication_failure_cancels_unpublished_token(prepared, monkeypatch, stage):
    chip, pending = prepared

    if stage == "before_publication":

        def fail(*args):
            raise RuntimeError("injected publication failure")

        monkeypatch.setattr(chip.worker, "_accept_l2_run", fail)
    else:
        # _accept_l2_run publishes the native run, then fails adding the handle.
        class FailingAdd(set):
            def add(self, item):
                raise RuntimeError("injected publication failure")

        monkeypatch.setattr(chip.worker, "_accepted_run_handles", FailingAdd(chip.worker._accepted_run_handles))
    with pytest.raises(RuntimeError, match="injected publication failure"):
        chip.worker.prepare_submission(chip.callable, caller_stream=123)
    assert len(pending) == 1 and isinstance(pending[0].error, RunPreparationRefused)
    assert pending[0].retired
    assert not chip.worker._chip_runs and not chip.worker._chip_run_callable_ids
    assert not chip.worker._accepted_run_handles
    assert not chip.worker._chip_run_touched_identities and not chip.worker._source_context_failed
    assert not any(handle._worker is chip.worker for handle in worker_module._native_submission_roots)
    chip.worker.unregister(chip.callable)


@pytest.mark.parametrize("failure_stage", ["materialize", "native_prepare"])
def test_prepare_failure_releases_callable_reservation(prepared, monkeypatch, failure_stage):
    chip, pending = prepared

    def fail(*args, **kwargs):
        raise RuntimeError("injected preparation failure")

    if failure_stage == "materialize":
        monkeypatch.setattr(chip.worker, "_materialize_l2_args", fail)
    else:
        monkeypatch.setattr(chip.impl, "_prepare_chip_submission", fail)
    with pytest.raises(RuntimeError, match="injected preparation failure"):
        chip.worker.prepare_submission(chip.callable, caller_stream=123)
    assert not pending and not chip.runs
    assert not chip.worker._chip_run_callable_ids
    assert not chip.worker._chip_run_touched_identities
    assert not chip.worker._source_context_failed
    assert not any(handle._worker is chip.worker for handle in worker_module._native_submission_roots)
    chip.worker.unregister(chip.callable)


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("handoff", [False, True])
def test_host_control_values_survive_delayed_consumption(prepared, monkeypatch, strided, handoff):
    import ctypes  # noqa: PLC0415

    from simpler.buffer import (  # noqa: PLC0415
        AccessMode,
        TensorTransfer,
        create_host_shared_buffer,
        mint_owner_instance_id,
    )
    from simpler.task_interface import (  # noqa: PLC0415
        ArgDirection,
        ChipCallable,
        DataType,
        TaskArgs,
        Tensor,
        TensorArgType,
    )

    chip, pending = prepared
    callable = chip.worker.register(
        ChipCallable.build(signature=[ArgDirection.IN], func_name="host_values", binary=b"x", children=[])
    )
    backing = create_host_shared_buffer(32, mint_owner_instance_id(), 1)
    try:
        backing.access = AccessMode.READ
        values = (ctypes.c_int32 * 8).from_address(backing.base)
        values[:] = [99, 7, 11, 13, 17, 23, 29, 31]
        args = TaskArgs()
        args.add_tensor(
            Tensor(backing, shapes=(2,), strides=(2 if strided else 1,), dtype=DataType.INT32, byte_offset=4),
            TensorArgType.INPUT,
            transfer=TensorTransfer.NONE,
        )
        args.add_scalar(41)
        recorded = []
        submit = chip.impl._submit_chip_run_direct

        def record(cid, native_args, config, **kwargs):
            recorded.append(native_args)
            return submit(cid, native_args, config, **kwargs)

        monkeypatch.setattr(chip.impl, "_submit_chip_run_direct", record)
        ticket = (
            chip.worker.prepare_submission(callable, args, caller_stream=123)
            if handoff
            else chip.worker.submit(callable, args, caller_stream=123)
        )
        values[1] = 19
        values[3 if strided else 2] = 37
        args.clear()
        native_args = pending[-1].values[1] if handoff else recorded[0]
        tensor = native_args.tensor(0)
        observed = (ctypes.c_int32 * (3 if strided else 2)).from_address(tensor.data)
        assert observed[0] == 7
        assert observed[2 if strided else 1] == (13 if strided else 11)
        assert tensor.shapes == (2,)
        assert tensor.strides == (2 if strided else 1,)
        assert tensor.data != backing.base + 4
        assert native_args.scalar(0) == 41
        del args, values
        backing.close()
        gc.collect()
        assert observed[0] == 7
        if handoff:
            pending[-1].submit()
        ticket.wait()
        if handoff:
            assert ticket.safe_to_release
    finally:
        backing.close()


@pytest.mark.parametrize("direction", ["OUT", "INOUT"])
def test_host_writable_handoff_preserves_external_destination(prepared, direction):
    import ctypes  # noqa: PLC0415

    from simpler.buffer import TensorTransfer, create_host_shared_buffer, mint_owner_instance_id  # noqa: PLC0415
    from simpler.task_interface import (  # noqa: PLC0415
        ArgDirection,
        ChipCallable,
        DataType,
        TaskArgs,
        Tensor,
        TensorArgType,
    )

    chip, pending = prepared
    callable = chip.worker.register(
        ChipCallable.build(
            signature=[getattr(ArgDirection, direction)], func_name="host_write", binary=b"x", children=[]
        )
    )
    backing = create_host_shared_buffer(16, mint_owner_instance_id(), 2)
    try:
        args = TaskArgs()
        args.add_tensor(
            Tensor(backing, shapes=(1,), dtype=DataType.INT32), TensorArgType.INOUT, transfer=TensorTransfer.NONE
        )
        ticket = chip.worker.prepare_submission(callable, args, caller_stream=123)
        native = pending[-1].values[1].tensor(0)
        ctypes.c_int32.from_address(native.data).value = 23
        assert ctypes.c_int32.from_address(backing.base).value == 23
        assert ticket.cancel() and ticket.safe_to_release
    finally:
        backing.close()


@pytest.mark.parametrize("safe", [False, True])
def test_call_value_retention_follows_native_failure_proof(prepared, monkeypatch, safe):
    chip, _ = prepared
    failure = RunPreparationRefused("refused before execution") if safe else RuntimeError("unknown native outcome")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(chip.impl, "_submit_chip_run_direct", fail)
    with pytest.raises(type(failure)) as caught:
        chip.worker.submit(chip.callable)
    assert caught.value is failure
    assert bool(chip.worker._chip_call_values) is not safe
    chip.worker.close()
    assert not chip.worker._chip_call_values


def test_snapshot_preparation_defers_stream_admission_but_ordinary_submit_keeps_it(prepared, monkeypatch):
    chip, pending = prepared
    checked = []

    def reject_stream(stream):
        checked.append(stream)
        raise RunPreparationRefused("stream is already capturing")

    monkeypatch.setattr(chip.impl, "_validate_caller_stream", reject_stream, raising=False)
    handle = chip.worker.prepare_submission(chip.callable, caller_stream=123)
    assert not checked and not chip.runs
    assert handle.native_handle is pending[0]
    assert handle.cancel()
    with pytest.raises(RunPreparationRefused, match="stream is already capturing"):
        chip.worker.submit(chip.callable, caller_stream=123)
    assert checked == [123]
    assert not chip.runs
