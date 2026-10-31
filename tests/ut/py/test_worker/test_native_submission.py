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

        def _retire(self):
            self.retired = True

        def capsule(self):
            return self

    def prepare(cid, args, config, *, caller_stream):
        assert caller_stream == 123
        result = Pending(cid, args, config)
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
    with pytest.raises(RuntimeError, match="in-flight L2 run"):
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


@pytest.mark.parametrize("replace_callable", [False, True])
def test_unregister_during_preparation_refuses_original_handle_without_source_leak(
    prepared, monkeypatch, replace_callable
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
    original_submit = worker._submit_l2_locked

    def pause_before_reservation(*args, **kwargs):
        resolved.set()
        assert resume.wait(5)
        return original_submit(*args, **kwargs)

    monkeypatch.setattr(worker, "_submit_l2_locked", pause_before_reservation)

    def prepare():
        try:
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
            handle.cancel()
        # The transport is fake; clean up a failed regression run's leaked use.
        if not backing.closed:
            source = worker._device_buffer_source
            while source._records[backing.identity].uses:
                source.release((backing.identity,))
            backing.close()
