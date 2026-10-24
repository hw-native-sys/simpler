# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public L2 close must cross accepted run fences before its workspace gate.

The real Worker submits, waits and finalizes; only the native transport and its
workspace report are recording boundaries. No runtime or device is executed.
"""

import gc
import threading
import weakref
from types import SimpleNamespace

import pytest
from simpler.buffer import Buffer
from simpler.task_interface import (
    ArgDirection,
    ChipCallable,
    DataType,
    ExecutionContextRefused,
    TaskArgs,
    Tensor,
    TensorArgType,
)
from simpler.worker import Worker, _Lifecycle

from simpler_setup import runtime_builder

from ._harness import install_fake_chip


@pytest.fixture(params=["host_build_graph", "tensormap_and_ringbuffer"])
def chip(monkeypatch, request):
    install_fake_chip(monkeypatch)
    monkeypatch.setattr(
        runtime_builder,
        "RuntimeBuilder",
        lambda *_: SimpleNamespace(
            get_binaries=lambda *_: SimpleNamespace(host_path="unused", aicpu_path="unused", aicore_path="unused")
        ),
    )
    worker = Worker(level=2, device_id=0, platform="a2a3sim", runtime=request.param)
    callable = worker.register(ChipCallable.build(signature=[], func_name="close_probe", binary=b"\x00", children=[]))
    worker.init()
    impl = worker._chip_worker
    assert impl is not None
    live, events, native_runs = set(), [], []
    state = SimpleNamespace(completes=True, error=None, entered=None, release=None)
    impl.latch_workspace_budget(4096)
    original_report = impl.workspace_report

    def report():
        impl.workspace_fields = dict(blocks_published=1, live_blocked=len(live))
        return original_report()

    class NativeRun:
        wait_count = 0

        def done(self):
            return self not in live

        def wait(self, timeout=-1):
            self.wait_count += 1
            events.append("native wait")
            if state.entered is not None:
                state.entered.set()
                assert state.release.wait(5)
            if state.error is not None:
                raise state.error
            if not state.completes:
                return False
            live.remove(self)
            return True

    def submit(*args, **kwargs):
        run = NativeRun()
        live.add(run)
        native_runs.append(run)
        return run

    monkeypatch.setattr(impl, "workspace_report", report)
    monkeypatch.setattr(impl, "_submit_chip_run_direct", submit)
    yield SimpleNamespace(
        worker=worker, callable=callable, impl=impl, events=events, live=live, runs=native_runs, state=state
    )
    # No device exists: settle the recording transport if a failing assertion
    # prevented the real Worker's close from reaching this test's fake fences.
    live.clear()
    try:
        worker.close()
    except RuntimeError:
        pass


def test_close_drains_unwaited_l2_run_before_workspace_teardown(chip):
    handle = chip.worker.submit(chip.callable)
    assert chip.live and chip.runs[0].wait_count == 0
    chip.worker.close()
    assert chip.runs[0].wait_count == 1
    assert not chip.live and not chip.worker._chip_runs
    assert not chip.worker._accepted_run_handles
    assert handle.done and handle.result() is None and handle.wait() is None
    assert chip.runs[0].wait_count == 1
    assert chip.impl.lane_closed_before_finalize and chip.impl.finalized


def test_close_owns_a_dropped_handle_and_drains_each_run_once(chip):
    handles = [chip.worker.submit(chip.callable) for _ in range(3)]
    dropped = weakref.ref(handles.pop())
    gc.collect()
    assert dropped() is not None
    chip.worker.close()
    assert all(run.wait_count == 1 for run in chip.runs)
    assert all(handle.done and handle.result() is None for handle in handles)
    assert not chip.worker._accepted_run_handles and not chip.worker._chip_runs
    gc.collect()
    assert dropped() is None


def test_normal_wait_retires_the_accepted_set_before_close(chip):
    handle = chip.worker.submit(chip.callable)
    assert handle in chip.worker._accepted_run_handles
    handle.wait()
    assert handle._cleanup_published and handle._finalization_error is None
    assert not chip.worker._accepted_run_handles
    chip.worker.close()
    handle.wait()
    assert chip.runs[0].wait_count == 1


@pytest.mark.parametrize("refusal", ["timeout", "context"])
def test_incomplete_close_keeps_the_run_and_can_retry_after_closed(chip, refusal):
    handle = chip.worker.submit(chip.callable)
    keepalive = handle._keepalive
    if refusal == "timeout":
        chip.state.completes = False
        error_type = TimeoutError
    else:
        chip.state.error = ExecutionContextRefused("caller context is not current")
        error_type = ExecutionContextRefused
    with pytest.raises(error_type):
        chip.worker.close()
    assert chip.worker._lifecycle is _Lifecycle.CLOSED
    assert handle in chip.worker._accepted_run_handles and not handle._cleanup_published
    assert handle._keepalive is keepalive and chip.live
    assert not chip.impl.lane_closed and not chip.impl.finalized
    chip.state.completes, chip.state.error = True, None
    handle.wait()
    assert handle.done and not chip.worker._accepted_run_handles
    chip.worker.close()
    assert chip.impl.finalized


def test_close_waits_for_in_progress_submission_to_publish_its_handle(chip, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    submitted, errors = [], []
    original = chip.impl._submit_chip_run_direct

    def submit(*args, **kwargs):
        native = original(*args, **kwargs)
        entered.set()
        assert release.wait(5)
        return native

    monkeypatch.setattr(chip.impl, "_submit_chip_run_direct", submit)

    def submit_thread():
        try:
            submitted.append(chip.worker.submit(chip.callable))
        except BaseException as error:
            errors.append(error)

    def release_after_close_commits():
        with chip.worker._hierarchical_start_cv:
            assert chip.worker._hierarchical_start_cv.wait_for(
                lambda: chip.worker._lifecycle is _Lifecycle.CLOSED, timeout=5
            )
        release.set()

    thread = threading.Thread(target=submit_thread, daemon=True)
    releaser = threading.Thread(target=release_after_close_commits, daemon=True)
    thread.start()
    assert entered.wait(5)
    releaser.start()
    try:
        chip.worker.close()
    finally:
        release.set()
        thread.join(5)
        releaser.join(5)
    assert not thread.is_alive() and not releaser.is_alive() and not errors
    assert len(submitted) == 1 and submitted[0].done
    assert chip.runs[0].wait_count == 1 and not chip.worker._accepted_run_handles


def test_concurrent_waiter_and_close_cross_one_native_fence(chip):
    handle = chip.worker.submit(chip.callable)
    chip.state.entered, chip.state.release = threading.Event(), threading.Event()
    errors = []

    def wait():
        try:
            handle.wait()
        except BaseException as error:
            errors.append(error)

    def release_after_close_commits():
        with chip.worker._hierarchical_start_cv:
            assert chip.worker._hierarchical_start_cv.wait_for(
                lambda: chip.worker._lifecycle is _Lifecycle.CLOSED, timeout=5
            )
        chip.state.release.set()

    waiter = threading.Thread(target=wait, daemon=True)
    releaser = threading.Thread(target=release_after_close_commits, daemon=True)
    waiter.start()
    assert chip.state.entered.wait(5)
    releaser.start()
    try:
        chip.worker.close()
    finally:
        chip.state.release.set()
        waiter.join(5)
        releaser.join(5)
    assert not waiter.is_alive() and not releaser.is_alive() and not errors
    assert chip.runs[0].wait_count == 1 and handle.done


def test_close_releases_real_source_reservations_after_fence(chip):
    target = chip.worker.register(
        ChipCallable.build(signature=[ArgDirection.IN], func_name="source_probe", binary=b"\x00", children=[])
    )
    backing = Buffer.wrap(address=0x8000, nbytes=64, location=chip.worker.device_location)
    args = TaskArgs()
    args.add_tensor(Tensor(backing, shapes=(4,), dtype=DataType.FLOAT32), TensorArgType.INPUT)
    handle = chip.worker.submit(target, args)
    assert chip.worker._chip_source_uses
    with pytest.raises(RuntimeError, match="in-flight"):
        backing.close()
    chip.worker.close()
    assert handle.done and not chip.worker._chip_source_uses
    backing.close()
    assert backing.closed


def test_interrupted_l2_source_cleanup_uses_existing_abandonment(chip, monkeypatch):
    handle = chip.worker.submit(chip.callable)
    keepalive = handle._keepalive
    failure = KeyboardInterrupt("source cleanup interrupted")

    def interrupt(run_id):
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(chip.worker, "_release_source_uses", interrupt)
        with pytest.raises(KeyboardInterrupt) as caught:
            handle.wait()
    assert caught.value is failure and chip.runs[0].wait_count == 1
    assert not chip.live and not chip.worker._chip_runs and not chip.worker._accepted_run_handles
    assert handle._finalization_abandoned and handle in chip.worker._abandoned_run_handles
    assert handle._keepalive is keepalive and chip.worker._ordered_cleanup_error is not None
    with pytest.raises(RuntimeError, match="ordered cleanup"):
        chip.worker.submit(chip.callable)
    chip.worker.close()
    assert not chip.worker._abandoned_run_handles and handle._keepalive is None
    with pytest.raises(KeyboardInterrupt) as caught:
        handle.wait()
    assert caught.value is failure and chip.runs[0].wait_count == 1


def test_close_only_lane_failure_remains_visible_across_journal_retries(chip, monkeypatch):
    handle = chip.worker.submit(chip.callable)
    failure = RuntimeError("lane close failed")
    attempts = []

    def fail_close():
        attempts.append(True)
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(chip.impl, "_close_chip_run_lane", fail_close)
        for _ in range(2):
            with pytest.raises(RuntimeError) as caught:
                chip.worker.close()
            assert caught.value is failure
            assert handle.done and handle.result() is None
            assert not chip.worker._chip_runs and not chip.worker._accepted_run_handles
            assert not chip.impl.finalized
    assert len(attempts) == 2 and chip.runs[0].wait_count == 1
    chip.worker.close()
    assert chip.impl.finalized


def test_close_keeps_undelivered_error_fact_before_native_action_registration(chip, monkeypatch):
    handle = chip.worker.submit(chip.callable)
    with monkeypatch.context() as patch:
        patch.setattr(chip.impl, "workspace_report", lambda: ("unavailable", None))
        with pytest.raises(RuntimeError, match="accounting could not be read"):
            chip.worker.close()
    assert handle.done and not chip.worker._chip_runs and not chip.worker._accepted_run_handles
    assert chip.runs[0].wait_count == 1
    assert not chip.impl.lane_closed and not chip.impl.finalized
    failure = RuntimeError("lane close failed after the workspace retry")

    def fail_close():
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(chip.impl, "_close_chip_run_lane", fail_close)
        with pytest.raises(RuntimeError) as caught:
            chip.worker.close()
        assert caught.value is failure
        assert not chip.impl.finalized
    chip.worker.close()
    assert chip.impl.finalized and chip.runs[0].wait_count == 1


def test_close_does_not_repeat_poison_already_delivered_by_wait(chip, monkeypatch):
    handle = chip.worker.submit(chip.callable)
    native = chip.runs[0]
    failure = RuntimeError("native run failed")

    def failed_wait(timeout):
        native.wait_count += 1
        chip.live.remove(native)
        raise failure

    def failed_close():
        raise failure

    monkeypatch.setattr(native, "wait", failed_wait)
    monkeypatch.setattr(chip.impl, "_close_chip_run_lane", failed_close)
    with pytest.raises(RuntimeError) as caught:
        handle.wait()
    assert caught.value is failure and chip.worker._source_context_failed
    assert not chip.worker._accepted_run_handles
    chip.worker.close()
    assert chip.impl.finalized
    with pytest.raises(RuntimeError) as caught:
        handle.result()
    assert caught.value is failure and native.wait_count == 1
