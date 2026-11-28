# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit predecessor-success fences include outer completion and reverse waits."""

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import pytest
from simpler.worker import ExecutionContextRefused, RunHandle, Worker


class FenceWorker:
    def __init__(self):
        self._orch = self
        self.allowed = {1: True}
        self.events = []
        self.outputs = {1: 0, 2: 0, 3: 0}
        self.native_error = None

    def _resolve_held_run(self, run_id, success):
        self.events.append(("resolve", run_id, success))
        self.allowed[run_id] = success

    def _wait_run_handle(self, run_id, timeout):
        del timeout
        if not self.allowed.get(run_id, False):
            raise RuntimeError("predecessor failed before launch")
        self.events.append(("launch", run_id))
        if run_id == 1 and self.native_error is not None:
            raise self.native_error
        self.outputs[run_id] = 1
        return True

    def _invoke_completion_callback(self, callback, error):
        callback(error)

    def _complete_outer_callback(self, handle):
        handle._cleanup_published = True

    def _finalize_run_handle(self, handle, run_id, error):
        self.events.append(("cleanup", run_id))
        handle._cleanup_published = True
        return error


def chain(worker):
    handles = [RunHandle(worker, i, ()) for i in (1, 2, 3)]
    for predecessor, successor in zip(handles, handles[1:]):
        Worker._attach_success_predecessor(worker, successor, predecessor)
    return handles


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, ExecutionContextRefused])
def test_finalized_distinguishes_interrupted_wait_from_terminal_failure(error_type):
    worker = FenceWorker()
    handle = RunHandle(cast(Worker, worker), 1, ())
    interrupt = error_type("interrupted native wait")
    worker.native_error = interrupt
    with pytest.raises(error_type):
        handle.result()
    assert not handle.finalized
    worker.native_error = RuntimeError("terminal native failure")
    with pytest.raises(RuntimeError, match="terminal native failure"):
        handle.result()
    assert handle.finalized


@pytest.mark.parametrize("outer_failure", [False, True])
def test_late_failure_cancels_three_run_chain_and_reverse_wait_preserves_outputs(outer_failure):
    worker = FenceWorker()
    handles = chain(worker)
    failure = RuntimeError("outer copyback failed" if outer_failure else "late native failure")
    callbacks = []
    if not outer_failure:
        worker.native_error = failure

    def completion(error):
        callbacks.append(error)
        if outer_failure:
            raise failure

    handles[0]._completion_callback = completion
    with pytest.raises(RuntimeError, match="predecessor failed"):
        handles[2].result()
    assert worker.outputs[2] == worker.outputs[3] == 0
    assert not any(event[0] == "launch" and event[1] in (2, 3) for event in worker.events)
    assert len(callbacks) == 1
    with pytest.raises(RuntimeError) as caught:
        handles[0].result()
    assert caught.value is failure
    assert len(callbacks) == 1
    assert [event for event in worker.events if event[0] == "cleanup"] == [
        ("cleanup", 1),
        ("cleanup", 2),
        ("cleanup", 3),
    ]


def test_success_callback_precedes_successor_activation_and_executes_once():
    worker = FenceWorker()
    handles = chain(worker)
    callback_entered = threading.Event()
    callback_release = threading.Event()
    callbacks = []

    def completion(error):
        assert error is None
        callbacks.append("outer completed")
        callback_entered.set()
        assert callback_release.wait(3)
        worker.events.append(("outer", 1))

    handles[0]._completion_callback = completion
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(handles[0].result)
        assert callback_entered.wait(3)
        assert not handles[0].finalized
        assert not worker.allowed.get(2, False)
        reverse = executor.submit(handles[2].result)
        callback_release.set()
        first.result(timeout=3)
        reverse.result(timeout=3)
    handles[2].result()
    assert all(handle.finalized for handle in handles)
    assert callbacks == ["outer completed"]
    assert worker.outputs == {1: 1, 2: 1, 3: 1}
    assert worker.events.index(("outer", 1)) < worker.events.index(("resolve", 2, True))


def test_interrupted_finalization_still_runs_outer_cleanup_once_and_cancels_successor():
    worker = FenceWorker()
    handles = chain(worker)
    original = worker._finalize_run_handle
    failure = RuntimeError("finalization boundary failed")
    callbacks = []

    def finalize(handle, run_id, error):
        if run_id == 1:
            raise failure
        return original(handle, run_id, error)

    worker._finalize_run_handle = finalize
    handles[0]._completion_callback = callbacks.append
    with pytest.raises(RuntimeError, match="predecessor failed"):
        handles[2].result()
    assert callbacks == [failure]
    assert worker.outputs[2] == worker.outputs[3] == 0
    with pytest.raises(RuntimeError) as caught:
        handles[0].result()
    assert caught.value is failure
    assert callbacks == [failure]


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_cached_callback_interrupt_allows_successor_to_drain(error_type):
    worker = FenceWorker()
    first, second, third = chain(worker)
    failure = error_type("outer callback interrupted")

    def completion(_error):
        raise failure

    first._completion_callback = completion
    with pytest.raises((RuntimeError, error_type)) as cancelled:
        third.result()
    assert isinstance(cancelled.value, RuntimeError)
    assert "predecessor failed" in str(cancelled.value)
    assert all(handle.finalized for handle in (first, second, third))
    assert [event for event in worker.events if event[0] == "cleanup"] == [
        ("cleanup", 1),
        ("cleanup", 2),
        ("cleanup", 3),
    ]
    with pytest.raises(error_type) as caught:
        first.result()
    assert caught.value is failure


def test_successor_snapshot_can_outlive_worker_close():
    worker = Worker(level=3, num_sub_workers=0)
    worker.init()
    first = worker.submit(lambda *_args: None)
    second = worker.submit(lambda *_args: None, success_predecessor=first)
    close_started = False

    class CloseOnSnapshot(list[int]):
        def __iter__(self):
            nonlocal close_started
            snapshot = tuple(super().__iter__())
            if snapshot and not close_started:
                close_started = True
                # Drain the successor and close the native tree after the original
                # waiter has captured its run ID.
                worker.close()
            return iter(snapshot)

    first._success_successors = CloseOnSnapshot(first._success_successors)
    try:
        first.result(timeout=3)
        assert close_started
        assert second.finalized
        assert worker._orch is None
        second.result(timeout=0)
    finally:
        worker.close()


def test_callback_cannot_wait_on_its_held_successor():
    with Worker(level=3, num_sub_workers=0) as worker:
        worker.init()
        refused = []

        def completion(_error):
            try:
                second.result(timeout=0)
            except RuntimeError as error:
                refused.append(str(error))

        first = worker.submit(lambda *_args: None, completion_callback=completion)
        second = worker.submit(lambda *_args: None, success_predecessor=first)
        second.result(timeout=3)
        assert refused == ["RunHandle.wait() cannot wait from this run's completion callback"]


def test_async_callback_is_rejected_before_graph_construction():
    with Worker(level=3, num_sub_workers=0) as worker:
        worker.init()
        built = []

        async def completion(_error):
            pass

        callback = cast("Callable[[BaseException | None], None]", completion)
        with pytest.raises(TypeError, match="completion_callback must be synchronous"):
            worker.submit(lambda *_args: built.append(True), completion_callback=callback)
        assert not built


def test_callback_returning_coroutine_cancels_successor():
    with Worker(level=3, num_sub_workers=0) as worker:
        worker.init()

        async def finish():
            pass

        coroutine = finish()
        callback = cast("Callable[[BaseException | None], None]", lambda _error: coroutine)
        try:
            first = worker.submit(lambda *_args: None, completion_callback=callback)
            second = worker.submit(lambda *_args: None, success_predecessor=first)
            with pytest.raises(RuntimeError, match="predecessor failed"):
                second.result(timeout=3)
            with pytest.raises(TypeError, match="completion_callback returned an awaitable"):
                first.result(timeout=0)
            assert coroutine.cr_frame is None
        finally:
            coroutine.close()


def test_graph_failure_calls_completion_outside_submit_lock():
    with Worker(level=3, num_sub_workers=1) as worker:
        worker.init()
        buffer = worker.create_buffer(64)
        graph_error = RuntimeError("graph build failed")
        callback_errors = []

        def graph(*_args):
            raise graph_error

        def completion(error):
            # Fail the test without hanging if the serializer is still held.
            assert not worker._submit_mu._exclusive
            worker.release_buffer(buffer)
            callback_errors.append(error)
            raise ValueError("completion failed too")

        with pytest.raises(RuntimeError) as failed:
            worker.submit(graph, completion_callback=completion)
        assert failed.value is graph_error
        (completion_error,) = callback_errors
        assert isinstance(completion_error, RuntimeError)
        assert "graph build failed" in str(completion_error)


@pytest.mark.parametrize("background_wait", [False, True])
def test_close_keeps_tree_until_outer_callback_finishes(monkeypatch, background_wait):
    worker = Worker(level=3, num_sub_workers=0)
    worker.init()
    entered = threading.Event()
    release = threading.Event()
    close_waiting = threading.Event()
    teardown = threading.Event()
    failures = []
    original_teardown = worker._teardown_worker_tree
    original_wait = RunHandle.wait

    def observed_teardown(*args, **kwargs):
        teardown.set()
        return original_teardown(*args, **kwargs)

    def completion(error):
        assert error is None
        entered.set()
        assert release.wait(5)

    handle = worker.submit(lambda *_args: None, completion_callback=completion)

    def observed_wait(self, *args, **kwargs):
        if self is handle and threading.current_thread() is threading.main_thread():
            close_waiting.set()
        return original_wait(self, *args, **kwargs)

    monkeypatch.setattr(worker, "_teardown_worker_tree", observed_teardown)
    monkeypatch.setattr(RunHandle, "wait", observed_wait)

    def release_after_close_waits():
        try:
            assert entered.wait(5)
            assert close_waiting.wait(5)
            assert handle in worker._accepted_run_handles
            assert not handle._cleanup_published
            assert not teardown.is_set()
        except BaseException as error:
            failures.append(error)
        finally:
            release.set()

    helper = threading.Thread(target=release_after_close_waits)
    waiter = threading.Thread(target=handle.result) if background_wait else None
    try:
        helper.start()
        if waiter is not None:
            waiter.start()
            assert entered.wait(5)
        worker.close()
        assert teardown.is_set()
        assert not worker._accepted_run_handles
    finally:
        release.set()
        helper.join(5)
        if waiter is not None:
            waiter.join(5)
        worker.close()
    assert not failures


@pytest.mark.parametrize("failed", [False, True])
def test_cached_predecessor_result_resolves_new_held_run(failed):
    worker = Worker(level=3, num_sub_workers=0)
    worker.init()

    def completion(error):
        assert error is None
        if failed:
            raise RuntimeError("outer failure")

    try:
        first = worker.submit(lambda *_args: None, completion_callback=completion)
        if failed:
            with pytest.raises(RuntimeError, match="outer failure"):
                first.result()
        else:
            first.result()
        assert first._run_id is None
        second = worker.submit(lambda *_args: None, success_predecessor=first)
        if failed:
            with pytest.raises(RuntimeError, match="predecessor failed"):
                second.result()
        else:
            second.result()
        assert worker.wait_any([first], timeout=0) is first
    finally:
        worker.close()


def test_close_cancels_empty_completion_wait_and_drains_its_native_lease(monkeypatch):
    worker = Worker(level=3, num_sub_workers=0)
    worker.init()
    entered = threading.Event()
    completed = threading.Event()
    failures = []
    orch = worker._orch
    assert orch is not None
    original = orch._wait_any_run

    def observed_wait(*args):
        entered.set()
        return original(*args)

    monkeypatch.setattr(orch, "_wait_any_run", observed_wait)

    def wait():
        try:
            assert worker.wait_any([]) is None
        except BaseException as error:
            failures.append(error)
        finally:
            completed.set()

    waiter = threading.Thread(target=wait)
    try:
        waiter.start()
        assert entered.wait(3)
        worker.close()
        assert completed.wait(3)
    finally:
        worker.cancel_wait_any()
        waiter.join(3)
        worker.close()
    assert not failures


def test_completion_callback_cannot_close_its_own_worker():
    worker = Worker(level=3, num_sub_workers=0)
    worker.init()
    try:
        handle = worker.submit(lambda *_args: None, completion_callback=lambda _error: worker.close())
        with pytest.raises(RuntimeError, match="within run finalization"):
            handle.result()
    finally:
        worker.close()


def test_completion_observer_can_drain_after_execution_admission_failure():
    worker = Worker(level=3, num_sub_workers=0)
    worker.init()
    try:
        worker._ordered_cleanup_error = RuntimeError("prior cleanup failed")
        # The observer admits no execution; it must still drain cancelled work.
        assert worker.wait_any([], timeout=0) is None
        with pytest.raises(RuntimeError, match="ordered cleanup failed"):
            worker.submit(lambda *_args: None)
    finally:
        worker._ordered_cleanup_error = None
        worker.close()
