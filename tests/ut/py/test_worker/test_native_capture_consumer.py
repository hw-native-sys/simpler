# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Actual consumer and public Worker facade; CPU tensors and framework/device leaves.

The recording graph and model notice are framework and RTS test leaves.
"""

import contextlib
import importlib.util
import json
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from simpler import worker as worker_module

from . import test_native_submission

chip_fixture = test_native_submission.chip_fixture
prepared = test_native_submission.prepared


@pytest.fixture
def consumer():
    path = Path(__file__).resolve().parents[4] / "examples/workers/l2/framework_eager/native_capture.py"
    spec = importlib.util.spec_from_file_location("cold_consumer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def scene(prepared, consumer, monkeypatch):
    monkeypatch.setenv("TASK_QUEUE_ENABLE", "0")
    chip, pending = prepared
    chip.state.completes = False
    events = []
    tensors = tuple(torch.empty((2, 3)) for _ in range(3))
    frame = {"tensors": tensors, "result": torch.empty((2, 3)), "args": None, "buffers": []}
    del tensors  # The fixture must not keep a second RHS Tensor root.
    faults = SimpleNamespace(
        eager=False,
        stale=False,
        reset=False,
        notice=True,
        end=False,
        notice_on_end=False,
        publication=False,
        discard=False,
        check=False,
        storage_only=False,
    )

    class Graph:
        recording = False
        replay_count = 0

        def __init__(self):
            self.actions = []
            self.notice = False

        def replay(self):
            self.replay_count += 1
            if not faults.stale or self.replay_count == 1:
                for action in self.actions:
                    action()

        def reset(self):
            events.append("reset")
            if faults.reset:
                raise RuntimeError("destroy unknown")
            self.actions.clear()
            self.notice |= faults.notice
            chip.state.completes = True

    graph = Graph()

    @contextlib.contextmanager
    def capture(actual, **kwargs):
        assert actual is graph
        graph.recording = True
        try:
            yield
        finally:
            graph.recording = False
            if faults.end:
                graph.notice |= faults.notice_on_end
                raise RuntimeError("capture End failed")

    def add(a, b, *, out):
        def action():
            torch.add(a, b, out=out)

        if graph.recording:
            graph.actions.append(action)
        else:
            action()

    prepare = chip.worker.prepare_submission

    def prepare_with_model_leaves(*args, keepalive, **kwargs):
        if faults.storage_only:
            keepalive = tuple(owner for owner in keepalive if not isinstance(owner, torch.Tensor))
        ticket = prepare(*args, keepalive=keepalive, **kwargs)
        ticket_ref = weakref.ref(ticket)
        token = ticket._native_submission
        capsule = token.capsule

        def retain_model():
            assert graph.recording
            rooted_ticket = ticket_ref()
            assert rooted_ticket is not None
            _, owners = worker_module._native_submission_roots[rooted_ticket]
            if not faults.storage_only:
                assert all(any(owner is tensor for owner in owners) for tensor in frame["tensors"])
            events.append("retain-model")

        def export():
            assert events[-1] == "retain-model"
            events.append("export")
            return capsule()

        def release_model():
            if not graph.notice:
                raise RuntimeError("ACL model destroy notice missing")
            events.append("release-model")
            chip.state.completes = True
            return token.cancel()

        token._retain_model_resources = retain_model
        token._model_destroyed = lambda: graph.notice
        token._release_model_resources = release_model
        token.capsule = export
        return ticket

    monkeypatch.setattr(chip.worker, "prepare_submission", prepare_with_model_leaves)

    def enqueue(token, operands):
        assert graph.recording and token is pending[-1]
        assert operands is frame["tensors"]
        assert events[-1] == "export"
        events.append("enqueue")
        if not faults.discard:
            token.submit()
            # Replay keeps storage only, so the fixture cannot mask a missing
            # TensorImpl root on the real Worker ticket.
            lhs, rhs, output = operands
            storage = rhs.untyped_storage()
            dtype, shape, stride, offset = rhs.dtype, rhs.shape, rhs.stride(), rhs.storage_offset()

            def action():
                alias = torch.empty(0, dtype=dtype).set_(storage, offset, shape, stride)
                torch.add(lhs, alias, out=output)

            if faults.eager:
                action()
            graph.actions.append(action)
        if faults.publication:
            raise RuntimeError("publication unknown after callback")

        def check():
            assert graph.recording
            if faults.discard:
                raise RuntimeError("native callback is not complete")
            if faults.check:
                raise RuntimeError("queued callback failed")

        return SimpleNamespace(check=check)

    class Torch(SimpleNamespace):
        def __getattr__(self, name):
            return getattr(torch, name)

    proxy = Torch(
        _C=torch._C,
        add=add,
        full=torch.full,
        testing=torch.testing,
        npu=SimpleNamespace(graph=capture, stream=lambda stream: contextlib.nullcontext()),
    )
    slot = [None]
    stream = SimpleNamespace(npu_stream=123, synchronize=lambda: events.append("synchronize"))
    bridge = SimpleNamespace(enqueue_native=enqueue)
    scene = SimpleNamespace(
        chip=chip,
        pending=pending,
        frame=frame,
        graph=graph,
        torch=proxy,
        bridge=bridge,
        slot=slot,
        stream=stream,
        faults=faults,
        events=events,
        exercise=lambda: consumer._exercise(
            proxy, chip.worker, bridge, chip.callable, None, stream, graph, frame, slot
        ),
        close=lambda: consumer._close(chip.worker, graph, slot, [frame]),
    )
    try:
        yield scene
    finally:
        # Settle only this CPU transport even when a negative control aborts
        # the consumer before reset. Shared fixtures then retire the real Worker.
        chip.state.completes = True


def test_cold_consumer_uses_public_prepare_and_checked_retirement(scene):
    assert scene.exercise() == [10, 20, 44]
    assert len(scene.pending) == 1 and len(scene.chip.runs) == 1
    ticket = scene.slot[0]
    assert ticket in worker_module._native_submission_roots and not ticket.safe_to_release
    assert not scene.chip.impl.finalized
    assert not scene.frame["rhs_probe"].expired()
    scene.close()
    assert scene.frame["rhs_probe"].expired()
    assert scene.pending[0].retired and ticket.safe_to_release
    assert scene.chip.impl.finalized
    assert scene.events[-2:] == ["reset", "release-model"]


@pytest.mark.parametrize("fault", ["eager", "stale"])
def test_consumer_detects_compute_during_record_and_stale_replay(scene, fault):
    setattr(scene.faults, fault, True)
    with pytest.raises(AssertionError):
        scene.exercise()
    assert scene.slot[0] in worker_module._native_submission_roots
    scene.close()
    assert scene.slot[0].safe_to_release


@pytest.mark.parametrize("fault", ["publication", "check"])
def test_failed_queue_path_keeps_actual_ticket_until_reset(scene, fault):
    setattr(scene.faults, fault, True)
    with pytest.raises(RuntimeError):
        scene.exercise()
    ticket = scene.slot[0]
    assert ticket is not None and not ticket.safe_to_release
    assert ticket in worker_module._native_submission_roots
    assert scene.pending[0].ready.is_set()
    scene.close()
    assert ticket.safe_to_release and scene.pending[0].retired


def test_failed_destroy_does_not_close_worker_or_retire_original_ticket(scene):
    scene.exercise()
    scene.faults.reset = True
    with pytest.raises(RuntimeError, match="destroy unknown"):
        scene.close()
    ticket = scene.slot[0]
    assert not ticket.safe_to_release and not scene.pending[0].retired
    assert ticket in worker_module._native_submission_roots
    assert not scene.chip.impl.finalized
    assert not scene.frame["rhs_probe"].expired()
    scene.faults.reset = False
    scene.close()
    assert ticket.safe_to_release and scene.pending[0].retired
    assert scene.frame["rhs_probe"].expired()


def test_reset_return_without_model_notice_keeps_ticket_and_tensor(scene):
    scene.exercise()
    scene.faults.notice = False
    with pytest.raises(RuntimeError, match="ACL model destroy notice missing"):
        scene.close()
    assert not scene.slot[0].safe_to_release and not scene.frame["rhs_probe"].expired()
    assert not scene.chip.impl.finalized
    scene.faults.notice = True
    scene.close()
    assert scene.slot[0].safe_to_release and scene.frame["rhs_probe"].expired()


def test_discarded_callback_cannot_replay_and_original_ticket_is_cancelled(scene):
    scene.faults.discard = True
    with pytest.raises(RuntimeError, match="native callback is not complete"):
        scene.exercise()
    assert scene.graph.replay_count == 0
    assert not scene.pending[0].ready.is_set()
    scene.close()
    assert scene.pending[0].retired and scene.slot[0].safe_to_release


def test_end_notice_allows_checked_release_even_when_reset_raises(scene):
    scene.faults.end = scene.faults.notice_on_end = True
    scene.faults.check = True
    with pytest.raises(RuntimeError, match="queued callback failed") as caught:
        scene.exercise()
    assert str(caught.value.__cause__) == "capture End failed"
    assert scene.graph.replay_count == 0
    scene.faults.reset = True
    with pytest.raises(RuntimeError, match="destroy unknown"):
        scene.close()
    assert scene.pending[0].retired and scene.slot[0].safe_to_release
    assert scene.chip.impl.finalized


def test_retired_ticket_error_still_closes_worker_and_buffers(scene):
    scene.exercise()
    primary = RuntimeError("original native error")
    token = scene.pending[0]

    def wait_with_retired_error(timeout=-1):
        assert token.run.wait(timeout)
        raise primary

    token.wait = wait_with_retired_error
    token.recovery_required = False
    scene.frame["buffers"].append(SimpleNamespace(close=lambda: scene.events.append("buffer-closed")))
    with pytest.raises(RuntimeError, match="original native error") as caught:
        scene.close()
    assert caught.value is primary
    assert scene.slot[0].safe_to_release
    assert not scene.chip.live
    assert scene.chip.impl.finalized and scene.events[-1] == "buffer-closed", caught.value.__cause__


@pytest.fixture
def run_scene(scene, consumer, monkeypatch):
    # Only framework/device leaves are replaced. prepare, tracking, cancel,
    # completion and close still run the real public Worker's methods.
    class ExistingWorker:
        def register(self, _):
            return scene.chip.callable

        def init(self, **kwargs):
            pass

        def __getattr__(self, name):
            return getattr(scene.chip.worker, name)

    scene.torch.npu.set_device = lambda device: None
    scene.torch.npu.Stream = lambda **kwargs: scene.stream
    scene.torch.npu.NPUGraph = lambda: scene.graph
    monkeypatch.setitem(sys.modules, "torch", scene.torch)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(worker_module, "Worker", lambda **kwargs: ExistingWorker())
    monkeypatch.setattr(
        consumer,
        "_helpers",
        lambda: SimpleNamespace(
            _load_bridge=lambda path: scene.bridge,
            _compile=lambda *args: None,
            _prepare_frames=lambda *args: ([scene.frame], []),
        ),
    )
    return lambda: consumer.run("host_build_graph", 0, Path("not-loaded"))


@pytest.mark.parametrize("reset_fails", [False, True])
def test_run_preserves_publication_error_and_checked_cleanup_order(scene, run_scene, reset_fails, consumer):
    scene.faults.publication = True
    scene.faults.reset = reset_fails
    try:
        with pytest.raises(RuntimeError, match="publication unknown after callback") as caught:
            run_scene()
        if reset_fails:
            assert str(caught.value.__cause__) == "destroy unknown"
            assert not scene.chip.impl.finalized
            assert not scene.pending[0].retired
            assert any(handle._worker is scene.chip.worker for handle in worker_module._native_submission_roots)
        else:
            assert caught.value.__cause__ is None
            assert scene.chip.impl.finalized and scene.pending[0].retired
    finally:
        if reset_fails:
            scene.faults.reset = False
            # Recover only this CPU leaf after proving unknown cleanup retained
            # the real Worker's roots. No SDK state exists in this test.
            ticket = next(
                handle for handle in worker_module._native_submission_roots if handle._worker is scene.chip.worker
            )
            consumer._close(scene.chip.worker, scene.graph, [ticket], [scene.frame])


def test_storage_only_rhs_cannot_qualify_tensor_lifetime(scene):
    scene.faults.storage_only = True
    try:
        with pytest.raises(AssertionError, match="TensorImpl"):
            scene.exercise()
    finally:
        scene.close()


def test_storage_only_rhs_still_produces_correct_replays_without_qualification(scene, monkeypatch):
    monkeypatch.setenv("TASK_QUEUE_ENABLE", "1")
    scene.faults.storage_only = True
    try:
        assert scene.exercise() == [10, 20, 44]
        assert scene.frame["rhs_probe"].expired()
        torch.testing.assert_close(scene.frame["result"], torch.full((2, 3), 44.0), rtol=0, atol=0)
    finally:
        scene.close()


@pytest.fixture
def driver():
    path = Path(__file__).resolve().parents[3] / "st/worker/framework_eager/native_capture.py"
    spec = importlib.util.spec_from_file_location("cold_capture_driver", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("task_queue", ["0", "1"])
def test_actual_run_and_driver_report_only_the_selected_lifetime_qualification(
    scene, run_scene, driver, monkeypatch, capsys, task_queue
):
    monkeypatch.setenv("TASK_QUEUE_ENABLE", task_queue)
    run_scene()
    reports = [
        json.loads(line.removeprefix("NATIVE_COLD_CAPTURE_PASS "))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("NATIVE_COLD_CAPTURE_PASS ")
    ]
    assert len(reports) == 1
    assert reports[0]["task_queue"] == task_queue
    assert reports[0]["tensor_impl_lifetime"] == ("observed" if task_queue == "0" else "not_observed")
    driver._check_report(reports, "host_build_graph", "a2a3", task_queue)
    reports[0]["tensor_impl_lifetime"] = "not_observed" if task_queue == "0" else "observed"
    with pytest.raises(AssertionError):
        driver._check_report(reports, "host_build_graph", "a2a3", task_queue)
    assert scene.pending[0].retired


def test_unsupported_queue_mode_is_rejected_before_backend_import(consumer, monkeypatch):
    monkeypatch.setenv("TASK_QUEUE_ENABLE", "2")
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    with pytest.raises(ValueError, match="TASK_QUEUE_ENABLE=0 or 1"):
        consumer.run("host_build_graph", 0, Path("not-loaded"))


def test_retained_capture_driver_checks_actual_replays_and_independent_release(driver):
    report = {
        "runtime": "host_build_graph",
        "platform": "a2a3",
        "task_queue": "0",
        "grants": 2,
        "models": 3,
        "record_computed": False,
        "replays": [[10, 20, 44], [212, 222, 246], [414, 424, 448]],
        "release_order": [2, 1, 0],
        "survivor_replays": [1010, 1008],
        "checked_close": True,
        "tensor_impl_lifetime": "observed",
    }
    driver._check_retained_report([report], "host_build_graph", "0")
    for invalid in (
        {**report, "models": 2},
        {**report, "record_computed": True},
        {**report, "replays": [[10, 20, 44], [212, 222, 246], [414, 424, 446]]},
        {**report, "release_order": [0, 1, 2]},
        {**report, "survivor_replays": [1010, 1010]},
        {**report, "tensor_impl_lifetime": "not_observed"},
    ):
        with pytest.raises(AssertionError):
            driver._check_retained_report([invalid], "host_build_graph", "0")
