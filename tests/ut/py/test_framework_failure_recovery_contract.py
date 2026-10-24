# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host failure bookkeeping and the real ST's summary rejection contract.

Submission is a recording boundary here, not an emulated executor. The complete
consumer runs with real CPU tensors and synchronous recording boundaries. Torch
emission, device completion and native teardown require the hardware ST.
"""

import gc
import importlib.util
import json
import sys
import threading
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_CONSUMER = _ROOT / "examples/workers/l2/framework_eager/failure_recovery.py"
_ST = _ROOT / "tests/st/a2a3/framework_eager/test_failure_recovery.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def consumer(monkeypatch):
    reference = _load(_CONSUMER.with_name("main.py"), "framework_recovery_reference")
    monkeypatch.setitem(sys.modules, "main", reference)
    return _load(_CONSUMER, "framework_failure_recovery")


def test_callback_exception_keeps_accepted_run_and_allows_later_request(consumer):
    class Run:
        def wait(self):
            pytest.fail("a callback must not wait for its accepted run")

    accepted = []

    def submit(handle, args, *, caller_stream):
        assert (handle, args, caller_stream) == ("callable", "arguments", 17)
        run = Run()
        accepted.append(weakref.ref(run))
        return run

    frame = dict(worker=SimpleNamespace(submit=submit), callable="callable", args="arguments", raw_stream=17, index=0)
    runs, submissions = [], []
    with pytest.raises(RuntimeError, match="callback failed after Worker.submit accepted"):
        consumer._submit_frame(frame, runs, submissions)
    gc.collect()
    assert len(runs) == 1 and accepted[0]() is runs[0] and submissions == [0]
    assert frame["callback_thread"] == threading.get_native_id()
    frame = {**frame, "index": 1}
    consumer._submit_frame(frame, runs, submissions)
    assert len(runs) == 2 and submissions == [0, 1]


def test_submit_exception_does_not_publish_an_accepted_run(consumer):
    error = ValueError("request refused before acceptance")

    def submit(*args, **kwargs):
        raise error

    frame = dict(worker=SimpleNamespace(submit=submit), callable=None, args=None, raw_stream=17, index=0)
    runs, submissions = [], []
    with pytest.raises(ValueError) as caught:
        consumer._submit_frame(frame, runs, submissions)
    assert caught.value is error
    assert runs == submissions == []


@pytest.fixture
def recording_transport(consumer, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    monkeypatch.setenv("TASK_QUEUE_ENABLE", "0")
    import torch  # noqa: PLC0415 -- disable backend autoload before importing the CPU package

    events, run_references, workers = [], [], []
    stream = SimpleNamespace(npu_stream=17)
    boundary_error = RuntimeError("framework completion is unknown")
    state = SimpleNamespace(fail_boundary=False)

    class Run:
        def wait(self):
            pytest.fail("the full consumer must retire runs through Worker.close")

    class Worker:
        closed = False

        def __init__(self, **kwargs):
            workers.append(self)
            self.owner = threading.get_ident()

        def register(self, spec):
            return "callable"

        def init(self, **kwargs):
            assert kwargs == {"borrowed_context_generation": 1}

        def submit(self, handle, args, *, caller_stream):
            assert handle == "callable" and caller_stream == 17 and not self.closed
            events.append("submit")
            # This is only synchronous CPU arithmetic over the example's three
            # tensors; no device lane, event ordering or teardown is emulated.
            torch.add(args[0], args[1], out=args[2])
            run = Run()
            run_references.append(weakref.ref(run))
            return run

        def close(self):
            assert threading.get_ident() == self.owner
            assert "framework boundary" in events
            assert len(run_references) == 3 and all(ref() is not None for ref in run_references)
            events.append("worker close")
            self.closed = True

    class Buffer:
        closed = False

        def close(self):
            assert workers[0].closed
            events.append("buffer close")
            self.closed = True

    def prepare(torch, worker, handle, actual_stream, raw_stream, calls):
        assert actual_stream is stream and raw_stream == 17 and calls == 3
        frames, references = [], []
        for index in range(calls):
            tensors = tuple(torch.empty(consumer.reference.SHAPE) for _ in range(3))
            references.extend(weakref.ref(tensor) for tensor in tensors)
            frames.append(
                dict(
                    index=index,
                    tensors=tensors,
                    args=tensors,
                    buffers=[Buffer() for _ in tensors],
                    worker=worker,
                    callable=handle,
                    raw_stream=raw_stream,
                    result=torch.empty(consumer.reference.SHAPE),
                )
            )
        return frames, references

    def enqueue(callback):
        try:
            callback()
        except RuntimeError as error:
            failure = error
        else:
            failure = None

        def check():
            if failure is not None:
                raise failure

        return SimpleNamespace(check=check, thread_id=threading.get_native_id())

    def finish():
        events.append("framework boundary")
        if state.fail_boundary:
            raise boundary_error

    bridge = SimpleNamespace(enqueue=enqueue, finish_current_stream=finish)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            set_device=lambda device: None, Stream=lambda **kwargs: stream, stream=lambda actual: nullcontext()
        ),
        raising=False,
    )
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "simpler.worker", SimpleNamespace(Worker=Worker))
    monkeypatch.setattr(consumer.reference, "_compile", lambda runtime: "compiled")
    monkeypatch.setattr(consumer.reference, "_load_bridge", lambda path: bridge)
    monkeypatch.setattr(consumer.reference, "_prepare_frames", prepare)
    return SimpleNamespace(
        events=events, workers=workers, run_references=run_references, state=state, boundary_error=boundary_error
    )


def test_full_consumer_recovers_and_closes_unwaited_accepted_runs(consumer, recording_transport):
    summary = consumer.run("host_build_graph", 0, Path("recording-transport.so"))
    assert summary["observed"] == [8, 12, 16] and summary["callback_errors"] == [0]
    assert summary["submissions"] == [0, 1, 2]
    events = recording_transport.events
    assert events[:4] == ["submit"] * 3 + ["framework boundary"]
    assert events[4:] == ["worker close"] + ["buffer close"] * 9 + ["worker close", "framework boundary"]
    assert not consumer._UNPROVEN


def test_full_consumer_unknown_completion_retains_owners_and_primary_error(consumer, recording_transport, monkeypatch):
    recording_transport.state.fail_boundary = True

    def stderr_unavailable(*args):
        raise OSError("stderr is closed")

    monkeypatch.setattr(consumer.os, "write", stderr_unavailable)
    with pytest.raises(RuntimeError) as caught:
        consumer.run("tensormap_and_ringbuffer", 0, Path("recording-transport.so"))
    assert caught.value is recording_transport.boundary_error
    gc.collect()
    assert len(consumer._UNPROVEN) == 1
    worker, handle, frames, queued, runs, stream = consumer._UNPROVEN[0]
    assert worker is recording_transport.workers[0] and not worker.closed
    assert handle == "callable" and stream.npu_stream == 17
    assert len(frames) == len(queued) == len(runs) == 3
    assert all(not buffer.closed for frame in frames for buffer in frame["buffers"])
    assert all(ref() is run for ref, run in zip(recording_transport.run_references, runs))
    assert recording_transport.events == ["submit"] * 3 + ["framework boundary"]


@pytest.mark.parametrize("defect", [None, "lost_error", "missing_successor", "wrong_result", "early_buffer_release"])
def test_real_hardware_test_rejects_incomplete_recovery_summary(tmp_path, monkeypatch, defect):
    st = _load(_ST, "framework_failure_recovery_st")
    summary = dict(
        runtime="host_build_graph",
        task_queue="1",
        verified=True,
        calls=3,
        submissions=[0, 1, 2],
        callback_errors=[0],
        observed=[8, 12, 16],
        caller_tensors_retained=True,
        close_without_individual_wait=True,
        buffers_closed_after_worker=True,
        caller_stream_after_close=True,
        caller_thread=1,
        callback_threads=[2, 2, 2],
    )
    if defect == "lost_error":
        summary["callback_errors"] = []
    elif defect == "missing_successor":
        summary["submissions"] = [0, 1]
    elif defect == "wrong_result":
        summary["observed"] = [8, 12, -1998]
    elif defect == "early_buffer_release":
        summary["buffers_closed_after_worker"] = False
    result = SimpleNamespace(returncode=0, stdout="FRAMEWORK_FAILURE_RECOVERY_PASS " + json.dumps(summary), stderr="")
    monkeypatch.setattr(st, "subprocess", SimpleNamespace(run=lambda *args, **kwargs: result))
    request = SimpleNamespace(config=SimpleNamespace(getoption=lambda name: "0"))
    if defect is None:
        st.test_framework_recovers_after_accepted_callback_failure("host_build_graph", "1", tmp_path, tmp_path, request)
    else:
        with pytest.raises(AssertionError):
            st.test_framework_recovers_after_accepted_callback_failure(
                "host_build_graph", "1", tmp_path, tmp_path, request
            )
