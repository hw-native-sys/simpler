# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Actual lifecycle routing/receipt controls; device execution belongs to the ST."""

import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_CONSUMER = _ROOT / "examples/workers/l2/framework_eager/lifecycle.py"
_ST = _ROOT / "tests/st/worker/framework_eager/test_lifecycle.py"
_TMR = "tensormap_and_ringbuffer"
_HBG = "host_build_graph"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def consumer(monkeypatch):
    module = _load(_CONSUMER, "framework_lifecycle_consumer")
    events, workers = [], []

    class Tensor:
        def fill_(self, value):
            return self

        def copy_(self, value):
            return self

        def add_(self, value):
            return self

        def cpu(self):
            return self

        def item(self):
            return 17

    class Worker:
        def __init__(self, *, runtime, **kwargs):
            self.runtime = runtime
            self.backend = runtime
            self.handles = {}
            self.closed = False
            self._chip_worker = SimpleNamespace(
                _execution_environment=(len(workers),), _validate_execution_context=lambda: None
            )
            workers.append(self)
            events.append(("create", runtime))

        def register(self, spec):
            events.append(("register", self.runtime, spec))
            if spec[0] != self.runtime:
                raise ValueError("foreign runtime callable")
            handle = object()
            self.handles[handle] = len(self.handles)
            return handle

        def init(self, **kwargs):
            assert all(worker.closed for worker in workers[:-1])
            events.append(("init", self.runtime))

        def _resolve_handle(self, handle):
            return SimpleNamespace(slot_id=self.handles[handle])

        @property
        def host_dlopen_count(self):
            assert not self.closed
            events.append(("host_count", self.runtime))
            return 2 if self.backend == _HBG else 0

        @property
        def aicpu_dlopen_count(self):
            assert not self.closed
            events.append(("aicpu_count", self.runtime))
            return 2 if self.backend == _TMR else 0

        def close(self):
            self.closed = True
            events.append(("close", self.runtime))

    stream = SimpleNamespace(npu_stream=23)

    class StreamContext:
        def __enter__(self):
            return stream

        def __exit__(self, *args):
            return False

    torch = SimpleNamespace(
        npu=SimpleNamespace(
            set_device=lambda device: None,
            Stream=lambda **kwargs: stream,
            stream=lambda supplied: StreamContext(),
            current_stream=lambda **kwargs: stream,
        ),
        full=lambda *args, **kwargs: Tensor(),
        add=lambda *args, **kwargs: None,
    )

    def enqueue(callback):
        callback()
        return SimpleNamespace(thread_id=threading.get_native_id(), check=lambda: None)

    bridge = SimpleNamespace(enqueue=enqueue, finish_current_stream=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "simpler.worker", SimpleNamespace(Worker=Worker))
    monkeypatch.setenv("TASK_QUEUE_ENABLE", "0")
    monkeypatch.setattr(module, "_helpers", lambda: SimpleNamespace(_load_bridge=lambda path: bridge))
    monkeypatch.setattr(module, "_current_context", lambda: 31)
    monkeypatch.setattr(module, "_compile", lambda runtime, name, **kwargs: (runtime, name))

    def frame(torch, worker, stream, index):
        return dict(
            index=index,
            backings=[Tensor() for _ in range(3)],
            tensors=[Tensor() for _ in range(3)],
            result=Tensor(),
            buffers=[SimpleNamespace(close=lambda: None)],
        )

    def submit(worker, handle, frame, state):
        frame.update(callback_context=31, run=SimpleNamespace(wait=lambda: None))
        events.append(("submit", worker.runtime))

    # Only frame/device/queue transport is recorded here. The actual consumer
    # owns generation selection, registration, handle checks, close order,
    # counter reads and receipt construction. This is no numerical CPU oracle.
    monkeypatch.setattr(module, "_frame", frame)
    monkeypatch.setattr(module, "_submit", submit)
    monkeypatch.setattr(module, "_verify", lambda torch, frame: module.EXPECTED[frame["index"]])
    return module, events, workers


@pytest.mark.parametrize("first,second", [(_TMR, _TMR), (_HBG, _HBG), (_TMR, _HBG), (_HBG, _TMR)])
def test_actual_consumer_selects_generation_runtime_specs_and_counts(consumer, first, second):
    module, events, workers = consumer
    summary = module.run(first, 0, Path("recording.so"), successor_runtime=second)
    assert [worker.runtime for worker in workers] == [first, second]
    assert [(event[1], event[2]) for event in events if event[0] == "register"] == [
        (first, (first, "x")),
        (first, (first, "y")),
        (second, (second, "y")),
        (second, (second, "x")),
    ]
    assert summary["runtimes"] == [first, second]
    assert summary["orchestration_load_counts"] == [
        [2, 0] if runtime == _HBG else [0, 2] for runtime in (first, second)
    ]
    first_close = events.index(("close", first))
    assert first_close < [index for index, event in enumerate(events) if event[0] == "init"][1]
    for worker in workers:
        assert worker.closed
    assert [event[0] for event in events].count("submit") == 8


def test_actual_consumer_rejects_wrong_second_backend(consumer, monkeypatch):
    module, _, workers = consumer
    original_submit = module._submit

    def wrong_backend(worker, handle, frame, state):
        if len(workers) == 2:
            worker.backend = _TMR
        original_submit(worker, handle, frame, state)

    monkeypatch.setattr(module, "_submit", wrong_backend)
    with pytest.raises(AssertionError):
        module.run(_TMR, 0, Path("recording.so"), successor_runtime=_HBG)
    assert workers[0].closed and not workers[1].closed
    assert module._UNPROVEN


@pytest.mark.parametrize("generation", [0, 1])
def test_actual_consumer_checks_backend_counts_in_default_same_runtime_path(consumer, monkeypatch, generation):
    module, _, workers = consumer
    original_submit = module._submit

    def wrong_backend(worker, handle, frame, state):
        if len(workers) == generation + 1:
            worker.backend = _HBG
        original_submit(worker, handle, frame, state)

    monkeypatch.setattr(module, "_submit", wrong_backend)
    with pytest.raises(AssertionError):
        module.run(_TMR, 0, Path("recording.so"))
    assert len(workers) == generation + 1 and not workers[-1].closed
    assert module._UNPROVEN


def test_actual_consumer_does_not_accept_predecessor_runtime_specs(consumer, monkeypatch):
    module, events, workers = consumer
    monkeypatch.setattr(module, "_compile", lambda runtime, name, **kwargs: (_TMR, name))
    with pytest.raises(ValueError, match="foreign runtime callable"):
        module.run(_TMR, 0, Path("recording.so"), successor_runtime=_HBG)
    assert workers[0].closed and not workers[1].closed
    assert ("init", _HBG) not in events
    assert module._UNPROVEN


@pytest.mark.parametrize("successor", [None, _TMR, _HBG])
def test_actual_cli_routes_successor_without_changing_default(consumer, monkeypatch, tmp_path, successor):
    module, _, _ = consumer
    (tmp_path / "_simpler_framework_queue_cpu.so").touch()
    calls = []
    monkeypatch.setattr(module, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    argv = [str(_CONSUMER), "--runtime", _TMR, "--build-dir", str(tmp_path)]
    if successor is not None:
        argv += ["--successor-runtime", successor]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    assert len(calls) == 1
    assert calls[0][0][:2] == (_TMR, 0)
    assert calls[0][1] == dict(platform="a2a3", successor_runtime=successor)


def _summary(first, second, queue) -> dict:
    return dict(
        verified=True,
        platform="a5",
        runtime=first,
        runtimes=[first, second],
        task_queue=queue,
        calls=8,
        generations=2,
        observed=[3, 7, 13, 17, 22, 34, 41, 57],
        expected=[3, 7, 13, 17, 22, 34, 41, 57],
        generation_calls=[4, 4],
        orchestration_load_counts=[[2, 0] if runtime == _HBG else [0, 2] for runtime in (first, second)],
        original_handles_preserved=True,
        guards_verified=True,
        caller_tensors_retained=True,
        caller_stream_after_each_close=True,
        caller_context=31,
        callback_contexts=[31] * 8,
        caller_thread=1,
        callback_threads=[1 if queue == "0" else 2] * 8,
    )


def _invoke_st(tmp_path, monkeypatch, summary, first, second, queue):
    module = _load(_ST, "framework_lifecycle_st")

    def run(command, **kwargs):
        assert Path(command[2]) == _CONSUMER
        assert command[command.index("--runtime") + 1] == first
        assert kwargs["env"]["TASK_QUEUE_ENABLE"] == queue
        if first != second:
            assert command[command.index("--successor-runtime") + 1] == second
        else:
            assert "--successor-runtime" not in command
        return SimpleNamespace(returncode=0, stdout="FRAMEWORK_LIFECYCLE_PASS " + json.dumps(summary), stderr="")

    monkeypatch.setattr(module, "subprocess", SimpleNamespace(run=run))
    request = SimpleNamespace(
        config=SimpleNamespace(getoption=lambda name: {"--device": "0", "--platform": "a5"}[name])
    )
    if first == second:
        module.test_framework_worker_close_recreate(first, queue, tmp_path, tmp_path, request)
    else:
        module.test_framework_worker_runtime_transition(first, queue, tmp_path, tmp_path, request)


@pytest.mark.parametrize("first", [_TMR, _HBG])
@pytest.mark.parametrize("queue", ["0", "1"])
@pytest.mark.parametrize("cross", [False, True])
def test_actual_st_accepts_same_and_cross_runtime_evidence(tmp_path, monkeypatch, first, queue, cross):
    second = (_HBG if first == _TMR else _TMR) if cross else first
    _invoke_st(tmp_path, monkeypatch, _summary(first, second, queue), first, second, queue)


@pytest.mark.parametrize("cross", [False, True])
@pytest.mark.parametrize(
    "corrupt", ["missing_pair", "wrong_pair", "missing_counts", "wrong_first", "wrong_second", "unregistered"]
)
def test_actual_st_rejects_wrong_backend_evidence_even_with_correct_arithmetic(tmp_path, monkeypatch, corrupt, cross):
    second = _HBG if cross else _TMR
    summary = _summary(_TMR, second, "1")
    if corrupt == "missing_pair":
        del summary["runtimes"]
    elif corrupt == "wrong_pair":
        summary["runtimes"] = [_TMR, _TMR if cross else _HBG]
    elif corrupt == "missing_counts":
        del summary["orchestration_load_counts"]
    elif corrupt == "wrong_first":
        summary["orchestration_load_counts"][0] = [2, 0]
    elif corrupt == "wrong_second":
        summary["orchestration_load_counts"][1] = [0, 2] if cross else [2, 0]
    else:
        summary["orchestration_load_counts"][1] = [0, 0]
    with pytest.raises((AssertionError, KeyError)):
        _invoke_st(tmp_path, monkeypatch, summary, _TMR, second, "1")
