# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real Worker provenance and recording-only context/consumer contracts.

RTS context calls, native arithmetic/completion and framework emission are private
recording boundaries. No test here qualifies a real RTS context or device task.
"""

import ctypes
import importlib.util
import json
import os
import sys
import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from simpler.task_interface import ExecutionContextRefused
from simpler.worker import Worker, _Lifecycle

_ROOT = Path(__file__).resolve().parents[3]
_CONSUMER = _ROOT / "examples/workers/l2/framework_eager/preflight_recovery.py"
_ST = _ROOT / "tests/st/worker/framework_eager/test_preflight_recovery.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def consumer():
    return _load(_CONSUMER, "framework_preflight_consumer")


class _ContextApi:
    def __init__(self):
        self.current = 101
        self.events = []
        self.create_error = self.destroy_error = 0
        self.fail_restore = self.ignore_restore = False
        self.query_failures = 0

    def aclrtGetCurrentContext(self, output):
        self.events.append(("get", self.current))
        if self.query_failures:
            self.query_failures -= 1
            return -7
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = self.current
        return 0

    def aclrtCreateContext(self, output, device):
        self.events.append(("create", device))
        self.current = 202
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = 202
        return self.create_error

    def aclrtSetCurrentContext(self, context):
        value = context.value if isinstance(context, ctypes.c_void_p) else context
        self.events.append(("set", value))
        if value == 101 and self.fail_restore:
            return -8
        if value != 101 or not self.ignore_restore:
            self.current = value
        return 0

    def aclrtDestroyContext(self, context):
        value = context.value
        assert self.current == 101 and value == 202
        self.events.append(("destroy", value))
        self.current = None
        return self.destroy_error


def _refusal_frame(api, *, error=None, accepted=None):
    def submit(*args, **kwargs):
        assert api.current == 202
        if accepted is not None:
            return accepted
        raise error if error is not None else ExecutionContextRefused("context mismatch")

    return dict(worker=SimpleNamespace(submit=submit), callable=object(), args=object(), raw_stream=17, index=1)


def test_context_capsule_restores_after_create_refusal_and_destroys_only_alternate(consumer):
    api = _ContextApi()
    contexts = consumer._CallerContexts(api, 0)
    contexts.create()
    assert api.current == contexts.base == 101 and contexts.alternate.value == 202
    runs = []
    contexts.refuse(_refusal_frame(api), runs)
    assert api.current == 101 and contexts.restored and not runs
    contexts.close()
    assert api.current == 101 and contexts.destroyed
    assert [event for event in api.events if event[0] == "destroy"] == [("destroy", 202)]


@pytest.mark.parametrize("failure", ["create", "restore_error", "restore_lie", "destroy"])
def test_context_lifecycle_failure_does_not_publish_success_or_destroy_unknown_context(consumer, failure):
    api = _ContextApi()
    contexts = consumer._CallerContexts(api, 0)
    if failure == "create":
        api.create_error = -9
        with pytest.raises(RuntimeError):
            contexts.create()
        assert api.current == 101
    else:
        contexts.create()
        if failure == "destroy":
            api.destroy_error = -10
            with pytest.raises(RuntimeError):
                contexts.close()
            assert api.current == 101
        else:
            api.fail_restore = failure == "restore_error"
            api.ignore_restore = failure == "restore_lie"
            with pytest.raises(RuntimeError, match="restoration"):
                contexts.refuse(_refusal_frame(api), [])
            assert not contexts.restored
    assert not contexts.destroyed
    assert sum(event[0] == "destroy" for event in api.events) == (1 if failure == "destroy" else 0)


@pytest.mark.parametrize("kind", ["unexpected", "accepted", "query"])
def test_context_refusal_keeps_unexpected_result_and_restores_on_every_exit(consumer, kind):
    api = _ContextApi()
    contexts = consumer._CallerContexts(api, 0)
    contexts.create()
    error, accepted = RuntimeError("unrelated failure"), object()
    runs = []
    frame = _refusal_frame(
        api, error=error if kind == "unexpected" else None, accepted=accepted if kind == "accepted" else None
    )
    if kind == "query":
        api.query_failures = 1
    with pytest.raises(AssertionError if kind == "accepted" else RuntimeError) as caught:
        contexts.refuse(frame, runs)
    if kind == "unexpected":
        assert caught.value is error
    assert runs == ([accepted] if kind == "accepted" else [])
    assert api.current == 101
    assert not any(event[0] == "destroy" for event in api.events)


@pytest.fixture
def transport(consumer, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    monkeypatch.setenv("TASK_QUEUE_ENABLE", "0")
    import torch  # noqa: PLC0415 -- disable backend autoload before CPU import

    reference, api = consumer._helpers(), _ContextApi()
    events, native_calls, runs = [], [], []
    state = SimpleNamespace(fail_boundary=False, fail_restore=False, native_error=None)
    error = RuntimeError("recording completion failed")
    stream = SimpleNamespace(npu_stream=17)
    worker = Worker(level=2)

    def validate():
        if api.current != 101:
            if state.fail_restore:
                api.fail_restore = True
            raise ExecutionContextRefused("recording context mismatch")

    class NativeRun:
        def wait(self, timeout):
            assert events.count("native submit") == 3 and "framework boundary" in events
            events.append("retire")
            return True

        def done(self):
            return True

    def submit(cid, args, cfg, *, caller_stream):
        validate()
        assert cid == 3 and caller_stream == 17
        if state.native_error is not None:
            raise state.native_error
        native_calls.append(args)
        events.append("native submit")
        # CPU recording arithmetic is not native/device execution. The actual
        # Worker still resolves descriptors, reserves uses and owns RunHandles.
        arrays = [(ctypes.c_float * (128 * 128)).from_address(args.tensor(i).data) for i in range(3)]
        for index in range(128 * 128):
            arrays[2][index] = arrays[0][index] + arrays[1][index]
        run = NativeRun()
        runs.append(run)
        return run

    worker._chip_worker = SimpleNamespace(
        _execution_environment=(os.getpid(), 101, 1, 0, True),
        _validate_execution_context=validate,
        _validate_caller_stream=lambda stream: None,
        _submit_chip_run_direct=submit,
    )
    worker._lifecycle = _Lifecycle.READY
    monkeypatch.setattr(worker, "_resolve_handle", lambda *args, **kwargs: SimpleNamespace(slot_id=3))
    monkeypatch.setattr(worker, "register", lambda spec: object())
    monkeypatch.setattr(worker, "init", lambda **kwargs: None)

    def close():
        assert not worker._accepted_run_handles and not worker._chip_runs and not worker._chip_source_uses
        events.append("close")
        worker._lifecycle = _Lifecycle.CLOSED

    monkeypatch.setattr(worker, "close", close)
    monkeypatch.setitem(sys.modules, "simpler.worker", SimpleNamespace(Worker=lambda **kwargs: worker))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(consumer, "_helpers", lambda: reference)
    monkeypatch.setattr(consumer, "_context_api", lambda: api)
    monkeypatch.setattr(reference, "_compile", lambda runtime, *, platform="a2a3": object())
    empty = torch.empty

    def cpu_empty(*args, **kwargs):
        assert kwargs.pop("device") == "npu"
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            set_device=lambda device: None, Stream=lambda **kwargs: stream, stream=lambda actual: nullcontext()
        ),
        raising=False,
    )

    def enqueue(callback):
        callback()
        return SimpleNamespace(check=lambda: None, thread_id=threading.get_native_id())

    def finish():
        events.append("framework boundary")
        if state.fail_boundary:
            raise error

    monkeypatch.setattr(
        reference, "_load_bridge", lambda path: SimpleNamespace(enqueue=enqueue, finish_current_stream=finish)
    )
    yield SimpleNamespace(
        worker=worker,
        api=api,
        torch=torch,
        reference=reference,
        events=events,
        native_calls=native_calls,
        state=state,
        error=error,
        stream=stream,
    )
    # Test-only disposal: no device exists at this recording boundary.
    for run_id in tuple(worker._chip_source_uses):
        worker._release_source_uses(run_id)
    worker._chip_runs.clear()
    worker._accepted_run_handles.clear()
    worker._close_chip_import_registry()
    worker._chip_worker = None
    worker._lifecycle = _Lifecycle.CLOSED


def test_real_worker_rejects_stale_source_before_native_and_reuses_fresh_registration(consumer, transport):
    t = transport
    frames, _ = t.reference._prepare_frames(t.torch, t.worker, object(), t.stream, 17, 1)
    frame = frames[0]
    frame["stale_args"] = frame["args"]
    old_ids = [buffer.identity for buffer in frame["buffers"]]
    frame["buffers"][0].close()
    with pytest.raises(ValueError) as refused:
        t.worker.submit(frame["callable"], frame["stale_args"], caller_stream=17)
    assert str(refused.value) == (
        f"orch.submit: device argument (arg 0, {frame['stale_args'].tensor(0).buffer.identity}) is not a live "
        "allocation on target worker 0 (wrong worker, or stale)"
    )
    assert not t.native_calls and not t.worker._accepted_run_handles
    assert not t.worker._chip_source_uses and not t.worker._chip_run_touched_identities
    assert not t.worker._source_context_failed and t.worker._chip_run_seq == 0
    consumer._renew_registrations(frame)
    assert all(buffer.closed for buffer in frame["retired_buffers"])
    assert not set(old_ids) & {buffer.identity for buffer in frame["buffers"]}
    handle = t.worker.submit(frame["callable"], frame["args"], caller_stream=17)
    assert len(t.native_calls) == 1 and handle in t.worker._accepted_run_handles
    t.worker._finalize_run_handle(handle, handle._run_id, None)
    assert not t.worker._chip_source_uses
    for buffer in frame["buffers"]:
        buffer.close()


@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
@pytest.mark.parametrize("platform", ["a2a3", "a5"])
def test_actual_consumer_recording_keeps_recovery_and_retirement_order(consumer, transport, runtime, platform):
    summary = consumer.run(runtime, 0, Path("recording.so"), platform=platform)
    assert summary["platform"] == platform
    assert summary["observed"] == [8, 12, 16] and summary["refusals"] == ["source", "context"]
    assert summary["source_registration_replaced"] and summary["alternate_destroyed"]
    assert summary["caller_context_preserved"] and summary["caller_stream_after_close"]
    assert transport.events == ["native submit"] * 3 + ["framework boundary"] + ["retire"] * 3 + [
        "close",
        "framework boundary",
    ]
    assert transport.api.current == 101 and not consumer._UNPROVEN


def test_actual_consumer_unknown_completion_keeps_contexts_and_storage(consumer, transport):
    transport.state.fail_boundary = True
    with pytest.raises(RuntimeError) as caught:
        consumer.run("host_build_graph", 0, Path("recording.so"))
    assert caught.value is transport.error
    capsule = next(iter(consumer._UNPROVEN.values()))
    assert capsule["worker"] is transport.worker and len(capsule["runs"]) == 3
    assert len(capsule["frames"]) == 3 and all(
        not buffer.closed for frame in capsule["frames"] for buffer in frame["buffers"]
    )
    assert not any(event[0] == "destroy" for event in transport.api.events)
    assert transport.api.current == 101 and "close" not in transport.events


@pytest.mark.parametrize("failure", ["create", "callback_restore", "destroy"])
def test_actual_consumer_context_failure_retains_capsule_without_unsafe_cleanup(consumer, transport, failure):
    if failure == "create":
        transport.api.create_error = -9
    elif failure == "callback_restore":
        transport.state.fail_restore = True
    else:
        transport.api.destroy_error = -10
    with pytest.raises(RuntimeError):
        consumer.run("host_build_graph", 0, Path("recording.so"))
    capsule = next(iter(consumer._UNPROVEN.values()))
    if failure == "create":
        assert capsule["worker"] is None and not transport.native_calls
    elif failure == "callback_restore":
        assert len(capsule["runs"]) == len(transport.native_calls) == 1
        assert "failure_restore_error" in capsule and transport.api.current == 202
        assert "close" not in transport.events
    else:
        assert len(capsule["runs"]) == 3 and "close" in transport.events
    assert sum(event[0] == "destroy" for event in transport.api.events) == (1 if failure == "destroy" else 0)


def test_unexpected_source_acceptance_retains_handle_before_failing(consumer):
    accepted = object()
    frame = dict(
        index=0,
        worker=SimpleNamespace(submit=lambda *args, **kwargs: accepted),
        callable=object(),
        stale_args=object(),
        raw_stream=17,
    )
    runs = []
    with pytest.raises(AssertionError, match="Revoked source registration was accepted"):
        consumer._submit_frame(frame, None, runs, [])
    assert runs == [accepted]


def test_context_loader_uses_existing_runtime_only_and_does_not_fallback(consumer, monkeypatch):
    calls = []
    error = OSError("library is not already loaded")

    def unavailable(name, *, mode):
        calls.append((name, mode))
        raise error

    monkeypatch.setattr(consumer.ctypes, "CDLL", unavailable)
    with pytest.raises(OSError) as caught:
        consumer._context_api()
    assert caught.value is error
    assert calls == [("libascendcl.so", os.RTLD_LOCAL | os.RTLD_NOLOAD)]


def test_wrong_context_restore_failure_cannot_be_counted_or_followed_by_healthy_submit(consumer):
    api = _ContextApi()
    contexts = consumer._CallerContexts(api, 0)
    contexts.create()
    api.fail_restore = True
    runs, refusals = [], []
    with pytest.raises(RuntimeError, match="restoration"):
        consumer._submit_frame(_refusal_frame(api), contexts, runs, refusals)
    assert not runs and not refusals


def test_unexpected_source_runtime_error_is_not_a_clean_refusal(consumer, monkeypatch):
    error = RuntimeError("unexpected native failure")
    calls, refusals = [], []

    def fail(*args, **kwargs):
        calls.append(True)
        raise error

    frame = dict(
        index=0,
        worker=SimpleNamespace(submit=fail),
        callable=object(),
        stale_args=object(),
        args=object(),
        raw_stream=17,
    )
    monkeypatch.setattr(consumer, "_renew_registrations", lambda frame: None)
    with pytest.raises(RuntimeError) as caught:
        consumer._submit_frame(frame, None, [], refusals)
    assert caught.value is error
    assert calls == [True] and not refusals


@pytest.mark.parametrize(
    "message",
    [
        "borrowed submission requires caller_stream",
        "borrowed submission does not copy HOST tensors to DEVICE",
        "Worker no longer has a live device context for source registration",
    ],
)
def test_actual_consumer_rejects_unrelated_source_value_error(consumer, transport, monkeypatch, message):
    submit = transport.worker.submit
    calls = []

    def unexpected_value_error(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise ValueError(message)
        return submit(*args, **kwargs)

    monkeypatch.setattr(transport.worker, "submit", unexpected_value_error)
    with pytest.raises(AssertionError, match="Unexpected source preflight refusal"):
        consumer.run("host_build_graph", 0, Path("recording.so"))
    assert calls == [True] and not transport.native_calls
    capsule = next(iter(consumer._UNPROVEN.values()))
    assert not capsule["runs"]
    assert "registration_replaced" not in capsule["frames"][0]
    assert "close" not in transport.events
    assert not any(event[0] == "destroy" for event in transport.api.events)


def _summary(runtime, queue, platform="a2a3") -> dict:
    return dict(
        platform=platform,
        runtime=runtime,
        task_queue=queue,
        calls=3,
        observed=[8, 12, 16],
        refusals=["source", "context"],
        source_registration_replaced=True,
        caller_context_preserved=True,
        alternate_destroyed=True,
        caller_stream_after_close=True,
        caller_tensors_retained=True,
        caller_thread=10,
        callback_threads=[10 if queue == "0" else 11] * 3,
        verified=True,
    )


def _invoke_st(tmp_path, monkeypatch, records, runtime, queue, code=0, *, platform="a2a3"):
    st = _load(_ST, "framework_preflight_st")

    def run(command, **kwargs):
        assert Path(command[2]) == _CONSUMER
        assert command[command.index("--platform") + 1] == platform
        assert kwargs["env"]["TASK_QUEUE_ENABLE"] == queue
        assert kwargs["env"]["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
        return SimpleNamespace(
            returncode=code,
            stdout="".join("FRAMEWORK_PREFLIGHT_RECOVERY_PASS " + json.dumps(record) + "\n" for record in records),
            stderr="",
        )

    monkeypatch.setattr(st, "subprocess", SimpleNamespace(run=run))
    request = SimpleNamespace(
        config=SimpleNamespace(getoption=lambda name: {"--device": "0", "--platform": platform}[name])
    )
    st.test_framework_preflight_recovery(runtime, queue, tmp_path, tmp_path, request)


@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
@pytest.mark.parametrize("queue", ["0", "1"])
@pytest.mark.parametrize("platform", ["a2a3", "a5"])
def test_actual_st_accepts_complete_result(tmp_path, monkeypatch, runtime, queue, platform):
    _invoke_st(tmp_path, monkeypatch, [_summary(runtime, queue, platform)], runtime, queue, platform=platform)


@pytest.mark.parametrize(
    "corrupt",
    [
        "absent",
        "duplicate",
        "exit",
        "golden",
        "refusal",
        "renew",
        "context",
        "destroy",
        "stream",
        "retention",
        "thread",
        "count",
    ],
)
def test_actual_st_rejects_incomplete_result(tmp_path, monkeypatch, corrupt):
    runtime, queue = "host_build_graph", "1"
    summary = _summary(runtime, queue)
    records, code = [summary], 0
    if corrupt == "absent":
        records = []
    elif corrupt == "duplicate":
        records = [summary, summary]
    elif corrupt == "exit":
        code = 1
    elif corrupt == "golden":
        summary["observed"] = [8, 12, -999]
    elif corrupt == "refusal":
        summary["refusals"] = ["source"]
    elif corrupt == "thread":
        summary["callback_threads"] = [10] * 3
    elif corrupt == "count":
        summary["calls"] = 1
    else:
        summary[
            dict(
                renew="source_registration_replaced",
                context="caller_context_preserved",
                destroy="alternate_destroyed",
                stream="caller_stream_after_close",
                retention="caller_tensors_retained",
            )[corrupt]
        ] = False
    with pytest.raises(AssertionError):
        _invoke_st(tmp_path, monkeypatch, records, runtime, queue, code)
