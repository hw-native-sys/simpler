# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU geometry/ownership and the real alias ST's result rejection contract.

CPU tensors stand in only for caller storage; no worker, queue or executor is
emulated. Silicon ordering is covered by the torch_npu environment ST.
"""

import gc
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from simpler.buffer import Buffer, BufferLocation, LocalEndpointBufferIdentityAllocator, _DeviceBufferSource
from simpler.task_interface import TensorArgType

_ROOT = Path(__file__).resolve().parents[3]
_CONSUMER = _ROOT / "examples/workers/l2/framework_eager/alias_inout.py"
_ST = _ROOT / "tests/st/worker/framework_eager/test_alias_inout.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def consumer():
    return _load(_CONSUMER, "framework_alias_consumer")


@pytest.fixture
def torch(monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch  # noqa: PLC0415 -- disable automatic device backend before importing the CPU package

    return torch


@pytest.fixture
def worker():
    # Only descriptor registration is exercised. The CPU pointer is never
    # submitted or dereferenced by a device/runtime.
    source = _DeviceBufferSource(LocalEndpointBufferIdentityAllocator(b"ALIASUT0"), 0, lambda: True)
    return SimpleNamespace(device_location=BufferLocation(source))


def _storage(torch, consumer, alias):
    backing = torch.arange(consumer.BACKING_ELEMENTS, dtype=torch.float32)
    views = tuple(backing.as_strided(consumer.SHAPE, consumer.STRIDES, offset) for offset in consumer._offsets(alias))
    return backing, views


@pytest.mark.parametrize("alias", ["full", "partial"])
def test_native_arguments_share_one_buffer_and_preserve_inout_geometry(consumer, torch, worker, alias):
    backing, views = _storage(torch, consumer, alias)
    args, buffer, arguments = consumer._task_args(worker, backing, views, alias)
    try:
        assert args.tensor_count() == 2
        assert (args.tag(0), args.tag(1)) == (TensorArgType.INPUT, TensorArgType.INOUT)
        assert arguments[0] is not arguments[1]
        for index, offset in enumerate((64, 72 if alias == "partial" else 64)):
            tensor = args.tensor(index)
            assert tensor.buffer == buffer.to_descriptor()
            assert tensor.buffer.nbytes == backing.numel() * 4
            assert tensor.byte_offset == offset * 4
            assert tensor.shapes == (64, 64)
            assert tensor.strides == (96, 1)
    finally:
        buffer.close()


@pytest.mark.parametrize("invalid", ["shape", "stride", "origin", "foreign", "dtype", "extent", "backing_origin"])
def test_geometry_rejected_before_registering_caller_buffer(consumer, torch, worker, monkeypatch, invalid):
    backing, views = _storage(torch, consumer, "partial")
    source, target = views
    if invalid == "shape":
        source = source[:63]
    elif invalid == "stride":
        source = backing.as_strided((64, 64), (95, 1), 64)
    elif invalid == "origin":
        target = backing.as_strided((64, 64), (96, 1), 73)
    elif invalid == "foreign":
        source = backing.clone().as_strided((64, 64), (96, 1), 64)
    elif invalid == "dtype":
        backing = backing.to(torch.float64)
    elif invalid == "extent":
        backing = backing[:-1]
    else:
        backing = torch.empty(consumer.BACKING_ELEMENTS + 1)[1:]

    def forbidden_wrap(**kwargs):
        pytest.fail("invalid geometry reached source registration")

    monkeypatch.setattr(Buffer, "wrap", forbidden_wrap)
    with pytest.raises(ValueError):
        consumer._task_args(worker, backing, (source, target), "partial")


@pytest.mark.parametrize("alias", ["full", "partial"])
def test_reference_snapshots_all_sources_and_preserves_non_target_elements(consumer, torch, alias):
    backing = torch.arange(consumer.BACKING_ELEMENTS, dtype=torch.float32).remainder_(19)
    expected = backing.tolist()
    source, target = 64, 72 if alias == "partial" else 64
    target_indices = {target + row * 96 + col for row in range(64) for col in range(64)}
    for invocation in range(3):
        previous = [value + invocation + 1 for value in expected]
        expected = previous.copy()
        # Independent scalar indexing reads only the pre-call image, including
        # source elements which an earlier destination row/column overlaps.
        for row in range(64):
            for col in range(64):
                index = row * 96 + col
                expected[target + index] = previous[source + index] + previous[target + index]
        snapshot = consumer._reference_step(backing, alias, invocation)
        assert backing.tolist() == expected
        assert snapshot.tolist() == [value + invocation + 2 for value in expected]
        assert all(expected[index] == previous[index] for index in range(len(expected)) if index not in target_indices)


def test_caller_frames_retain_real_tensors_after_construction_and_gc(consumer, torch, worker, monkeypatch):
    empty = torch.empty
    stream, handle = object(), object()
    recorded = []

    def cpu_empty(*args, **kwargs):
        assert kwargs.pop("device") == "npu"
        return empty(*args, **kwargs)

    # Only the allocation location/record-stream transport is substituted.
    # Actual Torch views, Buffer registration and native TaskArgs are used.
    monkeypatch.setattr(torch, "empty", cpu_empty)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, supplied: recorded.append(supplied))
    frames, references = consumer._prepare_frames(torch, worker, handle, stream, 1234, 3)
    gc.collect()
    assert len(references) == 12 and all(reference() is not None for reference in references)
    assert recorded == [stream] * 12
    assert all(
        frame["stream"] is stream and frame["worker"] is worker and frame["callable"] is handle for frame in frames
    )
    for frame in frames:
        frame["buffer"].close()
    frames.clear()
    del frame
    gc.collect()
    assert all(reference() is None for reference in references)


def _summary(queue="1"):
    return dict(
        platform="a2a3",
        runtime="host_build_graph",
        task_queue=queue,
        verified=True,
        calls_per_alias=3,
        calls=6,
        aliases=["full", "partial"],
        shape=[64, 64],
        strides=[96, 1],
        offsets={"full": [256, 256], "partial": [256, 288]},
        caller_frame_retained=True,
        guard_verified=True,
        state_reused=True,
        caller_thread=10,
        callback_threads=[10 if queue == "0" else 11] * 6,
    )


def _invoke_st(tmp_path, monkeypatch, records, *, queue="1", returncode=0):
    module = _load(_ST, "framework_alias_st")

    def run(command, **kwargs):
        assert Path(command[2]) == _CONSUMER
        assert command[-2:] == ["--calls", "3"]
        assert kwargs["env"]["TASK_QUEUE_ENABLE"] == queue
        assert kwargs["env"]["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
        stdout = "".join("FRAMEWORK_ALIAS_INOUT_PASS " + json.dumps(record) + "\n" for record in records)
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(module, "subprocess", SimpleNamespace(run=run))
    request = SimpleNamespace(
        config=SimpleNamespace(getoption=lambda name: {"--device": "0", "--platform": "a2a3"}[name])
    )
    module.test_framework_alias_inout("host_build_graph", queue, tmp_path, tmp_path, request)


@pytest.mark.parametrize("queue", ["0", "1"])
def test_actual_hardware_test_accepts_complete_result_only(tmp_path, monkeypatch, queue):
    _invoke_st(tmp_path, monkeypatch, [_summary(queue)], queue=queue)


@pytest.mark.parametrize(
    "corrupt", ["absent", "duplicate", "exit", "guard", "retention", "state", "geometry", "offset", "threads", "count"]
)
def test_actual_hardware_test_rejects_false_success(tmp_path, monkeypatch, corrupt):
    summary = _summary()
    records, code = [summary], 0
    if corrupt == "absent":
        records = []
    elif corrupt == "duplicate":
        records = [summary, summary]
    elif corrupt == "exit":
        code = 1
    elif corrupt == "guard":
        summary["guard_verified"] = False
    elif corrupt == "retention":
        summary["caller_frame_retained"] = False
    elif corrupt == "state":
        summary["state_reused"] = False
    elif corrupt == "geometry":
        summary["strides"] = [64, 1]
    elif corrupt == "offset":
        summary["offsets"] = {"full": [256, 256], "partial": [0, 32]}
    elif corrupt == "threads":
        summary["callback_threads"] = [10] * 6
    else:
        summary["calls"] = 3
    with pytest.raises(AssertionError):
        _invoke_st(tmp_path, monkeypatch, records, returncode=code)
