# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real scalar encoding and the declared value-argument ST's result boundary."""

import ctypes
import importlib.util
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
from simpler.buffer import create_host_shared_buffer
from simpler.task_interface import DataType, Tensor, TensorArgType, TensorTransfer

_ROOT = Path(__file__).resolve().parents[3]
_CONSUMER = _ROOT / "examples/workers/l2/framework_eager/value_arguments.py"
_ST = _ROOT / "tests/st/a2a3/framework_eager/test_value_arguments.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def consumer():
    return _load(_CONSUMER, "framework_value_consumer")


@pytest.fixture
def tensors():
    buffers = [create_host_shared_buffer(16, b"VALUEUT0", buffer_id=index + 1) for index in range(3)]
    yield [Tensor(buffer, shapes=(4,), dtype=DataType.FLOAT32) for buffer in buffers]
    for buffer in buffers:
        buffer.close()


@pytest.mark.parametrize("host", [False, True])
@pytest.mark.parametrize("index", [0, 1, 2])
def test_native_task_args_encode_widths_sign_and_high_bits(consumer, tensors, host, index):
    values = consumer._VALUES[index]
    args, wrappers = consumer._arguments(tensors[:2], values, host_control=tensors[2] if host else None)
    expected = (
        struct.unpack("<I", struct.pack("<f", values[0]))[0],
        values[1] & 0xFF,
        values[2],
        struct.unpack("<Q", struct.pack("<d", values[3]))[0],
    )
    assert args.tensor_count() == (3 if host else 2)
    assert args.scalar_count() == 4
    assert tuple(args.scalar(i) for i in range(4)) == expected
    assert [args.tag(i) for i in range(args.tensor_count())] == [TensorArgType.INPUT, TensorArgType.OUTPUT_EXISTING] + (
        [TensorArgType.INPUT] if host else []
    )
    assert all(args.transfer(i) == TensorTransfer.NONE for i in range(args.tensor_count()))
    # Encoding owns values rather than retaining references to ctypes payloads.
    for wrapper in wrappers:
        wrapper.value = 0
    assert tuple(args.scalar(i) for i in range(4)) == expected


def test_task_args_snapshot_survives_clearing_original_container(consumer, tensors):
    from _task_interface import _snapshot_local_task_args  # noqa: PLC0415

    args, wrappers = consumer._arguments(tensors[:2], consumer._VALUES[0])
    snapshot = _snapshot_local_task_args(args)
    args.clear()
    for wrapper in wrappers:
        wrapper.value = 0
    assert len(args) == 0
    assert snapshot.tensor_count() == 2 and snapshot.scalar_count() == 4
    assert [snapshot.scalar(i) for i in range(4)] == [0x3FA00000, 0xFD, 0x8000000000000001, 0x3FF0000000001000]


def test_precision_and_width_choices_change_the_reference_result(consumer):
    values = consumer._VALUES
    assert ctypes.c_float(values[0][3]).value == ctypes.c_float(values[1][3]).value == 1.0
    assert values[0][3] > 1.0 and values[1][3] < 1.0
    assert int(float(values[0][2])) != values[0][2]  # A numeric float conversion loses the low selector bit.
    assert struct.unpack("<Q", struct.pack("<d", values[2][3]))[0] == 1 << 63
    assert [consumer._expected(i) for i in range(3)] == [49.0, 8.0, 83.5]
    assert [consumer._expected(i, True) for i in range(3)] == [54.0, 1.0, 92.5]


def _summary(runtime, queue):
    host = runtime == "host_build_graph"
    return dict(
        runtime=runtime,
        task_queue=queue,
        verified=True,
        calls=6 if host else 3,
        observed=[49.0, 8.0, 83.5] + ([54.0, 1.0, 92.5] if host else []),
        caller_args_cleared=True,
        wrappers_mutated=True,
        host_backing_retained=True,
        host_controls_changed=host,
        clean_host_refusals=0 if host else 1,
        caller_thread=10,
        callback_threads=[10 if queue == "0" else 11] * (6 if host else 3),
    )


def _invoke_st(tmp_path, monkeypatch, records, runtime, queue, returncode=0):
    module = _load(_ST, "framework_value_st")

    def run(command, **kwargs):
        assert Path(command[2]) == _CONSUMER
        assert command[command.index("--runtime") + 1] == runtime
        assert kwargs["env"]["TASK_QUEUE_ENABLE"] == queue
        assert kwargs["env"]["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
        stdout = "".join("FRAMEWORK_VALUE_ARGUMENTS_PASS " + json.dumps(record) + "\n" for record in records)
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(module, "subprocess", SimpleNamespace(run=run))
    request = SimpleNamespace(config=SimpleNamespace(getoption=lambda _: "0"))
    module.test_framework_value_arguments(runtime, queue, tmp_path, tmp_path, request)


@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
@pytest.mark.parametrize("queue", ["0", "1"])
def test_actual_st_accepts_complete_result(tmp_path, monkeypatch, runtime, queue):
    _invoke_st(tmp_path, monkeypatch, [_summary(runtime, queue)], runtime, queue)


@pytest.mark.parametrize(
    "corrupt",
    ["absent", "duplicate", "exit", "golden", "args", "wrappers", "backing", "host", "refusal", "thread", "count"],
)
def test_actual_st_rejects_incomplete_or_wrong_result(tmp_path, monkeypatch, corrupt):
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
        summary["observed"] = [49.0, 8.0, 83.5, 1049.0, 1009.0, 1085.5]
    elif corrupt == "args":
        summary["caller_args_cleared"] = False
    elif corrupt == "wrappers":
        summary["wrappers_mutated"] = False
    elif corrupt == "backing":
        summary["host_backing_retained"] = False
    elif corrupt == "host":
        summary["host_controls_changed"] = False
    elif corrupt == "refusal":
        summary["clean_host_refusals"] = 1
    elif corrupt == "thread":
        summary["callback_threads"] = [10] * 6
    else:
        summary["calls"] = 3
    with pytest.raises(AssertionError):
        _invoke_st(tmp_path, monkeypatch, records, runtime, queue, code)
