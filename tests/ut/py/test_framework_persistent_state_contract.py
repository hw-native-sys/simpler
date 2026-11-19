# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU native argument/reference contracts, without emulating an executor.

CPU Torch allocations exercise descriptor geometry and caller ownership only.
The declared hardware test owns all claims about device execution and ordering.
"""

import gc
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from simpler.buffer import AccessMode, Buffer, BufferLocation, LocalEndpointBufferIdentityAllocator, _DeviceBufferSource
from simpler.task_interface import DataType, TensorArgType, TensorTransfer

_ROOT = Path(__file__).resolve().parents[3]
_CONSUMER = _ROOT / "examples/workers/l2/framework_eager/persistent_state.py"
_ST = _ROOT / "tests/st/worker/framework_eager/test_persistent_state.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def consumer():
    return _load(_CONSUMER, "framework_persistent_consumer")


@pytest.fixture
def torch(monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch  # noqa: PLC0415 -- backend autoload is disabled before the CPU package import

    return torch


@pytest.fixture
def worker():
    # Only native metadata is constructed; these CPU addresses never reach a device.
    source = _DeviceBufferSource(LocalEndpointBufferIdentityAllocator(b"STATEUT0"), 0, lambda: True)
    return SimpleNamespace(device_location=BufferLocation(source))


def _storage(torch):
    weights = torch.arange(1, 33, dtype=torch.float32)
    backing = torch.zeros(640)
    cache = backing.as_strided((8, 32), (64, 1), 64)
    proposals = torch.zeros((4, 32))
    control = torch.zeros(2, dtype=torch.int32)
    return [weights, backing, cache, proposals, control]


def test_native_arguments_keep_device_controls_separate_from_scalars_and_cache_inout(consumer, torch, worker):
    storage = _storage(torch)
    args, buffers, tensors = consumer._task_args(worker, *storage)
    try:
        assert args.tensor_count() == 4 and args.scalar_count() == 0
        assert [args.tag(i) for i in range(4)] == [
            TensorArgType.INPUT,
            TensorArgType.INOUT,
            TensorArgType.INPUT,
            TensorArgType.INPUT,
        ]
        assert all(args.transfer(i) == TensorTransfer.NONE for i in range(4))
        assert [buffer.access for buffer in buffers] == [
            AccessMode.READ,
            AccessMode.READWRITE,
            AccessMode.READ,
            AccessMode.READ,
        ]
        assert args.tensor(1).shapes == (8, 32) and args.tensor(1).strides == (64, 1)
        assert args.tensor(1).byte_offset == 256
        assert args.tensor(1).buffer == buffers[1].to_descriptor()
        assert args.tensor(1).buffer.nbytes == 2560
        assert args.tensor(3).dtype == DataType.INT32.value and args.tensor(3).shapes == (2,)
        assert len({buffer.identity for buffer in buffers}) == 4
        assert all(tensor.buffer == buffer.to_descriptor() for tensor, buffer in zip(tensors, buffers))
    finally:
        for buffer in buffers:
            buffer.close()


@pytest.mark.parametrize(
    "invalid",
    [
        "weight_dtype",
        "backing_extent",
        "backing_origin",
        "cache_shape",
        "cache_stride",
        "cache_origin",
        "foreign",
        "proposal_shape",
        "control_dtype",
        "control_origin",
    ],
)
def test_invalid_geometry_is_rejected_before_any_buffer_registration(consumer, torch, worker, monkeypatch, invalid):
    storage = _storage(torch)
    if invalid == "weight_dtype":
        storage[0] = storage[0].to(torch.float64)
    elif invalid == "backing_extent":
        storage[1] = storage[1][:-1]
    elif invalid == "backing_origin":
        storage[1] = torch.zeros(641)[1:]
    elif invalid == "cache_shape":
        storage[2] = storage[2][:7]
    elif invalid == "cache_stride":
        storage[2] = storage[1].as_strided((8, 32), (63, 1), 64)
    elif invalid == "cache_origin":
        storage[2] = storage[1].as_strided((8, 32), (64, 1), 65)
    elif invalid == "foreign":
        storage[2] = storage[1].clone().as_strided((8, 32), (64, 1), 64)
    elif invalid == "proposal_shape":
        storage[3] = storage[3][:3]
    elif invalid == "control_dtype":
        storage[4] = storage[4].to(torch.int64)
    else:
        storage[4] = torch.zeros(3, dtype=torch.int32)[1:]

    def forbidden(**kwargs):
        pytest.fail("Invalid geometry reached Buffer.wrap")

    monkeypatch.setattr(Buffer, "wrap", forbidden)
    with pytest.raises(ValueError):
        consumer._task_args(worker, *storage)


def _independent_states(*, accept_all=False, wrong_cursor=False, reset=False):
    # Scalar indexing and explicit input values are independent of the consumer's
    # vectorized Torch reference and its constants/helpers.
    initial = [float(i % 23 - 11) for i in range(640)]
    state, snapshots = initial.copy(), []
    for index, (cursor, accepted) in enumerate(
        ((0, 0), (1, 2), (6, 1), (0, 4), (4, 0), (3, 3), (2, 2), (7, 1), (4, 4), (0, 0), (5, 2), (1, 3))
    ):
        if reset:
            state = initial.copy()
        if wrong_cursor:
            cursor = 0
        if accept_all:
            # This variant remains in bounds while ignoring zero/partial acceptance.
            accepted = min(4, 8 - cursor)
        for row in range(accepted):
            proposal = (index + 1) * (row + 1) * (-1 if (index + row) % 2 else 1)
            for feature in range(32):
                state[64 + (cursor + row) * 64 + feature] += proposal * (feature + 1)
        snapshots.append(state.copy())
    return snapshots


def test_independent_reference_checks_every_update_zero_acceptance_and_untouched_element(consumer, torch):
    expected = _independent_states()
    actual = consumer._reference_snapshots(torch)
    assert len(actual) == 12 and [snapshot.tolist() for snapshot in actual] == expected
    active = {64 + row * 64 + feature for row in range(8) for feature in range(32)}
    initial, _ = consumer._initial(torch)
    for index, snapshot in enumerate(expected):
        previous = initial.tolist() if index == 0 else expected[index - 1]
        cursor, accepted = consumer.STEPS[index]
        updated = {64 + row * 64 + feature for row in range(cursor, cursor + accepted) for feature in range(32)}
        assert all(value == previous[position] for position, value in enumerate(snapshot) if position not in updated)
        assert all(value == initial[position] for position, value in enumerate(snapshot) if position not in active)
    assert expected[4] == expected[3] and expected[9] == expected[8]
    assert sum(expected[-1]) == -35414


@pytest.mark.parametrize("change", ["accept_all", "wrong_cursor", "reset", "guard", "weights"])
def test_actual_result_assertions_reject_ignored_controls_reset_and_corruption(consumer, torch, change):
    initial, weights = consumer._initial(torch)
    frame = dict(
        initial=initial, host_weights=weights, weights=weights.clone(), expected=consumer._reference_snapshots(torch)
    )
    states = _independent_states(**({change: True} if change in ("accept_all", "wrong_cursor", "reset") else {}))
    frame["snapshots"] = [torch.tensor(state) for state in states]
    if change == "guard":
        frame["snapshots"][-1][0] += 1
    elif change == "weights":
        frame["weights"][0] = -999
    # These are the consumer's actual post-completion assertions, fed CPU byte
    # images of valid in-bounds errors; no device/queue execution is emulated.
    with pytest.raises(AssertionError):
        consumer._verify_snapshots(torch, frame)


def test_actual_result_assertions_accept_independently_computed_images(consumer, torch):
    initial, weights = consumer._initial(torch)
    frame = dict(
        initial=initial, host_weights=weights, weights=weights.clone(), expected=consumer._reference_snapshots(torch)
    )
    frame["snapshots"] = [torch.tensor(state) for state in _independent_states()]
    assert consumer._verify_snapshots(torch, frame) == [
        -38,
        1018,
        2602,
        6826,
        6826,
        490,
        -3206,
        -7430,
        -16934,
        -16934,
        -22742,
        -35414,
    ]


def test_caller_frame_owns_all_backing_after_construction_scope_and_gc(consumer, torch, worker, monkeypatch):
    empty = torch.empty
    stream, handle = object(), object()
    records = []

    def cpu_empty(*args, **kwargs):
        assert kwargs.pop("device") == "npu"
        return empty(*args, **kwargs)

    # Allocation location/stream notification are substituted, not submission.
    monkeypatch.setattr(torch, "empty", cpu_empty)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, supplied: records.append(supplied))
    frame, references = consumer._prepare_frame(torch, worker, handle, stream, 1234)
    gc.collect()
    assert len(references) == 17 and all(reference() is not None for reference in references)
    assert records == [stream] * 17
    assert frame["worker"] is worker and frame["callable"] is handle and frame["stream"] is stream
    assert frame["raw_stream"] == 1234
    for buffer in frame["buffers"]:
        buffer.close()
    frame.clear()
    gc.collect()
    assert all(reference() is None for reference in references)


def _summary(runtime, queue) -> dict:
    return dict(
        platform="a2a3",
        runtime=runtime,
        task_queue=queue,
        calls=12,
        controls=[[0, 0], [1, 2], [6, 1], [0, 4], [4, 0], [3, 3], [2, 2], [7, 1], [4, 4], [0, 0], [5, 2], [1, 3]],
        cache_shape=[8, 32],
        cache_strides=[64, 1],
        cache_byte_offset=256,
        checksums=[-38, 1018, 2602, 6826, 6826, 490, -3206, -7430, -16934, -16934, -22742, -35414],
        caller_thread=10,
        callback_threads=[10 if queue == "0" else 11] * 12,
        caller_frame_retained=True,
        guard_verified=True,
        zero_accept_verified=True,
        weights_verified=True,
        persistent_state_verified=True,
        verified=True,
    )


def _invoke_st(tmp_path, monkeypatch, records, runtime, queue, code=0):
    module = _load(_ST, "framework_persistent_st")

    def run(command, **kwargs):
        assert Path(command[2]) == _CONSUMER
        assert command[command.index("--runtime") + 1] == runtime
        assert kwargs["env"]["TASK_QUEUE_ENABLE"] == queue
        assert kwargs["env"]["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
        stdout = "".join("FRAMEWORK_PERSISTENT_STATE_PASS " + json.dumps(record) + "\n" for record in records)
        return SimpleNamespace(returncode=code, stdout=stdout, stderr="")

    monkeypatch.setattr(module, "subprocess", SimpleNamespace(run=run))
    request = SimpleNamespace(
        config=SimpleNamespace(getoption=lambda name: {"--device": "0", "--platform": "a2a3"}[name])
    )
    module.test_framework_persistent_state(runtime, queue, tmp_path, tmp_path, request)


@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
@pytest.mark.parametrize("queue", ["0", "1"])
def test_actual_st_accepts_complete_result(tmp_path, monkeypatch, runtime, queue):
    _invoke_st(tmp_path, monkeypatch, [_summary(runtime, queue)], runtime, queue)


@pytest.mark.parametrize(
    "corrupt",
    [
        "absent",
        "duplicate",
        "exit",
        "checksum",
        "controls",
        "count",
        "geometry",
        "offset",
        "guard",
        "zero",
        "weights",
        "state",
        "retention",
        "thread",
        "verified",
    ],
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
    elif corrupt == "checksum":
        summary["checksums"][-1] = 0
    elif corrupt == "controls":
        summary["controls"][-1] = [0, 0]
    elif corrupt == "count":
        summary["calls"] = 1
    elif corrupt == "geometry":
        summary["cache_strides"] = [32, 1]
    elif corrupt == "offset":
        summary["cache_byte_offset"] = 0
    elif corrupt == "thread":
        summary["callback_threads"] = [10] * 12
    else:
        field = dict(
            guard="guard_verified",
            zero="zero_accept_verified",
            weights="weights_verified",
            state="persistent_state_verified",
            retention="caller_frame_retained",
            verified="verified",
        )[corrupt]
        summary[field] = False
    with pytest.raises(AssertionError):
        _invoke_st(tmp_path, monkeypatch, records, runtime, queue, code)
