# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Actual capture/parser checks; native arithmetic/queue/exports below are recording boundaries."""

import ctypes
import importlib.util
import json
import os
import struct
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from simpler.task_interface import DataType, RunPreparationRefused
from simpler.worker import Worker, _Lifecycle

_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLE = _ROOT / "examples/workers/l2/framework_eager/argument_capture.py"
_ST = _ROOT / "tests/st/worker/framework_eager/test_argument_capture.py"
_KNOWN = (
    {
        "a": [-8, -7, -6, -5, -4, -3],
        "b": [-6, -3, 0, 3, 6, -4],
        "mid": [-14, -10, -6, -2, 2, -7],
        "out": [-20, -13, -6, 1, 8, -11],
    },
    {
        "a": [-3, -2, -1, 0, 1, 2],
        "b": [-7, -4, -1, 2, 5, -5],
        "mid": [-10, -6, -2, 2, 6, -3],
        "out": [-17, -10, -3, 4, 11, -8],
    },
    {
        "a": [2, 3, 4, 5, 6, 7],
        "b": [-8, -5, -2, 1, 4, -6],
        "mid": [-6, -2, 2, 6, 10, 1],
        "out": [-14, -7, 0, 7, 14, -5],
    },
    {
        "a": [7, 8, 9, 10, 11, 12],
        "b": [-9, -6, -3, 0, 3, -7],
        "mid": [-2, 2, 6, 10, 14, 5],
        "out": [-11, -4, 3, 10, 17, -2],
    },
)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def consumer():
    module = _load(_EXAMPLE, "argument_capture_contract")
    module.SHAPE, module.COUNT = (2, 3), 6
    return module


def _seed(folder, invocation, level, *, unknown_func=False, materialized=None):
    dump = folder / "args_dump"
    dump.mkdir(parents=True)
    payload, records = bytearray(), []
    # IDs run backwards relative to submission order and repeat across runs.
    for task, names in ((1, ("mid", "b", "out")), (0, ("a", "b", "mid"))):
        for slot, name in enumerate(names):
            selected = level == 3 or (task, slot) in ((0, 2), (1, 0), (1, 2))
            record = dict(
                task_id=("0x70", "0x02")[task],
                run_epoch=invocation + 100,
                func_id=[-1] if unknown_func else [0],
                arg_index=slot,
                role="input" if slot < 2 else "output",
                stage="before_dispatch" if slot < 2 else "after_completion",
                kind="tensor",
                dtype=DataType.FLOAT32.name,
                shape=[6] if name == "a" else [2, 3],
                strides=[1] if name == "a" else [3, 1],
                start_offset=0,
                numel=6,
                is_contiguous=True,
                truncated=False,
                bin_offset=len(payload) if selected else 0,
                bin_size=24 if selected else 0,
            )
            if materialized is not None and name in materialized:
                record.update(materialized[name])
            records.append(record)
            if selected:
                payload.extend(struct.pack("<6f", *_KNOWN[invocation][name]))
    data: dict = dict(
        run_dir="args_dump",
        dump_args_level=level,
        bin_format={"type": "logical_contiguous", "byte_order": "little_endian"},
        total_args=6,
        before_dispatch=4,
        after_completion=2,
        input_args=4,
        output_args=2,
        inout_args=0,
        truncated_args=0,
        dropped_records=0,
        dropped_records_complete=True,
        bin_file="args.bin",
        args=records,
    )
    (dump / "args.bin").write_bytes(payload)
    (dump / "args_dump.json").write_text(json.dumps(data))
    return data, dump / "args.bin", dump / "args_dump.json"


@pytest.mark.parametrize("level", [2, 3])
def test_checker_accepts_real_schema_with_opaque_task_ids_and_exact_bytes(consumer, tmp_path, level):
    _seed(tmp_path, 1, level)
    assert consumer._check_dump(tmp_path, 1, level) == 101
    assert consumer._values(1) == _KNOWN[1]


@pytest.mark.parametrize("level", [2, 3])
def test_actual_checker_rejects_unknown_kernel_attribution(consumer, tmp_path, level):
    _seed(tmp_path, 1, level, unknown_func=True)
    with pytest.raises(AssertionError):
        consumer._check_dump(tmp_path, 1, level)


@pytest.mark.parametrize("level", [2, 3])
def test_actual_checker_rejects_lowercase_dtype_schema(consumer, tmp_path, level):
    data, _, manifest = _seed(tmp_path, 1, level)
    for record in data["args"]:
        record["dtype"] = "float32"
    manifest.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        consumer._check_dump(tmp_path, 1, level)


@pytest.mark.parametrize(
    "target,field,value",
    [
        (3, "start_offset", 1),
        (3, "start_offset", 32),
        (0, "shape", [3, 2]),
        (0, "strides", [1, 2]),
        (0, "dtype", "INT32"),
        (-1, "run_dir", "foreign"),
        (0, "func_id", [1]),
        (0, "run_epoch", 0),
        (0, "run_epoch", 99),
        (0, "host_discarded", True),
        (0, "truncated", True),
        (0, "bin_offset", -1),
        (0, "stage", "after_completion"),
        (-1, "dump_args_level", 2),
        (-1, "dropped_records", 1),
        (-1, "dropped_records_complete", False),
        (-1, "counts_unknown", True),
        (-1, "collection_verdict", "UNKNOWN"),
        (-1, "before_dispatch", 3),
        (-1, "host_discarded_args", 1),
    ],
)
def test_actual_checker_rejects_wrong_metadata(consumer, tmp_path, target, field, value):
    data, _, manifest = _seed(tmp_path, 1, 3)
    (data if target == -1 else data["args"][target])[field] = value
    manifest.write_text(json.dumps(data))
    with pytest.raises((AssertionError, KeyError)):
        consumer._check_dump(tmp_path, 1, 3)


@pytest.mark.parametrize(
    "corrupt",
    ["zero", "stale", "poison", "wrong_task", "missing", "duplicate", "overlap", "trailing_bytes", "extra_artifact"],
)
def test_actual_checker_rejects_incomplete_or_wrong_payloads(consumer, tmp_path, corrupt):
    data, binary, manifest = _seed(tmp_path, 1, 3)
    records, payload = data["args"], bytearray(binary.read_bytes())
    if corrupt in ("zero", "stale", "poison"):
        values = [0] * 6 if corrupt == "zero" else _KNOWN[0]["mid"] if corrupt == "stale" else [-999] * 6
        payload[:24] = struct.pack("<6f", *values)
    elif corrupt == "wrong_task":
        records[0]["bin_offset"], records[2]["bin_offset"] = records[2]["bin_offset"], records[0]["bin_offset"]
    elif corrupt == "missing":
        records.pop()
    elif corrupt == "duplicate":
        records[1] = records[0].copy()
    elif corrupt == "overlap":
        # Identical B inputs make bounds/data checks pass without interval checks.
        records[4]["bin_offset"] = records[1]["bin_offset"]
    elif corrupt == "trailing_bytes":
        payload.extend(b"trailing")
    else:
        (manifest.parent / "args.e200.json").write_text("{}")
    binary.write_bytes(payload)
    manifest.write_text(json.dumps(data))
    with pytest.raises((AssertionError, KeyError)):
        consumer._check_dump(tmp_path, 1, 3)


def test_hybrid_must_not_capture_unmarked_inputs(consumer, tmp_path):
    data, _, manifest = _seed(tmp_path, 2, 2)
    data["args"][1]["bin_size"] = 24
    manifest.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        consumer._check_dump(tmp_path, 2, 2)


@pytest.mark.parametrize("failure", ["manifest", "payload", "malformed"])
def test_actual_checker_requires_readable_complete_artifacts(consumer, tmp_path, failure):
    _, binary, manifest = _seed(tmp_path, 1, 3)
    if failure == "malformed":
        manifest.write_text("{")
    else:
        (manifest if failure == "manifest" else binary).unlink()
    with pytest.raises(json.JSONDecodeError if failure == "malformed" else FileNotFoundError):
        consumer._check_dump(tmp_path, 1, 3)


@pytest.mark.parametrize("filename", ["args_dump/args_dump.json", "args_dump/args.e900.bin", "args.e17.json"])
def test_off_checks_retained_dump_forms_but_allows_host_logs(consumer, tmp_path, filename):
    (tmp_path / "host.log").write_text("permitted host log")
    assert consumer._check_dump(tmp_path, 0, 0) is None
    path = tmp_path / filename
    path.parent.mkdir(exist_ok=True)
    path.write_text("unexpected dump")
    with pytest.raises(AssertionError):
        consumer._check_dump(tmp_path, 0, 0)


@pytest.fixture
def transport(consumer, monkeypatch, tmp_path):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    monkeypatch.setenv("TASK_QUEUE_ENABLE", "0")
    import torch  # noqa: PLC0415

    worker, events = Worker(level=2), []
    state = SimpleNamespace(fail_boundary=False, materialized=[], refuse_dump=False, defect=None, active_stream=None)
    stream = SimpleNamespace(npu_stream=17)

    class NativeRun:
        def wait(self, timeout):
            assert events.count("submit") == 4 and "boundary" in events
            events.append("retire")
            return True

        def done(self):
            return True

    def submit(cid, args, cfg, *, caller_stream):
        assert cid == 3 and caller_stream == 17
        index = events.count("submit")
        events.append("submit")
        # Native arithmetic and export are recording boundaries. Public Worker
        # still binds real descriptors, reserves source uses and retires handles.
        geometry = {
            name: dict(
                shape=list(args.tensor(i).shapes),
                strides=list(args.tensor(i).strides),
                start_offset=args.tensor(i).start_offset,
                dtype=args.tensor(i).dtype.name,
            )
            for i, name in enumerate(("a", "b", "out"))
        }
        state.materialized.append(geometry)
        arrays = [(ctypes.c_float * 6).from_address(args.tensor(i).data) for i in range(3)]
        if state.refuse_dump and cfg.enable_dump_args and state.defect != "silent_accept":
            events.append("refuse")
            if state.defect == "output_write":
                arrays[2][0] = 123
            if state.defect == "dump_files":
                _seed(Path(cfg.output_prefix), index, cfg.enable_dump_args, materialized=geometry)
            status = -1004 if state.defect == "wrong_status" else -1001
            message = (
                f"prepare_native_run failed with code {status} "
                f"(run_id=0 slot=0 generation={index + 1} dispatch_id=0 run_epoch={index + 1})"
            )
            if state.defect == "wrong_reason":
                message = "another refusal with code -1001"
            error = RuntimeError if state.defect == "untyped" else RunPreparationRefused
            raise error(message)
        for item in range(6):
            arrays[2][item] = arrays[0][item] + 2 * arrays[1][item]
        if state.defect == "recovery" and index == 3:
            arrays[2][0] += 1
        if cfg.enable_dump_args:
            _seed(Path(cfg.output_prefix), index, cfg.enable_dump_args, materialized=geometry)
        return NativeRun()

    worker._chip_worker = SimpleNamespace(
        _execution_environment=(os.getpid(), 101, 1, 0, True),
        _validate_execution_context=lambda: None,
        _validate_caller_stream=lambda value: None,
        _submit_chip_run_direct=submit,
    )
    worker._lifecycle = _Lifecycle.READY
    monkeypatch.setattr(worker, "_resolve_handle", lambda *args, **kwargs: SimpleNamespace(slot_id=3))
    monkeypatch.setattr(worker, "register", lambda image: object())
    monkeypatch.setattr(worker, "init", lambda **kwargs: None)

    def close():
        assert not worker._accepted_run_handles and not worker._chip_source_uses and not worker._chip_runs
        events.append("close")
        worker._lifecycle = _Lifecycle.CLOSED

    monkeypatch.setattr(worker, "close", close)
    monkeypatch.setitem(sys.modules, "simpler.worker", SimpleNamespace(Worker=lambda **kwargs: worker))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(consumer, "_compile", lambda runtime, *, platform="a2a3": object())
    empty = torch.empty

    def cpu_empty(*args, **kwargs):
        assert kwargs.pop("device") == "npu"
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)

    @contextmanager
    def stream_scope(actual):
        previous, state.active_stream = state.active_stream, actual
        try:
            yield
        finally:
            state.active_stream = previous

    cpu = torch.Tensor.cpu

    def checked_cpu(tensor, *args, **kwargs):
        if "close" in events:
            assert state.active_stream is stream, "post-close read must use the caller stream"
        return cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", checked_cpu)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(set_device=lambda device: None, Stream=lambda **kwargs: stream, stream=stream_scope),
        raising=False,
    )

    def enqueue(callback):
        callback()
        return SimpleNamespace(check=lambda: None, thread_id=threading.get_native_id())

    def finish():
        assert state.active_stream is stream, "completion must drain the caller stream"
        events.append("boundary")
        if state.fail_boundary:
            raise RuntimeError("recorded unknown completion")

    bridge = SimpleNamespace(enqueue=enqueue, finish_current_stream=finish)
    monkeypatch.setattr(consumer, "_helpers", lambda: SimpleNamespace(_load_bridge=lambda path: bridge))
    yield SimpleNamespace(worker=worker, events=events, state=state, torch=torch, root=tmp_path / "captures")
    for run_id in tuple(worker._chip_source_uses):
        worker._release_source_uses(run_id)
    worker._chip_runs.clear()
    worker._accepted_run_handles.clear()
    worker._close_chip_import_registry()
    worker._chip_worker, worker._lifecycle = None, _Lifecycle.CLOSED


@pytest.mark.parametrize(
    "platform,runtime",
    [("a2a3", "tensormap_and_ringbuffer"), ("a2a3", "host_build_graph"), ("a5", "tensormap_and_ringbuffer")],
)
def test_whole_consumer_records_intended_modes_and_retirement(consumer, transport, runtime, platform):
    result = consumer.run(runtime, 0, Path("recording.so"), transport.root, platform=platform)
    assert result["platform"] == platform
    assert result["payloads_verified"] and result["config_snapshot_verified"] and result["guards_verified"]
    # This trace belongs to the synchronous recording enqueue, not to real Torch scheduling.
    assert transport.events == ["submit"] * 4 + ["boundary"] + ["retire"] * 4 + ["close"]
    assert not consumer._UNPROVEN


def test_a5_hbg_refuses_only_enabled_dumps_and_recovers_on_the_same_worker(consumer, transport):
    transport.state.refuse_dump = True
    result = consumer.run("host_build_graph", 0, Path("recording.so"), transport.root, platform="a5")
    assert result["accepted_calls"] == result["calls"] == 2
    assert result["args_dump_supported"] is False and result["payloads_verified"] is False
    assert [(item["index"], item["level"]) for item in result["refusals"]] == [(1, 3), (2, 2)]
    assert result["off_calls_verified"] and result["refusal_outputs_unchanged"]
    assert result["caller_tensors_retained"] and result["caller_stream_after_close"]
    assert result["payload_counts"] == [0, 0, 0, 0] and result["run_epochs"] == [None] * 4
    assert transport.events == [
        "submit",
        "submit",
        "refuse",
        "submit",
        "refuse",
        "submit",
        "boundary",
        "retire",
        "retire",
        "close",
        "boundary",
    ]
    assert not consumer._dump_files(transport.root) and not consumer._UNPROVEN


@pytest.mark.parametrize(
    "defect", ["wrong_status", "wrong_reason", "silent_accept", "output_write", "dump_files", "recovery"]
)
def test_a5_hbg_contract_rejects_false_refusals_writes_and_broken_recovery(consumer, transport, defect):
    transport.state.refuse_dump, transport.state.defect = True, defect
    with pytest.raises(AssertionError):
        consumer.run("host_build_graph", 0, Path("recording.so"), transport.root, platform="a5")


def test_a5_hbg_does_not_translate_an_untyped_failure_into_a_refusal(consumer, transport):
    transport.state.refuse_dump, transport.state.defect = True, "untyped"
    with pytest.raises(RuntimeError, match="prepare_native_run failed"):
        consumer.run("host_build_graph", 0, Path("recording.so"), transport.root, platform="a5")
    assert consumer._UNPROVEN and "close" not in transport.events


@pytest.mark.parametrize("platform,runtime", [("a2a3", "host_build_graph"), ("a5", "tensormap_and_ringbuffer")])
def test_supported_dump_refusal_is_still_a_failure(consumer, transport, platform, runtime):
    transport.state.refuse_dump = True
    with pytest.raises(RunPreparationRefused):
        consumer.run(runtime, 0, Path("recording.so"), transport.root, platform=platform)
    assert consumer._UNPROVEN and "close" not in transport.events


def test_a5_hbg_unknown_boundary_retains_refused_and_accepted_caller_frames(consumer, transport):
    transport.state.refuse_dump = transport.state.fail_boundary = True
    with pytest.raises(RuntimeError, match="unknown completion"):
        consumer.run("host_build_graph", 0, Path("recording.so"), transport.root, platform="a5")
    worker, _, frames, queued, runs, _ = consumer._UNPROVEN[0]
    assert worker is transport.worker and len(frames) == len(queued) == 4 and len(runs) == 2
    assert all(not buffer.closed for frame in frames for buffer in frame["buffers"])
    assert "retire" not in transport.events and "close" not in transport.events


def test_unknown_completion_keeps_every_caller_frame(consumer, transport):
    transport.state.fail_boundary = True
    with pytest.raises(RuntimeError, match="unknown completion"):
        consumer.run("host_build_graph", 0, Path("recording.so"), transport.root)
    worker, handle, frames, queued, runs, stream = consumer._UNPROVEN[0]
    assert worker is transport.worker and handle is not None and stream.npu_stream == 17
    assert len(frames) == len(queued) == len(runs) == 4
    assert all(not buffer.closed for frame in frames for buffer in frame["buffers"])
    assert "retire" not in transport.events and "close" not in transport.events


@pytest.mark.parametrize("invalid", [None, "dtype", "offset", "backing", "stride"])
def test_real_arguments_preserve_caller_view_and_no_caller_intermediate(consumer, transport, invalid):
    from simpler.task_interface import TensorArgType, TensorTransfer  # noqa: PLC0415

    torch = transport.torch
    backing = torch.zeros(consumer.COUNT + 2 * consumer.OFFSET)
    a = backing[consumer.OFFSET : consumer.OFFSET + consumer.COUNT]
    b, out = torch.zeros(consumer.SHAPE), torch.zeros(consumer.SHAPE)
    if invalid == "dtype":
        b = b.to(torch.int32)
    elif invalid == "offset":
        a = backing[: consumer.COUNT]
    elif invalid == "backing":
        other = torch.zeros_like(backing)
        a = other[consumer.OFFSET : consumer.OFFSET + consumer.COUNT]
    elif invalid == "stride":
        b = torch.zeros((consumer.SHAPE[0], 2 * consumer.SHAPE[1]))[:, ::2]
    if invalid is not None:
        with pytest.raises(AssertionError):
            consumer._task_args(transport.worker, backing, a, b, out)
        assert not transport.worker._chip_source_uses
        return
    args, buffers = consumer._task_args(transport.worker, backing, a, b, out)
    assert args.tensor_count() == 3 and args.scalar_count() == 0
    assert [args.tag(i) for i in range(3)] == [TensorArgType.INPUT, TensorArgType.INPUT, TensorArgType.OUTPUT_EXISTING]
    assert all(args.transfer(i) == TensorTransfer.NONE for i in range(3))
    assert args.tensor(0).buffer == buffers[0].to_descriptor()
    assert args.tensor(0).byte_offset == consumer.OFFSET * 4
    assert args.tensor(0).shapes == (consumer.COUNT,) and args.tensor(0).strides == (1,)
    for buffer in buffers:
        buffer.close()


def test_real_worker_materialization_folds_view_origin_into_address(consumer, transport, monkeypatch):
    from simpler.task_interface import CallConfig  # noqa: PLC0415

    torch = transport.torch
    backing = torch.zeros(consumer.COUNT + 2 * consumer.OFFSET)
    a = backing[consumer.OFFSET : consumer.OFFSET + consumer.COUNT]
    b, out = torch.zeros(consumer.SHAPE), torch.zeros(consumer.SHAPE)
    args, buffers = consumer._task_args(transport.worker, backing, a, b, out)
    observed = []

    def submit(cid, materialized, cfg, *, caller_stream):
        assert cid == 3 and caller_stream == 17
        assert materialized.tensor_count() == 3
        for index, tensor in enumerate((a, b, out)):
            actual = materialized.tensor(index)
            observed.append(actual.data)
            assert actual.data == tensor.data_ptr()
            assert actual.start_offset == 0
            assert actual.shapes == tuple(tensor.shape)
            assert actual.strides == tensor.stride()
        return SimpleNamespace(wait=lambda timeout: True, done=lambda: True)

    monkeypatch.setattr(transport.worker._chip_worker, "_submit_chip_run_direct", submit)
    run = transport.worker.submit(object(), args, CallConfig(), caller_stream=17)
    run.wait()
    assert len(observed) == 3 and observed[0] == backing.data_ptr() + consumer.OFFSET * 4
    assert not transport.worker._chip_source_uses and not transport.worker._accepted_run_handles
    for buffer in buffers:
        buffer.close()


def _summary(runtime, queue):
    return dict(
        platform="a2a3",
        runtime=runtime,
        task_queue=queue,
        calls=4,
        accepted_calls=4,
        args_dump_supported=True,
        refusals=[],
        levels=[0, 3, 2, 0],
        shape=[128, 128],
        run_epochs=[None, 100, 200, None],
        payload_counts=[0, 6, 3, 0],
        records_per_enabled_call=6,
        caller_thread=10,
        callback_threads=[10 if queue == "0" else 11] * 4,
        guards_verified=True,
        payloads_verified=True,
        config_snapshot_verified=True,
        verified=True,
    )


def _invoke_st(tmp_path, monkeypatch, summary, *, code=0, copies=1, platform="a2a3"):
    module = _load(_ST, "actual_capture_st")
    marker = (
        "FRAMEWORK_ARGUMENT_CAPTURE_PASS "
        if summary["args_dump_supported"]
        else "FRAMEWORK_ARGUMENT_DUMP_REFUSAL_RECOVERY_PASS "
    )
    result = SimpleNamespace(returncode=code, stdout=(marker + json.dumps(summary) + "\n") * copies, stderr="")
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: result)
    request = SimpleNamespace(
        config=SimpleNamespace(getoption=lambda name: {"--device": "0", "--platform": platform}[name])
    )
    module.test_framework_argument_dump_contract("host_build_graph", "1", Path("bridge"), tmp_path, request)


def test_actual_st_requires_complete_result(tmp_path, monkeypatch):
    _invoke_st(tmp_path, monkeypatch, _summary("host_build_graph", "1"))


def _refusal_summary() -> dict:
    summary = _summary("host_build_graph", "1")
    summary.update(
        platform="a5",
        calls=2,
        accepted_calls=2,
        args_dump_supported=False,
        payloads_verified=False,
        run_epochs=[None] * 4,
        payload_counts=[0] * 4,
        records_per_enabled_call=0,
        off_calls_verified=True,
        refusal_outputs_unchanged=True,
        caller_tensors_retained=True,
        caller_stream_after_close=True,
        refusals=[
            dict(
                index=i,
                level=level,
                reason=(
                    f"prepare_native_run failed with code -1001 (run_id=0 slot=0 generation={i + 1} "
                    f"dispatch_id=0 run_epoch={i + 1})"
                ),
            )
            for i, level in [(1, 3), (2, 2)]
        ],
    )
    return summary


def test_actual_st_requires_the_explicit_a5_hbg_refusal_contract(tmp_path, monkeypatch):
    _invoke_st(tmp_path, monkeypatch, _refusal_summary(), platform="a5")


@pytest.mark.parametrize(
    "field,value",
    [
        ("calls", 4),
        ("accepted_calls", 4),
        ("args_dump_supported", True),
        ("payloads_verified", True),
        ("refusals", []),
        ("off_calls_verified", False),
        ("refusal_outputs_unchanged", False),
        ("caller_tensors_retained", False),
        ("caller_stream_after_close", False),
        ("payload_counts", [0, 6, 3, 0]),
        ("run_epochs", [None, 1, 2, None]),
        ("records_per_enabled_call", 6),
        ("guards_verified", False),
        ("config_snapshot_verified", False),
    ],
)
def test_actual_st_rejects_false_a5_hbg_recovery_claims(tmp_path, monkeypatch, field, value):
    summary = _refusal_summary()
    summary[field] = value
    with pytest.raises(AssertionError):
        _invoke_st(tmp_path, monkeypatch, summary, platform="a5")


def test_actual_st_rejects_wrong_a5_hbg_refusal_reason(tmp_path, monkeypatch):
    summary = _refusal_summary()
    summary["refusals"][0]["reason"] = summary["refusals"][0]["reason"].replace("-1001", "-1004")
    with pytest.raises(AssertionError):
        _invoke_st(tmp_path, monkeypatch, summary, platform="a5")


@pytest.mark.parametrize("code,copies", [(1, 1), (0, 0), (0, 2)])
def test_actual_st_rejects_failed_missing_or_duplicate_result(tmp_path, monkeypatch, code, copies):
    with pytest.raises(AssertionError):
        _invoke_st(tmp_path, monkeypatch, _summary("host_build_graph", "1"), code=code, copies=copies)


@pytest.mark.parametrize(
    "field,value",
    [
        ("calls", 3),
        ("levels", [0, 2, 3, 0]),
        ("payloads_verified", False),
        ("run_epochs", [None, 100, 100, None]),
        ("payload_counts", [0, 3, 3, 0]),
        ("callback_threads", [10] * 4),
        ("config_snapshot_verified", False),
    ],
)
def test_actual_st_rejects_incomplete_result(tmp_path, monkeypatch, field, value):
    summary = _summary("host_build_graph", "1")
    summary[field] = value
    with pytest.raises(AssertionError):
        _invoke_st(tmp_path, monkeypatch, summary)
