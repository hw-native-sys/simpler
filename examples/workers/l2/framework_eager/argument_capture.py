# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Capture task-time arguments after framework producers and before successors.

Two dependent tasks use a runtime-owned intermediate. Supported runtimes check
FULL bytes and HYBRID metadata/selected payloads. A5 HBG checks typed argument-dump
refusal and recovery without claiming per-task payload collection.
Collection and admission may wait; this reference does not measure overlap.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import re
import struct
import threading
import weakref
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHAPE = (128, 128)
COUNT = 128 * 128
OFFSET = 32
GUARD = -909.0
LEVELS = (0, 3, 2, 0)
_UNPROVEN = []


def _helpers():
    spec = importlib.util.spec_from_file_location("framework_capture_helpers", HERE / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load framework bridge helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compile(runtime, platform="a2a3"):
    from simpler.task_interface import ArgDirection as D  # noqa: PLC0415

    from simpler_setup.scene_test import compile_chip_callable_spec  # noqa: PLC0415

    signature = [D.IN, D.IN, D.OUT]
    return compile_chip_callable_spec(
        {
            "orchestration": {
                "source": str(HERE / "kernels/argument_capture_orchestration.cpp"),
                "function_name": "framework_argument_capture",
                "signature": signature,
            },
            "incores": [
                {
                    "func_id": 0,
                    "source": str(HERE.parent / "vector_add/kernels/aiv/vector_add_kernel.cpp"),
                    "core_type": "aiv",
                    "signature": signature,
                }
            ],
        },
        platform,
        runtime,
        f"framework-argument-capture-{platform}-{runtime}",
    )


def _values(invocation):
    a = [float(index % 17 - 8 + 5 * invocation) for index in range(COUNT)]
    b = [float((3 * index) % 13 - 6 - invocation) for index in range(COUNT)]
    return dict(a=a, b=b, mid=[x + y for x, y in zip(a, b)], out=[x + 2 * y for x, y in zip(a, b)])


def _dump_files(folder):
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix in (".json", ".bin")
        and (
            any(part.startswith("args_dump") for part in path.relative_to(folder).parts)
            or path.name.startswith("args.e")
        )
    )


def _check_dump(folder, invocation, level):
    artifacts = _dump_files(folder)
    if level == 0:
        assert not artifacts, artifacts
        return None
    dump = folder / "args_dump"
    manifest = dump / "args_dump.json"
    data = json.loads(manifest.read_text())
    assert data["run_dir"] == "args_dump"
    assert data["dump_args_level"] == level
    assert data["bin_format"] == {"type": "logical_contiguous", "byte_order": "little_endian"}
    for name, expected in dict(
        total_args=6,
        before_dispatch=4,
        after_completion=2,
        input_args=4,
        output_args=2,
        inout_args=0,
        truncated_args=0,
        dropped_records=0,
    ).items():
        assert data[name] == expected, (name, data)
    assert data["dropped_records_complete"] is True
    assert not data.get("counts_unknown", False)
    assert "collection_verdict" not in data
    assert not data.get("host_discarded_args", 0) and not data.get("metadata_discarded_records", 0)
    assert data["bin_file"] == "args.bin"
    binary = dump / data["bin_file"]
    payload = binary.read_bytes()
    assert artifacts == sorted([manifest, binary]), artifacts
    records = data["args"]
    assert len(records) == 6
    epochs = {record["run_epoch"] for record in records}
    assert len(epochs) == 1 and all(type(epoch) is int and epoch > 0 for epoch in epochs)
    tasks = {}
    for record in records:
        assert record["kind"] == "tensor" and record["dtype"] == "FLOAT32"
        assert record["numel"] == COUNT and record["is_contiguous"] is True
        assert record["truncated"] is False and not record.get("host_discarded", False)
        assert record["func_id"] == [0]
        key = (record["arg_index"], record["stage"], record["role"])
        group = tasks.setdefault(record["task_id"], {})
        assert key not in group, (record["task_id"], key)
        group[key] = record
    assert len(tasks) == 2
    input_key = (0, "before_dispatch", "input")
    # Materialization folds the caller byte offset into A's address. Its 1D
    # shape distinguishes task0 from task1's 2D runtime-owned intermediate.
    first = [group for group in tasks.values() if group[input_key]["shape"] == [COUNT]]
    second = [group for group in tasks.values() if group[input_key]["shape"] == list(SHAPE)]
    assert len(first) == len(second) == 1
    expected = _values(invocation)
    intervals = []
    for task, names in enumerate((("a", "b", "mid"), ("mid", "b", "out"))):
        group = (first, second)[task][0]
        assert set(group) == {
            (0, "before_dispatch", "input"),
            (1, "before_dispatch", "input"),
            (2, "after_completion", "output"),
        }
        for slot, name in enumerate(names):
            record = group[
                (slot, "before_dispatch" if slot < 2 else "after_completion", "input" if slot < 2 else "output")
            ]
            assert record["start_offset"] == 0
            assert record["shape"] == ([COUNT] if name == "a" else list(SHAPE))
            assert record["strides"] == ([1] if name == "a" else [SHAPE[1], 1])
            selected = level == 3 or (task, slot) in ((0, 2), (1, 0), (1, 2))
            size = record["bin_size"]
            assert size == (COUNT * 4 if selected else 0), record
            if selected:
                offset = record["bin_offset"]
                assert type(offset) is int and 0 <= offset <= len(payload) - size
                wanted = struct.pack("<" + "f" * COUNT, *expected[name])
                assert payload[offset : offset + size] == wanted, (invocation, task, slot, name)
                intervals.append((offset, offset + size))
    assert len(payload) == (6 if level == 3 else 3) * COUNT * 4
    ordered = sorted(intervals)
    assert ordered[0][0] == 0 and ordered[-1][1] == len(payload)
    assert all(left[1] == right[0] for left, right in zip(ordered, ordered[1:]))
    return next(iter(epochs))


def _task_args(worker, backing, a, b, out):
    import torch  # noqa: PLC0415
    from simpler.buffer import Buffer  # noqa: PLC0415
    from simpler.task_interface import DataType, TaskArgs, Tensor, TensorArgType  # noqa: PLC0415

    assert backing.numel() == COUNT + 2 * OFFSET and backing.storage_offset() == 0
    assert tuple(a.shape) == (COUNT,) and tuple(b.shape) == tuple(out.shape) == SHAPE
    assert a.stride() == (1,) and b.stride() == out.stride() == (SHAPE[1], 1)
    assert a.storage_offset() == OFFSET and a.data_ptr() == backing.data_ptr() + OFFSET * 4
    assert a.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
    assert b.storage_offset() == out.storage_offset() == 0
    assert all(tensor.dtype == torch.float32 for tensor in (backing, a, b, out))
    assert backing.is_contiguous() and all(tensor.device == backing.device for tensor in (a, b, out))
    args, buffers = TaskArgs(), []
    for storage, offset, shapes, direction in zip(
        (backing, b, out),
        (OFFSET, 0, 0),
        ((COUNT,), SHAPE, SHAPE),
        (TensorArgType.INPUT, TensorArgType.INPUT, TensorArgType.OUTPUT_EXISTING),
    ):
        buffer = Buffer.wrap(address=storage.data_ptr(), nbytes=storage.numel() * 4, location=worker.device_location)
        buffers.append(buffer)
        args.add_tensor(Tensor(buffer, shapes=shapes, dtype=DataType.FLOAT32, byte_offset=offset * 4), direction)
    return args, buffers


def _submit(frame, runs):
    runs.append(
        frame["worker"].submit(frame["callable"], frame["args"], frame["config"], caller_stream=frame["raw_stream"])
    )
    frame["config"].enable_dump_args = 1
    frame["config"].output_prefix = str(frame["poison"])


def _expect_dump_refusal(frame, runs, refusals):
    from simpler.task_interface import RunPreparationRefused  # noqa: PLC0415

    accepted = len(runs)
    try:
        _submit(frame, runs)
    except RunPreparationRefused as error:
        reason = str(error)
        assert re.fullmatch(
            r"prepare_native_run failed with code -1001 "
            r"\(run_id=\d+ slot=\d+ generation=\d+ dispatch_id=\d+ run_epoch=\d+\)",
            reason,
        ), reason
        assert len(runs) == accepted
        refusals.append(dict(index=frame["index"], level=frame["level"], reason=reason))
        frame["config"].enable_dump_args = 1
        frame["config"].output_prefix = str(frame["poison"])
    else:
        raise AssertionError("A5 HBG accepted unsupported per-task argument capture")


def _check_outputs(frames, args_dump_supported, torch):
    for index, frame in enumerate(frames):
        backing, a, b, out, result = frame["tensors"]
        if not args_dump_supported and frame["level"]:
            assert out.cpu().flatten().tolist() == [-999.0] * COUNT
            assert result.cpu().flatten().tolist() == [-1998.0] * COUNT
            assert a.cpu().flatten().tolist() == _values(index)["a"]
            assert b.cpu().flatten().tolist() == _values(index)["b"]
        else:
            assert result.cpu().flatten().tolist() == [2 * value for value in _values(index)["out"]]
        host_backing = backing.cpu()
        assert torch.all(host_backing[:OFFSET] == GUARD) and torch.all(host_backing[-OFFSET:] == GUARD)


def _check_dump_contract(root, frames, args_dump_supported, stream, bridge, torch):
    if args_dump_supported:
        epochs = [_check_dump(root / str(index), index, level) for index, level in enumerate(LEVELS)]
        assert len({epoch for epoch in epochs if epoch is not None}) == 2
    else:
        epochs = [_check_dump(root / str(index), index, 0) for index in range(len(LEVELS))]
        assert epochs == [None] * len(LEVELS) and not _dump_files(root)
        with torch.npu.stream(stream):
            frames[-1]["tensors"][-1].add_(1)
            bridge.finish_current_stream()
            assert frames[-1]["tensors"][-1].cpu().flatten().tolist() == [
                2 * value + 1 for value in _values(len(LEVELS) - 1)["out"]
            ]
    return epochs


def run(runtime, device, bridge_path, root, *, platform="a2a3"):
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415, F401 -- registers the framework backend
    from simpler.task_interface import CallConfig  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    root.mkdir(parents=True, exist_ok=False)
    poison = root / "poison"
    poison.mkdir()
    bridge = _helpers()._load_bridge(bridge_path)
    torch.npu.set_device(device)
    stream = torch.npu.Stream(device=device)
    worker = Worker(level=2, device_id=device, platform=platform, runtime=runtime)
    handle = worker.register(_compile(runtime, platform=platform))
    frames, queued, runs, refusals = [], [], [], []
    args_dump_supported = not (platform == "a5" and runtime == "host_build_graph")
    retired = False
    try:
        worker.init(borrowed_context_generation=1)
        base = _values(0)
        references = []
        with torch.npu.stream(stream):
            for index, level in enumerate(LEVELS):
                backing = torch.empty(COUNT + 2 * OFFSET, dtype=torch.float32, device="npu")
                a = backing[OFFSET : OFFSET + COUNT]
                b, out, result = (torch.empty(SHAPE, dtype=torch.float32, device="npu") for _ in range(3))
                args, buffers = _task_args(worker, backing, a, b, out)
                for tensor in (backing, a, b, out, result):
                    tensor.record_stream(stream)
                    references.append(weakref.ref(tensor))
                folder = root / str(index)
                folder.mkdir()
                config = CallConfig()
                config.enable_dump_args, config.output_prefix = level, str(folder)
                frames.append(
                    dict(
                        index=index,
                        level=level,
                        worker=worker,
                        callable=handle,
                        args=args,
                        buffers=buffers,
                        tensors=(backing, a, b, out, result),
                        config=config,
                        raw_stream=stream.npu_stream,
                        poison=poison,
                    )
                )
                backing.fill_(GUARD)
                a.copy_(torch.tensor(base["a"]))
                b.copy_(torch.tensor(base["b"]).view(SHAPE))
            gc.collect()
            assert all(reference() is not None for reference in references)
            caller_thread, task_queue = threading.get_native_id(), os.environ.get("TASK_QUEUE_ENABLE", "1")
            for index, frame in enumerate(frames):
                _, a, b, out, result = frame["tensors"]
                a.add_(5 * index)
                b.add_(-index)
                out.fill_(-999)
                if not args_dump_supported and frame["level"]:
                    queued.append(bridge.enqueue(lambda frame=frame: _expect_dump_refusal(frame, runs, refusals)))
                else:
                    queued.append(bridge.enqueue(lambda frame=frame: _submit(frame, runs)))
                torch.add(out, out, out=result)
            bridge.finish_current_stream()
        for callback in queued:
            callback.check()
            assert (callback.thread_id != caller_thread) == (task_queue == "1")
        assert len(runs) == (len(LEVELS) if args_dump_supported else 2)
        assert [(item["index"], item["level"]) for item in refusals] == (
            [] if args_dump_supported else [(1, 3), (2, 2)]
        )
        _check_outputs(frames, args_dump_supported, torch)
        for run_handle in runs:
            run_handle.wait()
        worker.close()
        for frame in frames:
            for buffer in frame["buffers"]:
                buffer.close()
        epochs = _check_dump_contract(root, frames, args_dump_supported, stream, bridge, torch)
        assert not _dump_files(poison)
        assert all(reference() is not None for reference in references)
        summary = dict(
            platform=platform,
            runtime=runtime,
            task_queue=task_queue,
            calls=len(runs),
            accepted_calls=len(runs),
            args_dump_supported=args_dump_supported,
            refusals=refusals,
            levels=list(LEVELS),
            shape=list(SHAPE),
            run_epochs=epochs,
            payload_counts=[0, 6, 3, 0] if args_dump_supported else [0, 0, 0, 0],
            records_per_enabled_call=6 if args_dump_supported else 0,
            caller_thread=caller_thread,
            callback_threads=[callback.thread_id for callback in queued],
            guards_verified=True,
            payloads_verified=args_dump_supported,
            config_snapshot_verified=all(
                frame["config"].enable_dump_args == 1 and frame["config"].output_prefix == str(poison)
                for frame in frames
            ),
            verified=True,
        )
        if not args_dump_supported:
            summary.update(
                off_calls_verified=True,
                refusal_outputs_unchanged=True,
                caller_tensors_retained=True,
                caller_stream_after_close=True,
            )
        retired = True
        marker = (
            "FRAMEWORK_ARGUMENT_CAPTURE_PASS "
            if args_dump_supported
            else "FRAMEWORK_ARGUMENT_DUMP_REFUSAL_RECOVERY_PASS "
        )
        print(marker + json.dumps(summary, sort_keys=True), flush=True)
        return summary
    finally:
        if not retired:
            _UNPROVEN.append((worker, handle, frames, queued, runs, stream))
            try:
                os.write(2, b"FRAMEWORK_ARGUMENT_CAPTURE_UNPROVEN: caller owners retained after failure.\n")
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["tensormap_and_ringbuffer", "host_build_graph"], required=True)
    parser.add_argument("--platform", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    modules = list(options.build_dir.glob("_simpler_framework_queue*.so"))
    if not modules:
        modules = [_helpers().build_bridge(options.build_dir)]
    if len(modules) != 1:
        parser.error("--build-dir must contain exactly one framework queue bridge")
    run(options.runtime, options.device, modules[0], options.output_dir, platform=options.platform)


if __name__ == "__main__":
    main()
