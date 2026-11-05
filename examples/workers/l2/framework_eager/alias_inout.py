# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Framework-owned aliased views through the emission queue and public Worker.

Each invocation computes y = snapshot(x) + snapshot(y). Both views share one
caller allocation. One AIV task loads both complete views before writing y;
partial overlap therefore has the same snapshot semantics as complete overlap.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import threading
import weakref
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHAPE = (64, 64)
STRIDES = (96, 1)
GUARD = 64
BACKING_ELEMENTS = 2 * GUARD + SHAPE[0] * STRIDES[0]
ALIASES = ("full", "partial")
_UNPROVEN = []


def _bridge_helpers():
    spec = importlib.util.spec_from_file_location("framework_queue_reference", HERE / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the framework queue build helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _offsets(alias):
    if alias not in ALIASES:
        raise ValueError(f"Unknown alias layout: {alias}")
    return GUARD, GUARD + (8 if alias == "partial" else 0)


def _validate_views(backing, source, target, alias):
    import torch  # noqa: PLC0415

    if backing.dtype != torch.float32 or backing.ndim != 1 or not backing.is_contiguous():
        raise ValueError("Alias backing must be a contiguous FLOAT32 vector")
    if backing.numel() != BACKING_ELEMENTS or backing.storage_offset() != 0:
        raise ValueError("Alias backing has the wrong extent or origin")
    for view, offset in zip((source, target), _offsets(alias)):
        if view.dtype != backing.dtype or tuple(view.shape) != SHAPE or view.stride() != STRIDES:
            raise ValueError("Alias views require 64x64 FLOAT32 geometry with strides (96, 1)")
        if view.untyped_storage().data_ptr() != backing.untyped_storage().data_ptr():
            raise ValueError("Both alias views must share the caller backing")
        if view.storage_offset() != offset or view.data_ptr() != backing.data_ptr() + offset * backing.element_size():
            raise ValueError("Alias view has the wrong origin")


def _task_args(worker, backing, views, alias):
    from simpler.buffer import Buffer  # noqa: PLC0415
    from simpler.task_interface import DataType, TaskArgs, Tensor, TensorArgType  # noqa: PLC0415

    _validate_views(backing, *views, alias)
    buffer = Buffer.wrap(
        address=backing.data_ptr(),
        nbytes=backing.numel() * backing.element_size(),
        location=worker.device_location,
    )
    arguments = tuple(
        Tensor(buffer, shapes=SHAPE, dtype=DataType.FLOAT32, strides=STRIDES, byte_offset=offset * 4)
        for offset in _offsets(alias)
    )
    args = TaskArgs()
    args.add_tensor(arguments[0], TensorArgType.INPUT)
    args.add_tensor(arguments[1], TensorArgType.INOUT)
    return args, buffer, arguments


def _compile(runtime, platform="a2a3"):
    from simpler.task_interface import ArgDirection  # noqa: PLC0415

    from simpler_setup.scene_test import compile_chip_callable_spec  # noqa: PLC0415

    signature = [ArgDirection.IN, ArgDirection.INOUT]
    return compile_chip_callable_spec(
        {
            "orchestration": {
                "source": str(HERE / "kernels/alias_inout_orchestration.cpp"),
                "function_name": "framework_alias_inout",
                "signature": signature,
            },
            "incores": [
                {
                    "func_id": 0,
                    "source": str(HERE / "kernels/aiv/alias_inout.cpp"),
                    "core_type": "aiv",
                    "signature": signature,
                }
            ],
        },
        platform,
        runtime,
        f"framework-alias-inout-{platform}-{runtime}",
    )


def _reference_step(backing, alias, index):
    source_offset, target_offset = _offsets(alias)
    backing.add_(index + 1)
    source = backing.as_strided(SHAPE, STRIDES, source_offset).clone()
    target = backing.as_strided(SHAPE, STRIDES, target_offset)
    target.copy_(source + target.clone())
    return backing + index + 2


def _prepare_frames(torch, worker, handle, stream, raw_stream, calls):
    frames, weak_tensors = [], []
    for alias in ALIASES:
        initial = torch.arange(BACKING_ELEMENTS, dtype=torch.float32).remainder_(19)
        backing = torch.empty(BACKING_ELEMENTS, dtype=torch.float32, device="npu")
        views = tuple(backing.as_strided(SHAPE, STRIDES, offset) for offset in _offsets(alias))
        snapshots = [torch.empty_like(backing) for _ in range(calls)]
        args, buffer, arguments = _task_args(worker, backing, views, alias)
        for tensor in (backing, *views, *snapshots):
            tensor.record_stream(stream)
            weak_tensors.append(weakref.ref(tensor))
        frames.append(
            {
                "alias": alias,
                "initial": initial,
                "reference": initial.clone(),
                "backing": backing,
                "views": views,
                "snapshots": snapshots,
                "expected": [],
                "args": args,
                "buffer": buffer,
                "argument_views": arguments,
                "stream": stream,
                "raw_stream": raw_stream,
                "worker": worker,
                "callable": handle,
            }
        )
    return frames, weak_tensors


def run(runtime, device, bridge_path, calls=3, *, platform="a2a3"):
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415, F401 -- registers the framework backend
    from simpler.worker import Worker  # noqa: PLC0415

    bridge = _bridge_helpers()._load_bridge(bridge_path)
    torch.npu.set_device(device)
    stream = torch.npu.Stream(device=device)
    worker = Worker(level=2, device_id=device, platform=platform, runtime=runtime)
    handle = worker.register(_compile(runtime, platform=platform))
    worker.init(borrowed_context_generation=1)
    frames, queued, runs = [], [], []
    caller_thread = threading.get_native_id()
    task_queue = os.environ.get("TASK_QUEUE_ENABLE", "1")
    retired = False
    try:
        with torch.npu.stream(stream):
            raw_stream = stream.npu_stream
            frames, weak_tensors = _prepare_frames(torch, worker, handle, stream, raw_stream, calls)
            # Caller frames own all backing, views and parameter wrappers after construction.
            gc.collect()
            assert all(reference() is not None for reference in weak_tensors)
            for frame in frames:
                frame["backing"].copy_(frame["initial"])
            for frame in frames:
                for index in range(calls):
                    frame["backing"].add_(index + 1)

                    def submit(frame=frame):
                        runs.append(
                            frame["worker"].submit(frame["callable"], frame["args"], caller_stream=frame["raw_stream"])
                        )

                    queued.append(bridge.enqueue(submit))
                    torch.add(frame["backing"], index + 2, out=frame["snapshots"][index])
                    frame["expected"].append(_reference_step(frame["reference"], frame["alias"], index))
            bridge.finish_current_stream()
            assert len(runs) == calls * len(ALIASES)
            for call in queued:
                call.check()
                assert (call.thread_id != caller_thread) == (task_queue == "1")
            for frame in frames:
                untouched = torch.ones(BACKING_ELEMENTS, dtype=torch.bool)
                untouched.as_strided(SHAPE, STRIDES, _offsets(frame["alias"])[1]).fill_(False)
                for index, (snapshot, expected) in enumerate(zip(frame["snapshots"], frame["expected"])):
                    actual = snapshot.cpu()
                    assert torch.equal(actual, expected), f"alias={frame['alias']} invocation={index} backing mismatch"
                    assert torch.equal(actual[untouched], expected[untouched]), "guard or padding was modified"
            assert all(reference() is not None for reference in weak_tensors)
        for run_handle in runs:
            run_handle.wait()
        worker.close()
        for frame in frames:
            frame["buffer"].close()
        retired = True
        summary = {
            "platform": platform,
            "runtime": runtime,
            "task_queue": task_queue,
            "aliases": list(ALIASES),
            "calls_per_alias": calls,
            "calls": len(runs),
            "shape": list(SHAPE),
            "strides": list(STRIDES),
            "offsets": {alias: [offset * 4 for offset in _offsets(alias)] for alias in ALIASES},
            "caller_thread": caller_thread,
            "callback_threads": [call.thread_id for call in queued],
            "caller_frame_retained": all(reference() is not None for reference in weak_tensors),
            "guard_verified": True,
            "state_reused": True,
            "verified": True,
        }
        print("FRAMEWORK_ALIAS_INOUT_PASS " + json.dumps(summary, sort_keys=True), flush=True)
        return summary
    finally:
        if not retired:
            _UNPROVEN.append((worker, handle, frames, queued, runs, stream))
            try:
                os.write(2, b"FRAMEWORK_ALIAS_UNPROVEN_RETAINED: caller owners retained after unproven completion.\n")
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["tensormap_and_ringbuffer", "host_build_graph"], required=True)
    parser.add_argument("--platform", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=3)
    options = parser.parse_args()
    if options.calls < 2:
        parser.error("--calls must be at least two to qualify state reuse")
    modules = list(options.build_dir.glob("_simpler_framework_queue*.so"))
    if not modules:
        modules = [_bridge_helpers().build_bridge(options.build_dir)]
    if len(modules) != 1:
        parser.error("--build-dir must contain exactly one framework queue bridge")
    run(options.runtime, options.device, modules[0], options.calls, platform=options.platform)


if __name__ == "__main__":
    main()
