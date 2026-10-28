# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One eager Worker owns many callables; a clean close admits its successor.

Two sequential generations register X/Y in opposite slot order and execute one
8-call numerical chain on a still-live Torch stream/context. Caller I/O and
boundary guards survive each close; an unresolved operation retains its owners.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import importlib.util
import json
import os
import threading
import weakref
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHAPE = (128, 128)
COUNT = 128 * 128
PADDING = 16
GUARD = -909.0
SCHEDULE = ("x", "y", "y", "x", "x", "y", "x", "y")
EXPECTED = (3, 7, 13, 17, 22, 34, 41, 57)
_UNPROVEN = {}


def _helpers():
    spec = importlib.util.spec_from_file_location("framework_lifecycle_helpers", HERE / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load framework bridge helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compile(runtime, name="x", platform="a2a3"):
    from simpler.task_interface import ArgDirection as D  # noqa: PLC0415

    from simpler_setup.scene_test import compile_chip_callable_spec  # noqa: PLC0415

    assert name in ("x", "y")
    signature = [D.IN, D.IN, D.OUT]
    return compile_chip_callable_spec(
        {
            "orchestration": {
                "source": str(HERE / "kernels/lifecycle_orchestration.cpp"),
                "function_name": "framework_owner_" + name,
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
        f"framework-lifecycle-{platform}-{runtime}-{name}",
    )


def _task_args(worker, backings, tensors):
    import torch  # noqa: PLC0415
    from simpler.buffer import Buffer  # noqa: PLC0415
    from simpler.task_interface import DataType, TaskArgs, Tensor, TensorArgType  # noqa: PLC0415

    assert len(backings) == len(tensors) == 3
    args, buffers = TaskArgs(), []
    for index, (backing, view) in enumerate(zip(backings, tensors)):
        assert backing.dtype == view.dtype == torch.float32 and backing.device == view.device
        assert backing.is_contiguous() and backing.numel() == COUNT + 2 * PADDING
        assert tuple(view.shape) == SHAPE and view.stride() == (SHAPE[1], 1)
        assert view.storage_offset() == PADDING and view.data_ptr() == backing.data_ptr() + PADDING * 4
        assert view.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
        # The registered range is exactly the contiguous view; surrounding
        # Torch-owned storage is retained separately for the guard oracle.
        buffer = Buffer.wrap(address=view.data_ptr(), nbytes=COUNT * 4, location=worker.device_location)
        buffers.append(buffer)
        args.add_tensor(
            Tensor(buffer, shapes=SHAPE, dtype=DataType.FLOAT32),
            TensorArgType.INPUT if index < 2 else TensorArgType.OUTPUT_EXISTING,
        )
    return args, buffers


def _current_context():
    query = ctypes.CDLL("libascendcl.so", mode=os.RTLD_LOCAL | os.RTLD_NOLOAD).aclrtGetCurrentContext
    query.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    query.restype = ctypes.c_int
    context = ctypes.c_void_p()
    status = query(ctypes.byref(context))
    if status != 0 or not context.value:
        raise RuntimeError(f"Cannot observe current caller RTS context: status={status}")
    return context.value


def _frame(torch, worker, stream, index) -> dict:
    backings = tuple(torch.empty(COUNT + 2 * PADDING, dtype=torch.float32, device="npu") for _ in range(3))
    tensors = tuple(backing[PADDING : PADDING + COUNT].view(SHAPE) for backing in backings)
    result = torch.empty(SHAPE, dtype=torch.float32, device="npu")
    args, buffers = _task_args(worker, backings, tensors)
    for tensor in (*backings, *tensors, result):
        tensor.record_stream(stream)
    for backing in backings:
        backing.fill_(GUARD)
    return dict(index=index, backings=backings, tensors=tensors, result=result, args=args, buffers=buffers)


def _assert_caller(torch, state):
    assert _current_context() == state["context"], "caller RTS context changed"
    assert torch.npu.current_stream(device=state["device"]).npu_stream == state["raw_stream"]


def _submit(worker, handle, frame, state):
    assert _current_context() == state["context"], "callback RTS context changed"
    frame["callback_context"] = _current_context()
    frame["run"] = worker.submit(handle, frame["args"], caller_stream=state["raw_stream"])


def _verify(torch, frame):
    index = frame["index"]
    for tensor, value in zip(frame["tensors"], (2 if index == 0 else EXPECTED[index - 1], index + 1, EXPECTED[index])):
        actual = tensor.cpu()
        assert torch.equal(actual, torch.full(SHAPE, value, dtype=actual.dtype)), index
    actual = frame["result"].cpu()
    assert torch.equal(actual, torch.full(SHAPE, EXPECTED[index] * 2, dtype=actual.dtype)), index
    for backing in frame["backings"]:
        actual = backing.cpu()
        assert torch.all(actual[:PADDING] == GUARD) and torch.all(actual[-PADDING:] == GUARD), index
    return frame["tensors"][2].cpu()[0, 0].item()


def run(runtime, device, bridge_path, *, platform="a2a3", successor_runtime=None):
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415, F401 -- registers framework backend
    from simpler.worker import Worker  # noqa: PLC0415

    torch.npu.set_device(device)
    stream = torch.npu.Stream(device=device)
    bridge = _helpers()._load_bridge(bridge_path)
    first_worker = Worker(level=2, device_id=device, platform=platform, runtime=runtime)
    state: dict = dict(device=device, frames=[], workers=[], callbacks=[], closed=[], context=_current_context())
    _UNPROVEN[id(state)] = state
    runtimes = (runtime, successor_runtime or runtime)
    specs = {
        backend: {name: _compile(backend, name, platform=platform) for name in ("x", "y")}
        for backend in dict.fromkeys(runtimes)
    }
    task_queue = os.environ.get("TASK_QUEUE_ENABLE", "1")
    caller_thread = threading.get_native_id()
    refs, observed, environments, slots, orchestration_load_counts = [], [], [], [], []
    with torch.npu.stream(stream):
        state["raw_stream"] = stream.npu_stream
        for generation, order in enumerate((("x", "y"), ("y", "x"))):
            worker = (
                first_worker
                if generation == 0
                else Worker(level=2, device_id=device, platform=platform, runtime=runtimes[generation])
            )
            state["workers"].append(worker)
            handles = {name: worker.register(specs[runtimes[generation]][name]) for name in order}
            originals = tuple(handles.items())
            worker.init(borrowed_context_generation=1)
            _assert_caller(torch, state)
            chip = worker._chip_worker
            assert chip is not None and chip._execution_environment is not None
            chip._validate_execution_context()
            environments.append(tuple(chip._execution_environment))
            slots.append({name: worker._resolve_handle(handle).slot_id for name, handle in handles.items()})
            phase = range(generation * 4, generation * 4 + 4)
            for index in phase:
                frame = _frame(torch, worker, stream, index)
                state["frames"].append(frame)
                refs.extend(weakref.ref(tensor) for tensor in (*frame["backings"], *frame["tensors"], frame["result"]))
                a, b, output = frame["tensors"]
                if index == 0:
                    a.fill_(2)
                else:
                    a.copy_(state["frames"][index - 1]["tensors"][2])
                b.fill_(index + 1)
                output.fill_(-999)
                callback = bridge.enqueue(
                    lambda frame=frame, handle=handles[SCHEDULE[index]]: _submit(worker, handle, frame, state)
                )
                state["callbacks"].append(callback)
                torch.add(output, output, out=frame["result"])
            # All four callbacks and Torch successors are queued before waiting.
            bridge.finish_current_stream()
            for index in phase:
                callback = state["callbacks"][index]
                callback.check()
                assert (callback.thread_id != caller_thread) == (task_queue == "1")
                frame = state["frames"][index]
                frame["run"].wait()
                observed.append(_verify(torch, frame))
            assert tuple(handles.items()) == originals
            # These committed per-runner counts distinguish host orchestration
            # from device orchestration even though both produce the same values.
            counts = [worker.host_dlopen_count, worker.aicpu_dlopen_count]
            assert counts == ([2, 0] if runtimes[generation] == "host_build_graph" else [0, 2]), counts
            orchestration_load_counts.append(counts)
            worker.close()
            state["closed"].append(True)
            for index in phase:
                for buffer in state["frames"][index]["buffers"]:
                    buffer.close()
            _assert_caller(torch, state)
            # Torch business storage and ordinary operators remain usable after
            # each close, including the predecessor for the successor Worker.
            alive = torch.full((1,), 11.0, device="npu")
            alive.add_(6)
            bridge.finish_current_stream()
            assert alive.cpu().item() == 17
        gc.collect()
        assert all(reference() is not None for reference in refs)
        assert slots[0]["x"] == slots[1]["y"] and slots[0]["y"] == slots[1]["x"]
        assert environments[0] != environments[1]
        assert observed == list(EXPECTED)
        # Check every earlier tensor after both close transactions as well.
        assert [_verify(torch, frame) for frame in state["frames"]] == list(EXPECTED)
    summary = dict(
        platform=platform,
        runtime=runtime,
        runtimes=list(runtimes),
        task_queue=task_queue,
        calls=8,
        generations=2,
        generation_calls=[4, 4],
        orchestration_load_counts=orchestration_load_counts,
        observed=observed,
        expected=list(EXPECTED),
        slots=slots,
        original_handles_preserved=True,
        caller_tensors_retained=True,
        guards_verified=True,
        caller_stream_after_each_close=True,
        caller_context=state["context"],
        callback_contexts=[frame["callback_context"] for frame in state["frames"]],
        caller_thread=caller_thread,
        callback_threads=[call.thread_id for call in state["callbacks"]],
        environments=environments,
        verified=True,
    )
    _UNPROVEN.pop(id(state))
    print("FRAMEWORK_LIFECYCLE_PASS " + json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["tensormap_and_ringbuffer", "host_build_graph"], required=True)
    parser.add_argument("--successor-runtime", choices=["tensormap_and_ringbuffer", "host_build_graph"])
    parser.add_argument("--platform", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--build-dir", type=Path, required=True)
    options = parser.parse_args()
    modules = list(options.build_dir.glob("_simpler_framework_queue*.so"))
    if not modules:
        modules = [_helpers().build_bridge(options.build_dir)]
    if len(modules) != 1:
        parser.error("--build-dir must contain exactly one framework queue bridge")
    run(
        options.runtime,
        options.device,
        modules[0],
        platform=options.platform,
        successor_runtime=options.successor_runtime,
    )


if __name__ == "__main__":
    main()
