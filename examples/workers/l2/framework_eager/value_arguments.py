# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Typed scalar values and explicit HOST/NONE controls in framework callbacks.

The callback submits fresh values, then clears the caller's TaskArgs and mutates
its scalar wrappers. HOST backing stays owned until retirement even though its
post-submit contents differ from the control consumed by host orchestration.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import importlib.util
import json
import os
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHAPE = (64, 64)
_VALUES = (
    (1.25, -3, 0x8000000000000001, 1.0 + 2.0**-40),
    (-0.5, 7, 0x4000000000000002, 1.0 - 2.0**-40),
    (2.0, -8, 0x8000000000000003, -0.0),
)
_CONTROLS = (5, -7, 9)
_UNPROVEN = []


def _reference():
    spec = importlib.util.spec_from_file_location("framework_value_reference", HERE / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load framework bridge helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compile(runtime, platform="a2a3", *, host_control=False):
    from simpler.task_interface import ArgDirection as D  # noqa: PLC0415

    from simpler_setup.scene_test import compile_chip_callable_spec  # noqa: PLC0415

    signature = [D.IN, D.OUT] + ([D.IN] if host_control else []) + [D.SCALAR] * 4
    entry = "framework_host_value_arguments" if host_control else "framework_value_arguments"
    return compile_chip_callable_spec(
        {
            "orchestration": {
                "source": str(HERE / "kernels/value_arguments_orchestration.cpp"),
                "function_name": entry,
                "signature": signature,
            },
            "incores": [
                {
                    "func_id": 0,
                    "source": str(HERE / "kernels/aiv/value_arguments.cpp"),
                    "core_type": "aiv",
                    "signature": [D.IN, D.OUT, D.SCALAR, D.SCALAR],
                }
            ],
        },
        platform,
        runtime,
        f"framework-values-{platform}-{runtime}-{host_control}",
    )


def _arguments(tensors, values, *, host_control=None):
    from simpler.task_interface import TaskArgs, TensorArgType, TensorTransfer  # noqa: PLC0415

    args = TaskArgs()
    args.add_tensor(tensors[0], TensorArgType.INPUT, transfer=TensorTransfer.NONE)
    args.add_tensor(tensors[1], TensorArgType.OUTPUT_EXISTING, transfer=TensorTransfer.NONE)
    if host_control is not None:
        args.add_tensor(host_control, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
    wrappers = (
        ctypes.c_float(values[0]),
        ctypes.c_int8(values[1]),
        ctypes.c_uint64(values[2]),
        ctypes.c_double(values[3]),
    )
    for value in wrappers:
        args.add_scalar(value)
    return args, wrappers


def _expected(index, host_control=False):
    # Hand-derived constants preserve independent width/sign/high-bit checks.
    # The double in call0 is >1 only at FLOAT64 precision; call2 is negative zero.
    scale, bias = ((1.25, 46), (-0.5, 9), (2.0, 75))[index]
    return (index + 2) * scale + bias + (_CONTROLS[index] if host_control else 0) + 0.5


def run(runtime, device, bridge_path, *, platform="a2a3"):
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415, F401 -- registers the NPU framework backend
    from simpler.buffer import Buffer  # noqa: PLC0415
    from simpler.task_interface import DataType, RunPreparationRefused, Tensor  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    reference = _reference()
    bridge = reference._load_bridge(bridge_path)
    torch.npu.set_device(device)
    stream = torch.npu.Stream(device=device)
    raw_stream = stream.npu_stream
    worker = Worker(level=2, device_id=device, platform=platform, runtime=runtime)
    handles = {False: worker.register(_compile(runtime, platform=platform))}
    if runtime == "host_build_graph":
        handles[True] = worker.register(_compile(runtime, platform=platform, host_control=True))
    worker.init(borrowed_context_generation=1)
    frames, queued, runs, refusals = [], [], [], []
    caller_thread, queue = threading.get_native_id(), os.environ.get("TASK_QUEUE_ENABLE", "1")
    retired = False
    try:
        # CPU backing remains caller-owned through every run's finalize.
        control = torch.zeros(1, dtype=torch.int32)
        control_arg = worker.make_tensor_arg(control, shapes=(1,), dtype=DataType.INT32)
        refusal_source = torch.zeros(SHAPE, dtype=torch.float32)
        refusal_arg = worker.make_tensor_arg(refusal_source, shapes=SHAPE, dtype=DataType.FLOAT32)
        with torch.npu.stream(stream):
            for host_control in handles:
                for index in range(3):
                    source, output, successor = (torch.empty(SHAPE, device="npu") for _ in range(3))
                    buffers = tuple(
                        Buffer.wrap(address=t.data_ptr(), nbytes=t.numel() * 4, location=worker.device_location)
                        for t in (source, output)
                    )
                    tensors = tuple(Tensor(buffer, shapes=SHAPE, dtype=DataType.FLOAT32) for buffer in buffers)
                    for tensor in (source, output, successor):
                        tensor.record_stream(stream)
                    frames.append(
                        dict(
                            index=index,
                            host_control=host_control,
                            source=source,
                            output=output,
                            successor=successor,
                            buffers=buffers,
                            tensors=tensors,
                        )
                    )
            gc.collect()
            for frame in frames:
                index, host_control = frame["index"], frame["host_control"]
                frame["source"].fill_(index + 2)
                frame["output"].fill_(-999)

                def submit(frame=frame):
                    index, host_control = frame["index"], frame["host_control"]
                    if runtime == "tensormap_and_ringbuffer" and index == 0:
                        invalid, _ = _arguments((refusal_arg, frame["tensors"][1]), _VALUES[index])
                        try:
                            worker.submit(handles[False], invalid, caller_stream=raw_stream)
                        except RunPreparationRefused:
                            refusals.append("HOST/NONE unsupported")
                        else:
                            raise AssertionError("TMR accepted a HOST/NONE input")
                    control.fill_(_CONTROLS[index])
                    args, wrappers = _arguments(
                        frame["tensors"], _VALUES[index], host_control=control_arg if host_control else None
                    )
                    runs.append(worker.submit(handles[host_control], args, caller_stream=raw_stream))
                    args.clear()
                    for wrapper in wrappers:
                        wrapper.value = 0
                    control.fill_(1000 + index)
                    frame["caller_args_cleared"] = len(args) == 0
                    frame["wrappers_mutated"] = all(wrapper.value == 0 for wrapper in wrappers)

                queued.append(bridge.enqueue(submit))
                torch.add(frame["output"], 0.5, out=frame["successor"])
            bridge.finish_current_stream()
            for call in queued:
                call.check()
                assert (call.thread_id != caller_thread) == (queue == "1")
            observed = []
            for frame in frames:
                actual = frame["successor"].cpu()
                expected = _expected(frame["index"], frame["host_control"])
                assert torch.equal(actual, torch.full(SHAPE, expected)), (frame["index"], frame["host_control"])
                assert frame["caller_args_cleared"] and frame["wrappers_mutated"]
                observed.append(actual[0, 0].item())
        assert len(runs) == len(frames) == (6 if runtime == "host_build_graph" else 3)
        assert refusals == (["HOST/NONE unsupported"] if runtime == "tensormap_and_ringbuffer" else [])
        assert control.item() == 1002
        for run_handle in runs:
            run_handle.wait()
        worker.close()
        for frame in frames:
            for buffer in frame["buffers"]:
                buffer.close()
        retired = True
        summary = dict(
            platform=platform,
            runtime=runtime,
            task_queue=queue,
            calls=len(runs),
            observed=observed,
            caller_thread=caller_thread,
            callback_threads=[call.thread_id for call in queued],
            caller_args_cleared=True,
            wrappers_mutated=True,
            host_backing_retained=True,
            host_controls_changed=runtime == "host_build_graph",
            clean_host_refusals=len(refusals),
            verified=True,
        )
        print("FRAMEWORK_VALUE_ARGUMENTS_PASS " + json.dumps(summary, sort_keys=True), flush=True)
        return summary
    finally:
        if not retired:
            _UNPROVEN.append(
                (
                    worker,
                    handles,
                    frames,
                    queued,
                    runs,
                    stream,
                    locals().get("control"),
                    locals().get("refusal_source"),
                    locals().get("control_arg"),
                    locals().get("refusal_arg"),
                )
            )
            try:
                os.write(2, b"FRAMEWORK_VALUES_UNPROVEN: caller owners retained after failure.\n")
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["tensormap_and_ringbuffer", "host_build_graph"], required=True)
    parser.add_argument("--platform", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--build-dir", type=Path, required=True)
    options = parser.parse_args()
    modules = list(options.build_dir.glob("_simpler_framework_queue*.so"))
    if len(modules) > 1:
        parser.error("--build-dir contains multiple framework queue bridges")
    path = modules[0] if modules else _reference().build_bridge(options.build_dir)
    run(options.runtime, options.device, path, platform=options.platform)


if __name__ == "__main__":
    main()
