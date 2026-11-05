# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real torch_npu queue → public Worker.submit → framework successor.

The optional C++ module only places a Python consumer in torch_npu's own
emission queue. Registration, arguments, execution and retirement stay on
the existing public Worker. No PyPTO process/lifecycle adapter is provided.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import threading
import time
import weakref
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SHAPE = (128, 128)


def build_bridge(directory: Path) -> Path:
    """Compile the optional host extension without importing torch_npu."""
    from setuptools import Distribution  # noqa: PLC0415
    from torch.utils.cpp_extension import BuildExtension, CppExtension  # noqa: PLC0415

    spec = importlib.util.find_spec("torch_npu")
    if spec is None or spec.origin is None:
        raise RuntimeError("This reference consumer requires matching torch and torch_npu")
    package = Path(spec.origin).parent
    directory.mkdir(parents=True, exist_ok=True)
    extension = CppExtension(
        name="_simpler_framework_queue",
        sources=[str(HERE / "queue_callback.cpp")],
        include_dirs=[
            str(package / "include"),
            str(package / "include/third_party/acl/inc"),
            str(package / "include/third_party/hccl/inc"),
        ],
        library_dirs=[str(package / "lib")],
        libraries=["torch_npu"],
        runtime_library_dirs=[str(package / "lib")],
        extra_compile_args=["-O2", "-std=c++17"],
    )
    distribution = Distribution({"ext_modules": [extension], "cmdclass": {"build_ext": BuildExtension}})
    command = distribution.get_command_obj("build_ext")
    command.build_lib = str(directory)
    command.build_temp = str(directory / "objects")
    distribution.run_command("build_ext")
    return Path(command.get_ext_fullpath(extension.name))


def _load_bridge(path: Path):
    spec = importlib.util.spec_from_file_location("_simpler_framework_queue", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load framework queue consumer {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compile(runtime, platform="a2a3"):
    from simpler.task_interface import ArgDirection  # noqa: PLC0415

    from simpler_setup.scene_test import compile_chip_callable_spec  # noqa: PLC0415

    directions = [ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT]
    spec = {
        "orchestration": {
            "source": str(HERE / "kernels/orchestration.cpp"),
            "function_name": "framework_add",
            "signature": directions,
        },
        "incores": [
            {
                "func_id": 0,
                "source": str(HERE.parent / "vector_add/kernels/aiv/vector_add_kernel.cpp"),
                "core_type": "aiv",
                "signature": directions,
            }
        ],
    }
    return compile_chip_callable_spec(spec, platform, runtime, f"framework-eager-{platform}-{runtime}")


def _prepare_frames(torch, worker, handle, stream, raw_stream, calls):
    from simpler.buffer import Buffer  # noqa: PLC0415
    from simpler.task_interface import DataType, TaskArgs, Tensor, TensorArgType  # noqa: PLC0415

    frames, weak_tensors = [], []
    for index in range(calls):
        tensors = tuple(torch.empty(SHAPE, device="npu") for _ in range(3))
        args, buffers = TaskArgs(), []
        for tensor, direction in zip(
            tensors, (TensorArgType.INPUT, TensorArgType.INPUT, TensorArgType.OUTPUT_EXISTING)
        ):
            tensor.record_stream(stream)
            buffer = Buffer.wrap(
                address=tensor.data_ptr(),
                nbytes=tensor.numel() * tensor.element_size(),
                location=worker.device_location,
            )
            buffers.append(buffer)
            args.add_tensor(Tensor(buffer, shapes=SHAPE, dtype=DataType.FLOAT32), direction)
            weak_tensors.append(weakref.ref(tensor))
        result = torch.empty(SHAPE, device="npu")
        result.record_stream(stream)
        frames.append(
            {
                "index": index,
                "args": args,
                "buffers": buffers,
                "tensors": tensors,
                "stream": stream,
                "raw_stream": raw_stream,
                "worker": worker,
                "callable": handle,
                "result": result,
            }
        )
    return frames, weak_tensors


def run(runtime: str, device: int, bridge_path: Path, calls: int = 3, *, platform="a2a3"):
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415, F401 -- registers the NPU framework backend
    from simpler.worker import Worker  # noqa: PLC0415

    bridge = _load_bridge(bridge_path)
    torch.npu.set_device(device)
    stream = torch.npu.Stream(device=device)
    worker = Worker(level=2, device_id=device, platform=platform, runtime=runtime)
    handle = worker.register(_compile(runtime, platform=platform))
    worker.init(borrowed_context_generation=1)
    frames, queued, runs, refusals = [], [], [], []
    caller_thread = threading.get_native_id()
    task_queue = os.environ.get("TASK_QUEUE_ENABLE", "1")
    retired = False
    try:
        with torch.npu.stream(stream):
            # This getter may flush host emission, so resolve its integer once
            # on the caller before the consumer's producer/callback/successor chain.
            raw_stream = stream.npu_stream
            frames, weak_tensors = _prepare_frames(torch, worker, handle, stream, raw_stream, calls)
            # The construction scope and temporary local wrappers have ended.
            # Only the retained frames keep framework tensor objects alive.
            gc.collect()
            assert all(reference() is not None for reference in weak_tensors)
            print("FRAMEWORK_ENQUEUE_BEGIN", flush=True)
            for index, frame in enumerate(frames):
                a, b, output = frame["tensors"]
                started = time.monotonic()
                a.fill_(index + 1)
                b.fill_(index + 3)
                output.fill_(-999)
                print(
                    f"FRAMEWORK_PREDECESSORS_ENQUEUED index={index} seconds={time.monotonic() - started:.3f}",
                    flush=True,
                )

                def submit(frame=frame):
                    print("FRAMEWORK_CALLBACK_ENTER thread=" + str(threading.get_native_id()), flush=True)
                    if frame["index"] == 0:
                        try:
                            frame["worker"].submit(frame["callable"], frame["args"], caller_stream=0)
                        except ValueError:
                            refusals.append("zero caller stream")
                        else:
                            raise AssertionError("invalid caller stream was accepted")
                    runs.append(
                        frame["worker"].submit(frame["callable"], frame["args"], caller_stream=frame["raw_stream"])
                    )
                    print("FRAMEWORK_CALLBACK_SUBMITTED", flush=True)

                queued.append(bridge.enqueue(submit))
                started = time.monotonic()
                torch.add(output, output, out=frame["result"])
                print(
                    f"FRAMEWORK_SUCCESSOR_ENQUEUED index={index} seconds={time.monotonic() - started:.3f}", flush=True
                )
                frame["expected"] = (2 * index + 4) * 2
            print("FRAMEWORK_ENQUEUE_END", flush=True)

            def callback_failure():
                raise RuntimeError("reference callback exception is observable")

            failed_callback = bridge.enqueue(callback_failure)
            # This is the first explicit completion wait: all framework successors were
            # enqueued above. The C++ boundary explicitly releases the GIL.
            bridge.finish_current_stream()
            for call in queued:
                call.check()
                assert (call.thread_id != caller_thread) == (task_queue == "1")
            try:
                failed_callback.check()
            except RuntimeError as error:
                assert "reference callback exception is observable" in str(error)
            else:
                raise AssertionError("callback exception was lost")
            assert refusals == ["zero caller stream"]
            assert len(runs) == calls
            for frame in frames:
                result = frame["result"].cpu()
                assert torch.equal(result, torch.full(SHAPE, frame["expected"], dtype=result.dtype))
        for run_handle in runs:
            run_handle.wait()
        worker.close()
        for frame in frames:
            for buffer in frame["buffers"]:
                buffer.close()
        # The framework owns its stream and context after borrowed Worker close.
        with torch.npu.stream(stream):
            torch.add(frames[0]["result"], frames[0]["result"], out=frames[0]["result"])
            bridge.finish_current_stream()
            result = frames[0]["result"].cpu()
            assert torch.equal(result, torch.full(SHAPE, frames[0]["expected"] * 2, dtype=result.dtype))
        retired = True
        summary = {
            "platform": platform,
            "runtime": runtime,
            "task_queue": task_queue,
            "calls": calls,
            "caller_thread": caller_thread,
            "callback_threads": [call.thread_id for call in queued],
            "verified": True,
            "caller_tensors_retained": all(reference() is not None for reference in weak_tensors),
            "zero_stream_preflight_refusals": len(refusals),
            "callback_error_observed": True,
            "caller_stream_after_close": True,
        }
        print("FRAMEWORK_EAGER_PASS " + json.dumps(summary, sort_keys=True), flush=True)
        return summary
    finally:
        # An unproven device drain is not permission to release caller storage.
        # Keep this failing process's owners reachable for its teardown report.
        if not retired:
            _UNPROVEN.append((worker, handle, frames, queued, runs, stream))
            try:
                os.write(
                    2,
                    b"FRAMEWORK_UNPROVEN_RETAINED: owners kept alive intentionally; "
                    b"exit-time nanobind leak warnings may follow.\n",
                )
            except OSError:
                pass


_UNPROVEN = []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime", choices=["tensormap_and_ringbuffer", "host_build_graph"], default="tensormap_and_ringbuffer"
    )
    parser.add_argument("--platform", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/framework_eager")
    parser.add_argument("--compile-only", action="store_true")
    options = parser.parse_args()
    if options.calls < 1:
        parser.error("--calls must be positive")
    path = build_bridge(options.build_dir)
    print("FRAMEWORK_QUEUE_MODULE " + str(path), flush=True)
    if not options.compile_only:
        run(options.runtime, options.device, path, options.calls, platform=options.platform)


if __name__ == "__main__":
    main()
