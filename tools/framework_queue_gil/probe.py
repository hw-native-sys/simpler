# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Diagnose Torch NPU host-queue/GIL waits in isolated, bounded processes.

This probe does not execute a Simpler kernel or qualify a native submit API.
Run hardware cases only inside the repository's device allocation mechanism.
"""

from __future__ import annotations

import argparse
import faulthandler
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def build(directory: Path) -> Path:
    from setuptools import Distribution  # noqa: PLC0415
    from torch.utils.cpp_extension import BuildExtension, CppExtension  # noqa: PLC0415

    spec = importlib.util.find_spec("torch_npu")
    if spec is None or spec.origin is None:
        raise RuntimeError("matching torch and torch_npu are required")
    package = Path(spec.origin).parent
    directory.mkdir(parents=True, exist_ok=True)
    extension = CppExtension(
        name="_framework_queue_gil_probe",
        sources=[str(Path(__file__).with_name("queue_probe.cpp"))],
        include_dirs=[
            str(Path(__file__).resolve().parents[2] / "src/common/worker"),
            str(package / "include"),
            str(package / "include/third_party/acl/inc"),
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
    return Path(command.get_ext_fullpath(extension.name)).resolve()


def child(bridge: Path, device: int, operation: str, callback: str) -> None:
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415

    faulthandler.enable()
    torch.npu.set_device(device)
    stream = torch.npu.Stream(device=device)
    torch.npu.set_stream(stream)
    # Initialization and its queue drain precede the gated callback.
    stream.synchronize()
    spec = importlib.util.spec_from_file_location("_framework_queue_gil_probe", bridge)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load queue probe from {bridge}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Torch NPU drains its first queue entry before returning from enqueue.
    module.warmup()
    print(json.dumps({"torch": torch.__version__, "torch_npu": torch_npu.__version__, "device": device}), flush=True)

    def python_callback():
        pass

    faulthandler.dump_traceback_later(10)
    print("GIL_PROBE ENQUEUE_ENTER", flush=True)
    gate = module.enqueue(python_callback, callback == "python")
    print("GIL_PROBE ENQUEUE_RETURN", flush=True)
    module.wait_ready(gate)
    print("GIL_PROBE CALLBACK_READY", flush=True)
    operations = {
        "query": stream.query,
        "empty_cache": torch.npu.empty_cache,
        "query_released": module.query_released,
        "empty_cache_released": module.empty_cache_released,
    }
    # No incidental interpreter timeslice between opening the gate and drain.
    sys.setswitchinterval(60)
    module.drain(gate, operations[operation])
    faulthandler.cancel_dump_traceback_later()
    print("GIL_PROBE SUCCESS", flush=True)


def classify(log: str, returncode: int | None, timed_out: bool) -> str:
    if timed_out:
        required = ("CALLBACK_READY", "DRAIN_ENTER_GIL_HELD", "CALLBACK_WAIT_GIL")
        if all(marker in log for marker in required) and "CALLBACK_ACQUIRED_GIL" not in log:
            return "gil_wait_timeout"
        return "unclassified_timeout"
    if returncode == 0 and "GIL_PROBE SUCCESS" in log and "CALLBACK_COMPLETE" in log:
        return "completed"
    return "failed"


def run(bridge: Path, device: int, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for operation, callback in (
        ("query", "native"),
        ("empty_cache", "native"),
        ("query_released", "python"),
        ("empty_cache_released", "python"),
        ("query", "python"),
        ("empty_cache", "python"),
    ):
        name = f"{operation}-{callback}"
        env = dict(os.environ, TASK_QUEUE_ENABLE="1", TORCH_DEVICE_BACKEND_AUTOLOAD="0")
        command = [
            sys.executable,
            "-u",
            __file__,
            "child",
            "--bridge",
            str(bridge),
            "--device",
            str(device),
            "--operation",
            operation,
            "--callback",
            callback,
        ]
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        timed_out = False
        try:
            log, _ = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                stack = subprocess.run(
                    [
                        "gdb",
                        "-batch",
                        "-ex",
                        "set pagination off",
                        "-ex",
                        "thread apply all bt",
                        "-p",
                        str(process.pid),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                (output / f"{name}.native-stack.log").write_text(stack.stdout + stack.stderr)
            except (OSError, subprocess.TimeoutExpired) as error:
                (output / f"{name}.native-stack.log").write_text(str(error))
            process.kill()
            log, _ = process.communicate()
        (output / f"{name}.log").write_text(log)
        result = {
            "case": name,
            "outcome": classify(log, process.returncode, timed_out),
            "returncode": process.returncode,
            "timed_out": timed_out,
        }
        results.append(result)
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if len(results) <= 4 and result["outcome"] != "completed":
            raise RuntimeError("positive control failed; remaining cases were not run")
    if any(item["outcome"] != "completed" for item in results[:4]):
        raise RuntimeError("a positive control failed; no GIL attribution accepted")
    if any(item["outcome"] in ("failed", "unclassified_timeout") for item in results[4:]):
        raise RuntimeError("target failure is not classified as a GIL wait")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "run", "child"))
    parser.add_argument("--output", type=Path, default=Path("build/framework-queue-gil"))
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--device", type=int)
    parser.add_argument("--operation", choices=("query", "empty_cache", "query_released", "empty_cache_released"))
    parser.add_argument("--callback", choices=("python", "native"))
    args = parser.parse_args()
    if args.action == "build":
        print(build(args.output))
    elif args.bridge is None or args.device is None:
        parser.error("run/child require --bridge and --device")
    elif args.action == "run":
        run(args.bridge.resolve(), args.device, args.output)
    elif args.operation is None or args.callback is None:
        parser.error("child requires --operation and --callback")
    else:
        child(args.bridge.resolve(), args.device, args.operation, args.callback)


if __name__ == "__main__":
    main()
