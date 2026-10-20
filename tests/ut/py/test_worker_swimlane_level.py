# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Successive public calls select DFX granularity from their own descriptors."""

import array
import ctypes
import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_kernel_caller_stream import _callable
from test_kernel_mode_c_api import _CallerContexts, _load


def _run(arch, runtime, device, borrowed, root):
    from simpler.task_interface import (  # noqa: PLC0415
        CallConfig,
        DataType,
        TaskArgs,
        Tensor,
        TensorArgType,
    )
    from simpler.worker import Worker  # noqa: PLC0415

    lib = _load(arch, "onboard", runtime)
    contexts = None
    worker = None
    lib.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.aclrtDestroyStream.argtypes = [ctypes.c_void_p]
    lib.aclrtSynchronizeStreamWithTimeout.argtypes = [ctypes.c_void_p, ctypes.c_int]
    stream = ctypes.c_void_p()
    buffers = []
    try:
        if borrowed:
            lib.rtSetDevice.argtypes = [ctypes.c_int]
            assert lib.rtSetDevice(device) == 0
            contexts = _CallerContexts(lib, device)
        worker = Worker(level=2, device_id=device, platform=arch, runtime=runtime)
        handle = worker.register(_callable(arch, runtime))
        worker.init(borrowed_context_generation=77 if borrowed else None)
        assert lib.aclrtCreateStream(ctypes.byref(stream)) == 0
        count = 128 * 128
        for _ in range(3):
            buffers.append(worker.malloc(count * 4))
        worker.copy_to(buffers[0], array.array("f", [2.0]) * count)
        worker.copy_to(buffers[1], array.array("f", [3.0]) * count)
        args = TaskArgs()
        for i, backing in enumerate(buffers):
            args.add_tensor(
                Tensor(backing, shapes=(count,), dtype=DataType.FLOAT32),
                TensorArgType.INPUT if i < 2 else TensorArgType.OUTPUT_EXISTING,
            )
        for invocation, level in enumerate((4, 1, 0, 3, 2)):
            folder = root / str(invocation)
            folder.mkdir()
            config = CallConfig()
            config.enable_chip_swimlane = level
            config.output_prefix = str(folder)
            worker.submit(handle, args, config, caller_stream=stream.value).wait()
            result = array.array("f", [0.0]) * count
            worker.copy_from(result, buffers[2])
            expected = 47.0 if runtime == "tensormap_and_ringbuffer" else 42.0
            assert all(value == expected for value in result), (level, result[:8])
            artifact = folder / "chip_swimlane_records.json"
            if level == 0:
                assert not artifact.exists()
                continue
            data = json.loads(artifact.read_text())
            assert data["chip_swimlane_level"] == level
            assert data["aicore_tasks"]
            if level >= 3:
                assert any(data["scheduler_records"]), (level, data["scheduler_records"])
            else:
                assert "scheduler_records" not in data
    finally:
        # The caller stream may still hold a wait after the kernel boundaries.
        if stream.value:
            assert lib.aclrtSynchronizeStreamWithTimeout(stream, 60000) == 0
        if worker is not None:
            for backing in buffers:
                worker.free(backing)
        if stream.value:
            assert lib.aclrtDestroyStream(stream) == 0
        if worker is not None:
            worker.close()
        if contexts is not None:
            contexts.close()


@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize(
    ("arch", "runtime"),
    [
        pytest.param(arch, runtime, marks=[pytest.mark.requires_hardware, pytest.mark.platforms([arch])])
        for arch in ("a2a3", "a5")
        for runtime in ("host_build_graph", "tensormap_and_ringbuffer")
    ],
)
def test_worker_uses_each_invocations_swimlane_level(arch, runtime, borrowed, tmp_path, request):
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), arch, runtime, device, str(int(borrowed)), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _run(sys.argv[1], sys.argv[2], int(sys.argv[3]), bool(int(sys.argv[4])), Path(sys.argv[5]))
