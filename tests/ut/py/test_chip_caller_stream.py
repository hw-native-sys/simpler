# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The direct chip lane enqueues caller dependencies before returning a run."""

from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path

import pytest
from test_kernel_caller_stream import _callable
from test_kernel_mode_c_api import _load

_ROOT = Path(__file__).resolve().parents[3]


def _run(arch, runtime, device, depth):
    from _task_interface import _ChipWorker  # noqa: PLC0415
    from simpler.task_interface import (  # noqa: PLC0415
        AddressSpace,
        CallConfig,
        ChipStorageTaskArgs,
        ChipTensor,
        DataType,
    )

    base = _ROOT / "build/lib" / arch / "onboard" / runtime
    worker = _ChipWorker()
    worker.configure_launch_depth(depth)
    worker.init(
        str(base / "libhost_runtime.so"),
        str(base / "libaicpu_kernel.so"),
        str(base / "aicore_kernel.o"),
        str(_ROOT / "build/lib" / arch / "dispatcher/libsimpler_aicpu_dispatcher.so"),
        device,
    )
    lib = _load(arch, "onboard", runtime)
    lib.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.aclrtDestroyStream.argtypes = [ctypes.c_void_p]
    lib.aclrtSynchronizeStreamWithTimeout.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.aclrtMallocHost.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.aclrtFreeHost.argtypes = [ctypes.c_void_p]
    lib.aclrtMemcpyAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    streams = [ctypes.c_void_p(), ctypes.c_void_p()]
    host_buffers = []
    device_buffers = []
    runs = []
    outputs = []
    size = 128 * 128
    nbytes = size * ctypes.sizeof(ctypes.c_float)
    try:
        worker.register_callable(0, _callable(arch, runtime))
        for stream in streams:
            assert lib.aclrtCreateStream(ctypes.byref(stream)) == 0
        for invocation in range(4):
            stream = streams[invocation % 2]
            args = ChipStorageTaskArgs()
            current_device = []
            current_host = []
            for _ in range(3):
                address = worker.malloc(nbytes)
                device_buffers.append(address)
                current_device.append(address)
                args.add_tensor(ChipTensor.make(address, (size,), DataType.FLOAT32, address_space=AddressSpace.DEVICE))
                host = ctypes.c_void_p()
                assert lib.aclrtMallocHost(ctypes.byref(host), nbytes) == 0
                assert host.value is not None
                host_buffers.append(host)
                current_host.append((ctypes.c_float * size).from_address(host.value))
            for index in range(2):
                current_host[index][:] = [float(invocation + index + 1)] * size
                assert (
                    lib.aclrtMemcpyAsync(
                        current_device[index], nbytes, ctypes.addressof(current_host[index]), nbytes, 1, stream
                    )
                    == 0
                )
            run = worker._submit_chip_run_direct(0, args, CallConfig(), caller_stream=stream.value)
            assert run.launched
            runs.append(run)
            assert (
                lib.aclrtMemcpyAsync(ctypes.addressof(current_host[2]), nbytes, current_device[2], nbytes, 2, stream)
                == 0
            )
            total = float(2 * invocation + 3)
            expected = (total + 1) * (total + 2)
            if runtime == "tensormap_and_ringbuffer":
                expected += total
            outputs.append((current_host[2], expected))
        # No RunHandle poll or wait can cause a deferred launch on this path.
        # Caller synchronization alone must see all four results.
        for stream in streams:
            assert lib.aclrtSynchronizeStreamWithTimeout(stream, 60000) == 0
        for values, expected in outputs:
            assert all(value == expected for value in values), (list(values[:8]), expected)
        for run in runs:
            assert run.wait(-1.0)
    finally:
        worker._close_chip_run_lane()
        for address in device_buffers:
            worker.free(address)
        for host in host_buffers:
            assert lib.aclrtFreeHost(host) == 0
        for stream in streams:
            if stream.value:
                assert lib.aclrtDestroyStream(stream) == 0
        worker.finalize()


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize(
    ("arch", "runtime"),
    [
        pytest.param(arch, runtime, marks=[pytest.mark.requires_hardware, pytest.mark.platforms([arch])])
        for arch in ("a2a3", "a5")
        for runtime in ("host_build_graph", "tensormap_and_ringbuffer")
    ],
)
def test_direct_chip_submit_orders_caller_work(arch, runtime, depth, request):
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), arch, runtime, device, str(depth)],
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _run(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
