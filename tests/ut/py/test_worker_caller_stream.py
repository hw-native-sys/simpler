# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Dependent public Worker calls cross caller streams through events and D2D."""

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_kernel_caller_stream import _callable
from test_kernel_mode_c_api import _CallerContexts, _load


def _run(arch, runtime, device, borrowed):  # noqa: PLR0912 -- one subprocess owns caller resources through retirement
    from simpler.buffer import Buffer  # noqa: PLC0415
    from simpler.task_interface import (  # noqa: PLC0415
        DataType,
        ExecutionContextRefused,
        TaskArgs,
        Tensor,
        TensorArgType,
    )
    from simpler.worker import Worker  # noqa: PLC0415

    lib = _load(arch, "onboard", runtime)
    lib.rtSetDevice.argtypes = [ctypes.c_int]
    contexts = None
    if borrowed:
        assert lib.rtSetDevice(device) == 0
        contexts = _CallerContexts(lib, device)
    owner = Worker(level=2, device_id=device, platform=arch, runtime=runtime)
    handle = owner.register(_callable(arch, runtime))
    owner.init(borrowed_context_generation=77 if borrowed else None)
    lib.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
    lib.aclrtFree.argtypes = [ctypes.c_void_p]
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
    lib.aclrtCreateEventExWithFlag.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32]
    lib.aclrtDestroyEvent.argtypes = [ctypes.c_void_p]
    lib.aclrtRecordEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.aclrtStreamWaitEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    streams = [ctypes.c_void_p(), ctypes.c_void_p()]
    handoffs = [ctypes.c_void_p(), ctypes.c_void_p()]
    device_buffers, host_buffers, sources, runs, outputs = [], [], [], [], []
    size = 128 * 128
    nbytes = size * ctypes.sizeof(ctypes.c_float)
    try:
        for stream in streams:
            assert lib.aclrtCreateStream(ctypes.byref(stream)) == 0
        for event in handoffs:
            assert lib.aclrtCreateEventExWithFlag(ctypes.byref(event), 1) == 0  # ACL_EVENT_SYNC
        previous_output, previous_expected = None, 0.0
        # Three calls keep every integer golden result exactly representable in float32.
        for invocation in range(3):
            stream = streams[invocation % 2]
            if invocation:
                assert lib.aclrtStreamWaitEvent(stream, handoffs[invocation - 1]) == 0
            args = TaskArgs()
            device_views, host_views = [], []
            for index in range(3):
                allocation, host = ctypes.c_void_p(), ctypes.c_void_p()
                assert lib.aclrtMalloc(ctypes.byref(allocation), nbytes, 0) == 0
                assert allocation.value is not None
                device_buffers.append(allocation)
                device_views.append(allocation)
                assert lib.aclrtMallocHost(ctypes.byref(host), nbytes) == 0
                assert host.value is not None
                host_buffers.append(host)
                host_views.append((ctypes.c_float * size).from_address(host.value))
                backing = Buffer.wrap(address=allocation.value, nbytes=nbytes, location=owner.device_location)
                sources.append(backing)
                args.add_tensor(
                    Tensor(backing, shapes=(size,), dtype=DataType.FLOAT32),
                    TensorArgType.INPUT if index < 2 else TensorArgType.OUTPUT_EXISTING,
                )
            for index in range(2):
                value = -31.0 if invocation and index == 0 else float(invocation + index + 1)
                host_views[index][:] = [value] * size
                assert (
                    lib.aclrtMemcpyAsync(device_views[index], nbytes, host_buffers[-3 + index], nbytes, 1, stream) == 0
                )
            if invocation:
                assert lib.aclrtMemcpyAsync(device_views[0], nbytes, previous_output, nbytes, 3, stream) == 0
            run = owner.submit(handle, args, caller_stream=stream.value)
            runs.append(run)
            with pytest.raises(RuntimeError, match="in-flight or unproven"):
                sources[-1].close()
            assert lib.aclrtMemcpyAsync(host_buffers[-1], nbytes, device_views[2], nbytes, 2, stream) == 0
            if invocation < len(handoffs):
                assert lib.aclrtRecordEvent(handoffs[invocation], stream) == 0
            total = previous_expected + invocation + 2 if invocation else 3.0
            expected = (total + 1) * (total + 2) + (total if runtime == "tensormap_and_ringbuffer" else 0)
            outputs.append((host_views[2], expected))
            previous_output, previous_expected = device_views[2], expected
        if contexts is not None:
            with contexts.other():
                for operation in (
                    runs[-1].wait,
                    runs[-1].result,
                    lambda: owner.submit(handle, caller_stream=streams[0].value),
                    owner.close,
                ):
                    with pytest.raises(ExecutionContextRefused):
                        operation()
        # No successful RunHandle query precedes the caller's read.
        for stream in streams:
            assert lib.aclrtSynchronizeStreamWithTimeout(stream, 60000) == 0
        for values, expected in outputs:
            assert all(value == expected for value in values), (list(values[:8]), expected)
        for run in runs:
            run.wait()
            assert run.done
    finally:
        # A failed drain stops cleanup before caller allocations can be released.
        for run in runs:
            run.wait()
        # Caller copies and event waits outlive the kernel completion fence.
        for stream in streams:
            if stream.value:
                assert lib.aclrtSynchronizeStreamWithTimeout(stream, 60000) == 0
        for event in handoffs:
            if event.value:
                assert lib.aclrtDestroyEvent(event) == 0
        for backing in sources:
            backing.close()
        for pointers, free in ((device_buffers, lib.aclrtFree), (host_buffers, lib.aclrtFreeHost)):
            for ptr in pointers:
                assert free(ptr) == 0
        if borrowed:
            assert contexts is not None
            owner.close()
            for stream in streams:
                assert lib.aclrtSynchronizeStreamWithTimeout(stream, 60000) == 0
            current = ctypes.c_void_p()
            assert lib.aclrtGetCurrentContext(ctypes.byref(current)) == 0
            assert current.value == contexts.owner.value
        for stream in streams:
            if stream.value:
                assert lib.aclrtDestroyStream(stream) == 0
        if not borrowed:
            owner.close()
        if contexts is not None:
            contexts.close()


@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("task_queue", ["0", "1"])
@pytest.mark.parametrize(
    ("arch", "runtime"),
    [
        pytest.param(arch, runtime, marks=[pytest.mark.requires_hardware, pytest.mark.platforms([arch])])
        for arch in ("a2a3", "a5")
        for runtime in ("host_build_graph", "tensormap_and_ringbuffer")
    ],
)
def test_public_worker_orders_caller_work(arch, runtime, borrowed, task_queue, request):
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    environment = {**os.environ, "TASK_QUEUE_ENABLE": task_queue}
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), arch, runtime, device, str(int(borrowed))],
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
        env=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _run(sys.argv[1], sys.argv[2], int(sys.argv[3]), bool(int(sys.argv[4])))
