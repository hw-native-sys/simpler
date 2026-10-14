# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Caller H2D → native borrowed execution → caller D2H, on the real device."""

from __future__ import annotations

import ctypes
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from test_kernel_mode_c_api import (
    CallConfig,
    NativeRunDescriptor,
    _binaries,
    _CallerContexts,
    _load,
    _native_run_storage,
)

_ROOT = Path(__file__).resolve().parents[3]


def _callable(arch, runtime):
    tree = "tests/st" if runtime == "host_build_graph" else "examples"
    path = _ROOT / tree / arch / runtime / "vector_example/test_vector_example.py"
    spec = importlib.util.spec_from_file_location(f"caller_vector_{arch}_{runtime}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    case = next(
        value
        for value in vars(module).values()
        if isinstance(value, type) and value.__module__ == spec.name and "CALLABLE" in vars(value)
    )
    return case.compile_chip_callable(arch)


def _run(arch, runtime, device):
    from simpler.task_interface import AddressSpace, ChipStorageTaskArgs, ChipTensor, DataType  # noqa: PLC0415

    chip = _callable(arch, runtime)
    image = ctypes.string_at(int(chip.buffer_ptr()), int(chip.buffer_size()))
    lib = _load(arch, "onboard", runtime)
    lib.rtSetDevice.argtypes = [ctypes.c_int]
    assert lib.rtSetDevice(device) == 0
    contexts = _CallerContexts(lib, device)
    for name in ("aclrtCreateStream",):
        getattr(lib, name).argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.aclrtDestroyStream.argtypes = [ctypes.c_void_p]
    lib.aclrtSynchronizeStreamWithTimeout.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
    lib.aclrtMallocHost.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.aclrtFree.argtypes = lib.aclrtFreeHost.argtypes = [ctypes.c_void_p]
    lib.aclrtMemcpyAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.simpler_launch_run_on_stream.argtypes = [ctypes.c_void_p] * 4
    lib.simpler_wait_run.argtypes = lib.simpler_poll_run.argtypes = [ctypes.c_void_p] * 2
    streams = [ctypes.c_void_p(), ctypes.c_void_p()]
    device_buffers = [ctypes.c_void_p() for _ in range(3)]
    host_buffers = [ctypes.c_void_p() for _ in range(3)]
    size = 128 * 128
    nbytes = size * ctypes.sizeof(ctypes.c_float)
    ctx = lib.create_device_context()
    run = None
    pending = False
    try:
        for stream in streams:
            assert lib.aclrtCreateStream(ctypes.byref(stream)) == 0
        for ptr in device_buffers:
            assert lib.aclrtMalloc(ctypes.byref(ptr), nbytes, 0) == 0
        for ptr in host_buffers:
            assert lib.aclrtMallocHost(ctypes.byref(ptr), nbytes) == 0
        aicpu, aicore, dispatcher = _binaries(arch, runtime)
        config = CallConfig()
        config.aicpu_thread_num = 2
        assert (
            lib.simpler_kernel_mode_init(
                ctx,
                device,
                aicpu,
                len(aicpu),
                aicore,
                len(aicore),
                dispatcher,
                len(dispatcher),
                ctypes.byref(config),
                1,
            )
            == 0
        )
        assert lib.simpler_kernel_mode_prepare_callable(ctx, 0, image, len(image)) == 0
        storage, run = _native_run_storage(lib)
        args = ChipStorageTaskArgs()
        for ptr in device_buffers:
            args.add_tensor(ChipTensor.make(ptr.value, (size,), DataType.FLOAT32, address_space=AddressSpace.DEVICE))
        views = []
        for ptr in host_buffers:
            assert ptr.value is not None
            views.append((ctypes.c_float * size).from_address(ptr.value))
        for epoch in range(1, 5):
            stream = streams[epoch % 2]
            descriptor = NativeRunDescriptor(generation=1, run_epoch=epoch)
            assert (
                lib.simpler_prepare_run(ctx, run, 0, args.__ptr__(), ctypes.byref(config), ctypes.byref(descriptor))
                == 0
            )
            pending = True
            # No synchronization separates caller writes from launch. These
            # buffers stay alive through the caller's final read below.
            for index in range(2):
                views[index][:] = [float(epoch + index)] * size
                assert lib.aclrtMemcpyAsync(device_buffers[index], nbytes, host_buffers[index], nbytes, 1, stream) == 0
            with contexts.other():
                assert lib.simpler_launch_run_on_stream(ctx, run, stream, None) == -1003
            assert lib.simpler_launch_run_on_stream(ctx, run, stream, None) == 0
            assert lib.aclrtMemcpyAsync(host_buffers[2], nbytes, device_buffers[2], nbytes, 2, stream) == 0
            # This is the user's stream, not either runtime stream. Numeric
            # output must be ready before simpler_wait_run/finalize is called.
            assert lib.aclrtSynchronizeStreamWithTimeout(stream, 60000) == 0
            total = float(2 * epoch + 1)
            expected = (total + 1) * (total + 2)
            if runtime == "tensormap_and_ringbuffer":
                expected += total
            assert all(value == expected for value in views[2]), (epoch, list(views[2][:8]), expected)
            assert lib.simpler_wait_run(ctx, run) == 0
            assert lib.simpler_finalize_run(ctx, run) == 0
            pending = False
        assert storage is not None  # owns the aligned native run storage
    finally:
        if pending:
            lib.simpler_finalize_run(ctx, run)
        assert lib.finalize_device(ctx) == 0
        lib.destroy_device_context(ctx)
        for pointers, free in (
            (streams, lib.aclrtDestroyStream),
            (device_buffers, lib.aclrtFree),
            (host_buffers, lib.aclrtFreeHost),
        ):
            for ptr in pointers:
                if ptr.value:
                    assert free(ptr) == 0
        contexts.close()


@pytest.mark.parametrize(
    ("arch", "runtime"),
    [
        pytest.param(arch, runtime, marks=[pytest.mark.requires_hardware, pytest.mark.platforms([arch])])
        for arch in ("a2a3", "a5")
        for runtime in ("host_build_graph", "tensormap_and_ringbuffer")
    ],
)
def test_borrowed_launch_orders_caller_work(arch, runtime, request):
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), arch, runtime, device],
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _run(sys.argv[1], sys.argv[2], int(sys.argv[3]))
