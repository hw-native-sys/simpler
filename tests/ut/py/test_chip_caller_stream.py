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
from test_kernel_mode_c_api import _CallerContexts, _load

_ROOT = Path(__file__).resolve().parents[3]


def _allocate_device(worker, lib, nbytes, borrowed):
    if not borrowed:
        return worker.malloc(nbytes)
    allocation = ctypes.c_void_p()
    assert lib.aclrtMalloc(ctypes.byref(allocation), nbytes, 0) == 0
    assert allocation.value is not None
    return allocation.value


def _cleanup(worker, lib, contexts, device_buffers, host_buffers, streams):
    worker._close_chip_run_lane()
    for address in device_buffers:
        if contexts is not None:
            assert lib.aclrtFree(address) == 0
        else:
            worker.free(address)
    for host in host_buffers:
        assert lib.aclrtFreeHost(host) == 0
    if contexts is not None:
        worker.finalize()
        # Successful borrowed close preserves caller streams and context.
        for stream in streams:
            assert lib.aclrtSynchronizeStreamWithTimeout(stream, 60000) == 0
        current = ctypes.c_void_p()
        assert lib.aclrtGetCurrentContext(ctypes.byref(current)) == 0
        assert current.value == contexts.owner.value
    for stream in streams:
        if stream.value:
            assert lib.aclrtDestroyStream(stream) == 0
    if contexts is not None:
        contexts.close()
    else:
        worker.finalize()


def _prepare_callable(worker, contexts, callable_image):
    from simpler.task_interface import ExecutionContextRefused  # noqa: PLC0415

    if contexts is None:
        assert worker.register_callable(callable_image) == 0
        return
    with contexts.other():
        for operation in (
            lambda: worker.register_callable(0, callable_image),
            lambda: worker.register_callable(callable_image),
        ):
            with pytest.raises(ExecutionContextRefused, match="context validation failed"):
                operation()
    assert worker.register_callable(callable_image) == 0


def _run(arch, runtime, device, depth, borrowed):
    from _task_interface import _ChipWorker  # noqa: PLC0415
    from simpler.task_interface import (  # noqa: PLC0415
        AddressSpace,
        CallConfig,
        ChipStorageTaskArgs,
        ChipTensor,
        DataType,
        ExecutionContextRefused,
    )
    from simpler.worker import RunHandle, Worker, _Lifecycle  # noqa: PLC0415

    base = _ROOT / "build/lib" / arch / "onboard" / runtime
    lib = _load(arch, "onboard", runtime)
    contexts = None
    if borrowed:
        lib.rtSetDevice.argtypes = [ctypes.c_int]
        assert lib.rtSetDevice(device) == 0
        contexts = _CallerContexts(lib, device)
    worker = _ChipWorker()
    worker.configure_launch_depth(depth)
    worker.init(
        str(base / "libhost_runtime.so"),
        str(base / "libaicpu_kernel.so"),
        str(base / "aicore_kernel.o"),
        str(_ROOT / "build/lib" / arch / "dispatcher/libsimpler_aicpu_dispatcher.so"),
        device,
        borrowed_context_generation=77 if borrowed else None,
    )
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
    streams = [ctypes.c_void_p(), ctypes.c_void_p()]
    host_buffers = []
    device_buffers = []
    runs = []
    python_runs = []
    handle_owner = Worker(level=2)
    handle_owner._chip_worker = worker
    outputs = []
    size = 128 * 128
    nbytes = size * ctypes.sizeof(ctypes.c_float)
    try:
        callable_image = _callable(arch, runtime)
        _prepare_callable(worker, contexts, callable_image)
        public_callable = handle_owner.register(callable_image)
        handle_owner._lifecycle = _Lifecycle.READY
        for stream in streams:
            assert lib.aclrtCreateStream(ctypes.byref(stream)) == 0
        for invocation in range(4):
            stream = streams[invocation % 2]
            args = ChipStorageTaskArgs()
            current_device = []
            current_host = []
            for _ in range(3):
                address = _allocate_device(worker, lib, nbytes, borrowed)
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
            # This test probes native admission directly. These handles bridge
            # real borrowed native runs to its shared Python completion path.
            run_id = invocation + 1
            handle_owner._chip_runs[run_id] = run
            python_runs.append(RunHandle(handle_owner, run_id, (args,)))
            assert (
                lib.aclrtMemcpyAsync(ctypes.addressof(current_host[2]), nbytes, current_device[2], nbytes, 2, stream)
                == 0
            )
            total = float(2 * invocation + 3)
            expected = (total + 1) * (total + 2)
            if runtime == "tensormap_and_ringbuffer":
                expected += total
            outputs.append((current_host[2], expected))
        if contexts is not None:
            with contexts.other():
                handle = python_runs[-1]
                keepalive = handle._keepalive
                for operation in (lambda: handle.done, handle.wait, handle.result):
                    with pytest.raises(ExecutionContextRefused, match="context validation failed"):
                        operation()
                    assert not handle._terminal and not handle._wait_in_progress
                    assert handle._run_id == len(runs) and handle._keepalive is keepalive
                    assert len(handle_owner._chip_runs) == len(runs)
                    assert not handle_owner._source_context_failed
                for operation in (
                    lambda: handle_owner.submit(public_callable),
                    lambda: handle_owner.register(callable_image),
                    lambda: handle_owner.unregister(public_callable),
                    handle_owner.close,
                    lambda: worker.unregister_callable(0),
                    lambda: worker.malloc(16),
                    lambda: worker.free(device_buffers[0]),
                    lambda: worker.copy_to(device_buffers[0], host_buffers[0].value, nbytes),
                    lambda: worker.copy_from(host_buffers[0].value, device_buffers[0], nbytes),
                    worker.device_memory_info,
                    lambda: worker.flush_diagnostics(0),
                    worker.finalize,
                ):
                    with pytest.raises(ExecutionContextRefused, match="context validation failed"):
                        operation()
                    assert handle_owner._lifecycle is _Lifecycle.READY
                    assert handle_owner._close_completion is None
                    assert handle_owner._resolve_handle(public_callable).ref_count == 1
                    assert not handle_owner._source_context_failed and handle_owner._chip_run_seq == 0
                assert worker.initialized
                assert runs[-1].launched
        # No RunHandle poll or wait can cause a deferred launch on this path.
        # Caller synchronization alone must see all four results.
        for stream in streams:
            assert lib.aclrtSynchronizeStreamWithTimeout(stream, 60000) == 0
        for values, expected in outputs:
            assert all(value == expected for value in values), (list(values[:8]), expected)
        for handle in python_runs:
            handle.wait()
            assert handle.done
        assert not handle_owner._chip_runs
    finally:
        # Native teardown owns this bridge's resources; the public borrowed
        # initialization and admission path is not under test here.
        handle_owner._chip_worker = None
        handle_owner._lifecycle = _Lifecycle.CLOSED
        _cleanup(worker, lib, contexts, device_buffers, host_buffers, streams)


@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize(
    ("arch", "runtime"),
    [
        pytest.param(arch, runtime, marks=[pytest.mark.requires_hardware, pytest.mark.platforms([arch])])
        for arch in ("a2a3", "a5")
        for runtime in ("host_build_graph", "tensormap_and_ringbuffer")
    ],
)
def test_direct_chip_submit_orders_caller_work(arch, runtime, depth, borrowed, request):
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), arch, runtime, device, str(depth), str(int(borrowed))],
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _run(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), bool(int(sys.argv[5])))
