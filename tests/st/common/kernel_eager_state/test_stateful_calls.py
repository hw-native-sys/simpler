# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A scalar or HOST control selects empty/active calls over one persistent DEVICE view."""

import ctypes
import struct
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[4]


def _compile(platform, runtime, host_controls):
    from simpler.task_interface import ArgDirection as D  # noqa: PLC0415

    from simpler_setup.scene_test import compile_chip_callable_spec, l3_compile_cache_key  # noqa: PLC0415

    arch = platform.removesuffix("sim")
    function = "state_from_host" if host_controls else "state_from_scalars"
    spec = {
        "orchestration": {
            "source": str(Path(__file__).with_name("state_orchestration.cpp")),
            "function_name": function,
            "signature": [D.INOUT, D.IN] if host_controls else [D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": str(
                    _ROOT
                    / "tests/st"
                    / arch
                    / "tensormap_and_ringbuffer/predicated_dispatch/kernels/aic/kernel_write_gate.cpp"
                ),
                "core_type": "aic",
                "signature": [D.INOUT],
            }
        ],
    }
    key = l3_compile_cache_key(__name__, "StatefulEager", function, platform, runtime)
    return compile_chip_callable_spec(spec, platform, runtime, key)


def _run(platform, runtime, device, borrowed, host_controls):  # noqa: PLR0912 -- one subprocess owns all resources through retirement
    from simpler.buffer import Buffer  # noqa: PLC0415
    from simpler.task_interface import DataType, TaskArgs, Tensor, TensorArgType, TensorTransfer  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

    target = _compile(platform, runtime, host_controls)
    onboard = not platform.endswith("sim")
    owner = Worker(level=2, device_id=device, platform=platform, runtime=runtime)
    lib = None
    stream, allocation, host = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    state, control, runs = None, None, []
    nbytes = 16 * ctypes.sizeof(ctypes.c_int32)
    try:
        if onboard:
            lib = ctypes.CDLL(str(RuntimeBuilder(platform).get_binaries(runtime).host_path))
            lib.rtSetDevice.argtypes = [ctypes.c_int]
            lib.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            lib.aclrtDestroyStream.argtypes = [ctypes.c_void_p]
            lib.aclrtSynchronizeStreamWithTimeout.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
            lib.aclrtFree.argtypes = [ctypes.c_void_p]
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
            if borrowed:
                assert lib.rtSetDevice(device) == 0
        handle = owner.register(target)
        owner.init(borrowed_context_generation=77 if borrowed else None)
        if lib is not None:
            assert lib.aclrtCreateStream(ctypes.byref(stream)) == 0
            assert lib.aclrtMalloc(ctypes.byref(allocation), nbytes, 0) == 0
            assert allocation.value is not None
            assert lib.aclrtMallocHost(ctypes.byref(host), nbytes) == 0
            assert host.value is not None
            state = Buffer.wrap(address=allocation.value, nbytes=nbytes, location=owner.device_location)
            values = (ctypes.c_int32 * 16).from_address(host.value)
            values[:] = [7] * 16
            assert lib.aclrtMemcpyAsync(allocation, nbytes, host, nbytes, 1, stream) == 0
        else:
            state = owner.malloc(nbytes)
            owner.copy_to(state, bytearray(struct.pack("16i", *([7] * 16))))
        view = Tensor(state, shapes=(16,), dtype=DataType.INT32)
        identity = state.identity
        if host_controls:
            control = owner.create_buffer(8)
            controls = (ctypes.c_int32 * 2).from_address(control.base)
        steps = [(0, 99, 7), (1, 11, 11), (0, 123, 11), (1, 23, 23), (0, 456, 23)]
        if borrowed and runtime == "host_build_graph":
            # Borrowed arenas cannot grow after commit; this reserves the one-task image.
            steps.insert(0, (1, 7, 7))
        for active, value, expected in steps:
            args = TaskArgs()
            args.add_tensor(view, TensorArgType.INOUT)
            if control is not None:
                controls[:] = [active, value]
                args.add_tensor(
                    Tensor(control, shapes=(2,), dtype=DataType.INT32),
                    TensorArgType.INPUT,
                    transfer=TensorTransfer.NONE,
                )
            else:
                args.add_scalar(active)
                args.add_scalar(value)
            options = {"caller_stream": stream.value} if onboard else {}
            run = owner.submit(handle, args, **options)
            runs.append(run)
            if control is not None:
                # All host decisions for this invocation precede the successful return.
                controls[:] = [1, 777]
            if lib is not None:
                assert lib.aclrtMemcpyAsync(host, nbytes, allocation, nbytes, 2, stream) == 0
                assert lib.aclrtSynchronizeStreamWithTimeout(stream, 60000) == 0
                actual = list(values)
                run.wait()
            else:
                run.wait()
                output = bytearray(nbytes)
                owner.copy_from(output, state)
                actual = list(struct.unpack("16i", output))
            assert actual == [expected] + [7] * 15, (active, value, actual)
            assert state.identity == identity and run.done
    finally:
        # Failed retirement supplies no permission to free the caller's allocation.
        for run in runs:
            run.wait()
        if control is not None:
            owner.release_buffer(control)
        if lib is not None:
            if state is not None:
                state.close()
            if allocation.value:
                assert lib.aclrtFree(allocation) == 0
            if host.value:
                assert lib.aclrtFreeHost(host) == 0
            if borrowed:
                owner.close()
                assert lib.aclrtSynchronizeStreamWithTimeout(stream, 60000) == 0
            if stream.value:
                assert lib.aclrtDestroyStream(stream) == 0
        elif state is not None:
            owner.free(state)
        owner.close()


def _check(st_platform, st_device_ids, runtime, borrowed, host_controls):
    if borrowed and st_platform.endswith("sim"):
        pytest.skip("borrowed contexts require onboard RTS")
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            st_platform,
            runtime,
            str(st_device_ids[0]),
            str(int(borrowed)),
            str(int(host_controls)),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


_OWNERSHIP = (False, True)


@pytest.mark.platforms(["a2a3sim", "a2a3", "a5sim", "a5"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(1)
@pytest.mark.parametrize("borrowed", _OWNERSHIP)
def test_tmr_empty_and_active_calls_retain_state(st_platform, st_device_ids, borrowed):
    _check(st_platform, st_device_ids, "tensormap_and_ringbuffer", borrowed, False)


@pytest.mark.platforms(["a2a3sim", "a2a3", "a5sim", "a5"])
@pytest.mark.runtime("host_build_graph")
@pytest.mark.device_count(1)
@pytest.mark.parametrize("borrowed", _OWNERSHIP)
@pytest.mark.parametrize("host_controls", [False, True])
def test_hbg_empty_and_active_calls_retain_state(st_platform, st_device_ids, borrowed, host_controls):
    _check(st_platform, st_device_ids, "host_build_graph", borrowed, host_controls)


if __name__ == "__main__":
    _run(sys.argv[1], sys.argv[2], int(sys.argv[3]), bool(int(sys.argv[4])), bool(int(sys.argv[5])))
