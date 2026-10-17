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


def _run(platform, runtime, device, borrowed, host_controls, workspace_budget=None):  # noqa: PLR0912, PLR0915 -- one subprocess owns all resources through retirement
    from simpler.buffer import Buffer  # noqa: PLC0415
    from simpler.task_interface import (  # noqa: PLC0415
        CallConfig,
        DataType,
        TaskArgs,
        Tensor,
        TensorArgType,
        TensorTransfer,
    )
    from simpler.worker import Worker  # noqa: PLC0415

    from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

    target = _compile(platform, runtime, host_controls)
    onboard = not platform.endswith("sim")
    owner = Worker(
        level=2,
        device_id=device,
        platform=platform,
        runtime=runtime,
        **({"workspace_budget_bytes": workspace_budget} if workspace_budget else {}),
    )
    lib = None
    stream, allocation, host = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    state, control, runs = None, None, []
    caller_context = ctypes.c_void_p()
    last_expected = 7
    refused = False
    nbytes = 16 * ctypes.sizeof(ctypes.c_int32)
    try:
        if onboard:
            lib = ctypes.CDLL(str(RuntimeBuilder(platform).get_binaries(runtime).host_path))
            lib.rtSetDevice.argtypes = [ctypes.c_int]
            lib.rtCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
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
                assert lib.rtCtxGetCurrent(ctypes.byref(caller_context)) == 0
        handle = owner.register(target)
        owner.init(borrowed_context_generation=77 if borrowed else None)
        if workspace_budget is not None:
            status, report = owner._chip_worker.workspace_report()
            assert status == "available"
            assert report["limit_bytes"] == workspace_budget
            assert report["budget_enforced"] == bool(workspace_budget)
            assert report["coverage_is_partial"] == 1
        config = CallConfig()
        if workspace_budget is not None:
            config.runtime_env.ring_task_window = 64
            config.runtime_env.ring_heap = 1 << 20
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
        steps = [(0, 99, 7), (1, 11, 11), (4, 17, 17), (0, 123, 17), (8, 23, 23), (2, 29, 29), (0, 456, 29)]
        if workspace_budget == 1024:
            steps = [(1, 11, 11)]
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
            if workspace_budget == 1024:
                with pytest.raises(RuntimeError):
                    owner.submit(handle, args, config=config, **options)
                refused = True
                status, report = owner._chip_worker.workspace_report()
                assert status == "available" and report["reserved_bytes"] <= workspace_budget
                assert lib.aclrtMemcpyAsync(host, nbytes, allocation, nbytes, 2, stream) == 0
                assert lib.aclrtSynchronizeStreamWithTimeout(stream, 60000) == 0
                assert list(values) == [7] * 16
                break
            run = owner.submit(handle, args, config=config, **options)
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
            last_expected = expected
            if workspace_budget is not None:
                status, report = owner._chip_worker.workspace_report()
                assert status == "available" and report["blocks_published"] > 0
                assert report["reserved_bytes"] > 0 and report["live_blocked"] == 0
                if workspace_budget:
                    assert report["reserved_bytes"] <= workspace_budget
    finally:
        # Failed retirement supplies no permission to free the caller's allocation.
        for run in runs:
            run.wait()
        if control is not None:
            owner.release_buffer(control)
        if lib is not None:
            if borrowed:
                owner.close()
                after_close = ctypes.c_void_p()
                assert lib.rtCtxGetCurrent(ctypes.byref(after_close)) == 0
                assert after_close.value == caller_context.value
                if allocation.value and host.value:
                    assert lib.aclrtMemcpyAsync(host, nbytes, allocation, nbytes, 2, stream) == 0
                    assert lib.aclrtSynchronizeStreamWithTimeout(stream, 60000) == 0
                    assert list(values) == [last_expected] + [7] * 15
            if state is not None:
                if refused:
                    # Native exceptions leave source uses unproven in the public
                    # API; this subprocess retains their device backing until exit.
                    with pytest.raises(RuntimeError, match="unproven device uses"):
                        state.close()
                else:
                    state.close()
            if allocation.value and not refused:
                assert lib.aclrtFree(allocation) == 0
            if host.value:
                assert lib.aclrtFreeHost(host) == 0
            if stream.value:
                assert lib.aclrtDestroyStream(stream) == 0
        elif state is not None:
            owner.free(state)
        owner.close()


def _check(st_platform, st_device_ids, runtime, borrowed, host_controls, workspace_budget=None):
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
            str(workspace_budget) if workspace_budget is not None else "none",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if workspace_budget == 1024:
        assert "the configured workspace budget has no room for it" in result.stdout + result.stderr


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


@pytest.mark.platforms(["a2a3", "a5"])
@pytest.mark.device_count(1)
@pytest.mark.parametrize(
    "runtime",
    [
        pytest.param("host_build_graph", marks=pytest.mark.runtime("host_build_graph")),
        pytest.param("tensormap_and_ringbuffer", marks=pytest.mark.runtime("tensormap_and_ringbuffer")),
    ],
)
@pytest.mark.parametrize("budget", [0, 1024, 128 << 20], ids=["unlimited", "refused", "bounded"])
def test_borrowed_workspace_budget_preserves_caller_resources(st_platform, st_device_ids, runtime, budget):
    _check(st_platform, st_device_ids, runtime, True, False, budget)


if __name__ == "__main__":
    _run(
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        bool(int(sys.argv[4])),
        bool(int(sys.argv[5])),
        None if sys.argv[6] == "none" else int(sys.argv[6]),
    )
