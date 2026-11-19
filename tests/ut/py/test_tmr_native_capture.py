# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""TMR replay through native launch and common prepared-resource retention."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_kernel_caller_stream import _callable
from test_kernel_mode_c_api import (
    CallConfig,
    NativeRunDescriptor,
    _binaries,
    _CallerContexts,
    _load,
    _native_run_storage,
    _reset_caller_device,
)

_ROOT = Path(__file__).resolve().parents[3]
_COUNT = 128 * 128
_BYTES = _COUNT * ctypes.sizeof(ctypes.c_float)
_POISON = -123.0
_PREPARE_CAPTURE = 1
pytestmark = pytest.mark.runtime("tensormap_and_ringbuffer")


def _fatal_callable(arch):
    from simpler.task_interface import ArgDirection as D  # noqa: PLC0415

    from simpler_setup.scene_test import compile_chip_callable_spec, l3_compile_cache_key  # noqa: PLC0415

    function = "aicpu_orchestration_entry"
    spec = {
        "orchestration": {
            "source": str(_ROOT / "tests/st/automatic_reported_fatal.cpp"),
            "function_name": function,
            "signature": [D.IN, D.IN, D.OUT],
        },
        "incores": [],
    }
    key = l3_compile_cache_key(__name__, "TmrNativeCaptureFatal", function, arch, "tensormap_and_ringbuffer")
    return compile_chip_callable_spec(spec, arch, "tensormap_and_ringbuffer", key)


def _bind(lib):
    lib.simpler_set_run_resource_grant_ctx.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.simpler_retain_run_resources.argtypes = [ctypes.c_void_p] * 3 + [ctypes.POINTER(ctypes.c_void_p)]
    lib.simpler_release_run_resources.argtypes = [ctypes.c_void_p] * 2
    lib.simpler_launch_run_on_stream.argtypes = [ctypes.c_void_p] * 4
    lib.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.aclrtDestroyStream.argtypes = [ctypes.c_void_p]
    lib.aclrtSynchronizeStreamWithTimeout.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.aclmdlRICaptureBegin.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.aclmdlRICaptureEnd.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.aclmdlRIExecuteAsync.argtypes = [ctypes.c_void_p] * 2
    lib.aclmdlRIDestroy.argtypes = [ctypes.c_void_p]
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
    lib.aclrtMemcpy.argtypes = lib.aclrtMemcpyAsync.argtypes[:-1]


class _Recording:
    def __init__(self, lib, ctx, caller, config, epoch, slot, *, capture=True):
        from simpler.task_interface import AddressSpace, ChipStorageTaskArgs, ChipTensor, DataType  # noqa: PLC0415

        self.lib, self.ctx, self.caller = lib, ctx, caller
        self.device = [ctypes.c_void_p() for _ in range(3)]
        self.host = [ctypes.c_void_p() for _ in range(3)]
        self.views = []
        self.reference = ctypes.c_void_p()
        for device, host in zip(self.device, self.host):
            assert lib.aclrtMalloc(ctypes.byref(device), _BYTES, 0) == 0
            assert lib.aclrtMallocHost(ctypes.byref(host), _BYTES) == 0
            assert host.value is not None
            self.views.append((ctypes.c_float * _COUNT).from_address(host.value))
        self.views[2][:] = [_POISON] * _COUNT
        assert lib.aclrtMemcpy(self.device[2], _BYTES, self.host[2], _BYTES, 1) == 0
        args = ChipStorageTaskArgs()
        for ptr in self.device:
            args.add_tensor(ChipTensor.make(ptr.value, (_COUNT,), DataType.FLOAT32, address_space=AddressSpace.DEVICE))
        self.storage, self.run = _native_run_storage(lib)
        descriptor = NativeRunDescriptor(
            pipeline_slot=slot,
            generation=1,
            dispatch_id=epoch,
            run_epoch=epoch,
            prepare_purpose=_PREPARE_CAPTURE if capture else 0,
        )
        assert (
            lib.simpler_prepare_run(ctx, self.run, 0, args.__ptr__(), ctypes.byref(config), ctypes.byref(descriptor))
            == 0
        )
        if capture:
            retain_rc = lib.simpler_retain_run_resources(ctx, self.run, caller, ctypes.byref(self.reference))
            assert retain_rc == 0, f"simpler_retain_run_resources failed: {retain_rc}"
            assert self.reference.value is not None
        # The launch consumes its prepared snapshot, not this mutable source object.
        args.clear()
        args.add_scalar(111)

    def record(self):
        for index in range(2):
            assert self.lib.aclrtMemcpyAsync(self.device[index], _BYTES, self.host[index], _BYTES, 1, self.caller) == 0
        assert self.lib.simpler_launch_run_on_stream(self.ctx, self.run, self.caller, None) == 0
        assert self.lib.aclrtMemcpyAsync(self.host[2], _BYTES, self.device[2], _BYTES, 2, self.caller) == 0

    def retire_original_run(self):
        assert self.lib.simpler_finalize_run(self.ctx, self.run) == 0
        ctypes.memset(ctypes.addressof(self.storage), 0xA5, ctypes.sizeof(self.storage))

    def check_not_executed(self):
        assert self.lib.aclrtMemcpy(self.host[2], _BYTES, self.device[2], _BYTES, 2) == 0
        assert all(value == _POISON for value in self.views[2]), "prepare/capture executed user work"

    def set_inputs(self, value):
        for index in range(2):
            self.views[index][:] = [float(value + index)] * _COUNT
        self.views[2][:] = [_POISON] * _COUNT
        assert self.lib.aclrtMemcpyAsync(self.device[2], _BYTES, self.host[2], _BYTES, 1, self.caller) == 0

    def check_output(self, value):
        total = float(2 * value + 1)
        expected = (total + 1) * (total + 2) + total
        assert all(result == expected for result in self.views[2]), (value, list(self.views[2][:8]), expected)

    def close(self):
        if self.reference.value:
            assert self.lib.simpler_release_run_resources(self.ctx, self.reference) == 0
        for ptr in self.device:
            assert self.lib.aclrtFree(ptr) == 0
        for ptr in self.host:
            assert self.lib.aclrtFreeHost(ptr) == 0


class _Model:
    def __init__(self, lib, ctx, caller, config, epochs):
        self.lib, self.caller = lib, caller
        self.nodes = [_Recording(lib, ctx, caller, config, epoch, slot) for slot, epoch in enumerate(epochs)]
        self.model = ctypes.c_void_p()
        assert lib.aclmdlRICaptureBegin(caller, 0) == 0
        for node in self.nodes:
            node.record()
            # Retained resources outlive the recording handle; no device wait
            # or private-result observation is needed to record the next node.
            node.retire_original_run()
        assert lib.aclmdlRICaptureEnd(caller, ctypes.byref(self.model)) == 0
        assert self.model.value is not None
        for node in self.nodes:
            node.check_not_executed()

    def replay(self, value):
        for index, node in enumerate(self.nodes):
            node.set_inputs(value + index)
        assert self.lib.aclmdlRIExecuteAsync(self.model, self.caller) == 0
        assert self.lib.aclrtSynchronizeStreamWithTimeout(self.caller, 60000) == 0
        for index, node in enumerate(self.nodes):
            node.check_output(value + index)

    def close(self):
        assert self.lib.aclmdlRIDestroy(self.model) == 0
        for node in self.nodes:
            node.close()


def _run(arch, device, scenario):
    chip = _fatal_callable(arch) if scenario == "fatal" else _callable(arch, "tensormap_and_ringbuffer")
    image = ctypes.string_at(int(chip.buffer_ptr()), int(chip.buffer_size()))
    lib = _load(arch, "onboard", "tensormap_and_ringbuffer")
    _bind(lib)
    lib.rtSetDevice.argtypes = [ctypes.c_int]
    assert lib.rtSetDevice(device) == 0
    contexts = _CallerContexts(lib, device)
    ctx, caller = lib.create_device_context(), ctypes.c_void_p()
    assert ctx
    if scenario == "two-nodes":
        assert lib.simpler_set_run_resource_grant_ctx(ctx, 2) == 0
    assert lib.aclrtCreateStream(ctypes.byref(caller)) == 0
    aicpu, aicore, dispatcher = _binaries(arch, "tensormap_and_ringbuffer")
    config = CallConfig(aicpu_thread_num=2)
    assert (
        lib.simpler_kernel_mode_init(
            ctx, device, aicpu, len(aicpu), aicore, len(aicore), dispatcher, len(dispatcher), ctypes.byref(config), 1
        )
        == 0
    )
    assert lib.simpler_kernel_mode_prepare_callable(ctx, 0, image, len(image)) == 0
    epochs = (1, 2) if scenario == "two-nodes" else (1,)
    models = [_Model(lib, ctx, caller, config, epochs)]
    if scenario == "serial-models":
        # The first model still holds its code, source image and device backing.
        models.append(_Model(lib, ctx, caller, config, (2,)))
    assert lib.simpler_unregister_callable(ctx, 0) != 0
    assert lib.simpler_kernel_mode_prepare_callable(ctx, 1, image, len(image)) == 0
    assert lib.simpler_unregister_callable(ctx, 1) == 0
    if scenario == "fatal":
        assert lib.aclmdlRIExecuteAsync(models[0].model, caller) == 0
        # The first error observer is RTS; no private terminal read precedes it.
        rc = lib.aclrtSynchronizeStreamWithTimeout(caller, 60000)
        assert rc == 507011, rc
        committed = lib.committed_device_memory_ctx(ctx)
        assert lib.finalize_device(ctx) != 0
        assert lib.committed_device_memory_ctx(ctx) == committed
        print(f"TMR_COMMON_CAPTURE_PASS {arch} fatal rts={rc} retained={committed}", flush=True)
        # An execution exception does not prove safe release of borrowed state.
        os._exit(0)
    rounds = 0 if scenario == "destroy-unexecuted" else 8
    for round_index in range(rounds):
        for index, model in enumerate(models):
            model.replay(10 * round_index + index)
    for index, model in enumerate(models):
        model.close()
        if index + 1 < len(models):
            assert lib.simpler_unregister_callable(ctx, 0) != 0
            models[index + 1].replay(100)
    assert lib.simpler_unregister_callable(ctx, 0) == 0
    assert lib.finalize_device(ctx) == 0
    lib.destroy_device_context(ctx)
    assert lib.aclrtDestroyStream(caller) == 0
    contexts.close()
    _reset_caller_device(lib, device)
    print(
        f"TMR_COMMON_CAPTURE_PASS {arch} {scenario} models={len(models)} rounds={rounds} verified={_COUNT}",
        flush=True,
    )


@pytest.mark.parametrize("scenario", ["replay", "serial-models", "two-nodes", "destroy-unexecuted", "fatal"])
@pytest.mark.parametrize(
    "arch",
    [
        pytest.param(arch, marks=[pytest.mark.requires_hardware, pytest.mark.platforms([arch])])
        for arch in ("a2a3", "a5")
    ],
)
def test_tmr_raw_replay_uses_common_prepared_resources(arch, scenario, request):
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), arch, device, scenario],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "TMR_COMMON_CAPTURE_PASS" in result.stdout


if __name__ == "__main__":
    _run(sys.argv[1], int(sys.argv[2]), sys.argv[3])
