# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native CANN capture/replay with a retained TMR execution image."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_kernel_caller_stream import _callable
from test_kernel_mode_c_api import (
    PTO_RUNTIME_ERR_INVALID_ARGUMENT,
    PTO_RUNTIME_ERR_INVALID_STATE,
    CallConfig,
    NativeRunDescriptor,
    _binaries,
    _CallerContexts,
    _load,
    _native_run_storage,
    kernel_close_faults,  # noqa: F401 -- shared fault-injection fixture
)

_ROOT = Path(__file__).resolve().parents[3]


def _capture_api(lib):
    lib.simpler_register_callable.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
    lib.simpler_register_callable.restype = ctypes.c_int
    lib.simpler_kernel_capture_supported.argtypes = [ctypes.c_void_p]
    lib.simpler_prepare_run.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.POINTER(CallConfig),
        ctypes.POINTER(NativeRunDescriptor),
    ]
    lib.simpler_retain_run_resources.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.simpler_release_run_resources.argtypes = [ctypes.c_void_p] * 2
    lib.simpler_finalize_run.argtypes = [ctypes.c_void_p] * 2
    lib.aclmdlRICaptureBegin.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.aclmdlRICaptureEnd.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.aclmdlRIExecuteAsync.argtypes = [ctypes.c_void_p] * 2
    lib.aclmdlRIDestroy.argtypes = [ctypes.c_void_p]
    lib.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.aclrtDestroyStream.argtypes = [ctypes.c_void_p]
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
    lib.aclrtSynchronizeStreamWithTimeout.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.simpler_launch_run_on_stream.argtypes = [ctypes.c_void_p] * 4
    lib.simpler_wait_run.argtypes = [ctypes.c_void_p] * 2


def _run_vectors(arch, device, revoked, release_retry=False, prepare_refusal=False, recapture=False):  # noqa: PLR0912, PLR0915 -- one child owns the complete native lifecycle
    from simpler.task_interface import AddressSpace, ChipStorageTaskArgs, ChipTensor, DataType  # noqa: PLC0415

    chip = _callable(arch, "tensormap_and_ringbuffer")
    image = ctypes.string_at(int(chip.buffer_ptr()), int(chip.buffer_size()))
    lib = _load(arch, "onboard", "tensormap_and_ringbuffer")
    _capture_api(lib)
    faults = ctypes.CDLL(None)
    if release_retry or prepare_refusal or recapture:
        faults.bind_test_log.argtypes = [ctypes.c_void_p]
        assert faults.bind_test_log(lib._handle) == 0
    if recapture:
        faults.capture_control_address.restype = ctypes.c_uint64
        faults.capture_control_generation.restype = ctypes.c_uint64
    lib.rtSetDevice.argtypes = [ctypes.c_int]
    assert lib.rtSetDevice(device) == 0
    contexts = _CallerContexts(lib, device)
    ctx = lib.create_device_context()
    stream = ctypes.c_void_p()
    model = ctypes.c_void_p()
    reference = ctypes.c_void_p()
    device_buffers = [ctypes.c_void_p() for _ in range(3)]
    host_buffers = [ctypes.c_void_p() for _ in range(3)]
    count = 128 * 128
    nbytes = count * ctypes.sizeof(ctypes.c_float)
    prepared = capturing = False
    core_launches = 0
    storage = run = None
    try:
        assert lib.simpler_kernel_capture_supported(ctx) == 1
        assert lib.aclrtCreateStream(ctypes.byref(stream)) == 0
        for ptr in device_buffers:
            assert lib.aclrtMalloc(ctypes.byref(ptr), nbytes, 0) == 0
        for ptr in host_buffers:
            assert lib.aclrtMallocHost(ctypes.byref(ptr), nbytes) == 0
        views = []
        for ptr in host_buffers:
            assert ptr.value is not None
            views.append((ctypes.c_float * count).from_address(ptr.value))
        aicpu, aicore, dispatcher = _binaries(arch, "tensormap_and_ringbuffer")
        config = CallConfig(aicpu_thread_num=2)
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
                0x2625,
            )
            == 0
        )
        assert lib.simpler_kernel_mode_prepare_callable(ctx, 0, image, len(image)) == 0
        args = ChipStorageTaskArgs()
        for ptr in device_buffers:
            args.add_tensor(ChipTensor.make(ptr.value, (count,), DataType.FLOAT32, address_space=AddressSpace.DEVICE))
        storage, run = _native_run_storage(lib)
        descriptor = NativeRunDescriptor(generation=1, run_epoch=1)
        if prepare_refusal:
            # Establish the slot's pinned source without launching either
            # processor, then fail capture's publication from that source.
            assert (
                lib.simpler_prepare_run(ctx, run, 0, args.__ptr__(), ctypes.byref(config), ctypes.byref(descriptor))
                == 0
            )
            assert lib.simpler_finalize_run(ctx, run) == 0
            assert faults.arm_descriptor_copy_failure() == 0
            core_launches = faults.core_launch_call_count()
        prepared = True
        if recapture:
            faults.arm_capture_control_reuse()
        assert lib.simpler_prepare_run(ctx, run, 0, args.__ptr__(), ctypes.byref(config), ctypes.byref(descriptor)) == 0
        rc = lib.simpler_retain_run_resources(ctx, run, stream, ctypes.byref(reference))
        if prepare_refusal:
            assert rc == -4321
            assert faults.descriptor_copy_failures() == 1
            assert faults.descriptor_kernel_guard_hits() == 0
            assert faults.core_launch_call_count() == core_launches
            assert faults.descriptor_source_frees() == 0
            assert reference.value is None
            assert lib.simpler_finalize_run(ctx, run) == 0
            prepared = False
            assert faults.descriptor_source_frees() == 0
            print("NATIVE_CAPTURE_PASS", arch, "prepare-refusal", flush=True)
            return
        assert rc == 0, ("resource prewarm", rc)
        assert reference.value is not None
        first_address = faults.capture_control_address() if recapture else 0
        previous_generation = faults.capture_control_generation() if recapture else 0
        if recapture:
            assert first_address != 0 and previous_generation != 0
            print("CAPTURE_CONTROL", 0, hex(first_address), previous_generation, flush=True)
        if revoked:
            auto_id = ctypes.c_int32(999)
            resident = lib.committed_device_memory_ctx(ctx)
            with contexts.other():
                assert lib.simpler_register_callable(ctx, 1, image) == PTO_RUNTIME_ERR_INVALID_STATE
                assert (
                    lib.simpler_kernel_mode_prepare_callable(ctx, 2, image, len(image)) == PTO_RUNTIME_ERR_INVALID_STATE
                )
                assert (
                    lib.simpler_register_callable_v2(ctx, image, len(image), ctypes.byref(auto_id))
                    == PTO_RUNTIME_ERR_INVALID_STATE
                )
                assert auto_id.value == -1
                assert lib.simpler_unregister_callable(ctx, 0) == PTO_RUNTIME_ERR_INVALID_STATE
            assert lib.committed_device_memory_ctx(ctx) == resident
            assert lib.simpler_register_callable(ctx, 0, image) == PTO_RUNTIME_ERR_INVALID_ARGUMENT
            assert (
                lib.simpler_kernel_mode_prepare_callable(ctx, 0, image, len(image)) == PTO_RUNTIME_ERR_INVALID_ARGUMENT
            )
            assert lib.simpler_register_callable(ctx, 1, image) == 0
            assert lib.simpler_kernel_mode_prepare_callable(ctx, 2, image, len(image)) == 0
            assert lib.simpler_register_callable_v2(ctx, image, len(image), ctypes.byref(auto_id)) == 0
            assert auto_id.value == 3
            for callable_id in (1, 2, 3):
                assert lib.simpler_unregister_callable(ctx, callable_id) == 0
        assert lib.finalize_device(ctx) != 0
        other_storage, other_run = _native_run_storage(lib)
        assert (
            lib.simpler_prepare_run(
                ctx,
                other_run,
                0,
                args.__ptr__(),
                ctypes.byref(config),
                ctypes.byref(descriptor),
            )
            != 0
        )
        # Mutating the source argument object after preparation cannot change
        # the captured counts or payload; the registered vector has no scalars.
        args.add_scalar(111)
        print("CAPTURE_BEGIN", arch, flush=True)
        assert lib.aclmdlRICaptureBegin(stream, 0) == 0
        capturing = True
        rc = lib.simpler_launch_run_on_stream(ctx, run, stream, None)
        print("CAPTURE_RECORD", rc, flush=True)
        assert rc == 0
        assert lib.aclmdlRICaptureEnd(stream, ctypes.byref(model)) == 0
        capturing = False
        assert model.value is not None
        assert lib.simpler_launch_run_on_stream(ctx, run, stream, None) != 0
        assert lib.simpler_finalize_run(ctx, run) == 0
        prepared = False
        assert ctypes.string_at(run, 8) == bytes(8)
        rounds = 2 if revoked else 8
        for epoch in range(1, rounds + 1):
            for index in range(2):
                views[index][:] = [float(epoch + index)] * count
                assert lib.aclrtMemcpyAsync(device_buffers[index], nbytes, host_buffers[index], nbytes, 1, stream) == 0
            assert lib.aclmdlRIExecuteAsync(model, stream) == 0
            if revoked and epoch == rounds:
                assert lib.simpler_register_callable(ctx, 1, image) == 0
                assert lib.simpler_unregister_callable(ctx, 1) == 0
            assert lib.aclrtMemcpyAsync(host_buffers[2], nbytes, device_buffers[2], nbytes, 2, stream) == 0
            rc = lib.simpler_wait_run(ctx, reference)
            print("REPLAY", epoch, rc, flush=True)
            assert rc == 0
            total = float(2 * epoch + 1)
            expected = (total + 1) * (total + 2) + total
            assert all(value == expected for value in views[2]), (epoch, list(views[2][:8]), expected)
        if revoked:
            assert lib.simpler_unregister_callable(ctx, 0) == 0
            auto_id = ctypes.c_int32(999)
            assert lib.simpler_register_callable_v2(ctx, image, len(image), ctypes.byref(auto_id)) == 0
            assert auto_id.value == 0
            views[2][:] = [-123.0] * count
            assert lib.aclrtMemcpyAsync(device_buffers[2], nbytes, host_buffers[2], nbytes, 1, stream) == 0
            # All three execute requests are queued before observing the first
            # refusal. Each retires through the platform handshake without
            # creating a user Runtime or dispatching user tasks.
            for _ in range(3):
                assert lib.aclmdlRIExecuteAsync(model, stream) == 0
            assert lib.aclrtMemcpyAsync(host_buffers[2], nbytes, device_buffers[2], nbytes, 2, stream) == 0
            assert lib.simpler_wait_run(ctx, reference) == -1003
            assert lib.simpler_wait_run(ctx, reference) == -1003
            assert all(value == -123.0 for value in views[2])
        assert lib.aclmdlRIDestroy(model) == 0
        model.value = None
        if release_retry:
            faults = ctypes.CDLL(None)
            faults.arm_destroy_failure.argtypes = [ctypes.c_int]
            committed = lib.committed_device_memory_ctx(ctx)
            faults.arm_destroy_failure(4)
            assert lib.simpler_release_run_resources(ctx, reference) == -4321
            assert lib.committed_device_memory_ctx(ctx) == committed
            assert lib.finalize_device(ctx) != 0
        assert lib.simpler_release_run_resources(ctx, reference) == 0
        reference.value = None
        # Reuse a host epoch equal to an old replay epoch. The released slot
        # must reinitialize its reports before eager execution can reuse it.
        fresh_args = ChipStorageTaskArgs()
        for ptr in device_buffers:
            fresh_args.add_tensor(
                ChipTensor.make(ptr.value, (count,), DataType.FLOAT32, address_space=AddressSpace.DEVICE)
            )
        if recapture:
            for lease in range(1, 4):
                descriptor = NativeRunDescriptor(generation=lease + 1, run_epoch=1)
                prepared = True
                assert (
                    lib.simpler_prepare_run(
                        ctx, run, 0, fresh_args.__ptr__(), ctypes.byref(config), ctypes.byref(descriptor)
                    )
                    == 0
                )
                assert lib.simpler_retain_run_resources(ctx, run, stream, ctypes.byref(reference)) == 0
                address = faults.capture_control_address()
                generation = faults.capture_control_generation()
                print("CAPTURE_CONTROL", lease, hex(address), generation, flush=True)
                assert address == first_address and generation > previous_generation
                previous_generation = generation
                assert lib.aclmdlRICaptureBegin(stream, 0) == 0
                capturing = True
                assert lib.simpler_launch_run_on_stream(ctx, run, stream, None) == 0
                assert lib.aclmdlRICaptureEnd(stream, ctypes.byref(model)) == 0
                capturing = False
                assert lib.simpler_finalize_run(ctx, run) == 0
                prepared = False
                assert ctypes.string_at(run, 8) == bytes(8)
                for epoch in (1, 2):
                    value = float(20 * lease + epoch)
                    for index in range(2):
                        views[index][:] = [value + index] * count
                        assert (
                            lib.aclrtMemcpyAsync(device_buffers[index], nbytes, host_buffers[index], nbytes, 1, stream)
                            == 0
                        )
                    assert lib.aclmdlRIExecuteAsync(model, stream) == 0
                    assert lib.aclrtMemcpyAsync(host_buffers[2], nbytes, device_buffers[2], nbytes, 2, stream) == 0
                    assert lib.simpler_wait_run(ctx, reference) == 0
                    total = 2 * value + 1
                    expected = (total + 1) * (total + 2) + total
                    assert all(item == expected for item in views[2])
                assert lib.aclmdlRIDestroy(model) == 0
                model.value = None
                assert lib.simpler_release_run_resources(ctx, reference) == 0
                reference.value = None
            assert faults.release_capture_control_reuse() == 0
        descriptor = NativeRunDescriptor(generation=2, run_epoch=1)
        assert (
            lib.simpler_prepare_run(
                ctx,
                run,
                0,
                fresh_args.__ptr__(),
                ctypes.byref(config),
                ctypes.byref(descriptor),
            )
            == 0
        )
        prepared = True
        assert lib.simpler_launch_run_on_stream(ctx, run, stream, None) == 0
        assert lib.aclrtMemcpyAsync(host_buffers[2], nbytes, device_buffers[2], nbytes, 2, stream) == 0
        assert lib.aclrtSynchronizeStreamWithTimeout(stream, 60000) == 0
        assert all(value == expected for value in views[2])
        assert lib.simpler_wait_run(ctx, run) == 0
        assert lib.simpler_finalize_run(ctx, run) == 0
        prepared = False
        assert other_storage is not None and storage is not None
        print("NATIVE_CAPTURE_PASS", arch, rounds, revoked, flush=True)
    finally:
        if capturing:
            lib.aclmdlRICaptureEnd(stream, ctypes.byref(model))
        if model.value:
            assert lib.aclmdlRIDestroy(model) == 0
        if prepared:
            rc = lib.simpler_finalize_run(ctx, run)
            if rc != 0:
                raise RuntimeError(f"run cleanup retained resources: {rc}")
        if reference.value:
            rc = lib.simpler_release_run_resources(ctx, reference)
            if rc != 0:
                raise RuntimeError(f"graph resource cleanup retained resources: {rc}")
            reference.value = None
        if recapture:
            assert faults.release_capture_control_reuse() == 0
        assert lib.finalize_device(ctx) == 0
        if prepare_refusal:
            assert faults.descriptor_source_frees() == 1
        lib.destroy_device_context(ctx)
        for buffers, release in ((device_buffers, lib.aclrtFree), (host_buffers, lib.aclrtFreeHost)):
            for ptr in buffers:
                if ptr.value:
                    assert release(ptr) == 0
        if stream.value:
            assert lib.aclrtDestroyStream(stream) == 0
        contexts.close()


@pytest.mark.parametrize("scenario", ["replay", "stale-queued", "release-retry", "prepare-refusal", "recapture-reuse"])
@pytest.mark.parametrize(
    "arch",
    [
        pytest.param(arch, marks=[pytest.mark.requires_hardware, pytest.mark.platforms([arch])])
        for arch in ("a2a3", "a5")
    ],
)
def test_native_kernel_capture(arch, scenario, kernel_close_faults, request):  # noqa: F811 -- imported pytest fixture
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    env = dict(os.environ)
    if scenario in ("release-retry", "prepare-refusal", "recapture-reuse"):
        env["LD_PRELOAD"] = str(kernel_close_faults) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), arch, device, scenario],
        check=False,
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NATIVE_CAPTURE_PASS" in result.stdout
    if scenario == "recapture-reuse":
        print(result.stdout)
        controls = [line.split() for line in result.stdout.splitlines() if line.startswith("CAPTURE_CONTROL ")]
        assert len(controls) == 4
        assert len({row[2] for row in controls}) == 1
        assert len({row[3] for row in controls}) == 4


def _state_callable(arch):
    from simpler.task_interface import ArgDirection as D  # noqa: PLC0415

    from simpler_setup.scene_test import compile_chip_callable_spec, l3_compile_cache_key  # noqa: PLC0415

    spec = {
        "orchestration": {
            "source": str(Path(__file__).with_name("kernel_capture_state.cpp")),
            "function_name": "captured_state",
            "signature": [D.INOUT, D.INOUT, D.SCALAR],
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
                "signature": [D.INOUT, D.SCALAR],
            }
        ],
    }
    key = l3_compile_cache_key(__name__, "CapturedState", "captured_state", arch, "tensormap_and_ringbuffer")
    return compile_chip_callable_spec(spec, arch, "tensormap_and_ringbuffer", key)


def _run_state(arch, device, fail):  # noqa: PLR0912, PLR0915 -- one child owns every native resource
    from simpler.task_interface import AddressSpace, ChipStorageTaskArgs, ChipTensor, DataType  # noqa: PLC0415

    chip = _state_callable(arch)
    image = ctypes.string_at(int(chip.buffer_ptr()), int(chip.buffer_size()))
    lib = _load(arch, "onboard", "tensormap_and_ringbuffer")
    _capture_api(lib)
    lib.rtSetDevice.argtypes = [ctypes.c_int]
    assert lib.rtSetDevice(device) == 0
    contexts = _CallerContexts(lib, device)
    ctx = lib.create_device_context()
    stream = ctypes.c_void_p()
    model = ctypes.c_void_p()
    reference = ctypes.c_void_p()
    output, control = ctypes.c_void_p(), ctypes.c_void_p()
    host_output, host_control = ctypes.c_void_p(), ctypes.c_void_p()
    commands = (2, 1, 1) if fail else (0, 1, 0, 1, 1, 0, 1)
    host_commands = [ctypes.c_void_p() for _ in commands]
    prepared = capturing = False
    storage = run = None
    try:
        assert lib.aclrtCreateStream(ctypes.byref(stream)) == 0
        assert lib.aclrtMalloc(ctypes.byref(output), 4, 0) == 0
        assert lib.aclrtMalloc(ctypes.byref(control), 8, 0) == 0
        assert lib.aclrtMallocHost(ctypes.byref(host_output), 4) == 0
        assert lib.aclrtMallocHost(ctypes.byref(host_control), 8) == 0
        for ptr, value in zip(host_commands, commands):
            assert lib.aclrtMallocHost(ctypes.byref(ptr), 4) == 0
            assert ptr.value is not None
            ctypes.c_int32.from_address(ptr.value).value = value
        assert host_output.value is not None and host_control.value is not None
        output_view = ctypes.c_int32.from_address(host_output.value)
        control_view = (ctypes.c_int32 * 2).from_address(host_control.value)
        output_view.value = -9
        control_view[:] = (0, 0)
        assert lib.aclrtMemcpyAsync(output, 4, host_output, 4, 1, stream) == 0
        assert lib.aclrtMemcpyAsync(control, 8, host_control, 8, 1, stream) == 0
        assert lib.aclrtSynchronizeStreamWithTimeout(stream, 60000) == 0
        aicpu, aicore, dispatcher = _binaries(arch, "tensormap_and_ringbuffer")
        config = CallConfig(aicpu_thread_num=2)
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
                0x2626,
            )
            == 0
        )
        assert lib.simpler_kernel_mode_prepare_callable(ctx, 0, image, len(image)) == 0
        args = ChipStorageTaskArgs()
        tensors = [
            ChipTensor.make(output.value, (1,), DataType.INT32, address_space=AddressSpace.DEVICE),
            ChipTensor.make(control.value, (2,), DataType.INT32, address_space=AddressSpace.DEVICE),
        ]
        for tensor in tensors:
            args.add_tensor(tensor)
        args.add_scalar(37)
        storage, run = _native_run_storage(lib)
        descriptor = NativeRunDescriptor(generation=1, run_epoch=1)
        prepared = True
        assert (
            lib.simpler_prepare_run(
                ctx,
                run,
                0,
                args.__ptr__(),
                ctypes.byref(config),
                ctypes.byref(descriptor),
            )
            == 0
        )
        assert lib.simpler_retain_run_resources(ctx, run, stream, ctypes.byref(reference)) == 0
        assert reference.value is not None
        args.clear()
        for tensor in tensors:
            args.add_tensor(tensor)
        args.add_scalar(99)
        assert lib.aclmdlRICaptureBegin(stream, 0) == 0
        capturing = True
        assert lib.simpler_launch_run_on_stream(ctx, run, stream, None) == 0
        assert lib.aclmdlRICaptureEnd(stream, ctypes.byref(model)) == 0
        capturing = False
        assert lib.simpler_finalize_run(ctx, run) == 0
        prepared = False
        assert ctypes.string_at(run, 8) == bytes(8)
        expected = -9
        for epoch, (command, host) in enumerate(zip(commands, host_commands), 1):
            assert lib.aclrtMemcpyAsync(control, 4, host, 4, 1, stream) == 0
            assert lib.aclmdlRIExecuteAsync(model, stream) == 0
            if not fail:
                assert lib.aclrtMemcpyAsync(host_output, 4, output, 4, 2, stream) == 0
                assert lib.aclrtMemcpyAsync(host_control, 8, control, 8, 2, stream) == 0
                assert lib.simpler_wait_run(ctx, reference) == 0
                if command:
                    expected = 37
                assert output_view.value == expected
                assert control_view[1] == epoch
        if fail:
            # R1's explicit fatal is submitted before both healthy successors,
            # with separate pinned H2D sources so their values cannot race.
            assert lib.aclrtMemcpyAsync(host_output, 4, output, 4, 2, stream) == 0
            assert lib.aclrtMemcpyAsync(host_control, 8, control, 8, 2, stream) == 0
            assert lib.simpler_wait_run(ctx, reference) == -9
            assert lib.simpler_wait_run(ctx, reference) == -9
            assert output_view.value == -9
            assert control_view[1] == 1
        assert lib.aclmdlRIDestroy(model) == 0
        model.value = None
        assert lib.simpler_release_run_resources(ctx, reference) == 0
        reference.value = None
        assert storage is not None
        print("NATIVE_CAPTURE_STATE_PASS", arch, fail, flush=True)
    finally:
        if capturing:
            lib.aclmdlRICaptureEnd(stream, ctypes.byref(model))
        if model.value:
            assert lib.aclmdlRIDestroy(model) == 0
        if prepared:
            rc = lib.simpler_finalize_run(ctx, run)
            if rc != 0:
                raise RuntimeError(f"state run cleanup retained resources: {rc}")
        if reference.value:
            rc = lib.simpler_release_run_resources(ctx, reference)
            if rc != 0:
                raise RuntimeError(f"state graph cleanup retained resources: {rc}")
            reference.value = None
        assert lib.finalize_device(ctx) == 0
        lib.destroy_device_context(ctx)
        for ptr in (output, control):
            if ptr.value:
                assert lib.aclrtFree(ptr) == 0
        for ptr in (host_output, host_control, *host_commands):
            if ptr.value:
                assert lib.aclrtFreeHost(ptr) == 0
        if stream.value:
            assert lib.aclrtDestroyStream(stream) == 0
        contexts.close()


@pytest.mark.parametrize("fail", [False, True], ids=["empty-active-snapshot", "error-queued"])
@pytest.mark.parametrize(
    "arch",
    [
        pytest.param(arch, marks=[pytest.mark.requires_hardware, pytest.mark.platforms([arch])])
        for arch in ("a2a3", "a5")
    ],
)
def test_native_capture_state(arch, fail, request):
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), arch, device, "state-error" if fail else "state"],
        check=False,
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NATIVE_CAPTURE_STATE_PASS" in result.stdout


@pytest.mark.parametrize("arch", ["a2a3", "a5"])
@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
def test_simulated_capture_capability_and_refusal(arch, runtime):
    lib = _load(arch, "sim", runtime)
    lib.simpler_kernel_capture_supported.argtypes = [ctypes.c_void_p]
    lib.simpler_retain_run_resources.argtypes = [ctypes.c_void_p] * 3 + [ctypes.POINTER(ctypes.c_void_p)]
    lib.simpler_release_run_resources.argtypes = [ctypes.c_void_p] * 2
    ctx = lib.create_device_context()
    storage, run = _native_run_storage(lib)
    reference = ctypes.c_void_p(1)
    try:
        assert lib.simpler_kernel_capture_supported(ctx) == 0
        assert lib.simpler_retain_run_resources(ctx, run, ctypes.c_void_p(1), ctypes.byref(reference)) == -1001
        assert reference.value is None
        assert lib.simpler_release_run_resources(ctx, ctypes.c_void_p(1)) == -1001
        assert storage is not None
    finally:
        lib.destroy_device_context(ctx)


if __name__ == "__main__":
    if sys.argv[3].startswith("state"):
        _run_state(sys.argv[1], int(sys.argv[2]), sys.argv[3] == "state-error")
    else:
        _run_vectors(
            sys.argv[1],
            int(sys.argv[2]),
            sys.argv[3] == "stale-queued",
            sys.argv[3] == "release-retry",
            sys.argv[3] == "prepare-refusal",
            sys.argv[3] == "recapture-reuse",
        )
