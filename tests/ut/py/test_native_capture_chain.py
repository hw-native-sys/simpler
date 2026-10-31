# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real common HBG capture, replay after native finalize, and eager alternation."""

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
)


def _run(device, worker_mode=0):  # noqa: PLR0912, PLR0915 - subprocess owns all three entry variants through device teardown.
    from _task_interface import _ChipWorker  # noqa: PLC0415
    from simpler.buffer import Buffer  # noqa: PLC0415
    from simpler.task_interface import (  # noqa: PLC0415  # noqa: PLC0415
        AddressSpace,
        ChipStorageTaskArgs,
        ChipTensor,
        DataType,
        TaskArgs,
        Tensor,
        TensorArgType,
    )
    from simpler.task_interface import CallConfig as WorkerCallConfig  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    probe = ctypes.CDLL(None)
    probe.capture_cold_watch.argtypes = [ctypes.c_int]
    probe.capture_cold_violations.restype = probe.capture_cold_observed.restype = ctypes.c_uint64
    lib = _load("a2a3", "onboard", "host_build_graph")
    lib.rtSetDevice.argtypes = [ctypes.c_int]
    assert lib.rtSetDevice(device) == 0
    contexts = _CallerContexts(lib, device)
    lib.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
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
    lib.aclmdlRICaptureBegin.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.aclmdlRICaptureEnd.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.aclmdlRIExecuteAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.aclmdlRIDestroy.argtypes = [ctypes.c_void_p]
    lib.simpler_poll_run.argtypes = lib.simpler_wait_run.argtypes = [ctypes.c_void_p] * 2
    lib.simpler_retire_capture_resources.argtypes = [ctypes.c_void_p]
    lib.simpler_reserve_capture_resources.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    ctx = None if worker_mode else lib.create_device_context()
    worker = _ChipWorker() if worker_mode == 1 else None
    owner = Worker(level=2, device_id=device, platform="a2a3", runtime="host_build_graph") if worker_mode == 2 else None
    public_callable = None
    public_args = TaskArgs()
    backings = []
    stream, model = ctypes.c_void_p(), ctypes.c_void_p()
    devices, hosts = [ctypes.c_void_p() for _ in range(3)], [ctypes.c_void_p() for _ in range(3)]
    size = 128 * 128
    nbytes = size * ctypes.sizeof(ctypes.c_float)
    assert lib.aclrtCreateStream(ctypes.byref(stream)) == 0
    for ptr in devices:
        assert lib.aclrtMalloc(ctypes.byref(ptr), nbytes, 0) == 0
    for ptr in hosts:
        assert lib.aclrtMallocHost(ctypes.byref(ptr), nbytes) == 0
    views = []
    for ptr in hosts:
        assert ptr.value is not None
        views.append((ctypes.c_float * size).from_address(ptr.value))
    aicpu, aicore, dispatcher = _binaries("a2a3", "host_build_graph")
    config = CallConfig()
    # Exercise multiple AICPU participants in replay completion accounting.
    config.aicpu_thread_num = 2
    from simpler.task_interface import ArgDirection as D  # noqa: PLC0415

    from simpler_setup.scene_test import compile_chip_callable_spec  # noqa: PLC0415

    kernels = Path(__file__).resolve().parents[2] / "st/a2a3/host_build_graph/vector_example/kernels/aiv"
    spec = {
        "orchestration": {
            "source": str(Path(__file__).with_name("support") / "native_capture_version_orch.cpp"),
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT],
        },
        "incores": [
            {"func_id": i, "source": str(kernels / name), "core_type": "aiv", "signature": signature}
            for i, name, signature in [
                (0, "kernel_add.cpp", [D.IN, D.IN, D.OUT]),
                (1, "kernel_add_scalar.cpp", [D.IN, D.OUT]),
                (2, "kernel_mul.cpp", [D.IN, D.IN, D.OUT]),
            ]
        ],
    }
    chip = compile_chip_callable_spec(spec, "a2a3", "host_build_graph", (__name__, "parameter_versions"))
    scalars = (1.0, 2.0)
    if owner is not None:
        public_callable = owner.register(chip)
        owner.init(borrowed_context_generation=77)
        for index, ptr in enumerate(devices):
            assert ptr.value is not None
            backing = Buffer.wrap(address=ptr.value, nbytes=nbytes, location=owner.device_location)
            backings.append(backing)
            public_args.add_tensor(
                Tensor(backing, shapes=(size,), dtype=DataType.FLOAT32),
                TensorArgType.INPUT if index < 2 else TensorArgType.OUTPUT_EXISTING,
            )
    elif worker is not None:
        root = Path(__file__).resolve().parents[3] / "build/lib/a2a3"
        base = root / "onboard/host_build_graph"
        worker.init(
            str(base / "libhost_runtime.so"),
            str(base / "libaicpu_kernel.so"),
            str(base / "aicore_kernel.o"),
            str(root / "dispatcher/libsimpler_aicpu_dispatcher.so"),
            device,
            borrowed_context_generation=77,
        )
        assert worker.register_callable(chip) == 0
    else:
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
        image = ctypes.string_at(int(chip.buffer_ptr()), int(chip.buffer_size()))
        callable_id = ctypes.c_int32(-1)
        assert lib.simpler_register_callable_v2(ctx, image, len(image), ctypes.byref(callable_id)) == 0
        assert callable_id.value == 0
    storage, run = _native_run_storage(lib)
    args = ChipStorageTaskArgs()
    for ptr in devices:
        args.add_tensor(ChipTensor.make(ptr.value, (size,), DataType.FLOAT32, address_space=AddressSpace.DEVICE))
    for scalar in scalars:
        args.add_scalar(scalar)
        public_args.add_scalar(scalar)
    epoch = 0

    def prepare():
        nonlocal epoch
        epoch += 1
        receipt = ctypes.c_int32(17)
        descriptor = NativeRunDescriptor(
            generation=1,
            run_epoch=epoch,
            caller_stream=stream.value,
            accepted_state=ctypes.pointer(receipt),
            accepted_value=23,
            joinable_boundary=1,
        )
        rc = lib.simpler_prepare_run(ctx, run, 0, args.__ptr__(), ctypes.byref(config), ctypes.byref(descriptor))
        assert rc == 0, ("prepare", rc)
        return descriptor, receipt

    def inputs(value):
        for i in range(2):
            views[i][:] = [float(value + i)] * size
            assert lib.aclrtMemcpyAsync(devices[i], nbytes, hosts[i], nbytes, 1, stream) == 0
        # A skipped replay must not pass by leaving a previous correct output.
        views[2][:] = [-999.0] * size
        assert lib.aclrtMemcpyAsync(devices[2], nbytes, hosts[2], nbytes, 1, stream) == 0

    def output(value):
        assert lib.aclrtMemcpyAsync(hosts[2], nbytes, devices[2], nbytes, 2, stream) == 0
        rc = lib.aclrtSynchronizeStreamWithTimeout(stream, 60000)
        assert rc == 0, ("caller drain", rc)
        total = 2 * value + 1
        expected = float((total + scalars[0]) * (total + scalars[1]))
        assert all(x == expected for x in views[2]), (value, list(views[2][:8]), expected)

    def eager(value):
        if owner is not None:
            inputs(value)
            eager_run = owner.submit(public_callable, public_args, caller_stream=stream.value)
            assert not eager_run.recorded
            output(value)
            eager_run.wait()
            return
        if worker is not None:
            inputs(value)
            eager_run = worker._submit_chip_run_direct(0, args, WorkerCallConfig(), caller_stream=stream.value)
            assert eager_run.launched and not eager_run.recorded
            output(value)
            eager_run.wait()
            return
        descriptor, receipt = prepare()
        inputs(value)
        assert lib.simpler_launch_run(ctx, run) == 0
        output(value)
        assert lib.simpler_wait_run(ctx, run) == 0
        assert lib.simpler_finalize_run(ctx, run) == 0
        assert receipt.value == 23
        assert descriptor.run_epoch == epoch

    def record():
        if owner is not None:
            recorded = owner.submit(public_callable, public_args, caller_stream=stream.value)
            assert recorded.recorded
            recorded.wait()
            assert recorded.done
            return
        if worker is not None:
            recorded = worker._submit_chip_run_direct(0, args, WorkerCallConfig(), caller_stream=stream.value)
            assert recorded.recorded and not recorded.launched
            recorded.wait()
            return
        descriptor, receipt = prepare()
        rc = lib.simpler_launch_run(ctx, run)
        assert rc == -1008, ("record", rc)
        assert receipt.value == 17
        assert lib.simpler_poll_run(ctx, run) == -1008
        assert lib.simpler_wait_run(ctx, run) == -1003
        assert lib.simpler_finalize_run(ctx, run) == 0
        assert ctypes.string_at(run, 8) == bytes(8)
        assert receipt.value == 17

    def retire():
        if owner is not None:
            owner.retire_capture_resources()
        elif worker is not None:
            worker.retire_capture_resources()
        else:
            assert lib.simpler_retire_capture_resources(ctx) == 0

    def reserve():
        assert stream.value is not None
        if owner is not None:
            owner.reserve_capture_resources(caller_stream=stream.value)
        elif worker is not None:
            worker.reserve_capture_resources(stream.value)
        else:
            assert lib.simpler_reserve_capture_resources(ctx, stream, 0, 0) == 0

    def begin():
        assert lib.aclmdlRICaptureBegin(stream, 0) == 0
        probe.capture_cold_watch(1)

    def end():
        probe.capture_cold_watch(0)
        rc = lib.aclmdlRICaptureEnd(stream, ctypes.byref(model))
        assert probe.capture_cold_violations() == 0
        return rc

    # Warm the existing slot, descriptor and private streams before capture.
    eager(1)
    print("warm eager passed", flush=True)
    assert probe.capture_cold_observed() > 0
    # Warm eager alone is not permission to allocate missing capture resources.
    begin()
    try:
        if owner is not None:
            with pytest.raises(RuntimeError, match="-1001"):
                owner.submit(public_callable, public_args, caller_stream=stream.value)
        elif worker is not None:
            with pytest.raises(RuntimeError, match="-1001"):
                worker._submit_chip_run_direct(0, args, WorkerCallConfig(), caller_stream=stream.value)
        else:
            epoch += 1
            descriptor = NativeRunDescriptor(generation=1, run_epoch=epoch, caller_stream=stream.value)
            assert (
                lib.simpler_prepare_run(ctx, run, 0, args.__ptr__(), ctypes.byref(config), ctypes.byref(descriptor))
                == -1001
            )
    finally:
        end()
        if model.value is not None:
            assert lib.aclmdlRIDestroy(model) == 0
        model = ctypes.c_void_p()
    print("missing reservation refused without cold allocation", flush=True)
    reserve()
    reserved_operations = probe.capture_cold_observed()
    reserve()
    assert probe.capture_cold_observed() == reserved_operations
    begin()
    try:
        record()
    finally:
        assert end() == 0
    print("common capture and native finalize passed", flush=True)
    if not worker_mode:
        assert lib.finalize_device(ctx) == -1003
        # A distinct image publishes new code and marks the private stream stale.
        # The old graph must survive the subsequent eager stream rotation.
        unrelated = _callable("a2a3", "host_build_graph")
        unrelated_image = ctypes.string_at(int(unrelated.buffer_ptr()), int(unrelated.buffer_size()))
        unrelated_id = ctypes.c_int32(-1)
        assert (
            lib.simpler_register_callable_v2(ctx, unrelated_image, len(unrelated_image), ctypes.byref(unrelated_id))
            == 0
        )
        assert unrelated_id.value != 0
        lib.simpler_unregister_callable.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        assert lib.simpler_unregister_callable(ctx, 0) == -1003
        assert lib.simpler_unregister_callable(ctx, unrelated_id.value) == 0
    # The captured native epoch stays fixed while replay count and the epochs
    # of interleaved eager calls advance independently.
    for value in range(2, 34):
        inputs(value)
        assert lib.aclmdlRIExecuteAsync(model, stream) == 0
        output(value)
        print(f"replay {value - 1} passed", flush=True)
        eager(value + 20)
    # Publish a new immutable parameter version with new addresses and scalars.
    # Both model handles remain usable: this is not an in-place RTS graph update.
    previous = (model, devices, args, public_args, scalars)
    devices = [ctypes.c_void_p() for _ in range(3)]
    for ptr in devices:
        assert lib.aclrtMalloc(ctypes.byref(ptr), nbytes, 0) == 0
    args, public_args = ChipStorageTaskArgs(), TaskArgs()
    for index, ptr in enumerate(devices):
        assert ptr.value is not None
        args.add_tensor(ChipTensor.make(ptr.value, (size,), DataType.FLOAT32, address_space=AddressSpace.DEVICE))
        if owner is not None:
            backing = Buffer.wrap(address=ptr.value, nbytes=nbytes, location=owner.device_location)
            backings.append(backing)
            public_args.add_tensor(
                Tensor(backing, shapes=(size,), dtype=DataType.FLOAT32),
                TensorArgType.INPUT if index < 2 else TensorArgType.OUTPUT_EXISTING,
            )
    scalars = (7.0, 11.0)
    for scalar in scalars:
        args.add_scalar(scalar)
        public_args.add_scalar(scalar)
    eager(61)
    reserve()
    model = ctypes.c_void_p()
    begin()
    record()
    assert end() == 0
    current = (model, devices, args, public_args, scalars)
    for version in (previous, current, previous, current):
        model, devices, args, public_args, scalars = version
        inputs(73)
        assert lib.aclmdlRIExecuteAsync(model, stream) == 0
        output(73)
    assert lib.aclmdlRIDestroy(current[0]) == 0
    model, devices, args, public_args, scalars = previous
    inputs(79)
    assert lib.aclmdlRIExecuteAsync(model, stream) == 0
    output(79)
    print("old/new graph versions preserve distinct tensor addresses and scalar values", flush=True)
    if not worker_mode:
        assert lib.simpler_retire_capture_resources(ctx) == -1003
    else:
        with pytest.raises(RuntimeError, match="-1003"):
            retire()
    assert lib.aclmdlRIDestroy(model) == 0
    retire()
    retire()
    for ptr in current[1]:
        assert lib.aclrtFree(ptr) == 0
    # Destruction of a never-executed graph must not wait on captured events.
    reserve()
    begin()
    record()
    assert end() == 0
    assert lib.aclmdlRIDestroy(model) == 0
    retire()
    print("never-replayed graph retirement passed", flush=True)
    # An unjoined fork makes CaptureEnd fail; its model destroy notification
    # must still permit explicit retirement without waiting on recorded events.
    lib.aclrtCreateEventExWithFlag.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32]
    lib.aclrtRecordEvent.argtypes = lib.aclrtStreamWaitEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.aclrtDestroyEvent.argtypes = [ctypes.c_void_p]
    child, fork = ctypes.c_void_p(), ctypes.c_void_p()
    assert lib.aclrtCreateStream(ctypes.byref(child)) == 0
    assert lib.aclrtCreateEventExWithFlag(ctypes.byref(fork), 1) == 0
    model = ctypes.c_void_p()
    reserve()
    begin()
    record()
    assert lib.aclrtRecordEvent(fork, stream) == 0
    assert lib.aclrtStreamWaitEvent(child, fork) == 0
    assert end() != 0
    assert model.value is None
    retire()
    assert lib.aclrtDestroyEvent(fork) == 0
    assert lib.aclrtDestroyStream(child) == 0
    eager(44)
    print("failed capture end retirement passed", flush=True)
    if owner is not None:
        for backing in backings:
            backing.close()
        owner.close()
    elif worker is not None:
        worker.finalize()
    else:
        assert lib.finalize_device(ctx) == 0
        lib.destroy_device_context(ctx)
    for ptr in devices:
        assert lib.aclrtFree(ptr) == 0
    for ptr in hosts:
        assert lib.aclrtFreeHost(ptr) == 0
    assert lib.aclrtDestroyStream(stream) == 0
    contexts.close()
    assert storage is not None
    assert probe.capture_cold_violations() == 0
    print("graph retirement and final eager passed; capture cold operations = 0", flush=True)


_COLD_PROBE_REASON = "the cold probe intercepts libdl symbols through LD_PRELOAD, which only Linux honours"


@pytest.fixture(scope="module")
def cold_probe(tmp_path_factory):
    if sys.platform != "linux":
        pytest.skip(_COLD_PROBE_REASON)
    output = tmp_path_factory.mktemp("capture-cold-probe") / "probe.so"
    source = Path(__file__).with_name("support") / "native_capture_cold_probe.cpp"
    subprocess.run(["g++", "-shared", "-fPIC", "-std=c++17", str(source), "-ldl", "-o", str(output)], check=True)
    return output


@pytest.mark.skipif(sys.platform != "linux", reason=_COLD_PROBE_REASON)
def test_cold_probe_observes_allocations_frees_and_local_hal_unmap(cold_probe, tmp_path):
    source = tmp_path / "hal.cpp"
    library = tmp_path / "hal.so"
    source.write_text('extern "C" int halHostUnregister(void *, unsigned) { return 73; }\n')
    subprocess.run(["g++", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
    script = r"""
import ctypes
import sys
probe = ctypes.CDLL(None)
hal = ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_LOCAL)
assert hal.halHostUnregister(None, 0) == 73
probe.capture_cold_watch(1)
for name, args in [
    ("rtMalloc", (None, 0, 0, 0)), ("rtMallocHost", (None, 0, 0)),
    ("aclrtMalloc", (None, 0, 0)), ("aclrtMallocHost", (None, 0)),
    ("aclrtCreateEventExWithFlag", (None, 0)), ("rtStreamCreate", (None, 0)),
    ("rtStreamCreateWithFlags", (None, 0, 0)), ("rtFree", (None,)),
    ("rtFreeHost", (None,)), ("aclrtFree", (None,)), ("aclrtFreeHost", (None,))
]:
    assert getattr(probe, name)(*args) == -778, name
assert hal.halHostUnregister(None, 0) == -778
assert probe.capture_cold_violations() == 12
probe.capture_cold_watch(0)
assert hal.halHostUnregister(None, 0) == 73
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(library)],
        env=dict(os.environ, LD_PRELOAD=str(cold_probe)),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("worker_mode", [0, 1, 2], ids=["c_api", "chip_worker", "worker"])
@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
def test_native_capture_replay_survives_finalize_and_alternating_eager(request, worker_mode, cold_probe):
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    env = dict(os.environ)
    env["LD_PRELOAD"] = str(cold_probe) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), device, str(int(worker_mode))],
        capture_output=True,
        env=env,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)


if __name__ == "__main__":
    _run(int(sys.argv[1]), int(sys.argv[2]))
