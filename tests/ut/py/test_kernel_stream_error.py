# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The caller observes native execution status before any Simpler host drain."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_A5_LEGACY_GRAPH = 3


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3", "a5"])
@pytest.mark.parametrize(
    ("runtime", "error_origin"),
    [
        ("tensormap_and_ringbuffer", "aicpu"),
        ("tensormap_and_ringbuffer", "aicore"),
        ("host_build_graph", "aicore"),
    ],
)
@pytest.mark.parametrize("failure", [False, True], ids=["success", "reported-error"])
def test_kernel_error_reaches_caller_stream(runtime, error_origin, failure, request, tmp_path):
    arch = request.config.getoption("--platform")
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    logs = tmp_path / "ascend"
    logs.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            arch,
            runtime,
            error_origin,
            device,
            str(int(failure)),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "ASCEND_PROCESS_LOG_PATH": str(logs)},
    )
    (tmp_path / "native-caller.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    records = [
        line.removeprefix("NATIVE_CALLER_STATUS ")
        for line in result.stdout.splitlines()
        if line.startswith("NATIVE_CALLER_STATUS ")
    ]
    assert len(records) == 1, result.stdout + result.stderr
    observed = json.loads(records[0])
    assert observed["runtime"] == runtime
    assert observed["error_origin"] == error_origin
    assert observed["failure"] is failure
    assert observed["peek_before"] == 0, observed
    assert observed["caller_sync_rc"] == (507018 if failure else 0), observed
    if error_origin == "aicore" and not failure:
        assert observed["output_marker"] == 42, observed
    if arch == "a5" and runtime == "host_build_graph":
        assert observed["scheduler_runtime_mode"] == _A5_LEGACY_GRAPH, observed
        assert observed["scheduler_mode_publish_count"] > 0, observed


def _callable(arch, runtime, error_origin, failure, output):
    from simpler.task_interface import ArgDirection, ChipCallable, CoreCallable  # noqa: PLC0415

    from simpler_setup.elf_parser import extract_text_section  # noqa: PLC0415
    from simpler_setup.kernel_compiler import KernelCompiler  # noqa: PLC0415
    from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: PLC0415

    compiler = KernelCompiler(arch)
    children = []
    signature = []
    kernel_source = None
    kernel_binary = None
    if error_origin == "aicore":
        source = Path(__file__).with_name("support") / (
            "kernel_stream_graph.cpp" if runtime == "host_build_graph" else "kernel_stream_task.cpp"
        )
        kernel_source = (
            _ROOT / "tests/st/runtime_fatal_codes/kernels/aiv" / "kernel_async_completion_invalid.cpp"
            if failure
            else Path(__file__).with_name("support") / "kernel_stream_output.cpp"
        )
        kernel = compiler.compile_incore(
            source_path=str(kernel_source),
            core_type="aiv",
            pto_isa_root=ensure_pto_isa_root(),
            extra_include_dirs=list(compiler.get_orchestration_include_dirs(runtime)) + [str(_ROOT / "src/common")],
        )
        signature = [ArgDirection.INOUT]
        kernel_binary = extract_text_section(kernel)
        children = [(0, CoreCallable.build(signature=signature, binary=kernel_binary))]
    else:
        source = (
            _ROOT / "tests/st/automatic_reported_fatal.cpp"
            if failure
            else Path(__file__).with_name("support") / "kernel_stream_success.cpp"
        )
    binary = compiler.compile_orchestration(runtime_name=runtime, source_path=str(source), build_dir=str(output))
    chip = ChipCallable.build(
        signature=signature, func_name="aicpu_orchestration_entry", binary=binary, children=children
    )
    hashes = {
        "fixture_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "orchestration_sha256": hashlib.sha256(binary).hexdigest(),
    }
    hashes["callable_image_sha256"] = hashlib.sha256(
        ctypes.string_at(int(chip.buffer_ptr()), int(chip.buffer_size()))
    ).hexdigest()
    if kernel_source is not None:
        assert kernel_binary is not None
        hashes["kernel_fixture_sha256"] = hashlib.sha256(kernel_source.read_bytes()).hexdigest()
        hashes["core_binary_sha256"] = hashlib.sha256(kernel_binary).hexdigest()
    return chip, hashes


def _probe(arch, runtime, error_origin, device, failure, output):
    from simpler.task_interface import AddressSpace, ChipStorageTaskArgs, ChipTensor, DataType  # noqa: PLC0415
    from test_kernel_mode_c_api import (  # noqa: PLC0415
        CallConfig,
        NativeRunDescriptor,
        _binaries,
        _load,
        _native_run_storage,
    )

    chip, hashes = _callable(arch, runtime, error_origin, failure, output)
    image = ctypes.string_at(int(chip.buffer_ptr()), int(chip.buffer_size()))
    (output / "callable.bin").write_bytes(image)
    lib = _load(arch, "onboard", runtime)
    lib.rtSetDevice.argtypes = [ctypes.c_int]
    lib.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.aclrtDestroyStream.argtypes = [ctypes.c_void_p]
    lib.aclrtPeekAtLastError.argtypes = [ctypes.c_int]
    lib.aclrtSynchronizeStreamWithTimeout.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.simpler_prepare_run.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.POINTER(CallConfig),
        ctypes.POINTER(NativeRunDescriptor),
    ]
    lib.simpler_launch_run_on_stream.argtypes = [ctypes.c_void_p] * 4
    lib.simpler_finalize_run.argtypes = [ctypes.c_void_p] * 2
    assert lib.rtSetDevice(device) == 0
    caller = ctypes.c_void_p()
    assert lib.aclrtCreateStream(ctypes.byref(caller)) == 0
    ctx = lib.create_device_context()
    device_tensor = ctypes.c_void_p()
    if error_origin == "aicore":
        lib.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
        lib.aclrtFree.argtypes = [ctypes.c_void_p]
        lib.aclrtMemset.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_size_t]
        lib.aclrtMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        assert lib.aclrtMalloc(ctypes.byref(device_tensor), 4, 0) == 0
        assert lib.aclrtMemset(device_tensor, 4, 0, 4) == 0
    config = CallConfig()
    config.aicpu_thread_num = 2
    aicpu, aicore, dispatcher = _binaries(arch, runtime)
    assert (
        lib.simpler_kernel_mode_init(
            ctx, device, aicpu, len(aicpu), aicore, len(aicore), dispatcher, len(dispatcher), ctypes.byref(config), 1
        )
        == 0
    )
    assert lib.simpler_kernel_mode_prepare_callable(ctx, 0, image, len(image)) == 0
    storage, run = _native_run_storage(lib)
    args = ChipStorageTaskArgs()
    if device_tensor.value is not None:
        args.add_tensor(ChipTensor.make(device_tensor.value, (1,), DataType.INT32, address_space=AddressSpace.DEVICE))
    descriptor = NativeRunDescriptor(generation=1, run_epoch=1)
    assert lib.simpler_prepare_run(ctx, run, 0, args.__ptr__(), ctypes.byref(config), ctypes.byref(descriptor)) == 0
    assert lib.simpler_launch_run_on_stream(ctx, run, caller, None) == 0
    # This is the first execution observer: neither native wait nor finalize ran.
    peek_before = lib.aclrtPeekAtLastError(0)
    started = time.monotonic()
    rc = lib.aclrtSynchronizeStreamWithTimeout(caller, 60000)
    elapsed = time.monotonic() - started
    mode = count = 0
    if arch == "a5" and runtime == "host_build_graph":
        lib.get_last_scheduler_runtime_mode.argtypes = [ctypes.c_void_p]
        lib.get_last_scheduler_runtime_mode.restype = ctypes.c_uint32
        lib.get_scheduler_mode_publish_count.argtypes = [ctypes.c_void_p]
        lib.get_scheduler_mode_publish_count.restype = ctypes.c_size_t
        mode = lib.get_last_scheduler_runtime_mode(ctx)
        count = lib.get_scheduler_mode_publish_count(ctx)
    marker = ctypes.c_int32(-1)
    if error_origin == "aicore" and not failure and rc == 0:
        assert lib.aclrtMemcpy(ctypes.byref(marker), 4, device_tensor, 4, 2) == 0
    print(
        "NATIVE_CALLER_STATUS "
        + json.dumps(
            {
                "runtime": runtime,
                "error_origin": error_origin,
                "scheduler_runtime_mode": mode,
                "scheduler_mode_publish_count": count,
                "output_marker": marker.value,
                "observer_seconds": elapsed,
                "failure": failure,
                "caller_sync_rc": rc,
                "peek_before": peek_before,
                **hashes,
            }
        ),
        flush=True,
    )
    if failure:
        # A fresh child samples exactly one error observer; teardown cannot add another.
        time.sleep(1)
        os._exit(0)
    assert lib.simpler_finalize_run(ctx, run) == 0
    assert lib.finalize_device(ctx) == 0
    lib.destroy_device_context(ctx)
    assert lib.aclrtSynchronizeStreamWithTimeout(caller, 60000) == 0
    assert lib.aclrtDestroyStream(caller) == 0
    if device_tensor.value is not None:
        assert lib.aclrtFree(device_tensor) == 0
    assert storage is not None


if __name__ == "__main__":
    _probe(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), bool(int(sys.argv[5])), Path(sys.argv[6]))
