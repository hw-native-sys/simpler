# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Dedicated HBG production entry, fresh snapshots, capture and lifetime ownership."""

from __future__ import annotations

import ctypes
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest
from test_kernel_mode_c_api import CallConfig, _binaries, _CallerContexts, _load

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def hot_path_probe(tmp_path_factory):
    output = tmp_path_factory.mktemp("hbg-hot-path") / "probe.so"
    subprocess.run(
        [
            "c++",
            "-shared",
            "-std=c++17",
            "-fPIC",
            "-I" + str(_PROJECT_ROOT / "src/common"),
            str(Path(__file__).with_name("support") / "hbg_kernel_launch_probe.cpp"),
            "-ldl",
            "-o",
            str(output),
        ],
        check=True,
    )
    return output


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.parametrize(
    "scenario",
    [
        "eager",
        "queued",
        "capture",
        "capture_unused",
        "enqueue_failure",
        "close_retry",
        "capacity",
        "graph_contract",
        "graph_program",
        "graph_eager",
        "graph_capture",
        "register_enqueue_failure",
        "launch_exception",
        "binary_identity",
        "slot_enqueue_failure",
        "register_sync_failure",
    ],
)
@pytest.mark.parametrize("profile", ["default", "compact"])
def test_hbg_kernel_launch(request, scenario, hot_path_probe, profile):
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    env = dict(os.environ, LD_PRELOAD=str(hot_path_probe))
    result = subprocess.run(
        [sys.executable, __file__, device, scenario, profile],
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _build_eager_callable(arch, runtime, intermediate=False, graph=False):
    import tempfile  # noqa: PLC0415

    from simpler.task_interface import (  # noqa: PLC0415
        ArgDirection,
        ChipCallable,
        CoreCallable,
    )

    from simpler_setup.elf_parser import extract_text_section  # noqa: PLC0415
    from simpler_setup.kernel_compiler import KernelCompiler  # noqa: PLC0415
    from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: PLC0415

    compiler = KernelCompiler(arch)
    kernel = (
        _PROJECT_ROOT / f"examples/{arch}/tensormap_and_ringbuffer/vector_example/kernels/aiv/kernel_add_scalar.cpp"
    )
    with tempfile.TemporaryDirectory(prefix="kernel-eager-") as build_dir:
        orchestration = compiler.compile_orchestration(
            runtime, str(Path(__file__).with_name("kernel_eager_orchestration.cpp")), build_dir=build_dir
        )
        incore = compiler.compile_incore(
            str(kernel),
            core_type="aiv",
            pto_isa_root=ensure_pto_isa_root(),
            extra_include_dirs=compiler.get_orchestration_include_dirs(runtime),
            build_dir=build_dir,
        )
    signature = [ArgDirection.IN, ArgDirection.OUT, ArgDirection.SCALAR]
    child = CoreCallable.build(signature=signature, binary=extract_text_section(incore))
    return ChipCallable.build(
        signature=signature,
        func_name="kernel_eager_graph"
        if graph
        else ("kernel_eager_intermediate" if intermediate else "kernel_eager_orchestration"),
        binary=orchestration,
        children=[(0, child)],
    )


def _launch(lib, probe, ctx, cid, args, stream):
    probe.hbg_guard_hot_path(1)
    try:
        rc = lib.simpler_kernel_mode_launch(ctx, cid, args.__ptr__(), stream)
    finally:
        probe.hbg_guard_hot_path(0)
    assert probe.hbg_forbidden_hot_calls() == 0
    return rc


def _close(lib, ctx, stream, device, addresses):
    assert lib.finalize_device(ctx) == 0
    assert lib.committed_device_memory_ctx(ctx) == 0
    lib.destroy_device_context(ctx)
    for address in reversed(addresses):
        assert lib.aclrtFree(address) == 0
    assert lib.aclrtDestroyStream(stream) == 0
    assert lib.aclrtResetDevice(device) == 0
    assert lib.aclFinalize() == 0


def _graph_contract(lib, ctx, config, binaries, device, probe):
    first, second = ctypes.c_uint64(), ctypes.c_uint64()
    assert lib.simpler_kernel_mode_retain_graph(ctx, ctypes.byref(first)) == 0
    assert lib.simpler_kernel_mode_retain_graph(ctx, ctypes.byref(second)) == 0
    assert first.value not in (second.value, 0)
    probe.hbg_fail_host_new()
    failed = ctypes.c_uint64(999)
    assert lib.simpler_kernel_mode_retain_graph(ctx, ctypes.byref(failed)) == -1000
    assert failed.value == 0
    assert lib.finalize_device(ctx) == -1003
    contexts = _CallerContexts(lib, device)
    with contexts.other():
        rejected = ctypes.c_uint64(999)
        assert lib.simpler_kernel_mode_retain_graph(ctx, ctypes.byref(rejected)) == -1003
        assert rejected.value == 0
        assert lib.simpler_kernel_mode_release_graph(ctx, first.value) == -1003
    contexts.close()
    assert lib.simpler_kernel_mode_release_graph(ctx, first.value) == 0
    assert lib.finalize_device(ctx) == -1003
    probe.hbg_fail_owned_sync()
    assert lib.simpler_kernel_mode_release_graph(ctx, second.value) == -4328
    assert lib.finalize_device(ctx) == -1003
    probe.hbg_throw_owned_sync()
    assert lib.simpler_kernel_mode_release_graph(ctx, second.value) == -1000
    assert lib.finalize_device(ctx) == -1003
    assert lib.simpler_kernel_mode_release_graph(ctx, second.value) == 0
    assert lib.finalize_device(ctx) == 0
    lib.destroy_device_context(ctx)
    replacement = lib.create_device_context()
    cpu, core, dispatcher = binaries
    assert (
        lib.simpler_kernel_mode_init(
            replacement, device, cpu, len(cpu), core, len(core), dispatcher, len(dispatcher), ctypes.byref(config), 72
        )
        == 0
    )
    current = ctypes.c_uint64()
    # Retention requires a prepared callable and the launch-ready phase.
    assert lib.simpler_kernel_mode_retain_graph(replacement, ctypes.byref(current)) == -1003
    assert current.value == 0
    chip = _build_eager_callable("a2a3", "host_build_graph")
    assert lib.simpler_kernel_mode_prepare_callable(replacement, 0, chip.buffer_ptr(), chip.buffer_size()) == 0
    assert lib.simpler_kernel_mode_retain_graph(replacement, ctypes.byref(current)) == 0
    assert current.value not in (first.value, second.value)
    assert lib.simpler_kernel_mode_release_graph(replacement, first.value) == -1004
    assert lib.finalize_device(replacement) == -1003
    assert lib.simpler_kernel_mode_release_graph(replacement, current.value) == 0
    assert lib.simpler_kernel_mode_release_graph(replacement, current.value) == -1004
    assert lib.finalize_device(replacement) == 0
    lib.destroy_device_context(replacement)


def _graph_program(lib, device, config, binaries):
    from simpler.task_interface import ChipStorageTaskArgs  # noqa: PLC0415

    ctx = lib.create_device_context()
    cpu, core, dispatcher = binaries
    assert (
        lib.simpler_init(
            ctx, device, cpu, len(cpu), core, len(core), dispatcher, len(dispatcher), ctypes.byref(config), 0, None, 0
        )
        == 0
    )
    assert lib.simpler_kernel_mode_supported(ctx) == 1
    token = ctypes.c_uint64(999)
    committed = lib.committed_device_memory_ctx(ctx)
    assert lib.simpler_kernel_mode_retain_graph(ctx, ctypes.byref(token)) == -1003
    assert token.value == 0
    assert lib.simpler_kernel_mode_release_graph(ctx, 1) == -1003
    args = ChipStorageTaskArgs()
    assert lib.simpler_kernel_mode_launch(ctx, 0, args.__ptr__(), 1) == -1003
    assert lib.committed_device_memory_ctx(ctx) == committed
    assert lib.finalize_device(ctx) == 0
    lib.destroy_device_context(ctx)


def _assert_retention_refused(lib, ctx):
    token = ctypes.c_uint64(999)
    assert lib.simpler_kernel_mode_retain_graph(ctx, ctypes.byref(token)) == -1003
    assert token.value == 0


def _lifecycle_only(lib, ctx, stream, device, config, binaries, probe, callable_id, scenario):
    from simpler.task_interface import ChipStorageTaskArgs  # noqa: PLC0415

    if scenario == "graph_contract":
        _graph_contract(lib, ctx, config, binaries, device, probe)
        assert lib.aclrtDestroyStream(stream) == 0
        assert lib.aclrtResetDevice(device) == 0
        assert lib.aclFinalize() == 0
        return True
    if scenario == "close_retry":
        probe.hbg_fail_free_once()
        assert lib.finalize_device(ctx) == -4327
        assert lib.committed_device_memory_ctx(ctx) > 0
        _assert_retention_refused(lib, ctx)
        assert lib.simpler_kernel_mode_launch(ctx, callable_id, ChipStorageTaskArgs().__ptr__(), stream) == -1003
        _close(lib, ctx, stream, device, [])
        return True
    return False


def _prepare_or_check_failure(lib, ctx, stream, device, config, binaries, probe, chip, scenario):
    aicpu, aicore, dispatcher = binaries
    if scenario == "slot_enqueue_failure":
        probe.hbg_fail_slot_publish()
    rc = lib.simpler_kernel_mode_init(
        ctx, device, aicpu, len(aicpu), aicore, len(aicore), dispatcher, len(dispatcher), ctypes.byref(config), 71
    )
    if scenario == "slot_enqueue_failure":
        assert rc == -4329
        _close(lib, ctx, stream, device, [])
        return None
    assert rc == 0, f"init: {rc}"
    if scenario == "binary_identity":
        probe.hbg_slot_binary_id.restype = ctypes.c_uint64
        probe.hbg_expected_binary_id.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        probe.hbg_expected_binary_id.restype = ctypes.c_uint64
        assert probe.hbg_slot_binary_id() == probe.hbg_expected_binary_id(aicpu, len(aicpu))
        _close(lib, ctx, stream, device, [])
        return None
    assert lib.simpler_kernel_mode_supported(ctx) == 1
    cid = ctypes.c_int32(-1)
    if scenario in ("register_enqueue_failure", "register_sync_failure"):
        if scenario == "register_enqueue_failure":
            probe.hbg_fail_cpu_launch(1)
        else:
            probe.hbg_fail_owned_sync()
        rc = lib.simpler_register_callable_v2(ctx, chip.buffer_ptr(), chip.buffer_size(), ctypes.byref(cid))
        assert rc == (-4329 if scenario == "register_enqueue_failure" else -4328), f"injected register: {rc}"
        assert cid.value == -1
        _close(lib, ctx, stream, device, [])
        return None
    rc = lib.simpler_register_callable_v2(ctx, chip.buffer_ptr(), chip.buffer_size(), ctypes.byref(cid))
    assert rc == 0, f"register: {rc}"
    return cid


def _arm_launch_failure(probe, scenario):
    if scenario == "enqueue_failure":
        probe.hbg_fail_record(2)
    if scenario == "launch_exception":
        probe.hbg_throw_cpu_launch()


def _run(device, scenario, profile="default"):
    from simpler.task_interface import ChipStorageTaskArgs, ChipTensor, DataType  # noqa: PLC0415

    arch, runtime = "a2a3", "host_build_graph"
    probe = ctypes.CDLL(None)
    probe.hbg_guard_hot_path.argtypes = [ctypes.c_int]
    chip = _build_eager_callable(
        arch,
        runtime,
        intermediate=scenario in ("queued", "capacity"),
        graph=scenario in ("graph_eager", "graph_capture"),
    )
    capture = scenario.startswith("capture") or scenario == "graph_capture"
    scalar_multiplier = 2 if scenario in ("queued", "graph_eager", "graph_capture") else 1
    lib = _load(arch, "onboard", runtime)
    signatures = {
        "aclInit": [ctypes.c_char_p],
        "aclFinalize": [],
        "aclrtSetDevice": [ctypes.c_int],
        "aclrtResetDevice": [ctypes.c_int],
        "aclrtCreateStream": [ctypes.POINTER(ctypes.c_void_p)],
        "aclrtDestroyStream": [ctypes.c_void_p],
        "aclrtSynchronizeStreamWithTimeout": [ctypes.c_void_p, ctypes.c_int32],
        "aclrtMalloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int],
        "aclrtFree": [ctypes.c_void_p],
        "aclrtMemcpy": [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
        "aclmdlRICaptureBegin": [ctypes.c_void_p, ctypes.c_int],
        "aclmdlRICaptureEnd": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
        "aclmdlRIExecuteAsync": [ctypes.c_void_p, ctypes.c_void_p],
        "aclmdlRIDestroy": [ctypes.c_void_p],
        "simpler_unregister_callable": [ctypes.c_void_p, ctypes.c_int32],
        "simpler_kernel_mode_retain_graph": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)],
        "simpler_kernel_mode_release_graph": [ctypes.c_void_p, ctypes.c_uint64],
    }
    for name, types in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = types, ctypes.c_int
    assert lib.aclInit(None) == 0
    assert lib.aclrtSetDevice(device) == 0
    aicpu, aicore, dispatcher = _binaries(arch, runtime)
    config = CallConfig()
    for i in range(4):
        config.runtime_env[i] = 1 if scenario == "capacity" else (8 if profile == "compact" else 64)
        config.runtime_env[4 + i] = 1 << 20
        config.runtime_env[8 + i] = 1024
    if scenario == "graph_program":
        _graph_program(lib, device, config, (aicpu, aicore, dispatcher))
        assert lib.aclFinalize() == 0
        return
    stream = ctypes.c_void_p()
    assert lib.aclrtCreateStream(ctypes.byref(stream)) == 0
    ctx = lib.create_device_context()
    cid = _prepare_or_check_failure(
        lib, ctx, stream, device, config, (aicpu, aicore, dispatcher), probe, chip, scenario
    )
    if cid is None:
        return
    committed = lib.committed_device_memory_ctx(ctx)
    assert lib.simpler_kernel_mode_launch(ctx, cid.value, ChipStorageTaskArgs().__ptr__(), stream) == -1004
    assert lib.committed_device_memory_ctx(ctx) == committed
    if _lifecycle_only(lib, ctx, stream, device, config, (aicpu, aicore, dispatcher), probe, cid.value, scenario):
        return
    count = 128 * 128
    array = ctypes.c_float * count
    nbytes = ctypes.sizeof(array)
    addresses, results = [], []
    rounds = (6 if profile == "compact" else 8) if scenario == "queued" else 2
    for index in range(rounds):
        source, destination = ctypes.c_void_p(), ctypes.c_void_p()
        for address in (source, destination):
            assert lib.aclrtMalloc(ctypes.byref(address), nbytes, 0) == 0
            addresses.append(address)
        values = array(*[float(i % 127 + index * 257) for i in range(count)])
        output = array(*([-999.0] * count))
        assert lib.aclrtMemcpy(source, nbytes, values, nbytes, 1) == 0
        assert lib.aclrtMemcpy(destination, nbytes, output, nbytes, 1) == 0
        scalar = (0.0, 2.5)[index % 2] if profile == "compact" else (1.25, -3.5)[index % 2]
        args = ChipStorageTaskArgs()
        args.add_tensor(ChipTensor.make(source.value, (count,), DataType.FLOAT32, child_memory=True))
        args.add_tensor(ChipTensor.make(destination.value, (count,), DataType.FLOAT32, child_memory=True))
        args.add_scalar(int.from_bytes(struct.pack("<f", scalar), "little"))
        graph, lease = ctypes.c_void_p(), ctypes.c_uint64()
        if capture:
            assert lib.simpler_kernel_mode_retain_graph(ctx, ctypes.byref(lease)) == 0
            assert lib.aclmdlRICaptureBegin(stream, 0) == 0
        _arm_launch_failure(probe, scenario)
        rc = _launch(lib, probe, ctx, cid.value, args, stream)
        if scenario == "capacity" and index == 0:
            assert rc < 0
            assert lib.committed_device_memory_ctx(ctx) == committed
            assert lib.simpler_unregister_callable(ctx, cid.value) == 0
            chip = _build_eager_callable(arch, runtime)
            assert lib.simpler_kernel_mode_prepare_callable(ctx, cid.value, chip.buffer_ptr(), chip.buffer_size()) == 0
            rc = _launch(lib, probe, ctx, cid.value, args, stream)
        if scenario in ("enqueue_failure", "launch_exception"):
            assert rc == (-4326 if scenario == "enqueue_failure" else -1000)
            _assert_retention_refused(lib, ctx)
            assert lib.simpler_kernel_mode_launch(ctx, cid.value, args.__ptr__(), stream) == -1003
            break
        assert rc == 0, f"launch {index}: {rc}"
        args.clear()
        if capture:
            assert lib.aclmdlRICaptureEnd(stream, ctypes.byref(graph)) == 0
            assert lib.simpler_unregister_callable(ctx, cid.value) == -1003
            assert lib.finalize_device(ctx) == -1003
            assert lib.committed_device_memory_ctx(ctx) == committed
            for _ in range(0 if scenario == "capture_unused" else 3):
                assert lib.aclmdlRIExecuteAsync(graph, stream) == 0
                assert lib.aclrtSynchronizeStreamWithTimeout(stream, 60000) == 0
                assert lib.aclrtMemcpy(output, nbytes, destination, nbytes, 2) == 0
                assert list(output) == [v + scalar * scalar_multiplier for v in values]
            assert lib.aclmdlRIDestroy(graph) == 0
            assert lib.simpler_kernel_mode_release_graph(ctx, lease.value) == 0
            assert lib.simpler_kernel_mode_release_graph(ctx, lease.value) == -1004
        if scenario != "queued":
            assert lib.aclrtSynchronizeStreamWithTimeout(stream, 60000) == 0
        assert lib.committed_device_memory_ctx(ctx) == committed
        expected = (
            [-999.0] * count if scenario == "capture_unused" else [v + scalar * scalar_multiplier for v in values]
        )
        results.append((destination, expected))
    assert lib.aclrtSynchronizeStreamWithTimeout(stream, 60000) == 0
    for destination, expected in results:
        output = array()
        assert lib.aclrtMemcpy(output, nbytes, destination, nbytes, 2) == 0
        assert list(output) == expected
    if scenario not in ("enqueue_failure", "launch_exception"):
        assert lib.simpler_unregister_callable(ctx, cid.value) == 0
    assert lib.simpler_kernel_mode_launch(ctx, cid.value, ChipStorageTaskArgs().__ptr__(), stream) == -1003
    _close(lib, ctx, stream, device, addresses)
    print(f"HBG_{scenario.upper()}_PASS", flush=True)


if __name__ == "__main__":
    _run(int(sys.argv[1]), sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "default")
