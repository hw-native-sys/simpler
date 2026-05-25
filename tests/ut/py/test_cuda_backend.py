# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CUDA backend bring-up tests."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys

import pytest

from simpler_setup.platform_info import parse_platform, to_platform
from simpler_setup.runtime_builder import RuntimeBuilder


def test_cuda_platform_maps_to_onboard_variant():
    assert parse_platform("cuda") == ("cuda", "onboard")
    assert to_platform("cuda", "onboard") == "cuda"


def test_cuda_runtime_builder_discovers_host_schedule():
    builder = RuntimeBuilder(platform="cuda")

    assert "host_schedule" in builder.list_runtimes()


class CudaHostCallable(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("op", ctypes.c_uint32),
        ("image", ctypes.c_void_p),
        ("image_size", ctypes.c_size_t),
        ("entry_name", ctypes.c_char_p),
        ("grid_dim", ctypes.c_uint32),
        ("block_dim", ctypes.c_uint32),
        ("shared_mem_bytes", ctypes.c_size_t),
    ]


class CudaVectorAddArgs(ctypes.Structure):
    _fields_ = [
        ("a", ctypes.c_void_p),
        ("b", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("n", ctypes.c_uint64),
    ]


class PtoRunTiming(ctypes.Structure):
    _fields_ = [
        ("host_wall_ns", ctypes.c_uint64),
        ("device_wall_ns", ctypes.c_uint64),
    ]


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required for CUDA runtime smoke test")
def test_cuda_host_schedule_runs_vector_add_with_real_device_data(tmp_path):
    kernel_src = tmp_path / "vector_add.cu"
    kernel_src.write_text(
        """
extern "C" __global__ void pto_vector_add_f32(
    const float *a, const float *b, float *out, unsigned long long n) {
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
""".lstrip()
    )
    ptx_path = tmp_path / "vector_add.ptx"
    subprocess.run(
        ["nvcc", "--ptx", "-std=c++17", "-arch=compute_80", str(kernel_src), "-o", str(ptx_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    ptx = ptx_path.read_bytes()
    ptx_buf = ctypes.create_string_buffer(ptx + b"\0")

    binaries = RuntimeBuilder(platform="cuda").get_binaries("host_schedule", build=True)
    runtime = ctypes.CDLL(str(binaries.host_path))

    runtime.create_device_context.restype = ctypes.c_void_p
    runtime.destroy_device_context.argtypes = [ctypes.c_void_p]
    runtime.simpler_init.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    runtime.simpler_init.restype = ctypes.c_int
    runtime.finalize_device.argtypes = [ctypes.c_void_p]
    runtime.finalize_device.restype = ctypes.c_int
    runtime.device_malloc_ctx.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    runtime.device_malloc_ctx.restype = ctypes.c_void_p
    runtime.device_free_ctx.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    runtime.copy_to_device_ctx.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    runtime.copy_to_device_ctx.restype = ctypes.c_int
    runtime.copy_from_device_ctx.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    runtime.copy_from_device_ctx.restype = ctypes.c_int
    runtime.prepare_callable.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
    runtime.prepare_callable.restype = ctypes.c_int
    runtime.run_prepared.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(PtoRunTiming),
    ]
    runtime.run_prepared.restype = ctypes.c_int
    runtime.unregister_callable.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    runtime.unregister_callable.restype = ctypes.c_int

    ctx = runtime.create_device_context()
    assert ctx
    try:
        assert runtime.simpler_init(ctx, 0, None, 0, None, 0) == 0

        n = 1024
        array_t = ctypes.c_float * n
        host_a = array_t(*[float(i) for i in range(n)])
        host_b = array_t(*[float(2 * i) for i in range(n)])
        host_out = array_t()
        nbytes = ctypes.sizeof(host_a)

        dev_a = runtime.device_malloc_ctx(ctx, nbytes)
        dev_b = runtime.device_malloc_ctx(ctx, nbytes)
        dev_out = runtime.device_malloc_ctx(ctx, nbytes)
        assert dev_a and dev_b and dev_out
        try:
            assert runtime.copy_to_device_ctx(ctx, dev_a, ctypes.byref(host_a), nbytes) == 0
            assert runtime.copy_to_device_ctx(ctx, dev_b, ctypes.byref(host_b), nbytes) == 0

            callable_manifest = CudaHostCallable(
                version=1,
                op=1,
                image=ctypes.cast(ptx_buf, ctypes.c_void_p),
                image_size=len(ptx) + 1,
                entry_name=b"pto_vector_add_f32",
                grid_dim=(n + 255) // 256,
                block_dim=256,
                shared_mem_bytes=0,
            )
            args = CudaVectorAddArgs(a=dev_a, b=dev_b, out=dev_out, n=n)
            timing = PtoRunTiming()

            assert runtime.prepare_callable(ctx, 0, ctypes.byref(callable_manifest)) == 0
            assert (
                runtime.run_prepared(
                    ctx,
                    None,
                    0,
                    ctypes.byref(args),
                    256,
                    0,
                    0,
                    0,
                    0,
                    0,
                    None,
                    ctypes.byref(timing),
                )
                == 0
            )
            assert timing.host_wall_ns > 0
            assert timing.device_wall_ns > 0
            assert runtime.copy_from_device_ctx(ctx, ctypes.byref(host_out), dev_out, nbytes) == 0

            assert list(host_out) == [float(3 * i) for i in range(n)]
            assert runtime.unregister_callable(ctx, 0) == 0
        finally:
            runtime.device_free_ctx(ctx, dev_a)
            runtime.device_free_ctx(ctx, dev_b)
            runtime.device_free_ctx(ctx, dev_out)
    finally:
        runtime.finalize_device(ctx)
        runtime.destroy_device_context(ctx)


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required for CUDA runtime smoke test")
def test_cuda_standalone_smoke_can_run_twice_in_one_process():
    script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".agents/skills/cuda-backend-eval/scripts").resolve()))
from cuda_smoke import run_smoke

for _ in range(2):
    run_smoke(device=0, n=1024, block_dim=256, arch="compute_80", build=False)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
