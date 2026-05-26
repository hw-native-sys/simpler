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
import json
import subprocess
import sys
import threading
import time

import pytest

from simpler_setup.cuda_callable_compiler import CudaVectorAddArgs
from simpler_setup.cuda_preflight import cuda_skip_reason
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.platform_info import parse_platform, to_platform
from simpler_setup.runtime_builder import RuntimeBuilder

_CUDA_SKIP_REASON = cuda_skip_reason(require_nvcc=True)
requires_cuda = pytest.mark.skipif(_CUDA_SKIP_REASON is not None, reason=_CUDA_SKIP_REASON or "")


def test_cuda_platform_maps_to_onboard_variant():
    assert parse_platform("cuda") == ("cuda", "onboard")
    assert to_platform("cuda", "onboard") == "cuda"


def test_cuda_runtime_builder_discovers_host_schedule():
    builder = RuntimeBuilder(platform="cuda")

    assert "host_schedule" in builder.list_runtimes()


def test_cuda_runtime_builder_discovers_persistent_device():
    builder = RuntimeBuilder(platform="cuda")

    assert "persistent_device" in builder.list_runtimes()


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


class CudaHostCallableV2(ctypes.Structure):
    _fields_ = CudaHostCallable._fields_ + [
        ("stream_id", ctypes.c_uint32),
    ]


class PtoRunTiming(ctypes.Structure):
    _fields_ = [
        ("host_wall_ns", ctypes.c_uint64),
        ("device_wall_ns", ctypes.c_uint64),
    ]


@pytest.fixture(scope="module")
def cuda_host_runtime_binaries():
    return RuntimeBuilder(platform="cuda").get_binaries("host_schedule", build=True)


@requires_cuda
def test_cuda_host_schedule_worker_run_accepts_raw_cuda_args():
    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/cuda_smoke.py",
            "--runner",
            "worker",
            "--device",
            "0",
            "--n",
            "1024",
            "--block-dim",
            "256",
            "--arch",
            "compute_80",
            "--no-build",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    payload = json.loads(result.stdout)
    assert payload["status"] == "pass"
    assert payload["runner"] == "worker"
    assert payload["runtime"] == "host_schedule"
    assert payload["mode"] == "worker/add"
    assert payload["ptx_arch"] == "compute_80"
    assert payload["ptx_source"] == "kernel-compiler-worker-task-body-compute_80"
    assert payload["host_wall_ns"] > 0
    assert payload["device_wall_ns"] > 0


@requires_cuda
def test_cuda_host_schedule_worker_run_accepts_multiply_task_body():
    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/cuda_smoke.py",
            "--runner",
            "worker",
            "--op",
            "mul",
            "--device",
            "0",
            "--n",
            "1024",
            "--block-dim",
            "256",
            "--arch",
            "compute_80",
            "--no-build",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    payload = json.loads(result.stdout)
    assert payload["status"] == "pass"
    assert payload["runner"] == "worker"
    assert payload["runtime"] == "host_schedule"
    assert payload["mode"] == "worker/mul"
    assert payload["op"] == "mul"
    assert payload["ptx_source"] == "kernel-compiler-worker-task-body-mul-compute_80"
    assert payload["host_wall_ns"] > 0
    assert payload["device_wall_ns"] > 0


@requires_cuda
def test_cuda_host_schedule_worker_run_accepts_affine_task_body():
    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/cuda_smoke.py",
            "--runner",
            "worker",
            "--op",
            "affine",
            "--device",
            "0",
            "--n",
            "1024",
            "--block-dim",
            "256",
            "--arch",
            "compute_80",
            "--no-build",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    payload = json.loads(result.stdout)
    assert payload["status"] == "pass"
    assert payload["runner"] == "worker"
    assert payload["runtime"] == "host_schedule"
    assert payload["mode"] == "worker/affine"
    assert payload["op"] == "affine"
    assert payload["ptx_source"] == "kernel-compiler-worker-task-body-affine-compute_80"
    assert payload["host_wall_ns"] > 0
    assert payload["device_wall_ns"] > 0


@requires_cuda
def test_cuda_host_schedule_worker_run_accepts_triad_task_body():
    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/cuda_smoke.py",
            "--runner",
            "worker",
            "--op",
            "triad",
            "--device",
            "0",
            "--n",
            "1024",
            "--block-dim",
            "256",
            "--arch",
            "compute_80",
            "--no-build",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    payload = json.loads(result.stdout)
    assert payload["status"] == "pass"
    assert payload["runner"] == "worker"
    assert payload["runtime"] == "host_schedule"
    assert payload["mode"] == "worker/triad"
    assert payload["op"] == "triad"
    assert payload["ptx_source"] == "kernel-compiler-worker-task-body-triad-compute_80"
    assert payload["host_wall_ns"] > 0
    assert payload["device_wall_ns"] > 0


@requires_cuda
def test_cuda_host_schedule_runs_kernel_compiler_task_body_with_real_device_data(tmp_path, cuda_host_runtime_binaries):
    task_src = tmp_path / "vector_add.pto.cu"
    task_src.write_text(
        """
unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < ctx->n) {
    ctx->out[i] = ctx->a[i] + ctx->b[i];
}
""".lstrip()
    )

    artifact = KernelCompiler(platform="cuda").compile_cuda_host_schedule(
        str(task_src),
        task_name="vector_add",
        arch="compute_80",
        cache_root=tmp_path / "cache",
        context_definition="""
struct PtoTaskContext {
    const float *a;
    const float *b;
    float *out;
    unsigned long long n;
};
""".strip(),
        host_parameters=(
            "const float *a",
            "const float *b",
            "float *out",
            "unsigned long long n",
        ),
        host_context_initializer="a, b, out, n",
    )
    ptx_buf = ctypes.create_string_buffer(artifact.ptx + b"\0")

    runtime = ctypes.CDLL(str(cuda_host_runtime_binaries.host_path))

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
                image_size=len(artifact.ptx) + 1,
                entry_name=artifact.entry_name.encode("utf-8"),
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


@requires_cuda
def test_cuda_host_schedule_runs_vector_add_with_real_device_data(tmp_path, cuda_host_runtime_binaries):
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

    runtime = ctypes.CDLL(str(cuda_host_runtime_binaries.host_path))

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


@requires_cuda
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


@requires_cuda
def test_cuda_persistent_device_smoke_runs_vector_add_tasks():
    script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".agents/skills/cuda-backend-eval/scripts").resolve()))
from cuda_persistent_smoke import run_persistent_smoke

result = run_persistent_smoke(device=0, task_count=2, n=1024, arch="compute_80")
assert result["status"] == "pass"
assert result["runtime"] == "persistent_device"
assert result["task_count"] == 2
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@requires_cuda
def test_cuda_persistent_device_smoke_runs_multi_block_vector_add_tasks():
    script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".agents/skills/cuda-backend-eval/scripts").resolve()))
from cuda_persistent_smoke import run_persistent_smoke

result = run_persistent_smoke(
    device=0,
    task_count=2,
    n=4096,
    arch="compute_80",
    mode="direct",
    worker_blocks_per_task=4,
)
assert result["status"] == "pass"
assert result["runtime"] == "persistent_device"
assert result["task_count"] == 2
assert result["worker_blocks_per_task"] == 4
assert result["worker_blocks"] == 8
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@requires_cuda
def test_cuda_persistent_device_smoke_runs_scheduler_worker_queue():
    script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".agents/skills/cuda-backend-eval/scripts").resolve()))
from cuda_persistent_smoke import run_persistent_smoke

result = run_persistent_smoke(device=0, task_count=4, n=1024, arch="compute_80", mode="queue")
assert result["status"] == "pass"
assert result["runtime"] == "persistent_device"
assert result["mode"] == "queue"
assert result["scheduler_blocks"] == 1
assert result["worker_blocks"] >= 1
assert result["completed_count"] == 4
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@requires_cuda
def test_cuda_persistent_device_smoke_runs_queue_with_explicit_resource_policy():
    script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".agents/skills/cuda-backend-eval/scripts").resolve()))
from cuda_persistent_smoke import run_persistent_smoke

result = run_persistent_smoke(
    device=0,
    task_count=6,
    n=1024,
    arch="compute_80",
    mode="queue",
    queue_capacity=2,
    worker_blocks=2,
    stream_id=1,
)
assert result["status"] == "pass"
assert result["runtime"] == "persistent_device"
assert result["mode"] == "queue"
assert result["queue_capacity"] == 2
assert result["scheduler_blocks"] == 1
assert result["worker_blocks"] == 2
assert result["stream_id"] == 1
assert result["resource_policy"] == {
    "scheduler_blocks": 1,
    "worker_blocks": 2,
    "worker_blocks_per_task": 1,
    "stream_id": 1,
    "block_dim": 256,
    "grid_dim": 3,
}
assert result["completed_count"] == 6
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@requires_cuda
def test_cuda_persistent_device_smoke_runs_bounded_ring_queue():
    script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".agents/skills/cuda-backend-eval/scripts").resolve()))
from cuda_persistent_smoke import run_persistent_smoke

result = run_persistent_smoke(
    device=0,
    task_count=6,
    n=1024,
    arch="compute_80",
    mode="queue",
    queue_capacity=2,
)
assert result["status"] == "pass"
assert result["runtime"] == "persistent_device"
assert result["mode"] == "queue"
assert result["queue_capacity"] == 2
assert result["completed_count"] == 6
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@requires_cuda
def test_cuda_persistent_device_smoke_runs_dispatch_dag():
    script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".agents/skills/cuda-backend-eval/scripts").resolve()))
from cuda_persistent_smoke import run_persistent_smoke

result = run_persistent_smoke(
    device=0,
    task_count=3,
    n=4096,
    arch="compute_80",
    mode="dag",
    queue_capacity=2,
)
assert result["status"] == "pass"
assert result["runtime"] == "persistent_device"
assert result["mode"] == "dag"
assert result["queue_capacity"] == 2
assert result["completed_count"] == 3
assert result["device_scheduler_errors"] == {"count": 0, "code": 0, "task_id": 0}
assert result["dispatch_func_ids"] == [1, 2, 1]
assert result["fanin_remaining"] == [0, 0, 0]
assert result["source_kind"] == "generated-dispatch"
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@requires_cuda
def test_cuda_persistent_device_smoke_runs_dag_with_explicit_worker_blocks():
    script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".agents/skills/cuda-backend-eval/scripts").resolve()))
from cuda_persistent_smoke import run_persistent_smoke

result = run_persistent_smoke(
    device=0,
    task_count=5,
    n=4096,
    arch="compute_80",
    mode="dag",
    queue_capacity=2,
    dag_shape="chain",
    worker_blocks=2,
    stream_id=1,
)
assert result["status"] == "pass"
assert result["runtime"] == "persistent_device"
assert result["mode"] == "dag"
assert result["dag_shape"] == "chain"
assert result["scheduler_blocks"] == 1
assert result["worker_blocks"] == 2
assert result["stream_id"] == 1
assert result["resource_policy"] == {
    "scheduler_blocks": 1,
    "worker_blocks": 2,
    "worker_blocks_per_task": 1,
    "stream_id": 1,
    "block_dim": 256,
    "grid_dim": 3,
}
assert result["completed_count"] == 5
assert result["dispatch_func_ids"] == [1, 2, 1, 2, 1]
assert result["fanin_remaining"] == [0, 0, 0, 0, 0]
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@requires_cuda
def test_cuda_persistent_device_smoke_reports_device_scheduler_errors():
    script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".agents/skills/cuda-backend-eval/scripts").resolve()))
from cuda_persistent_smoke import run_persistent_smoke

try:
    run_persistent_smoke(
        device=0,
        task_count=1,
        n=1024,
        arch="compute_80",
        mode="dag",
        queue_capacity=1,
        dag_shape="bad_func_id",
    )
except RuntimeError as exc:
    message = str(exc)
    assert "persistent dag scheduler error" in message
    assert "code=1" in message
    assert "task_id=0" in message
else:
    raise AssertionError("expected persistent DAG scheduler error")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@requires_cuda
def test_cuda_persistent_device_smoke_reports_bad_dependent_errors():
    script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".agents/skills/cuda-backend-eval/scripts").resolve()))
from cuda_persistent_smoke import run_persistent_smoke

try:
    run_persistent_smoke(
        device=0,
        task_count=1,
        n=1024,
        arch="compute_80",
        mode="dag",
        queue_capacity=1,
        dag_shape="bad_dependent",
    )
except RuntimeError as exc:
    message = str(exc)
    assert "persistent dag scheduler error" in message
    assert "code=2" in message
    assert "task_id=7" in message
else:
    raise AssertionError("expected persistent DAG dependent-id scheduler error")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@requires_cuda
def test_cuda_persistent_device_smoke_reports_bad_dependent_range_errors():
    script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".agents/skills/cuda-backend-eval/scripts").resolve()))
from cuda_persistent_smoke import run_persistent_smoke

try:
    run_persistent_smoke(
        device=0,
        task_count=1,
        n=1024,
        arch="compute_80",
        mode="dag",
        queue_capacity=1,
        dag_shape="bad_dependent_range",
    )
except RuntimeError as exc:
    message = str(exc)
    assert "persistent dag scheduler error" in message
    assert "code=3" in message
    assert "task_id=0" in message
else:
    raise AssertionError("expected persistent DAG dependent-range scheduler error")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@requires_cuda
def test_cuda_persistent_device_smoke_reports_bad_fanin_underflow_errors():
    script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".agents/skills/cuda-backend-eval/scripts").resolve()))
from cuda_persistent_smoke import run_persistent_smoke

try:
    run_persistent_smoke(
        device=0,
        task_count=3,
        n=1024,
        arch="compute_80",
        mode="dag",
        queue_capacity=2,
        dag_shape="bad_fanin_underflow",
    )
except RuntimeError as exc:
    message = str(exc)
    assert "persistent dag scheduler error" in message
    assert "code=4" in message
    assert "task_id=2" in message
else:
    raise AssertionError("expected persistent DAG fanin-underflow scheduler error")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@requires_cuda
def test_cuda_persistent_device_smoke_reports_initial_fanin_mismatch_errors():
    script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".agents/skills/cuda-backend-eval/scripts").resolve()))
from cuda_persistent_smoke import run_persistent_smoke

try:
    run_persistent_smoke(
        device=0,
        task_count=1,
        n=1024,
        arch="compute_80",
        mode="dag",
        queue_capacity=1,
        dag_shape="bad_initial_fanin",
    )
except RuntimeError as exc:
    message = str(exc)
    assert "persistent dag scheduler error" in message
    assert "code=5" in message
    assert "task_id=0" in message
else:
    raise AssertionError("expected persistent DAG initial-fanin scheduler error")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@requires_cuda
def test_cuda_persistent_device_smoke_reports_no_root_dag_errors():
    script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".agents/skills/cuda-backend-eval/scripts").resolve()))
from cuda_persistent_smoke import run_persistent_smoke

try:
    run_persistent_smoke(
        device=0,
        task_count=1,
        n=1024,
        arch="compute_80",
        mode="dag",
        queue_capacity=1,
        dag_shape="bad_no_root",
    )
except RuntimeError as exc:
    message = str(exc)
    assert "persistent dag scheduler error" in message
    assert "code=6" in message
    assert "task_id=0" in message
else:
    raise AssertionError("expected persistent DAG no-root scheduler error")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@requires_cuda
def test_cuda_persistent_device_smoke_runs_dispatch_dag_chain():
    script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".agents/skills/cuda-backend-eval/scripts").resolve()))
from cuda_persistent_smoke import run_persistent_smoke

result = run_persistent_smoke(
    device=0,
    task_count=5,
    n=4096,
    arch="compute_80",
    mode="dag",
    queue_capacity=2,
    dag_shape="chain",
)
assert result["status"] == "pass"
assert result["runtime"] == "persistent_device"
assert result["mode"] == "dag"
assert result["dag_shape"] == "chain"
assert result["task_count"] == 5
assert result["queue_capacity"] == 2
assert result["completed_count"] == 5
assert result["dispatch_func_ids"] == [1, 2, 1, 2, 1]
assert result["fanin_remaining"] == [0, 0, 0, 0, 0]
assert result["source_kind"] == "generated-dispatch"
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@requires_cuda
def test_cuda_persistent_device_smoke_reuses_prepared_dag_callable():
    script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".agents/skills/cuda-backend-eval/scripts").resolve()))
from cuda_persistent_smoke import run_persistent_smoke

result = run_persistent_smoke(
    device=0,
    task_count=5,
    n=4096,
    arch="compute_80",
    mode="dag",
    queue_capacity=2,
    dag_shape="chain",
    repeat_runs=2,
)
assert result["status"] == "pass"
assert result["runtime"] == "persistent_device"
assert result["mode"] == "dag"
assert result["dag_shape"] == "chain"
assert result["repeat_runs"] == 2
assert result["completed_count"] == 5
assert result["launch_completed_counts"] == [5, 5]
assert len(result["launch_device_wall_ns"]) == 2
assert result["device_scheduler_errors"] == {"count": 0, "code": 0, "task_id": 0}
assert result["fanin_remaining"] == [0, 0, 0, 0, 0]
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@requires_cuda
def test_cuda_persistent_device_smoke_runs_dispatch_dag_with_scratch_reuse():
    script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".agents/skills/cuda-backend-eval/scripts").resolve()))
from cuda_persistent_smoke import run_persistent_smoke

result = run_persistent_smoke(
    device=0,
    task_count=6,
    n=4096,
    arch="compute_80",
    mode="dag",
    queue_capacity=2,
    dag_shape="scratch_reuse",
)
assert result["status"] == "pass"
assert result["runtime"] == "persistent_device"
assert result["mode"] == "dag"
assert result["dag_shape"] == "scratch_reuse"
assert result["task_count"] == 6
assert result["queue_capacity"] == 2
assert result["completed_count"] == 6
assert result["dispatch_func_ids"] == [1, 2, 1, 2, 1, 1]
assert result["fanin_remaining"] == [0, 0, 0, 0, 0, 0]
assert result["source_kind"] == "generated-dispatch"
assert result["scratch_reuse"] == {"reused_buffer": "tmp0", "reuse_task": 4}
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@requires_cuda
def test_cuda_persistent_device_smoke_runs_dispatch_dag_tensor_tile():
    script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".agents/skills/cuda-backend-eval/scripts").resolve()))
from cuda_persistent_smoke import run_persistent_smoke

result = run_persistent_smoke(
    device=0,
    task_count=4,
    n=4096,
    arch="compute_80",
    mode="dag",
    queue_capacity=2,
    dag_shape="tensor_tile",
)
assert result["status"] == "pass"
assert result["runtime"] == "persistent_device"
assert result["mode"] == "dag"
assert result["dag_shape"] == "tensor_tile"
assert result["task_count"] == 4
assert result["queue_capacity"] == 2
assert result["completed_count"] == 4
assert result["dispatch_func_ids"] == [3, 1, 2, 1]
assert result["fanin_remaining"] == [0, 0, 0, 0]
assert result["source_kind"] == "generated-dispatch"
assert result["tensor_tile"] == {
    "rows": 16,
    "cols": 16,
    "inner": 16,
    "lda": 16,
    "ldb": 16,
    "ldc": 16,
    "a_batch_stride": 256,
    "b_batch_stride": 256,
    "out_batch_stride": 256,
    "tile_count": 16,
}
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@requires_cuda
def test_cuda_persistent_device_smoke_runs_dispatch_dag_scalar_axpy():
    script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".agents/skills/cuda-backend-eval/scripts").resolve()))
from cuda_persistent_smoke import run_persistent_smoke

result = run_persistent_smoke(
    device=0,
    task_count=3,
    n=4096,
    arch="compute_80",
    mode="dag",
    queue_capacity=2,
    dag_shape="scalar_axpy",
)
assert result["status"] == "pass"
assert result["runtime"] == "persistent_device"
assert result["mode"] == "dag"
assert result["dag_shape"] == "scalar_axpy"
assert result["task_count"] == 3
assert result["queue_capacity"] == 2
assert result["completed_count"] == 3
assert result["dispatch_func_ids"] == [4, 2, 1]
assert result["fanin_remaining"] == [0, 0, 0]
assert result["source_kind"] == "generated-dispatch"
assert result["scalar_args"] == {"scalar0": 1.5}
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@requires_cuda
def test_cuda_persistent_device_smoke_runs_dispatch_dag_scalar_affine():
    script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".agents/skills/cuda-backend-eval/scripts").resolve()))
from cuda_persistent_smoke import run_persistent_smoke

result = run_persistent_smoke(
    device=0,
    task_count=3,
    n=4096,
    arch="compute_80",
    mode="dag",
    queue_capacity=2,
    dag_shape="scalar_affine",
)
assert result["status"] == "pass"
assert result["runtime"] == "persistent_device"
assert result["mode"] == "dag"
assert result["dag_shape"] == "scalar_affine"
assert result["task_count"] == 3
assert result["queue_capacity"] == 2
assert result["completed_count"] == 3
assert result["dispatch_func_ids"] == [5, 2, 1]
assert result["fanin_remaining"] == [0, 0, 0]
assert result["source_kind"] == "generated-dispatch"
assert result["scalar_args"] == {"scalar0": 1.5, "scalar1": 0.5}
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@requires_cuda
def test_cuda_persistent_device_smoke_runs_dispatch_dag_triad():
    script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".agents/skills/cuda-backend-eval/scripts").resolve()))
from cuda_persistent_smoke import run_persistent_smoke

result = run_persistent_smoke(
    device=0,
    task_count=3,
    n=4096,
    arch="compute_80",
    mode="dag",
    queue_capacity=2,
    dag_shape="triad",
)
assert result["status"] == "pass"
assert result["runtime"] == "persistent_device"
assert result["mode"] == "dag"
assert result["dag_shape"] == "triad"
assert result["task_count"] == 3
assert result["queue_capacity"] == 2
assert result["completed_count"] == 3
assert result["dispatch_func_ids"] == [6, 2, 1]
assert result["fanin_remaining"] == [0, 0, 0]
assert result["source_kind"] == "generated-dispatch"
assert result["tensor_args"] == {"c": "tmp0"}
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@requires_cuda
def test_cuda_host_schedule_runs_independent_callables_on_multiple_streams(tmp_path, cuda_host_runtime_binaries):
    kernel_src = tmp_path / "slow_vector_add.cu"
    kernel_src.write_text(
        """
extern "C" __global__ void pto_vector_add_f32(
    const float *a, const float *b, float *out, unsigned long long n) {
    unsigned long long start = clock64();
    while (clock64() - start < 80000000ULL) {
    }
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
""".lstrip()
    )
    ptx_path = tmp_path / "slow_vector_add.ptx"
    subprocess.run(
        ["nvcc", "--ptx", "-std=c++17", "-arch=compute_80", str(kernel_src), "-o", str(ptx_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    ptx = ptx_path.read_bytes()
    ptx_buf = ctypes.create_string_buffer(ptx + b"\0")

    runtime = ctypes.CDLL(str(cuda_host_runtime_binaries.host_path))

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
        n = 1
        array_t = ctypes.c_float * n
        host_a = array_t(1.0)
        host_b = array_t(2.0)
        host_out_0 = array_t()
        host_out_1 = array_t()
        nbytes = ctypes.sizeof(host_a)

        dev_a = runtime.device_malloc_ctx(ctx, nbytes)
        dev_b = runtime.device_malloc_ctx(ctx, nbytes)
        dev_out_0 = runtime.device_malloc_ctx(ctx, nbytes)
        dev_out_1 = runtime.device_malloc_ctx(ctx, nbytes)
        assert dev_a and dev_b and dev_out_0 and dev_out_1
        try:
            assert runtime.copy_to_device_ctx(ctx, dev_a, ctypes.byref(host_a), nbytes) == 0
            assert runtime.copy_to_device_ctx(ctx, dev_b, ctypes.byref(host_b), nbytes) == 0
            manifests = [
                CudaHostCallableV2(
                    version=2,
                    op=1,
                    image=ctypes.cast(ptx_buf, ctypes.c_void_p),
                    image_size=len(ptx) + 1,
                    entry_name=b"pto_vector_add_f32",
                    grid_dim=1,
                    block_dim=1,
                    shared_mem_bytes=0,
                    stream_id=stream_id,
                )
                for stream_id in (0, 1)
            ]
            assert runtime.prepare_callable(ctx, 0, ctypes.byref(manifests[0])) == 0
            assert runtime.prepare_callable(ctx, 1, ctypes.byref(manifests[1])) == 0

            args = [
                CudaVectorAddArgs(a=dev_a, b=dev_b, out=dev_out_0, n=n),
                CudaVectorAddArgs(a=dev_a, b=dev_b, out=dev_out_1, n=n),
            ]
            timings = [PtoRunTiming(), PtoRunTiming()]

            def run(callable_id):
                assert (
                    runtime.run_prepared(
                        ctx,
                        None,
                        callable_id,
                        ctypes.byref(args[callable_id]),
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        None,
                        ctypes.byref(timings[callable_id]),
                    )
                    == 0
                )

            serial_start = time.perf_counter()
            run(0)
            run(1)
            serial_s = time.perf_counter() - serial_start

            parallel_start = time.perf_counter()
            threads = [threading.Thread(target=run, args=(callable_id,)) for callable_id in (0, 1)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            parallel_s = time.perf_counter() - parallel_start

            assert parallel_s < serial_s * 0.85
            assert runtime.copy_from_device_ctx(ctx, ctypes.byref(host_out_0), dev_out_0, nbytes) == 0
            assert runtime.copy_from_device_ctx(ctx, ctypes.byref(host_out_1), dev_out_1, nbytes) == 0
            assert list(host_out_0) == [3.0]
            assert list(host_out_1) == [3.0]
            assert runtime.unregister_callable(ctx, 0) == 0
            assert runtime.unregister_callable(ctx, 1) == 0
        finally:
            runtime.device_free_ctx(ctx, dev_a)
            runtime.device_free_ctx(ctx, dev_b)
            runtime.device_free_ctx(ctx, dev_out_0)
            runtime.device_free_ctx(ctx, dev_out_1)
    finally:
        runtime.finalize_device(ctx)
        runtime.destroy_device_context(ctx)
