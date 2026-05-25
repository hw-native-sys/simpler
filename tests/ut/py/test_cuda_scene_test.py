# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CUDA integration coverage for the SceneTestCase L2 path."""

from __future__ import annotations

import shutil

import pytest
from simpler.task_interface import ArgDirection

from simpler_setup.cuda_callable_compiler import (
    CudaHostScheduleCallableArtifact,
    PreparedCudaCallable,
)
from simpler_setup.scene_test import (
    SceneTestCase,
    TaskArgsBuilder,
    Tensor,
    _compile_chip_callable_from_spec,
    scene_test,
)

_VECTOR_ADD_BODY = """
unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < ctx->n) {
    ctx->out[i] = ctx->a[i] + ctx->b[i];
}
""".lstrip()

_VECTOR_ADD_CONTEXT = """
struct PtoTaskContext {
    const float *a;
    const float *b;
    float *out;
    unsigned long long n;
};
""".strip()

_VECTOR_ADD_HOST_PARAMS = (
    "const float *a",
    "const float *b",
    "float *out",
    "unsigned long long n",
)


def _cuda_vector_add_spec(source, *, arch="compute_80", grid_dim=4, block_dim=256):
    return {
        "cuda": {
            "source": str(source),
            "task_name": "vector_add",
            "arch": arch,
            "context_definition": _VECTOR_ADD_CONTEXT,
            "host_parameters": _VECTOR_ADD_HOST_PARAMS,
            "host_context_initializer": "a, b, out, n",
            "grid_dim": grid_dim,
            "block_dim": block_dim,
            "signature": [ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
            "arg_builder": "vector_add_f32",
            "args": ["a", "b", "out"],
        }
    }


def test_scene_test_compiles_cuda_host_schedule_callable(tmp_path, monkeypatch):
    source = tmp_path / "vector_add.pto.cu"
    source.write_text(_VECTOR_ADD_BODY)
    seen = {}

    def fake_compile_cuda_host_schedule(self, source_path, **kwargs):
        seen["platform"] = self.platform
        seen["source_path"] = source_path
        seen["kwargs"] = kwargs
        return CudaHostScheduleCallableArtifact(
            cache_key="scene-test-key",
            cache_hit=False,
            source_path=tmp_path / "generated_host_wrapper.cu",
            ptx_path=tmp_path / "pto_callable.ptx",
            manifest_path=tmp_path / "pto_callable.json",
            ptx=b"fake-scene-ptx",
            entry_name="pto_kernel_vector_add",
            persistent_entry_name="pto_task_vector_add",
            arch=kwargs["arch"],
            source_kind="task-body-wrapper",
        )

    monkeypatch.setattr(
        "simpler_setup.kernel_compiler.KernelCompiler.compile_cuda_host_schedule",
        fake_compile_cuda_host_schedule,
    )

    prepared = _compile_chip_callable_from_spec(
        _cuda_vector_add_spec(source),
        "cuda",
        "host_schedule",
        ("cuda-scene-compile", "cuda", "host_schedule"),
    )

    assert isinstance(prepared, PreparedCudaCallable)
    assert prepared.runtime == "host_schedule"
    assert prepared.manifest.grid_dim == 4
    assert prepared.manifest.block_dim == 256
    assert seen["platform"] == "cuda"
    assert seen["source_path"] == str(source)
    assert seen["kwargs"]["task_name"] == "vector_add"
    assert seen["kwargs"]["arch"] == "compute_80"
    assert seen["kwargs"]["context_definition"] == _VECTOR_ADD_CONTEXT
    assert seen["kwargs"]["host_parameters"] == _VECTOR_ADD_HOST_PARAMS
    assert seen["kwargs"]["host_context_initializer"] == "a, b, out, n"


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required for CUDA scene-test smoke")
def test_scene_test_runs_cuda_host_schedule_vector_add_with_real_data(tmp_path):
    torch = pytest.importorskip("torch")

    source = tmp_path / "vector_add.pto.cu"
    source.write_text(_VECTOR_ADD_BODY)

    @scene_test(level=2, runtime="host_schedule")
    class CudaVectorAddScene(SceneTestCase):
        CALLABLE = _cuda_vector_add_spec(source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            return TaskArgsBuilder(
                Tensor("a", torch.arange(n, dtype=torch.float32)),
                Tensor("b", torch.arange(n, dtype=torch.float32) * 2.0),
                Tensor("out", torch.zeros(n, dtype=torch.float32)),
            )

        def compute_golden(self, args, params):
            args.out[:] = args.a + args.b

    scene = CudaVectorAddScene()
    worker = CudaVectorAddScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(worker, callable_obj, CudaVectorAddScene.CASES[0])
    finally:
        worker.close()
