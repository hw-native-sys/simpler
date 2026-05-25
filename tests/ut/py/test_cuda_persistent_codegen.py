# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for CUDA persistent_device source generation."""

from __future__ import annotations

import json

import pytest

from simpler_setup import cuda_callable_compiler
from simpler_setup.cuda_callable_compiler import (
    CudaPersistentTaskFunction,
    compile_cuda_persistent_device,
    default_cuda_persistent_cache_root,
    render_persistent_dag_source,
)


def test_render_persistent_dag_source_generates_dispatch_switch():
    source = render_persistent_dag_source(
        [
            CudaPersistentTaskFunction(
                func_id=2,
                name="mul_f32",
                body="task->out[i] = task->a[i] * task->b[i];",
            ),
            CudaPersistentTaskFunction(
                func_id=1,
                name="add_f32",
                body="task->out[i] = task->a[i] + task->b[i];",
            ),
        ]
    )

    assert 'extern "C" __global__ void pto_persistent_dag_f32_executor' in source
    assert "__device__ void pto_task_add_f32" in source
    assert "__device__ void pto_task_mul_f32" in source
    assert "switch (task->func_id)" in source
    assert "case 1U:" in source
    assert "pto_task_add_f32(task);" in source
    assert "case 2U:" in source
    assert "pto_task_mul_f32(task);" in source
    assert source.index("case 1U:") < source.index("case 2U:")


def test_render_persistent_dag_source_rejects_duplicate_func_id():
    with pytest.raises(ValueError, match="duplicate func_id"):
        render_persistent_dag_source(
            [
                CudaPersistentTaskFunction(func_id=1, name="add_a", body=""),
                CudaPersistentTaskFunction(func_id=1, name="add_b", body=""),
            ]
        )


def test_compile_cuda_persistent_device_writes_manifest_and_reuses_cache(tmp_path, monkeypatch):
    calls = []

    def fake_run_nvcc(source_path, ptx_path, arch, nvcc):
        calls.append((source_path, ptx_path, arch, nvcc))
        ptx_path.write_bytes(b"fake-ptx")

    monkeypatch.setattr(cuda_callable_compiler, "_run_nvcc_ptx", fake_run_nvcc)

    tasks = [
        CudaPersistentTaskFunction(
            func_id=1,
            name="add_f32",
            body="task->out[i] = task->a[i] + task->b[i];",
        )
    ]
    first = compile_cuda_persistent_device(tasks, cache_root=tmp_path, arch="compute_80", nvcc="nvcc")
    second = compile_cuda_persistent_device(tasks, cache_root=tmp_path, arch="compute_80", nvcc="nvcc")

    assert first.cache_key == second.cache_key
    assert first.ptx_path == second.ptx_path
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(calls) == 1
    assert first.ptx == b"fake-ptx"
    assert first.source_path.read_text() == second.source_path.read_text()

    manifest = json.loads(first.manifest_path.read_text())
    assert manifest["runtime"] == "persistent_device"
    assert manifest["entry_name"] == "pto_persistent_dag_f32_executor"
    assert manifest["arch"] == "compute_80"
    assert manifest["source_kind"] == "generated-dispatch"
    assert manifest["task_functions"] == [{"func_id": 1, "name": "add_f32"}]


def test_compile_cuda_persistent_device_defaults_to_repo_cache(tmp_path, monkeypatch):
    def fake_run_nvcc(source_path, ptx_path, arch, nvcc):
        ptx_path.write_bytes(b"default-cache-ptx")

    monkeypatch.setattr(cuda_callable_compiler, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cuda_callable_compiler, "_run_nvcc_ptx", fake_run_nvcc)

    tasks = [
        CudaPersistentTaskFunction(
            func_id=1,
            name="add_f32",
            body="task->out[i] = task->a[i] + task->b[i];",
        )
    ]
    artifact = compile_cuda_persistent_device(tasks, arch="compute_80", nvcc="nvcc")

    expected_root = tmp_path / "build" / "cache" / "cuda" / "onboard" / "persistent_device"
    assert default_cuda_persistent_cache_root() == expected_root
    assert artifact.ptx == b"default-cache-ptx"
    assert artifact.ptx_path.parent.parent == expected_root / "callables"
    assert artifact.manifest_path.is_file()
