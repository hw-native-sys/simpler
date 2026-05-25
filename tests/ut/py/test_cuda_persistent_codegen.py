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

import ctypes
import json

import pytest

from simpler_setup import cuda_callable_compiler
from simpler_setup.cuda_callable_compiler import (
    CudaHostScheduleCallableArtifact,
    CudaPersistentCallableArtifact,
    CudaPersistentTaskBodyFunction,
    CudaPersistentTaskFunction,
    CudaTaskBody,
    compile_cuda_persistent_device,
    default_cuda_persistent_cache_root,
    render_persistent_dag_source,
)


def test_render_cuda_task_wrappers_share_one_task_body_between_runtimes():
    assert hasattr(cuda_callable_compiler, "CudaTaskBody")
    assert hasattr(cuda_callable_compiler, "render_cuda_task_wrappers")

    wrappers = cuda_callable_compiler.render_cuda_task_wrappers(
        cuda_callable_compiler.CudaTaskBody(
            name="vector_add",
            body="""
float *a = pto_arg<float *>(ctx, 0);
float *b = pto_arg<float *>(ctx, 1);
float *out = pto_arg<float *>(ctx, 2);
out[pto_linear_tid()] = a[pto_linear_tid()] + b[pto_linear_tid()];
""",
        )
    )

    assert wrappers.task_name == "vector_add"
    assert wrappers.body_name == "pto_task_body_vector_add"
    assert wrappers.host_entry_name == "pto_kernel_vector_add"
    assert wrappers.persistent_entry_name == "pto_task_vector_add"
    assert "__device__ void pto_task_body_vector_add(PtoTaskContext *ctx)" in wrappers.source
    assert 'extern "C" __global__ void pto_kernel_vector_add(PtoTaskContext ctx)' in wrappers.source
    assert "__device__ void pto_task_vector_add(PtoTaskContext *ctx)" in wrappers.source
    assert "pto_task_body_vector_add(&ctx);" in wrappers.source
    assert "pto_task_body_vector_add(ctx);" in wrappers.source
    assert wrappers.source.count("out[pto_linear_tid()] =") == 1


def test_render_cuda_task_wrappers_rejects_invalid_task_name():
    with pytest.raises(ValueError, match="invalid CUDA task body name"):
        cuda_callable_compiler.render_cuda_task_wrappers(cuda_callable_compiler.CudaTaskBody(name="not-valid", body=""))


def test_compile_cuda_host_schedule_writes_manifest_and_reuses_cache(tmp_path, monkeypatch):
    calls = []

    def fake_run_nvcc(source_path, ptx_path, arch, nvcc):
        calls.append((source_path, ptx_path, arch, nvcc))
        ptx_path.write_bytes(b"host-schedule-ptx")

    monkeypatch.setattr(cuda_callable_compiler, "_run_nvcc_ptx", fake_run_nvcc)

    assert hasattr(cuda_callable_compiler, "compile_cuda_host_schedule")

    task_body = cuda_callable_compiler.CudaTaskBody(
        name="vector_add",
        body="out[pto_linear_tid()] = a[pto_linear_tid()] + b[pto_linear_tid()];",
    )
    first = cuda_callable_compiler.compile_cuda_host_schedule(
        task_body, cache_root=tmp_path, arch="compute_80", nvcc="nvcc"
    )
    second = cuda_callable_compiler.compile_cuda_host_schedule(
        task_body, cache_root=tmp_path, arch="compute_80", nvcc="nvcc"
    )

    assert first.cache_key == second.cache_key
    assert first.ptx_path == second.ptx_path
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(calls) == 1
    assert first.ptx == b"host-schedule-ptx"
    assert "pto_kernel_vector_add" in first.source_path.read_text()

    manifest = json.loads(first.manifest_path.read_text())
    assert manifest["runtime"] == "host_schedule"
    assert manifest["entry_name"] == "pto_kernel_vector_add"
    assert manifest["persistent_entry_name"] == "pto_task_vector_add"
    assert manifest["arch"] == "compute_80"
    assert manifest["source_kind"] == "task-body-wrapper"
    assert manifest["task_body"] == {"context_type": "PtoTaskContext", "name": "vector_add"}


def test_prepare_cuda_host_schedule_callable_keeps_artifact_buffers_alive(tmp_path):
    assert hasattr(cuda_callable_compiler, "prepare_cuda_host_schedule_callable")

    artifact = CudaHostScheduleCallableArtifact(
        cache_key="abc",
        cache_hit=False,
        source_path=tmp_path / "generated_host_wrapper.cu",
        ptx_path=tmp_path / "pto_callable.ptx",
        manifest_path=tmp_path / "pto_callable.json",
        ptx=b"fake-host-ptx",
        entry_name="pto_kernel_vector_add",
        persistent_entry_name="pto_task_vector_add",
        arch="compute_80",
        source_kind="task-body-wrapper",
    )

    prepared = cuda_callable_compiler.prepare_cuda_host_schedule_callable(
        artifact,
        grid_dim=7,
        block_dim=128,
        shared_mem_bytes=64,
        stream_id=2,
    )

    assert prepared.runtime == "host_schedule"
    assert prepared.artifact is artifact
    assert prepared.manifest.version == 2
    assert prepared.manifest.op == 1
    assert prepared.manifest.image_size == len(artifact.ptx) + 1
    assert prepared.manifest.entry_name == b"pto_kernel_vector_add"
    assert prepared.manifest.grid_dim == 7
    assert prepared.manifest.block_dim == 128
    assert prepared.manifest.shared_mem_bytes == 64
    assert prepared.manifest.stream_id == 2
    assert prepared.byref()
    assert prepared.buffer_ptr() == ctypes.addressof(prepared.manifest)
    assert prepared.buffer_size() == ctypes.sizeof(prepared.manifest)
    assert prepared.to_bytes() == ctypes.string_at(prepared.buffer_ptr(), prepared.buffer_size())
    assert prepared.image_buffer.raw == b"fake-host-ptx\0"
    assert prepared.entry_name_buffer.value == artifact.entry_name.encode("utf-8")
    assert prepared.manifest.image == ctypes.cast(prepared.image_buffer, ctypes.c_void_p).value


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


def test_render_persistent_dag_source_can_use_shared_task_body_wrappers():
    source = render_persistent_dag_source(
        [
            CudaPersistentTaskBodyFunction(
                func_id=1,
                task_body=CudaTaskBody(
                    name="add_f32",
                    context_definition="""
struct PtoTaskContext {
    const PtoCudaPersistentDagTask *task;
    unsigned long long i;
};
""",
                    body="ctx->task->out[ctx->i] = ctx->task->a[ctx->i] + ctx->task->b[ctx->i];",
                ),
            )
        ]
    )

    assert "__device__ void pto_task_body_add_f32(PtoTaskContext *ctx)" in source
    assert "__device__ void pto_task_add_f32(PtoTaskContext *ctx)" in source
    assert "PtoTaskContext ctx{task, i};" in source
    assert "pto_task_add_f32(&ctx);" in source
    assert source.count("ctx->task->out[ctx->i] =") == 1


def test_render_persistent_dag_source_includes_tensor_descriptor_metadata():
    source = render_persistent_dag_source(
        [
            CudaPersistentTaskFunction(
                func_id=3,
                name="matmul_f32",
                body="task->out[i] = task->a[i];",
            )
        ]
    )

    assert "unsigned int rows;" in source
    assert "unsigned int cols;" in source
    assert "unsigned int inner;" in source
    assert "unsigned int lda;" in source
    assert "unsigned int ldb;" in source
    assert "unsigned int ldc;" in source
    assert "unsigned long long a_batch_stride;" in source
    assert "unsigned long long b_batch_stride;" in source
    assert "unsigned long long out_batch_stride;" in source


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


def test_prepare_cuda_persistent_device_callable_uses_generated_dispatch_entry(tmp_path):
    assert hasattr(cuda_callable_compiler, "prepare_cuda_persistent_device_callable")

    artifact = CudaPersistentCallableArtifact(
        cache_key="def",
        cache_hit=False,
        source_path=tmp_path / "generated_dispatch.cu",
        ptx_path=tmp_path / "pto_callable.ptx",
        manifest_path=tmp_path / "pto_callable.json",
        ptx=b"fake-persistent-ptx",
        entry_name="pto_persistent_dag_f32_executor",
        arch="compute_90",
        source_kind="generated-dispatch",
    )

    prepared = cuda_callable_compiler.prepare_cuda_persistent_device_callable(
        artifact,
        op=1003,
        grid_dim=5,
        block_dim=256,
    )

    assert prepared.runtime == "persistent_device"
    assert prepared.artifact is artifact
    assert prepared.manifest.version == 1
    assert prepared.manifest.op == 1003
    assert prepared.manifest.image_size == len(artifact.ptx) + 1
    assert prepared.manifest.entry_name == b"pto_persistent_dag_f32_executor"
    assert prepared.manifest.grid_dim == 5
    assert prepared.manifest.block_dim == 256
    assert prepared.manifest.shared_mem_bytes == 0
    assert prepared.byref()
    assert prepared.buffer_ptr() == ctypes.addressof(prepared.manifest)
    assert prepared.buffer_size() == ctypes.sizeof(prepared.manifest)
    assert prepared.to_bytes() == ctypes.string_at(prepared.buffer_ptr(), prepared.buffer_size())
    assert prepared.image_buffer.raw == b"fake-persistent-ptx\0"
    assert prepared.entry_name_buffer.value == artifact.entry_name.encode("utf-8")
    assert prepared.manifest.image == ctypes.cast(prepared.image_buffer, ctypes.c_void_p).value


def test_compile_cuda_persistent_device_task_body_manifest(tmp_path, monkeypatch):
    def fake_run_nvcc(source_path, ptx_path, arch, nvcc):
        ptx_path.write_bytes(b"task-body-ptx")

    monkeypatch.setattr(cuda_callable_compiler, "_run_nvcc_ptx", fake_run_nvcc)

    artifact = compile_cuda_persistent_device(
        [
            CudaPersistentTaskBodyFunction(
                func_id=1,
                task_body=CudaTaskBody(
                    name="add_f32",
                    context_definition="""
struct PtoTaskContext {
    const PtoCudaPersistentDagTask *task;
    unsigned long long i;
};
""",
                    body="ctx->task->out[ctx->i] = ctx->task->a[ctx->i] + ctx->task->b[ctx->i];",
                ),
            )
        ],
        cache_root=tmp_path,
        arch="compute_90",
        nvcc="nvcc",
    )

    manifest = json.loads(artifact.manifest_path.read_text())
    assert artifact.ptx == b"task-body-ptx"
    assert "pto_task_body_add_f32" in artifact.source_path.read_text()
    assert manifest["task_functions"] == [{"func_id": 1, "name": "add_f32"}]
    assert manifest["task_body_functions"][0]["func_id"] == 1
    assert manifest["task_body_functions"][0]["task_body"]["name"] == "add_f32"


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
