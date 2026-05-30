# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for CUDA KernelCompiler integration."""

from __future__ import annotations

from types import SimpleNamespace

from simpler_setup import kernel_compiler
from simpler_setup.kernel_compiler import KernelCompiler


def test_cuda_kernel_compiler_compiles_host_schedule_task_body(tmp_path, monkeypatch):
    seen = {}

    def fake_compile_cuda_host_schedule(task_body, arch, cache_root=None, nvcc="nvcc"):
        seen["task_body"] = task_body
        seen["arch"] = arch
        seen["cache_root"] = cache_root
        seen["nvcc"] = nvcc
        return SimpleNamespace(
            cache_key="fake-key",
            cache_hit=False,
            source_path=tmp_path / "generated_host_wrapper.cu",
            ptx_path=tmp_path / "pto_callable.ptx",
            manifest_path=tmp_path / "pto_callable.json",
            ptx=b"fake-ptx",
            entry_name="pto_kernel_vector_add",
            persistent_entry_name="pto_task_vector_add",
            arch=arch,
            source_kind="task-body-wrapper",
        )

    monkeypatch.setattr(kernel_compiler, "compile_cuda_host_schedule", fake_compile_cuda_host_schedule)
    source_path = tmp_path / "vector_add.pto.cu"
    source_path.write_text("out[i] = a[i] + b[i];\n")

    compiler = KernelCompiler(platform="cuda")
    artifact = compiler.compile_cuda_host_schedule(
        str(source_path),
        task_name="vector_add",
        arch="compute_80",
        cache_root=tmp_path / "cache",
        nvcc="nvcc-test",
    )

    assert artifact.ptx == b"fake-ptx"
    assert seen["task_body"].name == "vector_add"
    assert seen["task_body"].body == "out[i] = a[i] + b[i];\n"
    assert seen["task_body"].context_type == "PtoTaskContext"
    assert seen["arch"] == "compute_80"
    assert seen["cache_root"] == tmp_path / "cache"
    assert seen["nvcc"] == "nvcc-test"


def test_cuda_kernel_compiler_compiles_persistent_device_task_sources(tmp_path, monkeypatch):
    seen = {}

    def fake_compile_cuda_persistent_device(task_functions, arch, cache_root=None, nvcc="nvcc"):
        seen["task_functions"] = task_functions
        seen["arch"] = arch
        seen["cache_root"] = cache_root
        seen["nvcc"] = nvcc
        return SimpleNamespace(
            cache_key="persistent-key",
            cache_hit=False,
            source_path=tmp_path / "generated_dispatch.cu",
            ptx_path=tmp_path / "pto_callable.ptx",
            manifest_path=tmp_path / "pto_callable.json",
            ptx=b"persistent-ptx",
            entry_name="pto_persistent_dag_f32_executor",
            arch=arch,
            source_kind="generated-dispatch",
        )

    monkeypatch.setattr(kernel_compiler, "compile_cuda_persistent_device", fake_compile_cuda_persistent_device)
    add_src = tmp_path / "add.pto.cu"
    mul_src = tmp_path / "mul.pto.cu"
    add_src.write_text("task->out[i] = task->a[i] + task->b[i];\n")
    mul_src.write_text("task->out[i] = task->a[i] * task->b[i];\n")

    compiler = KernelCompiler(platform="cuda")
    artifact = compiler.compile_cuda_persistent_device(
        [
            {"func_id": 2, "task_name": "mul_f32", "source_path": str(mul_src)},
            {"func_id": 1, "task_name": "add_f32", "source_path": str(add_src)},
        ],
        arch="compute_90",
        cache_root=tmp_path / "cache",
        nvcc="/usr/local/cuda/bin/nvcc",
    )

    assert artifact.ptx == b"persistent-ptx"
    assert seen["arch"] == "compute_90"
    assert seen["cache_root"] == tmp_path / "cache"
    assert seen["nvcc"] == "/usr/local/cuda/bin/nvcc"
    assert [task.func_id for task in seen["task_functions"]] == [2, 1]
    assert [task.name for task in seen["task_functions"]] == ["mul_f32", "add_f32"]
    assert [task.body for task in seen["task_functions"]] == [
        "task->out[i] = task->a[i] * task->b[i];\n",
        "task->out[i] = task->a[i] + task->b[i];\n",
    ]
    assert [task.threading for task in seen["task_functions"]] == ["element", "element"]


def test_cuda_kernel_compiler_accepts_persistent_device_source_alias(tmp_path, monkeypatch):
    seen = {}

    def fake_compile_cuda_persistent_device(task_functions, arch, cache_root=None, nvcc="nvcc"):
        seen["task_functions"] = task_functions
        return SimpleNamespace(
            cache_key="persistent-key",
            cache_hit=False,
            source_path=tmp_path / "generated_dispatch.cu",
            ptx_path=tmp_path / "pto_callable.ptx",
            manifest_path=tmp_path / "pto_callable.json",
            ptx=b"persistent-ptx",
            entry_name="pto_persistent_dag_f32_executor",
            arch=arch,
            source_kind="generated-dispatch",
        )

    monkeypatch.setattr(kernel_compiler, "compile_cuda_persistent_device", fake_compile_cuda_persistent_device)
    add_src = tmp_path / "add.pto.cu"
    add_src.write_text("task->out[i] = task->a[i] + task->b[i];\n")

    compiler = KernelCompiler(platform="cuda")
    artifact = compiler.compile_cuda_persistent_device(
        [{"func_id": 1, "task_name": "add_f32", "source": str(add_src)}],
        arch="compute_80",
    )

    assert artifact.ptx == b"persistent-ptx"
    assert seen["task_functions"][0].body == "task->out[i] = task->a[i] + task->b[i];\n"


def test_cuda_kernel_compiler_preserves_persistent_task_threading(tmp_path, monkeypatch):
    seen = {}

    def fake_compile_cuda_persistent_device(task_functions, arch, cache_root=None, nvcc="nvcc"):
        seen["task_functions"] = task_functions
        return SimpleNamespace(
            cache_key="persistent-key",
            cache_hit=False,
            source_path=tmp_path / "generated_dispatch.cu",
            ptx_path=tmp_path / "pto_callable.ptx",
            manifest_path=tmp_path / "pto_callable.json",
            ptx=b"persistent-ptx",
            entry_name="pto_persistent_dag_f32_executor",
            arch=arch,
            source_kind="generated-dispatch",
        )

    monkeypatch.setattr(kernel_compiler, "compile_cuda_persistent_device", fake_compile_cuda_persistent_device)
    src = tmp_path / "wmma.pto.cu"
    src.write_text("if (threadIdx.x < 32) { task->out[threadIdx.x] = 1.0f; }\n")

    KernelCompiler(platform="cuda").compile_cuda_persistent_device(
        [
            {
                "func_id": 10,
                "task_name": "wmma_tile_f32",
                "source_path": str(src),
                "threading": "block",
            }
        ],
        arch="compute_90",
    )

    assert seen["task_functions"][0].threading == "block"


def test_cuda_kernel_compiler_compiles_persistent_device_task_bodies(tmp_path, monkeypatch):
    seen = {}

    def fake_compile_cuda_persistent_device(task_functions, arch, cache_root=None, nvcc="nvcc"):
        seen["task_functions"] = task_functions
        seen["arch"] = arch
        seen["cache_root"] = cache_root
        seen["nvcc"] = nvcc
        return SimpleNamespace(
            cache_key="task-body-key",
            cache_hit=False,
            source_path=tmp_path / "generated_dispatch.cu",
            ptx_path=tmp_path / "pto_callable.ptx",
            manifest_path=tmp_path / "pto_callable.json",
            ptx=b"task-body-ptx",
            entry_name="pto_persistent_dag_f32_executor",
            arch=arch,
            source_kind="generated-dispatch",
        )

    monkeypatch.setattr(kernel_compiler, "compile_cuda_persistent_device", fake_compile_cuda_persistent_device)
    add_src = tmp_path / "add.pto.cu"
    add_src.write_text("ctx->task->out[ctx->i] = ctx->task->a[ctx->i] + ctx->task->b[ctx->i];\n")

    compiler = KernelCompiler(platform="cuda")
    artifact = compiler.compile_cuda_persistent_device(
        [
            {
                "func_id": 1,
                "task_name": "add_f32",
                "source_path": str(add_src),
                "body_style": "task_body",
                "context_definition": """
struct PtoTaskContext {
    const PtoCudaPersistentDagTask *task;
    unsigned long long i;
};
""".strip(),
            }
        ],
        arch="compute_90",
        cache_root=tmp_path / "cache",
        nvcc="nvcc-test",
    )

    assert artifact.ptx == b"task-body-ptx"
    assert seen["arch"] == "compute_90"
    assert seen["cache_root"] == tmp_path / "cache"
    assert seen["nvcc"] == "nvcc-test"
    assert seen["task_functions"][0].func_id == 1
    assert seen["task_functions"][0].task_body.name == "add_f32"
    assert seen["task_functions"][0].task_body.context_type == "PtoTaskContext"
    assert "PtoCudaPersistentDagTask" in seen["task_functions"][0].task_body.context_definition
    assert seen["task_functions"][0].task_body.body == (
        "ctx->task->out[ctx->i] = ctx->task->a[ctx->i] + ctx->task->b[ctx->i];\n"
    )
