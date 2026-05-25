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
