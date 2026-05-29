# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for CUDA benchmark report helpers."""

from __future__ import annotations

import ctypes
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from simpler_setup import cuda_callable_compiler


def _load_benchmark_module():
    script_path = (
        Path(__file__).resolve().parents[3]
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "cuda_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("cuda_benchmark", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_smoke_module():
    script_path = (
        Path(__file__).resolve().parents[3] / ".agents" / "skills" / "cuda-backend-eval" / "scripts" / "cuda_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("cuda_smoke", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_persistent_smoke_module():
    script_dir = Path(__file__).resolve().parents[3] / ".agents" / "skills" / "cuda-backend-eval" / "scripts"
    script_path = script_dir / "cuda_persistent_smoke.py"
    sys.path.insert(0, str(script_dir))
    try:
        spec = importlib.util.spec_from_file_location("cuda_persistent_smoke", script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("cuda_persistent_smoke", None)
        sys.path.remove(str(script_dir))


def _load_artifact_index_module():
    script_path = (
        Path(__file__).resolve().parents[3]
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "cuda_artifact_index.py"
    )
    spec = importlib.util.spec_from_file_location("cuda_artifact_index", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_pair_benchmark_module():
    script_path = (
        Path(__file__).resolve().parents[3]
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "cuda_pair_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("cuda_pair_benchmark", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(spec.name, None)


def _load_tensor_shape_sweep_module():
    script_path = (
        Path(__file__).resolve().parents[3]
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "cuda_tensor_shape_sweep.py"
    )
    spec = importlib.util.spec_from_file_location("cuda_tensor_shape_sweep", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(spec.name, None)


def _load_pair_smoke_module():
    script_path = (
        Path(__file__).resolve().parents[3]
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "cuda_pair_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("cuda_pair_smoke", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(spec.name, None)


def _load_pair_persistent_smoke_module():
    script_path = (
        Path(__file__).resolve().parents[3]
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "cuda_pair_persistent_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("cuda_pair_persistent_smoke", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(spec.name, None)


def _load_persistent_lifecycle_matrix_module():
    script_dir = Path(__file__).resolve().parents[3] / ".agents" / "skills" / "cuda-backend-eval" / "scripts"
    script_path = script_dir / "cuda_persistent_lifecycle_matrix.py"
    sys.path.insert(0, str(script_dir))
    try:
        spec = importlib.util.spec_from_file_location("cuda_persistent_lifecycle_matrix", script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("cuda_persistent_lifecycle_matrix", None)
        sys.path.remove(str(script_dir))


def _load_current_summary_module():
    script_dir = Path(__file__).resolve().parents[3] / ".agents" / "skills" / "cuda-backend-eval" / "scripts"
    script_path = script_dir / "cuda_current_summary.py"
    sys.path.insert(0, str(script_dir))
    try:
        spec = importlib.util.spec_from_file_location("cuda_current_summary", script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("cuda_current_summary", None)
        sys.path.remove(str(script_dir))


def _load_capture_validator_module():
    script_path = (
        Path(__file__).resolve().parents[3]
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "cuda_validate_capture.py"
    )
    spec = importlib.util.spec_from_file_location("cuda_validate_capture", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(spec.name, None)


def _load_tensor_sweep_validator_module():
    script_path = (
        Path(__file__).resolve().parents[3]
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "cuda_validate_tensor_sweep.py"
    )
    spec = importlib.util.spec_from_file_location("cuda_validate_tensor_sweep", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(spec.name, None)


def _load_smoke_validator_module():
    script_path = (
        Path(__file__).resolve().parents[3]
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "cuda_validate_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("cuda_validate_smoke", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(spec.name, None)


def _load_lifecycle_matrix_validator_module():
    script_path = (
        Path(__file__).resolve().parents[3]
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "cuda_validate_lifecycle_matrix.py"
    )
    spec = importlib.util.spec_from_file_location("cuda_validate_lifecycle_matrix", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(spec.name, None)


def _load_smoke_report_module():
    script_path = (
        Path(__file__).resolve().parents[3]
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "cuda_smoke_report.py"
    )
    spec = importlib.util.spec_from_file_location("cuda_smoke_report", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_artifact_payload(path: Path, label: str, machine: str, baseline: str) -> None:
    path.mkdir(parents=True)
    payload = {
        "metadata": {
            "label": label,
            "git_commit": "abc123",
            "machine": machine,
        },
        "results": [{"baseline": baseline, "n": 1024, "device_wall_ns": 10}],
    }
    (path / "cuda-benchmark.json").write_text(json.dumps(payload) + "\n")


def test_persistent_smoke_builds_graph_descriptor_dag_shape():
    cuda_persistent_smoke = _load_persistent_smoke_module()

    fanin, dependents, tasks = cuda_persistent_smoke._make_dag_shape(
        "graph_descriptor",
        17,
        0x1000,
        0x2000,
        0x3000,
        0x4000,
        0x5000,
        0x6000,
        0x7000,
    )

    assert list(fanin) == [0, 0, 2]
    assert list(dependents) == [2, 2]
    assert [task.func_id for task in tasks] == [9, 2, 1]
    assert [task.initial_fanin for task in tasks] == [0, 0, 2]
    assert [task.dependent_count for task in tasks] == [1, 1, 0]
    assert tasks[0].out == 0x4000
    assert tasks[2].a == tasks[0].out
    assert tasks[2].b == tasks[1].out
    assert list(tasks[0].tensor_args)[:2] == [0x3000, 0x6000]
    assert list(tasks[0].scalar_args)[:2] == [1.5, 0.25]
    assert tasks[0].tensor_arg_count == 2
    assert tasks[0].scalar_arg_count == 2


def test_persistent_smoke_builds_tagged_graph_descriptor_dag_shape():
    cuda_persistent_smoke = _load_persistent_smoke_module()

    fanin, dependents, tasks = cuda_persistent_smoke._make_dag_shape(
        "graph_descriptor_tagged",
        17,
        0x1000,
        0x2000,
        0x3000,
        0x4000,
        0x5000,
        0x6000,
        0x7000,
    )

    assert list(fanin) == [0, 0, 2]
    assert list(dependents) == [2, 2]
    assert [task.func_id for task in tasks] == [9, 2, 1]
    assert [task.initial_fanin for task in tasks] == [0, 0, 2]
    assert [task.dependent_count for task in tasks] == [1, 1, 0]
    assert tasks[0].a == 0x1000
    assert tasks[0].b == 0x2000
    assert tasks[0].out == 0x4000
    assert tasks[1].out == 0x5000
    assert tasks[2].a == tasks[0].out
    assert tasks[2].b == tasks[1].out
    assert tasks[2].out == 0x7000
    assert list(tasks[0].tensor_args)[:2] == [0x3000, 0x6000]
    assert list(tasks[0].scalar_args)[:2] == [1.5, 0.25]
    assert tasks[0].tensor_arg_count == 2
    assert tasks[0].scalar_arg_count == 2


def test_persistent_smoke_builds_tagged_inout_graph_descriptor_dag_shape():
    cuda_persistent_smoke = _load_persistent_smoke_module()

    fanin, dependents, tasks = cuda_persistent_smoke._make_dag_shape(
        "graph_descriptor_tagged_inout",
        17,
        0x1000,
        0x2000,
        0x3000,
        0x4000,
        0x5000,
        0x6000,
        0x7000,
    )

    assert list(fanin) == [0, 1, 1]
    assert list(dependents) == [1, 2]
    assert [task.func_id for task in tasks] == [1, 1, 1]
    assert [task.initial_fanin for task in tasks] == [0, 1, 1]
    assert [task.dependent_count for task in tasks] == [1, 1, 0]
    assert tasks[0].a == 0x1000
    assert tasks[0].b == 0x2000
    assert tasks[0].out == 0x4000
    assert tasks[1].a == tasks[0].out
    assert tasks[1].b == 0x2000
    assert tasks[1].out == tasks[0].out
    assert tasks[2].a == tasks[1].out
    assert tasks[2].b == 0x1000
    assert tasks[2].out == 0x7000


def test_persistent_smoke_builds_graph_descriptor_generic_args4_dag_shape():
    cuda_persistent_smoke = _load_persistent_smoke_module()

    fanin, dependents, tasks = cuda_persistent_smoke._make_dag_shape(
        "graph_descriptor_generic_args4",
        17,
        0x1000,
        0x2000,
        0x3000,
        0x4000,
        0x5000,
        0x6000,
        0x7000,
    )

    assert list(fanin) == [0, 0, 2]
    assert list(dependents) == [2, 2]
    assert [task.func_id for task in tasks] == [9, 2, 1]
    assert list(tasks[0].tensor_args) == [0x3000, 0x6000, 0x1000, 0x2000]
    assert list(tasks[0].scalar_args) == [1.5, 0.25, 0.125, 0.0625]
    assert tasks[0].tensor_arg_count == 4
    assert tasks[0].scalar_arg_count == 4


def test_persistent_smoke_builds_reordered_graph_descriptor_dag_shape():
    cuda_persistent_smoke = _load_persistent_smoke_module()

    fanin, dependents, tasks = cuda_persistent_smoke._make_dag_shape(
        "graph_descriptor_reordered",
        17,
        0x1000,
        0x2000,
        0x3000,
        0x4000,
        0x5000,
        0x6000,
        0x7000,
    )

    assert list(fanin) == [2, 0, 0]
    assert list(dependents) == [0, 0]
    assert [task.func_id for task in tasks] == [1, 9, 2]
    assert [task.initial_fanin for task in tasks] == [2, 0, 0]
    assert [task.dependent_count for task in tasks] == [0, 1, 1]
    assert tasks[0].a == 0x4000
    assert tasks[0].b == 0x5000
    assert tasks[0].out == 0x7000
    assert tasks[1].out == tasks[0].a
    assert tasks[2].out == tasks[0].b
    assert list(tasks[1].tensor_args)[:2] == [0x3000, 0x6000]
    assert list(tasks[1].scalar_args)[:2] == [1.5, 0.25]


def test_persistent_smoke_builds_graph_descriptor_chain_dag_shape():
    cuda_persistent_smoke = _load_persistent_smoke_module()

    fanin, dependents, tasks = cuda_persistent_smoke._make_dag_shape(
        "graph_descriptor_chain",
        17,
        0x1000,
        0x2000,
        0x3000,
        0x4000,
        0x5000,
        0x6000,
        0x7000,
    )

    assert list(fanin) == [0, 0, 2, 1, 1]
    assert list(dependents) == [2, 2, 3, 4]
    assert [task.func_id for task in tasks] == [1, 2, 1, 2, 1]
    assert [task.initial_fanin for task in tasks] == [0, 0, 2, 1, 1]
    assert [task.dependent_count for task in tasks] == [1, 1, 1, 1, 0]
    assert tasks[0].out == 0x3000
    assert tasks[1].out == 0x4000
    assert tasks[2].a == tasks[0].out
    assert tasks[2].b == tasks[1].out
    assert tasks[2].out == 0x5000
    assert tasks[3].a == tasks[2].out
    assert tasks[3].out == 0x6000
    assert tasks[4].a == tasks[2].out
    assert tasks[4].b == tasks[3].out
    assert tasks[4].out == 0x7000


def test_persistent_smoke_builds_graph_tensor_tile_dag_shape():
    cuda_persistent_smoke = _load_persistent_smoke_module()
    descriptor = cuda_persistent_smoke._make_tensor_tile_descriptor(rows=8, cols=4, inner=12)

    fanin, dependents, tasks = cuda_persistent_smoke._make_dag_shape(
        "graph_tensor_tile",
        64,
        0x1000,
        0x2000,
        0x3000,
        0x4000,
        0x5000,
        0x6000,
        0x7000,
        tensor_tile=descriptor,
    )

    assert list(fanin) == [0, 1, 1, 2]
    assert list(dependents) == [1, 2, 3, 3]
    assert [task.func_id for task in tasks] == [3, 1, 2, 1]
    assert [task.initial_fanin for task in tasks] == [0, 1, 1, 2]
    assert [task.dependent_count for task in tasks] == [2, 1, 1, 0]
    assert tasks[0].rows == 8
    assert tasks[0].cols == 4
    assert tasks[0].inner == 12
    assert tasks[0].lda == 12
    assert tasks[0].ldb == 4
    assert tasks[0].ldc == 4
    assert tasks[0].a_batch_stride == 96
    assert tasks[0].b_batch_stride == 48
    assert tasks[0].out_batch_stride == 32
    assert tasks[3].out == 0x7000


def test_persistent_smoke_builds_graph_descriptor_triad_dag_shape():
    cuda_persistent_smoke = _load_persistent_smoke_module()

    fanin, dependents, tasks = cuda_persistent_smoke._make_dag_shape(
        "graph_descriptor_triad",
        17,
        0x1000,
        0x2000,
        0x3000,
        0x4000,
        0x5000,
        0x6000,
        0x7000,
    )

    assert list(fanin) == [0, 0, 2]
    assert list(dependents) == [2, 2]
    assert [task.func_id for task in tasks] == [6, 2, 1]
    assert [task.initial_fanin for task in tasks] == [0, 0, 2]
    assert tasks[0].a == 0x1000
    assert tasks[0].b == 0x2000
    assert tasks[0].c == 0x3000
    assert tasks[0].out == 0x4000
    assert tasks[2].a == tasks[0].out
    assert tasks[2].b == tasks[1].out
    assert tasks[2].out == 0x7000


def test_persistent_smoke_builds_graph_descriptor_quad_dag_shape():
    cuda_persistent_smoke = _load_persistent_smoke_module()

    fanin, dependents, tasks = cuda_persistent_smoke._make_dag_shape(
        "graph_descriptor_quad",
        17,
        0x1000,
        0x2000,
        0x3000,
        0x4000,
        0x5000,
        0x6000,
        0x7000,
    )

    assert list(fanin) == [0, 0, 2]
    assert list(dependents) == [2, 2]
    assert [task.func_id for task in tasks] == [8, 2, 1]
    assert [task.initial_fanin for task in tasks] == [0, 0, 2]
    assert tasks[0].a == 0x1000
    assert tasks[0].b == 0x2000
    assert tasks[0].c == 0x3000
    assert tasks[0].d == 0x6000
    assert tasks[0].out == 0x4000
    assert tasks[2].a == tasks[0].out
    assert tasks[2].b == tasks[1].out
    assert tasks[2].out == 0x7000


def test_persistent_smoke_builds_scalar_scale_dag_shape():
    cuda_persistent_smoke = _load_persistent_smoke_module()

    fanin, dependents, tasks = cuda_persistent_smoke._make_dag_shape(
        "scalar_scale",
        17,
        0x1000,
        0x2000,
        0x3000,
        0x4000,
        0x5000,
        0x6000,
        0x7000,
    )

    assert list(fanin) == [0, 0, 2]
    assert list(dependents) == [2, 2]
    assert [task.func_id for task in tasks] == [11, 2, 1]
    assert [task.initial_fanin for task in tasks] == [0, 0, 2]
    assert tasks[0].a == 0x1000
    assert tasks[0].b is None
    assert tasks[0].out == 0x3000
    assert tasks[0].scalar0 == 2.0
    assert tasks[2].a == tasks[0].out
    assert tasks[2].b == tasks[1].out
    assert tasks[2].out == 0x7000


def test_persistent_smoke_builds_graph_descriptor_scalar_scale_dag_shape():
    cuda_persistent_smoke = _load_persistent_smoke_module()

    fanin, dependents, tasks = cuda_persistent_smoke._make_dag_shape(
        "graph_descriptor_scalar_scale",
        17,
        0x1000,
        0x2000,
        0x3000,
        0x4000,
        0x5000,
        0x6000,
        0x7000,
    )

    assert list(fanin) == [0, 0, 2]
    assert list(dependents) == [2, 2]
    assert [task.func_id for task in tasks] == [11, 2, 1]
    assert [task.initial_fanin for task in tasks] == [0, 0, 2]
    assert tasks[0].a == 0x1000
    assert tasks[0].b is None
    assert tasks[0].out == 0x3000
    assert tasks[0].scalar0 == 2.0
    assert tasks[2].a == tasks[0].out
    assert tasks[2].b == tasks[1].out
    assert tasks[2].out == 0x7000


def test_persistent_smoke_builds_graph_descriptor_scalar_axpy_dag_shape():
    cuda_persistent_smoke = _load_persistent_smoke_module()

    fanin, dependents, tasks = cuda_persistent_smoke._make_dag_shape(
        "graph_descriptor_scalar_axpy",
        17,
        0x1000,
        0x2000,
        0x3000,
        0x4000,
        0x5000,
        0x6000,
        0x7000,
    )

    assert list(fanin) == [0, 0, 2]
    assert list(dependents) == [2, 2]
    assert [task.func_id for task in tasks] == [4, 2, 1]
    assert [task.initial_fanin for task in tasks] == [0, 0, 2]
    assert tasks[0].a == 0x1000
    assert tasks[0].b == 0x2000
    assert tasks[0].out == 0x3000
    assert tasks[0].scalar0 == 1.5
    assert tasks[2].a == tasks[0].out
    assert tasks[2].b == tasks[1].out
    assert tasks[2].out == 0x7000


def test_persistent_smoke_builds_graph_descriptor_scalar_affine_dag_shape():
    cuda_persistent_smoke = _load_persistent_smoke_module()

    fanin, dependents, tasks = cuda_persistent_smoke._make_dag_shape(
        "graph_descriptor_scalar_affine",
        17,
        0x1000,
        0x2000,
        0x3000,
        0x4000,
        0x5000,
        0x6000,
        0x7000,
    )

    assert list(fanin) == [0, 0, 2]
    assert list(dependents) == [2, 2]
    assert [task.func_id for task in tasks] == [5, 2, 1]
    assert [task.initial_fanin for task in tasks] == [0, 0, 2]
    assert tasks[0].a == 0x1000
    assert tasks[0].b == 0x2000
    assert tasks[0].out == 0x3000
    assert tasks[0].scalar0 == 1.5
    assert tasks[0].scalar1 == 0.5
    assert tasks[2].a == tasks[0].out
    assert tasks[2].b == tasks[1].out
    assert tasks[2].out == 0x7000


def test_cuda_smoke_report_renders_markdown_and_svg(tmp_path):
    cuda_smoke_report = _load_smoke_report_module()
    a100 = {
        "status": "pass",
        "runtime": "persistent_device",
        "mode": "dag",
        "dag_shape": "tensor_tile",
        "n": 4096,
        "ptx_arch": "compute_80",
        "ptx_source": "nvcc-persistent-generated-dispatch-compute_80",
        "device_wall_ns": 102400,
        "host_wall_ns": 122260,
        "repeat_runs": 2,
        "launch_completed_counts": [4, 4],
        "dispatch_func_ids": [3, 1, 2, 1],
        "device_scheduler_errors": {"count": 0, "code": 0, "task_id": 0},
        "resource_policy": {
            "scheduler_blocks": 1,
            "worker_blocks": 2,
            "worker_blocks_per_task": 1,
            "stream_id": 1,
            "block_dim": 256,
            "grid_dim": 3,
        },
        "scalar_args": {"scalar0": 1.5},
        "tensor_args": {"c": "tmp0"},
        "scratch_reuse": {"reused_buffer": "tmp0", "reuse_task": 4},
        "graph_task_args": {
            "task0": "input:a,input:b,output:tmp1",
            "task1": "input:a,input:b,output:tmp2",
        },
        "tensor_core": {
            "api": "wmma",
            "mma_shape": "m16n16k8",
            "input": "tf32",
            "accumulator": "f32",
        },
        "tensor_tile": {
            "rows": 16,
            "cols": 16,
            "inner": 16,
            "tile_count": 16,
        },
    }
    h200 = {
        **a100,
        "ptx_arch": "compute_90",
        "ptx_source": "nvcc-persistent-generated-dispatch-compute_90",
        "device_wall_ns": 70464,
        "host_wall_ns": 79788,
        "device_scheduler_errors": {"count": 1, "code": 7, "task_id": 3},
    }
    a100_path = tmp_path / "a100.json"
    h200_path = tmp_path / "h200.json"
    a100_path.write_text(json.dumps(a100) + "\n")
    h200_path.write_text(json.dumps(h200) + "\n")

    payload = cuda_smoke_report.load_smoke_payloads([a100_path, h200_path])
    markdown = cuda_smoke_report.render_markdown_report(payload, label="tensor-smoke")
    svg = cuda_smoke_report.render_svg_report(payload, label="tensor-smoke")

    assert (
        "| Tensor core | Dispatch | Scheduler errors | Repeat runs | Launch completions | "
        "Resource policy | Scalar args | Tensor args | Scratch reuse | Graph task args |" in markdown
    )
    assert "| a100 | pass | persistent_device | dag/tensor_tile | 4096 | `compute_80` | 102400 | 122260 |" in markdown
    assert "| h200 | pass | persistent_device | dag/tensor_tile | 4096 | `compute_90` | 70464 | 79788 |" in markdown
    assert (
        "| `wmma:m16n16k8:tf32->f32` | `3,1,2,1` | `count=0,code=0,task=0` | `2` | `4,4` | "
        "`sched=1,workers=2,wp=1,stream=1,block=256,grid=3` | "
        "`scalar0=1.5` | `c=tmp0` | "
        "`reused_buffer=tmp0,reuse_task=4` | "
        "`task0=input:a,input:b,output:tmp1;task1=input:a,input:b,output:tmp2` |" in markdown
    )
    assert (
        "| `3,1,2,1` | `count=1,code=7,task=3` | `2` | `4,4` | "
        "`sched=1,workers=2,wp=1,stream=1,block=256,grid=3` | "
        "`scalar0=1.5` | `c=tmp0` | `reused_buffer=tmp0,reuse_task=4` |" in markdown
    )
    assert "nvcc-persistent-generated-dispatch-compute_90" in markdown
    assert "<svg" in svg
    assert "tensor-smoke" in svg
    assert "h200" in svg
    assert "errors: count=1,code=7,task=3" in svg
    assert "policy: sched=1,workers=2,wp=1,stream=1,block=256,grid=3" in svg
    assert "lifecycle: repeat=2,completed=4,4" in svg
    assert "scalars: scalar0=1.5" in svg
    assert "tensors: c=tmp0" in svg
    assert "scratch: reused_buffer=tmp0,reuse_task=4" in svg
    assert "task args: task0=input:a,input:b,output:tmp1;task1=input:a,input:b,output:tmp2" in svg


def test_cuda_smoke_scripts_use_shared_callable_manifest_types():
    cuda_smoke = _load_smoke_module()
    cuda_persistent_smoke = _load_persistent_smoke_module()

    assert cuda_smoke.CudaHostCallable is cuda_callable_compiler.CudaHostScheduleCallable
    assert cuda_persistent_smoke.CudaPersistentCallable is cuda_callable_compiler.CudaPersistentDeviceCallable


def test_cuda_artifact_index_scans_benchmark_outputs(tmp_path):
    cuda_artifact_index = _load_artifact_index_module()
    artifact_dir = tmp_path / "combined-graph"
    _write_artifact_payload(
        artifact_dir,
        label="cuda-graph-a100-h200",
        machine="combined",
        baseline="direct_driver_graph",
    )
    (artifact_dir / "cuda-benchmark.md").write_text("# report\n")
    (artifact_dir / "cuda-benchmark.svg").write_text("<svg></svg>\n")
    (artifact_dir / "cuda-benchmark-throughput.svg").write_text("<svg></svg>\n")
    (artifact_dir / "cuda-benchmark-ratios.svg").write_text("<svg></svg>\n")
    (artifact_dir / "cuda-benchmark-dag-deltas.svg").write_text("<svg></svg>\n")

    entries = cuda_artifact_index.scan_artifacts(tmp_path)

    assert entries == [
        {
            "path": "combined-graph",
            "kind": "benchmark",
            "label": "cuda-graph-a100-h200",
            "machine": "combined",
            "git_commit": "abc123",
            "result_count": 1,
            "baselines": ["direct_driver_graph"],
            "sizes": [1024],
            "tensor_tiles": [],
            "source_papers": [],
            "has_command_examples": False,
            "has_markdown": True,
            "has_svg": True,
            "has_throughput_svg": True,
            "has_ratio_svg": True,
            "has_dag_delta_svg": True,
        }
    ]


def test_cuda_artifact_index_renders_markdown_and_writes_default_index(tmp_path):
    cuda_artifact_index = _load_artifact_index_module()
    artifact_dir = tmp_path / "a100-graph"
    _write_artifact_payload(
        artifact_dir,
        label="a100-graph",
        machine="hina",
        baseline="direct_driver_graph",
    )

    output = cuda_artifact_index.write_index(tmp_path)
    report = output.read_text()

    assert output == tmp_path / "index.md"
    assert "# CUDA Backend Artifact Index" in report
    assert (
        "| a100-graph | benchmark | a100-graph | hina | abc123 | 1 | 1024 |  |  |  |  |  |  |  |  |  |  | "
        " | no | direct_driver_graph |"
    ) in report
    assert "ratio SVG" in report
    assert "DAG delta SVG" in report


def test_cuda_artifact_index_records_benchmark_tensor_tiles(tmp_path):
    cuda_artifact_index = _load_artifact_index_module()
    artifact_dir = tmp_path / "combined-tensorflags"
    artifact_dir.mkdir()
    payload = {
        "metadata": {
            "label": "tensorflags",
            "git_commit": "abc123",
            "machine": "combined",
            "tensor_tile": {"rows": 8, "cols": 4, "inner": 12},
        },
        "results": [
            {
                "baseline": "pto_persistent_dag_tensor",
                "n": 64,
                "device_wall_ns": 10,
                "tensor_tile": {
                    "rows": 8,
                    "cols": 4,
                    "inner": 12,
                    "tile_count": 2,
                },
            }
        ],
    }
    (artifact_dir / "cuda-benchmark.json").write_text(json.dumps(payload) + "\n")

    entries = cuda_artifact_index.scan_artifacts(tmp_path)
    report = cuda_artifact_index.render_markdown(entries)

    assert entries[0]["tensor_tiles"] == ["8x4x12"]
    assert "| combined-tensorflags | benchmark | tensorflags | combined | abc123 | 1 | 64 | 8x4x12 |" in report


def test_cuda_artifact_index_records_benchmark_source_papers(tmp_path):
    cuda_artifact_index = _load_artifact_index_module()
    artifact_dir = tmp_path / "combined-provenance"
    artifact_dir.mkdir()
    payload = {
        "metadata": {
            "label": "provenance",
            "git_commit": "abc123",
            "machine": "combined",
            "source_papers": [
                {"id": "arXiv:2605.03190", "label": "VDCores"},
                {"id": "arXiv:2512.22219v1", "label": "MPK persistent kernel"},
            ],
        },
        "results": [{"baseline": "pto_host_schedule", "n": 1024, "device_wall_ns": 10}],
    }
    (artifact_dir / "cuda-benchmark.json").write_text(json.dumps(payload) + "\n")

    entries = cuda_artifact_index.scan_artifacts(tmp_path)
    report = cuda_artifact_index.render_markdown(entries)

    assert entries[0]["source_papers"] == ["arXiv:2512.22219v1", "arXiv:2605.03190"]
    assert "arXiv:2512.22219v1, arXiv:2605.03190 | no |" in report


def test_cuda_artifact_index_scans_tensor_shape_sweep_outputs(tmp_path):
    cuda_artifact_index = _load_artifact_index_module()
    artifact_dir = tmp_path / "tensor-shape-sweep-abc123"
    artifact_dir.mkdir()
    payload = {
        "metadata": {
            "label": "tensor-shape-sweep-abc123",
            "git_commit": "abc123",
            "n": 256,
            "baselines": ["pto_persistent_dag_tensor", "cublas_sgemm"],
            "shapes": ["16x16x16", "16x16x64"],
            "source_papers": [
                {"id": "arXiv:2605.03190", "label": "VDCores"},
                {"id": "arXiv:2512.22219v1", "label": "MPK persistent kernel"},
            ],
            "command_examples": {
                "local_sample": "env PYTHONPATH=$PWD:$PWD/python cuda_benchmark.py",
                "remote_sample": "ssh h200-box cuda_benchmark.py",
            },
        },
        "results": [
            {
                "artifact": "a100",
                "machine": "hina",
                "baseline": "pto_persistent_dag_tensor",
                "shape": "16x16x16",
                "tensor_tile": {"rows": 16, "cols": 16, "inner": 16, "tile_count": 1},
            },
            {
                "artifact": "h200",
                "machine": "dasys-h200x8",
                "baseline": "cublas_sgemm",
                "shape": "16x16x64",
                "tensor_tile": {"rows": 16, "cols": 16, "inner": 64, "tile_count": 1},
            },
        ],
    }
    (artifact_dir / "cuda-tensor-shape-sweep.json").write_text(json.dumps(payload) + "\n")
    (artifact_dir / "cuda-tensor-shape-sweep.md").write_text("# report\n")
    (artifact_dir / "cuda-tensor-shape-sweep.svg").write_text("<svg></svg>\n")
    (artifact_dir / "cuda-tensor-shape-throughput.svg").write_text("<svg></svg>\n")

    entries = cuda_artifact_index.scan_artifacts(tmp_path)
    report = cuda_artifact_index.render_markdown(entries)

    assert entries == [
        {
            "path": "tensor-shape-sweep-abc123",
            "kind": "tensor_sweep",
            "label": "tensor-shape-sweep-abc123",
            "machine": "combined",
            "git_commit": "abc123",
            "result_count": 2,
            "baselines": ["cublas_sgemm", "pto_persistent_dag_tensor"],
            "sizes": [256],
            "tensor_tiles": ["16x16x16", "16x16x64"],
            "source_papers": ["arXiv:2512.22219v1", "arXiv:2605.03190"],
            "has_command_examples": True,
            "has_markdown": True,
            "has_svg": True,
            "has_throughput_svg": True,
            "has_ratio_svg": False,
            "has_dag_delta_svg": False,
        }
    ]
    assert (
        "| tensor-shape-sweep-abc123 | tensor_sweep | tensor-shape-sweep-abc123 | "
        "combined | abc123 | 2 | 256 | 16x16x16, 16x16x64 |"
    ) in report
    assert "arXiv:2512.22219v1, arXiv:2605.03190 | yes |" in report


def test_cuda_artifact_index_scans_smoke_report_outputs(tmp_path):
    cuda_artifact_index = _load_artifact_index_module()
    artifact_dir = tmp_path / "tensor-descriptor-smoke"
    artifact_dir.mkdir()
    smoke_payload = {
        "status": "pass",
        "runtime": "persistent_device",
        "mode": "dag",
        "dag_shape": "tensor_tile",
        "n": 4096,
        "dispatch_func_ids": [3, 1, 2, 1],
        "device_scheduler_errors": {"count": 0, "code": 0, "task_id": 0},
        "resource_policy": {
            "scheduler_blocks": 1,
            "worker_blocks": 2,
            "worker_blocks_per_task": 1,
            "stream_id": 1,
            "block_dim": 256,
            "grid_dim": 3,
        },
        "scalar_args": {"scalar0": 1.5},
        "tensor_args": {"c": "tmp0"},
        "graph_task_args": {
            "task0": "input:a,input:b,output:tmp1",
            "task1": "input:a,input:b,output:tmp2",
        },
        "tensor_tile": {
            "rows": 16,
            "cols": 16,
            "inner": 16,
            "tile_count": 16,
        },
    }
    (artifact_dir / "a100.json").write_text(json.dumps(smoke_payload) + "\n")
    h200_payload = {
        **smoke_payload,
        "device_scheduler_errors": {"count": 1, "code": 7, "task_id": 3},
    }
    (artifact_dir / "h200.json").write_text(json.dumps(h200_payload) + "\n")
    (artifact_dir / "cuda-smoke-report.md").write_text("# CUDA Smoke Report\n\n- Label: `tensor-smoke`\n")
    (artifact_dir / "cuda-smoke-report.svg").write_text("<svg></svg>\n")

    entries = cuda_artifact_index.scan_artifacts(tmp_path)
    report = cuda_artifact_index.render_markdown(entries)

    assert entries == [
        {
            "path": "tensor-descriptor-smoke",
            "kind": "smoke",
            "label": "tensor-smoke",
            "machine": "combined",
            "git_commit": "unknown",
            "result_count": 2,
            "baselines": ["persistent_device"],
            "sizes": [4096],
            "smoke_modes": ["dag/tensor_tile"],
            "dispatches": ["3,1,2,1"],
            "scheduler_errors": [
                "count=0,code=0,task=0",
                "count=1,code=7,task=3",
            ],
            "repeat_runs": [],
            "launch_completed_counts": [],
            "resource_policies": ["sched=1,workers=2,wp=1,stream=1,block=256,grid=3"],
            "scalar_args": ["scalar0=1.5"],
            "tensor_args": ["c=tmp0"],
            "graph_task_args": ["task0=input:a,input:b,output:tmp1;task1=input:a,input:b,output:tmp2"],
            "tensor_tiles": ["16x16x16"],
            "has_markdown": True,
            "has_svg": True,
            "has_throughput_svg": False,
            "has_ratio_svg": False,
            "has_dag_delta_svg": False,
        }
    ]
    assert (
        "Smoke mode | Dispatch | Scheduler errors | Repeat runs | Launch completions | "
        "Resource policy | Scalar args | Tensor args |"
    ) in report
    assert "| tensor-descriptor-smoke | smoke | tensor-smoke | combined | unknown | 2 |" in report
    assert "| 4096 | 16x16x16 | dag/tensor_tile | 3,1,2,1 |" in report
    assert "count=0,code=0,task=0, count=1,code=7,task=3 |" in report
    assert "sched=1,workers=2,wp=1,stream=1,block=256,grid=3 |" in report
    assert "scalar0=1.5 | c=tmp0 | task0=input:a,input:b,output:tmp1;task1=input:a,input:b,output:tmp2 |" in report


def test_cuda_artifact_index_records_persistent_smoke_lifecycle_reuse(tmp_path):
    cuda_artifact_index = _load_artifact_index_module()
    artifact_dir = tmp_path / "graph-descriptor-repeat-smoke"
    artifact_dir.mkdir()
    smoke_payload = {
        "status": "pass",
        "runtime": "persistent_device",
        "mode": "dag",
        "dag_shape": "graph_descriptor",
        "n": 1024,
        "repeat_runs": 2,
        "launch_completed_counts": [3, 3],
        "dispatch_func_ids": [9, 2, 1],
        "device_scheduler_errors": {"count": 0, "code": 0, "task_id": 0},
    }
    (artifact_dir / "a100.json").write_text(json.dumps(smoke_payload) + "\n")
    (artifact_dir / "h200.json").write_text(json.dumps(smoke_payload) + "\n")
    (artifact_dir / "cuda-smoke-report.md").write_text(
        "# CUDA Smoke Report\n\n- Label: `graph-descriptor-repeat-smoke`\n"
    )

    [entry] = cuda_artifact_index.scan_artifacts(tmp_path)
    report = cuda_artifact_index.render_markdown([entry])

    assert entry["repeat_runs"] == [2]
    assert entry["launch_completed_counts"] == ["3,3"]
    assert "Repeat runs | Launch completions" in report
    assert "| dag/graph_descriptor | 9,2,1 | count=0,code=0,task=0 | 2 | 3,3 |" in report


def test_cuda_artifact_index_sorts_numeric_sizes_before_strings(tmp_path):
    cuda_artifact_index = _load_artifact_index_module()
    artifact_dir = tmp_path / "mixed-sizes"
    artifact_dir.mkdir()
    payload = {
        "metadata": {"label": "mixed-sizes"},
        "results": [
            {"baseline": "pto_host_schedule", "n": 1048576},
            {"baseline": "pto_host_schedule", "n": 1024},
            {"baseline": "pto_host_schedule", "n": "unknown"},
            {"baseline": "pto_host_schedule", "n": 65536},
        ],
    }
    (artifact_dir / "cuda-benchmark.json").write_text(json.dumps(payload) + "\n")

    [entry] = cuda_artifact_index.scan_artifacts(tmp_path)

    assert entry["sizes"] == [1024, 65536, 1048576, "unknown"]


def _paired_capture_payload():
    baselines = [
        "direct_driver",
        "direct_driver_graph",
        "pto_host_schedule",
        "pto_host_schedule_compiler",
    ]
    results = []
    for machine in ("hina", "dasys-h200x8"):
        for n in (1024, 65536):
            for baseline in baselines:
                for repeat in range(2):
                    results.append(
                        {
                            "machine": machine,
                            "baseline": baseline,
                            "n": n,
                            "repeat": repeat,
                            "status": "pass",
                            "device_wall_ns": 1024,
                        }
                    )
    return {
        "metadata": {
            "label": "combined-current-abc123",
            "git_commit": "abc123",
            "machine": "combined",
        },
        "results": results,
    }


def test_cuda_capture_validator_accepts_complete_capture(tmp_path):
    cuda_validate_capture = _load_capture_validator_module()
    artifact_dir = tmp_path / "combined-current-abc123"
    artifact_dir.mkdir()
    payload = _paired_capture_payload()
    json_path = artifact_dir / "cuda-benchmark.json"
    json_path.write_text(json.dumps(payload) + "\n")
    (artifact_dir / "cuda-benchmark.md").write_text("# report\n")
    (artifact_dir / "cuda-benchmark.svg").write_text("<svg></svg>\n")
    (artifact_dir / "cuda-benchmark-ratios.svg").write_text("<svg></svg>\n")
    (artifact_dir / "cuda-benchmark-dag-deltas.svg").write_text("<svg></svg>\n")
    (artifact_dir / "cuda-benchmark-throughput.svg").write_text("<svg></svg>\n")

    errors = cuda_validate_capture.validate_capture(
        payload,
        artifact_dir=artifact_dir,
        required_machines=["hina", "dasys-h200x8"],
        required_baselines=[
            "direct_driver",
            "direct_driver_graph",
            "pto_host_schedule",
            "pto_host_schedule_compiler",
        ],
        required_sizes=[1024, 65536],
        expected_repeats=2,
        require_report_files=True,
    )

    assert errors == []


def test_cuda_capture_validator_reports_missing_baseline_and_artifact(tmp_path):
    cuda_validate_capture = _load_capture_validator_module()
    artifact_dir = tmp_path / "combined-current-abc123"
    artifact_dir.mkdir()
    payload = _paired_capture_payload()
    payload["results"] = [row for row in payload["results"] if row["baseline"] != "direct_driver_graph"]

    errors = cuda_validate_capture.validate_capture(
        payload,
        artifact_dir=artifact_dir,
        required_machines=["hina", "dasys-h200x8"],
        required_baselines=[
            "direct_driver",
            "direct_driver_graph",
            "pto_host_schedule",
            "pto_host_schedule_compiler",
        ],
        required_sizes=[1024, 65536],
        expected_repeats=2,
        require_report_files=True,
    )

    assert "missing baseline direct_driver_graph on dasys-h200x8" in errors
    assert "missing baseline direct_driver_graph on hina" in errors
    assert "missing report file cuda-benchmark.md" in errors
    assert "missing report file cuda-benchmark.svg" in errors
    assert "missing report file cuda-benchmark-ratios.svg" in errors
    assert "missing report file cuda-benchmark-dag-deltas.svg" in errors
    assert "missing report file cuda-benchmark-throughput.svg" in errors


def test_cuda_capture_validator_requires_source_papers(tmp_path):
    cuda_validate_capture = _load_capture_validator_module()
    payload = _paired_capture_payload()
    source_root = tmp_path / "repo"
    source_dir = source_root / "tmp" / "sources"
    source_dir.mkdir(parents=True)

    errors = cuda_validate_capture.validate_capture(payload, source_paper_root=source_root)

    assert "missing metadata.paper_setup" in errors
    assert "missing metadata.source_papers arXiv:2605.03190" in errors
    assert "missing metadata.source_papers arXiv:2512.22219v1" in errors

    payload["metadata"]["paper_setup"] = "paired A100/H200 microbenchmark"
    payload["metadata"]["source_papers"] = [
        {
            "id": "arXiv:2605.03190",
            "label": "VDCores",
            "path": "/tmp/arxiv-2605.03190-vdcores.txt",
        },
        {
            "id": "arXiv:2512.22219v1",
            "label": "MPK persistent kernel",
            "path": "tmp/sources/arxiv-2512.22219v1-mirage-persistent-kernel.txt",
        },
    ]

    errors = cuda_validate_capture.validate_capture(payload, source_paper_root=source_root)

    assert "metadata.source_papers arXiv:2605.03190 path must stay under tmp/sources/" in errors

    payload["metadata"]["source_papers"][0]["path"] = "tmp/sources/arxiv-2605.03190-vdcores.txt"
    errors = cuda_validate_capture.validate_capture(payload, source_paper_root=source_root)

    assert ("missing metadata.source_papers arXiv:2605.03190 file tmp/sources/arxiv-2605.03190-vdcores.txt") in errors
    assert (
        "missing metadata.source_papers arXiv:2512.22219v1 file "
        "tmp/sources/arxiv-2512.22219v1-mirage-persistent-kernel.txt"
    ) in errors

    (source_dir / "arxiv-2605.03190-vdcores.txt").write_text("vdcores\n")
    (source_dir / "arxiv-2512.22219v1-mirage-persistent-kernel.txt").write_text("mpk\n")

    assert cuda_validate_capture.validate_capture(payload, source_paper_root=source_root) == []


def test_cuda_capture_validator_requires_sanitized_command_examples():
    cuda_validate_capture = _load_capture_validator_module()
    payload = _paired_capture_payload()

    errors = cuda_validate_capture.validate_capture(payload, require_command_examples=True)

    assert "missing metadata.command_examples.local_sample" in errors
    assert "missing metadata.command_examples.remote_sample" in errors

    payload["metadata"]["command_examples"] = {
        "local_sample": f"env PYTHONPATH={Path.cwd()}:{Path.cwd() / 'python'} cuda_benchmark.py",
        "remote_sample": "python3 cuda_benchmark.py",
    }

    errors = cuda_validate_capture.validate_capture(payload, require_command_examples=True)

    assert "metadata.command_examples.local_sample contains local checkout path" in errors
    assert "metadata.command_examples.remote_sample must use ssh" in errors

    payload["metadata"]["command_examples"] = {
        "local_sample": "env PYTHONPATH=$PWD:$PWD/python .venv/bin/python cuda_benchmark.py",
        "remote_sample": "ssh bizhaoh200 'cd /remote/pto-cu && cuda_benchmark.py'",
    }

    assert cuda_validate_capture.validate_capture(payload, require_command_examples=True) == []


def test_cuda_capture_validator_requires_zero_scheduler_errors():
    cuda_validate_capture = _load_capture_validator_module()
    payload = _paired_capture_payload()
    payload["results"].append(
        {
            "machine": "hina",
            "baseline": "pto_persistent_dag",
            "n": 1024,
            "repeat": 0,
            "status": "pass",
            "device_wall_ns": 1024,
            "device_scheduler_errors": {"count": 1, "code": 7, "task_id": 2},
        }
    )

    errors = cuda_validate_capture.validate_capture(
        payload,
        require_zero_scheduler_errors=True,
    )

    assert "scheduler error machine=hina baseline=pto_persistent_dag n=1024 count=1 code=7 task_id=2" in errors

    payload["results"][-1]["device_scheduler_errors"] = {"count": 0, "code": 0, "task_id": 0}

    assert cuda_validate_capture.validate_capture(payload, require_zero_scheduler_errors=True) == []


def test_cuda_capture_validator_requires_dispatch_sequence():
    cuda_validate_capture = _load_capture_validator_module()
    payload = _paired_capture_payload()
    payload["results"].append(
        {
            "machine": "hina",
            "baseline": "pto_persistent_dag_graph_diamond",
            "n": 1024,
            "repeat": 0,
            "status": "pass",
            "device_wall_ns": 1024,
            "dispatch_func_ids": [9, 2, 1],
        }
    )

    errors = cuda_validate_capture.validate_capture(
        payload,
        required_dispatch={"pto_persistent_dag_graph_diamond": "9,2,1,2,1"},
    )

    assert (
        "expected dispatch 9,2,1,2,1 for machine=hina baseline=pto_persistent_dag_graph_diamond n=1024, found 9,2,1"
    ) in errors

    payload["results"][-1]["dispatch_func_ids"] = [9, 2, 1, 2, 1]

    assert (
        cuda_validate_capture.validate_capture(
            payload,
            required_dispatch={"pto_persistent_dag_graph_diamond": "9,2,1,2,1"},
        )
        == []
    )


def test_cuda_capture_validator_requires_graph_descriptor_metadata():
    cuda_validate_capture = _load_capture_validator_module()
    payload = _paired_capture_payload()
    payload["results"].append(
        {
            "machine": "hina",
            "baseline": "pto_persistent_dag_graph_diamond",
            "n": 1024,
            "repeat": 0,
            "status": "pass",
            "device_wall_ns": 1024,
            "dispatch_func_ids": [9, 2, 1, 2, 1],
            "graph_descriptor": {
                "fanin": [0, 0, 2, 1, 2],
                "dependents": [2, 3, 2, 4, 4],
            },
        }
    )

    errors = cuda_validate_capture.validate_capture(
        payload,
        required_graph_fanin={"pto_persistent_dag_graph_diamond": "0,0,2,2,2"},
        required_graph_dependents={"pto_persistent_dag_graph_diamond": "2,3,2,3,4,4"},
    )

    assert (
        "expected graph_descriptor.fanin 0,0,2,2,2 for machine=hina "
        "baseline=pto_persistent_dag_graph_diamond n=1024, found 0,0,2,1,2"
    ) in errors
    assert (
        "expected graph_descriptor.dependents 2,3,2,3,4,4 for machine=hina "
        "baseline=pto_persistent_dag_graph_diamond n=1024, found 2,3,2,4,4"
    ) in errors

    payload["results"][-1]["graph_descriptor"]["fanin"] = [0, 0, 2, 2, 2]
    payload["results"][-1]["graph_descriptor"]["dependents"] = [2, 3, 2, 3, 4, 4]

    assert (
        cuda_validate_capture.validate_capture(
            payload,
            required_graph_fanin={"pto_persistent_dag_graph_diamond": "0,0,2,2,2"},
            required_graph_dependents={"pto_persistent_dag_graph_diamond": "2,3,2,3,4,4"},
        )
        == []
    )


def test_cuda_capture_validator_requires_tensor_tile_shape():
    cuda_validate_capture = _load_capture_validator_module()
    payload = _paired_capture_payload()
    payload["results"].append(
        {
            "machine": "hina",
            "baseline": "pto_persistent_dag_tensor_core",
            "n": 1024,
            "repeat": 0,
            "status": "pass",
            "device_wall_ns": 1024,
            "tensor_tile": {"rows": 16, "cols": 16, "inner": 32},
        }
    )

    errors = cuda_validate_capture.validate_capture(
        payload,
        required_tensor_tiles={"pto_persistent_dag_tensor_core": "16x16x16"},
    )

    assert (
        "expected tensor tile 16x16x16 for machine=hina baseline=pto_persistent_dag_tensor_core n=1024, found 16x16x32"
    ) in errors

    payload["results"][-1]["tensor_tile"]["inner"] = 16

    assert (
        cuda_validate_capture.validate_capture(
            payload,
            required_tensor_tiles={"pto_persistent_dag_tensor_core": "16x16x16"},
        )
        == []
    )


def test_cuda_capture_validator_requires_scratch_reuse_metadata():
    cuda_validate_capture = _load_capture_validator_module()
    payload = _paired_capture_payload()
    payload["results"].append(
        {
            "machine": "hina",
            "baseline": "pto_persistent_dag_graph_scratch_reuse",
            "n": 1024,
            "repeat": 0,
            "status": "pass",
            "device_wall_ns": 1024,
            "scratch_reuse": {"reused_buffer": "tmp1", "reuse_task": 4},
        }
    )

    errors = cuda_validate_capture.validate_capture(
        payload,
        required_scratch_reuse={"pto_persistent_dag_graph_scratch_reuse": "reused_buffer=tmp0,reuse_task=4"},
    )

    assert (
        "expected scratch_reuse reused_buffer=tmp0,reuse_task=4 for machine=hina "
        "baseline=pto_persistent_dag_graph_scratch_reuse n=1024, found reused_buffer=tmp1,reuse_task=4"
    ) in errors

    payload["results"][-1]["scratch_reuse"]["reused_buffer"] = "tmp0"

    assert (
        cuda_validate_capture.validate_capture(
            payload,
            required_scratch_reuse={"pto_persistent_dag_graph_scratch_reuse": "reused_buffer=tmp0,reuse_task=4"},
        )
        == []
    )


def test_cuda_capture_validator_requires_graph_task_args_metadata():
    cuda_validate_capture = _load_capture_validator_module()
    payload = _paired_capture_payload()
    payload["results"].append(
        {
            "machine": "hina",
            "baseline": "pto_persistent_dag_graph_tagged_inout",
            "n": 1024,
            "repeat": 0,
            "status": "pass",
            "device_wall_ns": 1024,
            "graph_task_args": {
                "task0": "input:a,input:b,output:tmp1",
                "task1": "input:tmp1,input:b,output:tmp2",
                "task2": "input:tmp1,input:a,output_existing:out",
            },
        }
    )

    expected = "task0=input:a,input:b,output:tmp1;task1=inout:tmp1,input:b;task2=input:tmp1,input:a,output_existing:out"
    errors = cuda_validate_capture.validate_capture(
        payload,
        required_graph_task_args={"pto_persistent_dag_graph_tagged_inout": expected},
    )

    assert (
        "expected graph_task_args "
        "task0=input:a,input:b,output:tmp1;task1=inout:tmp1,input:b;"
        "task2=input:tmp1,input:a,output_existing:out for machine=hina "
        "baseline=pto_persistent_dag_graph_tagged_inout n=1024, found "
        "task0=input:a,input:b,output:tmp1;task1=input:tmp1,input:b,output:tmp2;"
        "task2=input:tmp1,input:a,output_existing:out"
    ) in errors

    payload["results"][-1]["graph_task_args"]["task1"] = "inout:tmp1,input:b"

    assert (
        cuda_validate_capture.validate_capture(
            payload,
            required_graph_task_args={"pto_persistent_dag_graph_tagged_inout": expected},
        )
        == []
    )


def test_cuda_capture_validator_paired_current_requires_generic_args_baseline():
    cuda_validate_capture = _load_capture_validator_module()
    args = cuda_validate_capture.parse_args(["capture.json", "--preset", "paired-current"])

    cuda_validate_capture._apply_preset(args)

    assert "pto_host_schedule_quad" in args.require_baseline
    assert "pto_host_schedule_generic_args" in args.require_baseline
    assert "pto_persistent_dag_quad" in args.require_baseline
    assert "pto_persistent_dag_generic_args" in args.require_baseline
    assert "pto_persistent_dag_graph" in args.require_baseline
    assert "pto_persistent_dag_graph_generic_args4" in args.require_baseline
    assert "pto_persistent_dag_graph_chain" in args.require_baseline
    assert "pto_persistent_dag_graph_scratch_reuse" in args.require_baseline
    assert "pto_persistent_dag_graph_tagged" in args.require_baseline
    assert "pto_persistent_dag_graph_tagged_inout" in args.require_baseline
    assert "pto_persistent_dag_graph_triad" in args.require_baseline
    assert "pto_persistent_dag_graph_quad" in args.require_baseline
    assert "pto_persistent_dag_graph_unary_square" in args.require_baseline
    assert "pto_persistent_dag_graph_tensor" in args.require_baseline
    assert "pto_persistent_dag_tensor_core" in args.require_baseline
    assert "pto_persistent_dag_graph_tensor_core" in args.require_baseline
    assert "cublas_sgemm" in args.require_baseline
    assert "cublas_sgemm_graph" in args.require_baseline
    assert "pto_persistent_dag_graph_generic_args4=9,2,1" in args.require_dispatch
    assert "pto_persistent_dag_graph_chain=1,2,1,2,1" in args.require_dispatch
    assert "pto_persistent_dag_graph_scratch_reuse=1,2,1,2,1,1" in args.require_dispatch
    assert "pto_persistent_dag_graph_scratch_reuse=reused_buffer=tmp0,reuse_task=4" in args.require_scratch_reuse
    assert "pto_persistent_dag_graph_tagged=9,2,1" in args.require_dispatch
    assert "pto_persistent_dag_graph_tagged_inout=1,1,1" in args.require_dispatch
    assert "pto_persistent_dag_graph_triad=6,2,1" in args.require_dispatch
    assert "pto_persistent_dag_graph_quad=8,2,1" in args.require_dispatch
    assert "pto_persistent_dag_graph_unary_square=7,1,1" in args.require_dispatch
    assert "pto_persistent_dag_graph_triad=0,0,2" in args.require_graph_fanin
    assert "pto_persistent_dag_graph_quad=0,0,2" in args.require_graph_fanin
    assert "pto_persistent_dag_graph_triad=2,2" in args.require_graph_dependents
    assert "pto_persistent_dag_graph_quad=2,2" in args.require_graph_dependents
    assert "pto_persistent_dag_graph_unary_square=0,1,1" in args.require_graph_fanin
    assert "pto_persistent_dag_graph_unary_square=1,2" in args.require_graph_dependents
    assert "pto_persistent_dag_graph_diamond=0,0,2,2,2" in args.require_graph_fanin
    assert "pto_persistent_dag_graph_diamond=2,3,2,3,4,4" in args.require_graph_dependents
    assert (
        "pto_persistent_dag_graph_tagged="
        "task0=input:a,input:b,output:tmp1,scalar:scalar_args[0],scalar:scalar_args[1];"
        "task1=input:a,input:b,output:tmp2;task2=input:tmp1,input:tmp2,output_existing:out"
    ) in args.require_graph_task_args
    assert (
        "pto_persistent_dag_graph_tagged_inout="
        "task0=input:a,input:b,output:tmp1;task1=inout:tmp1,input:b;"
        "task2=input:tmp1,input:a,output_existing:out"
    ) in args.require_graph_task_args
    assert "pto_persistent_dag_graph_diamond=9,2,1,2,1" in args.require_dispatch
    assert "pto_persistent_dag_tensor_core=10,1,2,1" in args.require_dispatch
    assert "pto_persistent_dag_graph_tensor_core=10,1,2,1" in args.require_dispatch
    assert "pto_persistent_dag_tensor=16x16x16" in args.require_tensor_tile
    assert "pto_persistent_dag_graph_tensor_core=16x16x16" in args.require_tensor_tile
    assert "pto_persistent_dag_graph_tensor_core=0,1,1,2" in args.require_graph_fanin
    assert "pto_persistent_dag_graph_tensor_core=1,2,3,3" in args.require_graph_dependents
    assert "cublas_sgemm=16x16x16" in args.require_tensor_tile
    assert "cublas_sgemm_graph=16x16x16" in args.require_tensor_tile
    assert args.expected_result_count == 1008


def test_cuda_capture_validator_compact_current_preset_matches_docs_gate():
    cuda_validate_capture = _load_capture_validator_module()
    args = cuda_validate_capture.parse_args(["capture.json", "--preset", "compact-current"])

    cuda_validate_capture._apply_preset(args)

    assert args.require_machine == ["hina", "dasys-h200x8"]
    assert args.require_size == ["1024"]
    assert args.expected_repeats == 1
    assert args.expected_result_count == 78
    assert args.require_report_files is True
    assert args.require_command_examples is True
    assert args.require_zero_scheduler_errors is True
    assert args.require_source_papers is True
    assert "pto_host_schedule_generic_args" in args.require_baseline
    assert "pto_persistent_dag_scalar_scale" in args.require_baseline
    assert "pto_persistent_dag_graph_generic_args4" in args.require_baseline
    assert "pto_persistent_dag_graph_chain" in args.require_baseline
    assert "pto_persistent_dag_graph_scratch_reuse" in args.require_baseline
    assert "pto_persistent_dag_graph_diamond" in args.require_baseline
    assert "pto_persistent_dag_graph_tagged" in args.require_baseline
    assert "pto_persistent_dag_graph_tagged_inout" in args.require_baseline
    assert "pto_persistent_dag_graph_triad" in args.require_baseline
    assert "pto_persistent_dag_graph_quad" in args.require_baseline
    assert "pto_persistent_dag_graph_unary_square" in args.require_baseline
    assert "pto_persistent_dag_graph_tensor" in args.require_baseline
    assert "pto_persistent_dag_graph_tensor_core" in args.require_baseline
    assert "cublas_sgemm_graph" in args.require_baseline
    assert "pto_persistent_dag_graph_generic_args4=9,2,1" in args.require_dispatch
    assert "pto_persistent_dag_graph_chain=1,2,1,2,1" in args.require_dispatch
    assert "pto_persistent_dag_graph_scratch_reuse=1,2,1,2,1,1" in args.require_dispatch
    assert "pto_persistent_dag_graph_scratch_reuse=reused_buffer=tmp0,reuse_task=4" in args.require_scratch_reuse
    assert "pto_persistent_dag_graph_tagged=9,2,1" in args.require_dispatch
    assert "pto_persistent_dag_graph_tagged_inout=1,1,1" in args.require_dispatch
    assert "pto_persistent_dag_graph_unary_square=7,1,1" in args.require_dispatch
    assert "pto_persistent_dag_graph_diamond=0,0,2,2,2" in args.require_graph_fanin
    assert "pto_persistent_dag_graph_diamond=2,3,2,3,4,4" in args.require_graph_dependents
    assert "pto_persistent_dag_graph_unary_square=0,1,1" in args.require_graph_fanin
    assert "pto_persistent_dag_graph_unary_square=1,2" in args.require_graph_dependents
    assert (
        "pto_persistent_dag_graph_tagged="
        "task0=input:a,input:b,output:tmp1,scalar:scalar_args[0],scalar:scalar_args[1];"
        "task1=input:a,input:b,output:tmp2;task2=input:tmp1,input:tmp2,output_existing:out"
    ) in args.require_graph_task_args
    assert (
        "pto_persistent_dag_graph_tagged_inout="
        "task0=input:a,input:b,output:tmp1;task1=inout:tmp1,input:b;"
        "task2=input:tmp1,input:a,output_existing:out"
    ) in args.require_graph_task_args
    assert "pto_persistent_dag_graph_tensor=3,1,2,1" in args.require_dispatch
    assert "pto_persistent_dag_tensor_core=10,1,2,1" in args.require_dispatch
    assert "pto_persistent_dag_graph_tensor_core=10,1,2,1" in args.require_dispatch
    assert "pto_persistent_dag_graph_tensor=16x16x16" in args.require_tensor_tile
    assert "pto_persistent_dag_graph_tensor_core=16x16x16" in args.require_tensor_tile
    assert "pto_persistent_dag_graph_tensor_core=0,1,1,2" in args.require_graph_fanin
    assert "pto_persistent_dag_graph_tensor_core=1,2,3,3" in args.require_graph_dependents
    assert "cublas_sgemm=16x16x16" in args.require_tensor_tile
    assert "cublas_sgemm_graph=16x16x16" in args.require_tensor_tile


def _tensor_sweep_payload():
    results = []
    baselines = {
        "pto_persistent_dag_tensor": [3, 1, 2, 1],
        "pto_persistent_dag_tensor_core": [10, 1, 2, 1],
        "cublas_sgemm": [],
    }
    for artifact in ("a100", "h200"):
        for baseline, dispatch in baselines.items():
            for shape in ("16x16x16", "16x16x64"):
                rows, cols, inner = (int(part) for part in shape.split("x"))
                for repeat in range(2):
                    results.append(
                        {
                            "artifact": artifact,
                            "machine": artifact,
                            "baseline": baseline,
                            "n": 256,
                            "shape": shape,
                            "repeat": repeat,
                            "status": "pass",
                            "device_wall_ns": 1000,
                            "host_wall_ns": 1500,
                            "dispatch_func_ids": dispatch,
                            "tensor_tile": {"rows": rows, "cols": cols, "inner": inner, "tile_count": 1},
                        }
                    )
    return {
        "metadata": {
            "label": "tensor-shape-sweep-abc123",
            "git_commit": "abc123",
            "n": 256,
            "repeats": 2,
            "baselines": list(baselines),
            "shapes": ["16x16x16", "16x16x64"],
        },
        "results": results,
    }


def test_cuda_tensor_sweep_validator_accepts_complete_artifact(tmp_path):
    cuda_validate_tensor_sweep = _load_tensor_sweep_validator_module()
    artifact_dir = tmp_path / "tensor-shape-sweep-abc123"
    artifact_dir.mkdir()
    payload = _tensor_sweep_payload()
    (artifact_dir / "cuda-tensor-shape-sweep.json").write_text(json.dumps(payload) + "\n")
    (artifact_dir / "cuda-tensor-shape-sweep.md").write_text("# report\n")
    (artifact_dir / "cuda-tensor-shape-sweep.svg").write_text("<svg></svg>\n")
    (artifact_dir / "cuda-tensor-shape-throughput.svg").write_text("<svg></svg>\n")

    errors = cuda_validate_tensor_sweep.validate_tensor_sweep(
        payload,
        artifact_dir=artifact_dir,
        required_artifacts=["a100", "h200"],
        required_baselines=["pto_persistent_dag_tensor", "pto_persistent_dag_tensor_core", "cublas_sgemm"],
        required_sizes=[256],
        required_shapes=["16x16x16", "16x16x64"],
        expected_repeats=2,
        expected_result_count=24,
        required_dispatch={
            "pto_persistent_dag_tensor": "3,1,2,1",
            "pto_persistent_dag_tensor_core": "10,1,2,1",
        },
        require_report_files=True,
    )

    assert errors == []


def test_cuda_tensor_sweep_validator_reports_missing_rows_and_metadata(tmp_path):
    cuda_validate_tensor_sweep = _load_tensor_sweep_validator_module()
    artifact_dir = tmp_path / "tensor-shape-sweep-abc123"
    artifact_dir.mkdir()
    payload = _tensor_sweep_payload()
    payload["results"] = [
        row
        for row in payload["results"]
        if not (
            row["artifact"] == "h200"
            and row["baseline"] == "pto_persistent_dag_tensor_core"
            and row["shape"] == "16x16x64"
        )
    ]
    payload["results"][0]["status"] = "fail"
    payload["results"][1]["tensor_tile"]["inner"] = 32
    payload["results"][2]["dispatch_func_ids"] = [9]

    errors = cuda_validate_tensor_sweep.validate_tensor_sweep(
        payload,
        artifact_dir=artifact_dir,
        required_artifacts=["a100", "h200"],
        required_baselines=["pto_persistent_dag_tensor", "pto_persistent_dag_tensor_core", "cublas_sgemm"],
        required_sizes=[256],
        required_shapes=["16x16x16", "16x16x64"],
        expected_repeats=2,
        expected_result_count=24,
        required_dispatch={"pto_persistent_dag_tensor": "3,1,2,1"},
        require_report_files=True,
    )

    assert "expected 24 results, found 22" in errors
    assert "non-pass row artifact=a100 baseline=pto_persistent_dag_tensor n=256 shape=16x16x16" in errors
    assert (
        "expected tensor tile 16x16x16 for artifact=a100 baseline=pto_persistent_dag_tensor n=256, found 16x16x32"
    ) in errors
    assert "expected dispatch 3,1,2,1 for artifact=a100 baseline=pto_persistent_dag_tensor n=256, found 9" in errors
    assert "missing baseline pto_persistent_dag_tensor_core artifact=h200 n=256 shape=16x16x64" in errors
    assert "missing report file cuda-tensor-shape-sweep.md" in errors
    assert "missing report file cuda-tensor-shape-sweep.svg" in errors
    assert "missing report file cuda-tensor-shape-throughput.svg" in errors


def test_cuda_tensor_sweep_validator_requires_sanitized_command_examples():
    cuda_validate_tensor_sweep = _load_tensor_sweep_validator_module()
    payload = _tensor_sweep_payload()

    errors = cuda_validate_tensor_sweep.validate_tensor_sweep(
        payload,
        require_command_examples=True,
    )

    assert "missing metadata.command_examples.local_sample" in errors
    assert "missing metadata.command_examples.remote_sample" in errors

    payload["metadata"]["command_examples"] = {
        "local_sample": (f"env PYTHONPATH={Path.cwd()}:{Path.cwd() / 'python'} cuda_benchmark.py"),
        "remote_sample": "python3 cuda_benchmark.py",
    }

    errors = cuda_validate_tensor_sweep.validate_tensor_sweep(
        payload,
        require_command_examples=True,
    )

    assert "metadata.command_examples.local_sample contains local checkout path" in errors
    assert "metadata.command_examples.remote_sample must use ssh" in errors

    payload["metadata"]["command_examples"] = {
        "local_sample": "env PYTHONPATH=$PWD:$PWD/python .venv/bin/python cuda_benchmark.py",
        "remote_sample": "ssh bizhaoh200 'cd /remote/pto-cu && cuda_benchmark.py'",
    }

    assert (
        cuda_validate_tensor_sweep.validate_tensor_sweep(
            payload,
            require_command_examples=True,
        )
        == []
    )


def test_cuda_tensor_sweep_validator_requires_source_papers(tmp_path):
    cuda_validate_tensor_sweep = _load_tensor_sweep_validator_module()
    payload = _tensor_sweep_payload()
    source_root = tmp_path / "repo"
    source_dir = source_root / "tmp" / "sources"
    source_dir.mkdir(parents=True)

    errors = cuda_validate_tensor_sweep.validate_tensor_sweep(
        payload,
        require_source_papers=True,
        source_root=source_root,
    )

    assert "missing metadata.paper_setup" in errors
    assert "missing metadata.source_papers arXiv:2605.03190" in errors
    assert "missing metadata.source_papers arXiv:2512.22219v1" in errors

    payload["metadata"]["paper_setup"] = "paired A100/H200 tensor sweep"
    payload["metadata"]["source_papers"] = [
        {
            "id": "arXiv:2605.03190",
            "label": "VDCores",
            "path": "/home/user/private.pdf",
        },
        {
            "id": "arXiv:2512.22219v1",
            "label": "MPK persistent kernel",
            "path": "tmp/sources/arxiv-2512.22219v1-mirage-persistent-kernel.txt",
        },
    ]

    errors = cuda_validate_tensor_sweep.validate_tensor_sweep(
        payload,
        require_source_papers=True,
        source_root=source_root,
    )

    assert "metadata.source_papers arXiv:2605.03190 path must stay under tmp/sources/" in errors

    payload["metadata"]["source_papers"][0]["path"] = "tmp/sources/arxiv-2605.03190-vdcores.txt"

    errors = cuda_validate_tensor_sweep.validate_tensor_sweep(
        payload,
        require_source_papers=True,
        source_root=source_root,
    )

    assert ("missing metadata.source_papers arXiv:2605.03190 file tmp/sources/arxiv-2605.03190-vdcores.txt") in errors
    assert (
        "missing metadata.source_papers arXiv:2512.22219v1 file "
        "tmp/sources/arxiv-2512.22219v1-mirage-persistent-kernel.txt"
    ) in errors

    (source_dir / "arxiv-2605.03190-vdcores.txt").write_text("vdcores\n")
    (source_dir / "arxiv-2512.22219v1-mirage-persistent-kernel.txt").write_text("mpk\n")

    assert (
        cuda_validate_tensor_sweep.validate_tensor_sweep(
            payload,
            require_source_papers=True,
            source_root=source_root,
        )
        == []
    )


def test_cuda_tensor_sweep_validator_requires_each_size():
    cuda_validate_tensor_sweep = _load_tensor_sweep_validator_module()
    payload = _tensor_sweep_payload()
    payload["results"] = [
        row
        for row in payload["results"]
        if not (row["artifact"] == "h200" and row["baseline"] == "cublas_sgemm" and row["shape"] == "16x16x64")
    ]

    errors = cuda_validate_tensor_sweep.validate_tensor_sweep(
        payload,
        required_artifacts=["a100", "h200"],
        required_baselines=["cublas_sgemm"],
        required_sizes=[256, 4096],
        required_shapes=["16x16x16", "16x16x64"],
        expected_repeats=2,
    )

    assert "missing baseline cublas_sgemm artifact=a100 n=4096 shape=16x16x16" in errors
    assert "missing baseline cublas_sgemm artifact=h200 n=256 shape=16x16x64" in errors


def test_cuda_tensor_sweep_validator_compact_preset_keeps_dispatch_commas():
    cuda_validate_tensor_sweep = _load_tensor_sweep_validator_module()
    args = cuda_validate_tensor_sweep.parse_args(["sweep.json", "--preset", "compact-tensor-baselines"])

    cuda_validate_tensor_sweep._apply_preset(args)
    required_dispatch = cuda_validate_tensor_sweep._parse_required_dispatch(args.require_dispatch)

    assert args.expected_repeats == 3
    assert args.expected_result_count == 72
    assert "cublas_sgemm_graph" in args.require_baseline
    assert "pto_persistent_dag_graph_tensor_core" in args.require_baseline
    assert required_dispatch == {
        "pto_persistent_dag_tensor": "3,1,2,1",
        "pto_persistent_dag_graph_tensor": "3,1,2,1",
        "pto_persistent_dag_tensor_core": "10,1,2,1",
        "pto_persistent_dag_graph_tensor_core": "10,1,2,1",
    }


def test_cuda_smoke_validator_accepts_paired_lifecycle_artifact(tmp_path):
    cuda_validate_smoke = _load_smoke_validator_module()
    artifact_dir = tmp_path / "persistent-graph-smoke"
    artifact_dir.mkdir()
    payload = {
        "status": "pass",
        "runtime": "persistent_device",
        "mode": "dag",
        "dag_shape": "graph_descriptor",
        "n": 1024,
        "repeat_runs": 2,
        "launch_completed_counts": [3, 3],
        "dispatch_func_ids": [9, 2, 1],
        "device_scheduler_errors": {"count": 0, "code": 0, "task_id": 0},
    }
    (artifact_dir / "a100.json").write_text(json.dumps(payload) + "\n")
    (artifact_dir / "h200.json").write_text(json.dumps(payload) + "\n")
    (artifact_dir / "cuda-smoke-report.md").write_text("# report\n")
    (artifact_dir / "cuda-smoke-report.svg").write_text("<svg></svg>\n")

    payloads = cuda_validate_smoke.load_smoke_payloads([artifact_dir / "a100.json", artifact_dir / "h200.json"])
    errors = cuda_validate_smoke.validate_smoke(
        payloads,
        expectation=cuda_validate_smoke.SmokeValidationExpectation(
            artifact_dir=artifact_dir,
            required_artifacts=["a100", "h200"],
            runtime="persistent_device",
            mode="dag",
            dag_shape="graph_descriptor",
            repeat_runs=2,
            completed_count=3,
            dispatch="9,2,1",
            require_report_files=True,
        ),
    )

    assert errors == []


def test_cuda_smoke_validator_checks_tensor_tile_shape(tmp_path):
    cuda_validate_smoke = _load_smoke_validator_module()
    artifact_dir = tmp_path / "persistent-tensor-tile-smoke"
    artifact_dir.mkdir()
    payload = {
        "status": "pass",
        "runtime": "persistent_device",
        "mode": "dag",
        "dag_shape": "tensor_tile",
        "n": 768,
        "repeat_runs": 1,
        "launch_completed_counts": [4],
        "dispatch_func_ids": [3, 1, 2, 1],
        "device_scheduler_errors": {"count": 0, "code": 0, "task_id": 0},
        "tensor_tile": {"rows": 8, "cols": 4, "inner": 16, "tile_count": 24},
    }
    (artifact_dir / "a100.json").write_text(json.dumps(payload) + "\n")

    payloads = cuda_validate_smoke.load_smoke_payloads([artifact_dir / "a100.json"])
    errors = cuda_validate_smoke.validate_smoke(
        payloads,
        expectation=cuda_validate_smoke.SmokeValidationExpectation(
            runtime="persistent_device",
            mode="dag",
            dag_shape="tensor_tile",
            completed_count=4,
            tensor_tile="8x4x12",
        ),
    )

    assert "expected tensor tile 8x4x12 for artifact=a100, found 8x4x16" in errors


def test_cuda_smoke_validator_checks_resource_policy(tmp_path):
    cuda_validate_smoke = _load_smoke_validator_module()
    artifact_dir = tmp_path / "persistent-resource-policy-smoke"
    artifact_dir.mkdir()
    payload = {
        "status": "pass",
        "runtime": "persistent_device",
        "mode": "dag",
        "dag_shape": "chain",
        "n": 1024,
        "repeat_runs": 1,
        "launch_completed_counts": [5],
        "dispatch_func_ids": [1, 2, 1, 2, 1],
        "device_scheduler_errors": {"count": 0, "code": 0, "task_id": 0},
        "resource_policy": {
            "scheduler_blocks": 1,
            "worker_blocks": 2,
            "worker_blocks_per_task": 1,
            "stream_id": 1,
            "block_dim": 256,
            "grid_dim": 3,
        },
    }
    (artifact_dir / "a100.json").write_text(json.dumps(payload) + "\n")

    payloads = cuda_validate_smoke.load_smoke_payloads([artifact_dir / "a100.json"])
    errors = cuda_validate_smoke.validate_smoke(
        payloads,
        expectation=cuda_validate_smoke.SmokeValidationExpectation(
            resource_policy=cuda_validate_smoke.ResourcePolicyExpectation(
                scheduler_blocks=1,
                worker_blocks=4,
                worker_blocks_per_task=1,
                stream_id=0,
                block_dim=256,
                grid_dim=3,
            ),
        ),
    )

    assert "expected resource_policy.worker_blocks 4 for artifact=a100, found 2" in errors
    assert "expected resource_policy.stream_id 0 for artifact=a100, found 1" in errors
    assert "expected resource_policy.scheduler_blocks 1 for artifact=a100" not in errors


def test_cuda_smoke_validator_checks_graph_descriptor_metadata(tmp_path):
    cuda_validate_smoke = _load_smoke_validator_module()
    artifact_dir = tmp_path / "persistent-graph-descriptor-smoke"
    artifact_dir.mkdir()
    payload = {
        "status": "pass",
        "runtime": "persistent_device",
        "mode": "dag",
        "dag_shape": "graph_descriptor_diamond",
        "n": 1024,
        "repeat_runs": 2,
        "launch_completed_counts": [5, 5],
        "dispatch_func_ids": [9, 2, 1, 2, 1],
        "device_scheduler_errors": {"count": 0, "code": 0, "task_id": 0},
        "graph_descriptor": {
            "tasks": 5,
            "fanin": [0, 0, 2, 2, 2],
            "dependents": [2, 3, 2, 3, 4, 4],
        },
    }
    (artifact_dir / "a100.json").write_text(json.dumps(payload) + "\n")

    payloads = cuda_validate_smoke.load_smoke_payloads([artifact_dir / "a100.json"])
    errors = cuda_validate_smoke.validate_smoke(
        payloads,
        expectation=cuda_validate_smoke.SmokeValidationExpectation(
            graph_fanin="0,0,2,1,2",
            graph_dependents="2,3,4",
        ),
    )

    assert "expected graph_descriptor.fanin 0,0,2,1,2 for artifact=a100, found 0,0,2,2,2" in errors
    assert "expected graph_descriptor.dependents 2,3,4 for artifact=a100, found 2,3,2,3,4,4" in errors


def test_cuda_smoke_validator_checks_graph_task_args_metadata(tmp_path):
    cuda_validate_smoke = _load_smoke_validator_module()
    artifact_dir = tmp_path / "persistent-graph-descriptor-tagged-smoke"
    artifact_dir.mkdir()
    payload = {
        "status": "pass",
        "runtime": "persistent_device",
        "mode": "dag",
        "dag_shape": "graph_descriptor_tagged",
        "n": 1024,
        "repeat_runs": 2,
        "launch_completed_counts": [3, 3],
        "dispatch_func_ids": [9, 2, 1],
        "device_scheduler_errors": {"count": 0, "code": 0, "task_id": 0},
        "graph_descriptor": {
            "tasks": 3,
            "fanin": [0, 0, 2],
            "dependents": [2, 2],
        },
        "graph_task_args": {
            "task0": "input:a,input:b,output:tmp1",
            "task1": "input:a,input:b,output:tmp2",
        },
    }
    (artifact_dir / "a100.json").write_text(json.dumps(payload) + "\n")

    payloads = cuda_validate_smoke.load_smoke_payloads([artifact_dir / "a100.json"])
    errors = cuda_validate_smoke.validate_smoke(
        payloads,
        expectation=cuda_validate_smoke.SmokeValidationExpectation(
            graph_task_args="task0=input:a,input:b,output:tmp1;task2=input:tmp1,input:tmp2,output_existing:out",
        ),
    )

    assert (
        "expected graph_task_args "
        "task0=input:a,input:b,output:tmp1;task2=input:tmp1,input:tmp2,output_existing:out "
        "for artifact=a100, found task0=input:a,input:b,output:tmp1;task1=input:a,input:b,output:tmp2"
    ) in errors


def test_cuda_smoke_validator_checks_scratch_reuse_metadata(tmp_path):
    cuda_validate_smoke = _load_smoke_validator_module()
    artifact_dir = tmp_path / "persistent-graph-descriptor-scratch-reuse-smoke"
    artifact_dir.mkdir()
    payload = {
        "status": "pass",
        "runtime": "persistent_device",
        "mode": "dag",
        "dag_shape": "graph_descriptor_scratch_reuse",
        "n": 1024,
        "repeat_runs": 2,
        "launch_completed_counts": [6, 6],
        "dispatch_func_ids": [1, 2, 1, 2, 1, 1],
        "device_scheduler_errors": {"count": 0, "code": 0, "task_id": 0},
        "scratch_reuse": {"reused_buffer": "tmp1", "reuse_task": 4},
    }
    (artifact_dir / "a100.json").write_text(json.dumps(payload) + "\n")

    payloads = cuda_validate_smoke.load_smoke_payloads([artifact_dir / "a100.json"])
    errors = cuda_validate_smoke.validate_smoke(
        payloads,
        expectation=cuda_validate_smoke.SmokeValidationExpectation(
            scratch_reuse="reused_buffer=tmp0,reuse_task=4",
        ),
    )

    assert (
        "expected scratch_reuse reused_buffer=tmp0,reuse_task=4 "
        "for artifact=a100, found reused_buffer=tmp1,reuse_task=4"
    ) in errors


def test_cuda_smoke_validator_reports_lifecycle_and_report_errors(tmp_path):
    cuda_validate_smoke = _load_smoke_validator_module()
    artifact_dir = tmp_path / "persistent-graph-smoke"
    artifact_dir.mkdir()
    (artifact_dir / "a100.json").write_text(
        json.dumps(
            {
                "status": "fail",
                "runtime": "persistent_device",
                "mode": "dag",
                "dag_shape": "graph_descriptor",
                "n": 1024,
                "repeat_runs": 2,
                "launch_completed_counts": [3, 2],
                "dispatch_func_ids": [9, 2, 1],
                "device_scheduler_errors": {"count": 1, "code": 7, "task_id": 0},
            }
        )
        + "\n"
    )

    payloads = cuda_validate_smoke.load_smoke_payloads([artifact_dir / "a100.json"])
    errors = cuda_validate_smoke.validate_smoke(
        payloads,
        expectation=cuda_validate_smoke.SmokeValidationExpectation(
            artifact_dir=artifact_dir,
            required_artifacts=["a100", "h200"],
            runtime="persistent_device",
            mode="dag",
            dag_shape="graph_descriptor",
            repeat_runs=2,
            completed_count=3,
            dispatch="9,2,1",
            require_report_files=True,
        ),
    )

    assert "missing artifact h200" in errors
    assert "non-pass artifact=a100 status=fail" in errors
    assert "scheduler error artifact=a100 count=1 code=7 task=0" in errors
    assert "expected completed count 3 for artifact=a100 launch=1, found 2" in errors
    assert "missing report file cuda-smoke-report.md" in errors
    assert "missing report file cuda-smoke-report.svg" in errors


def test_cuda_lifecycle_matrix_validator_accepts_default_matrix(tmp_path):
    cuda_validate_lifecycle = _load_lifecycle_matrix_validator_module()
    artifact_dir = tmp_path / "persistent-lifecycle-matrix-test"
    artifact_dir.mkdir()
    rows = [
        {
            "scenario": "direct",
            "artifact": "a100",
            "status": "pass",
            "runtime": "persistent_device",
            "mode": "direct",
            "n": 1024,
            "repeat_runs": 2,
            "launch_completed_counts": [2, 2],
            "completed_count": 2,
            "resource_policy": {
                "scheduler_blocks": 0,
                "worker_blocks": 4,
                "worker_blocks_per_task": 2,
                "stream_id": 1,
                "block_dim": 256,
                "grid_dim": 4,
            },
        },
        {
            "scenario": "queue",
            "artifact": "h200",
            "status": "pass",
            "runtime": "persistent_device",
            "mode": "queue",
            "n": 1024,
            "repeat_runs": 2,
            "launch_completed_counts": [4, 4],
            "completed_count": 4,
            "resource_policy": {
                "scheduler_blocks": 1,
                "worker_blocks": 2,
                "worker_blocks_per_task": 1,
                "stream_id": 1,
                "block_dim": 256,
                "grid_dim": 3,
            },
        },
        {
            "scenario": "dag-chain",
            "artifact": "a100",
            "status": "pass",
            "runtime": "persistent_device",
            "mode": "dag",
            "dag_shape": "chain",
            "n": 1024,
            "repeat_runs": 2,
            "launch_completed_counts": [5, 5],
            "completed_count": 5,
            "dispatch_func_ids": [1, 2, 1, 2, 1],
            "device_scheduler_errors": {"count": 0, "code": 0, "task_id": 0},
            "resource_policy": {
                "scheduler_blocks": 1,
                "worker_blocks": 2,
                "worker_blocks_per_task": 1,
                "stream_id": 1,
                "block_dim": 256,
                "grid_dim": 3,
            },
        },
        {
            "scenario": "graph-scratch-reuse",
            "artifact": "h200",
            "status": "pass",
            "runtime": "persistent_device",
            "mode": "dag",
            "dag_shape": "graph_descriptor_scratch_reuse",
            "n": 1024,
            "repeat_runs": 2,
            "launch_completed_counts": [6, 6],
            "completed_count": 6,
            "dispatch_func_ids": [1, 2, 1, 2, 1, 1],
            "device_scheduler_errors": {"count": 0, "code": 0, "task_id": 0},
            "resource_policy": {
                "scheduler_blocks": 1,
                "worker_blocks": 2,
                "worker_blocks_per_task": 1,
                "stream_id": 1,
                "block_dim": 256,
                "grid_dim": 3,
            },
        },
    ]
    (artifact_dir / "cuda-lifecycle-matrix.json").write_text(json.dumps({"label": "test", "rows": rows}) + "\n")
    (artifact_dir / "cuda-lifecycle-matrix.md").write_text("# matrix\n")
    (artifact_dir / "cuda-lifecycle-matrix.svg").write_text("<svg></svg>\n")

    errors = cuda_validate_lifecycle.validate_lifecycle_matrix(
        cuda_validate_lifecycle.load_lifecycle_matrix(artifact_dir / "cuda-lifecycle-matrix.json"),
        artifact_dir=artifact_dir,
        expected_repeat_runs=2,
        required_scenarios=["direct", "queue", "dag-chain", "graph-scratch-reuse"],
        require_artifacts=["a100", "h200"],
        require_report_files=True,
    )

    assert errors == []


def test_cuda_lifecycle_matrix_validator_reports_contract_errors(tmp_path):
    cuda_validate_lifecycle = _load_lifecycle_matrix_validator_module()
    artifact_dir = tmp_path / "persistent-lifecycle-matrix-test"
    artifact_dir.mkdir()
    rows = [
        {
            "scenario": "dag-chain",
            "artifact": "a100",
            "status": "fail",
            "runtime": "persistent_device",
            "mode": "dag",
            "dag_shape": "chain",
            "n": 1024,
            "repeat_runs": 2,
            "launch_completed_counts": [5, 4],
            "completed_count": 5,
            "dispatch_func_ids": [1, 2, 1],
            "device_scheduler_errors": {"count": 1, "code": 7, "task_id": 1},
        }
    ]
    payload = {"label": "test", "rows": rows}

    errors = cuda_validate_lifecycle.validate_lifecycle_matrix(
        payload,
        artifact_dir=artifact_dir,
        expected_repeat_runs=2,
        required_scenarios=["direct", "dag-chain"],
        require_artifacts=["a100", "h200"],
        require_report_files=True,
    )

    assert "missing scenario direct" in errors
    assert "missing artifact h200" in errors
    assert "missing report file cuda-lifecycle-matrix.md" in errors
    assert "non-pass scenario=dag-chain artifact=a100 status=fail" in errors
    assert "scheduler error scenario=dag-chain artifact=a100 count=1 code=7 task=1" in errors
    assert "expected dispatch 1,2,1,2,1 for scenario=dag-chain artifact=a100, found 1,2,1" in errors
    assert "expected completed count 5 for scenario=dag-chain artifact=a100 launch=1, found 4" in errors


def test_cuda_pair_benchmark_builds_current_a100_h200_workflow(tmp_path):
    cuda_pair_benchmark = _load_pair_benchmark_module()
    config = cuda_pair_benchmark.PairedBenchmarkConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        branch="design/nvidia-backend",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
    )

    local = cuda_pair_benchmark.build_local_benchmark_command(config, "abc123")
    remote = cuda_pair_benchmark.build_remote_benchmark_command(config, "abc123")
    scp = cuda_pair_benchmark.build_scp_command(config, "abc123")
    merge = cuda_pair_benchmark.build_merge_command(config, "abc123")
    validate = cuda_pair_benchmark.build_validate_command(config, "abc123")
    index = cuda_pair_benchmark.build_index_command(config)

    assert local[:2] == ["env", f"PYTHONPATH={Path.cwd()}:{Path.cwd() / 'python'}"]
    assert ".agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py" in local
    assert "--arch" in local
    assert "compute_80" in local
    assert "--tensor-rows" in local
    assert "16" in local
    assert "--tensor-cols" in local
    assert "--tensor-inner" in local
    assert "a100-current-abc123" in local
    assert str(tmp_path / "cuda-backend" / "a100-current-abc123") in local

    assert remote[:6] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "h200-box"]
    remote_shell = remote[-1]
    assert "cd /remote/pto-cu" in remote_shell
    assert (
        "timeout 60 git -c http.lowSpeedLimit=1 -c http.lowSpeedTime=30 "
        "fetch origin design/nvidia-backend" in remote_shell
    )
    assert "git checkout -B design/nvidia-backend FETCH_HEAD >/dev/null" in remote_shell
    assert "CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH PYTHONPATH=$PWD:$PWD/python" in remote_shell
    assert "--arch compute_90" in remote_shell
    assert "--tensor-rows 16" in remote_shell
    assert "--tensor-cols 16" in remote_shell
    assert "--tensor-inner 16" in remote_shell
    assert "h200-current-abc123" in remote_shell

    assert scp == [
        "scp",
        "-r",
        f"h200-box:/remote/pto-cu/{tmp_path / 'cuda-backend' / 'h200-current-abc123'}",
        str(tmp_path / "cuda-backend"),
    ]
    assert "--merge-json" in merge
    assert str(tmp_path / "cuda-backend" / "a100-current-abc123" / "cuda-benchmark.json") in merge
    assert str(tmp_path / "cuda-backend" / "h200-current-abc123" / "cuda-benchmark.json") in merge
    assert "combined-current-abc123" in merge
    assert ".agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py" in validate
    assert str(tmp_path / "cuda-backend" / "combined-current-abc123" / "cuda-benchmark.json") in validate
    assert "--expected-result-count" in validate
    assert "1008" in validate
    assert "--require-baseline" in validate
    assert "pto_host_schedule_generic_args" in validate
    assert "pto_persistent_dag_graph_generic_args4" in validate
    assert "pto_persistent_dag_graph_chain" in validate
    assert "pto_persistent_dag_graph_scratch_reuse" in validate
    assert "pto_persistent_dag_graph_diamond" in validate
    assert "pto_persistent_dag_graph_tagged" in validate
    assert "pto_persistent_dag_graph_tagged_inout" in validate
    assert "pto_persistent_dag_graph_triad" in validate
    assert "pto_persistent_dag_graph_quad" in validate
    assert "pto_persistent_dag_graph_unary_square" in validate
    assert "pto_persistent_dag_graph_tensor" in validate
    assert "pto_persistent_dag_tensor_core" in validate
    assert "pto_persistent_dag_graph_tensor_core" in validate
    assert "cublas_sgemm" in validate
    assert "cublas_sgemm_graph" in validate
    assert "--require-dispatch" in validate
    assert "pto_persistent_dag_graph_generic_args4=9,2,1" in validate
    assert "pto_persistent_dag_graph_chain=1,2,1,2,1" in validate
    assert "pto_persistent_dag_graph_scratch_reuse=1,2,1,2,1,1" in validate
    assert "pto_persistent_dag_graph_scratch_reuse=reused_buffer=tmp0,reuse_task=4" in validate
    assert "pto_persistent_dag_graph_diamond=9,2,1,2,1" in validate
    assert "pto_persistent_dag_graph_tagged=9,2,1" in validate
    assert "pto_persistent_dag_graph_tagged_inout=1,1,1" in validate
    assert "pto_persistent_dag_graph_triad=6,2,1" in validate
    assert "pto_persistent_dag_graph_quad=8,2,1" in validate
    assert "pto_persistent_dag_graph_unary_square=7,1,1" in validate
    assert "pto_persistent_dag_graph_triad=0,0,2" in validate
    assert "pto_persistent_dag_graph_quad=0,0,2" in validate
    assert "pto_persistent_dag_graph_triad=2,2" in validate
    assert "pto_persistent_dag_graph_quad=2,2" in validate
    assert "pto_persistent_dag_graph_diamond=0,0,2,2,2" in validate
    assert "pto_persistent_dag_graph_diamond=2,3,2,3,4,4" in validate
    assert "pto_persistent_dag_graph_unary_square=0,1,1" in validate
    assert "pto_persistent_dag_graph_unary_square=1,2" in validate
    assert (
        "pto_persistent_dag_graph_tagged="
        "task0=input:a,input:b,output:tmp1,scalar:scalar_args[0],scalar:scalar_args[1];"
        "task1=input:a,input:b,output:tmp2;task2=input:tmp1,input:tmp2,output_existing:out"
    ) in validate
    assert (
        "pto_persistent_dag_graph_tagged_inout="
        "task0=input:a,input:b,output:tmp1;task1=inout:tmp1,input:b;"
        "task2=input:tmp1,input:a,output_existing:out"
    ) in validate
    assert "pto_persistent_dag_tensor=3,1,2,1" in validate
    assert "pto_persistent_dag_graph_tensor=3,1,2,1" in validate
    assert "pto_persistent_dag_tensor_core=10,1,2,1" in validate
    assert "pto_persistent_dag_graph_tensor_core=10,1,2,1" in validate
    assert "--require-tensor-tile" in validate
    assert "pto_persistent_dag_tensor=16x16x16" in validate
    assert "pto_persistent_dag_graph_tensor=16x16x16" in validate
    assert "pto_persistent_dag_tensor_core=16x16x16" in validate
    assert "pto_persistent_dag_graph_tensor_core=16x16x16" in validate
    assert "pto_persistent_dag_graph_tensor_core=0,1,1,2" in validate
    assert "pto_persistent_dag_graph_tensor_core=1,2,3,3" in validate
    assert "cublas_sgemm=16x16x16" in validate
    assert "cublas_sgemm_graph=16x16x16" in validate
    assert "--require-command-examples" in validate
    assert "--require-source-papers" in validate
    assert "--require-zero-scheduler-errors" in validate
    assert index[-2:] == ["--root", str(tmp_path / "cuda-backend")]


def test_cuda_pair_benchmark_validate_command_matches_configured_capture(tmp_path):
    cuda_pair_benchmark = _load_pair_benchmark_module()
    config = cuda_pair_benchmark.PairedBenchmarkConfig(
        output_root=tmp_path / "cuda-backend",
        sizes=(1024, 2048),
        repeats=2,
        batch_tasks=(2,),
        worker_blocks_per_task=(4, 8),
        local_python=".venv/bin/python",
    )

    validate = cuda_pair_benchmark.build_validate_command(config, "local123", "remote456")

    assert any("combined-current-local123-remote456" in part for part in validate)
    assert "--require-size" in validate
    assert "1024,2048" in validate
    assert "--expected-repeats" in validate
    assert "2" in validate
    assert "--expected-result-count" in validate
    assert "320" in validate
    assert "--require-baseline" in validate
    baselines = [validate[index + 1] for index, part in enumerate(validate) if part == "--require-baseline"]
    assert "pto_host_schedule_generic_args" in baselines
    assert "pto_persistent_dag_graph_generic_args4" in baselines
    assert "pto_persistent_dag_graph_chain" in baselines
    assert "pto_persistent_dag_graph_scratch_reuse" in baselines
    assert "pto_persistent_dag_graph_diamond" in baselines
    assert "pto_persistent_dag_graph_tagged" in baselines
    assert "pto_persistent_dag_graph_tagged_inout" in baselines
    assert "pto_persistent_dag_graph_triad" in baselines
    assert "pto_persistent_dag_graph_quad" in baselines
    assert "pto_persistent_dag_graph_unary_square" in baselines
    assert "pto_persistent_dag_graph_tensor" in baselines
    assert "pto_persistent_dag_graph_tensor_core" in baselines
    assert "cublas_sgemm_graph" in baselines
    assert "pto_host_schedule_batch" in baselines
    assert "pto_persistent_device_grid_batch" in baselines
    dispatch = [validate[index + 1] for index, part in enumerate(validate) if part == "--require-dispatch"]
    assert "pto_persistent_dag_graph_generic_args4=9,2,1" in dispatch
    assert "pto_persistent_dag_graph_chain=1,2,1,2,1" in dispatch
    assert "pto_persistent_dag_graph_scratch_reuse=1,2,1,2,1,1" in dispatch
    assert "pto_persistent_dag_graph_diamond=9,2,1,2,1" in dispatch
    assert "pto_persistent_dag_graph_tagged=9,2,1" in dispatch
    assert "pto_persistent_dag_graph_tagged_inout=1,1,1" in dispatch
    assert "pto_persistent_dag_graph_triad=6,2,1" in dispatch
    assert "pto_persistent_dag_graph_quad=8,2,1" in dispatch
    assert "pto_persistent_dag_graph_unary_square=7,1,1" in dispatch
    assert "pto_persistent_dag_tensor_core=10,1,2,1" in dispatch
    assert "pto_persistent_dag_graph_tensor_core=10,1,2,1" in dispatch
    tensor_tiles = [validate[index + 1] for index, part in enumerate(validate) if part == "--require-tensor-tile"]
    assert "pto_persistent_dag_tensor=16x16x16" in tensor_tiles
    assert "pto_persistent_dag_graph_tensor_core=16x16x16" in tensor_tiles
    assert "cublas_sgemm=16x16x16" in tensor_tiles
    assert "cublas_sgemm_graph=16x16x16" in tensor_tiles
    graph_task_args = [
        validate[index + 1] for index, part in enumerate(validate) if part == "--require-graph-task-args"
    ]
    assert (
        "pto_persistent_dag_graph_tagged="
        "task0=input:a,input:b,output:tmp1,scalar:scalar_args[0],scalar:scalar_args[1];"
        "task1=input:a,input:b,output:tmp2;task2=input:tmp1,input:tmp2,output_existing:out"
    ) in graph_task_args


def test_cuda_pair_benchmark_omits_empty_batch_sweeps(tmp_path):
    cuda_pair_benchmark = _load_pair_benchmark_module()
    config = cuda_pair_benchmark.PairedBenchmarkConfig(
        output_root=tmp_path / "cuda-backend",
        sizes=(1024,),
        repeats=1,
        batch_tasks=(),
        worker_blocks_per_task=(),
        local_python=".venv/bin/python",
    )

    local = cuda_pair_benchmark.build_local_benchmark_command(config, "abc123")
    validate = cuda_pair_benchmark.build_validate_command(config, "abc123")

    assert "--batch-tasks" not in local
    assert "--worker-blocks-per-task" not in local
    assert "pto_host_schedule_batch" not in validate
    assert "pto_persistent_device_grid_batch" not in validate
    assert "--expected-result-count" in validate
    assert "70" in validate


def test_cuda_pair_benchmark_merge_command_records_sanitized_examples(tmp_path):
    cuda_pair_benchmark = _load_pair_benchmark_module()
    config = cuda_pair_benchmark.PairedBenchmarkConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        branch="design/nvidia-backend",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        sync_remote_tree=True,
    )

    merge = cuda_pair_benchmark.build_merge_command(config, "abc123")
    examples = [merge[index + 1] for index, part in enumerate(merge) if part == "--command-example"]

    assert len(examples) == 3
    assert all(str(Path.cwd()) not in example for example in examples)
    assert any(example.startswith("local_sample=env PYTHONPATH=$PWD:$PWD/python") for example in examples)
    assert any(example.startswith("remote_sample=ssh") and "--arch compute_90" in example for example in examples)
    assert any(example.startswith("sync_remote_tree=rsync") for example in examples)


def test_cuda_tensor_shape_sweep_builds_single_baseline_commands(tmp_path):
    cuda_tensor_shape_sweep = _load_tensor_shape_sweep_module()
    shape = cuda_tensor_shape_sweep.TensorShape(rows=16, cols=16, inner=64)
    config = cuda_tensor_shape_sweep.TensorShapeSweepConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        n=4096,
    )

    local = cuda_tensor_shape_sweep.build_local_sample_command(config, shape)
    remote = cuda_tensor_shape_sweep.build_remote_sample_command(config, shape)

    assert ".agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py" in local
    assert "--single-baseline" in local
    assert "pto_persistent_dag_tensor" in local
    assert "--sizes" in local
    assert "4096" in local
    assert "--arch" in local
    assert "compute_80" in local
    assert "--tensor-rows" in local
    assert "16" in local
    assert "--tensor-cols" in local
    assert "--tensor-inner" in local
    assert "64" in local

    assert remote[:6] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "h200-box"]
    assert "cd /remote/pto-cu" in remote[-1]
    assert "--single-baseline pto_persistent_dag_tensor" in remote[-1]
    assert "--arch compute_90" in remote[-1]
    assert "--tensor-rows 16" in remote[-1]
    assert "--tensor-cols 16" in remote[-1]
    assert "--tensor-inner 64" in remote[-1]


def test_cuda_tensor_shape_sweep_builds_size_specific_commands(tmp_path):
    cuda_tensor_shape_sweep = _load_tensor_shape_sweep_module()
    shape = cuda_tensor_shape_sweep.TensorShape(rows=16, cols=16, inner=16)
    config = cuda_tensor_shape_sweep.TensorShapeSweepConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        sizes=(256, 4096),
    )

    local = cuda_tensor_shape_sweep.build_local_sample_command(config, shape, n=4096)
    remote = cuda_tensor_shape_sweep.build_remote_sample_command(config, shape, n=256)

    assert "--sizes" in local
    assert "4096" in local
    assert "--sizes 256" in remote[-1]


def test_cuda_tensor_shape_sweep_builds_configured_baseline_commands(tmp_path):
    cuda_tensor_shape_sweep = _load_tensor_shape_sweep_module()
    shape = cuda_tensor_shape_sweep.TensorShape(rows=16, cols=16, inner=16)
    config = cuda_tensor_shape_sweep.TensorShapeSweepConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        n=256,
        baselines=("cublas_sgemm", "pto_persistent_dag_tensor_core"),
    )

    local = cuda_tensor_shape_sweep.build_local_sample_command(config, shape, "cublas_sgemm")
    remote = cuda_tensor_shape_sweep.build_remote_sample_command(
        config,
        shape,
        "pto_persistent_dag_tensor_core",
    )

    assert "--single-baseline" in local
    assert "cublas_sgemm" in local
    assert "--sizes" in local
    assert "256" in local

    assert "--single-baseline pto_persistent_dag_tensor_core" in remote[-1]
    assert "--tensor-rows 16" in remote[-1]
    assert "--tensor-cols 16" in remote[-1]
    assert "--tensor-inner 16" in remote[-1]


def test_cuda_tensor_shape_sweep_accepts_graph_tensor_baseline(tmp_path):
    cuda_tensor_shape_sweep = _load_tensor_shape_sweep_module()
    shape = cuda_tensor_shape_sweep.TensorShape(rows=16, cols=16, inner=16)
    config = cuda_tensor_shape_sweep.TensorShapeSweepConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        n=256,
        baselines=cuda_tensor_shape_sweep.parse_baselines("pto_persistent_dag_tensor,pto_persistent_dag_graph_tensor"),
    )

    local = cuda_tensor_shape_sweep.build_local_sample_command(config, shape, "pto_persistent_dag_graph_tensor")
    remote = cuda_tensor_shape_sweep.build_remote_sample_command(config, shape, "pto_persistent_dag_graph_tensor")

    assert "--single-baseline" in local
    assert "pto_persistent_dag_graph_tensor" in local
    assert "--single-baseline pto_persistent_dag_graph_tensor" in remote[-1]
    assert "--tensor-rows 16" in remote[-1]
    assert "--tensor-cols 16" in remote[-1]
    assert "--tensor-inner 16" in remote[-1]


def test_cuda_tensor_shape_sweep_accepts_graph_tensor_core_baseline(tmp_path):
    cuda_tensor_shape_sweep = _load_tensor_shape_sweep_module()
    shape = cuda_tensor_shape_sweep.TensorShape(rows=16, cols=16, inner=16)
    config = cuda_tensor_shape_sweep.TensorShapeSweepConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        n=256,
        baselines=cuda_tensor_shape_sweep.parse_baselines(
            "pto_persistent_dag_tensor_core,pto_persistent_dag_graph_tensor_core"
        ),
    )

    local = cuda_tensor_shape_sweep.build_local_sample_command(config, shape, "pto_persistent_dag_graph_tensor_core")
    remote = cuda_tensor_shape_sweep.build_remote_sample_command(config, shape, "pto_persistent_dag_graph_tensor_core")

    assert "--single-baseline" in local
    assert "pto_persistent_dag_graph_tensor_core" in local
    assert "--single-baseline pto_persistent_dag_graph_tensor_core" in remote[-1]
    assert "--tensor-rows 16" in remote[-1]
    assert "--tensor-cols 16" in remote[-1]
    assert "--tensor-inner 16" in remote[-1]


def test_cuda_tensor_shape_sweep_accepts_cublas_graph_baseline(tmp_path):
    cuda_tensor_shape_sweep = _load_tensor_shape_sweep_module()
    shape = cuda_tensor_shape_sweep.TensorShape(rows=16, cols=16, inner=16)
    config = cuda_tensor_shape_sweep.TensorShapeSweepConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        n=256,
        baselines=cuda_tensor_shape_sweep.parse_baselines("cublas_sgemm,cublas_sgemm_graph"),
    )

    local = cuda_tensor_shape_sweep.build_local_sample_command(config, shape, "cublas_sgemm_graph")
    remote = cuda_tensor_shape_sweep.build_remote_sample_command(config, shape, "cublas_sgemm_graph")

    assert "--single-baseline" in local
    assert "cublas_sgemm_graph" in local
    assert "--single-baseline cublas_sgemm_graph" in remote[-1]
    assert "--tensor-rows 16" in remote[-1]
    assert "--tensor-cols 16" in remote[-1]
    assert "--tensor-inner 16" in remote[-1]


def test_cuda_tensor_shape_sweep_parses_shapes_and_renders_report():
    cuda_tensor_shape_sweep = _load_tensor_shape_sweep_module()

    shapes = cuda_tensor_shape_sweep.parse_shapes("8x4x12,16x16x64")
    assert shapes == (
        cuda_tensor_shape_sweep.TensorShape(rows=8, cols=4, inner=12),
        cuda_tensor_shape_sweep.TensorShape(rows=16, cols=16, inner=64),
    )
    assert cuda_tensor_shape_sweep.parse_sizes("256,4096") == (256, 4096)

    payload = {
        "metadata": {
            "label": "tensor-shape-sweep-abc123",
            "git_commit": "abc123",
            "n": 4096,
            "sizes": [4096, 8192],
            "repeats": 2,
            "shapes": ["8x4x12"],
            "baselines": ["pto_persistent_dag_tensor", "pto_persistent_dag_graph_tensor", "cublas_sgemm"],
            "source_papers": [
                {"id": "arXiv:2605.03190", "label": "VDCores"},
                {"id": "arXiv:2512.22219v1", "label": "MPK persistent kernel"},
            ],
            "command_examples": {
                "local_sample": "env PYTHONPATH=$PWD:$PWD/python .venv/bin/python cuda_benchmark.py",
                "remote_sample": "ssh h200-box 'cd /remote/pto-cu && cuda_benchmark.py'",
            },
        },
        "results": [
            {
                "artifact": "a100",
                "machine": "hina",
                "baseline": "pto_persistent_dag_tensor",
                "n": 4096,
                "shape": "8x4x12",
                "repeat": 0,
                "status": "pass",
                "device_wall_ns": 1000,
                "host_wall_ns": 1500,
                "dispatch_func_ids": [3, 1, 2, 1],
                "tensor_tile": {"rows": 8, "cols": 4, "inner": 12, "tile_count": 128},
            },
            {
                "artifact": "h200",
                "baseline": "cublas_sgemm",
                "n": 8192,
                "shape": "8x4x12",
                "repeat": 0,
                "status": "pass",
                "device_wall_ns": 800,
                "host_wall_ns": 1200,
                "dispatch_func_ids": [3, 1, 2, 1],
                "tensor_tile": {"rows": 8, "cols": 4, "inner": 12, "tile_count": 128},
            },
            {
                "artifact": "a100",
                "machine": "hina",
                "baseline": "pto_persistent_dag_graph_tensor",
                "n": 4096,
                "shape": "8x4x12",
                "repeat": 0,
                "status": "pass",
                "device_wall_ns": 1100,
                "host_wall_ns": 1600,
                "dispatch_func_ids": [3, 1, 2, 1],
                "tensor_tile": {"rows": 8, "cols": 4, "inner": 12, "tile_count": 128},
            },
        ],
    }

    markdown = cuda_tensor_shape_sweep.render_markdown(payload)
    svg = cuda_tensor_shape_sweep.render_svg(payload)
    throughput_svg = cuda_tensor_shape_sweep.render_throughput_svg(payload)

    assert "- Source setup: `arXiv:2605.03190` VDCores; `arXiv:2512.22219v1` MPK persistent kernel." in markdown
    assert "- Workloads:" in markdown
    assert "  - `pto_persistent_dag_tensor`: PTO persistent DAG with scalar tiled GEMM work." in markdown
    assert (
        "  - `pto_persistent_dag_graph_tensor`: PTO persistent DAG with explicit graph scalar tiled GEMM work."
        in markdown
    )
    assert "  - `cublas_sgemm`: CUDA Runtime API plus cuBLAS SGEMM over the same descriptor." in markdown
    assert "- Workload: `pto_persistent_dag_tensor` scalar tiled GEMM DAG." not in markdown
    assert "- Baselines: `pto_persistent_dag_tensor`, `pto_persistent_dag_graph_tensor`, `cublas_sgemm`." in markdown
    assert "- Sizes: `4096`, `8192`" in markdown
    assert "- Local sample command: `env PYTHONPATH=$PWD:$PWD/python .venv/bin/python cuda_benchmark.py`" in markdown
    assert "- Remote sample command: `ssh h200-box 'cd /remote/pto-cu && cuda_benchmark.py'`" in markdown
    assert (
        "| a100 | hina | pto_persistent_dag_tensor | 4096 | 8x4x12 | 0 | pass | 1000 | 1500 | 128 | `3,1,2,1` |"
        in markdown
    )
    assert (
        "| a100 | hina | pto_persistent_dag_graph_tensor | 4096 | 8x4x12 | 0 | pass | 1100 | 1600 | 128 | `3,1,2,1` |"
        in markdown
    )
    assert "| h200 | h200 | cublas_sgemm | 8192 | 8x4x12 | 0 | pass | 800 | 1200 | 128 | `3,1,2,1` |" in markdown
    assert "## Median Summary" in markdown
    assert "| a100 | hina | pto_persistent_dag_tensor | 4096 | 8x4x12 | 1000 | 1500 | 98.30 | 1 |" in markdown
    assert "| h200 | h200 | cublas_sgemm | 8192 | 8x4x12 | 800 | 1200 | 122.88 | 1 |" in markdown
    assert "| a100 | hina | pto_persistent_dag_graph_tensor | 4096 | 8x4x12 | 1100 | 1600 | 89.37 | 1 |" in markdown
    assert "tensor-shape-sweep-abc123" in svg
    assert "Median device ns" in svg
    assert "samples=1" in svg
    assert "8x4x12" in svg
    assert "cublas_sgemm" in svg
    assert "r0" not in svg
    assert "Median GF/s" in throughput_svg
    assert "98.30" in throughput_svg
    assert "122.88" in throughput_svg
    assert "cublas_sgemm" in throughput_svg


def test_cuda_tensor_shape_sweep_dry_run_records_source_papers(tmp_path):
    cuda_tensor_shape_sweep = _load_tensor_shape_sweep_module()

    def fake_runner(command, **kwargs):
        class Result:
            stdout = "abc123\n"

        assert command == ["git", "rev-parse", "--short", "HEAD"]
        return Result()

    payload = cuda_tensor_shape_sweep.run_tensor_shape_sweep(
        cuda_tensor_shape_sweep.TensorShapeSweepConfig(
            output_root=tmp_path,
            n=256,
            repeats=1,
            baselines=("pto_persistent_dag_tensor_core",),
            shapes=(cuda_tensor_shape_sweep.TensorShape(rows=16, cols=16, inner=16),),
            refresh_remote=False,
        ),
        runner=fake_runner,
        dry_run=True,
    )

    assert payload["metadata"]["paper_setup"] == (
        "Model-shaped tensor tile sweep using fixed GPU work, repeated samples, "
        "selected launch/library baselines, local A100, and remote H200."
    )
    assert payload["metadata"]["source_papers"] == [
        {
            "id": "arXiv:2605.03190",
            "label": "VDCores",
            "path": "tmp/sources/arxiv-2605.03190-vdcores.txt",
        },
        {
            "id": "arXiv:2512.22219v1",
            "label": "MPK persistent kernel",
            "path": "tmp/sources/arxiv-2512.22219v1-mirage-persistent-kernel.txt",
        },
    ]
    command_examples = payload["metadata"]["command_examples"]
    assert "local_sample" in command_examples
    assert "remote_sample" in command_examples
    assert "sync_remote_tree" not in command_examples
    assert "$PWD:$PWD/python" in command_examples["local_sample"]
    assert str(Path.cwd()) not in command_examples["local_sample"]
    assert "--single-baseline pto_persistent_dag_tensor_core" in command_examples["local_sample"]
    assert "ssh" in command_examples["remote_sample"]
    assert "--arch compute_90" in command_examples["remote_sample"]


def test_cuda_tensor_shape_sweep_main_renders_existing_json(tmp_path, capsys):
    cuda_tensor_shape_sweep = _load_tensor_shape_sweep_module()
    payload = {
        "metadata": {
            "label": "tensor-shape-sweep-existing",
            "git_commit": "abc123",
            "n": 4096,
            "sizes": [4096],
            "repeats": 1,
            "shapes": ["8x4x12"],
            "baselines": ["pto_persistent_dag_tensor"],
            "source_papers": [],
        },
        "results": [
            {
                "artifact": "a100",
                "machine": "hina",
                "baseline": "pto_persistent_dag_tensor",
                "n": 4096,
                "shape": "8x4x12",
                "repeat": 0,
                "status": "pass",
                "device_wall_ns": 1000,
                "host_wall_ns": 1500,
                "tensor_tile": {"rows": 8, "cols": 4, "inner": 12, "tile_count": 128},
            }
        ],
    }
    input_json = tmp_path / "input.json"
    output_dir = tmp_path / "rerendered"
    input_json.write_text(json.dumps(payload) + "\n")

    cuda_tensor_shape_sweep.main(
        [
            "--render-json",
            str(input_json),
            "--render-output-dir",
            str(output_dir),
        ]
    )

    printed = capsys.readouterr().out
    assert str(output_dir / "cuda-tensor-shape-sweep.json") in printed
    assert str(output_dir / "cuda-tensor-shape-throughput.svg") in printed
    assert "98.30" in (output_dir / "cuda-tensor-shape-sweep.md").read_text()
    assert "Median GF/s" in (output_dir / "cuda-tensor-shape-throughput.svg").read_text()


def test_cuda_pair_benchmark_can_reuse_remote_checkout(tmp_path):
    cuda_pair_benchmark = _load_pair_benchmark_module()
    config = cuda_pair_benchmark.PairedBenchmarkConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        branch="design/nvidia-backend",
        output_root=tmp_path / "cuda-backend",
        remote_python=".venv/bin/python",
        refresh_remote=False,
    )

    remote = cuda_pair_benchmark.build_remote_benchmark_command(config, "abc123")
    remote_shell = remote[-1]

    assert "cd /remote/pto-cu" in remote_shell
    assert "git fetch" not in remote_shell
    assert "git checkout" not in remote_shell
    assert ".agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py" in remote_shell
    assert "--arch compute_90" in remote_shell
    assert "h200-current-abc123" in remote_shell


def test_cuda_pair_benchmark_reused_checkout_uses_remote_commit(tmp_path):
    cuda_pair_benchmark = _load_pair_benchmark_module()
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs))
        if command == ["git", "rev-parse", "--short", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, stdout="local123\n", stderr="")
        if command[0] == "ssh" and "git rev-parse --short HEAD" in command[-1]:
            return subprocess.CompletedProcess(command, 0, stdout="remote456\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    config = cuda_pair_benchmark.PairedBenchmarkConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        refresh_remote=False,
    )

    commands = cuda_pair_benchmark.run_paired_benchmark(config, runner=fake_runner, dry_run=True)

    assert calls == [
        (["git", "rev-parse", "--short", "HEAD"], {"check": True, "capture_output": True, "text": True}),
        (
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=8",
                "h200-box",
                "cd /remote/pto-cu && git rev-parse --short HEAD",
            ],
            {"check": True, "capture_output": True, "text": True},
        ),
    ]
    assert "a100-current-local123" in commands[0]
    assert "h200-current-remote456" in commands[1][-1]
    assert "h200-current-remote456" in commands[2][2]
    assert str(tmp_path / "cuda-backend" / "a100-current-local123" / "cuda-benchmark.json") in commands[3]
    assert str(tmp_path / "cuda-backend" / "h200-current-remote456" / "cuda-benchmark.json") in commands[3]
    assert "combined-current-local123-remote456" in commands[3]
    assert any("combined-current-local123-remote456" in part for part in commands[4])
    assert commands[5][-2:] == ["--root", str(tmp_path / "cuda-backend")]


def test_cuda_pair_benchmark_can_sync_local_tree_to_remote(tmp_path):
    cuda_pair_benchmark = _load_pair_benchmark_module()
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="local123\n", stderr="")

    config = cuda_pair_benchmark.PairedBenchmarkConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        sync_remote_tree=True,
    )

    commands = cuda_pair_benchmark.run_paired_benchmark(config, runner=fake_runner, dry_run=True)

    assert calls == [(["git", "rev-parse", "--short", "HEAD"], {"check": True, "capture_output": True, "text": True})]
    assert len(commands) == 7
    sync = commands[1]
    assert sync[:3] == ["rsync", "-a", "--delete"]
    assert "--exclude=.git" not in sync
    assert sync[-2:] == [f"{Path.cwd()}/", "h200-box:/remote/pto-cu/"]
    remote_shell = commands[2][-1]
    assert "git fetch" not in remote_shell
    assert "git checkout" not in remote_shell
    assert "git rev-parse" not in remote_shell
    assert "h200-current-local123" in remote_shell
    assert "h200-current-local123" in commands[3][2]
    assert "combined-current-local123" in commands[4]
    assert any("combined-current-local123" in part for part in commands[5])


def test_cuda_pair_benchmark_dry_run_does_not_launch_benchmarks(tmp_path):
    cuda_pair_benchmark = _load_pair_benchmark_module()
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="abc123\n", stderr="")

    config = cuda_pair_benchmark.PairedBenchmarkConfig(
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
    )
    commands = cuda_pair_benchmark.run_paired_benchmark(config, runner=fake_runner, dry_run=True)

    assert calls == [(["git", "rev-parse", "--short", "HEAD"], {"check": True, "capture_output": True, "text": True})]
    assert len(commands) == 6
    assert commands[0][0] == "env"
    assert commands[1][0] == "ssh"
    assert commands[2][0] == "scp"
    assert commands[4][0] == "env"


def test_cuda_pair_smoke_builds_worker_mul_a100_h200_workflow(tmp_path):
    cuda_pair_smoke = _load_pair_smoke_module()
    config = cuda_pair_smoke.PairedSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        branch="design/nvidia-backend",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        op="mul",
    )

    local = cuda_pair_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_smoke.build_remote_smoke_command(config, "abc123")
    scp = cuda_pair_smoke.build_scp_command(config, "abc123")
    report = cuda_pair_smoke.build_report_command(config, "abc123")
    index = cuda_pair_smoke.build_index_command(config)

    assert local[:2] == ["env", f"PYTHONPATH={Path.cwd()}:{Path.cwd() / 'python'}"]
    assert ".agents/skills/cuda-backend-eval/scripts/cuda_smoke.py" in local
    assert "--runner" in local
    assert "worker" in local
    assert "--op" in local
    assert "mul" in local
    assert "--arch" in local
    assert "compute_80" in local
    assert str(tmp_path / "cuda-backend" / "worker-mul-smoke-abc123" / "a100.json") in local

    assert remote[:6] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "h200-box"]
    remote_shell = remote[-1]
    assert "cd /remote/pto-cu" in remote_shell
    assert (
        "timeout 60 git -c http.lowSpeedLimit=1 -c http.lowSpeedTime=30 "
        "fetch origin design/nvidia-backend" in remote_shell
    )
    assert "git checkout -B design/nvidia-backend FETCH_HEAD >/dev/null" in remote_shell
    assert "CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH PYTHONPATH=$PWD:$PWD/python" in remote_shell
    assert ".agents/skills/cuda-backend-eval/scripts/cuda_smoke.py" in remote_shell
    assert "--runner worker" in remote_shell
    assert "--op mul" in remote_shell
    assert "--arch compute_90" in remote_shell
    assert "worker-mul-smoke-abc123/h200.json" in remote_shell

    assert scp == [
        "scp",
        f"h200-box:/remote/pto-cu/{tmp_path / 'cuda-backend' / 'worker-mul-smoke-abc123' / 'h200.json'}",
        str(tmp_path / "cuda-backend" / "worker-mul-smoke-abc123" / "h200.json"),
    ]
    assert ".agents/skills/cuda-backend-eval/scripts/cuda_smoke_report.py" in report
    assert str(tmp_path / "cuda-backend" / "worker-mul-smoke-abc123" / "a100.json") in report
    assert str(tmp_path / "cuda-backend" / "worker-mul-smoke-abc123" / "h200.json") in report
    assert "worker-mul-smoke-abc123" in report
    assert index[-2:] == ["--root", str(tmp_path / "cuda-backend")]


def test_cuda_pair_smoke_accepts_affine_op():
    cuda_pair_smoke = _load_pair_smoke_module()

    args = cuda_pair_smoke.parse_args(["--op", "affine", "--sync-remote-tree"])

    assert args.op == "affine"
    assert args.sync_remote_tree is True


def test_cuda_pair_smoke_accepts_triad_op():
    cuda_pair_smoke = _load_pair_smoke_module()

    args = cuda_pair_smoke.parse_args(["--op", "triad", "--sync-remote-tree"])

    assert args.op == "triad"
    assert args.sync_remote_tree is True


def test_cuda_pair_smoke_accepts_quad_op():
    cuda_pair_smoke = _load_pair_smoke_module()

    args = cuda_pair_smoke.parse_args(["--op", "quad", "--sync-remote-tree"])

    assert args.op == "quad"
    assert args.sync_remote_tree is True


def test_cuda_pair_smoke_accepts_generic_args_op():
    cuda_pair_smoke = _load_pair_smoke_module()

    args = cuda_pair_smoke.parse_args(["--op", "generic_args", "--sync-remote-tree"])

    assert args.op == "generic_args"
    assert args.sync_remote_tree is True


def test_cuda_pair_smoke_can_sync_local_tree_and_dry_run(tmp_path):
    cuda_pair_smoke = _load_pair_smoke_module()
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="local123\n", stderr="")

    config = cuda_pair_smoke.PairedSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        sync_remote_tree=True,
    )

    commands = cuda_pair_smoke.run_paired_smoke(config, runner=fake_runner, dry_run=True)

    assert calls == [(["git", "rev-parse", "--short", "HEAD"], {"check": True, "capture_output": True, "text": True})]
    assert len(commands) == 6
    sync = commands[1]
    assert sync[:3] == ["rsync", "-a", "--delete"]
    assert sync[-2:] == [f"{Path.cwd()}/", "h200-box:/remote/pto-cu/"]
    assert "git fetch" not in commands[2][-1]
    assert "git checkout" not in commands[2][-1]
    assert any("worker-add-smoke-local123" in part for part in commands[0])
    assert "worker-add-smoke-local123" in commands[2][-1]
    assert "worker-add-smoke-local123" in commands[3][1]
    assert any("worker-add-smoke-local123" in part for part in commands[4])


def test_cuda_pair_smoke_can_request_runtime_build(tmp_path):
    cuda_pair_smoke = _load_pair_smoke_module()
    config = cuda_pair_smoke.PairedSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        build_runtime=True,
    )

    local = cuda_pair_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_smoke.build_remote_smoke_command(config, "abc123")

    assert "--no-build" not in local
    assert "--no-build" not in remote[-1]


def test_cuda_persistent_smoke_main_writes_output_json(tmp_path, monkeypatch, capsys):
    cuda_persistent_smoke = _load_persistent_smoke_module()
    output = tmp_path / "persistent-smoke.json"

    monkeypatch.setattr(
        cuda_persistent_smoke,
        "run_persistent_smoke",
        lambda **kwargs: {
            "status": "pass",
            "runtime": "persistent_device",
            "mode": kwargs["mode"],
            "dag_shape": kwargs["dag_shape"],
            "device": kwargs["device"],
            "n": kwargs["n"],
            "ptx_arch": kwargs["arch"],
            "resource_policy": {"block_dim": kwargs["block_dim"]},
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cuda_persistent_smoke.py",
            "--device",
            "1",
            "--task-count",
            "5",
            "--n",
            "64",
            "--arch",
            "compute_90",
            "--mode",
            "dag",
            "--queue-capacity",
            "3",
            "--dag-shape",
            "chain",
            "--block-dim",
            "128",
            "--output-json",
            str(output),
        ],
    )

    cuda_persistent_smoke.main()

    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text())
    assert printed == written
    assert written["runtime"] == "persistent_device"
    assert written["mode"] == "dag"
    assert written["dag_shape"] == "chain"
    assert written["device"] == 1
    assert written["ptx_arch"] == "compute_90"
    assert written["resource_policy"]["block_dim"] == 128


def test_cuda_pair_persistent_smoke_builds_chain_a100_h200_workflow(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        branch="design/nvidia-backend",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        dag_shape="chain",
        task_count=5,
        queue_capacity=3,
        worker_blocks=2,
        stream_id=1,
        block_dim=128,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")
    validate = cuda_pair_persistent_smoke.build_validate_command(config, "abc123")
    scp = cuda_pair_persistent_smoke.build_scp_command(config, "abc123")
    report = cuda_pair_persistent_smoke.build_report_command(config, "abc123")
    index = cuda_pair_persistent_smoke.build_index_command(config)

    assert local[:2] == ["env", f"PYTHONPATH={Path.cwd()}:{Path.cwd() / 'python'}"]
    assert ".agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py" in local
    assert "--mode" in local
    assert "dag" in local
    assert "--dag-shape" in local
    assert "chain" in local
    assert "--task-count" in local
    assert "5" in local
    assert "--queue-capacity" in local
    assert "3" in local
    assert "--worker-blocks" in local
    assert "2" in local
    assert "--stream-id" in local
    assert "1" in local
    assert "--block-dim" in local
    assert "128" in local
    assert "--arch" in local
    assert "compute_80" in local
    assert str(tmp_path / "cuda-backend" / "persistent-chain-smoke-abc123" / "a100.json") in local

    assert remote[:6] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "h200-box"]
    remote_shell = remote[-1]
    assert "cd /remote/pto-cu" in remote_shell
    assert (
        "timeout 60 git -c http.lowSpeedLimit=1 -c http.lowSpeedTime=30 "
        "fetch origin design/nvidia-backend" in remote_shell
    )
    assert "git checkout -B design/nvidia-backend FETCH_HEAD >/dev/null" in remote_shell
    assert "CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH PYTHONPATH=$PWD:$PWD/python" in remote_shell
    assert ".agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py" in remote_shell
    assert "--mode dag" in remote_shell
    assert "--dag-shape chain" in remote_shell
    assert "--worker-blocks 2" in remote_shell
    assert "--stream-id 1" in remote_shell
    assert "--block-dim 128" in remote_shell
    assert "--arch compute_90" in remote_shell
    assert "persistent-chain-smoke-abc123/h200.json" in remote_shell

    assert scp == [
        "scp",
        f"h200-box:/remote/pto-cu/{tmp_path / 'cuda-backend' / 'persistent-chain-smoke-abc123' / 'h200.json'}",
        str(tmp_path / "cuda-backend" / "persistent-chain-smoke-abc123" / "h200.json"),
    ]
    assert ".agents/skills/cuda-backend-eval/scripts/cuda_smoke_report.py" in report
    assert str(tmp_path / "cuda-backend" / "persistent-chain-smoke-abc123" / "a100.json") in report
    assert str(tmp_path / "cuda-backend" / "persistent-chain-smoke-abc123" / "h200.json") in report
    assert "persistent-chain-smoke-abc123" in report
    assert ".agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py" in validate
    assert str(tmp_path / "cuda-backend" / "persistent-chain-smoke-abc123" / "a100.json") in validate
    assert str(tmp_path / "cuda-backend" / "persistent-chain-smoke-abc123" / "h200.json") in validate
    assert "--require-artifact" in validate
    assert "a100" in validate
    assert "h200" in validate
    assert "--expected-runtime" in validate
    assert "persistent_device" in validate
    assert "--expected-mode" in validate
    assert "dag" in validate
    assert "--expected-dag-shape" in validate
    assert "chain" in validate
    assert "--expected-completed-count" in validate
    assert "5" in validate
    assert "--expected-scheduler-blocks" in validate
    assert "1" in validate
    assert "--expected-worker-blocks" in validate
    assert "2" in validate
    assert "--expected-worker-blocks-per-task" in validate
    assert "--expected-stream-id" in validate
    assert "--expected-block-dim" in validate
    assert "128" in validate
    assert "--expected-grid-dim" in validate
    assert "3" in validate
    assert "--require-report-files" in validate
    assert index[-2:] == ["--root", str(tmp_path / "cuda-backend")]


def test_cuda_pair_persistent_smoke_passes_repeat_runs(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        dag_shape="chain",
        task_count=5,
        repeat_runs=2,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")

    assert "--repeat-runs" in local
    assert "2" in local
    assert "persistent-chain-repeat2-smoke-abc123" in " ".join(local)
    assert "--repeat-runs 2" in remote[-1]
    assert "persistent-chain-repeat2-smoke-abc123/h200.json" in remote[-1]


def test_cuda_pair_persistent_smoke_labels_queue_mode_by_mode(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        mode="queue",
        task_count=4,
        queue_capacity=2,
        repeat_runs=2,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")

    assert str(tmp_path / "cuda-backend" / "persistent-queue-repeat2-smoke-abc123" / "a100.json") in local
    assert "persistent-fork_join-repeat2-smoke-abc123" not in " ".join(local)


def test_cuda_pair_persistent_smoke_builds_tensor_tile_descriptor_workflow(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        dag_shape="tensor_tile",
        task_count=4,
        n=768,
        tensor_rows=8,
        tensor_cols=4,
        tensor_inner=12,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")
    validate = cuda_pair_persistent_smoke.build_validate_command(config, "abc123")
    report = cuda_pair_persistent_smoke.build_report_command(config, "abc123")

    assert "persistent-tensor_tile-8x4x12-smoke-abc123" in str(local)
    assert "--tensor-rows" in local
    assert "8" in local
    assert "--tensor-cols" in local
    assert "4" in local
    assert "--tensor-inner" in local
    assert "12" in local
    assert "--dag-shape tensor_tile" in remote[-1]
    assert "--tensor-rows 8" in remote[-1]
    assert "--tensor-cols 4" in remote[-1]
    assert "--tensor-inner 12" in remote[-1]
    assert "--expected-dispatch" in validate
    assert "3,1,2,1" in validate
    assert "--expected-tensor-tile" in validate
    assert "8x4x12" in validate
    assert "persistent-tensor_tile-8x4x12-smoke-abc123" in report


def test_cuda_pair_persistent_smoke_builds_graph_tensor_tile_workflow(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        dag_shape="graph_tensor_tile",
        task_count=4,
        n=768,
        tensor_rows=8,
        tensor_cols=4,
        tensor_inner=12,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")
    validate = cuda_pair_persistent_smoke.build_validate_command(config, "abc123")
    report = cuda_pair_persistent_smoke.build_report_command(config, "abc123")

    assert "persistent-graph_tensor_tile-8x4x12-smoke-abc123" in str(local)
    assert "--dag-shape graph_tensor_tile" in remote[-1]
    assert "--tensor-rows 8" in remote[-1]
    assert "--tensor-cols 4" in remote[-1]
    assert "--tensor-inner 12" in remote[-1]
    assert "--expected-dag-shape" in validate
    assert "graph_tensor_tile" in validate
    assert "--expected-completed-count" in validate
    assert "4" in validate
    assert "--expected-dispatch" in validate
    assert "3,1,2,1" in validate
    assert "--expected-tensor-tile" in validate
    assert "8x4x12" in validate
    assert "persistent-graph_tensor_tile-8x4x12-smoke-abc123" in report


def test_cuda_pair_persistent_smoke_builds_tensor_core_tile_workflow(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        dag_shape="tensor_core_tile",
        task_count=4,
        n=256,
        tensor_rows=16,
        tensor_cols=16,
        tensor_inner=16,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")
    validate = cuda_pair_persistent_smoke.build_validate_command(config, "abc123")
    report = cuda_pair_persistent_smoke.build_report_command(config, "abc123")

    assert "persistent-tensor_core_tile-16x16x16-smoke-abc123" in str(local)
    assert "--dag-shape tensor_core_tile" in remote[-1]
    assert "--tensor-rows 16" in remote[-1]
    assert "--expected-completed-count" in validate
    assert "4" in validate
    assert "--expected-dispatch" in validate
    assert "10,1,2,1" in validate
    assert "--expected-tensor-tile" in validate
    assert "16x16x16" in validate
    assert "persistent-tensor_core_tile-16x16x16-smoke-abc123" in report


def test_cuda_pair_persistent_smoke_builds_graph_tensor_core_tile_workflow(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        dag_shape="graph_tensor_core_tile",
        task_count=4,
        n=256,
        tensor_rows=16,
        tensor_cols=16,
        tensor_inner=16,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")
    validate = cuda_pair_persistent_smoke.build_validate_command(config, "abc123")
    report = cuda_pair_persistent_smoke.build_report_command(config, "abc123")

    assert "persistent-graph_tensor_core_tile-16x16x16-smoke-abc123" in str(local)
    assert "--dag-shape graph_tensor_core_tile" in remote[-1]
    assert "--tensor-rows 16" in remote[-1]
    assert "--expected-dag-shape" in validate
    assert "graph_tensor_core_tile" in validate
    assert "--expected-completed-count" in validate
    assert "4" in validate
    assert "--expected-dispatch" in validate
    assert "10,1,2,1" in validate
    assert "--expected-tensor-tile" in validate
    assert "16x16x16" in validate
    assert "--expected-graph-fanin" in validate
    assert "0,1,1,2" in validate
    assert "--expected-graph-dependents" in validate
    assert "1,2,3,3" in validate
    assert "persistent-graph_tensor_core_tile-16x16x16-smoke-abc123" in report


def test_cuda_pair_persistent_smoke_builds_scalar_axpy_workflow(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        dag_shape="scalar_axpy",
        task_count=3,
        queue_capacity=2,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")
    validate = cuda_pair_persistent_smoke.build_validate_command(config, "abc123")

    assert "persistent-scalar_axpy-smoke-abc123" in str(local)
    assert "--dag-shape" in local
    assert "scalar_axpy" in local
    assert "--dag-shape scalar_axpy" in remote[-1]
    assert "persistent-scalar_axpy-smoke-abc123/h200.json" in remote[-1]
    assert "--expected-dispatch" in validate
    assert "4,2,1" in validate


def test_cuda_pair_persistent_smoke_builds_scalar_scale_workflow(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        dag_shape="scalar_scale",
        task_count=3,
        queue_capacity=2,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")
    validate = cuda_pair_persistent_smoke.build_validate_command(config, "abc123")

    assert "persistent-scalar_scale-smoke-abc123" in str(local)
    assert "--dag-shape" in local
    assert "scalar_scale" in local
    assert "--dag-shape scalar_scale" in remote[-1]
    assert "persistent-scalar_scale-smoke-abc123/h200.json" in remote[-1]
    assert "--expected-dispatch" in validate
    assert "11,2,1" in validate


def test_cuda_pair_persistent_smoke_builds_graph_descriptor_scalar_scale_workflow(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        dag_shape="graph_descriptor_scalar_scale",
        task_count=3,
        queue_capacity=2,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")
    validate = cuda_pair_persistent_smoke.build_validate_command(config, "abc123")

    assert "persistent-graph_descriptor_scalar_scale-smoke-abc123" in str(local)
    assert "--dag-shape" in local
    assert "graph_descriptor_scalar_scale" in local
    assert "--dag-shape graph_descriptor_scalar_scale" in remote[-1]
    assert "persistent-graph_descriptor_scalar_scale-smoke-abc123/h200.json" in remote[-1]
    assert "--expected-dispatch" in validate
    assert "11,2,1" in validate
    assert "--expected-graph-fanin" in validate
    assert "0,0,2" in validate
    assert "--expected-graph-dependents" in validate
    assert "2,2" in validate


def test_cuda_pair_persistent_smoke_builds_graph_descriptor_scalar_axpy_workflow(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        dag_shape="graph_descriptor_scalar_axpy",
        task_count=3,
        queue_capacity=2,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")
    validate = cuda_pair_persistent_smoke.build_validate_command(config, "abc123")

    assert "persistent-graph_descriptor_scalar_axpy-smoke-abc123" in str(local)
    assert "--dag-shape" in local
    assert "graph_descriptor_scalar_axpy" in local
    assert "--dag-shape graph_descriptor_scalar_axpy" in remote[-1]
    assert "persistent-graph_descriptor_scalar_axpy-smoke-abc123/h200.json" in remote[-1]
    assert "--expected-dispatch" in validate
    assert "4,2,1" in validate
    assert "--expected-graph-fanin" in validate
    assert "0,0,2" in validate
    assert "--expected-graph-dependents" in validate
    assert "2,2" in validate


def test_cuda_pair_persistent_smoke_builds_graph_descriptor_scalar_affine_workflow(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        dag_shape="graph_descriptor_scalar_affine",
        task_count=3,
        queue_capacity=2,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")
    validate = cuda_pair_persistent_smoke.build_validate_command(config, "abc123")

    assert "persistent-graph_descriptor_scalar_affine-smoke-abc123" in str(local)
    assert "--dag-shape" in local
    assert "graph_descriptor_scalar_affine" in local
    assert "--dag-shape graph_descriptor_scalar_affine" in remote[-1]
    assert "persistent-graph_descriptor_scalar_affine-smoke-abc123/h200.json" in remote[-1]
    assert "--expected-dispatch" in validate
    assert "5,2,1" in validate
    assert "--expected-graph-fanin" in validate
    assert "0,0,2" in validate
    assert "--expected-graph-dependents" in validate
    assert "2,2" in validate


def test_cuda_pair_persistent_smoke_builds_unary_square_workflow(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        dag_shape="unary_square",
        task_count=3,
        queue_capacity=2,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")
    validate = cuda_pair_persistent_smoke.build_validate_command(config, "abc123")

    assert "persistent-unary_square-smoke-abc123" in str(local)
    assert "--dag-shape" in local
    assert "unary_square" in local
    assert "--dag-shape unary_square" in remote[-1]
    assert "persistent-unary_square-smoke-abc123/h200.json" in remote[-1]
    assert "--expected-dispatch" in validate
    assert "7,1,1" in validate


def test_cuda_pair_persistent_smoke_builds_graph_descriptor_unary_square_workflow(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()
    args = cuda_pair_persistent_smoke.parse_args(
        [
            "--dag-shape",
            "graph_descriptor_unary_square",
            "--repeat-runs",
            "2",
        ]
    )
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        dag_shape=args.dag_shape,
        task_count=3,
        queue_capacity=2,
        repeat_runs=args.repeat_runs,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")
    validate = cuda_pair_persistent_smoke.build_validate_command(config, "abc123")

    assert "persistent-graph_descriptor_unary_square-repeat2-smoke-abc123" in str(local)
    assert "--dag-shape" in local
    assert "graph_descriptor_unary_square" in local
    assert "--dag-shape graph_descriptor_unary_square" in remote[-1]
    assert "persistent-graph_descriptor_unary_square-repeat2-smoke-abc123/h200.json" in remote[-1]
    assert "--expected-dispatch" in validate
    assert "7,1,1" in validate
    assert "--expected-graph-fanin" in validate
    assert "0,1,1" in validate
    assert "--expected-graph-dependents" in validate
    assert "1,2" in validate


def test_cuda_pair_persistent_smoke_builds_graph_descriptor_triad_workflow(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        dag_shape="graph_descriptor_triad",
        task_count=3,
        queue_capacity=2,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")
    validate = cuda_pair_persistent_smoke.build_validate_command(config, "abc123")

    assert "persistent-graph_descriptor_triad-smoke-abc123" in str(local)
    assert "--dag-shape" in local
    assert "graph_descriptor_triad" in local
    assert "--dag-shape graph_descriptor_triad" in remote[-1]
    assert "persistent-graph_descriptor_triad-smoke-abc123/h200.json" in remote[-1]
    assert "--expected-dispatch" in validate
    assert "6,2,1" in validate
    assert "--expected-graph-fanin" in validate
    assert "0,0,2" in validate
    assert "--expected-graph-dependents" in validate
    assert "2,2" in validate


def test_cuda_pair_persistent_smoke_builds_graph_descriptor_quad_workflow(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        dag_shape="graph_descriptor_quad",
        task_count=3,
        queue_capacity=2,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")
    validate = cuda_pair_persistent_smoke.build_validate_command(config, "abc123")

    assert "persistent-graph_descriptor_quad-smoke-abc123" in str(local)
    assert "--dag-shape" in local
    assert "graph_descriptor_quad" in local
    assert "--dag-shape graph_descriptor_quad" in remote[-1]
    assert "persistent-graph_descriptor_quad-smoke-abc123/h200.json" in remote[-1]
    assert "--expected-dispatch" in validate
    assert "8,2,1" in validate
    assert "--expected-graph-fanin" in validate
    assert "0,0,2" in validate
    assert "--expected-graph-dependents" in validate
    assert "2,2" in validate


def test_cuda_pair_persistent_smoke_builds_generic_args_workflow(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        dag_shape="generic_args",
        task_count=3,
        queue_capacity=2,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")
    validate = cuda_pair_persistent_smoke.build_validate_command(config, "abc123")

    assert "persistent-generic_args-smoke-abc123" in str(local)
    assert "--dag-shape" in local
    assert "generic_args" in local
    assert "--dag-shape generic_args" in remote[-1]
    assert "persistent-generic_args-smoke-abc123/h200.json" in remote[-1]
    assert "--expected-dispatch" in validate
    assert "9,2,1" in validate


def test_cuda_pair_persistent_smoke_accepts_generic_args4_workflow(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        dag_shape="generic_args4",
        task_count=3,
        queue_capacity=2,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")
    validate = cuda_pair_persistent_smoke.build_validate_command(config, "abc123")

    assert "persistent-generic_args4-smoke-abc123" in str(local)
    assert "--dag-shape" in local
    assert "generic_args4" in local
    assert "--dag-shape generic_args4" in remote[-1]
    assert "persistent-generic_args4-smoke-abc123/h200.json" in remote[-1]
    assert "--expected-dispatch" in validate
    assert "9,2,1" in validate


def test_cuda_pair_persistent_smoke_accepts_graph_descriptor_generic_args4_workflow(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        dag_shape="graph_descriptor_generic_args4",
        task_count=3,
        queue_capacity=2,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")
    validate = cuda_pair_persistent_smoke.build_validate_command(config, "abc123")

    assert "persistent-graph_descriptor_generic_args4-smoke-abc123" in str(local)
    assert "--dag-shape" in local
    assert "graph_descriptor_generic_args4" in local
    assert "--dag-shape graph_descriptor_generic_args4" in remote[-1]
    assert "persistent-graph_descriptor_generic_args4-smoke-abc123/h200.json" in remote[-1]
    assert "--expected-dispatch" in validate
    assert "9,2,1" in validate


def test_cuda_pair_persistent_smoke_accepts_tagged_graph_descriptor_workflow(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()
    args = cuda_pair_persistent_smoke.parse_args(
        [
            "--dag-shape",
            "graph_descriptor_tagged",
            "--repeat-runs",
            "2",
            "--sync-remote-tree",
        ]
    )
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        dag_shape=args.dag_shape,
        repeat_runs=args.repeat_runs,
        sync_remote_tree=args.sync_remote_tree,
        refresh_remote=not args.skip_remote_refresh and not args.sync_remote_tree,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")
    validate = cuda_pair_persistent_smoke.build_validate_command(config, "abc123")

    assert "persistent-graph_descriptor_tagged-repeat2-smoke-abc123" in str(local)
    assert "graph_descriptor_tagged" in local
    assert "--dag-shape graph_descriptor_tagged" in remote[-1]
    assert "persistent-graph_descriptor_tagged-repeat2-smoke-abc123/h200.json" in remote[-1]
    assert "--expected-dispatch" in validate
    assert "9,2,1" in validate
    assert "--expected-graph-fanin" in validate
    assert "0,0,2" in validate
    assert "--expected-graph-dependents" in validate
    assert "2,2" in validate
    assert "--expected-graph-task-args" in validate
    assert "task0=input:a,input:b,output:tmp1,scalar:scalar_args[0],scalar:scalar_args[1]" in " ".join(validate)
    assert "task2=input:tmp1,input:tmp2,output_existing:out" in " ".join(validate)


def test_cuda_pair_persistent_smoke_accepts_tagged_inout_graph_descriptor_workflow(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()
    args = cuda_pair_persistent_smoke.parse_args(
        [
            "--dag-shape",
            "graph_descriptor_tagged_inout",
            "--repeat-runs",
            "2",
            "--sync-remote-tree",
        ]
    )
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        dag_shape=args.dag_shape,
        repeat_runs=args.repeat_runs,
        sync_remote_tree=args.sync_remote_tree,
        refresh_remote=not args.skip_remote_refresh and not args.sync_remote_tree,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")
    validate = cuda_pair_persistent_smoke.build_validate_command(config, "abc123")

    assert "persistent-graph_descriptor_tagged_inout-repeat2-smoke-abc123" in str(local)
    assert "graph_descriptor_tagged_inout" in local
    assert "--dag-shape graph_descriptor_tagged_inout" in remote[-1]
    assert "persistent-graph_descriptor_tagged_inout-repeat2-smoke-abc123/h200.json" in remote[-1]
    assert "--expected-dispatch" in validate
    assert "1,1,1" in validate
    assert "--expected-graph-fanin" in validate
    assert "0,1,1" in validate
    assert "--expected-graph-dependents" in validate
    assert "1,2" in validate
    assert "--expected-graph-task-args" in validate
    assert "task1=inout:tmp1,input:b" in " ".join(validate)


def test_cuda_pair_persistent_smoke_accepts_graph_descriptor_repeat_runs(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()

    args = cuda_pair_persistent_smoke.parse_args(
        [
            "--dag-shape",
            "graph_descriptor",
            "--repeat-runs",
            "2",
            "--sync-remote-tree",
        ]
    )
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        dag_shape=args.dag_shape,
        repeat_runs=args.repeat_runs,
        sync_remote_tree=args.sync_remote_tree,
        refresh_remote=not args.skip_remote_refresh and not args.sync_remote_tree,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")
    validate = cuda_pair_persistent_smoke.build_validate_command(config, "abc123")

    assert config.dag_shape == "graph_descriptor"
    assert config.repeat_runs == 2
    assert config.sync_remote_tree is True
    assert "persistent-graph_descriptor-repeat2-smoke-abc123" in str(local)
    assert "--dag-shape" in local
    assert "graph_descriptor" in local
    assert "--repeat-runs" in local
    assert "2" in local
    assert "--dag-shape graph_descriptor" in remote[-1]
    assert "--repeat-runs 2" in remote[-1]
    assert "persistent-graph_descriptor-repeat2-smoke-abc123/h200.json" in remote[-1]
    assert "--expected-repeat-runs" in validate
    assert "2" in validate
    assert "--expected-dispatch" in validate
    assert "9,2,1" in validate


def test_cuda_pair_persistent_smoke_accepts_graph_descriptor_chain(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()

    args = cuda_pair_persistent_smoke.parse_args(
        [
            "--dag-shape",
            "graph_descriptor_chain",
            "--task-count",
            "5",
            "--repeat-runs",
            "2",
            "--sync-remote-tree",
        ]
    )
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        dag_shape=args.dag_shape,
        task_count=args.task_count,
        repeat_runs=args.repeat_runs,
        sync_remote_tree=args.sync_remote_tree,
        refresh_remote=not args.skip_remote_refresh and not args.sync_remote_tree,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")
    validate = cuda_pair_persistent_smoke.build_validate_command(config, "abc123")

    assert "persistent-graph_descriptor_chain-repeat2-smoke-abc123" in str(local)
    assert "graph_descriptor_chain" in local
    assert "--dag-shape graph_descriptor_chain" in remote[-1]
    assert "persistent-graph_descriptor_chain-repeat2-smoke-abc123/h200.json" in remote[-1]
    assert "--expected-completed-count" in validate
    assert "5" in validate
    assert "--expected-dispatch" in validate
    assert "1,2,1,2,1" in validate
    assert "--expected-graph-fanin" in validate
    assert "0,0,2,1,1" in validate
    assert "--expected-graph-dependents" in validate
    assert "2,2,3,4" in validate


def test_cuda_pair_persistent_smoke_accepts_reordered_graph_descriptor(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()

    args = cuda_pair_persistent_smoke.parse_args(
        [
            "--dag-shape",
            "graph_descriptor_reordered",
            "--repeat-runs",
            "2",
            "--sync-remote-tree",
        ]
    )
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        dag_shape=args.dag_shape,
        repeat_runs=args.repeat_runs,
        sync_remote_tree=args.sync_remote_tree,
        refresh_remote=not args.skip_remote_refresh and not args.sync_remote_tree,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")
    validate = cuda_pair_persistent_smoke.build_validate_command(config, "abc123")

    assert "persistent-graph_descriptor_reordered-repeat2-smoke-abc123" in str(local)
    assert "graph_descriptor_reordered" in local
    assert "--dag-shape graph_descriptor_reordered" in remote[-1]
    assert "persistent-graph_descriptor_reordered-repeat2-smoke-abc123/h200.json" in remote[-1]
    assert "--expected-repeat-runs" in validate
    assert "--expected-dispatch" in validate
    assert "1,9,2" in validate
    assert "--expected-graph-fanin" in validate
    assert "2,0,0" in validate
    assert "--expected-graph-dependents" in validate
    assert "0,0" in validate


def test_cuda_pair_persistent_smoke_accepts_diamond_graph_descriptor(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()

    args = cuda_pair_persistent_smoke.parse_args(
        [
            "--dag-shape",
            "graph_descriptor_diamond",
            "--repeat-runs",
            "2",
            "--sync-remote-tree",
        ]
    )
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        dag_shape=args.dag_shape,
        repeat_runs=args.repeat_runs,
        sync_remote_tree=args.sync_remote_tree,
        refresh_remote=not args.skip_remote_refresh and not args.sync_remote_tree,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")
    validate = cuda_pair_persistent_smoke.build_validate_command(config, "abc123")

    assert "persistent-graph_descriptor_diamond-repeat2-smoke-abc123" in str(local)
    assert "graph_descriptor_diamond" in local
    assert "--dag-shape graph_descriptor_diamond" in remote[-1]
    assert "persistent-graph_descriptor_diamond-repeat2-smoke-abc123/h200.json" in remote[-1]
    assert "--expected-completed-count" in validate
    assert "5" in validate
    assert "--expected-dispatch" in validate
    assert "9,2,1,2,1" in validate
    assert "--expected-graph-fanin" in validate
    assert "0,0,2,2,2" in validate
    assert "--expected-graph-dependents" in validate
    assert "2,3,2,3,4,4" in validate


def test_cuda_pair_persistent_smoke_accepts_scratch_reuse_graph_descriptor(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()

    args = cuda_pair_persistent_smoke.parse_args(
        [
            "--dag-shape",
            "graph_descriptor_scratch_reuse",
            "--task-count",
            "6",
            "--repeat-runs",
            "2",
            "--sync-remote-tree",
        ]
    )
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        dag_shape=args.dag_shape,
        task_count=args.task_count,
        repeat_runs=args.repeat_runs,
        sync_remote_tree=args.sync_remote_tree,
        refresh_remote=not args.skip_remote_refresh and not args.sync_remote_tree,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")
    validate = cuda_pair_persistent_smoke.build_validate_command(config, "abc123")

    assert "persistent-graph_descriptor_scratch_reuse-repeat2-smoke-abc123" in str(local)
    assert "graph_descriptor_scratch_reuse" in local
    assert "--dag-shape graph_descriptor_scratch_reuse" in remote[-1]
    assert "persistent-graph_descriptor_scratch_reuse-repeat2-smoke-abc123/h200.json" in remote[-1]
    assert "--expected-completed-count" in validate
    assert "6" in validate
    assert "--expected-dispatch" in validate
    assert "1,2,1,2,1,1" in validate
    assert "--expected-graph-fanin" in validate
    assert "0,0,2,1,1,2" in validate
    assert "--expected-graph-dependents" in validate
    assert "2,2,3,4,5,5" in validate
    assert "--expected-scratch-reuse" in validate
    assert "reused_buffer=tmp0,reuse_task=4" in validate


def test_cuda_pair_persistent_smoke_builds_scalar_affine_workflow(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        dag_shape="scalar_affine",
        task_count=3,
        queue_capacity=2,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")
    validate = cuda_pair_persistent_smoke.build_validate_command(config, "abc123")

    assert "persistent-scalar_affine-smoke-abc123" in str(local)
    assert "--dag-shape" in local
    assert "scalar_affine" in local
    assert "--dag-shape scalar_affine" in remote[-1]
    assert "persistent-scalar_affine-smoke-abc123/h200.json" in remote[-1]
    assert "--expected-dispatch" in validate
    assert "5,2,1" in validate


def test_cuda_pair_persistent_smoke_builds_triad_workflow(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        dag_shape="triad",
        task_count=3,
        queue_capacity=2,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")
    validate = cuda_pair_persistent_smoke.build_validate_command(config, "abc123")

    assert "persistent-triad-smoke-abc123" in str(local)
    assert "--dag-shape" in local
    assert "triad" in local
    assert "--dag-shape triad" in remote[-1]
    assert "persistent-triad-smoke-abc123/h200.json" in remote[-1]
    assert "--expected-dispatch" in validate
    assert "6,2,1" in validate


def test_cuda_pair_persistent_smoke_builds_quad_workflow(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()
    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        dag_shape="quad",
        task_count=3,
        queue_capacity=2,
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")
    validate = cuda_pair_persistent_smoke.build_validate_command(config, "abc123")

    assert "persistent-quad-smoke-abc123" in str(local)
    assert "--dag-shape" in local
    assert "quad" in local
    assert "--dag-shape quad" in remote[-1]
    assert "persistent-quad-smoke-abc123/h200.json" in remote[-1]
    assert "--expected-dispatch" in validate
    assert "8,2,1" in validate


def test_cuda_pair_persistent_smoke_can_sync_local_tree_and_dry_run(tmp_path):
    cuda_pair_persistent_smoke = _load_pair_persistent_smoke_module()
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="local123\n", stderr="")

    config = cuda_pair_persistent_smoke.PairedPersistentSmokeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        sync_remote_tree=True,
        dag_shape="scratch_reuse",
        task_count=6,
    )

    commands = cuda_pair_persistent_smoke.run_paired_persistent_smoke(config, runner=fake_runner, dry_run=True)

    assert calls == [(["git", "rev-parse", "--short", "HEAD"], {"check": True, "capture_output": True, "text": True})]
    assert len(commands) == 7
    sync = commands[1]
    assert sync[:3] == ["rsync", "-a", "--delete"]
    assert sync[-2:] == [f"{Path.cwd()}/", "h200-box:/remote/pto-cu/"]
    assert "git fetch" not in commands[2][-1]
    assert "git checkout" not in commands[2][-1]
    assert any("persistent-scratch_reuse-smoke-local123" in part for part in commands[0])
    assert "persistent-scratch_reuse-smoke-local123" in commands[2][-1]
    assert "persistent-scratch_reuse-smoke-local123" in commands[3][1]
    assert any("persistent-scratch_reuse-smoke-local123" in part for part in commands[4])
    assert ".agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py" in commands[5]
    assert "--expected-completed-count" in commands[5]
    assert "6" in commands[5]
    assert "--expected-dispatch" in commands[5]
    assert "1,2,1,2,1,1" in commands[5]
    assert commands[6][-2:] == ["--root", str(tmp_path / "cuda-backend")]


def test_cuda_persistent_lifecycle_matrix_builds_default_workflow(tmp_path):
    cuda_lifecycle_matrix = _load_persistent_lifecycle_matrix_module()
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="abc123\n", stderr="")

    config = cuda_lifecycle_matrix.LifecycleMatrixConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=tmp_path / "cuda-backend",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        sync_remote_tree=True,
    )

    commands = cuda_lifecycle_matrix.run_lifecycle_matrix(config, runner=fake_runner, dry_run=True)
    command_text = "\n".join(" ".join(command) for command in commands)

    assert calls[0] == (["git", "rev-parse", "--short", "HEAD"], {"check": True, "capture_output": True, "text": True})
    assert sum(1 for command in commands if command[:3] == ["rsync", "-a", "--delete"]) == 1
    assert "persistent-direct-repeat2-smoke-abc123/a100.json" in command_text
    assert "persistent-queue-repeat2-smoke-abc123/a100.json" in command_text
    assert "persistent-chain-repeat2-smoke-abc123/a100.json" in command_text
    assert "persistent-graph_descriptor_scratch_reuse-repeat2-smoke-abc123/a100.json" in command_text
    assert "--mode direct" in command_text
    assert "--mode queue" in command_text
    assert "--mode dag" in command_text
    assert "--dag-shape graph_descriptor_scratch_reuse" in command_text
    assert "--worker-blocks-per-task 2" in command_text
    assert "--worker-blocks 2" in command_text
    assert "--stream-id 1" in command_text
    validate_script = ".agents/skills/cuda-backend-eval/scripts/cuda_validate_lifecycle_matrix.py"
    validate_commands = [command for command in commands if validate_script in command]
    assert len(validate_commands) == 1
    matrix_json = tmp_path / "cuda-backend" / "persistent-lifecycle-matrix-abc123" / "cuda-lifecycle-matrix.json"
    assert str(matrix_json) in validate_commands[0]
    assert validate_commands[0][-2:] == ["--preset", "default"]
    assert commands[-2] == validate_commands[0]
    assert commands[-1][-2:] == ["--root", str(tmp_path / "cuda-backend")]
    assert not any("cuda-lifecycle-matrix.md" in part for command in commands for part in command)


def test_cuda_persistent_lifecycle_matrix_validates_written_report(tmp_path):
    cuda_lifecycle_matrix = _load_persistent_lifecycle_matrix_module()
    output_root = tmp_path / "cuda-backend"
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="abc123\n", stderr="")

    def write_payload(directory, completed_count, *, mode, dag_shape=None, dispatch=None):
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "pass",
            "runtime": "persistent_device",
            "mode": mode,
            "n": 1024,
            "device_wall_ns": 4096,
            "host_wall_ns": 8192,
            "repeat_runs": 2,
            "completed_count": completed_count,
            "launch_completed_counts": [completed_count, completed_count],
            "resource_policy": {
                "scheduler_blocks": 1 if mode != "direct" else 0,
                "worker_blocks": 2,
                "worker_blocks_per_task": 1 if mode != "direct" else 2,
                "stream_id": 1,
                "block_dim": 256,
                "grid_dim": 3,
            },
        }
        if dag_shape is not None:
            payload["dag_shape"] = dag_shape
            payload["device_scheduler_errors"] = {"count": 0, "code": 0, "task_id": 0}
        if dispatch is not None:
            payload["dispatch_func_ids"] = dispatch
        for artifact in ("a100", "h200"):
            (directory / f"{artifact}.json").write_text(json.dumps(payload) + "\n")

    write_payload(output_root / "persistent-direct-repeat2-smoke-abc123", 2, mode="direct")
    write_payload(output_root / "persistent-queue-repeat2-smoke-abc123", 4, mode="queue")
    write_payload(
        output_root / "persistent-chain-repeat2-smoke-abc123",
        5,
        mode="dag",
        dag_shape="chain",
        dispatch=[1, 2, 1, 2, 1],
    )
    write_payload(
        output_root / "persistent-graph_descriptor_scratch_reuse-repeat2-smoke-abc123",
        6,
        mode="dag",
        dag_shape="graph_descriptor_scratch_reuse",
        dispatch=[1, 2, 1, 2, 1, 1],
    )
    config = cuda_lifecycle_matrix.LifecycleMatrixConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        output_root=output_root,
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        sync_remote_tree=True,
    )

    commands = cuda_lifecycle_matrix.run_lifecycle_matrix(config, runner=fake_runner, dry_run=False)

    validate_script = ".agents/skills/cuda-backend-eval/scripts/cuda_validate_lifecycle_matrix.py"
    validate_commands = [command for command in commands if validate_script in command]
    assert len(validate_commands) == 1
    validate_command = validate_commands[0]
    assert validate_command[:2] == [
        ".venv/bin/python",
        validate_script,
    ]
    json_path = output_root / "persistent-lifecycle-matrix-abc123" / "cuda-lifecycle-matrix.json"
    assert str(json_path) in validate_command
    assert validate_command[-2:] == ["--preset", "default"]
    assert commands[-2] == validate_command
    assert commands[-1][-2:] == ["--root", str(output_root)]
    assert calls[-2] == (validate_command, {"check": True})


def test_cuda_persistent_lifecycle_matrix_renders_report(tmp_path):
    cuda_lifecycle_matrix = _load_persistent_lifecycle_matrix_module()
    rows = [
        {
            "scenario": "direct",
            "artifact": "a100",
            "mode": "direct",
            "status": "pass",
            "runtime": "persistent_device",
            "n": 1024,
            "device_wall_ns": 4096,
            "host_wall_ns": 8192,
            "repeat_runs": 2,
            "launch_completed_counts": [2, 2],
            "device_scheduler_errors": {"count": 0, "code": 0, "task_id": 0},
            "resource_policy": {
                "scheduler_blocks": 0,
                "worker_blocks": 4,
                "worker_blocks_per_task": 2,
                "stream_id": 1,
                "block_dim": 256,
                "grid_dim": 4,
            },
        },
        {
            "scenario": "dag-chain",
            "artifact": "h200",
            "mode": "dag",
            "dag_shape": "chain",
            "status": "pass",
            "runtime": "persistent_device",
            "n": 1024,
            "device_wall_ns": 2048,
            "host_wall_ns": 4096,
            "repeat_runs": 2,
            "launch_completed_counts": [5, 5],
            "dispatch_func_ids": [1, 2, 1, 2, 1],
            "device_scheduler_errors": {"count": 0, "code": 0, "task_id": 0},
            "resource_policy": {
                "scheduler_blocks": 1,
                "worker_blocks": 2,
                "worker_blocks_per_task": 1,
                "stream_id": 1,
                "block_dim": 256,
                "grid_dim": 3,
            },
        },
        {
            "scenario": "graph-scratch-reuse",
            "artifact": "a100",
            "mode": "dag",
            "dag_shape": "graph_descriptor_scratch_reuse",
            "status": "pass",
            "runtime": "persistent_device",
            "n": 1024,
            "device_wall_ns": 3072,
            "host_wall_ns": 6144,
            "repeat_runs": 2,
            "launch_completed_counts": [6, 6],
            "dispatch_func_ids": [1, 2, 1, 2, 1, 1],
            "device_scheduler_errors": {"count": 0, "code": 0, "task_id": 0},
            "resource_policy": {
                "scheduler_blocks": 1,
                "worker_blocks": 2,
                "worker_blocks_per_task": 1,
                "stream_id": 1,
                "block_dim": 256,
                "grid_dim": 3,
            },
        },
    ]

    markdown_path, svg_path = cuda_lifecycle_matrix.write_lifecycle_report(
        rows,
        tmp_path,
        "lifecycle-test",
    )

    report = markdown_path.read_text()
    assert markdown_path.name == "cuda-lifecycle-matrix.md"
    assert svg_path.name == "cuda-lifecycle-matrix.svg"
    assert "| direct | a100 | pass | persistent_device | direct |" in report
    assert "`sched=0,workers=4,wp=2,stream=1,block=256,grid=4`" in report
    assert "| dag-chain | h200 | pass | persistent_device | dag/chain |" in report
    assert "`1,2,1,2,1`" in report
    assert ("| graph-scratch-reuse | a100 | pass | persistent_device | dag/graph_descriptor_scratch_reuse |") in report
    assert "`1,2,1,2,1,1`" in report
    assert "lifecycle-test" in svg_path.read_text()


def test_find_nvcc_uses_cuda_home_when_nvcc_is_not_on_path(tmp_path, monkeypatch):
    cuda_smoke = _load_smoke_module()
    cuda_home = tmp_path / "cuda-12.8"
    nvcc = cuda_home / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\n")
    nvcc.chmod(0o755)

    monkeypatch.setenv("CUDA_HOME", str(cuda_home))
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(cuda_smoke.shutil, "which", lambda name: None)

    assert cuda_smoke._find_nvcc() == str(nvcc)


def test_cuda_worker_smoke_generates_multiply_task_body():
    cuda_smoke = _load_smoke_module()

    body = cuda_smoke._worker_task_body("mul")

    assert "ctx->out[i] = ctx->a[i] * ctx->b[i];" in body
    assert "ctx->a[i] + ctx->b[i]" not in body


def test_cuda_worker_smoke_generates_scale_task_body():
    cuda_smoke = _load_smoke_module()

    body = cuda_smoke._worker_task_body("scale")

    assert "ctx->out[i] = ctx->a[i] * ctx->alpha;" in body
    assert "ctx->b[i]" not in body


def test_cuda_worker_smoke_generates_square_task_body():
    cuda_smoke = _load_smoke_module()

    body = cuda_smoke._worker_task_body("square")

    assert "ctx->out[i] = ctx->a[i] * ctx->a[i];" in body
    assert "ctx->b[i]" not in body
    assert "ctx->alpha" not in body


def test_cuda_worker_smoke_square_expected_output_uses_float32_rounding():
    cuda_smoke = _load_smoke_module()

    expected = cuda_smoke._worker_expected_output("square", 65536)

    assert expected[-1] == ctypes.c_float(float(65535) * float(65535)).value
    assert expected[-1] != float(65535 * 65535)


def test_cuda_worker_smoke_generates_axpy_task_body():
    cuda_smoke = _load_smoke_module()

    body = cuda_smoke._worker_task_body("axpy")

    assert "ctx->out[i] = ctx->alpha * ctx->a[i] + ctx->b[i];" in body


def test_cuda_worker_smoke_generates_affine_task_body():
    cuda_smoke = _load_smoke_module()

    body = cuda_smoke._worker_task_body("affine")

    assert "ctx->out[i] = ctx->alpha * ctx->a[i] + ctx->beta * ctx->b[i];" in body


def test_cuda_worker_smoke_affine_helpers_use_two_scalars():
    cuda_smoke = _load_smoke_module()

    assert cuda_smoke._worker_expected_output("affine", 4) == [0.0, 2.5, 5.0, 7.5]
    assert "float beta;" in cuda_smoke._worker_context_definition("affine")
    assert cuda_smoke._worker_host_parameters("affine") == (
        "const float *a",
        "const float *b",
        "float *out",
        "float alpha",
        "float beta",
        "unsigned long long n",
    )
    assert cuda_smoke._worker_host_context_initializer("affine") == "a, b, out, alpha, beta, n"
    assert cuda_smoke._worker_host_op("affine") == 5


def test_cuda_worker_smoke_triad_helpers_use_three_tensors():
    cuda_smoke = _load_smoke_module()

    body = cuda_smoke._worker_task_body("triad")

    assert "ctx->out[i] = ctx->a[i] * ctx->b[i] + ctx->c[i];" in body
    assert cuda_smoke._worker_expected_output("triad", 4) == [0.0, 5.0, 14.0, 27.0]
    assert "const float *c;" in cuda_smoke._worker_context_definition("triad")
    assert cuda_smoke._worker_host_parameters("triad") == (
        "const float *a",
        "const float *b",
        "const float *c",
        "float *out",
        "unsigned long long n",
    )
    assert cuda_smoke._worker_host_context_initializer("triad") == "a, b, c, out, n"
    assert cuda_smoke._worker_host_op("triad") == 6


def test_cuda_worker_smoke_generic_args_helpers_use_aux_tensor_and_scalar_slots():
    cuda_smoke = _load_smoke_module()

    body = cuda_smoke._worker_task_body("generic_args")

    assert "ctx->scalar0 * ctx->a[i]" in body
    assert "ctx->tensor0[i]" in body
    assert "ctx->scalar1 * ctx->tensor1[i]" in body
    assert cuda_smoke._worker_expected_output("generic_args", 4) == [0.0, 7.5, 15.0, 22.5]
    assert "const float *tensor0;" in cuda_smoke._worker_context_definition("generic_args")
    assert "float scalar1;" in cuda_smoke._worker_context_definition("generic_args")
    assert cuda_smoke._worker_host_parameters("generic_args") == (
        "const float *a",
        "const float *b",
        "float *out",
        "const float *tensor0",
        "const float *tensor1",
        "float scalar0",
        "float scalar1",
        "unsigned long long n",
    )
    assert cuda_smoke._worker_host_context_initializer("generic_args") == (
        "a, b, out, tensor0, tensor1, scalar0, scalar1, n"
    )
    assert cuda_smoke._worker_host_op("generic_args") == 8


def test_cuda_worker_smoke_generic_args4_helpers_use_all_aux_slots():
    cuda_smoke = _load_smoke_module()

    body = cuda_smoke._worker_task_body("generic_args4")

    assert "ctx->tensor2[i]" in body
    assert "ctx->scalar3 * ctx->tensor3[i]" in body
    assert cuda_smoke._worker_expected_output("generic_args4", 4) == [0.0, 8.5, 17.0, 25.5]
    assert "const float *tensor3;" in cuda_smoke._worker_context_definition("generic_args4")
    assert "float scalar3;" in cuda_smoke._worker_context_definition("generic_args4")
    assert cuda_smoke._worker_host_parameters("generic_args4") == (
        "const float *a",
        "const float *b",
        "float *out",
        "const float *tensor0",
        "const float *tensor1",
        "const float *tensor2",
        "const float *tensor3",
        "float scalar0",
        "float scalar1",
        "float scalar2",
        "float scalar3",
        "unsigned long long n",
    )
    assert cuda_smoke._worker_host_context_initializer("generic_args4") == (
        "a, b, out, tensor0, tensor1, tensor2, tensor3, scalar0, scalar1, scalar2, scalar3, n"
    )
    assert cuda_smoke._worker_host_op("generic_args4") == 9


def test_cuda_smoke_main_writes_output_json(tmp_path, monkeypatch, capsys):
    cuda_smoke = _load_smoke_module()
    output = tmp_path / "smoke.json"

    monkeypatch.setattr(
        cuda_smoke,
        "run_worker_smoke",
        lambda device, n, block_dim, arch, build, op: {
            "status": "pass",
            "runner": "worker",
            "runtime": "host_schedule",
            "mode": f"worker/{op}",
            "op": op,
            "device": device,
            "n": n,
            "block_dim": block_dim,
            "ptx_arch": arch,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cuda_smoke.py",
            "--runner",
            "worker",
            "--op",
            "mul",
            "--device",
            "1",
            "--n",
            "64",
            "--block-dim",
            "32",
            "--arch",
            "compute_90",
            "--no-build",
            "--output-json",
            str(output),
        ],
    )

    cuda_smoke.main()

    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text())
    assert printed == written
    assert written["op"] == "mul"
    assert written["runtime"] == "host_schedule"
    assert written["mode"] == "worker/mul"
    assert written["device"] == 1
    assert written["ptx_arch"] == "compute_90"


def test_cuda_smoke_main_accepts_scale_output_json(tmp_path, monkeypatch, capsys):
    cuda_smoke = _load_smoke_module()
    output = tmp_path / "scale-smoke.json"

    monkeypatch.setattr(
        cuda_smoke,
        "run_worker_smoke",
        lambda device, n, block_dim, arch, build, op: {
            "status": "pass",
            "runner": "worker",
            "runtime": "host_schedule",
            "mode": f"worker/{op}",
            "op": op,
            "device": device,
            "n": n,
            "block_dim": block_dim,
            "ptx_arch": arch,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cuda_smoke.py",
            "--runner",
            "worker",
            "--op",
            "scale",
            "--device",
            "1",
            "--n",
            "64",
            "--block-dim",
            "32",
            "--arch",
            "compute_90",
            "--no-build",
            "--output-json",
            str(output),
        ],
    )

    cuda_smoke.main()

    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text())
    assert printed == written
    assert written["op"] == "scale"
    assert written["mode"] == "worker/scale"


def test_cuda_smoke_main_accepts_square_output_json(tmp_path, monkeypatch, capsys):
    cuda_smoke = _load_smoke_module()
    output = tmp_path / "square-smoke.json"

    monkeypatch.setattr(
        cuda_smoke,
        "run_worker_smoke",
        lambda device, n, block_dim, arch, build, op: {
            "status": "pass",
            "runner": "worker",
            "runtime": "host_schedule",
            "mode": f"worker/{op}",
            "op": op,
            "device": device,
            "n": n,
            "block_dim": block_dim,
            "ptx_arch": arch,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cuda_smoke.py",
            "--runner",
            "worker",
            "--op",
            "square",
            "--device",
            "1",
            "--n",
            "64",
            "--block-dim",
            "32",
            "--arch",
            "compute_90",
            "--no-build",
            "--output-json",
            str(output),
        ],
    )

    cuda_smoke.main()

    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text())
    assert printed == written
    assert written["op"] == "square"
    assert written["mode"] == "worker/square"


def test_cuda_smoke_main_accepts_axpy_output_json(tmp_path, monkeypatch, capsys):
    cuda_smoke = _load_smoke_module()
    output = tmp_path / "axpy-smoke.json"

    monkeypatch.setattr(
        cuda_smoke,
        "run_worker_smoke",
        lambda device, n, block_dim, arch, build, op: {
            "status": "pass",
            "runner": "worker",
            "runtime": "host_schedule",
            "mode": f"worker/{op}",
            "op": op,
            "device": device,
            "n": n,
            "block_dim": block_dim,
            "ptx_arch": arch,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cuda_smoke.py",
            "--runner",
            "worker",
            "--op",
            "axpy",
            "--device",
            "1",
            "--n",
            "64",
            "--block-dim",
            "32",
            "--arch",
            "compute_90",
            "--no-build",
            "--output-json",
            str(output),
        ],
    )

    cuda_smoke.main()

    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text())
    assert printed == written
    assert written["op"] == "axpy"
    assert written["mode"] == "worker/axpy"


def test_cuda_smoke_main_accepts_affine_output_json(tmp_path, monkeypatch, capsys):
    cuda_smoke = _load_smoke_module()
    output = tmp_path / "affine-smoke.json"

    monkeypatch.setattr(
        cuda_smoke,
        "run_worker_smoke",
        lambda device, n, block_dim, arch, build, op: {
            "status": "pass",
            "runner": "worker",
            "runtime": "host_schedule",
            "mode": f"worker/{op}",
            "op": op,
            "device": device,
            "n": n,
            "block_dim": block_dim,
            "ptx_arch": arch,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cuda_smoke.py",
            "--runner",
            "worker",
            "--op",
            "affine",
            "--device",
            "1",
            "--n",
            "64",
            "--block-dim",
            "32",
            "--arch",
            "compute_90",
            "--no-build",
            "--output-json",
            str(output),
        ],
    )

    cuda_smoke.main()

    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text())
    assert printed == written
    assert written["op"] == "affine"
    assert written["mode"] == "worker/affine"


def test_cuda_smoke_main_accepts_triad_output_json(tmp_path, monkeypatch, capsys):
    cuda_smoke = _load_smoke_module()
    output = tmp_path / "triad-smoke.json"

    monkeypatch.setattr(
        cuda_smoke,
        "run_worker_smoke",
        lambda device, n, block_dim, arch, build, op: {
            "status": "pass",
            "runner": "worker",
            "runtime": "host_schedule",
            "mode": f"worker/{op}",
            "op": op,
            "device": device,
            "n": n,
            "block_dim": block_dim,
            "ptx_arch": arch,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cuda_smoke.py",
            "--runner",
            "worker",
            "--op",
            "triad",
            "--device",
            "1",
            "--n",
            "64",
            "--block-dim",
            "32",
            "--arch",
            "compute_90",
            "--no-build",
            "--output-json",
            str(output),
        ],
    )

    cuda_smoke.main()

    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text())
    assert printed == written
    assert written["op"] == "triad"
    assert written["mode"] == "worker/triad"


def test_cuda_smoke_main_accepts_generic_args_output_json(tmp_path, monkeypatch, capsys):
    cuda_smoke = _load_smoke_module()
    output = tmp_path / "generic-args-smoke.json"

    monkeypatch.setattr(
        cuda_smoke,
        "run_worker_smoke",
        lambda device, n, block_dim, arch, build, op: {
            "status": "pass",
            "runner": "worker",
            "runtime": "host_schedule",
            "mode": f"worker/{op}",
            "op": op,
            "device": device,
            "n": n,
            "block_dim": block_dim,
            "ptx_arch": arch,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cuda_smoke.py",
            "--runner",
            "worker",
            "--op",
            "generic_args",
            "--device",
            "1",
            "--n",
            "64",
            "--block-dim",
            "32",
            "--arch",
            "compute_90",
            "--no-build",
            "--output-json",
            str(output),
        ],
    )

    cuda_smoke.main()

    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text())
    assert printed == written
    assert written["op"] == "generic_args"
    assert written["mode"] == "worker/generic_args"


def test_persistent_direct_launch_can_use_multiple_worker_blocks_per_task():
    cuda_persistent_smoke = _load_persistent_smoke_module()
    ptx_buf = ctypes.create_string_buffer(b"ptx\0")

    launch = cuda_persistent_smoke._make_direct_launch(
        ptx_buf=ptx_buf,
        ptx_size=128,
        task_count=3,
        dev_tasks=12345,
        worker_blocks_per_task=4,
    )

    assert launch.scheduler_blocks == 0
    assert launch.worker_blocks == 12
    assert launch.manifest.grid_dim == 12
    assert launch.manifest.version == 2
    assert launch.manifest.stream_id == 0
    assert launch.args.worker_blocks_per_task == 4


def test_persistent_direct_smoke_reuses_prepared_callable_for_repeat_runs(monkeypatch):
    cuda_persistent_smoke = _load_persistent_smoke_module()
    run_timings = []

    class FakeRuntime:
        def __init__(self):
            self.next_ptr = 1000
            self.run_calls = 0

        def create_device_context(self):
            return 1

        def simpler_init(self, *args):
            return 0

        def device_malloc_ctx(self, ctx, size):
            ptr = self.next_ptr
            self.next_ptr += max(8, int(size))
            return ptr

        def copy_to_device_ctx(self, *args):
            return 0

        def prepare_callable(self, *args):
            return 0

        def run_prepared(self, *args):
            self.run_calls += 1
            timing = args[-1]._obj
            timing.host_wall_ns = 10 * self.run_calls
            timing.device_wall_ns = 5 * self.run_calls
            run_timings.append((timing.host_wall_ns, timing.device_wall_ns))
            return 0

        def copy_from_device_ctx(self, ctx, dst, src, size):
            value_count = int(size) // ctypes.sizeof(ctypes.c_float)
            expected_t = ctypes.c_float * value_count
            expected = expected_t(*[float(3 * i) for i in range(value_count)])
            ctypes.memmove(dst, ctypes.byref(expected), size)
            return 0

        def unregister_callable(self, *args):
            return 0

        def device_free_ctx(self, *args):
            return 0

        def finalize_device(self, *args):
            return 0

        def destroy_device_context(self, *args):
            return 0

    fake_runtime = FakeRuntime()
    fake_binaries = type("FakeBinaries", (), {"host_path": Path("libhost_runtime.so")})()
    monkeypatch.setattr(
        cuda_persistent_smoke,
        "_compile_persistent_ptx",
        lambda work_dir, arch, mode: (b"ptx", f"fake-{mode}-{arch}", None),
    )
    monkeypatch.setattr(cuda_persistent_smoke, "_load_persistent_runtime", lambda: (fake_runtime, fake_binaries))

    result = cuda_persistent_smoke.run_persistent_smoke(
        device=0,
        task_count=2,
        n=8,
        arch="compute_80",
        mode="direct",
        repeat_runs=3,
    )

    assert fake_runtime.run_calls == 3
    assert result["repeat_runs"] == 3
    assert result["launch_completed_counts"] == [2, 2, 2]
    assert result["launch_host_wall_ns"] == [10, 20, 30]
    assert result["launch_device_wall_ns"] == [5, 10, 15]
    assert result["host_wall_ns"] == 60
    assert result["device_wall_ns"] == 30
    assert run_timings == [(10, 5), (20, 10), (30, 15)]


def test_tensor_tile_dag_shape_uses_caller_provided_descriptor():
    cuda_persistent_smoke = _load_persistent_smoke_module()

    descriptor = cuda_persistent_smoke._make_tensor_tile_descriptor(rows=8, cols=4, inner=12)
    lengths = cuda_persistent_smoke._tensor_tile_buffer_lengths(n=64, descriptor=descriptor)
    _, _, tasks = cuda_persistent_smoke._make_dag_shape(
        "tensor_tile",
        64,
        101,
        102,
        201,
        202,
        203,
        204,
        301,
        tensor_tile=descriptor,
    )

    assert descriptor == {
        "rows": 8,
        "cols": 4,
        "inner": 12,
        "lda": 12,
        "ldb": 4,
        "ldc": 4,
        "a_batch_stride": 96,
        "b_batch_stride": 48,
        "out_batch_stride": 32,
    }
    assert lengths == {"a": 192, "b": 96, "output": 64, "tile_count": 2}
    assert tasks[0].rows == 8
    assert tasks[0].cols == 4
    assert tasks[0].inner == 12
    assert tasks[0].a_batch_stride == 96
    assert tasks[0].b_batch_stride == 48
    assert tasks[0].out_batch_stride == 32


def test_tensor_core_tile_dag_shape_uses_block_wide_wmma_task():
    cuda_persistent_smoke = _load_persistent_smoke_module()

    descriptor = cuda_persistent_smoke._make_tensor_tile_descriptor(rows=16, cols=16, inner=16)
    _, _, tasks = cuda_persistent_smoke._make_dag_shape(
        "tensor_core_tile",
        256,
        101,
        102,
        201,
        202,
        203,
        204,
        301,
        tensor_tile=descriptor,
    )

    assert [task.func_id for task in tasks] == [10, 1, 2, 1]
    assert tasks[0].rows == 16
    assert tasks[0].cols == 16
    assert tasks[0].inner == 16


def test_graph_tensor_core_tile_dag_shape_uses_block_wide_wmma_task():
    cuda_persistent_smoke = _load_persistent_smoke_module()

    descriptor = cuda_persistent_smoke._make_tensor_tile_descriptor(rows=16, cols=16, inner=16)
    fanin, dependents, tasks = cuda_persistent_smoke._make_dag_shape(
        "graph_tensor_core_tile",
        256,
        101,
        102,
        201,
        202,
        203,
        204,
        301,
        tensor_tile=descriptor,
    )

    assert list(fanin) == [0, 1, 1, 2]
    assert list(dependents) == [1, 2, 3, 3]
    assert [task.func_id for task in tasks] == [10, 1, 2, 1]
    assert [task.initial_fanin for task in tasks] == [0, 1, 1, 2]
    assert tasks[0].rows == 16
    assert tasks[0].cols == 16
    assert tasks[0].inner == 16
    assert tasks[3].out == 301


def test_scalar_affine_dag_shape_uses_two_scalar_descriptor_fields():
    cuda_persistent_smoke = _load_persistent_smoke_module()

    _, _, tasks = cuda_persistent_smoke._make_dag_shape(
        "scalar_affine",
        64,
        101,
        102,
        201,
        202,
        203,
        204,
        301,
    )

    assert [task.func_id for task in tasks] == [5, 2, 1]
    assert tasks[0].scalar0 == ctypes.c_float(1.5).value
    assert tasks[0].scalar1 == ctypes.c_float(0.5).value


def test_triad_dag_shape_uses_third_tensor_descriptor_field():
    cuda_persistent_smoke = _load_persistent_smoke_module()

    _, _, tasks = cuda_persistent_smoke._make_dag_shape(
        "triad",
        64,
        101,
        102,
        201,
        202,
        203,
        204,
        301,
    )

    assert [task.func_id for task in tasks] == [6, 2, 1]
    assert tasks[0].a == 101
    assert tasks[0].b == 102
    assert tasks[0].c == 201
    assert tasks[0].out == 202


def test_quad_dag_shape_uses_two_extra_tensor_descriptor_fields():
    cuda_persistent_smoke = _load_persistent_smoke_module()

    _, _, tasks = cuda_persistent_smoke._make_dag_shape(
        "quad",
        64,
        101,
        102,
        201,
        202,
        203,
        204,
        301,
    )

    assert [task.func_id for task in tasks] == [8, 2, 1]
    assert tasks[0].a == 101
    assert tasks[0].b == 102
    assert tasks[0].c == 201
    assert tasks[0].d == 204
    assert tasks[0].out == 202


def test_quad_dag_expected_value_matches_cuda_fused_multiply_add():
    cuda_persistent_smoke = _load_persistent_smoke_module()

    i = 4383
    a = cuda_persistent_smoke._f32(i)
    b = cuda_persistent_smoke._f32(2 * i)
    c = cuda_persistent_smoke._f32(3 * i)
    d = cuda_persistent_smoke._f32(4 * i)
    separately_rounded = cuda_persistent_smoke._f32(
        cuda_persistent_smoke._f32(a * b) + cuda_persistent_smoke._f32(c * d)
    )

    expected = cuda_persistent_smoke._fma_f32(a, b, cuda_persistent_smoke._f32(c * d))

    assert expected != separately_rounded
    assert expected == 268949664.0


def test_unary_square_dag_shape_uses_single_input_task_body():
    cuda_persistent_smoke = _load_persistent_smoke_module()

    host_fanin, dependents, tasks = cuda_persistent_smoke._make_dag_shape(
        "unary_square",
        64,
        101,
        102,
        201,
        202,
        203,
        204,
        301,
    )

    assert list(host_fanin) == [0, 1, 1]
    assert list(dependents) == [1, 2]
    assert [task.func_id for task in tasks] == [7, 1, 1]
    assert tasks[0].a == 101
    assert tasks[0].b is None
    assert tasks[0].out == 201
    assert tasks[1].a == 201
    assert tasks[1].b == 102
    assert tasks[2].a == 202
    assert tasks[2].b == 101


def test_graph_descriptor_unary_square_dag_shape_uses_single_input_task_body():
    cuda_persistent_smoke = _load_persistent_smoke_module()

    host_fanin, dependents, tasks = cuda_persistent_smoke._make_dag_shape(
        "graph_descriptor_unary_square",
        64,
        101,
        102,
        201,
        202,
        203,
        204,
        301,
    )

    assert list(host_fanin) == [0, 1, 1]
    assert list(dependents) == [1, 2]
    assert [task.func_id for task in tasks] == [7, 1, 1]
    assert tasks[0].a == 101
    assert tasks[0].b is None
    assert tasks[0].out == 201
    assert tasks[1].a == 201
    assert tasks[1].b == 102
    assert tasks[2].a == 202
    assert tasks[2].b == 101


def test_diamond_graph_descriptor_dag_shape_fans_out_to_two_consumers():
    cuda_persistent_smoke = _load_persistent_smoke_module()

    host_fanin, dependents, tasks = cuda_persistent_smoke._make_dag_shape(
        "graph_descriptor_diamond",
        64,
        101,
        102,
        201,
        202,
        203,
        204,
        301,
    )

    assert list(host_fanin) == [0, 0, 2, 2, 2]
    assert list(dependents) == [2, 3, 2, 3, 4, 4]
    assert [task.func_id for task in tasks] == [9, 2, 1, 2, 1]
    assert [task.dependent_count for task in tasks] == [2, 2, 1, 1, 0]
    assert tasks[0].out == 202
    assert tasks[1].out == 203
    assert tasks[2].a == 202
    assert tasks[2].b == 203
    assert tasks[2].out == 201
    assert tasks[3].a == 202
    assert tasks[3].b == 203
    assert tasks[3].out == 204
    assert tasks[4].a == 201
    assert tasks[4].b == 204
    assert tasks[4].out == 301


def test_scratch_reuse_graph_descriptor_reuses_temporary_after_last_use():
    cuda_persistent_smoke = _load_persistent_smoke_module()

    host_fanin, dependents, tasks = cuda_persistent_smoke._make_dag_shape(
        "graph_descriptor_scratch_reuse",
        64,
        101,
        102,
        201,
        202,
        203,
        204,
        301,
    )

    assert list(host_fanin) == [0, 0, 2, 1, 1, 2]
    assert list(dependents) == [2, 2, 3, 4, 5, 5]
    assert [task.func_id for task in tasks] == [1, 2, 1, 2, 1, 1]
    assert [task.dependent_count for task in tasks] == [1, 1, 2, 1, 1, 0]
    assert tasks[0].out == 201
    assert tasks[2].out == 203
    assert tasks[4].out == 201
    assert tasks[5].a == 201
    assert tasks[5].b == 204
    assert tasks[5].out == 301


def test_bad_no_root_dag_shape_has_no_initial_ready_tasks():
    cuda_persistent_smoke = _load_persistent_smoke_module()

    host_fanin, dependents, tasks = cuda_persistent_smoke._make_dag_shape(
        "bad_no_root",
        64,
        101,
        102,
        201,
        202,
        203,
        204,
        301,
    )

    assert list(host_fanin) == [1]
    assert list(dependents) == [0]
    assert len(tasks) == 1
    assert tasks[0].initial_fanin == 1
    assert tasks[0].dependent_count == 0


def test_persistent_dag_compiler_path_uses_kernel_compiler(tmp_path, monkeypatch):
    cuda_persistent_smoke = _load_persistent_smoke_module()
    seen = {}

    class FakeKernelCompiler:
        def __init__(self, platform):
            seen["platform"] = platform

        def compile_cuda_persistent_device(self, task_sources, **kwargs):
            seen["task_sources"] = task_sources
            seen.update(kwargs)
            return type(
                "Artifact",
                (),
                {
                    "ptx": b"persistent-dag-ptx",
                    "source_kind": "generated-dispatch",
                },
            )()

    monkeypatch.setattr(cuda_persistent_smoke, "_find_nvcc", lambda: "/usr/local/cuda/bin/nvcc")
    monkeypatch.setattr(cuda_persistent_smoke, "KernelCompiler", FakeKernelCompiler)

    ptx, source_kind, artifact = cuda_persistent_smoke._compile_persistent_dag_ptx(tmp_path, "compute_90")

    assert ptx == b"persistent-dag-ptx"
    assert source_kind == "nvcc-persistent-generated-dispatch-compute_90"
    assert artifact.ptx == b"persistent-dag-ptx"
    assert seen["platform"] == "cuda"
    assert seen["arch"] == "compute_90"
    assert seen["nvcc"] == "/usr/local/cuda/bin/nvcc"
    assert [task["func_id"] for task in seen["task_sources"]] == [1, 2, 3, 4, 11, 5, 6, 7, 8, 9, 10]
    assert [task["task_name"] for task in seen["task_sources"]] == [
        "add_f32",
        "mul_f32",
        "matmul_tile_f32",
        "axpy_f32",
        "scale_f32",
        "affine_f32",
        "triad_f32",
        "square_f32",
        "quad_f32",
        "generic_args_f32",
        "wmma_m16n16k8_f32",
    ]
    assert {task.get("body_style", "raw") for task in seen["task_sources"]} == {"raw", "task_body"}
    task_body_sources = [task for task in seen["task_sources"] if task.get("body_style") == "task_body"]
    raw_sources = [task for task in seen["task_sources"] if task.get("body_style") != "task_body"]
    assert all("PtoCudaPersistentDagTask" in task["context_definition"] for task in task_body_sources)
    assert raw_sources == [
        {
            "func_id": 10,
            "task_name": "wmma_m16n16k8_f32",
            "source_path": raw_sources[0]["source_path"],
            "threading": "block",
        }
    ]
    assert all(Path(task["source_path"]).is_file() for task in seen["task_sources"])


def test_cuda_current_summary_renders_launch_table():
    cuda_current_summary = _load_current_summary_module()
    payload = {
        "results": [
            {"machine": "hina", "baseline": "pto_host_schedule", "n": 1024, "device_wall_ns": 1000},
            {"machine": "hina", "baseline": "pto_host_schedule_compiler", "n": 1024, "device_wall_ns": 900},
            {"machine": "hina", "baseline": "direct_driver", "n": 1024, "device_wall_ns": 1200},
            {"machine": "hina", "baseline": "direct_driver_graph", "n": 1024, "device_wall_ns": 500},
            {"machine": "dasys-h200x8", "baseline": "pto_host_schedule", "n": 1024, "device_wall_ns": 800},
            {
                "machine": "dasys-h200x8",
                "baseline": "pto_host_schedule_compiler",
                "n": 1024,
                "device_wall_ns": 840,
            },
            {"machine": "dasys-h200x8", "baseline": "direct_driver", "n": 1024, "device_wall_ns": 880},
            {"machine": "dasys-h200x8", "baseline": "direct_driver_graph", "n": 1024, "device_wall_ns": 400},
        ],
    }

    table = cuda_current_summary.render_launch_table(payload)

    assert "| GPU | N | PTO host ns | Compiler ns | Driver ns | Graph ns | Compiler/PTO | Graph/PTO |" in table
    assert "| A100 | 1024 | 1000 | 900 | 1200 | 500 | 0.90x | 0.50x |" in table
    assert "| H200 | 1024 | 800 | 840 | 880 | 400 | 1.05x | 0.50x |" in table


def test_cuda_current_summary_renders_unary_square_table():
    cuda_current_summary = _load_current_summary_module()
    payload = {
        "results": [
            {
                "machine": "hina",
                "baseline": "pto_host_schedule_unary_square",
                "n": 65536,
                "device_wall_ns": 3000,
            },
            {
                "machine": "hina",
                "baseline": "pto_host_schedule_unary_square",
                "n": 65536,
                "device_wall_ns": 1000,
            },
            {
                "machine": "hina",
                "baseline": "pto_host_schedule_unary_square",
                "n": 65536,
                "device_wall_ns": 2000,
            },
            {
                "machine": "hina",
                "baseline": "pto_host_schedule_quad",
                "n": 65536,
                "device_wall_ns": 2500,
            },
        ],
    }

    table = cuda_current_summary.render_unary_square_table(payload)

    assert "| GPU | N | Unary square ns | Quad ns |" in table
    assert "| A100 | 65536 | 2000 | 2500 |" in table


def test_cuda_current_summary_renders_worker_and_dag_tables():
    cuda_current_summary = _load_current_summary_module()
    payload = {
        "results": [
            {
                "machine": "hina",
                "baseline": "pto_host_schedule_batch",
                "n": 65536,
                "task_count": 6,
                "device_wall_ns": 10000,
            },
            {
                "machine": "hina",
                "baseline": "pto_persistent_device_grid_batch",
                "n": 65536,
                "task_count": 6,
                "worker_blocks_per_task": 32,
                "device_wall_ns": 7000,
            },
            {
                "machine": "hina",
                "baseline": "pto_persistent_device_grid_batch",
                "n": 65536,
                "task_count": 6,
                "worker_blocks_per_task": 128,
                "device_wall_ns": 3000,
            },
            {"machine": "hina", "baseline": "pto_persistent_dag", "n": 65536, "task_count": 3, "device_wall_ns": 2000},
            {
                "machine": "hina",
                "baseline": "pto_persistent_dag_chain",
                "n": 65536,
                "task_count": 5,
                "device_wall_ns": 3000,
            },
            {
                "machine": "hina",
                "baseline": "pto_persistent_dag_reuse",
                "n": 65536,
                "task_count": 6,
                "device_wall_ns": 4000,
            },
            {
                "machine": "hina",
                "baseline": "pto_persistent_dag_scalar_axpy",
                "n": 65536,
                "task_count": 3,
                "device_wall_ns": 2500,
            },
            {
                "machine": "hina",
                "baseline": "pto_persistent_dag_scalar_scale",
                "n": 65536,
                "task_count": 3,
                "device_wall_ns": 2450,
            },
            {
                "machine": "hina",
                "baseline": "pto_persistent_dag_scalar_affine",
                "n": 65536,
                "task_count": 3,
                "device_wall_ns": 2600,
            },
            {
                "machine": "hina",
                "baseline": "pto_persistent_dag_triad",
                "n": 65536,
                "task_count": 3,
                "device_wall_ns": 2700,
            },
            {
                "machine": "hina",
                "baseline": "pto_persistent_dag_quad",
                "n": 65536,
                "task_count": 3,
                "device_wall_ns": 2800,
            },
            {
                "machine": "hina",
                "baseline": "pto_persistent_dag_generic_args",
                "n": 65536,
                "task_count": 3,
                "device_wall_ns": 2200,
            },
            {
                "machine": "hina",
                "baseline": "pto_persistent_dag_graph",
                "n": 65536,
                "task_count": 3,
                "device_wall_ns": 2300,
            },
            {
                "machine": "hina",
                "baseline": "pto_persistent_dag_graph_scratch_reuse",
                "n": 65536,
                "task_count": 6,
                "device_wall_ns": 3600,
            },
            {
                "machine": "hina",
                "baseline": "pto_persistent_dag_unary_square",
                "n": 65536,
                "task_count": 3,
                "device_wall_ns": 2400,
            },
            {
                "machine": "hina",
                "baseline": "pto_persistent_dag_tensor",
                "n": 65536,
                "task_count": 4,
                "device_wall_ns": 5000,
            },
        ],
    }

    worker_table = cuda_current_summary.render_worker_grid_table(payload)
    dag_table = cuda_current_summary.render_dag_shape_table(payload)

    assert "| A100 | 65536 | 6 | 128 | 3000 | 0.30x |" in worker_table
    assert (
        "| A100 | 65536 | 1.50x | 2.00x | 1.25x | 1.23x | 1.30x | 1.35x | 1.40x | "
        "1.10x | 1.15x | - | 1.80x | - | 1.20x | 2.50x |"
    ) in dag_table


def test_cuda_current_summary_keeps_old_captures_without_scalar_affine():
    cuda_current_summary = _load_current_summary_module()
    payload = {
        "results": [
            {"machine": "hina", "baseline": "pto_persistent_dag", "n": 65536, "task_count": 3, "device_wall_ns": 2000},
            {
                "machine": "hina",
                "baseline": "pto_persistent_dag_chain",
                "n": 65536,
                "task_count": 5,
                "device_wall_ns": 3000,
            },
            {
                "machine": "hina",
                "baseline": "pto_persistent_dag_reuse",
                "n": 65536,
                "task_count": 6,
                "device_wall_ns": 4000,
            },
            {
                "machine": "hina",
                "baseline": "pto_persistent_dag_scalar_axpy",
                "n": 65536,
                "task_count": 3,
                "device_wall_ns": 2500,
            },
            {
                "machine": "hina",
                "baseline": "pto_persistent_dag_tensor",
                "n": 65536,
                "task_count": 4,
                "device_wall_ns": 5000,
            },
        ],
    }

    dag_table = cuda_current_summary.render_dag_shape_table(payload)

    assert (
        "| GPU | N | Chain/DAG | Reuse/DAG | Scalar AXPY/DAG | Scalar Scale/DAG | Scalar Affine/DAG | "
        "Triad/DAG | Quad/DAG | Generic Args/DAG | Graph Descriptor/DAG | Graph Diamond/DAG | "
        "Graph Scratch Reuse/DAG | Graph Tagged Inout/DAG | Unary Square/DAG | "
        "Tensor/DAG |"
    ) in dag_table
    assert "| A100 | 65536 | 1.50x | 2.00x | 1.25x | - | - | - | - | - | - | - | - | - | - | 2.50x |" in dag_table


def test_cuda_current_summary_renders_tensor_sweep_table():
    cuda_current_summary = _load_current_summary_module()
    payload = {
        "results": [
            {
                "artifact": "a100",
                "machine": "a100",
                "baseline": "pto_persistent_dag_tensor",
                "n": 256,
                "shape": "16x16x16",
                "device_wall_ns": 1000,
                "status": "pass",
            },
            {
                "artifact": "a100",
                "machine": "a100",
                "baseline": "pto_persistent_dag_tensor",
                "n": 256,
                "shape": "16x16x16",
                "device_wall_ns": 1200,
                "status": "pass",
            },
            {
                "artifact": "a100",
                "machine": "a100",
                "baseline": "pto_persistent_dag_tensor_core",
                "n": 256,
                "shape": "16x16x16",
                "device_wall_ns": 900,
                "status": "pass",
            },
            {
                "artifact": "a100",
                "machine": "a100",
                "baseline": "pto_persistent_dag_graph_tensor_core",
                "n": 256,
                "shape": "16x16x16",
                "device_wall_ns": 1000,
                "status": "pass",
            },
            {
                "artifact": "a100",
                "machine": "a100",
                "baseline": "pto_persistent_dag_graph_tensor",
                "n": 256,
                "shape": "16x16x16",
                "device_wall_ns": 1300,
                "status": "pass",
            },
            {
                "artifact": "a100",
                "machine": "a100",
                "baseline": "cublas_sgemm",
                "n": 256,
                "shape": "16x16x16",
                "device_wall_ns": 1500,
                "status": "pass",
            },
            {
                "artifact": "a100",
                "machine": "a100",
                "baseline": "cublas_sgemm_graph",
                "n": 256,
                "shape": "16x16x16",
                "device_wall_ns": 500,
                "status": "pass",
            },
            {
                "artifact": "h200",
                "machine": "h200",
                "baseline": "pto_persistent_dag_tensor",
                "n": 256,
                "shape": "16x16x16",
                "device_wall_ns": 800,
                "status": "pass",
            },
            {
                "artifact": "h200",
                "machine": "h200",
                "baseline": "pto_persistent_dag_tensor_core",
                "n": 256,
                "shape": "16x16x16",
                "device_wall_ns": 1000,
                "status": "pass",
            },
            {
                "artifact": "h200",
                "machine": "h200",
                "baseline": "pto_persistent_dag_graph_tensor_core",
                "n": 256,
                "shape": "16x16x16",
                "device_wall_ns": 1200,
                "status": "pass",
            },
            {
                "artifact": "h200",
                "machine": "h200",
                "baseline": "pto_persistent_dag_graph_tensor",
                "n": 256,
                "shape": "16x16x16",
                "device_wall_ns": 700,
                "status": "pass",
            },
            {
                "artifact": "h200",
                "machine": "h200",
                "baseline": "cublas_sgemm",
                "n": 256,
                "shape": "16x16x16",
                "device_wall_ns": 1600,
                "status": "pass",
            },
            {
                "artifact": "h200",
                "machine": "h200",
                "baseline": "cublas_sgemm_graph",
                "n": 256,
                "shape": "16x16x16",
                "device_wall_ns": 400,
                "status": "pass",
            },
        ]
    }

    table = cuda_current_summary.render_tensor_sweep_table(payload)

    assert (
        "| GPU | N | Shape | Scalar tensor ns | Graph tensor ns | Tensor-core ns | Graph tensor-core ns | "
        "cuBLAS ns | cuBLAS Graph ns | Scalar GF/s | Graph tensor GF/s | Tensor-core GF/s | "
        "Graph tensor-core GF/s | cuBLAS GF/s | cuBLAS Graph GF/s | Graph/scalar | Tensor-core/scalar | "
        "Graph tensor-core/scalar | cuBLAS/scalar | cuBLAS Graph/scalar |" in table
    )
    assert (
        "| A100 | 256 | 16x16x16 | 1100 | 1300 | 900 | 1000 | 1500 | 500 | 7.45 | 6.30 | 9.10 | 8.19 | "
        "5.46 | 16.38 | 1.18x | 0.82x | 0.91x | 1.36x | 0.45x |" in table
    )
    assert (
        "| H200 | 256 | 16x16x16 | 800 | 700 | 1000 | 1200 | 1600 | 400 | 10.24 | 11.70 | 8.19 | 6.83 | "
        "5.12 | 20.48 | 0.88x | 1.25x | 1.50x | 2.00x | 0.50x |" in table
    )


def test_cuda_current_summary_renders_benchmark_tensor_throughput_table():
    cuda_current_summary = _load_current_summary_module()
    payload = {
        "results": [
            {
                "machine": "hina",
                "baseline": "pto_persistent_dag_tensor",
                "n": 512,
                "device_wall_ns": 2048,
                "tensor_tile": {"rows": 16, "cols": 16, "inner": 16, "tile_count": 2},
                "status": "pass",
            },
            {
                "machine": "hina",
                "baseline": "pto_persistent_dag_tensor_core",
                "n": 512,
                "device_wall_ns": 1024,
                "tensor_tile": {"rows": 16, "cols": 16, "inner": 16, "tile_count": 2},
                "status": "pass",
            },
            {
                "machine": "hina",
                "baseline": "pto_persistent_dag_graph_tensor_core",
                "n": 512,
                "device_wall_ns": 1536,
                "tensor_tile": {"rows": 16, "cols": 16, "inner": 16, "tile_count": 2},
                "status": "pass",
            },
            {
                "machine": "hina",
                "baseline": "pto_persistent_dag_graph_tensor",
                "n": 512,
                "device_wall_ns": 4096,
                "tensor_tile": {"rows": 16, "cols": 16, "inner": 16, "tile_count": 2},
                "status": "pass",
            },
            {
                "machine": "hina",
                "baseline": "cublas_sgemm",
                "n": 512,
                "device_wall_ns": 8192,
                "tensor_tile": {"rows": 16, "cols": 16, "inner": 16, "tile_count": 2},
                "status": "pass",
            },
            {
                "machine": "hina",
                "baseline": "cublas_sgemm_graph",
                "n": 512,
                "device_wall_ns": 6144,
                "tensor_tile": {"rows": 16, "cols": 16, "inner": 16, "tile_count": 2},
                "status": "pass",
            },
            {
                "machine": "dasys-h200x8",
                "baseline": "pto_persistent_dag_tensor",
                "n": 512,
                "device_wall_ns": 1024,
                "tensor_tile": {"rows": 16, "cols": 16, "inner": 16, "tile_count": 2},
                "status": "pass",
            },
            {
                "machine": "dasys-h200x8",
                "baseline": "pto_persistent_dag_tensor_core",
                "n": 512,
                "device_wall_ns": 2048,
                "tensor_tile": {"rows": 16, "cols": 16, "inner": 16, "tile_count": 2},
                "status": "pass",
            },
            {
                "machine": "dasys-h200x8",
                "baseline": "pto_persistent_dag_graph_tensor_core",
                "n": 512,
                "device_wall_ns": 2560,
                "tensor_tile": {"rows": 16, "cols": 16, "inner": 16, "tile_count": 2},
                "status": "pass",
            },
            {
                "machine": "dasys-h200x8",
                "baseline": "cublas_sgemm",
                "n": 512,
                "device_wall_ns": 4096,
                "tensor_tile": {"rows": 16, "cols": 16, "inner": 16, "tile_count": 2},
                "status": "pass",
            },
            {
                "machine": "dasys-h200x8",
                "baseline": "cublas_sgemm_graph",
                "n": 512,
                "device_wall_ns": 3072,
                "tensor_tile": {"rows": 16, "cols": 16, "inner": 16, "tile_count": 2},
                "status": "pass",
            },
        ]
    }

    table = cuda_current_summary.render_benchmark_tensor_throughput_table(payload)

    assert (
        "| GPU | N | Shape | Scalar ns | Graph ns | Tensor-core ns | Graph tensor-core ns | cuBLAS ns | "
        "cuBLAS graph ns | Scalar GF/s | Graph GF/s | Tensor-core GF/s | Graph tensor-core GF/s | "
        "cuBLAS GF/s | cuBLAS graph GF/s | Tensor-core/scalar | Graph tensor-core/scalar | "
        "cuBLAS/scalar | cuBLAS graph/scalar |"
    ) in table
    assert (
        "| --- | - | ----- | --------- | -------- | -------------- | -------------------- | --------- | "
        "--------------- | ----------- | ---------- | ---------------- | ---------------------- | ----------- | "
        "----------------- | ------------------ | ------------------------ | ------------- | ------------------- |"
    ) in table
    assert (
        "| A100 | 512 | 16x16x16 | 2048 | 4096 | 1024 | 1536 | 8192 | 6144 | 8.00 | 4.00 | 16.00 | 10.67 | "
        "2.00 | 2.67 | 0.50x | 0.75x | 4.00x | 3.00x |" in table
    )
    assert (
        "| H200 | 512 | 16x16x16 | 1024 | - | 2048 | 2560 | 4096 | 3072 | 16.00 | - | 8.00 | 6.40 | "
        "4.00 | 5.33 | 2.00x | 2.50x | 4.00x | 3.00x |" in table
    )


def test_cuda_current_summary_renders_graph_tensor_core_without_scalar_reference():
    cuda_current_summary = _load_current_summary_module()
    payload = {
        "results": [
            {
                "artifact": "a100",
                "machine": "a100",
                "baseline": "pto_persistent_dag_graph_tensor_core",
                "n": 256,
                "shape": "16x16x16",
                "device_wall_ns": 1000,
                "status": "pass",
            }
        ]
    }

    table = cuda_current_summary.render_tensor_sweep_table(payload)

    assert "| A100 | 256 | 16x16x16 | - | - | - | 1000 | - | - | - | - | - | 8.19 | - | - | - | - | - | - |" in table


def test_summarize_results_groups_by_machine_and_baseline():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "results": [
            {"machine": "a100-local", "baseline": "pto_host_schedule", "n": 1024, "device_wall_ns": 1000},
            {"machine": "a100-local", "baseline": "pto_host_schedule", "n": 1024, "device_wall_ns": 2000},
            {"machine": "a100-local", "baseline": "direct_driver", "n": 1024, "device_wall_ns": 500},
            {"machine": "h200-remote", "baseline": "pto_host_schedule", "n": 2048, "device_wall_ns": 800},
        ]
    }

    summary = cuda_benchmark.summarize_results(payload)

    assert summary[("a100-local", "pto_host_schedule", 1024, 1, 1)]["median_device_wall_ns"] == 1500
    assert summary[("a100-local", "direct_driver", 1024, 1, 1)]["samples"] == 1
    assert summary[("h200-remote", "pto_host_schedule", 2048, 1, 1)]["median_device_wall_ns"] == 800


def test_summarize_results_separates_worker_blocks_per_task():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "results": [
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_device_grid_batch",
                "n": 1024,
                "task_count": 6,
                "worker_blocks_per_task": 2,
                "device_wall_ns": 5000,
            },
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_device_grid_batch",
                "n": 1024,
                "task_count": 6,
                "worker_blocks_per_task": 4,
                "device_wall_ns": 3000,
            },
        ]
    }

    summary = cuda_benchmark.summarize_results(payload)
    report = cuda_benchmark.render_markdown_report(payload)

    assert summary[("a100-local", "pto_persistent_device_grid_batch", 1024, 6, 2)]["median_device_wall_ns"] == 5000
    assert summary[("a100-local", "pto_persistent_device_grid_batch", 1024, 6, 4)]["median_device_wall_ns"] == 3000
    assert "| Machine | Baseline | N | Tasks | Worker blocks/task | Samples |" in report
    assert "| a100-local | pto_persistent_device_grid_batch | 1024 | 6 | 2 | 1 | 5000 | 5000 | - |" in report
    assert "| a100-local | pto_persistent_device_grid_batch | 1024 | 6 | 4 | 1 | 3000 | 3000 | - |" in report


def test_render_report_highlights_best_worker_grid_rows():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "grid-best-unit",
            "git_commit": "abc123",
            "paper_setup": "microbenchmarks only",
        },
        "results": [
            {
                "machine": "a100-local",
                "baseline": "pto_host_schedule_batch",
                "n": 1048576,
                "task_count": 6,
                "device_wall_ns": 100000,
            },
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_device_grid_batch",
                "n": 1048576,
                "task_count": 6,
                "worker_blocks_per_task": 4,
                "device_wall_ns": 450000,
            },
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_device_grid_batch",
                "n": 1048576,
                "task_count": 6,
                "worker_blocks_per_task": 16,
                "device_wall_ns": 150000,
            },
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)

    assert "## Best Worker Grid Rows" in report
    expected_header = (
        "| Machine | N | Tasks | Best worker blocks/task | Median device ns | Device vs matched host_schedule |"
    )
    assert expected_header in report
    assert "| a100-local | 1048576 | 6 | 16 | 150000 | 1.50x |" in report


def test_render_report_contains_table_and_svg():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "unit",
            "git_commit": "abc123",
            "paper_setup": "microbenchmarks only",
            "source_papers": [
                {"id": "arXiv:2605.03190", "label": "VDCores"},
                {"id": "arXiv:2512.22219v1", "label": "MPK persistent kernel"},
            ],
        },
        "results": [
            {"machine": "a100-local", "baseline": "pto_host_schedule", "n": 1024, "device_wall_ns": 1000},
            {"machine": "a100-local", "baseline": "direct_driver", "n": 1024, "device_wall_ns": 500},
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)
    svg = cuda_benchmark.render_svg(cuda_benchmark.summarize_results(payload))

    assert "CUDA Backend Microbenchmark Report" in report
    assert "- Source papers: `arXiv:2605.03190` VDCores; `arXiv:2512.22219v1` MPK persistent kernel" in report
    expected_header = (
        "| Machine | Baseline | N | Tasks | Worker blocks/task | Samples | Median device ns | "
        "Median host ns | Device vs matched reference |"
    )
    assert expected_header in report
    assert "a100-local" in report
    assert "<svg" in svg
    assert "direct_driver" in svg


def test_render_report_includes_host_schedule_relative_ratios():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "ratio-unit",
            "git_commit": "abc123",
            "paper_setup": "microbenchmarks only",
        },
        "results": [
            {"machine": "a100-local", "baseline": "pto_host_schedule", "n": 1024, "device_wall_ns": 1000},
            {"machine": "a100-local", "baseline": "direct_driver", "n": 1024, "device_wall_ns": 500},
            {
                "machine": "a100-local",
                "baseline": "direct_driver_graph",
                "n": 1024,
                "device_wall_ns": 450,
            },
            {"machine": "a100-local", "baseline": "pto_persistent_dag", "n": 1024, "device_wall_ns": 2500},
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)

    assert "| a100-local | pto_host_schedule | 1024 | 1 | 1 | 1 | 1000 | 1000 | 1.00x |" in report
    assert "| a100-local | direct_driver | 1024 | 1 | 1 | 1 | 500 | 500 | 0.50x |" in report
    assert ("| a100-local | direct_driver_graph | 1024 | 1 | 1 | 1 | 450 | 450 | 0.45x |") in report
    assert "| a100-local | pto_persistent_dag | 1024 | 1 | 1 | 1 | 2500 | 2500 | 2.50x |" in report
    assert "`direct_driver_graph` measures a CUDA Graph replay path" in report
    assert "Non-stream ratio columns are relative to the matched" in report
    assert "`pto_host_schedule` row for the same machine, `N`, and task count" in report
    assert "for the same machine, `N`, and task count" in report


def test_render_ratio_svg_visualizes_matched_reference_ratios():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "ratio-svg-unit",
            "git_commit": "abc123",
            "paper_setup": "microbenchmarks only",
        },
        "results": [
            {"machine": "a100-local", "baseline": "pto_host_schedule", "n": 1024, "device_wall_ns": 1000},
            {"machine": "a100-local", "baseline": "direct_driver", "n": 1024, "device_wall_ns": 500},
            {
                "machine": "a100-local",
                "baseline": "direct_driver_graph",
                "n": 1024,
                "device_wall_ns": 250,
            },
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)
    summary = cuda_benchmark.summarize_results(payload)
    svg = cuda_benchmark.render_ratio_svg(summary)

    assert "![Device ratio chart](cuda-benchmark-ratios.svg)" in report
    assert "Device time ratio vs matched reference" in svg
    assert "direct_driver_graph" in svg
    assert "0.25x" in svg
    assert "reference=1.00x" in svg


def test_write_report_writes_ratio_svg(tmp_path):
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "write-ratio-unit",
            "git_commit": "abc123",
            "paper_setup": "microbenchmarks only",
        },
        "results": [
            {"machine": "a100-local", "baseline": "pto_host_schedule", "n": 1024, "device_wall_ns": 1000},
            {"machine": "a100-local", "baseline": "direct_driver", "n": 1024, "device_wall_ns": 500},
        ],
    }

    cuda_benchmark.write_report(payload, tmp_path)

    assert (tmp_path / "cuda-benchmark-ratios.svg").exists()


def test_write_report_writes_dag_delta_svg(tmp_path):
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "write-dag-delta-unit",
            "git_commit": "abc123",
            "paper_setup": "microbenchmarks only",
        },
        "results": [
            {"machine": "a100-local", "baseline": "pto_persistent_dag", "n": 1024, "device_wall_ns": 1000},
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag_tensor_core",
                "n": 1024,
                "task_count": 4,
                "device_wall_ns": 1250,
            },
        ],
    }

    cuda_benchmark.write_report(payload, tmp_path)

    assert (tmp_path / "cuda-benchmark-dag-deltas.svg").exists()


def test_render_report_includes_tensor_throughput_rows():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "tensor-throughput-unit",
            "git_commit": "abc123",
            "paper_setup": "microbenchmarks only",
            "tensor_tile": {"rows": 16, "cols": 16, "inner": 16},
        },
        "results": [
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag_tensor_core",
                "n": 512,
                "task_count": 4,
                "device_wall_ns": 1024,
                "tensor_tile": {"rows": 16, "cols": 16, "inner": 16, "tile_count": 2},
            },
            {
                "machine": "a100-local",
                "baseline": "cublas_sgemm",
                "n": 512,
                "task_count": 1,
                "device_wall_ns": 2048,
                "tensor_tile": {"rows": 16, "cols": 16, "inner": 16, "tile_count": 2},
            },
            {
                "machine": "a100-local",
                "baseline": "cublas_sgemm_graph",
                "n": 512,
                "task_count": 1,
                "device_wall_ns": 1536,
                "tensor_tile": {"rows": 16, "cols": 16, "inner": 16, "tile_count": 2},
            },
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)
    svg = cuda_benchmark.render_tensor_throughput_svg(payload)

    assert "## Tensor Throughput Rows" in report
    assert "| a100-local | pto_persistent_dag_tensor_core | 512 | 16x16x16 | 1024 | 16.00 |" in report
    assert "| a100-local | cublas_sgemm | 512 | 16x16x16 | 2048 | 8.00 |" in report
    assert "| a100-local | cublas_sgemm_graph | 512 | 16x16x16 | 1536 | 10.67 |" in report
    assert "![Tensor throughput chart](cuda-benchmark-throughput.svg)" in report
    assert "Tensor throughput by baseline" in svg
    assert "16.00 GF/s" in svg
    assert "cublas_sgemm_graph" in svg


def test_write_report_writes_tensor_throughput_svg(tmp_path):
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "write-throughput-unit",
            "git_commit": "abc123",
            "paper_setup": "microbenchmarks only",
            "tensor_tile": {"rows": 16, "cols": 16, "inner": 16},
        },
        "results": [
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag_tensor_core",
                "n": 512,
                "task_count": 4,
                "device_wall_ns": 1024,
                "tensor_tile": {"rows": 16, "cols": 16, "inner": 16, "tile_count": 2},
            }
        ],
    }

    cuda_benchmark.write_report(payload, tmp_path)

    assert (tmp_path / "cuda-benchmark-throughput.svg").exists()


def test_render_report_uses_batch_host_schedule_reference_for_batch_rows():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "batch-ratio-unit",
            "git_commit": "abc123",
            "paper_setup": "microbenchmarks only",
        },
        "results": [
            {
                "machine": "a100-local",
                "baseline": "pto_host_schedule",
                "n": 1024,
                "task_count": 1,
                "device_wall_ns": 1000,
            },
            {
                "machine": "a100-local",
                "baseline": "pto_host_schedule_batch",
                "n": 1024,
                "task_count": 6,
                "device_wall_ns": 6000,
            },
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_queue_batch",
                "n": 1024,
                "task_count": 6,
                "device_wall_ns": 3000,
            },
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)

    assert "| a100-local | pto_host_schedule_batch | 1024 | 6 | 1 | 1 | 6000 | 6000 | 1.00x |" in report
    assert "| a100-local | pto_persistent_queue_batch | 1024 | 6 | 1 | 1 | 3000 | 3000 | 0.50x |" in report
    assert "same-work batch rows use `pto_host_schedule_batch` as their reference" in report
    assert "match descriptor count, not intra-task grid shape" in report


def test_render_report_describes_worker_grid_batch_rows():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "grid-batch-unit",
            "git_commit": "abc123",
            "paper_setup": "microbenchmarks only",
        },
        "results": [
            {
                "machine": "a100-local",
                "baseline": "pto_host_schedule_batch",
                "n": 4096,
                "task_count": 6,
                "device_wall_ns": 6000,
            },
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_device_grid_batch",
                "n": 4096,
                "task_count": 6,
                "device_wall_ns": 3000,
            },
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)
    svg = cuda_benchmark.render_svg(cuda_benchmark.summarize_results(payload))

    assert "| a100-local | pto_persistent_device_grid_batch | 4096 | 6 | 1 | 1 | 3000 | 3000 | 0.50x |" in report
    assert "`pto_persistent_device_grid_batch` assigns multiple worker blocks per task descriptor" in report
    assert "pto_persistent_device_grid_batch" in svg


def test_render_report_describes_dag_chain_rows():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "dag-chain-unit",
            "git_commit": "abc123",
            "paper_setup": "microbenchmarks only",
        },
        "results": [
            {"machine": "a100-local", "baseline": "pto_host_schedule", "n": 1024, "device_wall_ns": 1000},
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag_chain",
                "n": 1024,
                "task_count": 5,
                "dag_shape": "chain",
                "device_wall_ns": 3000,
            },
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)
    svg = cuda_benchmark.render_svg(cuda_benchmark.summarize_results(payload))

    assert "| a100-local | pto_persistent_dag_chain | 1024 | 5 | 1 | 1 | 3000 | 3000 | - |" in report
    assert "`pto_persistent_dag_chain` extends the DAG smoke to five tasks" in report
    assert "pto_persistent_dag_chain" in svg


def test_render_report_describes_dag_reuse_rows():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "dag-reuse-unit",
            "git_commit": "abc123",
            "paper_setup": "microbenchmarks only",
        },
        "results": [
            {"machine": "a100-local", "baseline": "pto_host_schedule", "n": 1024, "device_wall_ns": 1000},
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag_reuse",
                "n": 1024,
                "task_count": 6,
                "dag_shape": "scratch_reuse",
                "device_wall_ns": 4000,
            },
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)
    svg = cuda_benchmark.render_svg(cuda_benchmark.summarize_results(payload))

    assert "| a100-local | pto_persistent_dag_reuse | 1024 | 6 | 1 | 1 | 4000 | 4000 | - |" in report
    assert "`pto_persistent_dag_reuse` uses a six-task DAG with scratch-buffer reuse" in report
    assert "pto_persistent_dag_reuse" in svg


def test_render_report_describes_dag_tensor_rows():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "dag-tensor-unit",
            "git_commit": "abc123",
            "paper_setup": "microbenchmarks only",
        },
        "results": [
            {"machine": "a100-local", "baseline": "pto_host_schedule", "n": 1024, "device_wall_ns": 1000},
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag_tensor",
                "n": 1024,
                "task_count": 4,
                "dag_shape": "tensor_tile",
                "device_wall_ns": 5000,
            },
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)
    svg = cuda_benchmark.render_svg(cuda_benchmark.summarize_results(payload))

    assert "| a100-local | pto_persistent_dag_tensor | 1024 | 4 | 1 | 1 | 5000 | 5000 | - |" in report
    assert "`pto_persistent_dag_tensor` uses the default 16x16x16 tiled GEMM" in report
    assert "pto_persistent_dag_tensor" in svg


def test_render_report_describes_dag_graph_tensor_rows():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "dag-graph-tensor-unit",
            "git_commit": "abc123",
            "paper_setup": "microbenchmarks only",
            "tensor_tile": {"rows": 16, "cols": 16, "inner": 16},
        },
        "results": [
            {"machine": "a100-local", "baseline": "pto_host_schedule", "n": 512, "device_wall_ns": 1000},
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag_graph_tensor",
                "n": 512,
                "task_count": 4,
                "dag_shape": "graph_tensor_tile",
                "graph_descriptor": {"tasks": 4, "dependents": [1, 2, 3, 3], "fanin": [0, 1, 1, 2]},
                "tensor_tile": {
                    "rows": 16,
                    "cols": 16,
                    "inner": 16,
                    "tile_count": 2,
                },
                "device_wall_ns": 4700,
            },
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)
    svg = cuda_benchmark.render_svg(cuda_benchmark.summarize_results(payload))

    expected_row = "| a100-local | pto_persistent_dag_graph_tensor | 512 | 4 | 1 | 1 | 4700 | 4700 | - |"
    assert expected_row in report
    assert "`pto_persistent_dag_graph_tensor` uses an explicit graph descriptor" in report
    assert "16x16x16" in report
    assert "pto_persistent_dag_graph_tensor" in svg


def test_render_report_describes_dag_tensor_core_rows():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "dag-tensor-core-unit",
            "git_commit": "abc123",
            "paper_setup": "microbenchmarks only",
            "tensor_tile": {"rows": 16, "cols": 16, "inner": 16},
        },
        "results": [
            {"machine": "a100-local", "baseline": "pto_host_schedule", "n": 256, "device_wall_ns": 1000},
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag_tensor_core",
                "n": 256,
                "task_count": 4,
                "dag_shape": "tensor_core_tile",
                "device_wall_ns": 3200,
                "tensor_core": {
                    "api": "wmma",
                    "mma_shape": "m16n16k8",
                    "input": "tf32",
                    "accumulator": "f32",
                },
            },
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)
    svg = cuda_benchmark.render_svg(cuda_benchmark.summarize_results(payload))

    assert "| a100-local | pto_persistent_dag_tensor_core | 256 | 4 | 1 | 1 | 3200 | 3200 | - |" in report
    assert "`pto_persistent_dag_tensor_core` uses a WMMA tensor-core task" in report
    assert "m16n16k8" in report
    assert "pto_persistent_dag_tensor_core" in svg


def test_render_report_describes_cublas_sgemm_rows():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "cublas-unit",
            "git_commit": "abc123",
            "paper_setup": "microbenchmarks only",
            "tensor_tile": {"rows": 16, "cols": 16, "inner": 16},
        },
        "results": [
            {"machine": "a100-local", "baseline": "pto_host_schedule", "n": 256, "device_wall_ns": 1000},
            {
                "machine": "a100-local",
                "baseline": "cublas_sgemm",
                "n": 256,
                "task_count": 1,
                "batch_count": 1,
                "library": "cublas",
                "device_wall_ns": 2400,
            },
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)
    svg = cuda_benchmark.render_svg(cuda_benchmark.summarize_results(payload))

    assert "| a100-local | cublas_sgemm | 256 | 1 | 1 | 1 | 2400 | 2400 | 2.40x |" in report
    assert "`cublas_sgemm` measures cuBLAS SGEMM" in report
    assert "16x16x16" in report
    assert "cublas_sgemm" in svg


def test_render_report_describes_cublas_sgemm_graph_rows():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "cublas-graph-unit",
            "git_commit": "abc123",
            "paper_setup": "microbenchmarks only",
            "tensor_tile": {"rows": 16, "cols": 16, "inner": 16},
        },
        "results": [
            {"machine": "a100-local", "baseline": "pto_host_schedule", "n": 256, "device_wall_ns": 1000},
            {
                "machine": "a100-local",
                "baseline": "cublas_sgemm_graph",
                "n": 256,
                "task_count": 1,
                "batch_count": 1,
                "library": "cublas",
                "launch_mode": "cuda_graph",
                "device_wall_ns": 1800,
            },
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)
    svg = cuda_benchmark.render_svg(cuda_benchmark.summarize_results(payload))

    assert "| a100-local | cublas_sgemm_graph | 256 | 1 | 1 | 1 | 1800 | 1800 | 1.80x |" in report
    assert "`cublas_sgemm_graph` measures cuBLAS SGEMM captured into a CUDA Graph" in report
    assert "16x16x16" in report
    assert "cublas_sgemm_graph" in svg


def test_render_report_describes_dag_scalar_axpy_rows():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "dag-scalar-unit",
            "git_commit": "abc123",
            "paper_setup": "microbenchmarks only",
        },
        "results": [
            {"machine": "a100-local", "baseline": "pto_host_schedule", "n": 1024, "device_wall_ns": 1000},
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag_scalar_axpy",
                "n": 1024,
                "task_count": 3,
                "dag_shape": "scalar_axpy",
                "device_wall_ns": 2500,
            },
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)
    svg = cuda_benchmark.render_svg(cuda_benchmark.summarize_results(payload))

    expected_row = "| a100-local | pto_persistent_dag_scalar_axpy | 1024 | 3 | 1 | 1 | 2500 | 2500 | - |"
    assert expected_row in report
    assert "`pto_persistent_dag_scalar_axpy` uses the scalar0 task descriptor" in report
    assert "pto_persistent_dag_scalar_axpy" in svg


def test_render_report_describes_dag_scalar_affine_rows():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "dag-scalar-affine-unit",
            "git_commit": "abc123",
            "paper_setup": "microbenchmarks only",
        },
        "results": [
            {"machine": "a100-local", "baseline": "pto_host_schedule", "n": 1024, "device_wall_ns": 1000},
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag_scalar_affine",
                "n": 1024,
                "task_count": 3,
                "dag_shape": "scalar_affine",
                "device_wall_ns": 2600,
            },
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)
    svg = cuda_benchmark.render_svg(cuda_benchmark.summarize_results(payload))

    expected_row = "| a100-local | pto_persistent_dag_scalar_affine | 1024 | 3 | 1 | 1 | 2600 | 2600 | - |"
    assert expected_row in report
    assert "`pto_persistent_dag_scalar_affine` uses scalar0 and scalar1 task descriptor fields" in report
    assert "pto_persistent_dag_scalar_affine" in svg


def test_render_report_describes_dag_scalar_scale_rows():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "dag-scalar-scale-unit",
            "git_commit": "abc123",
            "paper_setup": "microbenchmarks only",
        },
        "results": [
            {"machine": "a100-local", "baseline": "pto_host_schedule", "n": 1024, "device_wall_ns": 1000},
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag_scalar_scale",
                "n": 1024,
                "task_count": 3,
                "dag_shape": "scalar_scale",
                "device_wall_ns": 2550,
            },
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)
    svg = cuda_benchmark.render_svg(cuda_benchmark.summarize_results(payload))

    expected_row = "| a100-local | pto_persistent_dag_scalar_scale | 1024 | 3 | 1 | 1 | 2550 | 2550 | - |"
    assert expected_row in report
    assert "`pto_persistent_dag_scalar_scale` uses scalar0 with one tensor input" in report
    assert "pto_persistent_dag_scalar_scale" in svg


def test_render_report_describes_dag_triad_rows():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "dag-triad-unit",
            "git_commit": "abc123",
            "paper_setup": "microbenchmarks only",
        },
        "results": [
            {"machine": "a100-local", "baseline": "pto_host_schedule", "n": 1024, "device_wall_ns": 1000},
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag_triad",
                "n": 1024,
                "task_count": 3,
                "dag_shape": "triad",
                "tensor_args": {"c": "tmp0"},
                "device_wall_ns": 2700,
            },
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)
    svg = cuda_benchmark.render_svg(cuda_benchmark.summarize_results(payload))

    expected_row = "| a100-local | pto_persistent_dag_triad | 1024 | 3 | 1 | 1 | 2700 | 2700 | - |"
    assert expected_row in report
    assert "`pto_persistent_dag_triad` uses the third tensor task descriptor field" in report
    assert "pto_persistent_dag_triad" in svg


def test_render_report_describes_dag_quad_rows():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "dag-quad-unit",
            "git_commit": "abc123",
            "paper_setup": "microbenchmarks only",
        },
        "results": [
            {"machine": "a100-local", "baseline": "pto_host_schedule", "n": 1024, "device_wall_ns": 1000},
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag_quad",
                "n": 1024,
                "task_count": 3,
                "dag_shape": "quad",
                "tensor_args": {"c": "tmp0", "d": "tmp3"},
                "device_wall_ns": 2800,
            },
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)
    svg = cuda_benchmark.render_svg(cuda_benchmark.summarize_results(payload))

    expected_row = "| a100-local | pto_persistent_dag_quad | 1024 | 3 | 1 | 1 | 2800 | 2800 | - |"
    assert expected_row in report
    assert "`pto_persistent_dag_quad` uses third and fourth tensor task descriptor fields" in report
    assert "pto_persistent_dag_quad" in svg


def test_render_report_describes_dag_unary_square_rows():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "dag-unary-square-unit",
            "git_commit": "abc123",
            "paper_setup": "microbenchmarks only",
        },
        "results": [
            {"machine": "a100-local", "baseline": "pto_host_schedule", "n": 1024, "device_wall_ns": 1000},
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag_unary_square",
                "n": 1024,
                "task_count": 3,
                "dag_shape": "unary_square",
                "device_wall_ns": 2400,
            },
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)
    svg = cuda_benchmark.render_svg(cuda_benchmark.summarize_results(payload))

    expected_row = "| a100-local | pto_persistent_dag_unary_square | 1024 | 3 | 1 | 1 | 2400 | 2400 | - |"
    assert expected_row in report
    assert "`pto_persistent_dag_unary_square` uses a one-input square task body" in report
    assert "pto_persistent_dag_unary_square" in svg


def test_render_report_describes_dag_generic_args_rows():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "dag-generic-args-unit",
            "git_commit": "abc123",
            "paper_setup": "microbenchmarks only",
        },
        "results": [
            {"machine": "a100-local", "baseline": "pto_host_schedule", "n": 1024, "device_wall_ns": 1000},
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag_generic_args",
                "n": 1024,
                "task_count": 3,
                "dag_shape": "generic_args",
                "generic_args": {
                    "tensor_args": {"0": "tmp0", "1": "tmp3"},
                    "scalar_args": [1.5, 0.25],
                },
                "device_wall_ns": 2200,
            },
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)
    svg = cuda_benchmark.render_svg(cuda_benchmark.summarize_results(payload))

    expected_row = "| a100-local | pto_persistent_dag_generic_args | 1024 | 3 | 1 | 1 | 2200 | 2200 | - |"
    assert expected_row in report
    assert "`pto_persistent_dag_generic_args` uses generic tensor/scalar task descriptor slots" in report
    assert "pto_persistent_dag_generic_args" in svg


def test_render_report_describes_dag_graph_rows():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "dag-graph-unit",
            "git_commit": "abc123",
            "paper_setup": "microbenchmarks only",
        },
        "results": [
            {"machine": "a100-local", "baseline": "pto_host_schedule", "n": 1024, "device_wall_ns": 1000},
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag_graph",
                "n": 1024,
                "task_count": 3,
                "dag_shape": "graph_descriptor",
                "graph_descriptor": {
                    "tasks": 3,
                    "dependents": [2, 2],
                    "fanin": [0, 0, 2],
                },
                "device_wall_ns": 2300,
            },
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)
    svg = cuda_benchmark.render_svg(cuda_benchmark.summarize_results(payload))

    expected_row = "| a100-local | pto_persistent_dag_graph | 1024 | 3 | 1 | 1 | 2300 | 2300 | - |"
    assert expected_row in report
    assert "`pto_persistent_dag_graph` uses an explicit runtime graph descriptor" in report
    assert "pto_persistent_dag_graph" in svg


def test_render_report_describes_tensor_tile_metadata():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "tensor-shape-unit",
            "tensor_tile": {"rows": 8, "cols": 4, "inner": 12},
        },
        "results": [
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag_tensor",
                "n": 64,
                "task_count": 4,
                "device_wall_ns": 4200,
                "tensor_tile": {
                    "rows": 8,
                    "cols": 4,
                    "inner": 12,
                    "tile_count": 2,
                },
            }
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)

    assert "- Tensor tile descriptor: `8x4x12`." in report
    assert "`pto_persistent_dag_tensor` uses the configured 8x4x12 tiled GEMM" in report


def test_render_report_highlights_dag_shape_rows():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "dag-shapes-unit",
            "git_commit": "abc123",
            "paper_setup": "microbenchmarks only",
        },
        "results": [
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag",
                "n": 4096,
                "task_count": 3,
                "device_wall_ns": 1000,
            },
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag_chain",
                "n": 4096,
                "task_count": 5,
                "device_wall_ns": 1800,
            },
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag_tensor",
                "n": 4096,
                "task_count": 4,
                "device_wall_ns": 4200,
            },
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag_scalar_axpy",
                "n": 4096,
                "task_count": 3,
                "device_wall_ns": 1300,
            },
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag_scalar_affine",
                "n": 4096,
                "task_count": 3,
                "device_wall_ns": 1400,
            },
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag_triad",
                "n": 4096,
                "task_count": 3,
                "device_wall_ns": 1500,
            },
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag_quad",
                "n": 4096,
                "task_count": 3,
                "device_wall_ns": 1600,
            },
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag_generic_args",
                "n": 4096,
                "task_count": 3,
                "device_wall_ns": 1250,
            },
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag_graph",
                "n": 4096,
                "task_count": 3,
                "device_wall_ns": 900,
            },
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag_unary_square",
                "n": 4096,
                "task_count": 3,
                "device_wall_ns": 1200,
            },
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)

    assert "## DAG Shape Rows" in report
    assert ("| Machine | N | Baseline | Tasks | Median device ns | Device vs pto_persistent_dag |") in report
    assert ("| a100-local | 4096 | pto_persistent_dag_chain | 5 | 1800 | 1.80x |") in report
    assert ("| a100-local | 4096 | pto_persistent_dag_scalar_axpy | 3 | 1300 | 1.30x |") in report
    assert ("| a100-local | 4096 | pto_persistent_dag_scalar_affine | 3 | 1400 | 1.40x |") in report
    assert ("| a100-local | 4096 | pto_persistent_dag_triad | 3 | 1500 | 1.50x |") in report
    assert ("| a100-local | 4096 | pto_persistent_dag_quad | 3 | 1600 | 1.60x |") in report
    assert ("| a100-local | 4096 | pto_persistent_dag_generic_args | 3 | 1250 | 1.25x |") in report
    assert ("| a100-local | 4096 | pto_persistent_dag_graph | 3 | 900 | 0.90x |") in report
    assert ("| a100-local | 4096 | pto_persistent_dag_unary_square | 3 | 1200 | 1.20x |") in report
    assert ("| a100-local | 4096 | pto_persistent_dag_tensor | 4 | 4200 | 4.20x |") in report
    assert "## DAG Increment Rows" in report
    assert (
        "| Machine | N | Baseline | Tasks | Base DAG ns | Median device ns | Increment ns | Increment vs base |"
    ) in report
    assert ("| a100-local | 4096 | pto_persistent_dag_tensor | 4 | 1000 | 4200 | 3200 | 3.20x |") in report
    assert ("| a100-local | 4096 | pto_persistent_dag_graph | 3 | 1000 | 900 | -100 | -0.10x |") in report
    assert "![DAG increment chart](cuda-benchmark-dag-deltas.svg)" in report


def test_render_dag_delta_svg_visualizes_increment_over_base_dag():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {"label": "dag-delta-svg-unit"},
        "results": [
            {"machine": "a100-local", "baseline": "pto_persistent_dag", "n": 1024, "device_wall_ns": 1000},
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag_tensor_core",
                "n": 1024,
                "task_count": 4,
                "device_wall_ns": 1500,
            },
        ],
    }
    summary = cuda_benchmark.summarize_results(payload)

    svg = cuda_benchmark.render_dag_delta_svg(summary)

    assert "DAG device-time increment over pto_persistent_dag" in svg
    assert "pto_persistent_dag_tensor_core" in svg
    assert "+500 ns" in svg


def test_render_dag_delta_svg_keeps_negative_deltas_in_view():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {"label": "dag-negative-delta-svg-unit"},
        "results": [
            {"machine": "a100-local", "baseline": "pto_persistent_dag", "n": 1024, "device_wall_ns": 1000},
            {
                "machine": "a100-local",
                "baseline": "pto_persistent_dag_graph",
                "n": 1024,
                "task_count": 3,
                "device_wall_ns": 500,
            },
        ],
    }
    summary = cuda_benchmark.summarize_results(payload)

    svg = cuda_benchmark.render_dag_delta_svg(summary)

    assert "-500 ns" in svg
    assert 'x="-' not in svg


def test_render_report_summarizes_ptx_sources_by_machine_and_baseline():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "ptx-source-unit",
            "git_commit": "abc123",
            "paper_setup": "microbenchmarks only",
        },
        "results": [
            {
                "machine": "a100-local",
                "baseline": "pto_host_schedule",
                "n": 1024,
                "device_wall_ns": 1000,
                "ptx_source": "nvcc-compute_80",
            },
            {
                "machine": "h200-remote",
                "baseline": "pto_host_schedule",
                "n": 1024,
                "device_wall_ns": 800,
                "ptx_source": "embedded-sm80-ptx",
            },
            {
                "machine": "h200-remote",
                "baseline": "pto_persistent_queue_batch",
                "n": 1024,
                "task_count": 6,
                "device_wall_ns": 700,
                "ptx_source": "embedded-sm80-persistent-queue-ptx",
            },
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)

    assert "## PTX Sources" in report
    assert "| a100-local | pto_host_schedule | `nvcc-compute_80` |" in report
    assert "| h200-remote | pto_host_schedule | `embedded-sm80-ptx` |" in report
    assert "| h200-remote | pto_persistent_queue_batch | `embedded-sm80-persistent-queue-ptx` |" in report
    assert "Embedded `sm_80` PTX means the local driver JIT compiled fallback code" in report


def test_merge_payloads_preserves_results_and_records_sources():
    cuda_benchmark = _load_benchmark_module()
    payloads = [
        {
            "metadata": {
                "label": "a100",
                "git_commit": "abc123",
                "tensor_tile": {"rows": 8, "cols": 4, "inner": 12},
                "stream_pool_size": 6,
            },
            "results": [{"machine": "a100-local", "baseline": "direct_driver", "n": 1024, "device_wall_ns": 500}],
        },
        {
            "metadata": {
                "label": "h200",
                "git_commit": "abc123",
                "tensor_tile": {"rows": 8, "cols": 4, "inner": 12},
                "stream_pool_size": 6,
            },
            "results": [{"machine": "h200-remote", "baseline": "direct_driver", "n": 1024, "device_wall_ns": 300}],
        },
    ]

    merged = cuda_benchmark.merge_payloads(payloads, label="combined")

    assert merged["metadata"]["label"] == "combined"
    assert merged["metadata"]["source_labels"] == ["a100", "h200"]
    assert merged["metadata"]["git_commits"] == ["abc123"]
    assert merged["metadata"]["tensor_tile"] == {"rows": 8, "cols": 4, "inner": 12}
    assert merged["metadata"]["stream_pool_size"] == 6
    assert [paper["id"] for paper in merged["metadata"]["source_papers"]] == [
        "arXiv:2605.03190",
        "arXiv:2512.22219v1",
    ]
    assert len(merged["results"]) == 2


def test_merge_payloads_records_command_examples():
    cuda_benchmark = _load_benchmark_module()
    payloads = [
        {
            "metadata": {"label": "a100", "git_commit": "abc123"},
            "results": [{"machine": "a100-local", "baseline": "direct_driver", "n": 1024, "device_wall_ns": 500}],
        },
        {
            "metadata": {"label": "h200", "git_commit": "abc123"},
            "results": [{"machine": "h200-remote", "baseline": "direct_driver", "n": 1024, "device_wall_ns": 300}],
        },
    ]

    merged = cuda_benchmark.merge_payloads(
        payloads,
        label="combined",
        command_examples={
            "local_sample": "env PYTHONPATH=$PWD:$PWD/python .venv/bin/python cuda_benchmark.py",
            "remote_sample": "ssh h200-box 'cd /remote/pto-cu && cuda_benchmark.py'",
        },
    )
    report = cuda_benchmark.render_markdown_report(merged)

    assert merged["metadata"]["command_examples"] == {
        "local_sample": "env PYTHONPATH=$PWD:$PWD/python .venv/bin/python cuda_benchmark.py",
        "remote_sample": "ssh h200-box 'cd /remote/pto-cu && cuda_benchmark.py'",
    }
    assert "- Local sample command: `env PYTHONPATH=$PWD:$PWD/python .venv/bin/python cuda_benchmark.py`" in report
    assert "- Remote sample command: `ssh h200-box 'cd /remote/pto-cu && cuda_benchmark.py'`" in report


def test_run_benchmark_uses_in_process_samples(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = []

    def fake_compile_ptx(work_dir, arch):
        return b"ptx", f"fake-{arch}"

    def fake_run_single_sample(baseline, device, n, block_dim, arch):
        seen.append((baseline, device, n, block_dim, arch))
        return {
            "baseline": baseline,
            "n": n,
            "block_dim": block_dim,
            "host_wall_ns": 20,
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "_compile_ptx", fake_compile_ptx)
    monkeypatch.setattr(cuda_benchmark, "run_single_sample", fake_run_single_sample)

    payload = cuda_benchmark.run_benchmark(
        device=3,
        sizes=[1024],
        repeats=1,
        block_dim=128,
        arch="compute_80",
        label="unit",
    )

    assert seen == [
        ("pto_host_schedule", 3, 1024, 128, "compute_80"),
        ("pto_host_schedule_compiler", 3, 1024, 128, "compute_80"),
        ("pto_host_schedule_unary_square", 3, 1024, 128, "compute_80"),
        ("pto_host_schedule_quad", 3, 1024, 128, "compute_80"),
        ("pto_host_schedule_generic_args", 3, 1024, 128, "compute_80"),
        ("direct_driver", 3, 1024, 128, "compute_80"),
        ("direct_driver_graph", 3, 1024, 128, "compute_80"),
    ]
    assert len(payload["results"]) == 7


def test_run_benchmark_records_source_paper_metadata(monkeypatch):
    cuda_benchmark = _load_benchmark_module()

    def fake_compile_ptx(work_dir, arch):
        return b"ptx", f"fake-{arch}"

    def fake_run_single_sample(baseline, device, n, block_dim, arch):
        return {
            "baseline": baseline,
            "n": n,
            "block_dim": block_dim,
            "host_wall_ns": 20,
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "_compile_ptx", fake_compile_ptx)
    monkeypatch.setattr(cuda_benchmark, "run_single_sample", fake_run_single_sample)

    payload = cuda_benchmark.run_benchmark(
        device=3,
        sizes=[1024],
        repeats=1,
        block_dim=128,
        arch="compute_80",
        label="unit",
    )

    assert payload["metadata"]["source_papers"] == [
        {
            "id": "arXiv:2605.03190",
            "label": "VDCores",
            "path": "tmp/sources/arxiv-2605.03190-vdcores.txt",
        },
        {
            "id": "arXiv:2512.22219v1",
            "label": "MPK persistent kernel",
            "path": "tmp/sources/arxiv-2512.22219v1-mirage-persistent-kernel.txt",
        },
    ]


def test_run_single_sample_dispatches_compiler_host_schedule(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    def fake_run_pto_compiler_sample(device, n, block_dim, arch):
        seen["args"] = (device, n, block_dim, arch)
        return {
            "baseline": "pto_host_schedule_compiler",
            "n": n,
            "block_dim": block_dim,
            "host_wall_ns": 20,
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "run_pto_compiler_sample", fake_run_pto_compiler_sample)

    result = cuda_benchmark.run_single_sample(
        baseline="pto_host_schedule_compiler",
        device=3,
        n=1024,
        block_dim=128,
        arch="compute_80",
    )

    assert seen["args"] == (3, 1024, 128, "compute_80")
    assert result["baseline"] == "pto_host_schedule_compiler"


def test_run_single_sample_dispatches_unary_square_host_schedule(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    def fake_run_pto_unary_square_sample(device, n, block_dim, arch):
        seen["args"] = (device, n, block_dim, arch)
        return {
            "baseline": "pto_host_schedule_unary_square",
            "n": n,
            "block_dim": block_dim,
            "host_wall_ns": 20,
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "run_pto_unary_square_sample", fake_run_pto_unary_square_sample)

    result = cuda_benchmark.run_single_sample(
        baseline="pto_host_schedule_unary_square",
        device=3,
        n=1024,
        block_dim=128,
        arch="compute_80",
    )

    assert seen["args"] == (3, 1024, 128, "compute_80")
    assert result["baseline"] == "pto_host_schedule_unary_square"


def test_run_single_sample_dispatches_quad_host_schedule(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    def fake_run_pto_quad_sample(device, n, block_dim, arch):
        seen["args"] = (device, n, block_dim, arch)
        return {
            "baseline": "pto_host_schedule_quad",
            "n": n,
            "block_dim": block_dim,
            "host_wall_ns": 20,
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "run_pto_quad_sample", fake_run_pto_quad_sample)

    result = cuda_benchmark.run_single_sample(
        baseline="pto_host_schedule_quad",
        device=3,
        n=1024,
        block_dim=128,
        arch="compute_80",
    )

    assert seen["args"] == (3, 1024, 128, "compute_80")
    assert result["baseline"] == "pto_host_schedule_quad"


def test_unary_square_benchmark_expected_output_uses_float32_rounding():
    cuda_benchmark = _load_benchmark_module()

    expected = cuda_benchmark._expected_unary_square_output(65536)

    assert expected[-1] == ctypes.c_float(float(65535) * float(65535)).value
    assert expected[-1] != float(65535 * 65535)


def test_compile_compiler_host_schedule_artifact_uses_discovered_nvcc(tmp_path, monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    class FakeKernelCompiler:
        def __init__(self, platform):
            seen["platform"] = platform

        def compile_cuda_host_schedule(self, source_path, **kwargs):
            seen["source_path"] = source_path
            seen.update(kwargs)
            return "artifact"

    monkeypatch.setattr(cuda_benchmark, "_find_nvcc", lambda: "/usr/local/cuda/bin/nvcc")
    monkeypatch.setattr(cuda_benchmark, "KernelCompiler", FakeKernelCompiler)

    artifact = cuda_benchmark._compile_compiler_host_schedule_artifact(tmp_path, "compute_90")

    assert artifact == "artifact"
    assert seen["platform"] == "cuda"
    assert seen["arch"] == "compute_90"
    assert seen["nvcc"] == "/usr/local/cuda/bin/nvcc"
    assert seen["task_name"] == "vector_add"
    assert Path(seen["source_path"]).name == "vector_add.pto.cu"


def test_compile_unary_square_host_schedule_artifact_uses_unary_abi(tmp_path, monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    class FakeKernelCompiler:
        def __init__(self, platform):
            seen["platform"] = platform

        def compile_cuda_host_schedule(self, source_path, **kwargs):
            seen["source_path"] = source_path
            seen.update(kwargs)
            return "artifact"

    monkeypatch.setattr(cuda_benchmark, "_find_nvcc", lambda: "/usr/local/cuda/bin/nvcc")
    monkeypatch.setattr(cuda_benchmark, "KernelCompiler", FakeKernelCompiler)

    artifact = cuda_benchmark._compile_unary_square_host_schedule_artifact(tmp_path, "compute_90")

    assert artifact == "artifact"
    assert seen["platform"] == "cuda"
    assert seen["arch"] == "compute_90"
    assert seen["nvcc"] == "/usr/local/cuda/bin/nvcc"
    assert seen["task_name"] == "vector_square"
    assert seen["host_parameters"] == (
        "const float *a",
        "float *out",
        "unsigned long long n",
    )
    assert seen["host_context_initializer"] == "a, out, n"
    assert "ctx->a[i] * ctx->a[i]" in Path(seen["source_path"]).read_text()


def test_run_benchmark_can_include_persistent_device_modes(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = []

    def fake_compile_ptx(work_dir, arch):
        return b"ptx", f"fake-{arch}"

    def fake_run_single_sample(baseline, device, n, block_dim, arch):
        seen.append(baseline)
        return {
            "baseline": baseline,
            "n": n,
            "block_dim": block_dim,
            "host_wall_ns": 20,
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "_compile_ptx", fake_compile_ptx)
    monkeypatch.setattr(cuda_benchmark, "run_single_sample", fake_run_single_sample)

    payload = cuda_benchmark.run_benchmark(
        device=3,
        sizes=[1024],
        repeats=1,
        block_dim=128,
        arch="compute_80",
        label="unit",
        include_persistent=True,
    )

    assert seen == [
        "pto_host_schedule",
        "pto_host_schedule_compiler",
        "pto_host_schedule_unary_square",
        "pto_host_schedule_quad",
        "pto_host_schedule_generic_args",
        "direct_driver",
        "direct_driver_graph",
        "pto_persistent_device",
        "pto_persistent_queue",
        "pto_persistent_dag",
        "pto_persistent_dag_chain",
        "pto_persistent_dag_reuse",
        "pto_persistent_dag_scalar_axpy",
        "pto_persistent_dag_scalar_scale",
        "pto_persistent_dag_scalar_affine",
        "pto_persistent_dag_triad",
        "pto_persistent_dag_quad",
        "pto_persistent_dag_generic_args",
        "pto_persistent_dag_graph",
        "pto_persistent_dag_graph_generic_args4",
        "pto_persistent_dag_graph_chain",
        "pto_persistent_dag_graph_scratch_reuse",
        "pto_persistent_dag_graph_diamond",
        "pto_persistent_dag_graph_tagged",
        "pto_persistent_dag_graph_tagged_inout",
        "pto_persistent_dag_graph_triad",
        "pto_persistent_dag_graph_quad",
        "pto_persistent_dag_graph_unary_square",
        "pto_persistent_dag_unary_square",
        "pto_persistent_dag_tensor",
        "pto_persistent_dag_graph_tensor",
        "pto_persistent_dag_tensor_core",
        "pto_persistent_dag_graph_tensor_core",
        "cublas_sgemm",
        "cublas_sgemm_graph",
    ]
    assert len(payload["results"]) == 35


def test_run_single_sample_dispatches_cublas_sgemm(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    def fake_run_cublas_sgemm_sample(device, n, tensor_tile):
        seen.update({"device": device, "n": n, "tensor_tile": tensor_tile})
        return {
            "baseline": "cublas_sgemm",
            "n": n,
            "task_count": 1,
            "device_wall_ns": 10,
            "status": "pass",
        }

    tensor_tile = {"rows": 16, "cols": 16, "inner": 16}
    monkeypatch.setattr(cuda_benchmark, "run_cublas_sgemm_sample", fake_run_cublas_sgemm_sample)

    result = cuda_benchmark.run_single_sample(
        baseline="cublas_sgemm",
        device=3,
        n=256,
        block_dim=128,
        arch="compute_80",
        tensor_tile=tensor_tile,
    )

    assert seen == {"device": 3, "n": 256, "tensor_tile": tensor_tile}
    assert result["baseline"] == "cublas_sgemm"


def test_run_single_sample_dispatches_cublas_sgemm_graph(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    def fake_run_cublas_sgemm_graph_sample(device, n, tensor_tile):
        seen.update({"device": device, "n": n, "tensor_tile": tensor_tile})
        return {
            "baseline": "cublas_sgemm_graph",
            "n": n,
            "task_count": 1,
            "device_wall_ns": 10,
            "status": "pass",
        }

    tensor_tile = {"rows": 16, "cols": 16, "inner": 16}
    monkeypatch.setattr(cuda_benchmark, "run_cublas_sgemm_graph_sample", fake_run_cublas_sgemm_graph_sample)

    result = cuda_benchmark.run_single_sample(
        baseline="cublas_sgemm_graph",
        device=3,
        n=256,
        block_dim=128,
        arch="compute_80",
        tensor_tile=tensor_tile,
    )

    assert seen == {"device": 3, "n": 256, "tensor_tile": tensor_tile}
    assert result["baseline"] == "cublas_sgemm_graph"


def test_run_single_sample_dispatches_scalar_axpy_dag(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    def fake_run_persistent_sample(
        device,
        n,
        arch,
        mode="direct",
        task_count=None,
        baseline=None,
        worker_blocks_per_task=1,
        dag_shape="fork_join",
        tensor_tile=None,
    ):
        seen.update(
            {
                "device": device,
                "n": n,
                "arch": arch,
                "mode": mode,
                "task_count": task_count,
                "baseline": baseline,
                "worker_blocks_per_task": worker_blocks_per_task,
                "dag_shape": dag_shape,
                "tensor_tile": tensor_tile,
            }
        )
        return {
            "baseline": baseline,
            "n": n,
            "task_count": task_count or 3,
            "dag_shape": dag_shape,
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "run_persistent_sample", fake_run_persistent_sample)

    result = cuda_benchmark.run_single_sample(
        baseline="pto_persistent_dag_scalar_axpy",
        device=3,
        n=1024,
        block_dim=128,
        arch="compute_80",
    )

    assert seen == {
        "device": 3,
        "n": 1024,
        "arch": "compute_80",
        "mode": "dag",
        "task_count": None,
        "baseline": "pto_persistent_dag_scalar_axpy",
        "worker_blocks_per_task": 1,
        "dag_shape": "scalar_axpy",
        "tensor_tile": None,
    }
    assert result["baseline"] == "pto_persistent_dag_scalar_axpy"


def test_run_single_sample_dispatches_scalar_affine_dag(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    def fake_run_persistent_sample(
        device,
        n,
        arch,
        mode="direct",
        task_count=None,
        baseline=None,
        worker_blocks_per_task=1,
        dag_shape="fork_join",
        tensor_tile=None,
    ):
        seen.update(
            {
                "device": device,
                "n": n,
                "arch": arch,
                "mode": mode,
                "task_count": task_count,
                "baseline": baseline,
                "worker_blocks_per_task": worker_blocks_per_task,
                "dag_shape": dag_shape,
                "tensor_tile": tensor_tile,
            }
        )
        return {
            "baseline": baseline,
            "n": n,
            "task_count": task_count or 3,
            "dag_shape": dag_shape,
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "run_persistent_sample", fake_run_persistent_sample)

    result = cuda_benchmark.run_single_sample(
        baseline="pto_persistent_dag_scalar_affine",
        device=3,
        n=1024,
        block_dim=128,
        arch="compute_80",
    )

    assert seen == {
        "device": 3,
        "n": 1024,
        "arch": "compute_80",
        "mode": "dag",
        "task_count": None,
        "baseline": "pto_persistent_dag_scalar_affine",
        "worker_blocks_per_task": 1,
        "dag_shape": "scalar_affine",
        "tensor_tile": None,
    }
    assert result["baseline"] == "pto_persistent_dag_scalar_affine"


def test_run_single_sample_dispatches_scalar_scale_dag(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    def fake_run_persistent_sample(
        device,
        n,
        arch,
        mode="direct",
        task_count=None,
        baseline=None,
        worker_blocks_per_task=1,
        dag_shape="fork_join",
        tensor_tile=None,
    ):
        seen.update(
            {
                "device": device,
                "n": n,
                "arch": arch,
                "mode": mode,
                "task_count": task_count,
                "baseline": baseline,
                "worker_blocks_per_task": worker_blocks_per_task,
                "dag_shape": dag_shape,
                "tensor_tile": tensor_tile,
            }
        )
        return {
            "baseline": baseline,
            "n": n,
            "task_count": task_count or 3,
            "dag_shape": dag_shape,
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "run_persistent_sample", fake_run_persistent_sample)

    result = cuda_benchmark.run_single_sample(
        baseline="pto_persistent_dag_scalar_scale",
        device=3,
        n=1024,
        block_dim=128,
        arch="compute_80",
    )

    assert seen == {
        "device": 3,
        "n": 1024,
        "arch": "compute_80",
        "mode": "dag",
        "task_count": None,
        "baseline": "pto_persistent_dag_scalar_scale",
        "worker_blocks_per_task": 1,
        "dag_shape": "scalar_scale",
        "tensor_tile": None,
    }
    assert result["baseline"] == "pto_persistent_dag_scalar_scale"


def test_run_single_sample_dispatches_triad_dag(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    def fake_run_persistent_sample(
        device,
        n,
        arch,
        mode="direct",
        task_count=None,
        baseline=None,
        worker_blocks_per_task=1,
        dag_shape="fork_join",
        tensor_tile=None,
    ):
        seen.update(
            {
                "device": device,
                "n": n,
                "arch": arch,
                "mode": mode,
                "task_count": task_count,
                "baseline": baseline,
                "worker_blocks_per_task": worker_blocks_per_task,
                "dag_shape": dag_shape,
                "tensor_tile": tensor_tile,
            }
        )
        return {
            "baseline": baseline,
            "n": n,
            "task_count": task_count or 3,
            "dag_shape": dag_shape,
            "tensor_args": {"c": "tmp0"},
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "run_persistent_sample", fake_run_persistent_sample)

    result = cuda_benchmark.run_single_sample(
        baseline="pto_persistent_dag_triad",
        device=3,
        n=1024,
        block_dim=128,
        arch="compute_80",
    )

    assert seen == {
        "device": 3,
        "n": 1024,
        "arch": "compute_80",
        "mode": "dag",
        "task_count": None,
        "baseline": "pto_persistent_dag_triad",
        "worker_blocks_per_task": 1,
        "dag_shape": "triad",
        "tensor_tile": None,
    }
    assert result["baseline"] == "pto_persistent_dag_triad"
    assert result["tensor_args"] == {"c": "tmp0"}


def test_run_single_sample_dispatches_quad_dag(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    def fake_run_persistent_sample(
        device,
        n,
        arch,
        mode="direct",
        task_count=None,
        baseline=None,
        worker_blocks_per_task=1,
        dag_shape="fork_join",
        tensor_tile=None,
    ):
        seen.update(
            {
                "device": device,
                "n": n,
                "arch": arch,
                "mode": mode,
                "task_count": task_count,
                "baseline": baseline,
                "worker_blocks_per_task": worker_blocks_per_task,
                "dag_shape": dag_shape,
                "tensor_tile": tensor_tile,
            }
        )
        return {
            "baseline": baseline,
            "n": n,
            "task_count": task_count or 3,
            "dag_shape": dag_shape,
            "tensor_args": {"c": "tmp0", "d": "tmp3"},
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "run_persistent_sample", fake_run_persistent_sample)

    result = cuda_benchmark.run_single_sample(
        baseline="pto_persistent_dag_quad",
        device=3,
        n=1024,
        block_dim=128,
        arch="compute_80",
    )

    assert seen == {
        "device": 3,
        "n": 1024,
        "arch": "compute_80",
        "mode": "dag",
        "task_count": None,
        "baseline": "pto_persistent_dag_quad",
        "worker_blocks_per_task": 1,
        "dag_shape": "quad",
        "tensor_tile": None,
    }
    assert result["baseline"] == "pto_persistent_dag_quad"
    assert result["tensor_args"] == {"c": "tmp0", "d": "tmp3"}


def test_run_single_sample_dispatches_unary_square_dag(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    def fake_run_persistent_sample(
        device,
        n,
        arch,
        mode="direct",
        task_count=None,
        baseline=None,
        worker_blocks_per_task=1,
        dag_shape="fork_join",
        tensor_tile=None,
    ):
        seen.update(
            {
                "device": device,
                "n": n,
                "arch": arch,
                "mode": mode,
                "task_count": task_count,
                "baseline": baseline,
                "worker_blocks_per_task": worker_blocks_per_task,
                "dag_shape": dag_shape,
                "tensor_tile": tensor_tile,
            }
        )
        return {
            "baseline": baseline,
            "n": n,
            "task_count": task_count or 3,
            "dag_shape": dag_shape,
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "run_persistent_sample", fake_run_persistent_sample)

    result = cuda_benchmark.run_single_sample(
        baseline="pto_persistent_dag_unary_square",
        device=3,
        n=1024,
        block_dim=128,
        arch="compute_80",
    )

    assert seen == {
        "device": 3,
        "n": 1024,
        "arch": "compute_80",
        "mode": "dag",
        "task_count": None,
        "baseline": "pto_persistent_dag_unary_square",
        "worker_blocks_per_task": 1,
        "dag_shape": "unary_square",
        "tensor_tile": None,
    }
    assert result["baseline"] == "pto_persistent_dag_unary_square"


def test_run_single_sample_dispatches_graph_unary_square_dag(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    def fake_run_persistent_sample(
        device,
        n,
        arch,
        mode="direct",
        task_count=None,
        baseline=None,
        worker_blocks_per_task=1,
        dag_shape="fork_join",
        tensor_tile=None,
    ):
        seen.update(
            {
                "device": device,
                "n": n,
                "arch": arch,
                "mode": mode,
                "task_count": task_count,
                "baseline": baseline,
                "worker_blocks_per_task": worker_blocks_per_task,
                "dag_shape": dag_shape,
                "tensor_tile": tensor_tile,
            }
        )
        return {
            "baseline": baseline,
            "n": n,
            "task_count": task_count or 3,
            "dag_shape": dag_shape,
            "dispatch_func_ids": [7, 1, 1],
            "graph_descriptor": {"tasks": 3, "dependents": [1, 2], "fanin": [0, 1, 1]},
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "run_persistent_sample", fake_run_persistent_sample)

    result = cuda_benchmark.run_single_sample(
        baseline="pto_persistent_dag_graph_unary_square",
        device=3,
        n=1024,
        block_dim=128,
        arch="compute_80",
    )

    assert seen == {
        "device": 3,
        "n": 1024,
        "arch": "compute_80",
        "mode": "dag",
        "task_count": None,
        "baseline": "pto_persistent_dag_graph_unary_square",
        "worker_blocks_per_task": 1,
        "dag_shape": "graph_descriptor_unary_square",
        "tensor_tile": None,
    }
    assert result["baseline"] == "pto_persistent_dag_graph_unary_square"
    assert result["dispatch_func_ids"] == [7, 1, 1]
    assert result["graph_descriptor"] == {"tasks": 3, "dependents": [1, 2], "fanin": [0, 1, 1]}


def test_run_single_sample_dispatches_generic_args_dag(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    def fake_run_persistent_sample(
        device,
        n,
        arch,
        mode="direct",
        task_count=None,
        baseline=None,
        worker_blocks_per_task=1,
        dag_shape="fork_join",
        tensor_tile=None,
    ):
        seen.update(
            {
                "device": device,
                "n": n,
                "arch": arch,
                "mode": mode,
                "task_count": task_count,
                "baseline": baseline,
                "worker_blocks_per_task": worker_blocks_per_task,
                "dag_shape": dag_shape,
                "tensor_tile": tensor_tile,
            }
        )
        return {
            "baseline": baseline,
            "n": n,
            "task_count": task_count or 3,
            "dag_shape": dag_shape,
            "generic_args": {"tensor_args": {"0": "tmp0", "1": "tmp3"}, "scalar_args": [1.5, 0.25]},
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "run_persistent_sample", fake_run_persistent_sample)

    result = cuda_benchmark.run_single_sample(
        baseline="pto_persistent_dag_generic_args",
        device=3,
        n=1024,
        block_dim=128,
        arch="compute_80",
    )

    assert seen == {
        "device": 3,
        "n": 1024,
        "arch": "compute_80",
        "mode": "dag",
        "task_count": None,
        "baseline": "pto_persistent_dag_generic_args",
        "worker_blocks_per_task": 1,
        "dag_shape": "generic_args",
        "tensor_tile": None,
    }
    assert result["baseline"] == "pto_persistent_dag_generic_args"
    assert result["generic_args"] == {"tensor_args": {"0": "tmp0", "1": "tmp3"}, "scalar_args": [1.5, 0.25]}


def test_run_single_sample_dispatches_graph_descriptor_dag(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    def fake_run_persistent_sample(
        device,
        n,
        arch,
        mode="direct",
        task_count=None,
        baseline=None,
        worker_blocks_per_task=1,
        dag_shape="fork_join",
        tensor_tile=None,
    ):
        seen.update(
            {
                "device": device,
                "n": n,
                "arch": arch,
                "mode": mode,
                "task_count": task_count,
                "baseline": baseline,
                "worker_blocks_per_task": worker_blocks_per_task,
                "dag_shape": dag_shape,
                "tensor_tile": tensor_tile,
            }
        )
        return {
            "baseline": baseline,
            "n": n,
            "task_count": task_count or 3,
            "dag_shape": dag_shape,
            "graph_descriptor": {"tasks": 3, "dependents": [2, 2], "fanin": [0, 0, 2]},
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "run_persistent_sample", fake_run_persistent_sample)

    result = cuda_benchmark.run_single_sample(
        baseline="pto_persistent_dag_graph",
        device=3,
        n=1024,
        block_dim=128,
        arch="compute_80",
    )

    assert seen == {
        "device": 3,
        "n": 1024,
        "arch": "compute_80",
        "mode": "dag",
        "task_count": None,
        "baseline": "pto_persistent_dag_graph",
        "worker_blocks_per_task": 1,
        "dag_shape": "graph_descriptor",
        "tensor_tile": None,
    }
    assert result["baseline"] == "pto_persistent_dag_graph"
    assert result["graph_descriptor"] == {"tasks": 3, "dependents": [2, 2], "fanin": [0, 0, 2]}


def test_run_single_sample_dispatches_graph_generic_args4_dag(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    def fake_run_persistent_sample(
        device,
        n,
        arch,
        mode="direct",
        task_count=None,
        baseline=None,
        worker_blocks_per_task=1,
        dag_shape="fork_join",
        tensor_tile=None,
    ):
        seen.update(
            {
                "device": device,
                "n": n,
                "arch": arch,
                "mode": mode,
                "task_count": task_count,
                "baseline": baseline,
                "worker_blocks_per_task": worker_blocks_per_task,
                "dag_shape": dag_shape,
                "tensor_tile": tensor_tile,
            }
        )
        return {
            "baseline": baseline,
            "n": n,
            "task_count": task_count or 3,
            "dag_shape": dag_shape,
            "graph_descriptor": {"tasks": 3, "dependents": [2, 2], "fanin": [0, 0, 2]},
            "tensor_args": {
                "tensor_args[0]": "tmp0",
                "tensor_args[1]": "tmp3",
                "tensor_args[2]": "a",
                "tensor_args[3]": "b",
            },
            "scalar_args": {
                "scalar_args[0]": 1.5,
                "scalar_args[1]": 0.25,
                "scalar_args[2]": 0.125,
                "scalar_args[3]": 0.0625,
            },
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "run_persistent_sample", fake_run_persistent_sample)

    result = cuda_benchmark.run_single_sample(
        baseline="pto_persistent_dag_graph_generic_args4",
        device=3,
        n=1024,
        block_dim=128,
        arch="compute_80",
    )

    assert seen == {
        "device": 3,
        "n": 1024,
        "arch": "compute_80",
        "mode": "dag",
        "task_count": None,
        "baseline": "pto_persistent_dag_graph_generic_args4",
        "worker_blocks_per_task": 1,
        "dag_shape": "graph_descriptor_generic_args4",
        "tensor_tile": None,
    }
    assert result["baseline"] == "pto_persistent_dag_graph_generic_args4"
    assert result["graph_descriptor"] == {"tasks": 3, "dependents": [2, 2], "fanin": [0, 0, 2]}
    assert result["tensor_args"]["tensor_args[2]"] == "a"
    assert result["scalar_args"]["scalar_args[3]"] == 0.0625


def test_run_single_sample_dispatches_graph_triad_dag(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    def fake_run_persistent_sample(
        device,
        n,
        arch,
        mode="direct",
        task_count=None,
        baseline=None,
        worker_blocks_per_task=1,
        dag_shape="fork_join",
        tensor_tile=None,
    ):
        seen.update(
            {
                "device": device,
                "n": n,
                "arch": arch,
                "mode": mode,
                "task_count": task_count,
                "baseline": baseline,
                "worker_blocks_per_task": worker_blocks_per_task,
                "dag_shape": dag_shape,
                "tensor_tile": tensor_tile,
            }
        )
        return {
            "baseline": baseline,
            "n": n,
            "task_count": task_count or 3,
            "dag_shape": dag_shape,
            "dispatch_func_ids": [6, 2, 1],
            "graph_descriptor": {"tasks": 3, "dependents": [2, 2], "fanin": [0, 0, 2]},
            "tensor_args": {"c": "tmp0"},
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "run_persistent_sample", fake_run_persistent_sample)

    result = cuda_benchmark.run_single_sample(
        baseline="pto_persistent_dag_graph_triad",
        device=3,
        n=1024,
        block_dim=128,
        arch="compute_80",
    )

    assert seen == {
        "device": 3,
        "n": 1024,
        "arch": "compute_80",
        "mode": "dag",
        "task_count": None,
        "baseline": "pto_persistent_dag_graph_triad",
        "worker_blocks_per_task": 1,
        "dag_shape": "graph_descriptor_triad",
        "tensor_tile": None,
    }
    assert result["baseline"] == "pto_persistent_dag_graph_triad"
    assert result["dispatch_func_ids"] == [6, 2, 1]
    assert result["graph_descriptor"] == {"tasks": 3, "dependents": [2, 2], "fanin": [0, 0, 2]}
    assert result["tensor_args"] == {"c": "tmp0"}


def test_run_single_sample_dispatches_graph_quad_dag(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    def fake_run_persistent_sample(
        device,
        n,
        arch,
        mode="direct",
        task_count=None,
        baseline=None,
        worker_blocks_per_task=1,
        dag_shape="fork_join",
        tensor_tile=None,
    ):
        seen.update(
            {
                "device": device,
                "n": n,
                "arch": arch,
                "mode": mode,
                "task_count": task_count,
                "baseline": baseline,
                "worker_blocks_per_task": worker_blocks_per_task,
                "dag_shape": dag_shape,
                "tensor_tile": tensor_tile,
            }
        )
        return {
            "baseline": baseline,
            "n": n,
            "task_count": task_count or 3,
            "dag_shape": dag_shape,
            "dispatch_func_ids": [8, 2, 1],
            "graph_descriptor": {"tasks": 3, "dependents": [2, 2], "fanin": [0, 0, 2]},
            "tensor_args": {"c": "tmp0", "d": "tmp3"},
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "run_persistent_sample", fake_run_persistent_sample)

    result = cuda_benchmark.run_single_sample(
        baseline="pto_persistent_dag_graph_quad",
        device=3,
        n=1024,
        block_dim=128,
        arch="compute_80",
    )

    assert seen == {
        "device": 3,
        "n": 1024,
        "arch": "compute_80",
        "mode": "dag",
        "task_count": None,
        "baseline": "pto_persistent_dag_graph_quad",
        "worker_blocks_per_task": 1,
        "dag_shape": "graph_descriptor_quad",
        "tensor_tile": None,
    }
    assert result["baseline"] == "pto_persistent_dag_graph_quad"
    assert result["dispatch_func_ids"] == [8, 2, 1]
    assert result["graph_descriptor"] == {"tasks": 3, "dependents": [2, 2], "fanin": [0, 0, 2]}
    assert result["tensor_args"] == {"c": "tmp0", "d": "tmp3"}


def test_run_single_sample_dispatches_graph_chain_dag(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    def fake_run_persistent_sample(
        device,
        n,
        arch,
        mode="direct",
        task_count=None,
        baseline=None,
        worker_blocks_per_task=1,
        dag_shape="fork_join",
        tensor_tile=None,
    ):
        seen.update(
            {
                "device": device,
                "n": n,
                "arch": arch,
                "mode": mode,
                "task_count": task_count,
                "baseline": baseline,
                "worker_blocks_per_task": worker_blocks_per_task,
                "dag_shape": dag_shape,
                "tensor_tile": tensor_tile,
            }
        )
        return {
            "baseline": baseline,
            "n": n,
            "task_count": task_count or 5,
            "dag_shape": dag_shape,
            "graph_descriptor": {
                "tasks": 5,
                "dependents": [2, 2, 3, 4],
                "fanin": [0, 0, 2, 1, 1],
            },
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "run_persistent_sample", fake_run_persistent_sample)

    result = cuda_benchmark.run_single_sample(
        baseline="pto_persistent_dag_graph_chain",
        device=3,
        n=1024,
        block_dim=128,
        arch="compute_80",
    )

    assert seen == {
        "device": 3,
        "n": 1024,
        "arch": "compute_80",
        "mode": "dag",
        "task_count": None,
        "baseline": "pto_persistent_dag_graph_chain",
        "worker_blocks_per_task": 1,
        "dag_shape": "graph_descriptor_chain",
        "tensor_tile": None,
    }
    assert result["baseline"] == "pto_persistent_dag_graph_chain"
    assert result["graph_descriptor"] == {
        "tasks": 5,
        "dependents": [2, 2, 3, 4],
        "fanin": [0, 0, 2, 1, 1],
    }


def test_run_single_sample_dispatches_graph_scratch_reuse_dag(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    def fake_run_persistent_sample(
        device,
        n,
        arch,
        mode="direct",
        task_count=None,
        baseline=None,
        worker_blocks_per_task=1,
        dag_shape="fork_join",
        tensor_tile=None,
    ):
        seen.update(
            {
                "device": device,
                "n": n,
                "arch": arch,
                "mode": mode,
                "task_count": task_count,
                "baseline": baseline,
                "worker_blocks_per_task": worker_blocks_per_task,
                "dag_shape": dag_shape,
                "tensor_tile": tensor_tile,
            }
        )
        return {
            "baseline": baseline,
            "n": n,
            "task_count": task_count or 6,
            "dag_shape": dag_shape,
            "graph_descriptor": {
                "tasks": 6,
                "dependents": [2, 2, 3, 4, 5, 5],
                "fanin": [0, 0, 2, 1, 1, 2],
            },
            "scratch_reuse": {"reused_buffer": "tmp0", "reuse_task": 4},
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "run_persistent_sample", fake_run_persistent_sample)

    result = cuda_benchmark.run_single_sample(
        baseline="pto_persistent_dag_graph_scratch_reuse",
        device=3,
        n=1024,
        block_dim=128,
        arch="compute_80",
    )

    assert seen == {
        "device": 3,
        "n": 1024,
        "arch": "compute_80",
        "mode": "dag",
        "task_count": None,
        "baseline": "pto_persistent_dag_graph_scratch_reuse",
        "worker_blocks_per_task": 1,
        "dag_shape": "graph_descriptor_scratch_reuse",
        "tensor_tile": None,
    }
    assert result["baseline"] == "pto_persistent_dag_graph_scratch_reuse"
    assert result["graph_descriptor"] == {
        "tasks": 6,
        "dependents": [2, 2, 3, 4, 5, 5],
        "fanin": [0, 0, 2, 1, 1, 2],
    }
    assert result["scratch_reuse"] == {"reused_buffer": "tmp0", "reuse_task": 4}


def test_run_single_sample_dispatches_graph_diamond_dag(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    def fake_run_persistent_sample(
        device,
        n,
        arch,
        mode="direct",
        task_count=None,
        baseline=None,
        worker_blocks_per_task=1,
        dag_shape="fork_join",
        tensor_tile=None,
    ):
        seen.update(
            {
                "device": device,
                "n": n,
                "arch": arch,
                "mode": mode,
                "task_count": task_count,
                "baseline": baseline,
                "worker_blocks_per_task": worker_blocks_per_task,
                "dag_shape": dag_shape,
                "tensor_tile": tensor_tile,
            }
        )
        return {
            "baseline": baseline,
            "n": n,
            "task_count": task_count or 5,
            "dag_shape": dag_shape,
            "graph_descriptor": {
                "tasks": 5,
                "dependents": [2, 3, 2, 3, 4, 4],
                "fanin": [0, 0, 2, 2, 2],
            },
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "run_persistent_sample", fake_run_persistent_sample)

    result = cuda_benchmark.run_single_sample(
        baseline="pto_persistent_dag_graph_diamond",
        device=3,
        n=1024,
        block_dim=128,
        arch="compute_80",
    )

    assert seen == {
        "device": 3,
        "n": 1024,
        "arch": "compute_80",
        "mode": "dag",
        "task_count": None,
        "baseline": "pto_persistent_dag_graph_diamond",
        "worker_blocks_per_task": 1,
        "dag_shape": "graph_descriptor_diamond",
        "tensor_tile": None,
    }
    assert result["baseline"] == "pto_persistent_dag_graph_diamond"
    assert result["graph_descriptor"] == {
        "tasks": 5,
        "dependents": [2, 3, 2, 3, 4, 4],
        "fanin": [0, 0, 2, 2, 2],
    }


def test_run_single_sample_dispatches_graph_tagged_inout_dag(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    def fake_run_persistent_sample(
        device,
        n,
        arch,
        mode="direct",
        task_count=None,
        baseline=None,
        worker_blocks_per_task=1,
        dag_shape="fork_join",
        tensor_tile=None,
    ):
        seen.update(
            {
                "device": device,
                "n": n,
                "arch": arch,
                "mode": mode,
                "task_count": task_count,
                "baseline": baseline,
                "worker_blocks_per_task": worker_blocks_per_task,
                "dag_shape": dag_shape,
                "tensor_tile": tensor_tile,
            }
        )
        return {
            "baseline": baseline,
            "n": n,
            "task_count": task_count or 3,
            "dag_shape": dag_shape,
            "graph_descriptor": {
                "tasks": 3,
                "dependents": [1, 2],
                "fanin": [0, 1, 1],
            },
            "graph_task_args": [
                {"task": 0, "a": "input:a", "b": "input:b", "out": "output:tmp1"},
                {"task": 1, "a": "inout:tmp1", "b": "input:b", "out": "inout:tmp1"},
                {"task": 2, "a": "input:tmp1", "b": "input:a", "out": "output_existing:out"},
            ],
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "run_persistent_sample", fake_run_persistent_sample)

    result = cuda_benchmark.run_single_sample(
        baseline="pto_persistent_dag_graph_tagged_inout",
        device=3,
        n=1024,
        block_dim=128,
        arch="compute_80",
    )

    assert seen == {
        "device": 3,
        "n": 1024,
        "arch": "compute_80",
        "mode": "dag",
        "task_count": None,
        "baseline": "pto_persistent_dag_graph_tagged_inout",
        "worker_blocks_per_task": 1,
        "dag_shape": "graph_descriptor_tagged_inout",
        "tensor_tile": None,
    }
    assert result["baseline"] == "pto_persistent_dag_graph_tagged_inout"
    assert result["graph_descriptor"] == {
        "tasks": 3,
        "dependents": [1, 2],
        "fanin": [0, 1, 1],
    }
    assert result["graph_task_args"][1]["a"] == "inout:tmp1"


def test_run_single_sample_dispatches_graph_tagged_scalar_dag(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    def fake_run_persistent_sample(
        device,
        n,
        arch,
        mode="direct",
        task_count=None,
        baseline=None,
        worker_blocks_per_task=1,
        dag_shape="fork_join",
        tensor_tile=None,
    ):
        seen.update(
            {
                "device": device,
                "n": n,
                "arch": arch,
                "mode": mode,
                "task_count": task_count,
                "baseline": baseline,
                "worker_blocks_per_task": worker_blocks_per_task,
                "dag_shape": dag_shape,
                "tensor_tile": tensor_tile,
            }
        )
        return {
            "baseline": baseline,
            "n": n,
            "task_count": task_count or 3,
            "dag_shape": dag_shape,
            "graph_descriptor": {
                "tasks": 3,
                "dependents": [2, 2],
                "fanin": [0, 0, 2],
            },
            "graph_task_args": {
                "task0": "input:a,input:b,output:tmp1,scalar:scalar_args[0],scalar:scalar_args[1]",
                "task1": "input:a,input:b,output:tmp2",
                "task2": "input:tmp1,input:tmp2,output_existing:out",
            },
            "scalar_args": {"scalar_args[0]": 1.5, "scalar_args[1]": 0.25},
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "run_persistent_sample", fake_run_persistent_sample)

    result = cuda_benchmark.run_single_sample(
        baseline="pto_persistent_dag_graph_tagged",
        device=3,
        n=1024,
        block_dim=128,
        arch="compute_80",
    )

    assert seen == {
        "device": 3,
        "n": 1024,
        "arch": "compute_80",
        "mode": "dag",
        "task_count": None,
        "baseline": "pto_persistent_dag_graph_tagged",
        "worker_blocks_per_task": 1,
        "dag_shape": "graph_descriptor_tagged",
        "tensor_tile": None,
    }
    assert result["baseline"] == "pto_persistent_dag_graph_tagged"
    assert result["graph_descriptor"] == {
        "tasks": 3,
        "dependents": [2, 2],
        "fanin": [0, 0, 2],
    }
    assert result["scalar_args"] == {"scalar_args[0]": 1.5, "scalar_args[1]": 0.25}


def test_run_single_sample_dispatches_graph_tensor_tile_dag(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    def fake_run_persistent_sample(
        device,
        n,
        arch,
        mode="direct",
        task_count=None,
        baseline=None,
        worker_blocks_per_task=1,
        dag_shape="fork_join",
        tensor_tile=None,
    ):
        seen.update(
            {
                "device": device,
                "n": n,
                "arch": arch,
                "mode": mode,
                "task_count": task_count,
                "baseline": baseline,
                "worker_blocks_per_task": worker_blocks_per_task,
                "dag_shape": dag_shape,
                "tensor_tile": tensor_tile,
            }
        )
        assert tensor_tile is not None
        return {
            "baseline": baseline,
            "n": n,
            "task_count": task_count or 4,
            "dag_shape": dag_shape,
            "graph_descriptor": {"tasks": 4, "dependents": [1, 2, 3, 3], "fanin": [0, 1, 1, 2]},
            "tensor_tile": {**tensor_tile, "tile_count": 2},
            "device_wall_ns": 10,
            "status": "pass",
        }

    tensor_tile = {"rows": 16, "cols": 16, "inner": 16}
    monkeypatch.setattr(cuda_benchmark, "run_persistent_sample", fake_run_persistent_sample)

    result = cuda_benchmark.run_single_sample(
        baseline="pto_persistent_dag_graph_tensor",
        device=3,
        n=512,
        block_dim=128,
        arch="compute_80",
        tensor_tile=tensor_tile,
    )

    assert seen == {
        "device": 3,
        "n": 512,
        "arch": "compute_80",
        "mode": "dag",
        "task_count": None,
        "baseline": "pto_persistent_dag_graph_tensor",
        "worker_blocks_per_task": 1,
        "dag_shape": "graph_tensor_tile",
        "tensor_tile": tensor_tile,
    }
    assert result["baseline"] == "pto_persistent_dag_graph_tensor"
    assert result["dag_shape"] == "graph_tensor_tile"
    assert result["graph_descriptor"] == {"tasks": 4, "dependents": [1, 2, 3, 3], "fanin": [0, 1, 1, 2]}


def test_run_persistent_sample_defaults_tensor_core_tile_to_four_tasks(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    def fake_run_persistent_smoke(**kwargs):
        seen.update(kwargs)
        return {
            "baseline": "pto_persistent_dag_tensor_core",
            "n": kwargs["n"],
            "task_count": kwargs["task_count"],
            "dag_shape": kwargs["dag_shape"],
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "run_persistent_smoke", fake_run_persistent_smoke)

    result = cuda_benchmark.run_persistent_sample(
        device=3,
        n=256,
        arch="compute_80",
        mode="dag",
        baseline="pto_persistent_dag_tensor_core",
        dag_shape="tensor_core_tile",
        tensor_tile={"rows": 16, "cols": 16, "inner": 16},
    )

    assert seen["task_count"] == 4
    assert seen["dag_shape"] == "tensor_core_tile"
    assert seen["tensor_rows"] == 16
    assert seen["tensor_cols"] == 16
    assert seen["tensor_inner"] == 16
    assert result["baseline"] == "pto_persistent_dag_tensor_core"


def test_run_persistent_sample_accepts_multi_wmma_tensor_core_tile(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    def fake_run_persistent_smoke(**kwargs):
        seen.update(kwargs)
        return {
            "baseline": "pto_persistent_dag_tensor_core",
            "n": kwargs["n"],
            "task_count": kwargs["task_count"],
            "dag_shape": kwargs["dag_shape"],
            "tensor_tile": {
                "rows": kwargs["tensor_rows"],
                "cols": kwargs["tensor_cols"],
                "inner": kwargs["tensor_inner"],
                "tile_count": 2,
            },
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "run_persistent_smoke", fake_run_persistent_smoke)

    result = cuda_benchmark.run_persistent_sample(
        device=3,
        n=1024,
        arch="compute_80",
        mode="dag",
        baseline="pto_persistent_dag_tensor_core",
        dag_shape="tensor_core_tile",
        tensor_tile={"rows": 32, "cols": 16, "inner": 16},
    )

    assert seen["task_count"] == 4
    assert seen["dag_shape"] == "tensor_core_tile"
    assert seen["tensor_rows"] == 32
    assert seen["tensor_cols"] == 16
    assert seen["tensor_inner"] == 16
    assert result["tensor_tile"]["rows"] == 32


def test_run_persistent_sample_rejects_incompatible_tensor_core_tile(monkeypatch):
    cuda_benchmark = _load_benchmark_module()

    def fake_run_persistent_smoke(**kwargs):
        raise AssertionError("incompatible tensor-core tile should fail before launch")

    monkeypatch.setattr(cuda_benchmark, "run_persistent_smoke", fake_run_persistent_smoke)

    try:
        cuda_benchmark.run_persistent_sample(
            device=3,
            n=1024,
            arch="compute_80",
            mode="dag",
            baseline="pto_persistent_dag_tensor_core",
            dag_shape="tensor_core_tile",
            tensor_tile={"rows": 32, "cols": 8, "inner": 16},
        )
    except ValueError as exc:
        assert "tensor_core" in str(exc)
        assert "multiples of 16" in str(exc)
    else:
        raise AssertionError("expected ValueError for incompatible tensor-core tile")


def test_run_single_sample_dispatches_tensor_core_dag(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    def fake_run_persistent_sample(
        device,
        n,
        arch,
        mode="direct",
        task_count=None,
        baseline=None,
        worker_blocks_per_task=1,
        dag_shape="fork_join",
        tensor_tile=None,
    ):
        seen.update(
            {
                "device": device,
                "n": n,
                "arch": arch,
                "mode": mode,
                "task_count": task_count,
                "baseline": baseline,
                "worker_blocks_per_task": worker_blocks_per_task,
                "dag_shape": dag_shape,
                "tensor_tile": tensor_tile,
            }
        )
        return {
            "baseline": baseline,
            "n": n,
            "task_count": task_count or 4,
            "dag_shape": dag_shape,
            "device_wall_ns": 10,
            "status": "pass",
        }

    tensor_tile = {"rows": 16, "cols": 16, "inner": 16}
    monkeypatch.setattr(cuda_benchmark, "run_persistent_sample", fake_run_persistent_sample)

    result = cuda_benchmark.run_single_sample(
        baseline="pto_persistent_dag_tensor_core",
        device=3,
        n=256,
        block_dim=128,
        arch="compute_80",
        tensor_tile=tensor_tile,
    )

    assert seen == {
        "device": 3,
        "n": 256,
        "arch": "compute_80",
        "mode": "dag",
        "task_count": None,
        "baseline": "pto_persistent_dag_tensor_core",
        "worker_blocks_per_task": 1,
        "dag_shape": "tensor_core_tile",
        "tensor_tile": tensor_tile,
    }
    assert result["baseline"] == "pto_persistent_dag_tensor_core"


def test_run_single_sample_dispatches_graph_tensor_core_dag(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    def fake_run_persistent_sample(
        device,
        n,
        arch,
        mode="direct",
        task_count=None,
        baseline=None,
        worker_blocks_per_task=1,
        dag_shape="fork_join",
        tensor_tile=None,
    ):
        seen.update(
            {
                "device": device,
                "n": n,
                "arch": arch,
                "mode": mode,
                "task_count": task_count,
                "baseline": baseline,
                "worker_blocks_per_task": worker_blocks_per_task,
                "dag_shape": dag_shape,
                "tensor_tile": tensor_tile,
            }
        )
        return {
            "baseline": baseline,
            "n": n,
            "task_count": task_count or 4,
            "dag_shape": dag_shape,
            "dispatch_func_ids": [10, 1, 2, 1],
            "graph_descriptor": {"tasks": 4, "dependents": [1, 2, 3, 3], "fanin": [0, 1, 1, 2]},
            "tensor_core": {"api": "wmma", "mma_shape": "m16n16k8", "input": "tf32", "accumulator": "f32"},
            "device_wall_ns": 10,
            "status": "pass",
        }

    tensor_tile = {"rows": 16, "cols": 16, "inner": 16}
    monkeypatch.setattr(cuda_benchmark, "run_persistent_sample", fake_run_persistent_sample)

    result = cuda_benchmark.run_single_sample(
        baseline="pto_persistent_dag_graph_tensor_core",
        device=3,
        n=256,
        block_dim=128,
        arch="compute_80",
        tensor_tile=tensor_tile,
    )

    assert seen == {
        "device": 3,
        "n": 256,
        "arch": "compute_80",
        "mode": "dag",
        "task_count": None,
        "baseline": "pto_persistent_dag_graph_tensor_core",
        "worker_blocks_per_task": 1,
        "dag_shape": "graph_tensor_core_tile",
        "tensor_tile": tensor_tile,
    }
    assert result["baseline"] == "pto_persistent_dag_graph_tensor_core"
    assert result["dag_shape"] == "graph_tensor_core_tile"
    assert result["dispatch_func_ids"] == [10, 1, 2, 1]
    assert result["graph_descriptor"] == {"tasks": 4, "dependents": [1, 2, 3, 3], "fanin": [0, 1, 1, 2]}
    assert result["tensor_core"]["api"] == "wmma"


def test_run_single_sample_dispatches_host_schedule_generic_args_baseline(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = {}

    def fake_run_pto_generic_args_sample(device, n, block_dim, arch):
        seen.update({"device": device, "n": n, "block_dim": block_dim, "arch": arch})
        return {"baseline": "pto_host_schedule_generic_args", "status": "pass"}

    monkeypatch.setattr(
        cuda_benchmark,
        "run_pto_generic_args_sample",
        fake_run_pto_generic_args_sample,
        raising=False,
    )

    result = cuda_benchmark.run_single_sample(
        baseline="pto_host_schedule_generic_args",
        device=3,
        n=64,
        block_dim=128,
        arch="compute_80",
    )

    assert seen == {"device": 3, "n": 64, "block_dim": 128, "arch": "compute_80"}
    assert result["baseline"] == "pto_host_schedule_generic_args"


def test_run_benchmark_passes_tensor_descriptor_to_tensor_dag(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = []

    def fake_compile_ptx(work_dir, arch):
        return b"ptx", f"fake-{arch}"

    def fake_run_single_sample(
        baseline,
        device,
        n,
        block_dim,
        arch,
        task_count=1,
        worker_blocks_per_task=1,
        tensor_tile=None,
    ):
        seen.append((baseline, tensor_tile))
        return {
            "baseline": baseline,
            "n": n,
            "task_count": task_count,
            "block_dim": block_dim,
            "host_wall_ns": 20,
            "device_wall_ns": 10,
            "status": "pass",
        }

    tensor_tile = {"rows": 8, "cols": 4, "inner": 12}
    monkeypatch.setattr(cuda_benchmark, "_compile_ptx", fake_compile_ptx)
    monkeypatch.setattr(cuda_benchmark, "run_single_sample", fake_run_single_sample)

    payload = cuda_benchmark.run_benchmark(
        device=3,
        sizes=[64],
        repeats=1,
        block_dim=128,
        arch="compute_80",
        label="unit",
        include_persistent=True,
        tensor_tile=tensor_tile,
    )

    tensor_baselines = {
        "pto_persistent_dag_tensor",
        "pto_persistent_dag_graph_tensor",
        "pto_persistent_dag_tensor_core",
        "pto_persistent_dag_graph_tensor_core",
        "cublas_sgemm",
        "cublas_sgemm_graph",
    }
    tensor_calls = [item for item in seen if item[0] in tensor_baselines]
    non_tensor_calls = [item for item in seen if item[0] not in tensor_baselines]
    assert payload["metadata"]["tensor_tile"] == tensor_tile
    assert tensor_calls == [
        ("pto_persistent_dag_tensor", tensor_tile),
        ("pto_persistent_dag_graph_tensor", tensor_tile),
        ("pto_persistent_dag_tensor_core", tensor_tile),
        ("pto_persistent_dag_graph_tensor_core", tensor_tile),
        ("cublas_sgemm", tensor_tile),
        ("cublas_sgemm_graph", tensor_tile),
    ]
    assert all(call[1] is None for call in non_tensor_calls)


def test_run_benchmark_can_include_same_work_batch_modes(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = []

    def fake_compile_ptx(work_dir, arch):
        return b"ptx", f"fake-{arch}"

    def fake_run_single_sample(baseline, device, n, block_dim, arch, task_count=1):
        seen.append((baseline, task_count))
        return {
            "baseline": baseline,
            "n": n,
            "task_count": task_count,
            "block_dim": block_dim,
            "host_wall_ns": 20,
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "_compile_ptx", fake_compile_ptx)
    monkeypatch.setattr(cuda_benchmark, "run_single_sample", fake_run_single_sample)

    payload = cuda_benchmark.run_benchmark(
        device=3,
        sizes=[1024],
        repeats=1,
        block_dim=128,
        arch="compute_80",
        label="unit",
        include_persistent=True,
        batch_tasks=6,
    )

    assert seen == [
        ("pto_host_schedule", 1),
        ("pto_host_schedule_compiler", 1),
        ("pto_host_schedule_unary_square", 1),
        ("pto_host_schedule_quad", 1),
        ("pto_host_schedule_generic_args", 1),
        ("direct_driver", 1),
        ("direct_driver_graph", 1),
        ("pto_persistent_device", 1),
        ("pto_persistent_queue", 1),
        ("pto_persistent_dag", 1),
        ("pto_persistent_dag_chain", 1),
        ("pto_persistent_dag_reuse", 1),
        ("pto_persistent_dag_scalar_axpy", 1),
        ("pto_persistent_dag_scalar_scale", 1),
        ("pto_persistent_dag_scalar_affine", 1),
        ("pto_persistent_dag_triad", 1),
        ("pto_persistent_dag_quad", 1),
        ("pto_persistent_dag_generic_args", 1),
        ("pto_persistent_dag_graph", 1),
        ("pto_persistent_dag_graph_generic_args4", 1),
        ("pto_persistent_dag_graph_chain", 1),
        ("pto_persistent_dag_graph_scratch_reuse", 1),
        ("pto_persistent_dag_graph_diamond", 1),
        ("pto_persistent_dag_graph_tagged", 1),
        ("pto_persistent_dag_graph_tagged_inout", 1),
        ("pto_persistent_dag_graph_triad", 1),
        ("pto_persistent_dag_graph_quad", 1),
        ("pto_persistent_dag_graph_unary_square", 1),
        ("pto_persistent_dag_unary_square", 1),
        ("pto_persistent_dag_tensor", 1),
        ("pto_persistent_dag_graph_tensor", 1),
        ("pto_persistent_dag_tensor_core", 1),
        ("pto_persistent_dag_graph_tensor_core", 1),
        ("cublas_sgemm", 1),
        ("cublas_sgemm_graph", 1),
        ("pto_host_schedule_batch", 6),
        ("pto_persistent_device_batch", 6),
        ("pto_persistent_queue_batch", 6),
    ]
    assert payload["metadata"]["batch_tasks"] == 6
    assert len(payload["results"]) == 38


def test_run_benchmark_can_include_worker_grid_batch_mode(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = []

    def fake_compile_ptx(work_dir, arch):
        return b"ptx", f"fake-{arch}"

    def fake_run_single_sample(baseline, device, n, block_dim, arch, task_count=1, worker_blocks_per_task=1):
        seen.append((baseline, task_count, worker_blocks_per_task))
        return {
            "baseline": baseline,
            "n": n,
            "task_count": task_count,
            "worker_blocks_per_task": worker_blocks_per_task,
            "block_dim": block_dim,
            "host_wall_ns": 20,
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "_compile_ptx", fake_compile_ptx)
    monkeypatch.setattr(cuda_benchmark, "run_single_sample", fake_run_single_sample)

    payload = cuda_benchmark.run_benchmark(
        device=3,
        sizes=[1024],
        repeats=1,
        block_dim=128,
        arch="compute_80",
        label="unit",
        include_persistent=True,
        batch_tasks=6,
        worker_blocks_per_task=4,
    )

    assert ("pto_persistent_device_grid_batch", 6, 4) in seen
    assert payload["metadata"]["worker_blocks_per_task"] == 4


def test_run_benchmark_can_sweep_worker_grid_batch_modes(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = []

    def fake_compile_ptx(work_dir, arch):
        return b"ptx", f"fake-{arch}"

    def fake_run_single_sample(baseline, device, n, block_dim, arch, task_count=1, worker_blocks_per_task=1):
        seen.append((baseline, task_count, worker_blocks_per_task))
        return {
            "baseline": baseline,
            "n": n,
            "task_count": task_count,
            "worker_blocks_per_task": worker_blocks_per_task,
            "block_dim": block_dim,
            "host_wall_ns": 20,
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "_compile_ptx", fake_compile_ptx)
    monkeypatch.setattr(cuda_benchmark, "run_single_sample", fake_run_single_sample)

    payload = cuda_benchmark.run_benchmark(
        device=3,
        sizes=[1024],
        repeats=1,
        block_dim=128,
        arch="compute_80",
        label="unit",
        include_persistent=True,
        batch_tasks=6,
        worker_blocks_per_task=[2, 4],
    )

    assert ("pto_persistent_device_grid_batch", 6, 2) in seen
    assert ("pto_persistent_device_grid_batch", 6, 4) in seen
    assert payload["metadata"]["worker_blocks_per_task_values"] == [2, 4]


def test_run_benchmark_can_sweep_batch_task_counts(monkeypatch):
    cuda_benchmark = _load_benchmark_module()
    seen = []

    def fake_compile_ptx(work_dir, arch):
        return b"ptx", f"fake-{arch}"

    def fake_run_single_sample(baseline, device, n, block_dim, arch, task_count=1, worker_blocks_per_task=1):
        seen.append((baseline, task_count, worker_blocks_per_task))
        return {
            "baseline": baseline,
            "n": n,
            "task_count": task_count,
            "worker_blocks_per_task": worker_blocks_per_task,
            "block_dim": block_dim,
            "host_wall_ns": 20,
            "device_wall_ns": 10,
            "status": "pass",
        }

    monkeypatch.setattr(cuda_benchmark, "_compile_ptx", fake_compile_ptx)
    monkeypatch.setattr(cuda_benchmark, "run_single_sample", fake_run_single_sample)

    payload = cuda_benchmark.run_benchmark(
        device=3,
        sizes=[1024],
        repeats=1,
        block_dim=128,
        arch="compute_80",
        label="unit",
        include_persistent=True,
        batch_tasks=[2, 6],
        worker_blocks_per_task=[4, 8],
    )

    assert ("pto_host_schedule_batch", 2, 1) in seen
    assert ("pto_persistent_device_batch", 6, 1) in seen
    assert ("pto_persistent_queue_batch", 6, 1) in seen
    assert ("pto_persistent_device_grid_batch", 2, 4) in seen
    assert ("pto_persistent_device_grid_batch", 2, 8) in seen
    assert ("pto_persistent_device_grid_batch", 6, 4) in seen
    assert ("pto_persistent_device_grid_batch", 6, 8) in seen
    assert payload["metadata"]["batch_tasks"] == [2, 6]
    assert payload["metadata"]["batch_task_values"] == [2, 6]


def test_parse_batch_tasks_accepts_comma_separated_values():
    cuda_benchmark = _load_benchmark_module()

    assert cuda_benchmark._parse_batch_tasks("0") == []
    assert cuda_benchmark._parse_batch_tasks("6") == [6]
    assert cuda_benchmark._parse_batch_tasks("2,6,2") == [2, 6]


def test_render_report_describes_stream_concurrency_rows():
    cuda_benchmark = _load_benchmark_module()
    payload = {
        "metadata": {
            "label": "stream-unit",
            "git_commit": "abc123",
            "paper_setup": "stream concurrency microbenchmark",
            "stream_pool_size": 6,
        },
        "results": [
            {"machine": "a100-local", "baseline": "pto_stream_serial", "n": 2, "device_wall_ns": 2000},
            {"machine": "a100-local", "baseline": "pto_stream_parallel", "n": 2, "device_wall_ns": 1200},
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)

    assert "pto_stream_serial" in report
    assert "- Host stream pool size: `6`." in report
    assert "| a100-local | pto_stream_parallel | 2 | 1 | 1 | 1 | 1200 | 1200 | 0.60x |" in report
    assert "`pto_stream_parallel` measures two independent PTO launches" in report
    assert "stream rows use `pto_stream_serial` as their reference" in report


def test_cuda_benchmark_stream_concurrency_cli_accepts_stream_pool_size(tmp_path, monkeypatch, capsys):
    cuda_benchmark = _load_benchmark_module()
    calls = []

    def fake_run_stream_concurrency_benchmark(device, repeats, arch, label, stream_pool_size):
        calls.append(
            {
                "device": device,
                "repeats": repeats,
                "arch": arch,
                "label": label,
                "stream_pool_size": stream_pool_size,
            }
        )
        return {
            "metadata": {
                "label": label,
                "git_commit": "abc123",
                "stream_pool_size": stream_pool_size,
            },
            "results": [],
        }

    written = []
    monkeypatch.setattr(cuda_benchmark, "run_stream_concurrency_benchmark", fake_run_stream_concurrency_benchmark)
    monkeypatch.setattr(
        cuda_benchmark, "write_report", lambda payload, output_dir: written.append((payload, output_dir))
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cuda_benchmark.py",
            "--stream-concurrency",
            "--stream-pool-size",
            "6",
            "--device",
            "1",
            "--repeats",
            "3",
            "--arch",
            "compute_90",
            "--label",
            "stream-cli",
            "--output-dir",
            str(tmp_path),
        ],
    )

    cuda_benchmark.main()
    captured = capsys.readouterr()

    assert calls == [
        {
            "device": 1,
            "repeats": 3,
            "arch": "compute_90",
            "label": "stream-cli",
            "stream_pool_size": 6,
        }
    ]
    assert written[0][0]["metadata"]["stream_pool_size"] == 6
    assert written[0][1] == tmp_path
    assert '"stream_pool_size": 6' in captured.out
