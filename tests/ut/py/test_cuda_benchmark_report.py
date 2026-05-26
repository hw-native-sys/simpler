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

    assert "| Dispatch | Scheduler errors | Resource policy | Scalar args | Tensor args |" in markdown
    assert "| a100 | pass | persistent_device | dag/tensor_tile | 4096 | `compute_80` | 102400 | 122260 |" in markdown
    assert "| h200 | pass | persistent_device | dag/tensor_tile | 4096 | `compute_90` | 70464 | 79788 |" in markdown
    assert (
        "| `3,1,2,1` | `count=0,code=0,task=0` | "
        "`sched=1,workers=2,wp=1,stream=1,block=256,grid=3` | "
        "`scalar0=1.5` | `c=tmp0` |" in markdown
    )
    assert (
        "| `3,1,2,1` | `count=1,code=7,task=3` | "
        "`sched=1,workers=2,wp=1,stream=1,block=256,grid=3` | "
        "`scalar0=1.5` | `c=tmp0` |" in markdown
    )
    assert "nvcc-persistent-generated-dispatch-compute_90" in markdown
    assert "<svg" in svg
    assert "tensor-smoke" in svg
    assert "h200" in svg
    assert "errors: count=1,code=7,task=3" in svg
    assert "policy: sched=1,workers=2,wp=1,stream=1,block=256,grid=3" in svg
    assert "scalars: scalar0=1.5" in svg
    assert "tensors: c=tmp0" in svg


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
    (artifact_dir / "cuda-benchmark-ratios.svg").write_text("<svg></svg>\n")

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
            "has_markdown": True,
            "has_svg": True,
            "has_ratio_svg": True,
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
        "| a100-graph | benchmark | a100-graph | hina | abc123 | 1 | 1024 |  |  |  |  |  |  |  | direct_driver_graph |"
    ) in report
    assert "ratio SVG" in report


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
            "resource_policies": ["sched=1,workers=2,wp=1,stream=1,block=256,grid=3"],
            "scalar_args": ["scalar0=1.5"],
            "tensor_args": ["c=tmp0"],
            "tensor_tiles": ["16x16x16"],
            "has_markdown": True,
            "has_svg": True,
            "has_ratio_svg": False,
        }
    ]
    assert "Smoke mode | Dispatch | Scheduler errors | Resource policy | Scalar args | Tensor args |" in report
    assert "| tensor-descriptor-smoke | smoke | tensor-smoke | combined | unknown | 2 |" in report
    assert "| 4096 | 16x16x16 | dag/tensor_tile | 3,1,2,1 |" in report
    assert "count=0,code=0,task=0, count=1,code=7,task=3 |" in report
    assert "sched=1,workers=2,wp=1,stream=1,block=256,grid=3 |" in report
    assert "scalar0=1.5 | c=tmp0 |" in report


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


def test_cuda_capture_validator_paired_current_requires_quad_baseline():
    cuda_validate_capture = _load_capture_validator_module()
    args = cuda_validate_capture.parse_args(["capture.json", "--preset", "paired-current"])

    cuda_validate_capture._apply_preset(args)

    assert "pto_persistent_dag_quad" in args.require_baseline
    assert args.expected_result_count == 666


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
    index = cuda_pair_benchmark.build_index_command(config)

    assert local[:2] == ["env", f"PYTHONPATH={Path.cwd()}:{Path.cwd() / 'python'}"]
    assert ".agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py" in local
    assert "--arch" in local
    assert "compute_80" in local
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
    assert index[-2:] == ["--root", str(tmp_path / "cuda-backend")]


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
    assert len(commands) == 6
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
    assert len(commands) == 5
    assert commands[0][0] == "env"
    assert commands[1][0] == "ssh"
    assert commands[2][0] == "scp"


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
    )

    local = cuda_pair_persistent_smoke.build_local_smoke_command(config, "abc123")
    remote = cuda_pair_persistent_smoke.build_remote_smoke_command(config, "abc123")
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
    assert "persistent-tensor_tile-8x4x12-smoke-abc123" in report


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

    assert "persistent-scalar_axpy-smoke-abc123" in str(local)
    assert "--dag-shape" in local
    assert "scalar_axpy" in local
    assert "--dag-shape scalar_axpy" in remote[-1]
    assert "persistent-scalar_axpy-smoke-abc123/h200.json" in remote[-1]


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

    assert "persistent-unary_square-smoke-abc123" in str(local)
    assert "--dag-shape" in local
    assert "unary_square" in local
    assert "--dag-shape unary_square" in remote[-1]
    assert "persistent-unary_square-smoke-abc123/h200.json" in remote[-1]


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

    assert "persistent-scalar_affine-smoke-abc123" in str(local)
    assert "--dag-shape" in local
    assert "scalar_affine" in local
    assert "--dag-shape scalar_affine" in remote[-1]
    assert "persistent-scalar_affine-smoke-abc123/h200.json" in remote[-1]


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

    assert "persistent-triad-smoke-abc123" in str(local)
    assert "--dag-shape" in local
    assert "triad" in local
    assert "--dag-shape triad" in remote[-1]
    assert "persistent-triad-smoke-abc123/h200.json" in remote[-1]


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

    assert "persistent-quad-smoke-abc123" in str(local)
    assert "--dag-shape" in local
    assert "quad" in local
    assert "--dag-shape quad" in remote[-1]
    assert "persistent-quad-smoke-abc123/h200.json" in remote[-1]


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
    assert len(commands) == 6
    sync = commands[1]
    assert sync[:3] == ["rsync", "-a", "--delete"]
    assert sync[-2:] == [f"{Path.cwd()}/", "h200-box:/remote/pto-cu/"]
    assert "git fetch" not in commands[2][-1]
    assert "git checkout" not in commands[2][-1]
    assert any("persistent-scratch_reuse-smoke-local123" in part for part in commands[0])
    assert "persistent-scratch_reuse-smoke-local123" in commands[2][-1]
    assert "persistent-scratch_reuse-smoke-local123" in commands[3][1]
    assert any("persistent-scratch_reuse-smoke-local123" in part for part in commands[4])


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
    assert [task["func_id"] for task in seen["task_sources"]] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert [task["task_name"] for task in seen["task_sources"]] == [
        "add_f32",
        "mul_f32",
        "matmul_tile_f32",
        "axpy_f32",
        "affine_f32",
        "triad_f32",
        "square_f32",
        "quad_f32",
    ]
    assert {task["body_style"] for task in seen["task_sources"]} == {"task_body"}
    assert all("PtoCudaPersistentDagTask" in task["context_definition"] for task in seen["task_sources"])
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
        ],
    }

    table = cuda_current_summary.render_unary_square_table(payload)

    assert "| GPU | N | Unary square ns |" in table
    assert "| A100 | 65536 | 2000 |" in table


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
    assert "| A100 | 65536 | 1.50x | 2.00x | 1.25x | 1.30x | 1.35x | 1.40x | 1.20x | 2.50x |" in dag_table


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
        "| GPU | N | Chain/DAG | Reuse/DAG | Scalar AXPY/DAG | Scalar Affine/DAG | Triad/DAG | "
        "Quad/DAG | Unary Square/DAG | Tensor/DAG |"
    ) in dag_table
    assert "| A100 | 65536 | 1.50x | 2.00x | 1.25x | - | - | - | - | 2.50x |" in dag_table


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
        },
        "results": [
            {"machine": "a100-local", "baseline": "pto_host_schedule", "n": 1024, "device_wall_ns": 1000},
            {"machine": "a100-local", "baseline": "direct_driver", "n": 1024, "device_wall_ns": 500},
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)
    svg = cuda_benchmark.render_svg(cuda_benchmark.summarize_results(payload))

    assert "CUDA Backend Microbenchmark Report" in report
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
    assert ("| a100-local | 4096 | pto_persistent_dag_unary_square | 3 | 1200 | 1.20x |") in report
    assert ("| a100-local | 4096 | pto_persistent_dag_tensor | 4 | 4200 | 4.20x |") in report


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
            "metadata": {"label": "a100", "git_commit": "abc123", "tensor_tile": {"rows": 8, "cols": 4, "inner": 12}},
            "results": [{"machine": "a100-local", "baseline": "direct_driver", "n": 1024, "device_wall_ns": 500}],
        },
        {
            "metadata": {"label": "h200", "git_commit": "abc123", "tensor_tile": {"rows": 8, "cols": 4, "inner": 12}},
            "results": [{"machine": "h200-remote", "baseline": "direct_driver", "n": 1024, "device_wall_ns": 300}],
        },
    ]

    merged = cuda_benchmark.merge_payloads(payloads, label="combined")

    assert merged["metadata"]["label"] == "combined"
    assert merged["metadata"]["source_labels"] == ["a100", "h200"]
    assert merged["metadata"]["git_commits"] == ["abc123"]
    assert merged["metadata"]["tensor_tile"] == {"rows": 8, "cols": 4, "inner": 12}
    assert len(merged["results"]) == 2


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
        ("direct_driver", 3, 1024, 128, "compute_80"),
        ("direct_driver_graph", 3, 1024, 128, "compute_80"),
    ]
    assert len(payload["results"]) == 5


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
        "direct_driver",
        "direct_driver_graph",
        "pto_persistent_device",
        "pto_persistent_queue",
        "pto_persistent_dag",
        "pto_persistent_dag_chain",
        "pto_persistent_dag_reuse",
        "pto_persistent_dag_scalar_axpy",
        "pto_persistent_dag_scalar_affine",
        "pto_persistent_dag_triad",
        "pto_persistent_dag_quad",
        "pto_persistent_dag_unary_square",
        "pto_persistent_dag_tensor",
    ]
    assert len(payload["results"]) == 16


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

    tensor_calls = [item for item in seen if item[0] == "pto_persistent_dag_tensor"]
    non_tensor_calls = [item for item in seen if item[0] != "pto_persistent_dag_tensor"]
    assert payload["metadata"]["tensor_tile"] == tensor_tile
    assert tensor_calls == [("pto_persistent_dag_tensor", tensor_tile)]
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
        ("direct_driver", 1),
        ("direct_driver_graph", 1),
        ("pto_persistent_device", 1),
        ("pto_persistent_queue", 1),
        ("pto_persistent_dag", 1),
        ("pto_persistent_dag_chain", 1),
        ("pto_persistent_dag_reuse", 1),
        ("pto_persistent_dag_scalar_axpy", 1),
        ("pto_persistent_dag_scalar_affine", 1),
        ("pto_persistent_dag_triad", 1),
        ("pto_persistent_dag_quad", 1),
        ("pto_persistent_dag_unary_square", 1),
        ("pto_persistent_dag_tensor", 1),
        ("pto_host_schedule_batch", 6),
        ("pto_persistent_device_batch", 6),
        ("pto_persistent_queue_batch", 6),
    ]
    assert payload["metadata"]["batch_tasks"] == 6
    assert len(payload["results"]) == 19


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
        },
        "results": [
            {"machine": "a100-local", "baseline": "pto_stream_serial", "n": 2, "device_wall_ns": 2000},
            {"machine": "a100-local", "baseline": "pto_stream_parallel", "n": 2, "device_wall_ns": 1200},
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)

    assert "pto_stream_serial" in report
    assert "| a100-local | pto_stream_parallel | 2 | 1 | 1 | 1 | 1200 | 1200 | 0.60x |" in report
    assert "`pto_stream_parallel` measures two independent PTO launches" in report
    assert "stream rows use `pto_stream_serial` as their reference" in report
