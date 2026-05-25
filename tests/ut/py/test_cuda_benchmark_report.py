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
import sys
from pathlib import Path


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
    }
    a100_path = tmp_path / "a100.json"
    h200_path = tmp_path / "h200.json"
    a100_path.write_text(json.dumps(a100) + "\n")
    h200_path.write_text(json.dumps(h200) + "\n")

    payload = cuda_smoke_report.load_smoke_payloads([a100_path, h200_path])
    markdown = cuda_smoke_report.render_markdown_report(payload, label="tensor-smoke")
    svg = cuda_smoke_report.render_svg_report(payload, label="tensor-smoke")

    assert (
        "| a100 | pass | persistent_device | dag/tensor_tile | 4096 | `compute_80` | 102400 | 122260 | 16x16x16 | 16 |"
    ) in markdown
    assert (
        "| h200 | pass | persistent_device | dag/tensor_tile | 4096 | `compute_90` | 70464 | 79788 | 16x16x16 | 16 |"
    ) in markdown
    assert "nvcc-persistent-generated-dispatch-compute_90" in markdown
    assert "<svg" in svg
    assert "tensor-smoke" in svg
    assert "h200" in svg


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
    assert ("| a100-graph | benchmark | a100-graph | hina | abc123 | 1 | 1024 |  | direct_driver_graph |") in report
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
        "tensor_tile": {
            "rows": 16,
            "cols": 16,
            "inner": 16,
            "tile_count": 16,
        },
    }
    (artifact_dir / "a100.json").write_text(json.dumps(smoke_payload) + "\n")
    (artifact_dir / "h200.json").write_text(json.dumps(smoke_payload) + "\n")
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
            "sizes": ["unknown"],
            "tensor_tiles": ["16x16x16"],
            "has_markdown": True,
            "has_svg": True,
            "has_ratio_svg": False,
        }
    ]
    assert "| tensor-descriptor-smoke | smoke | tensor-smoke | combined | unknown | 2 |" in report


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
    assert launch.args.worker_blocks_per_task == 4


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
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)

    assert "## DAG Shape Rows" in report
    assert ("| Machine | N | Baseline | Tasks | Median device ns | Device vs pto_persistent_dag |") in report
    assert ("| a100-local | 4096 | pto_persistent_dag_chain | 5 | 1800 | 1.80x |") in report
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
        ("direct_driver", 3, 1024, 128, "compute_80"),
        ("direct_driver_graph", 3, 1024, 128, "compute_80"),
    ]
    assert len(payload["results"]) == 4


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
        "direct_driver",
        "direct_driver_graph",
        "pto_persistent_device",
        "pto_persistent_queue",
        "pto_persistent_dag",
        "pto_persistent_dag_chain",
        "pto_persistent_dag_reuse",
        "pto_persistent_dag_tensor",
    ]
    assert len(payload["results"]) == 10


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
        ("direct_driver", 1),
        ("direct_driver_graph", 1),
        ("pto_persistent_device", 1),
        ("pto_persistent_queue", 1),
        ("pto_persistent_dag", 1),
        ("pto_persistent_dag_chain", 1),
        ("pto_persistent_dag_reuse", 1),
        ("pto_persistent_dag_tensor", 1),
        ("pto_host_schedule_batch", 6),
        ("pto_persistent_device_batch", 6),
        ("pto_persistent_queue_batch", 6),
    ]
    assert payload["metadata"]["batch_tasks"] == 6
    assert len(payload["results"]) == 13


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
