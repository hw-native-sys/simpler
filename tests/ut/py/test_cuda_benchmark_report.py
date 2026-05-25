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
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(script_dir))


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

    assert summary[("a100-local", "pto_host_schedule", 1024)]["median_device_wall_ns"] == 1500
    assert summary[("a100-local", "direct_driver", 1024)]["samples"] == 1
    assert summary[("h200-remote", "pto_host_schedule", 2048)]["median_device_wall_ns"] == 800


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
        "| Machine | Baseline | N | Tasks | Samples | Median device ns | Median host ns | "
        "Device vs matched host_schedule |"
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
            {"machine": "a100-local", "baseline": "pto_persistent_dag", "n": 1024, "device_wall_ns": 2500},
        ],
    }

    report = cuda_benchmark.render_markdown_report(payload)

    assert "| a100-local | pto_host_schedule | 1024 | 1 | 1 | 1000 | 1000 | 1.00x |" in report
    assert "| a100-local | direct_driver | 1024 | 1 | 1 | 500 | 500 | 0.50x |" in report
    assert "| a100-local | pto_persistent_dag | 1024 | 1 | 1 | 2500 | 2500 | 2.50x |" in report
    assert "relative to the matched `pto_host_schedule` row" in report
    assert "for the same machine, `N`, and task count" in report


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

    assert "| a100-local | pto_host_schedule_batch | 1024 | 6 | 1 | 6000 | 6000 | 1.00x |" in report
    assert "| a100-local | pto_persistent_queue_batch | 1024 | 6 | 1 | 3000 | 3000 | 0.50x |" in report
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

    assert "| a100-local | pto_persistent_device_grid_batch | 4096 | 6 | 1 | 3000 | 3000 | 0.50x |" in report
    assert "`pto_persistent_device_grid_batch` assigns multiple worker blocks per task descriptor" in report
    assert "pto_persistent_device_grid_batch" in svg


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
            "metadata": {"label": "a100", "git_commit": "abc123"},
            "results": [{"machine": "a100-local", "baseline": "direct_driver", "n": 1024, "device_wall_ns": 500}],
        },
        {
            "metadata": {"label": "h200", "git_commit": "abc123"},
            "results": [{"machine": "h200-remote", "baseline": "direct_driver", "n": 1024, "device_wall_ns": 300}],
        },
    ]

    merged = cuda_benchmark.merge_payloads(payloads, label="combined")

    assert merged["metadata"]["label"] == "combined"
    assert merged["metadata"]["source_labels"] == ["a100", "h200"]
    assert merged["metadata"]["git_commits"] == ["abc123"]
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
        ("direct_driver", 3, 1024, 128, "compute_80"),
    ]
    assert len(payload["results"]) == 2


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
        "direct_driver",
        "pto_persistent_device",
        "pto_persistent_queue",
        "pto_persistent_dag",
    ]
    assert len(payload["results"]) == 5


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
        ("direct_driver", 1),
        ("pto_persistent_device", 1),
        ("pto_persistent_queue", 1),
        ("pto_persistent_dag", 1),
        ("pto_host_schedule_batch", 6),
        ("pto_persistent_device_batch", 6),
        ("pto_persistent_queue_batch", 6),
    ]
    assert payload["metadata"]["batch_tasks"] == 6
    assert len(payload["results"]) == 8


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
    assert "`pto_stream_parallel` measures two independent PTO launches" in report
