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

import importlib.util
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
    assert "| Machine | Baseline | N | Samples | Median device ns |" in report
    assert "a100-local" in report
    assert "<svg" in svg
    assert "direct_driver" in svg
