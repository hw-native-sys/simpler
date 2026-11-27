#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

CASE_DIR = (
    Path(__file__).resolve().parents[4] / "examples" / "a2a3" / "host_build_graph" / "qwen3_14b_serving_effective"
)
sys.path.insert(0, str(CASE_DIR))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _load_module("qwen3_14b_hbg_benchmark", CASE_DIR / "benchmark.py")
benchmark_dual = _load_module("qwen3_14b_hbg_benchmark_dual", CASE_DIR / "benchmark_dual.py")
trace_effective = _load_module("qwen3_14b_hbg_trace_effective", CASE_DIR / "trace_effective.py")
artifact_builder = _load_module("qwen3_14b_hbg_artifact_builder", CASE_DIR / "build_hbg_artifact.py")


def _slot() -> dict[str, torch.Tensor]:
    block_table = torch.full((16 * 32,), -1, dtype=torch.int32)
    block_table.view(16, 32)[:, :27] = torch.arange(432, dtype=torch.int32).view(16, 27)
    return {
        "seq_lens": torch.zeros(16, dtype=torch.int32),
        "slot_mapping": torch.zeros(16, dtype=torch.int32),
        "block_table": block_table,
        "sampled_ids_host": torch.zeros((16, 8), dtype=torch.int32),
    }


def test_single_update_slot_installs_new_page() -> None:
    slot = _slot()
    page_ids = torch.arange(432, 448, dtype=torch.int32)
    golden = {
        "seq_lens": torch.full((127, 16), 3457, dtype=torch.int32),
        "slot_mapping": torch.zeros((127, 16), dtype=torch.int32),
    }
    golden["slot_mapping"][118] = page_ids * 128

    benchmark._update_slot(slot, golden, 118, page_size=128, blocks_per_row=32)

    assert torch.equal(slot["block_table"].view(16, 32)[:, 27], page_ids)


def test_dual_slot_updates_do_not_share_block_table() -> None:
    slots = [_slot(), _slot()]
    page_ids = torch.arange(432, 448, dtype=torch.int32)
    golden = {
        "seq_lens": torch.full((2, 16), 3457, dtype=torch.int32),
        "slot_mapping": torch.stack((page_ids * 128, page_ids * 128)),
    }

    benchmark_dual._update_slot(slots[0], golden, 0, page_size=128, blocks_per_row=32)
    assert torch.all(slots[1]["block_table"].view(16, 32)[:, 27] == -1)
    benchmark_dual._update_slot(slots[1], golden, 1, page_size=128, blocks_per_row=32)
    assert torch.equal(slots[0]["block_table"].view(16, 32)[:, 27], page_ids)
    assert torch.equal(slots[1]["block_table"].view(16, 32)[:, 27], page_ids)


def test_update_slot_rejects_page_offset_mismatch() -> None:
    slot = _slot()
    golden = {
        "seq_lens": torch.full((1, 16), 3457, dtype=torch.int32),
        "slot_mapping": torch.ones((1, 16), dtype=torch.int32),
    }

    with pytest.raises(RuntimeError, match="offset mismatch"):
        benchmark._update_slot(slot, golden, 0, page_size=128, blocks_per_row=32)


def test_update_slot_rejects_logical_block_overflow() -> None:
    slot = _slot()
    golden = {
        "seq_lens": torch.full((1, 16), 4097, dtype=torch.int32),
        "slot_mapping": torch.full((1, 16), 4096 * 128, dtype=torch.int32),
    }

    with pytest.raises(RuntimeError, match="logical block"):
        benchmark._update_slot(slot, golden, 0, page_size=128, blocks_per_row=32)


def test_trace_summary_uses_steady_completion_intervals(tmp_path: Path) -> None:
    lines = []
    for inv, timestamp in ((1, 1_000_000_000), (2, 1_040_000_000)):
        common = f"[STRACE] v=1 pid=1 tid=1 inv={inv} hid=abc depth=2"
        lines.extend(
            [
                (
                    f"{common} name=chip.run ts={timestamp} dur=40000000 "
                    f"dispatch_id={inv} slot_id=0 generation={inv} prepare_only=0"
                ),
                (
                    f"{common} name=node.dispatch ts={timestamp} dur=1 "
                    f"dispatch_id={inv} slot_id=0 generation={inv} prepare_only=0"
                ),
                f"{common} name=node.graph_build ts={timestamp} dur=2000000",
                f"{common} name=chip.run.bind ts={timestamp} dur=1000000",
                f"{common} name=chip.run.runner_run ts={timestamp} dur=39000000",
                f"{common} name=chip.run.runner_run.device_wall ts={timestamp} dur=38000000",
                f"{common} name=chip.run.validate ts={timestamp} dur=1000000",
            ]
        )
    log = tmp_path / "strace.log"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = trace_effective.invocation_rows(trace_effective.parse_spans(log))
    summary = trace_effective.summarize_campaign(rows, warmup_runs=0, measured_runs=1, steps=2, steady_skip=1)

    assert summary["dispatch_contract"]["total"] == 2
    assert summary["official_metrics"]["rts_completion_interval_ms"]["pooled_steady_stats"]["mean"] == pytest.approx(
        40.0
    )
    assert summary["official_metrics"]["effective_ms"]["pooled_steady_stats"]["mean"] == pytest.approx(38.0)
    assert summary["official_metrics"]["graph_build_ms"]["pooled_steady_stats"]["mean"] == pytest.approx(2.0)
    assert summary["official_metrics"]["bind_ms"]["pooled_steady_stats"]["mean"] == pytest.approx(1.0)
    assert summary["effective_source"] == "chip.run.runner_run.device_wall"


def test_trace_summary_associates_child_process_spans_by_root_window(tmp_path: Path) -> None:
    lines = [
        "[STRACE] v=1 pid=10 tid=10 inv=1 hid=abc depth=0 name=chip.run ts=1000 dur=100 dispatch_id=1 run_id=1",
        "[STRACE] v=1 pid=10 tid=10 inv=1 hid=abc depth=0 name=node.graph_build ts=1000 dur=10 run_id=1",
        "[STRACE] v=1 pid=20 tid=20 inv=1 hid=abc depth=1 name=chip.run.bind ts=1020 dur=5 run_id=1",
        "[STRACE] v=1 pid=20 tid=20 inv=1 hid=abc depth=1 name=chip.run.runner_run ts=1030 dur=50",
        "[STRACE] v=1 pid=20 tid=20 inv=1 hid=abc depth=2 name=chip.run.runner_run.device_wall ts=0 dur=45 clk=dev",
        "[STRACE] v=1 pid=20 tid=20 inv=1 hid=abc depth=1 name=chip.run.validate ts=1080 dur=2",
        "[STRACE] v=1 pid=30 tid=30 inv=1 hid=def depth=0 name=node.graph_build ts=2000 dur=10",
        "[STRACE] v=1 pid=30 tid=30 inv=1 hid=def depth=1 name=chip.run.bind ts=2020 dur=5",
        "[STRACE] v=1 pid=30 tid=30 inv=1 hid=def depth=1 name=chip.run.runner_run ts=2030 dur=50",
        "[STRACE] v=1 pid=30 tid=30 inv=1 hid=def depth=2 name=chip.run.runner_run.device_wall ts=0 dur=45 clk=dev",
        "[STRACE] v=1 pid=30 tid=30 inv=1 hid=def depth=1 name=chip.run.validate ts=2080 dur=2",
        "[STRACE] v=1 pid=30 tid=30 inv=1 hid=def depth=0 name=chip.run ts=2000 dur=100 dispatch_id=2 run_id=2",
    ]
    log = tmp_path / "multi-process.log"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = trace_effective.invocation_rows(trace_effective.parse_spans(log))
    assert [row["dispatch_id"] for row in rows] == [1, 2]


def test_artifact_builder_accepts_supported_decode_abis() -> None:
    common = [f"input_{index}" for index in range(20)]
    outputs = ["out", "embed_weight", "sampled_ids_in", "sampled_ids", "next_hidden"]
    artifact_builder._validate_param_names([*common, *outputs])
    artifact_builder._validate_param_names([*common, *outputs, "sampled_ids_host"])


@pytest.mark.parametrize("steps", [1, 3])
def test_zero_skip_intervals_require_two_completions(steps: int) -> None:
    rows = [
        {
            "dispatch_id": index + 1,
            "root_completion_ns": 1_000_000_000 + index * 40_000_000,
            "effective_source": "chip.run.runner_run.device_wall",
            "graph_build_ms": 2.0,
            "bind_ms": 1.0,
            "runner_run_ms": 39.0,
            "effective_ms": 38.0,
            "device_wall_ms": 38.0,
            "validate_ms": 1.0,
            "chip_run_lifecycle_ms": 40.0,
        }
        for index in range(steps)
    ]
    summary = trace_effective.summarize_campaign(rows, warmup_runs=0, measured_runs=1, steps=steps, steady_skip=0)
    metric = summary["official_metrics"]["rts_completion_interval_ms"]
    assert metric["pooled_steady_stats"]["count"] == steps - 1
    if steps == 1:
        assert metric["pooled_steady_stats"]["mean"] is None
        assert metric["run_means"] == []
    else:
        assert metric["pooled_steady_stats"]["mean"] == pytest.approx(40.0)
    assert summary["official_metrics"]["runner_run_ms"]["pooled_steady_stats"]["count"] == steps


def test_artifact_builder_rejects_inconsistent_host_output() -> None:
    common = [f"input_{index}" for index in range(20)]
    outputs = ["out", "embed_weight", "sampled_ids_in", "sampled_ids", "next_hidden"]
    with pytest.raises(RuntimeError, match="sampled_ids_host"):
        artifact_builder._validate_param_names([*common[:-1], *outputs, "sampled_ids_host"])


def test_artifact_builder_normalizes_generated_tensor_type(tmp_path: Path) -> None:
    source = tmp_path / "kernel.cpp"
    source.write_text("TaskTensor input; const TaskTensor& output = input;\n", encoding="utf-8")

    assert artifact_builder._normalize_tensor_type(source) == 2
    assert source.read_text(encoding="utf-8") == "Tensor input; const Tensor& output = input;\n"


def test_artifact_builder_counts_emitted_definition_tasks() -> None:
    source = """
static void decode_layer_definition(const GraphTaskArgs& args) {
  rt_submit_aiv_task(1, params);
  for (int64_t i = 0; i < 2; ++i) {
    rt_submit_aic_task(2, params);
  }
}
"""
    assert artifact_builder._definition_task_count(source) == 3


@pytest.mark.parametrize("tamper", [None, "metadata", "cpp", "so", "bin"])
def test_artifact_preflight_checks_frozen_files(tmp_path: Path, tamper: str | None) -> None:
    child = tmp_path / "next_levels" / "decode_fwd"
    (child / "orchestration").mkdir(parents=True)
    (child / "cache").mkdir()
    paths = {
        "metadata": tmp_path / "distributed_meta.json",
        "cpp": child / "orchestration" / "decode_fwd.cpp",
        "so": child / "orchestration" / "decode_fwd.so",
        "bin": child / "cache" / "incore_0.bin",
    }
    for path in paths.values():
        path.write_text("frozen", encoding="utf-8")
    for index in range(1, 39):
        (child / "cache" / f"incore_{index}.bin").write_text("kernel", encoding="utf-8")
    callable_spec = tmp_path / "callable_spec.json"
    callable_spec.write_text("{}", encoding="utf-8")
    manifest = {
        "schema": "simpler-hbg-pure-artifact-v1",
        "runtime": "host_build_graph",
        "graph_definition_task_count_per_layer": 277,
        "distributed_meta_sha256": artifact_builder._sha256(paths["metadata"]),
        "callable_spec_sha256": artifact_builder._sha256(callable_spec),
        "orchestration_cpp_sha256": artifact_builder._sha256(paths["cpp"]),
        "orchestration_so_sha256": artifact_builder._sha256(paths["so"]),
        "source_incore_bins": artifact_builder._bin_manifest(child / "cache"),
    }
    (tmp_path / "hbg_artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if tamper:
        paths[tamper].write_text("changed", encoding="utf-8")
        with pytest.raises(ValueError, match="checksum"):
            artifact_builder.verify_hbg_artifact(tmp_path)
    else:
        assert artifact_builder.verify_hbg_artifact(tmp_path) == manifest
