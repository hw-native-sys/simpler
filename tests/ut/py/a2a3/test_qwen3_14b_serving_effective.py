#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

CASE_DIR = (
    Path(__file__).resolve().parents[4]
    / "examples"
    / "a2a3"
    / "tensormap_and_ringbuffer"
    / "qwen3_14b_serving_effective"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _load_module("qwen3_14b_tmr_benchmark", CASE_DIR / "benchmark.py")
benchmark_dual = _load_module("qwen3_14b_tmr_benchmark_dual", CASE_DIR / "benchmark_dual.py")
trace_effective = _load_module("qwen3_14b_tmr_trace_effective", CASE_DIR / "trace_effective.py")


def _slot() -> dict[str, torch.Tensor]:
    block_table = torch.full((16 * 32,), -1, dtype=torch.int32)
    block_table.view(16, 32)[:, :27] = torch.arange(432, dtype=torch.int32).view(16, 27)
    return {
        "seq_lens": torch.zeros(16, dtype=torch.int32),
        "slot_mapping": torch.zeros(16, dtype=torch.int32),
        "block_table": block_table,
        "initial_block_table": block_table.clone(),
        "page_size": torch.tensor(128),
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

    benchmark._update_slot(slot, golden, 118)

    assert torch.equal(slot["block_table"].view(16, 32)[:, 27], page_ids)


def test_dual_slot_updates_do_not_share_block_table() -> None:
    slots = [_slot(), _slot()]
    page_ids = torch.arange(432, 448, dtype=torch.int32)
    golden = {
        "seq_lens": torch.full((2, 16), 3457, dtype=torch.int32),
        "slot_mapping": torch.stack((page_ids * 128, page_ids * 128)),
    }

    benchmark_dual._update_slot(slots[0], golden, 0)
    assert torch.all(slots[1]["block_table"].view(16, 32)[:, 27] == -1)
    benchmark_dual._update_slot(slots[1], golden, 1)
    assert torch.equal(slots[0]["block_table"].view(16, 32)[:, 27], page_ids)
    assert torch.equal(slots[1]["block_table"].view(16, 32)[:, 27], page_ids)


def test_update_slot_rejects_page_offset_mismatch() -> None:
    slot = _slot()
    golden = {
        "seq_lens": torch.full((1, 16), 3457, dtype=torch.int32),
        "slot_mapping": torch.ones((1, 16), dtype=torch.int32),
    }

    with pytest.raises(RuntimeError, match="offset mismatch"):
        benchmark._update_slot(slot, golden, 0)


def test_update_slot_uses_fixture_page_size_and_tensor_stride() -> None:
    block_table = torch.full((16 * 4,), -1, dtype=torch.int32)
    slot = {
        "seq_lens": torch.zeros(16, dtype=torch.int32),
        "slot_mapping": torch.zeros(16, dtype=torch.int32),
        "block_table": block_table,
        "page_size": torch.tensor(4),
        "sampled_ids_host": torch.zeros((16, 8), dtype=torch.int32),
    }
    page_ids = torch.arange(16, dtype=torch.int32)
    golden = {
        "seq_lens": torch.full((1, 16), 9, dtype=torch.int32),
        "slot_mapping": (page_ids * 4).unsqueeze(0),
    }

    benchmark._update_slot(slot, golden, 0)

    assert torch.equal(slot["block_table"].view(16, 4)[:, 2], page_ids)


def test_dual_host_sample_is_read_before_slot_reuse() -> None:
    class Handle:
        def __init__(self, target: torch.Tensor, token: int) -> None:
            self.target = target
            self.token = token

        def result(self) -> None:
            self.target[:, 0] = self.token

    class Runtime:
        def __init__(self) -> None:
            self.submissions = 0

        def copy_to(self, *_args) -> None:
            pass

        def submit(self, _compiled, sampled_ids_host, *, config):
            del config
            handle = Handle(sampled_ids_host, 10 + self.submissions)
            self.submissions += 1
            return handle

    steps = 3
    golden = {
        "decode_input_token_ids": torch.zeros((steps, 16), dtype=torch.int32),
        "decode_output_token_ids": torch.stack(
            [torch.full((16,), 10 + step, dtype=torch.int32) for step in range(steps)]
        ),
        "seq_lens": torch.ones((steps, 16), dtype=torch.int32),
        "slot_mapping": torch.zeros((steps, 16), dtype=torch.int32),
    }
    resident = {
        "weights": {},
        "initial_tokens": SimpleNamespace(data_ptr=0),
        "sampled": [None] * steps,
        "rope_cos": None,
        "rope_sin": None,
        "k_cache": None,
        "v_cache": None,
        "out": [None, None],
        "next_hidden": [None, None],
    }

    result = benchmark_dual._run_sequence(
        rt=Runtime(),
        compiled=None,
        run_config=None,
        param_names=["sampled_ids_host"],
        resident=resident,
        slots=[_slot(), _slot()],
        golden=golden,
        mode="dual",
        steps=steps,
        sampled_ids_host_abi=True,
    )

    assert result["token_rows"] == golden["decode_output_token_ids"].tolist()


def test_trace_summary_uses_steady_completion_intervals(tmp_path: Path) -> None:
    lines = []
    for inv, timestamp in ((1, 1_000_000_000), (2, 1_040_000_000)):
        common = f"[STRACE] v=1 pid=1 tid=1 inv={inv} hid=abc depth=2"
        lines.extend(
            [
                f"{common} name=chip.run ts={timestamp} dur=40000000 "
                f"dispatch_id={inv} slot_id=0 generation={inv} prepare_only=0",
                f"{common} name=node.dispatch ts={timestamp} dur=1 "
                f"dispatch_id={inv} slot_id=0 generation={inv} prepare_only=0",
                f"{common} name=chip.run.runner_run ts={timestamp} dur=39000000",
                f"{common} name=chip.run.runner_run.device_wall ts={timestamp} dur=38000000",
                f"{common} name=chip.run.runner_run.device_wall.sched ts={timestamp} dur=37000000",
                f"{common} name=chip.run.bind.args ts={timestamp} dur=1000000",
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
    assert summary["official_metrics"]["effective_ms"]["pooled_steady_stats"]["mean"] == pytest.approx(37.0)


def test_trace_summary_does_not_create_zero_interval_at_skip_zero(tmp_path: Path) -> None:
    lines = []
    for inv, timestamp in ((1, 1_000_000_000), (2, 1_040_000_000)):
        common = f"[STRACE] v=1 pid=1 tid=1 inv={inv} hid=abc depth=2"
        lines.extend(
            [
                f"{common} name=chip.run ts={timestamp} dur=40000000 dispatch_id={inv}",
                f"{common} name=chip.run.runner_run ts={timestamp} dur=39000000",
                f"{common} name=chip.run.runner_run.device_wall ts={timestamp} dur=38000000",
                f"{common} name=chip.run.runner_run.device_wall.sched ts={timestamp} dur=37000000",
                f"{common} name=chip.run.bind.args ts={timestamp} dur=1000000",
                f"{common} name=chip.run.validate ts={timestamp} dur=1000000",
            ]
        )
    log = tmp_path / "strace-zero-skip.log"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = trace_effective.invocation_rows(trace_effective.parse_spans(log))
    summary = trace_effective.summarize_campaign(rows, warmup_runs=0, measured_runs=1, steps=2, steady_skip=0)
    intervals = summary["official_metrics"]["rts_completion_interval_ms"]["pooled_steady_stats"]

    assert intervals["count"] == 1
    assert intervals["mean"] == pytest.approx(40.0)

    one_step = trace_effective.summarize_campaign(rows[:1], warmup_runs=0, measured_runs=1, steps=1, steady_skip=0)
    assert "rts_completion_interval_ms" not in one_step["official_metrics"]
