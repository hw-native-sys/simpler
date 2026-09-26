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
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

CASE = Path(__file__).resolve().parents[3] / "examples/a2a3/host_build_graph/qwen3_14b_serving_effective"
spec = importlib.util.spec_from_file_location("qwen_standalone_adapter", CASE / "standalone_adapter.py")
assert spec is not None and spec.loader is not None
adapter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = adapter
spec.loader.exec_module(adapter)


class Fixture:
    manifest = {"decode_dispatches_remaining": 2, "physical_layout": {"page_size": 4, "num_pages": 128}}
    metadata = {
        "first_generated_token_ids": torch.tensor([10, 11] + [12] * 14, dtype=torch.int32),
        "block_table": torch.full((16, 4), -1, dtype=torch.int32),
    }

    def load_golden(self):
        return {
            "decode_input_token_ids": torch.tensor([[10, 11] + [12] * 14, [20, 21] + [22] * 14], dtype=torch.int32),
            "decode_output_token_ids": torch.tensor([[20, 21] + [22] * 14, [30, 31] + [32] * 14], dtype=torch.int32),
            "seq_lens": torch.tensor([[5] * 16, [6] * 16], dtype=torch.int32),
            "slot_mapping": torch.tensor([[4] * 16, [5] * 16], dtype=torch.int32),
        }


def test_autoregressive_feedback_and_page_state():
    state = adapter.StandaloneDecodeAdapter(Fixture())
    first = state.next_step()
    assert first.input_token_ids.tolist()[:2] == [10, 11]
    assert first.block_table.view(16, 4)[:, 1].eq(1).all()
    state.complete_step(first.expected_output_token_ids)
    second = state.next_step()
    assert second.input_token_ids.tolist()[:2] == [20, 21]
    state.complete_step(second.expected_output_token_ids)
    assert state.finished


def test_step_cannot_be_reused_or_completed_before_submission():
    state = adapter.StandaloneDecodeAdapter(Fixture())
    with pytest.raises(RuntimeError, match="no decode step"):
        state.complete_step(torch.zeros(16, dtype=torch.int32))
    state.next_step()
    with pytest.raises(RuntimeError, match="previous decode step"):
        state.next_step()


def test_sampled_mismatch_is_rejected_and_stream_remains_inflight():
    state = adapter.StandaloneDecodeAdapter(Fixture())
    state.next_step()
    with pytest.raises(ValueError, match="sampled token mismatch"):
        state.complete_step(torch.zeros(16, dtype=torch.int32))
    with pytest.raises(RuntimeError, match="previous decode step"):
        state.next_step()


def test_submit_step_uses_worker_submit_and_worker_id():
    state = adapter.StandaloneDecodeAdapter(Fixture())
    events = []

    class Orchestrator:
        def submit_next_level(self, *args, **kwargs):
            events.append((args, kwargs))

    class Worker:
        def submit(self, callback, *, args, config):
            assert args is None
            callback(Orchestrator(), args, config)
            return SimpleNamespace(wait=lambda: None)

    step, handle = state.submit_step(Worker(), chip_handle="chip", task_args="args", config="cfg", worker_id=3)
    assert step.index == 0
    assert handle is not None
    assert events == [(("chip", "args", "cfg"), {"worker": 3})]


@pytest.mark.parametrize("columns", [1, 8])
def test_single_and_padded_sample_columns(columns):
    state = adapter.StandaloneDecodeAdapter(Fixture())
    step = state.next_step()
    values = step.expected_output_token_ids[:, None].expand(-1, columns).clone()
    state.complete_step(values)
    assert state.completed_steps == 1
