# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import json

import pytest

from simpler_setup.tools.l0_swimlane_converter import convert


def test_converts_structured_intervals_without_raw_binary(tmp_path):
    records = {
        "format": "simpler_a5_l0_events_v1",
        "run_epoch": 42,
        "physical_core_id": 107,
        "system_counter_frequency_hz": 1_000_000_000,
        "core_frequency_hz": 2_000_000_000,
        "busy_interval_columns": ["anchor_syscnt", "start_cycles", "end_cycles", "lane", "block_id", "clip_flags"],
        "busy_intervals": [[1000, 5, 15, 1, 7, 0]],
    }
    (tmp_path / "core_107.events.json").write_text(json.dumps(records), encoding="utf-8")
    output = tmp_path / "trace.json"

    assert convert(tmp_path, output) == 1
    trace = json.loads(output.read_text(encoding="utf-8"))
    assert trace["metadata"]["run_epoch"] == 42
    assert trace["traceEvents"][0] == {
        "ph": "X",
        "cat": "a5.l0",
        "name": "L0 VECTOR",
        "pid": 100107,
        "tid": 1,
        "ts": 1.0025,
        "dur": 0.005,
        "args": {"physical_core_id": 107, "block_id": 7, "clip_flags": 0},
    }


def test_rejects_missing_core_frequency_without_mislabeling_cycles(tmp_path):
    records = {
        "format": "simpler_a5_l0_events_v1",
        "run_epoch": 42,
        "physical_core_id": 107,
        "system_counter_frequency_hz": 1_000_000_000,
        "core_frequency_hz": 0,
        "core_frequency_source": "unavailable",
        "busy_interval_columns": ["anchor_syscnt", "start_cycles", "end_cycles", "lane", "block_id", "clip_flags"],
        "busy_intervals": [[1000, 5, 15, 1, 7, 0]],
    }
    (tmp_path / "core_107.events.json").write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot convert core cycles to time"):
        convert(tmp_path, tmp_path / "trace.json")
    assert not (tmp_path / "trace.json").exists()
