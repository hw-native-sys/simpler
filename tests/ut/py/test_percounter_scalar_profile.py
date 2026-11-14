# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Percounter nested execution intervals retain exact accounting and runtime identity."""

import json

import pytest
from _capture_builder import Capture

from simpler_setup.tools.swimlane_converter import generate_chrome_trace_json, read_perf_data


@pytest.mark.parametrize("level", [3, 4])
def test_nested_kernel_in_mix_service_exports_without_double_counting(tmp_path, level):
    capture = Capture(runtime="percounter", level=level, scheduler_producer="aicore")
    capture.task(task_id=7, core_id=0, start=40, end=60, run_epoch=0)
    for kind, begin, end, depth in [
        ("scalar_window", 10, 100, 0),
        ("scalar_percounter_serve_mix", 20, 80, 1),
        ("scalar_percounter_run_prepared", 30, 70, 2),
        ("scalar_bypass_load", 32, 35, 3),
    ]:
        metrics = {"scalar_depth": depth, "scalar_calls": 1}
        if depth == 0:
            metrics.update(scalar_schema=2, function_min_cycles=1 if level == 3 else 0)
        capture.aicore_sched_phase(phase=kind, start=begin, end=end, metrics=metrics)
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(capture.build()))
    data = read_perf_data(raw)
    profile = data["scalar_profiles"][0]
    assert profile["complete"]
    assert profile["kernel_cycles"] == 20
    assert profile["exclusive_cycles"]["scalar_percounter_run_prepared"] == 17
    assert profile["exclusive_cycles"]["scalar_percounter_serve_mix"] == 20
    assert sum(profile["exclusive_cycles"].values()) == 90
    trace = tmp_path / "trace.json"
    generate_chrome_trace_json(
        data["tasks"],
        str(trace),
        runtime_name="percounter",
        scheduler_phases=data["scheduler_records"],
        scheduler_streams=data["scheduler_streams"],
        scalar_profiles=data["scalar_profiles"],
        scheduler_task_producer="aicore",
        chip_swimlane_level=level,
    )
    output = json.loads(trace.read_text())
    assert output["metadata"]["runtime"] == "percounter"
    names = {e["name"] for e in output["traceEvents"] if e["ph"] == "X" and e["pid"] == 2}
    assert {"percounter_serve_mix", "percounter_run_prepared", "bypass_load"} <= names
    assert not any(name.startswith("scheduler_") for name in names)
