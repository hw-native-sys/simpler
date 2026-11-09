#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest

from simpler_setup.tools.swimlane_converter import read_perf_data

from .helpers import Case, CounterRunner, Scenario, l3_case

pytestmark = [
    pytest.mark.platforms(["a5sim", "a5"]),
    pytest.mark.runtime("host_build_graph"),
    pytest.mark.device_count(1),
]


@l3_case
def test_synchronous_consumer_acquires_updated_payload(run_case):
    run_case(Case(scenario=Scenario.VISIBILITY, delay_ms=20))


@l3_case
@pytest.mark.parametrize(
    "case",
    [
        Case(scenario=Scenario.VISIBILITY, delay_ms=20),
        Case(scenario=Scenario.GRAPH, instances=2, conditions=2, delay_ms=10),
    ],
    ids=["ordinary_visibility", "graph_instances"],
)
def test_execution_trace_keeps_deferred_completion(st_platform, st_device_ids, tmp_path, case):
    with CounterRunner(st_platform, st_device_ids, tmp_path, case, profiling_level=1) as runner:
        runner.run()
        captures = sorted(runner.capture.rglob("chip_swimlane_records.json"))
        assert captures
        for path in captures:
            parsed = read_perf_data(path)
            assert parsed["chip_swimlane_level"] == 1
