#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from dataclasses import replace

import pytest

from .helpers import Case, CounterRunner, Scenario, l3_case

pytestmark = [
    pytest.mark.manual,  # Acceptance contracts awaiting resident async integration; run with --manual include.
    pytest.mark.platforms(["a5sim", "a5"]),
    pytest.mark.runtime("host_build_graph"),
    pytest.mark.device_count(1),
]


@l3_case
def test_graph_instances_keep_completion_and_payload_separate(run_case):
    run_case(Case(scenario=Scenario.GRAPH, instances=2, delay_ms=20))


@pytest.mark.parametrize("epochs", [(1, 2, 3), (1, 1, 1)])
@l3_case
def test_retained_window_across_runs(st_platform, st_device_ids, tmp_path, epochs):
    case = Case()
    with CounterRunner(st_platform, st_device_ids, tmp_path, case) as runner:
        for index, epoch in enumerate(epochs):
            runner.run(replace(case, epoch=epoch, expected=epoch), reset_window=index == 0)
