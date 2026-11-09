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


@l3_case
def test_run_entry_refreshes_changed_async_mapping(st_platform, st_device_ids, tmp_path):
    case = Case(waiters=16, mask=7, conditions=64)
    shapes = (
        (4, 4, 7, 1, False),
        (1, 1, 2, 0, False),
        (3, 4, 1, 64, False),
        (2, 4, 6, 4, False),
        (2, 1, 2, 0, True),
        (4, 4, 7, 4, False),
    )
    with CounterRunner(st_platform, st_device_ids, tmp_path, case) as runner:
        for index, (waiters, blocks, mask, conditions, skip) in enumerate(shapes * 2):
            # The fixed physical-slot allocation has enough logical-record rows
            # for the single-block shape as well as the four-block shape.
            runner.capacity_waiters = runner.slots // (blocks * 3)
            runner.run(
                replace(
                    case,
                    waiters=waiters,
                    blocks=blocks,
                    mask=mask,
                    conditions=conditions,
                    skip=skip,
                    epoch=index + 1,
                    expected=index + 1,
                )
            )
