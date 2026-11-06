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

from .helpers import Case, Scenario, l3_case

pytestmark = [
    pytest.mark.platforms(["a5sim", "a5"]),
    pytest.mark.runtime("host_build_graph"),
    pytest.mark.device_count(1),
]


@pytest.mark.parametrize("delay,tail", [(30, 0), (0, 30)])
@l3_case
def test_both_fanin_dependencies_are_required(run_case, delay, tail):
    run_case(Case(scenario=Scenario.FANIN, delay_ms=delay, tail_ms=tail))


@pytest.mark.parametrize("mask", [1, 2, 3, 5, 6, 7])
@pytest.mark.parametrize("delay,tail", [(20, 0), (0, 20)])
@l3_case
def test_last_block_and_condition_orders(run_case, mask, delay, tail):
    run_case(Case(blocks=2, mask=mask, delay_ms=delay, tail_ms=tail))


@l3_case
def test_sync_start_releases_execution_before_waiting(run_case):
    run_case(Case(blocks=2, mask=7, sync_start=True, delay_ms=20))
