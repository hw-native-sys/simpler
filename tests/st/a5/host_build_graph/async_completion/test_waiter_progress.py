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


@pytest.mark.parametrize("delta", [-1, 0, 1])
@l3_case
def test_waiters_leave_capacity_for_the_publisher(run_case, delta):
    run_case(Case(scenario=Scenario.PROGRESS, capacity_delta=delta))


@l3_case
def test_sparse_wait_completes_alongside_ordinary_ready_chains(run_case):
    run_case(Case(scenario=Scenario.SPARSE_READY, conditions=1, delay_ms=1))
