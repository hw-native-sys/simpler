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

from .helpers import CONDITION_LIMIT, Case, CounterRunner, Scenario, error_log_cursor, failure_evidence, l3_case

pytestmark = [
    pytest.mark.platforms(["a5sim", "a5"]),
    pytest.mark.runtime("host_build_graph"),
    pytest.mark.device_count(1),
]


@pytest.mark.parametrize("count", [0, 1, CONDITION_LIMIT - 1, CONDITION_LIMIT])
@l3_case
def test_condition_count_boundaries(run_case, count):
    run_case(Case(conditions=count, delay_ms=10 if count > 1 else 0))


@pytest.mark.parametrize("initial,expected", [(0, 0), (0, 1), (1, 1), (2, 1), (2**31 - 2, 2**31 - 1)])
@l3_case
def test_ge_threshold(run_case, initial, expected):
    run_case(Case(initial_signal=initial, expected=expected))


@l3_case
def test_last_condition_controls_completion(run_case):
    run_case(Case(scenario=Scenario.TWO_CONDITIONS, conditions=2, delay_ms=20))


@l3_case
def test_capacity_overflow_is_not_success(st_platform, st_device_ids, tmp_path, capfd):
    case = Case(conditions=CONDITION_LIMIT + 1)
    with CounterRunner(st_platform, st_device_ids, tmp_path, case) as runner:
        runner.require_support()
        capfd.readouterr()
        cursor = error_log_cursor()
        with pytest.raises(RuntimeError) as failure:
            runner.submit().wait(timeout=30)
        captured = capfd.readouterr()
        evidence = failure_evidence(failure.value, captured, cursor)
        (tmp_path / "overflow.log").write_text(evidence)
        assert "ASYNC_WAIT_OVERFLOW" in evidence, evidence


@l3_case
def test_skipped_async_kernel_does_not_hold_successor(run_case):
    run_case(Case(conditions=0, skip=True))
