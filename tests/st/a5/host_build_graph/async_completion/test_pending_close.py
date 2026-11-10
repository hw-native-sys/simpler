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

from .helpers import Case, CounterRunner, Scenario, error_log_cursor, failure_evidence, l3_case

pytestmark = [
    pytest.mark.manual,  # Acceptance contracts awaiting resident async integration; run with --manual include.
    pytest.mark.platforms(["a5sim", "a5"]),
    pytest.mark.runtime("host_build_graph"),
    pytest.mark.device_count(1),
]


@l3_case
def test_close_drains_an_accepted_async_run(st_platform, st_device_ids, tmp_path):
    case = Case(delay_ms=200)
    with CounterRunner(st_platform, st_device_ids, tmp_path, case) as runner:
        runner.require_support()
        handle = runner.submit()
        assert not handle.done, "run already completed: pending-close boundary was not exercised"
        runner.close()
        assert handle.done
        handle.result()
        runner.check(check_guards=False)


@l3_case
def test_never_satisfied_condition_cannot_report_success(st_platform, st_device_ids, tmp_path, monkeypatch, capfd):
    monkeypatch.setenv("SIMPLER_SCHEDULER_TIMEOUT_MS", "1000")
    monkeypatch.setenv("SIMPLER_OP_EXECUTE_TIMEOUT_US", "3000000")
    monkeypatch.setenv("SIMPLER_STREAM_SYNC_TIMEOUT_MS", "5000")
    case = Case(scenario=Scenario.NEVER)
    with CounterRunner(st_platform, st_device_ids, tmp_path, case) as runner:
        runner.require_support()
        capfd.readouterr()
        cursor = error_log_cursor()
        with pytest.raises(RuntimeError) as failure:
            runner.submit().wait(timeout=15)
        captured = capfd.readouterr()
        evidence = failure_evidence(failure.value, captured, cursor)
        (tmp_path / "never_ready.log").write_text(evidence)
        assert "SCHEDULER_TIMEOUT" in evidence, evidence
        consumer = runner.slots
        assert int(runner.records[consumer, 0]) == 0, "consumer executed despite an unsatisfied condition"
        runner.close_after_expected_error()
