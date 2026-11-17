# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Global supervision ends failed or unsatisfied asynchronous runs."""

import json

import pytest

from tests.st.a5.host_build_graph.async_completion.helpers import (
    CONDITION_LIMIT,
    Case,
    CounterRunner,
    Scenario,
    assert_observations,
    error_log_cursor,
    failure_evidence,
    l3_case,
)

# Automatic coverage for AICPU supervision of a returned kernel's pending completion.
# The missing signal is explicit; the timeout is the contract under test.
pytestmark = [
    pytest.mark.platforms(["a5", "a5sim"]),
    pytest.mark.runtime("percounter"),
    pytest.mark.device_count(1),
]


@pytest.mark.parametrize(
    "conditions,expected_error",
    [(1, "SCHEDULER_TIMEOUT"), (CONDITION_LIMIT + 1, "ASYNC_WAIT_OVERFLOW")],
    ids=["pending_timeout", "registration_error"],
)
@l3_case
def test_supervisor_bounds_returned_kernel_with_uncompleted_dependency(
    st_platform, st_device_ids, tmp_path, monkeypatch, capfd, conditions, expected_error
):
    device_logs = tmp_path / "ascend"
    device_logs.mkdir()
    monkeypatch.setenv("ASCEND_PROCESS_LOG_PATH", str(device_logs))
    monkeypatch.setenv("SIMPLER_SCHEDULER_TIMEOUT_MS", "1000")
    monkeypatch.setenv("SIMPLER_OP_EXECUTE_TIMEOUT_US", "5000000")
    monkeypatch.setenv("SIMPLER_STREAM_SYNC_TIMEOUT_MS", "7000")
    with CounterRunner(st_platform, st_device_ids, tmp_path, Case(), runtime="percounter", profiling_level=1) as runner:
        runner.submit().wait(timeout=20)
        assert_observations(runner.case, runner.capacity_waiters, runner.records, runner.output, runner.control)
        capfd.readouterr()
        cursor = error_log_cursor()
        with pytest.raises(RuntimeError) as failure:
            runner.submit(Case(scenario=Scenario.NEVER, conditions=conditions)).wait(timeout=20)
        captured = capfd.readouterr()
        evidence = failure_evidence(failure.value, captured, cursor)
        evidence += "\n".join(
            path.read_text(errors="replace") for path in sorted(device_logs.rglob("*.log")) if path.is_file()
        )
        (tmp_path / "pending_completion_timeout.log").write_text(evidence)
        assert expected_error in evidence, evidence
        runner.close_after_expected_error()
        captures = sorted(runner.capture.rglob("chip_swimlane_records.json"))
        assert len(captures) == 1, captures
        records = json.loads(captures[0].read_text())["aicore_tasks"]
        # NEVER records waiter task 0 and dependent consumer task 1 without a publisher.
        # Failed runs skip tensor copy-back; the returned-kernel trace remains observable.
        assert len(records) == 1, records
        assert records[0][1] == 0, records
        assert records[0][4] > records[0][3], records
