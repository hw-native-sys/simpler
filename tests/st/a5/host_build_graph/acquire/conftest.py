#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Each invocation must publish a fresh calendar publication."""

import pytest

from tests.st.a5.host_build_graph.conftest import assert_cohorts_were_coresident as _cohort_check

assert_cohorts_were_coresident = _cohort_check


def pytest_configure(config):
    config.addinivalue_line("markers", "scheduler_mode(kind): scheduler selection contract on reused workloads")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call":
        item._calendar_call_report = report


@pytest.fixture(autouse=True)
def assert_calendar_runtime(request):
    if "st_worker" not in request.fixturenames:
        yield
        return
    worker = request.getfixturevalue("st_worker")
    before = worker.scheduler_mode_publish_count
    yield
    report = getattr(request.node, "_calendar_call_report", None)
    if report is not None and report.skipped:
        return
    assert worker.scheduler_mode_publish_count > before
    assert worker.scheduler_runtime_mode == 5
