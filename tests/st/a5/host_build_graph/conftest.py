#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check that marked A5 HBG runs publish a fresh calendar selection."""

import json
from pathlib import Path

import pytest

_CALENDAR_MODE = 5


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "scheduler_mode(kind): assert this test ran on the A5 HBG calendar",
    )


@pytest.fixture(autouse=True)
def _assert_scheduler_mode(request, st_platform):
    marker = request.node.get_closest_marker("scheduler_mode")
    if marker is None:
        yield
        return
    kind = marker.args[0] if marker.args else None
    if kind != "calendar":
        raise pytest.UsageError(f"scheduler_mode expects 'calendar', got {kind!r}")
    if not st_platform.startswith("a5") or "st_worker" not in request.fixturenames:
        yield
        return

    worker = request.getfixturevalue("st_worker")
    before = worker.scheduler_mode_publish_count
    yield

    assert worker.scheduler_mode_publish_count > before, "this run did not publish a fresh calendar selection"
    mode = worker.scheduler_runtime_mode
    assert mode == _CALENDAR_MODE, f"calendar expected, reported scheduler mode {mode}"


def _swimlane_records(output_prefix):
    matches = sorted(Path(output_prefix).rglob("chip_swimlane_records.json"))
    assert matches, f"no chip_swimlane_records.json under {output_prefix}"
    return json.loads(matches[-1].read_text())


@pytest.fixture
def assert_cohorts_were_coresident():
    """Decide co-residency from the run's own record, not from a timeout.

    A cohort is co-resident when every one of its subtasks held a core at the
    same instant, and the swimlane says that directly: the task's intervals all
    overlap exactly when its latest start precedes its earliest end.

    The alternative -- give the blocks a barrier and see whether it times out --
    measures a race against a wall clock, so the verdict moves with machine
    load -- the same build can report the defect when the case runs alone and
    report it fixed when it runs after the rest of the suite. This predicate
    cannot do that. Load shifts where the intervals sit, never whether they
    intersect.

    Cohort subtasks are identified as the tasks with more than one AICore
    record, which is what distinguishes them from the single-block work a case
    submits alongside. `expected_widths` is the number of records each cohort
    should have -- blocks times subtasks per block -- so a cohort that was
    never fully dispatched is reported as a missing task rather than silently
    passing on the blocks that did run.
    """

    def check(output_prefix, expected_widths, context=""):
        raw = _swimlane_records(output_prefix)
        rows = raw["aicore_tasks"]
        assert rows, f"{context}: no AICore task records, so the run produced no per-subtask trace"

        by_task = {}
        for row in rows:
            by_task.setdefault(int(row[1]), []).append((int(row[0]), int(row[3]), int(row[4])))
        cohorts = {token: lanes for token, lanes in by_task.items() if len(lanes) > 1}

        found = sorted(len(lanes) for lanes in cohorts.values())
        assert found == sorted(expected_widths), (
            f"{context}: expected cohorts of {sorted(expected_widths)} subtasks, the run recorded {found}. "
            f"A width that came up short means the cohort was never fully dispatched."
        )

        for token, lanes in sorted(cohorts.items()):
            latest_start = max(start for _, start, _ in lanes)
            earliest_end = min(end for _, _, end in lanes)
            if latest_start < earliest_end:
                continue
            origin = min(start for _, start, _ in lanes)
            spread = [
                (core, round((start - origin) / 1e3, 1), round((end - origin) / 1e3, 1))
                for core, start, end in sorted(lanes)
            ]
            raise AssertionError(
                f"{context}: task {token}'s {len(lanes)} subtasks were never all on cores at once -- its latest "
                f"start is {(latest_start - origin) / 1e3:.1f} us, after its earliest end at "
                f"{(earliest_end - origin) / 1e3:.1f} us. Sync-start promises the opposite, and a real Mix or "
                f"cohort kernel blocks on its siblings, so this is a stall or a hang in any graph that uses the "
                f"shape as intended.\n  (core, start_us, end_us) = {spread}"
            )

    return check
