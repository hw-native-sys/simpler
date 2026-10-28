#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Which scheduler every a5 host_build_graph test actually ran on.

The runtime falls back to the legacy AICPU scheduler silently — one LOG_INFO and
nothing a test can see. Every guard in this directory would stay green through
such a fallback, because legacy produces correct results; the resident AICore
scheduler simply would not have run. That makes result-only tests blind to the
one thing the per-core Scheduler work changes.

`Worker.scheduler_runtime_mode` closes that hole. This fixture reads it after
every test and fails when the selection is not the one the test declares, so a
change that quietly reroutes a test to the legacy scheduler is reported at that
test rather than discovered later.

Declaring the expectation:

    pytestmark = pytest.mark.scheduler_mode("legacy")        # whole module
    @pytest.mark.scheduler_mode("resident")                  # one class or test

Unmarked tests are not checked, so the coverage can be filled in gradually.
A marked test whose worker published no selection at all also fails: that means
either the test never ran a graph, or the hook is not wired on this build — both
are things a guard should report rather than pass over.
"""

import json
import os
from pathlib import Path

import pytest

# Mirrors SCHEDULER_RUNTIME_MODE_* in
# src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_layout.h.
_MODE_NAMES = {
    0: "none",
    1: "resident (pending)",
    2: "resident (ready)",
    3: "legacy (graph task)",
    4: "legacy (unsupported shape)",
}
_RESIDENT_MODES = frozenset({1, 2})
_LEGACY_MODES = frozenset({3, 4})
_EXPECTED = {"resident": _RESIDENT_MODES, "legacy": _LEGACY_MODES}


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "scheduler_mode(kind): assert this test ran on the 'resident' or 'legacy' a5 HBG scheduler",
    )


def _expected_kind(request):
    marker = request.node.get_closest_marker("scheduler_mode")
    if marker is None:
        return None
    kind = marker.args[0] if marker.args else None
    if kind not in _EXPECTED:
        raise pytest.UsageError(f"scheduler_mode expects 'resident' or 'legacy', got {kind!r}")
    return kind


@pytest.fixture(autouse=True)
def _assert_scheduler_mode(request, st_platform):
    kind = _expected_kind(request)
    # Discovery mode: report what every test selected instead of checking marks,
    # so the marks can be written from the measured distribution rather than
    # guessed. A wrong guess is a false failure, which is worse than no mark.
    report = os.environ.get("SIMPLER_REPORT_SCHEDULER_MODE") == "1"
    worker = None
    before = 0
    if (kind is not None or report) and st_platform.startswith("a5") and "st_worker" in request.fixturenames:
        worker = request.getfixturevalue("st_worker")
        before = worker.scheduler_mode_publish_count

    yield

    if worker is None:
        return

    published = worker.scheduler_mode_publish_count - before
    mode = worker.scheduler_runtime_mode
    name = _MODE_NAMES.get(mode, f"unknown ({mode})")

    if report:
        print(f"\n[SCHEDULER-MODE] {request.node.nodeid} mode={mode} ({name}) published={published}")
        if kind is None:
            return

    assert published > 0, (
        f"test declares scheduler_mode({kind!r}) but published no scheduler selection of its own "
        f"(latch reads {name}); either it bound no graph or this build does not export the hook"
    )
    assert mode in _EXPECTED[kind], (
        f"test declares scheduler_mode({kind!r}) but ran on {name}; "
        f"a silent fallback would leave a result-only guard green"
    )


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
