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

import os

import pytest

# Mirrors SCHEDULER_RUNTIME_MODE_* in
# src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_layout.h.
# Mode 4 named a shape the resident scheduler declined and is retired: SPMD and
# sync-start are its own now, and what it cannot run no other scheduler can.
_MODE_NAMES = {
    0: "none",
    1: "resident (pending)",
    2: "resident (ready)",
    3: "legacy (graph task)",
}
_RESIDENT_MODES = frozenset({1, 2})
_LEGACY_MODES = frozenset({3})
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
