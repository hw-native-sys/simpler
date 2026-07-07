# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for Resource phase failure summaries emitted by the root conftest."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from simpler_setup.parallel_scheduler import JobResult

_ROOT = Path(__file__).resolve().parents[3]


def _load_root_conftest():
    spec = importlib.util.spec_from_file_location("_root_conftest_resource_summary", _ROOT / "conftest.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_emit_resource_failure_summary_prints_nodeid_and_annotation(capsys):
    cf = _load_root_conftest()
    results = [
        JobResult(
            label="standalone pass",
            returncode=0,
            device_ids=[0],
            output="pass output\n",
            duration_s=1.0,
        ),
        JobResult(
            label="standalone bad%case\nname",
            returncode=-11,
            device_ids=[4, 5],
            output=(
                "line1\n"
                "E       RuntimeError: run_prepared failed with code 507018\n"
                "PTO2 runtime failed: orch_error_code=0 sched_error_code=100 runtime_status=-100\n"
                "PTO2 scheduler timeout sub_class=S1:running-stalled\n"
            ),
            duration_s=12.34,
            nodeid="tests/st/runtime_fatal_codes/test_probe.py::test_bad[param]",
        ),
    ]

    cf._emit_resource_failure_summary(results)

    out = capsys.readouterr().out
    assert "*** Resource phase failed: 1 child job(s) ***" in out
    assert (
        "::error title=Resource phase failed::"
        "tests/st/runtime_fatal_codes/test_probe.py::test_bad[param] "
        "(standalone bad%25case%0Aname) rc=-11 devices=[4,5]"
    ) in out
    assert "- nodeid=tests/st/runtime_fatal_codes/test_probe.py::test_bad[param]" in out
    assert "label=standalone bad%case\nname" in out
    assert "rc=-11 devices=[4, 5] duration=12.3s" in out
    assert "full output is in the Resource child group above" in out
    assert "hint:" not in out
    assert "line1" not in out
    assert "RuntimeError: run_prepared failed with code 507018" not in out
    assert "PTO2 runtime failed: orch_error_code=0 sched_error_code=100 runtime_status=-100" not in out
    assert "PTO2 scheduler timeout sub_class=S1:running-stalled" not in out
    assert "standalone pass" not in out


def test_emit_resource_failure_summary_can_emit_compact_recap(capsys):
    cf = _load_root_conftest()
    results = [
        JobResult(
            label="standalone failed",
            returncode=2,
            device_ids=[7],
            output="hidden tail\n",
            duration_s=3.0,
            nodeid="tests/st/test_failed.py::test_failed",
        )
    ]

    cf._emit_resource_failure_summary(
        results,
        emit_annotations=False,
        heading="Resource phase failed recap",
    )

    out = capsys.readouterr().out
    assert "*** Resource phase failed recap: 1 child job(s) ***" in out
    assert "::error" not in out
    assert "nodeid=tests/st/test_failed.py::test_failed" in out
    assert "label=standalone failed" in out
    assert "rc=2 devices=[7] duration=3.0s" in out
    assert "full output is in the Resource child group above" in out
    assert "hidden tail" not in out
