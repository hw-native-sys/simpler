# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Clock-log fixture scope must not include captures from earlier invocations."""

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def clock_guard(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "st/a5/host_build_graph/dfx/chip_swimlane/test_scheduler_phases.py"
    spec = importlib.util.spec_from_file_location("_hbg_clock_guard_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    scene = importlib.import_module("simpler_setup.scene_test")
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "finalized"

    monkeypatch.setattr(scene, "finalize_diagnostic_outputs", original)
    after_calls = []
    module._verify_runtime_clock_logs(monkeypatch, "a5", lambda *args: after_calls.append(args))
    return scene.finalize_diagnostic_outputs, calls, after_calls


def test_disabled_capture_never_walks_the_worktree(clock_guard, monkeypatch):
    finalize, calls, after_calls = clock_guard

    def unexpected_walk(*_args, **_kwargs):
        raise AssertionError("disabled profiling scanned unrelated captures")

    monkeypatch.setattr(Path, "rglob", unexpected_walk)
    assert finalize("case", Path("."), chip_swimlane=0) == "finalized"
    assert len(calls) == 1
    assert after_calls == [("case", Path("."))]


def test_enabled_capture_still_rejects_premature_alignment(clock_guard, tmp_path):
    finalize, calls, _after_calls = clock_guard
    raw = {"chip_swimlane_level": 3, "metadata": {"orchestrator_source": "host", "clock_alignment": {}}}
    (tmp_path / "chip_swimlane_records.json").write_text(json.dumps(raw))
    (tmp_path / "host_clock_alignment.1.log").write_text("")
    with pytest.raises(AssertionError):
        finalize("case", tmp_path, chip_swimlane=3)
    assert not calls


def test_enabled_capture_still_requires_runtime_clock_log(clock_guard, tmp_path):
    finalize, calls, _after_calls = clock_guard
    raw = {"chip_swimlane_level": 3, "metadata": {"orchestrator_source": "host"}}
    (tmp_path / "chip_swimlane_records.json").write_text(json.dumps(raw))
    with pytest.raises(AssertionError, match="runtime must export timing logs"):
        finalize("case", tmp_path, chip_swimlane=3)
    assert not calls


@pytest.mark.parametrize("level", [1, 2])
def test_lower_capture_levels_still_forbid_alignment_logs(clock_guard, tmp_path, level):
    finalize, calls, _after_calls = clock_guard
    raw = {"chip_swimlane_level": level, "metadata": {"orchestrator_source": "host"}}
    (tmp_path / "chip_swimlane_records.json").write_text(json.dumps(raw))
    (tmp_path / "host_clock_alignment.1.log").write_text("")
    with pytest.raises(AssertionError, match="must not export Host alignment logs"):
        finalize("case", tmp_path, chip_swimlane=level)
    assert not calls
