# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise the scheduler smoke's capture checks without running a device."""

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import pytest
from _capture_builder import Capture
from _task_ids import hbg_global

from simpler_setup.tools._runtime_dispatch import HBG_RUNTIME


@pytest.fixture
def validate_capture(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[2] / "st/a5/host_build_graph/dfx/chip_swimlane/test_scheduler_phases.py"
    spec = importlib.util.spec_from_file_location("_hbg_scheduler_phase_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_outputs_dir", lambda: tmp_path)
    case = module.TestSchedulerPhases()
    monkeypatch.setattr(case, "_effective_enable_chip_swimlane", lambda request: 3)
    monkeypatch.setattr(case, "_matching_cases", lambda *args: [{"name": "resolve_dummy"}])
    invocations = itertools.count()

    def validate(raw):
        def run(*args):
            output = tmp_path / f"TestSchedulerPhases_resolve_dummy_{next(invocations)}"
            output.mkdir()
            (output / "chip_swimlane_records.json").write_text(json.dumps(raw))

        monkeypatch.setattr(module.SceneTestCase, "test_run", run)
        case.test_run("a5sim", None, None, monkeypatch)

    return validate


def _capture(inbox_kinds):
    capture = Capture(runtime=HBG_RUNTIME, level=3, scheduler_producer="aicore")
    capture.aicore_sched_phase(phase="bootstrap", start=1, end=2)
    capture.aicpu_lifecycle(
        thread_id=0,
        config_start_cycles=1,
        topology_complete_cycles=2,
        **{
            field: index
            for index, field in enumerate(
                (
                    "handshake_start_cycles",
                    "handshake_complete_cycles",
                    "context_publish_start_cycles",
                    "context_publish_complete_cycles",
                    "bootstrap_wait_start_cycles",
                    "bootstrap_complete_cycles",
                    "register_release_start_cycles",
                    "register_release_end_cycles",
                    "exit_signal_start_cycles",
                    "exit_signal_end_cycles",
                    "exit_wait_start_cycles",
                    "exit_wait_end_cycles",
                ),
                start=3,
            )
        },
    )
    for index in range(34):
        task_id = hbg_global(index)
        start = 100 + index * 20
        capture.task(task_id=task_id, start=start + 4, end=start + 5)
        if index < 33:
            capture.aicore_sched_phase(phase="state_probe", task_id=task_id, start=start, end=start + 1)
            kind = inbox_kinds[index % len(inbox_kinds)]
        else:
            kind = "refill"
        capture.aicore_sched_phase(phase=kind, task_id=task_id, start=start + 2, end=start + 3)
        capture.aicore_sched_phase(phase="complete", task_id=task_id, start=start + 6, end=start + 7)
        capture.aicore_sched_phase(phase="resolve", task_id=task_id, start=start + 8, end=start + 9)
    capture.aicore_sched_phase(phase="idle", start=800, end=801)
    return capture.build()


@pytest.mark.parametrize("inbox_kinds", [("dispatch",), ("worksteal",), ("dispatch", "worksteal")])
def test_smoke_accepts_local_stolen_and_mixed_inbox_launches(validate_capture, inbox_kinds):
    validate_capture(_capture(inbox_kinds))


@pytest.mark.parametrize(
    "corruption",
    ("missing_launch", "duplicate_launch", "unknown_task", "missing_probe", "overflow", "producer", "timing"),
)
def test_smoke_rejects_incomplete_or_inconsistent_launch_evidence(validate_capture, corruption):
    raw = _capture(("dispatch", "worksteal"))
    stream = raw["scheduler_records"]["streams"][0]
    records = stream["records"]
    task_id = hbg_global(11)
    launch = next(record for record in records if record["task_id"] == task_id and record["kind"] == "worksteal")
    if corruption == "missing_launch":
        records.remove(launch)
    elif corruption == "duplicate_launch":
        records.append(dict(launch))
    elif corruption == "unknown_task":
        launch["task_id"] = hbg_global(34)
    elif corruption == "missing_probe":
        records[:] = [
            record for record in records if not (record["task_id"] == task_id and record["kind"] == "state_probe")
        ]
    elif corruption == "overflow":
        stream["capture"]["dropped"] = 1
    elif corruption == "producer":
        stream["producer"] = "aicpu"
    else:
        launch["start_cycles"] -= 2
    with pytest.raises(AssertionError):
        validate_capture(raw)
