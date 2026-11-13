# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reject incorrect native Graph benchmark captures."""

import importlib.util
import json
from pathlib import Path

import pytest

from simpler_setup.tools.hbg.task_id import TaskId

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "dsv41_graph_audit", ROOT / "tests/fdwic/dsv41_scheduler/production/audit_graph.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def capture(tmp_path):
    bundle, run = tmp_path / "bundle", tmp_path / "run"
    bundle.mkdir()
    run.mkdir()
    workload = tmp_path / "tasks.tsv"
    workload.write_text("8 vector 2 2 0\n")
    case = {"graph_benchmark": {"mode": "graph", "repetitions": 2}, "source": {"executions": 2}}
    submission = {"mode": "graph", "repetitions": 2, "shells": [0, 1], "recordings": 1}
    records = [
        {
            "task_id": TaskId.make_sub_task(parent, 0).raw,
            "block_idx": block,
            "block_num": 2,
            "subtask_slot": 1,
            "executed": True,
        }
        for parent in range(2)
        for block in range(2)
    ]
    (bundle / "case.json").write_text(json.dumps(case))
    (run / "source-task-map.txt.graph.json").write_text(json.dumps(submission))
    (run / "deps.json").write_text(json.dumps({"edges": [{"pred": "0", "succ": "1"}]}))
    data = {"aicore_tasks": [None] * 4, "aicore_execution_identities": {"records": records}}
    (run / "chip_swimlane_records.json").write_text(json.dumps(data))
    return bundle, run, workload, data


def test_accepts_complete_reused_graph(capture):
    bundle, run, workload, _ = capture
    assert MODULE.audit(bundle, run, workload)["executions"] == 4


@pytest.mark.parametrize("fault", ["duplicate", "missing_block", "wrong_path", "wrong_instance", "skipped"])
def test_rejects_incorrect_execution(capture, fault):
    bundle, run, workload, data = capture
    rows = data["aicore_execution_identities"]["records"]
    if fault == "duplicate":
        rows[-1] = rows[-2].copy()
    elif fault == "missing_block":
        rows[-1]["block_idx"] = 2
    elif fault == "wrong_path":
        rows[0]["task_id"] = TaskId.make_global(0).raw
    elif fault == "wrong_instance":
        rows[0]["task_id"] = TaskId.make_sub_task(8, 0).raw
    else:
        rows[0]["executed"] = False
    (run / "chip_swimlane_records.json").write_text(json.dumps(data))
    with pytest.raises(ValueError):
        MODULE.audit(bundle, run, workload)


def test_rejects_missing_graph_handoff(capture):
    bundle, run, workload, _ = capture
    (run / "deps.json").write_text(json.dumps({"edges": []}))
    with pytest.raises(ValueError, match="serial Graph completion"):
        MODULE.audit(bundle, run, workload)
