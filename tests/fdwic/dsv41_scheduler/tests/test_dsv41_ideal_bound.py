# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Dependency and resource constraints of the conditional ideal bound."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.manual

PRODUCTION = Path(__file__).resolve().parents[4] / "tests/fdwic/dsv41_scheduler/production"
SPEC = importlib.util.spec_from_file_location("ideal_bound", PRODUCTION / "ideal_bound.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(PRODUCTION))
try:
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path.pop(0)


def capture(rows, cores=None):
    return {
        "chip_swimlane_level": 1,
        "metadata": {"clock_freq_hz": 1000000000, "core_types": cores or ["aic", "aiv", "aiv"]},
        "aicore_tasks": rows,
    }


def dag(count, edges):
    return {
        "tasks": [{"task_id": str(i), "kernel_ids": [0, -1, -1]} for i in range(count)],
        "edges": [{"pred": str(a), "succ": str(b)} for a, b in edges],
    }


def test_dag_fork_join_uses_longest_path_not_total_kernel_time():
    records = capture([[0, 0, 0, 1, 101], [1, 1, 1, 101, 181], [2, 2, 2, 101, 121], [0, 3, 3, 181, 221]])
    deps = dag(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
    assert MODULE.ideal_lower_bound_ms(records, deps) == pytest.approx(0.00022)


def test_global_worker_capacity_bounds_independent_tasks():
    records = capture([[1, i, i, 1, 101] for i in range(4)])
    assert MODULE.ideal_lower_bound_ms(records, dag(4, [])) == pytest.approx(0.0002)


def test_task_capacity_bounds_oversubscribed_block_wave():
    records = capture([[1, 0, i, 1, 101] for i in range(4)] + [[0, 1, 4, 201, 251]])
    assert MODULE.ideal_lower_bound_ms(records, dag(2, [(0, 1)])) == pytest.approx(0.00025)


def test_model_rejects_a_cycle():
    records = capture([[0, 0, 0, 1, 101], [1, 1, 1, 1, 101]])
    with pytest.raises(ValueError, match="cycle"):
        MODULE.ideal_lower_bound_ms(records, dag(2, [(0, 1), (1, 0)]))


def test_model_rejects_missing_dependency_events():
    records = capture([[0, 0, 0, 1, 101]])
    with pytest.raises(ValueError, match="unobserved endpoint"):
        MODULE.ideal_lower_bound_ms(records, dag(1, [(0, 8)]))


def test_flat_model_adds_only_verified_host_allocation_nodes(tmp_path):
    mapping = tmp_path / "mapping.txt"
    mapping.write_text("10 0\n11 1\n12 2\n")
    deps = {
        "tasks": [{"task_id": "1", "kernel_ids": [0, -1, -1]}, {"task_id": "2", "kernel_ids": [0, -1, -1]}],
        "edges": [{"pred": "0", "succ": "1"}, {"pred": "1", "succ": "2"}],
    }
    completed = MODULE.complete_flat_dag(deps, mapping, {11, 12}, 1)
    assert {int(task["task_id"]) for task in completed["tasks"]} == {0, 1, 2}
    deps["edges"].append({"pred": "99", "succ": "2"})
    with pytest.raises(ValueError, match="verified Host allocation"):
        MODULE.complete_flat_dag(deps, mapping, {11, 12}, 1)


def test_missing_compute_node_timing_is_rejected():
    records = capture([[0, 0, 0, 1, 101]])
    with pytest.raises(ValueError, match="unmeasured kernel"):
        MODULE.ideal_lower_bound_ms(records, dag(2, []))


def test_communication_case_is_excluded(tmp_path):
    bundle, run = tmp_path / "bundle", tmp_path / "run"
    bundle.mkdir()
    run.mkdir()
    (bundle / "case.json").write_text('{"device_count": 2}')
    (run / "result.json").write_text('{"ranks": 2}')
    with pytest.raises(ValueError, match="Communication cases"):
        MODULE.read_case(bundle, run)


def test_mix_lanes_contribute_parallel_work_not_serial_sum():
    records = capture([[0, 0, 0, 1, 101], [1, 0, 1, 1, 81], [2, 0, 2, 1, 81], [0, 1, 3, 101, 111]])
    deps = dag(2, [(0, 1)])
    deps["tasks"][0]["kernel_ids"] = [0, 1, 2]
    assert MODULE.ideal_lower_bound_ms(records, deps) == pytest.approx(0.00011)
