# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check that resource and common-start semantics affect the model optimum."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.manual

pytest.importorskip("ortools.sat.python.cp_model")
PRODUCTION = Path(__file__).resolve().parents[4] / "tests/fdwic/dsv41_scheduler/production"
SPEC = importlib.util.spec_from_file_location("sync_bound", PRODUCTION / "sync_bound.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_sync_start_changes_optimum_with_identical_kernel_times_and_dag():
    tasks = {
        0: {"sync": False, "units": [(0, 1, "aiv", 100)]},
        1: {"sync": False, "units": [(0, 1, "aiv", 10), (1, 1, "aiv", 10)]},
    }
    independent = MODULE.solve_model(tasks, [], {"aiv": 2}, seconds=10)
    tasks[1]["sync"] = True
    together = MODULE.solve_model(tasks, [], {"aiv": 2}, seconds=10)
    assert independent["optimal"] and together["optimal"]
    assert independent["lower_ticks"] == independent["feasible_ticks"] == 100
    assert together["lower_ticks"] == together["feasible_ticks"] == 110


def test_mix_block_starts_together_and_task_completion_orders_successor():
    tasks = {
        0: {"sync": False, "units": [(0, 0, "aic", 10), (0, 1, "aiv", 4)]},
        1: {"sync": False, "units": [(0, 1, "aiv", 7)]},
    }
    result = MODULE.solve_model(tasks, [(0, 1)], {"aic": 1, "aiv": 1}, seconds=10)
    assert result["optimal"] and result["lower_ticks"] == 17
    mix = [unit for unit in result["schedule"] if unit["task"] == 0]
    successor = next(unit for unit in result["schedule"] if unit["task"] == 1)
    assert len({unit["start"] for unit in mix}) == 1
    assert successor["start"] >= max(unit["end"] for unit in mix)


def test_oversubscribed_independent_blocks_require_two_waves():
    tasks = {0: {"sync": False, "units": [(block, 1, "aiv", 10) for block in range(3)]}}
    result = MODULE.solve_model(tasks, [], {"aiv": 2}, seconds=10)
    assert result["optimal"] and result["lower_ticks"] == 20
    repeated = MODULE.solve_model(tasks, [], {"aiv": 2}, seconds=10, previous=result)
    assert repeated["optimal"] and repeated["lower_ticks"] == 20


def test_model_rejects_cycle_instead_of_publishing_a_bound():
    tasks = {i: {"sync": False, "units": [(0, 1, "aiv", 10)]} for i in (0, 1)}
    with pytest.raises(AssertionError, match="cyclic"):
        MODULE.solve_model(tasks, [(0, 1), (1, 0)], {"aiv": 2})
