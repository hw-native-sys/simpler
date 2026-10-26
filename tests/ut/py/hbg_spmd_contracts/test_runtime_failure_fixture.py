# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Bad-record checks for the resident-timeout fixture, not device failure evidence."""

import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def native(request):
    request.config.addinivalue_line("markers", "scheduler_mode(kind): required scheduler selection")
    path = Path(__file__).resolve().parents[3] / "st/a5/host_build_graph/resident_failure/test_resident_failure.py"
    spec = importlib.util.spec_from_file_location("_resident_failure_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        yield module


def _capture():
    return {
        "metadata": {"clock_freq_hz": 1_000_000_000, "num_cores": 3},
        "aicore_protocol": {
            "producer": "aicore",
            "scheduler_error": 10,
            "first_error": {"site": 95},
            "worker_count": 3,
            "active_worker_count": 3,
            "workers": [{"worker_index": worker, "finished": 2, "error_flags": 0} for worker in range(3)],
            "tasks": [
                {"task_ref": 0, "active_mask": 3, "kernel_ids": [0, 1, 65535], "block_num": 2, "dependencies": []},
                {"task_ref": 1, "active_mask": 2, "kernel_ids": [65535, 3, 65535], "block_num": 1, "dependencies": [0]},
            ],
            "executions": [
                {
                    "task_ref": 0,
                    "block_idx": 0,
                    "subtask_slot": 0,
                    "worker_index": 0,
                    "entry": [100, 1],
                    "returned": [2_500_000_100, 2],
                    "visible": [2_500_000_101, 3],
                }
            ],
            "task_controls": [
                {"task_ref": 0, "state": 0, "completed_blocks": 1},
                {"task_ref": 1, "state": 0, "completed_blocks": 0},
            ],
            "resolved_task_count": 0,
        },
        "aicpu_lifecycle_records": [
            {
                "aicpu_thread_id": 0,
                "exit_signal_start_cycles": 2_000_000_100,
                "exit_signal_end_cycles": 2_000_000_150,
                "exit_wait_start_cycles": 2_000_000_200,
                "exit_wait_end_cycles": 2_500_000_200,
            },
            {
                "aicpu_thread_id": 1,
                "exit_signal_start_cycles": 2_000_000_101,
                "exit_signal_end_cycles": 2_000_000_151,
                "exit_wait_start_cycles": 2_000_000_201,
                "exit_wait_end_cycles": 2_000_000_300,
            },
        ],
    }


@pytest.mark.parametrize("site", (95, 102))
def test_both_real_timeout_writers_have_the_same_containment_requirements(native, site):
    raw = _capture()
    raw["aicore_protocol"]["first_error"]["site"] = site
    native._check_failed_capture(raw, 2, 3, 0)


def test_nonreturning_probe_is_hardware_only_and_covers_each_executing_lane(native):
    test = native.test_nonreturning_kernel_timeout_keeps_failed_run_contained
    platforms = [mark.args[0] for mark in test.pytestmark if mark.name == "platforms"]
    assert platforms == [["a5"]]
    shapes = [mark.args[1] for mark in test.pytestmark if mark.name == "parametrize"]
    assert len(shapes) == 1
    assert sorted(shapes[0]) == [(mask, lane) for mask in range(1, 8) for lane in range(3) if mask & (1 << lane)]


def _nonreturning_capture():
    raw = _capture()
    protocol = raw["aicore_protocol"]
    protocol["workers"][0]["finished"] = 0
    execution = protocol["executions"][0]
    for field in ("returned", "visible", "completion_consumed", "trace_published"):
        execution[field] = [0, 0]
    raw["aicpu_lifecycle_records"][0]["exit_wait_end_cycles"] = 3_000_000_200
    return raw


def test_nonreturning_capture_proves_missing_return_and_exit_ack_timeout(native):
    native._check_failed_capture(_nonreturning_capture(), 2, 3, 0, nonreturning=True)


@pytest.mark.parametrize(
    "corruption",
    (
        "missing_entry",
        "return",
        "visible",
        "consumed",
        "trace",
        "fake_ack",
        "extra_unfinished",
        "wrong_partition",
        "short_exit_wait",
        "premature_exit",
        "late_signal_end",
        "consumer_entered",
        "task_done",
        "completed_blocks",
        "resolved_count",
    ),
)
def test_nonreturning_capture_rejects_fabricated_completion_or_unrelated_failure(native, corruption):
    raw = _nonreturning_capture()
    protocol = raw["aicore_protocol"]
    execution = protocol["executions"][0]
    if corruption == "missing_entry":
        execution["entry"] = [0, 0]
    elif corruption in ("return", "visible", "consumed", "trace"):
        field = {
            "return": "returned",
            "visible": "visible",
            "consumed": "completion_consumed",
            "trace": "trace_published",
        }[corruption]
        execution[field] = [2_500_000_100, 2]
    elif corruption == "fake_ack":
        protocol["workers"][0]["finished"] = 2
    elif corruption == "extra_unfinished":
        protocol["workers"][1]["finished"] = 0
    elif corruption == "wrong_partition":
        execution["worker_index"] = 2
        protocol["workers"][0]["finished"] = 2
        protocol["workers"][2]["finished"] = 0
    elif corruption == "short_exit_wait":
        raw["aicpu_lifecycle_records"][0]["exit_wait_end_cycles"] -= 1
    elif corruption == "late_signal_end":
        raw["aicpu_lifecycle_records"][0]["exit_signal_end_cycles"] = 3_100_000_000
    else:
        _corrupt_shutdown(raw, corruption)
    with pytest.raises(AssertionError):
        native._check_failed_capture(raw, 2, 3, 0, nonreturning=True)


@pytest.mark.parametrize(
    "corruption",
    (
        "no_error",
        "wrong_site",
        "unfinished_worker",
        "missing_worker",
        "duplicate_worker",
        "truncated_topology",
        "worker_error",
        "dependency",
        "wrong_mask",
        "wrong_block",
        "missing_return",
        "return_sequence",
        "short_stall",
        "premature_exit",
        "late_exit",
        "late_signal_end",
        "premature_retirement",
        "wrong_partition",
        "duplicate_partition",
        "consumer_entered",
        "task_done",
        "completed_blocks",
        "duplicate_control",
        "missing_control",
        "resolved_count",
    ),
)
def test_failed_capture_cannot_hide_missing_exit_or_successful_dependency_release(native, corruption):
    raw = _capture()
    protocol = raw["aicore_protocol"]
    execution = protocol["executions"][0]
    if corruption == "no_error":
        protocol["scheduler_error"] = 0
    elif corruption == "wrong_site":
        protocol["first_error"]["site"] = 93
    elif corruption == "unfinished_worker":
        protocol["workers"][0]["finished"] = 0
    elif corruption == "missing_worker":
        protocol["workers"].pop()
    elif corruption == "duplicate_worker":
        protocol["workers"][1] = copy.deepcopy(protocol["workers"][0])
    elif corruption == "truncated_topology":
        protocol["workers"].pop()
        protocol["active_worker_count"] = protocol["worker_count"] = 2
    elif corruption == "worker_error":
        protocol["workers"][0]["error_flags"] = 1
    elif corruption == "dependency":
        protocol["tasks"][1]["dependencies"] = []
    elif corruption == "wrong_mask":
        protocol["tasks"][0]["active_mask"] = 1
    elif corruption == "wrong_block":
        execution["block_idx"] = 1
    elif corruption == "missing_return":
        execution["returned"] = [0, 0]
    elif corruption == "return_sequence":
        execution["returned"][1] = execution["entry"][1]
    elif corruption == "short_stall":
        execution["returned"][0] -= 1
    else:
        _corrupt_shutdown(raw, corruption)
    with pytest.raises(AssertionError):
        native._check_failed_capture(raw, 2, 3, 0)


def _corrupt_shutdown(raw, corruption):
    protocol = raw["aicore_protocol"]
    execution = protocol["executions"][0]
    if corruption == "premature_exit":
        raw["aicpu_lifecycle_records"][0]["exit_signal_start_cycles"] = 99
    elif corruption == "late_exit":
        for record in raw["aicpu_lifecycle_records"]:
            record["exit_signal_start_cycles"] = execution["returned"][0] + 1
    elif corruption == "late_signal_end":
        raw["aicpu_lifecycle_records"][0]["exit_signal_end_cycles"] = execution["returned"][0] + 1
    elif corruption == "premature_retirement":
        for record in raw["aicpu_lifecycle_records"]:
            record["exit_wait_end_cycles"] = execution["returned"][0] - 1
    elif corruption == "wrong_partition":
        raw["aicpu_lifecycle_records"][0]["exit_wait_end_cycles"] = execution["returned"][0] - 1
        raw["aicpu_lifecycle_records"][1]["exit_wait_end_cycles"] = execution["returned"][0] + 100
    elif corruption == "duplicate_partition":
        raw["aicpu_lifecycle_records"][1]["aicpu_thread_id"] = 0
    elif corruption == "consumer_entered":
        protocol["executions"].append({**execution, "task_ref": 1})
    elif corruption == "task_done":
        protocol["task_controls"][0]["state"] = 2
    elif corruption == "completed_blocks":
        protocol["task_controls"][0]["completed_blocks"] = 2
    elif corruption == "duplicate_control":
        protocol["task_controls"].append(copy.deepcopy(protocol["task_controls"][0]))
    elif corruption == "missing_control":
        protocol["task_controls"].pop()
    else:
        protocol["resolved_task_count"] = 1


@pytest.mark.parametrize(
    "corruption", ("duplicate", "missing_return", "wrong_n", "guard", "consumer", "inactive", "padding")
)
def test_failed_output_checks_return_writes_and_untouched_successor(native, corruption):
    records = torch.zeros((4, 16), dtype=torch.int64)
    for lane in (0, 1):
        records[lane, :2] = 1
        records[lane, 8:14] = torch.tensor([1, 1, lane + 1, 1, 100, 200])
    values = [records.flatten(), torch.zeros(16, dtype=torch.int64)]
    native._check_output(values, 1, 3, failed=True)
    if corruption == "duplicate":
        records[0, 0] = 2
    elif corruption == "missing_return":
        records[0, 1] = 0
    elif corruption == "wrong_n":
        records[0, 9] = 2
    elif corruption == "guard":
        records[3, 0] = 1
    elif corruption == "consumer":
        values[1][0] = 2
    elif corruption == "inactive":
        records[2, 0] = 1
    else:
        records[0, 14] = 1
    with pytest.raises(AssertionError):
        native._check_output(values, 1, 3, failed=True)
