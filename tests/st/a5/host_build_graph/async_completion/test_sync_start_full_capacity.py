# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""One full-width async cohort; the numerical oracle remains the shared original."""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch

from simpler_setup.tools.swimlane_converter import read_perf_data

from .helpers import CONDITION_LIMIT, CONDITIONS_PER_EXECUTION, Case, CounterRunner, Scenario, l3_case, read_host_tensor

pytestmark = [
    pytest.mark.platforms(["a5sim", "a5"]),
    pytest.mark.runtime("host_build_graph"),
    pytest.mark.device_count(1),
]


def _native_capture(directory):
    paths = list(directory.rglob("chip_swimlane_records.json"))
    assert len(paths) == 1, paths
    raw = json.loads(paths[0].read_text())
    decoded = read_perf_data(paths[0])
    assert decoded["protocol_audit"]["protocol_verified"]
    return paths[0], raw["aicore_protocol"], decoded["protocol_audit"]


def _assert_topology(protocol, capacity):
    workers = protocol["workers"]
    assert protocol["cluster_count"] == capacity
    assert len(workers) == protocol["worker_count"] == 3 * capacity
    assert sum(worker["core_type"] == 0 for worker in workers) == capacity
    assert sum(worker["core_type"] == 1 for worker in workers) == 2 * capacity
    assert {worker["cluster_index"] for worker in workers} == set(range(capacity))
    return {worker["worker_index"] for worker in workers}


def _qualify_full_cohort(directory, capacity):
    path, protocol, audit = _native_capture(directory)
    assert protocol["schema"] == "hbg-native-protocol-v12"
    all_workers = _assert_topology(protocol, capacity)
    tasks = protocol["tasks"]
    main_tasks = [task for task in tasks if task["kernel_ids"] == [11, 9, 12]]
    assert len(main_tasks) == 1
    main = main_tasks[0]
    main_ref = main["task_ref"]
    assert main["active_mask"] == 7 and main["block_num"] == capacity
    assert main["flags"] & (4 | 8 | 16) == 4 | 8 | 16
    assert main["dependencies"] == []
    gates = [task for task in tasks if task["kernel_ids"] == [65535, 10, 65535]]
    consumers = [task for task in tasks if task["kernel_ids"] == [65535, 5, 65535]]
    publishers = [task for task in tasks if task["kernel_ids"] == [65535, 3, 65535]]
    assert len(gates) == 3 * capacity and len(consumers) == len(publishers) == 1
    assert len(tasks) == 3 * capacity + 3
    assert all(task["dependencies"] == [] for task in gates)
    assert consumers[0]["dependencies"] == [main_ref]
    assert len(publishers[0]["dependencies"]) == 3 * capacity
    assert set(publishers[0]["dependencies"]) == {task["task_ref"] for task in gates}

    executions = [row for row in protocol["executions"] if row["task_ref"] == main_ref]
    assert len(executions) == 3 * capacity
    assert {(row["block_idx"], row["subtask_slot"]) for row in executions} == {
        (block, lane) for block in range(capacity) for lane in range(3)
    }
    assert {row["worker_index"] for row in executions} == all_workers
    members = [row for row in protocol["members"] if row["task_ref"] == main_ref]
    assert len(members) == 3 * capacity and {row["worker_index"] for row in members} == all_workers
    blocks = [row for row in protocol["blocks"] if row["owner"]["task_ref"] == main_ref]
    assert len(blocks) == capacity
    assert {row["owner"]["block_idx"] for row in blocks} == set(range(capacity))
    assert all(len(row["lanes"]) == 3 for row in blocks)

    groups = [row for row in protocol["async"]["groups"] if row["task_ref"] == main_ref]
    assert len(groups) == 3 * capacity
    assert all(row["state"] == "READY" and row["registered_count"] == 1 for row in groups)
    assert all(row["conditions"][0]["type"] == 0 for row in groups)
    assert any(row["observations"] > row["registered_count"] for row in groups), (
        "diagnostic coverage missing: no main-cohort COUNTER was observed unmet"
    )
    releases = [row for row in protocol["async"]["releases"] if row["task_ref"] == main_ref]
    assert len(releases) == capacity and {row["block_idx"] for row in releases} == set(range(capacity))
    finishes = [row for row in protocol["async"]["finishes"] if row["task_ref"] == main_ref]
    assert len(finishes) == 1 and finishes[0]["state"] == "DRAINED"
    assert set(finishes[0]["group_indices"]) == {row["index"] for row in groups}
    return {"capture": str(path), "capacity_clusters": capacity, "main_task_ref": main_ref, "audit": audit}


@l3_case
def test_full_capacity_async_mix_releases_execution_for_its_publisher(st_platform, st_device_ids, tmp_path):
    # Reuse the existing public-capacity fixture. It records
    # rt_available_cluster_count() in Control::CAPACITY, without Host inference.
    probe_case = Case(scenario=Scenario.VISIBILITY, delay_ms=20)
    with CounterRunner(st_platform, st_device_ids, tmp_path / "capacity_probe", probe_case) as probe:
        probe.run()
        capacity = int(probe.control[0, 5, 0])
        assert 2 <= capacity <= 36, "the fixture needs a multi-block A5 cohort and at most108 gate fanins"
        _, protocol, _ = _native_capture(probe.capture)
        _assert_topology(protocol, capacity)

    case = Case(scenario=Scenario.FULL_SYNC_START, blocks=capacity, mask=7, sync_start=True, delay_ms=20)
    with CounterRunner(st_platform, st_device_ids, tmp_path / "full_cohort", case) as runner:
        evidence: dict[str, Any] = {"capacity_clusters": capacity, "error": None}
        try:
            runner.run()  # The unchanged numerical, registration and guard oracle.
            assert int(runner.control[0, 5, 0]) == capacity
            copied = read_host_tensor(runner.signal_staging, runner.initial_signals)
            expected = runner.initial_signals.clone()
            expected[0::CONDITIONS_PER_EXECUTION, 0] = case.expected
            expected[CONDITION_LIMIT::CONDITIONS_PER_EXECUTION, 0] = 1
            torch.testing.assert_close(copied, expected, rtol=0, atol=0)
            evidence["qualification"] = _qualify_full_cohort(runner.capture, capacity)
        except BaseException as error:
            evidence["error"] = repr(error)
            raise
        finally:
            evidence.update(
                records=runner.records.tolist(),
                output=runner.output.tolist(),
                control=runner.control.tolist(),
                note="Host buffers after an invocation failure do not prove device copyback completed.",
            )
            (tmp_path / "full-sync-start-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
