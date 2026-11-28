# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Synchronous and ordinary SPMD submissions in a direct WSE task list."""

import json
from functools import lru_cache
from pathlib import Path

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from tests.st.wse.host_build_graph.fixed_sram_layout import copy_from, copy_to, layout_view, release_view
from tests.st.wse.host_build_graph.scene_helpers import build_callable
from tests.st.wse.host_build_graph.tile_operations.test_execution import load_gate

pytestmark = [pytest.mark.platforms(["wsesim"]), pytest.mark.runtime("host_build_graph"), pytest.mark.device_count(1)]

BLOCK_BEGIN, BLOCK_END, TASK_DONE = 4, 5, 6
BLOCK_READY, SYNC_RELEASE = 18, 19


@lru_cache
def build(platform):
    return build_callable(
        platform,
        Path(__file__).parent / "kernels",
        "sync_start_orch.cpp",
        [D.OUT] * 3 + [D.IN] * 5,
        [("sync_start_member.cpp", [D.OUT] + [D.IN] * 4), ("sync_start_join.cpp", [D.IN, D.IN, D.OUT])],
    )


def task_args(worker, sync, blocks, second_core, probe, producer_count=0):
    args = TaskArgs()
    buffers = []
    for owner, count in zip((second_core, 2, 3), (1, 4, 1), strict=True):
        view = layout_view(worker, 4 * count, core=owner)
        copy_to(worker, view, torch.full((count,), -1.0))
        args.add_tensor(view.tensor(shapes=(count,), dtype=DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
        buffers.append(view)
    for value in (sync, blocks, second_core, probe, producer_count):
        args.add_scalar(int(value))
    return args, buffers


def check_outputs(worker, buffers):
    for view, expected in zip(buffers, ([7.0], [10.0, 11.0, 20.0, 21.0], [69.0]), strict=True):
        output = torch.zeros(len(expected))
        copy_from(worker, output, view)
        assert output.tolist() == expected


def audit(path, worker, buffers, *, sync, second_core):
    check_outputs(worker, buffers)
    header, *events = map(json.loads, (path / "wse_events.jsonl").read_text().splitlines())
    assert header["expected_tasks"] == 3
    calendar = json.loads((path / "calendar.json").read_text())
    assert calendar["epoch"] == header["epoch"]
    assert calendar["waits"][:2] == [[], []] and set(calendar["waits"][2]) == {0, 1}
    rows = [row for core in calendar["cores"] for row in core if row["task"] == 1]
    assert len(rows) == 2
    if sync:
        assert rows[0]["tick"] == rows[1]["tick"]

    def matching(task, kind):
        return sorted((e for e in events if e["task"] == task and e["kind"] == kind), key=lambda e: e["seq"])

    starts = matching(1, BLOCK_BEGIN)
    assert sorted((e["block"], e["core"]) for e in starts) == [(0, 2), (1, second_core)]
    for kind in (16, 17):
        pe_events = matching(1, kind)
        assert all(e["value"] == e["pe"] for e in pe_events)
        assert sorted((e["block"], e["core"], e["pe"]) for e in pe_events) == [
            (block, core, pe) for block, core in ((0, 2), (1, second_core)) for pe in (0, 1)
        ]
    start_by_block = {e["block"]: e["seq"] for e in starts}
    (busy_end_event,) = matching(0, BLOCK_END)
    assert busy_end_event["core"] == 2
    busy_end = busy_end_event["seq"]
    if second_core != 2:
        assert busy_end < start_by_block[0]
    ready, release = matching(1, BLOCK_READY), matching(1, SYNC_RELEASE)
    if sync:
        assert sorted((e["block"], e["core"]) for e in ready) == [(0, 2), (1, second_core)]
        assert len(release) == 1 and ready[-1]["seq"] < release[0]["seq"] < starts[0]["seq"]
        assert busy_end < release[0]["seq"]
    else:
        assert ready == release == []
        if second_core != 2:
            assert start_by_block[1] < busy_end  # Host held T0 until B1 actually entered its kernel.
    ends, done = matching(1, BLOCK_END), matching(1, TASK_DONE)
    assert len(ends) == 2 and len(done) == 1
    if second_core == 2:
        assert ends[0]["block"] == starts[0]["block"] and ends[0]["seq"] < starts[1]["seq"]
    (join_begin,) = matching(2, BLOCK_BEGIN)
    assert join_begin["core"] == 3 and ends[-1]["seq"] < done[0]["seq"] < join_begin["seq"]
    assert sorted(e["task"] for e in events if e["kind"] == TASK_DONE) == [0, 1, 2]


@pytest.mark.parametrize("sync", [False, True], ids=["independent", "sync_start"])
def test_direct_spmd_start(tmp_path, st_platform, sync):
    # The ordinary launch must enter B1 on the free Core before the host releases the busy Core.
    # For sync_start the host itself releases T0 after acknowledgement, independently of B1.
    gate = load_gate(tmp_path, "wse_sync_start_probe" if sync else "wse_independent_start_probe", 0 if sync else 2)
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    try:
        worker.init()
        handle = worker.register(build(st_platform))
        args, buffers = task_args(worker, sync, 2, 6, True)
        config = CallConfig()
        config.output_prefix = str(tmp_path)
        run = worker.submit(handle, args, config)
        try:
            assert gate.wait_probe()
        finally:
            gate.release_probe()
        run.wait(timeout=15)
        audit(tmp_path, worker, buffers, sync=sync, second_core=6)
        for view in buffers:
            release_view(worker, view)
    finally:
        gate.release_probe()
        worker.close()


def test_sync_start_rejects_single_and_duplicate_cores(tmp_path, st_platform):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    try:
        worker.init()
        handle = worker.register(build(st_platform))
        for blocks in (1, 2):
            args, buffers = task_args(worker, True, blocks, 2, False)
            config = CallConfig()
            config.output_prefix = str(tmp_path / str(blocks))
            with pytest.raises(RuntimeError):
                worker.run(handle, args, config)
            assert not (tmp_path / str(blocks) / "wse_events.jsonl").exists()
            for view in buffers:
                release_view(worker, view)

        # Identical placements are legal for ordinary SPMD and the Worker remains usable.
        args, buffers = task_args(worker, False, 2, 2, False)
        config = CallConfig()
        config.output_prefix = str(tmp_path)
        worker.run(handle, args, config)
        audit(tmp_path, worker, buffers, sync=False, second_core=2)
        for view in buffers:
            release_view(worker, view)
    finally:
        worker.close()


def test_busy_core_failure_wakes_sync_waiters(tmp_path, st_platform):
    gate = load_gate(tmp_path, "wse_sync_failure_probe", 0)
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    try:
        worker.init()
        handle = worker.register(build(st_platform))
        args, buffers = task_args(worker, True, 2, 6, 2)
        run = worker.submit(handle, args)
        try:
            assert gate.wait_probe()
        finally:
            gate.release_probe()
        with pytest.raises(RuntimeError):
            run.wait(timeout=15)
        for view in buffers:
            release_view(worker, view)

    finally:
        gate.release_probe()
        worker.close()

    # A device execution fault poisons the chip run lane. Closing it drains all PEs.
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    try:
        worker.init()
        handle = worker.register(build(st_platform))
        args, buffers = task_args(worker, True, 2, 6, False)
        config = CallConfig()
        config.output_prefix = str(tmp_path)
        worker.run(handle, args, config)
        audit(tmp_path, worker, buffers, sync=True, second_core=6)
        for view in buffers:
            release_view(worker, view)
    finally:
        worker.close()


def test_task_state_follows_explicit_core_placement(tmp_path, st_platform):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    try:
        worker.init()
        handle = worker.register(build(st_platform))

        def run(producer_core, producer_count):
            path = tmp_path / f"core-{producer_core}-ss-{producer_count}"
            config = CallConfig()
            config.output_prefix = str(path)
            args, buffers = task_args(worker, True, 2, producer_core, False, producer_count)
            try:
                worker.run(handle, args, config)
                check_outputs(worker, buffers)
            finally:
                for view in buffers:
                    release_view(worker, view)

            group = producer_count
            waits = json.loads((path / "calendar.json").read_text())["waits"]
            assert waits[group] == [group - 1] and group in waits[group + 1]
            header, *events = map(json.loads, (path / "wse_events.jsonl").read_text().splitlines())
            assert header["expected_tasks"] == group + 2

            def matching(task, kind):
                return sorted((e for e in events if e["task"] == task and e["kind"] == kind), key=lambda e: e["seq"])

            (producer_done,) = matching(group - 1, TASK_DONE)
            ready, begins, ends = (matching(group, kind) for kind in (BLOCK_READY, BLOCK_BEGIN, BLOCK_END))
            (release,) = matching(group, SYNC_RELEASE)
            (done,) = matching(group, TASK_DONE)
            (join,) = matching(group + 1, BLOCK_BEGIN)
            assert producer_done["core"] == producer_core and join["core"] == 3
            for members in (ready, begins, ends):
                assert sorted((e["block"], e["core"]) for e in members) == [(0, 2), (1, producer_core)]
            assert producer_done["seq"] < ready[0]["seq"] <= ready[-1]["seq"] < release["seq"] < begins[0]["seq"]
            assert ends[-1]["seq"] < done["seq"] < join["seq"]
            return json.loads((path / "wse_sram.json").read_text())

        one = run(producer_core=6, producer_count=1)
        many = run(producer_core=6, producer_count=16)
        swapped = run(producer_core=0, producer_count=16)

        def usage(report, core):
            return report["cores"][core]["runtime_used"]

        assert usage(many, 6) > usage(one, 6)
        for core in (0, 2):
            assert usage(many, core) == usage(one, core)
        assert usage(many, 6) == usage(swapped, 0)
        assert usage(many, 0) == usage(swapped, 6)
        for report, active in ((many, 6), (swapped, 0)):
            owners = {}
            signal_bytes = {}
            for section in report["sections"]:
                owner = section["owner"]
                owners.setdefault(section["name"], set()).add(owner)
                if section["name"] == "task_signal":
                    signal_bytes[owner] = signal_bytes.get(owner, 0) + section["bytes"]
            # Sixteen SS signals reside on their submitted Core. Only Block 0 owns the Group signal.
            assert signal_bytes[active] == 16 * signal_bytes[2] and signal_bytes[2] == signal_bytes[3]
            assert owners["core_control"] == set(range(len(report["cores"])))
            assert owners["local_task"] == owners["task_signal"] == owners["block_queue"] == {2, 3, active}
            assert owners["dependencies"] == {2, 3, active}
            for core in range(len(report["cores"])):
                if core not in {2, 3, active}:
                    assert {s["name"] for s in report["sections"] if s["owner"] == core} == {"core_control"}
    finally:
        worker.close()
