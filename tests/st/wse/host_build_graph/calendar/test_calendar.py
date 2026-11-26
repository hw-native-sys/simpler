# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Calendar dispatch of one SS kernel with single-Block and SPMD launches in a WSE Graph."""

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

pytestmark = [pytest.mark.platforms(["wsesim"]), pytest.mark.runtime("host_build_graph"), pytest.mark.device_count(1)]

BLOCK_BEGIN, BLOCK_END, TASK_DONE = 4, 5, 6


@lru_cache
def build_graph(platform, *, independent=False):
    return build_callable(
        platform,
        Path(__file__).parent / "kernels",
        "calendar_graph.cpp",
        [D.IN, D.IN, D.OUT, D.OUT, D.OUT] * (1 if independent else 2) + [D.IN],
        [("fused.cpp", [D.IN, D.IN, D.OUT, D.IN])],
    )


def run_graph(worker, handle, path, instances, *, independent=False, first_core=0):
    args = TaskArgs()
    views = []
    outputs = []
    try:
        for inputs, golden in instances:
            tensors = inputs + [torch.full_like(value, -999.0) for value in golden]
            for index, tensor in enumerate(tensors):
                view = layout_view(worker, tensor.nbytes, core=first_core + len(views))
                views.append(view)
                copy_to(worker, view, tensor)
                args.add_tensor(
                    view.tensor(shapes=tuple(tensor.shape), dtype=DataType.FLOAT32),
                    TensorArgType.INPUT if index < len(inputs) else TensorArgType.OUTPUT_EXISTING,
                )
                if index >= len(inputs):
                    outputs.append((tensor, view, golden[index - len(inputs)]))
        args.add_scalar(int(independent))
        config = CallConfig()
        config.output_prefix = str(path)
        config.enable_dep_gen = True
        worker.run(handle, args, config)
        for tensor, view, expected in outputs:
            copy_from(worker, tensor, view)
            torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
    finally:
        for view in views:
            release_view(worker, view)
    header, *events = [json.loads(line) for line in (path / "wse_events.jsonl").read_text().splitlines()]
    assert header["expected_tasks"] == 5 * len(instances)  # Enter, three kernel tasks, exit per Graph.
    return header, events


def assert_calendar_matches_dispatch(path, header, events):
    calendar = json.loads((path / "calendar.json").read_text())
    assert calendar["epoch"] == header["epoch"]
    assert len(calendar["cores"]) == header["core_count"]
    starts = sorted((e for e in events if e["kind"] == BLOCK_BEGIN), key=lambda e: e["seq"])
    for core, rows in enumerate(calendar["cores"]):
        order = [(e["task"], e["block"]) for e in starts if e["core"] == core]
        assert [(row["task"], row["block"]) for row in rows] == order
        ticks = [row["tick"] for row in rows]
        assert all(before < after for before, after in zip(ticks, ticks[1:]))
    return calendar["waits"]


def assert_fan_out_and_join(events, branches, first_task):
    # Compare events using IDs local to this invocation. Independent Graphs can interleave.
    events = [dict(e, task=e["task"] - first_task) for e in events if first_task <= e["task"] < first_task + 5]
    begins = {(e["task"], e["block"]): e for e in events if e["kind"] == BLOCK_BEGIN}
    ends = {(e["task"], e["block"]): e for e in events if e["kind"] == BLOCK_END}
    done = {e["task"]: e["seq"] for e in events if e["kind"] == TASK_DONE}
    expected = {(1, 0), (3, 0)} | {(2, b) for b in range(branches)}
    assert begins.keys() == ends.keys() == expected
    assert sum(e["kind"] == BLOCK_BEGIN for e in events) == sum(e["kind"] == BLOCK_END for e in events) == len(expected)
    assert sum(e["kind"] == TASK_DONE for e in events) == len(done) == 5
    assert done[0] < begins[1, 0]["seq"] < ends[1, 0]["seq"] < done[1]
    for block in range(branches):
        begin, end = begins[2, block], ends[2, block]
        assert begin["core"] == end["core"] == block
        assert done[1] < begin["seq"] < end["seq"] < done[2]
    assert done[2] < begins[3, 0]["seq"] < ends[3, 0]["seq"] < done[3] < done[4]


def test_native_graph_one_to_many_to_one(tmp_path, st_platform):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    weights = torch.tensor([[[1, 2], [1, -1]], [[2, -1], [0, 1]], [[1, 3], [-1, 2]], [[0, 1], [2, 3]]]).float()
    try:
        worker.init()
        handle = worker.register(build_graph(st_platform))
        epochs = []
        for repeat in range(2):
            instances = []
            for instance in range(2):
                x = torch.tensor([[2 + repeat + instance * 3, -3], [-1, 4 - repeat]]).float()
                seed = torch.relu(x @ weights[0])
                branches = torch.relu(seed @ weights)
                joined = torch.relu((branches @ weights).sum(dim=0))
                instances.append(([x, weights], [seed, branches, joined]))
            path = tmp_path / str(repeat)
            header, events = run_graph(worker, handle, path, instances, first_core=repeat)
            waits = assert_calendar_matches_dispatch(path, header, events)
            for first_task in (0, 5):
                assert_fan_out_and_join(events, len(weights), first_task)
                assert first_task + 1 in waits[first_task + 2] and first_task + 2 in waits[first_task + 3]
            shells = json.loads((path / "deps.json").read_text())["tasks"]
            assert len(shells) == 2
            boundary_ids = [{arg["tensor_id"] for arg in shell["args"]} for shell in shells]
            assert len(boundary_ids[0]) == len(boundary_ids[1]) == 5 and boundary_ids[0].isdisjoint(boundary_ids[1])
            epochs.append(header["epoch"])
        assert epochs[0] != epochs[1]
        worker.unregister(handle)
    finally:
        worker.close()


def test_ready_critical_path_precedes_independent_leaf(tmp_path, st_platform):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    try:
        worker.init()
        handle = worker.register(build_graph(st_platform, independent=True))
        x = torch.tensor([[3, -2], [1, 4]]).float()
        weight = torch.tensor([[[2, 1], [-1, 3]]]).float()
        first = torch.relu(x @ weight[0])
        golden = [first, first, torch.relu(first @ weight[0])]
        header, events = run_graph(worker, handle, tmp_path, [([x, weight], golden)], independent=True)
        starts = sorted((e["seq"], e["task"], e["core"]) for e in events if e["kind"] == BLOCK_BEGIN)
        assert len(starts) == 3 and all(core == 0 for _, _, core in starts)
        assert starts[0][1] == 2  # Task 1 is the leaf. Task 2 unlocks task 3 on the same Core.
        waits = assert_calendar_matches_dispatch(tmp_path, header, events)
        assert 2 in waits[3]
        worker.unregister(handle)
    finally:
        worker.close()
