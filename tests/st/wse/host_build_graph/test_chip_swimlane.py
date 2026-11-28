# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Correlate the offline swimlane with the simulator's actual events and Host calendar."""

import json
import shutil

import pytest
from simpler.task_interface import CallConfig
from simpler.worker import Worker

from tests.st.wse.host_build_graph.deployment.test_sram_ffn import build_sram_ffn, check_ffn, prepare_ffn
from tests.st.wse.host_build_graph.direct_launch.test_sync_start import build, check_outputs, task_args
from tests.st.wse.host_build_graph.scene_helpers import convert_swimlane
from tests.st.wse.host_build_graph.tile_operations.test_execution import load_gate

pytestmark = [pytest.mark.platforms(["wsesim"]), pytest.mark.runtime("host_build_graph"), pytest.mark.device_count(1)]
TASK_DONE, THREAD_BEGIN, THREAD_END, PE_ARRIVE, SYNC_RELEASE = 6, 16, 17, 18, 19
INSTANTS = {TASK_DONE: "TASK_DONE", PE_ARRIVE: "PE_ARRIVE", SYNC_RELEASE: "SYNC_RELEASE"}


@pytest.fixture
def worker(st_platform, st_device_ids):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=st_device_ids[0])
    worker.init()
    try:
        yield worker
    finally:
        worker.close()


def read_swimlane(path, level):
    header, *raw = map(json.loads, (path / "wse_events.jsonl").read_text().splitlines())
    assert (header["clock"], header["timestamp_unit"]) == ("CLOCK_MONOTONIC", "ns")
    by_seq = {row["seq"]: row for row in raw}
    calendar = json.loads((path / "calendar.json").read_text())
    capture = path / "chip_swimlane_records.json"
    records = json.loads(capture.read_text())
    assert records["chip_swimlane_level"] == level
    assert records["metadata"]["run_epoch"] == calendar["epoch"] == header["epoch"]
    assert [task["waits"] for task in sorted(records["tasks"], key=lambda task: task["task_id"])] == calendar["waits"]

    # The converter gets just the capture, in a fresh directory with no calendar or raw log.
    offline = path / "offline"
    offline.mkdir()
    shutil.copyfile(capture, offline / capture.name)
    converted = convert_swimlane(offline)
    trace = converted["traceEvents"]
    origin = converted["metadata"]["timeline_origin_ns"]
    metadata = {(e["name"], e["pid"], e.get("tid", 0)): e["args"] for e in trace if e["ph"] == "M"}
    cores = {
        pid: int(args["name"].removeprefix("Core "))
        for (name, pid, _tid), args in metadata.items()
        if name == "process_name" and args["name"].startswith("Core ")
    }
    assert sorted(cores.values()) == list(range(header["core_count"]))
    process_order = {
        pid: args["sort_index"] for (name, pid, _tid), args in metadata.items() if name == "process_sort_index"
    }
    assert [cores[pid] for pid in sorted(cores, key=process_order.__getitem__)] == list(range(header["core_count"]))
    physical = {}
    for pid, core in cores.items():
        pes = [
            (tid, args["name"])
            for (name, owner, tid), args in metadata.items()
            if name == "thread_name" and owner == pid
        ]
        pes.sort(key=lambda pair: metadata["thread_sort_index", pid, pair[0]]["sort_index"])
        assert [name for tid, name in pes] == [f"PE {i}" for i in range(4)]
        physical.update({(pid, tid): (core, pe) for pe, (tid, _name) in enumerate(pes)})

    placements = {coordinate: [] for coordinate in physical.values()}
    sources, destinations = {}, {}
    events = []
    for event in trace:
        phase = event["ph"]
        if phase == "M":
            continue
        args = event["args"]
        row = by_seq[args["seq"]]
        assert origin + round(event["ts"] * 1000) == row["monotonic_ns"]
        lane = (event["pid"], event["tid"])
        if row["core"] is not None:
            assert physical[lane] == (row["core"], row["pe"])
        if phase in ("s", "f"):
            endpoints = sources if phase == "s" else destinations
            assert event["id"] not in endpoints
            endpoints[event["id"]] = row["seq"]
            continue
        assert args.keys() <= {"seq", "end_seq", "epoch", "core_id", "pe_id", "task", "block_idx"}
        assert (args["epoch"], args["task"]) == (header["epoch"], row["task"])
        if row["core"] is not None:
            assert (args["core_id"], args["pe_id"]) == physical[lane]
        if phase == "X":
            assert event["cat"] == "wse.kernel" and row["kind"] == THREAD_BEGIN
            end = by_seq[args["end_seq"]]
            assert end["kind"] == THREAD_END
            assert round(event["dur"] * 1000) == end["monotonic_ns"] - row["monotonic_ns"]
            assert (end["core"], end["pe"]) == physical[lane]
            assert (end["task"], end["block"]) == (row["task"], args["block_idx"]) == (row["task"], row["block"])
            placements[physical[lane]].append((row["seq"], row["task"], row["block"]))
            events.append(end)
        else:
            assert (phase, event["cat"], event["name"]) == ("i", "wse.event", INSTANTS[row["kind"]])
        events.append(row)
    for core, queues in enumerate(calendar["pes"]):
        for pe, queue in enumerate(queues):
            assert [(task, block) for _, task, block in sorted(placements[core, pe])] == [
                (row["task"], row["block"]) for row in queue
            ]
    waited_tasks = {pred for queues in calendar["pes"] for queue in queues for row in queue for pred in row["waits"]}
    expected = [
        row["seq"]
        for row in raw
        if row["kind"] in (THREAD_BEGIN, THREAD_END, PE_ARRIVE, SYNC_RELEASE)
        or (row["kind"] == TASK_DONE and row["task"] in waited_tasks)
    ]
    assert sorted(row["seq"] for row in events) == sorted(expected)
    assert sources.keys() == destinations.keys()
    edges = {(sources[flow], destinations[flow]) for flow in sources}
    assert len(edges) == len(sources)
    assert all(by_seq[start]["monotonic_ns"] <= by_seq[end]["monotonic_ns"] for start, end in edges)
    return events, edges


@pytest.mark.parametrize("sync", [False, True], ids=["independent", "sync_start"])
def test_group_swimlane(tmp_path, worker, st_platform, sync):
    handle = worker.register(build(st_platform))
    gate = load_gate(
        tmp_path, "wse_sync_swimlane_probe" if sync else "wse_independent_swimlane_probe", 0 if sync else 0b11010
    )
    args, buffers = task_args(worker, sync, 2, 6, 3)
    config = CallConfig()
    config.output_prefix = str(tmp_path)
    config.enable_chip_swimlane = 2 if sync else 1
    run = worker.submit(handle, args, config)
    try:
        assert gate.wait_probe(), "the busy SS and the expected Group PEs did not enter their kernels"
    finally:
        gate.release_probe()
    run.wait(timeout=15)
    check_outputs(worker, buffers)

    events, edges = read_swimlane(tmp_path, config.enable_chip_swimlane)
    members = [e for e in events if e["task"] == 1]
    starts = [e for e in members if e["kind"] == THREAD_BEGIN]
    participants = [(core, block, pe) for core, block in ((2, 0), (6, 1)) for pe in (2, 3)]
    assert sorted((e["core"], e["block"], e["pe"]) for e in starts) == participants
    (busy_end,) = [e for e in events if e["task"] == 0 and e["kind"] == THREAD_END]
    delayed = (busy_end["core"], busy_end["pe"])
    assert delayed == (2, 3)
    arrivals = [e for e in members if e["kind"] == PE_ARRIVE]
    releases = [e for e in members if e["kind"] == SYNC_RELEASE]
    if sync:
        assert sorted((e["core"], e["block"], e["pe"]) for e in arrivals) == participants
        (delayed_arrival,) = [e for e in arrivals if (e["core"], e["pe"]) == delayed]
        assert busy_end["seq"] < delayed_arrival["seq"]
        (release,) = releases
        last_arrival = max(arrivals, key=lambda e: e["seq"])
        assert (release["core"], release["pe"]) == (last_arrival["core"], last_arrival["pe"])
        assert last_arrival["seq"] < release["seq"] < min(e["seq"] for e in starts)
        assert {(a["seq"], release["seq"]) for a in arrivals} == {edge for edge in edges if edge[1] == release["seq"]}
        assert {(release["seq"], s["seq"]) for s in starts} == {edge for edge in edges if edge[0] == release["seq"]}
    else:
        assert not arrivals and not releases
        for start in starts:
            assert (start["seq"] > busy_end["seq"]) == ((start["core"], start["pe"]) == delayed)


@pytest.mark.parametrize("sync", [False, True], ids=["independent", "sync_start"])
def test_group_predecessor_flows(tmp_path, worker, st_platform, sync):
    handle = worker.register(build(st_platform))
    args, buffers = task_args(worker, sync, 2, 6, False, producer_count=1)
    config = CallConfig()
    config.output_prefix = str(tmp_path)
    config.enable_chip_swimlane = 1
    worker.run(handle, args, config)
    check_outputs(worker, buffers)
    events, edges = read_swimlane(tmp_path, config.enable_chip_swimlane)
    (producer,) = [e for e in events if e["task"] == 0 and e["kind"] == TASK_DONE]
    group = [e for e in events if e["task"] == 1]
    destinations = [e for e in group if e["kind"] == (PE_ARRIVE if sync else THREAD_BEGIN)]
    assert sorted((e["core"], e["pe"]) for e in destinations) == [(2, 2), (2, 3), (6, 2), (6, 3)]
    group_seqs = {e["seq"] for e in group}
    assert {dst for src, dst in edges if src == producer["seq"] and dst in group_seqs} == {
        e["seq"] for e in destinations
    }


def test_ffn_swimlane(tmp_path, worker, st_platform):
    # Mode 2 rotates each Block to the next Core and assigns its first matrix row to PE 1.
    handle = worker.register(build_sram_ffn(st_platform, 1))
    args, buffers, tensors, goldens = prepare_ffn(worker, seed=0, mode=2)
    config = CallConfig()
    config.output_prefix = str(tmp_path)
    config.enable_chip_swimlane = 3
    worker.run(handle, args, config)
    check_ffn(worker, tmp_path, buffers, tensors, goldens, 2)

    events, edges = read_swimlane(tmp_path, config.enable_chip_swimlane)
    done = {e["task"]: e["seq"] for e in events if e["kind"] == TASK_DONE}
    for stage in (1, 2, 3):
        starts = [e for e in events if e["task"] == stage and e["kind"] == THREAD_BEGIN]
        assert len(starts) == 160
        actual = {(e["block"], e["core"], e["pe"]) for e in starts}
        assert actual == {(b, (b + 1) % 40, pe) for b in range(40) for pe in range(4)}
        for start in starts:
            if stage in (2, 3):
                assert (done[stage - 1], start["seq"]) in edges
    assert not any(e["kind"] in (PE_ARRIVE, SYNC_RELEASE) for e in events)
