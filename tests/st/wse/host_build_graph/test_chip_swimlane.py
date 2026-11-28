# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Read the simulator's native Perfetto file and correlate physical execution with the raw event log."""

import json

import pytest
from simpler.task_interface import CallConfig
from simpler.worker import Worker

from tests.st.wse.host_build_graph.deployment.test_sram_ffn import build_sram_ffn, check_ffn, prepare_ffn
from tests.st.wse.host_build_graph.direct_launch.test_sync_start import build, check_outputs, task_args
from tests.st.wse.host_build_graph.tile_operations.test_execution import load_gate

pytestmark = [pytest.mark.platforms(["wsesim"]), pytest.mark.runtime("host_build_graph"), pytest.mark.device_count(1)]
TASK_DONE, THREAD_BEGIN, THREAD_END, PE_ARRIVE, SYNC_RELEASE = 6, 16, 17, 18, 19
INSTANTS = {TASK_DONE: "TASK_DONE", PE_ARRIVE: "PE_ARRIVE", SYNC_RELEASE: "SYNC_RELEASE"}


@pytest.fixture
def perfetto():
    return pytest.importorskip("perfetto.protos.perfetto.trace.perfetto_trace_pb2")


@pytest.fixture
def worker(st_platform, st_device_ids):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=st_device_ids[0])
    worker.init()
    try:
        yield worker
    finally:
        worker.close()


def read_swimlane(path, proto):
    header, *raw = map(json.loads, (path / "wse_events.jsonl").read_text().splitlines())
    assert (header["clock"], header["timestamp_unit"]) == ("CLOCK_MONOTONIC", "ns")
    by_seq = {row["seq"]: row for row in raw}
    calendar = json.loads((path / "calendar.json").read_text())
    assert calendar["epoch"] == header["epoch"]
    trace = proto.Trace.FromString((path / "wse_swimlane.pftrace").read_bytes())
    tracks = {p.track_descriptor.uuid: p.track_descriptor for p in trace.packet if p.HasField("track_descriptor")}
    (root,) = [track for track in tracks.values() if not track.parent_uuid]
    assert root.name == "WSE" and root.child_ordering == proto.TrackDescriptor.EXPLICIT
    cores = sorted((t for t in tracks.values() if t.parent_uuid == root.uuid), key=lambda t: t.sibling_order_rank)
    assert [c.name for c in cores] == [f"Core {i}" for i in range(header["core_count"])]
    physical = {}
    for core_id, core in enumerate(cores):
        assert core.child_ordering == proto.TrackDescriptor.EXPLICIT
        pes = sorted((t for t in tracks.values() if t.parent_uuid == core.uuid), key=lambda t: t.sibling_order_rank)
        assert [pe.name for pe in pes] == [f"PE {i}" for i in range(4)]
        physical.update({pe.uuid: (core_id, i) for i, pe in enumerate(pes)})
    assert set(tracks) == {root.uuid, *(c.uuid for c in cores), *physical}

    placements = {coordinate: [] for coordinate in physical.values()}
    sources, destinations, active = {}, {}, {}
    events = []
    for packet in trace.packet:
        if not packet.HasField("track_event"):
            continue
        event = packet.track_event
        args = {arg.name: arg.uint_value for arg in event.debug_annotations}
        assert args.keys() <= {"seq", "epoch", "core_id", "pe_id", "task", "block_idx"}
        row = by_seq[args["seq"]]
        assert packet.timestamp == row["monotonic_ns"] > 0
        assert args["epoch"] == header["epoch"]
        if row["core"] is None:
            assert event.track_uuid == root.uuid
        else:
            assert physical[event.track_uuid] == (args["core_id"], args["pe_id"]) == (row["core"], row["pe"])
        if row["kind"] in INSTANTS:
            assert (event.name, event.type) == (INSTANTS[row["kind"]], proto.TrackEvent.TYPE_INSTANT)
            assert "wse.event" in event.categories
        if row["kind"] in (THREAD_BEGIN, THREAD_END):
            assert "wse.kernel" in event.categories
            key = (args["task"], args["block_idx"])
            assert key == (row["task"], row["block"])
            assert row["value"] == row["pe"]
            if row["kind"] == THREAD_BEGIN:
                assert event.type == proto.TrackEvent.TYPE_SLICE_BEGIN and event.track_uuid not in active
                active[event.track_uuid] = key
                placements[physical[event.track_uuid]].append(key)
            else:
                assert event.type == proto.TrackEvent.TYPE_SLICE_END and active.pop(event.track_uuid) == key
        for flow in event.flow_ids:
            assert flow not in sources
            sources[flow] = row["seq"]
        for flow in event.terminating_flow_ids:
            assert flow not in destinations
            destinations[flow] = row["seq"]
        events.append(row)
    assert not active
    assert placements == {
        (core_id, pe_id): [(row["task"], row["block"]) for row in queue]
        for core_id, queues in enumerate(calendar["pes"])
        for pe_id, queue in enumerate(queues)
    }
    assert sources.keys() == destinations.keys()
    waited_tasks = {pred for queues in calendar["pes"] for queue in queues for row in queue for pred in row["waits"]}
    expected = [
        row["seq"]
        for row in raw
        if row["kind"] in (THREAD_BEGIN, THREAD_END, PE_ARRIVE, SYNC_RELEASE)
        or (row["kind"] == TASK_DONE and row["task"] in waited_tasks)
    ]
    assert sorted(row["seq"] for row in events) == sorted(expected)
    edges = {(sources[flow], destinations[flow]) for flow in sources}
    assert all(by_seq[start]["monotonic_ns"] <= by_seq[end]["monotonic_ns"] for start, end in edges)
    return events, edges


@pytest.mark.parametrize("sync", [False, True], ids=["independent", "sync_start"])
def test_group_swimlane(tmp_path, worker, st_platform, perfetto, sync):
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

    events, edges = read_swimlane(tmp_path, perfetto)
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
def test_group_predecessor_flows(tmp_path, worker, st_platform, perfetto, sync):
    handle = worker.register(build(st_platform))
    args, buffers = task_args(worker, sync, 2, 6, False, producer_count=1)
    config = CallConfig()
    config.output_prefix = str(tmp_path)
    config.enable_chip_swimlane = 1
    worker.run(handle, args, config)
    check_outputs(worker, buffers)
    events, edges = read_swimlane(tmp_path, perfetto)
    (producer,) = [e for e in events if e["task"] == 0 and e["kind"] == TASK_DONE]
    group = [e for e in events if e["task"] == 1]
    destinations = [e for e in group if e["kind"] == (PE_ARRIVE if sync else THREAD_BEGIN)]
    assert sorted((e["core"], e["pe"]) for e in destinations) == [(2, 2), (2, 3), (6, 2), (6, 3)]
    group_seqs = {e["seq"] for e in group}
    assert {dst for src, dst in edges if src == producer["seq"] and dst in group_seqs} == {
        e["seq"] for e in destinations
    }


def test_ffn_swimlane(tmp_path, worker, st_platform, perfetto):
    # Mode 2 rotates each Block to the next Core and assigns its first matrix row to PE 1.
    handle = worker.register(build_sram_ffn(st_platform, 1))
    args, buffers, tensors, goldens = prepare_ffn(worker, seed=0, mode=2)
    config = CallConfig()
    config.output_prefix = str(tmp_path)
    config.enable_chip_swimlane = 3
    worker.run(handle, args, config)
    check_ffn(worker, tmp_path, buffers, tensors, goldens, 2)

    events, edges = read_swimlane(tmp_path, perfetto)
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
