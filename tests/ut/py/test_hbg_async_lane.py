# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Synthetic presentation tests; no runtime or hardware qualification."""

import copy
import json

import pytest
from _capture_builder import Capture
from test_hbg_native_async import add_async, async_capture
from test_hbg_native_causality import resequence
from test_hbg_native_causality import wire_capture as historical_capture
from test_hbg_native_graph import native_capture
from test_swimlane_converter import _write_hbg_l3_rank

from simpler_setup.tools import swimlane_converter as sc
from simpler_setup.tools.hbg import native_protocol
from simpler_setup.tools.hbg.native_protocol import STAMPS, _event_records


def wire_capture(protocol=None, identities=None):
    if protocol is None:
        protocol, identities = async_capture()
    assert identities is not None
    capture = Capture(
        runtime="host_build_graph",
        level=4,
        clock_freq_hz=1_000_000_000,
        core_types=("aic", "aiv", "aiv") * 2,
        scheduler_producer="aicore",
    )
    by_execution = {(row["task_ref"], row["block_idx"], row["subtask_slot"]): row for row in protocol["executions"]}
    for identity in identities:
        row = next(row for row in protocol["executions"] if row["index"] == identity["reg_task_id"])
        capture.task(
            task_id=identity["task_id"],
            core_id=identity["core_id"],
            reg_task_id=identity["reg_task_id"],
            run_epoch=protocol["run_epoch"],
            start=row["entry"][0],
            end=row["visible"][0],
        )
    assert len(by_execution) == len(identities)
    raw = capture.build()
    raw["aicore_protocol"] = protocol
    raw["aicore_execution_identities"] = dict(schema="hbg-execution-identity-v1", producer="aicore", records=identities)
    return raw


def trace_for(raw):
    decoded = sc._decode_perf_data(raw)
    extra = {"native_async_view": decoded["native_async_view"]} if "native_async_view" in decoded else {}
    return sc.generate_chrome_trace_json(
        decoded["tasks"],
        None,
        runtime_name="host_build_graph",
        timeline_metadata=decoded.get("timeline_metadata"),
        **extra,
    ), decoded


def points(trace):
    return [event for event in trace["traceEvents"] if event.get("cat") == "hbg_async"]


def test_converter_validates_each_capture_once(monkeypatch):
    original = native_protocol.read_native_protocol
    calls = []

    def read(section):
        calls.append(section)
        return original(section)

    monkeypatch.setattr(native_protocol, "read_native_protocol", read)
    raw = wire_capture()
    trace, _ = trace_for(raw)
    assert points(trace)
    assert calls == [raw["aicore_protocol"]]


def test_standalone_projection_still_validates_structure():
    from simpler_setup.tools.hbg.async_lane import project_async_protocol  # noqa: PLC0415

    raw, _ = async_capture()
    raw["async"]["groups"].clear()
    with pytest.raises(ValueError, match="table length"):
        project_async_protocol(raw, lambda _worker, cycles: cycles)


def test_success_draws_all_recorded_boundaries_and_keeps_business_rows():
    raw = wire_capture()
    trace, decoded = trace_for(raw)
    rendered = points(trace)
    assert {event["name"] for event in rendered} == {
        "registration_sealed",
        "condition_ready",
        "execution_release.before",
        "execution_release.after",
        "logical_done.before",
        "logical_done.after",
        "resolve_drained",
    }
    assert all(event["ph"] == "i" and "dur" not in event for event in rendered)
    terminal = next(event for event in rendered if event["name"] == "condition_ready")
    assert terminal["args"]["capture_pending_count"] == 2
    assert terminal["args"]["capture_observations"] == 1
    assert terminal["args"]["capture_active_cycles"] == 17
    assert "owner_generation" not in terminal["args"]["conditions"][0]
    assert terminal["args"]["condition_types"] == ["COUNTER"]
    assert "acquire" in terminal["args"]["pending_count_meaning"]
    assert terminal["args"]["capture_lease_generation"] == 7
    assert terminal["args"]["dispatch_generation"] == 1
    sealed = next(event for event in rendered if event["name"] == "registration_sealed")
    assert sealed["args"]["capture_state"] == "READY"
    assert sealed["args"]["capture_pending_count"] == 2
    assert "state" not in sealed["args"] and "pending_count" not in sealed["args"]
    assert "not state or counter values at this endpoint" in sealed["args"]["snapshot_note"]
    assert trace["metadata"]["hbg_async"]["groups"][0]["state"] == "READY"
    assert len(decoded["tasks"]) == raw["aicore_protocol"]["trace_count"]
    assert trace["metadata"]["hbg_async"]["protocol_verified"]
    assert any(event.get("name") == "logical_wait_endpoints" and event["ph"] == "s" for event in trace["traceEvents"])


def test_cross_worker_clock_reversal_never_becomes_busy_or_elapsed_time():
    raw, identities = async_capture()
    group = raw["async"]["groups"][0]
    seal_worker = group["sealed"]["worker_index"]
    for record, kind, _ in _event_records(raw):
        if record["worker_index"] == seal_worker:
            for field in STAMPS[kind].split():
                if record[field][1]:
                    record[field][0] += 1_000_000
            if kind == "execution":
                record["publication_before_cycles"] += 1_000_000
                record["publication_after_cycles"] += 1_000_000
    trace, _ = trace_for(wire_capture(raw, identities))
    rendered = points(trace)
    assert rendered
    sealed = next(event for event in rendered if event["name"] == "registration_sealed")
    terminal = next(event for event in rendered if event["name"] == "condition_ready")
    assert terminal["ts"] < sealed["ts"]
    assert all("dur" not in event for event in rendered)
    assert not any(
        event.get("ph") == "X" for event in trace["traceEvents"] if event.get("cat", "").startswith("hbg_async")
    )


def test_empty_group_has_a_real_seal_but_no_invented_check_terminal():
    raw, identities = async_capture(empty=True)
    trace, _ = trace_for(wire_capture(raw, identities))
    rendered = points(trace)
    assert any(
        event["name"] == "registration_sealed" and event["args"]["capture_state"] == "EMPTY" for event in rendered
    )
    assert not any(event["name"] == "condition_ready" for event in rendered)
    assert not any(event.get("name") == "logical_wait_endpoints" for event in trace["traceEvents"])


def test_graph_public_ids_remain_distinct_from_internal_task_refs():
    raw, identities = native_capture()
    add_async(raw, identities, refs=(2,))
    trace, _ = trace_for(wire_capture(raw, identities))
    sealed = next(event for event in points(trace) if event["name"] == "registration_sealed")
    assert sealed["args"]["task_ref"] == 2
    assert sealed["args"]["task_id"] == (1 << 62) | (1 << 32)


def test_standard_converter_still_rejects_partial_and_failed_native_protocol():
    for mutation in ("partial", "failed"):
        raw = wire_capture()
        if mutation == "partial":
            raw["aicore_protocol"]["async"]["releases"][0]["publication"]["after"] = None
        else:
            raw["aicore_protocol"]["async"]["groups"][0].update(state="FAILED", terminal=None, failure_code=-7)
        resequence(raw["aicore_protocol"])
        with pytest.raises(ValueError):
            sc._decode_perf_data(raw)


def test_explicit_projection_preserves_partial_zero_and_null_evidence():
    from simpler_setup.tools.hbg.async_lane import project_async_protocol  # noqa: PLC0415

    raw, _ = async_capture()
    group = raw["async"]["groups"][0]
    group.update(state="FAILED", terminal=None, failure_code=-7)
    group["sealed"]["at"][0] = 0
    raw["async"]["releases"][0]["publication"]["after"] = None
    raw["async"]["finishes"][0].update(state="PENDING", drained=None)
    observed = []

    def convert(worker, cycles):
        observed.append((worker, cycles))
        return worker * 10 + cycles / 1000

    before = copy.deepcopy(raw)
    view = project_async_protocol(raw, convert)
    assert view is not None
    assert raw == before
    assert not view["summary"]["protocol_verified"]
    assert (group["sealed"]["worker_index"], 0) in observed
    names = [point["name"] for point in view["points"]]
    assert "registration_sealed" in names and "execution_release.before" in names
    assert "execution_release.after" not in names and "resolve_drained" not in names
    assert "condition_failed" not in names  # A failure code alone is not a timestamp.
    assert "owner_generation" not in view["summary"]["groups"][0]["conditions"][0]
    assert not view["links"]


def test_production_projection_converts_recorded_zero_without_a_missing_stamp_sentinel():
    raw, identities = async_capture()
    # This worker performs terminal/FINISH work but no business kernel. Legacy
    # aicore_tasks correctly forbids a zero kernel start, independently of these
    # new protocol stamps whose positive sequence proves they were recorded.
    worker = raw["async"]["finishes"][0]["drained"]["worker_index"]
    for record, kind, _ in _event_records(raw):
        if record["worker_index"] == worker:
            for field in STAMPS[kind].split():
                if record[field][1]:
                    record[field][0] = 0
            if kind == "execution":
                record["publication_before_cycles"] = 0
                record["publication_after_cycles"] = 0
    trace, _ = trace_for(wire_capture(raw, identities))
    drained = next(event for event in points(trace) if event["name"] == "resolve_drained")
    assert drained["args"]["cycles"] == 0 and drained["args"]["sequence"] > 0
    assert drained["ts"] < 0  # Converted relative to the nonzero capture origin, not fabricated zero.


def test_release_after_can_follow_remote_logical_done_without_a_false_flow_edge():
    raw, identities = async_capture()
    release = raw["async"]["releases"][0]["publication"]
    release["after"][0] = raw["task_events"][0]["completion"]["after"][0] + 500
    resequence(raw)
    trace, _ = trace_for(wire_capture(raw, identities))
    rendered = {event["name"]: event for event in points(trace)}
    assert rendered["execution_release.after"]["ts"] > rendered["logical_done.after"]["ts"]
    joining = [
        event
        for event in trace["traceEvents"]
        if event.get("name") == "release_lower_bound_to_logical_done_upper_bound"
    ]
    assert next(event for event in joining if event["ph"] == "s")["ts"] == rendered["execution_release.before"]["ts"]


def test_pending_is_aggregate_and_mix_release_is_once_per_logical_block():
    raw, identities = async_capture(mask=7, blocks=2)
    for group in raw["async"]["groups"]:
        group.update(pending_count=1_000_000, slice_count=1_000_001)
    trace, _ = trace_for(wire_capture(raw, identities))
    rendered = points(trace)
    assert len(rendered) == 6 * 2 + 2 * 2 + 3
    assert sum(event["name"] == "execution_release.before" for event in rendered) == 2
    terminals = [event for event in rendered if event["name"] == "condition_ready"]
    assert len(terminals) == 6 and all(event["args"]["capture_pending_count"] == 1_000_000 for event in terminals)


def test_historical_schema_keeps_the_existing_trace_without_async_lanes():
    trace, decoded = trace_for(historical_capture())
    assert "native_async_view" not in decoded
    assert not points(trace)
    assert "hbg_async" not in trace["metadata"]


def test_single_capture_cli_plumbs_async_view(tmp_path, monkeypatch):
    source, target = tmp_path / "capture.json", tmp_path / "trace.json"
    source.write_text(json.dumps(wire_capture()))
    monkeypatch.setattr(sc.sys, "argv", ["swimlane_converter", str(source), "-o", str(target)])
    assert sc.main() == 0
    assert len(points(json.loads(target.read_text()))) == 7


def test_final_l3_trace_plumbs_and_namespaces_async_endpoints(tmp_path):
    root = tmp_path / "capture"
    for rank in (0, 1):
        directory = _write_hbg_l3_rank(root, rank, host_shift_ns=rank * 1_000_000, task_id=0)
        source = directory / "chip_swimlane_records.json"
        raw = json.loads(source.read_text())
        asynchronous = wire_capture()
        for key in ("aicore_tasks", "scheduler_tasks", "aicore_protocol", "aicore_execution_identities"):
            raw[key] = asynchronous[key]
        raw["metadata"]["core_types"] = asynchronous["metadata"]["core_types"]
        source.write_text(json.dumps(raw))
        log = next(directory.glob("host_clock_alignment.*.log"))
        log.write_text(log.read_text().replace("dur=8000", "dur=800000").replace("dur=2000", "dur=200000"))
    target = tmp_path / "merged.json"
    args = sc._build_parser().parse_args(
        [
            str(root),
            "--dispatch",
            "d0",
            "-o",
            str(target),
            "--rank-pid",
            "0=1000:1",
            "--rank-pid",
            "1=1001:1",
        ]
    )
    sc._generate_l3_trace(args, root)
    trace = json.loads(target.read_text())
    rendered = points(trace)
    assert len(rendered) == 14
    assert {event["args"]["rank"] for event in rendered} == {0, 1}
    assert {event["pid"] for event in rendered} == {107, 207}
    assert all(rank["hbg_async"]["group_count"] == 1 for rank in trace["metadata"]["ranks"])
    assert all(rank["hbg_protocol_audit"]["protocol_verified"] for rank in trace["metadata"]["ranks"])
    flow_ids = {event["id"] for event in trace["traceEvents"] if event.get("cat") == "hbg_async_flow"}
    assert flow_ids and all(value.startswith(("r0:", "r1:")) for value in flow_ids)
    assert any(value.startswith("r0:") for value in flow_ids) and any(value.startswith("r1:") for value in flow_ids)
