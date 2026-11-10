# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Independent native Graph expansion and control-event causality."""

import copy

import pytest
from _native_protocol_builder import ProtocolBuilder

from simpler_setup.tools.hbg.native_causality import audit_native_protocol
from simpler_setup.tools.hbg.native_graph import audit_graph_expansion, export_kernel_dependencies
from simpler_setup.tools.hbg.native_protocol import read_native_protocol
from simpler_setup.tools.sched_overhead_analysis import build_task_graph


def native_capture(version=9, *, deferred_tasks=()):
    # Outer kernel -> Graph(kernel -> dummy) -> outer consumer.
    raw, identities = ProtocolBuilder(
        [(2, 1, ()), (0, 1, (0,)), (2, 1, (1,)), (0, 1, (2,)), (0, 1, (3,)), (2, 1, (4,))],
        deferred_tasks=deferred_tasks,
    ).run()
    raw["schema"] = "hbg-native-protocol-v6"
    for control in raw["task_controls"]:
        control.update(claim_tickets=0, admission_claim_lock=0)
    raw["graph_source"] = dict(
        tasks=[
            dict(task_id=0, kind=0, graph_key=0, dependencies=[]),
            dict(task_id=1, kind=1, graph_key=11, dependencies=[0]),
            dict(task_id=2, kind=0, graph_key=0, dependencies=[1]),
        ],
        definitions=[
            dict(
                graph_key=11,
                children=[
                    dict(
                        active_mask=2,
                        block_num=1,
                        task_attrs=0,
                        predicate_slot=0,
                        kernel_ids=[-1, 0, -1],
                        dependencies=[],
                    ),
                    dict(
                        active_mask=0,
                        block_num=1,
                        task_attrs=0,
                        predicate_slot=0,
                        kernel_ids=[-1, -1, -1],
                        dependencies=[0],
                    ),
                ],
            )
        ],
    )
    raw["node_map"] = []
    for ref, kind, original, local in [
        (0, 0, 0, None),
        (1, 3, 1, None),
        (2, 1, 1, 0),
        (3, 2, 1, 1),
        (4, 4, 1, None),
        (5, 0, 2, None),
    ]:
        public = original if local is None else (1 << 62) | (original << 32) | local
        raw["node_map"].append(
            dict(
                node_ref=ref,
                kind=kind,
                original_task_id=original,
                local_child_id=(1 << 32) - 1 if local is None else local,
                public_task_id=public,
            )
        )
        if kind in (2, 3, 4):
            raw["tasks"][ref].update(flags=2, active_mask=0)
            next(event for event in raw["task_events"] if event["index"] == ref)["completion"]["flags"] = 2
    for identity in identities:
        identity["task_id"] = raw["node_map"][identity["task_id"]]["public_task_id"]
    if version == 9:
        raw.update(schema="hbg-native-protocol-v9", close_seal_stride=128)
        for control in raw["task_controls"]:
            control["claim_seal"] = 0
        for member in raw["members"]:
            member.update(claim_seal_seen=0, claim_cursor_seen=0)
        for block in raw["blocks"]:
            block["claim"] = None
            for lane in block["lanes"]:
                lane["previous_lane_seq"] = [0, 0, 0]
    return raw, identities


@pytest.mark.parametrize("version", (6, 9))
def test_native_graph_full_causality_and_public_identity_join(version):
    raw, identities = native_capture(version)
    result = audit_native_protocol(raw, identities)
    assert result["protocol_verified"] and result["device_logical_tasks"] == 6
    assert result["executions"] == 3


@pytest.mark.parametrize("version", (6, 9))
@pytest.mark.parametrize("task_ref", (0, 2, 5))
def test_native_graph_deferred_finish_preserves_public_identity_and_dependencies(version, task_ref):
    raw, identities = native_capture(version, deferred_tasks=(task_ref,))
    result = audit_native_protocol(raw, identities)
    assert result["protocol_verified"]
    assert result["device_logical_tasks"] == 6 and result["executions"] == 3
    completion = next(event for event in raw["task_events"] if event["index"] == task_ref)["completion"]
    completion["flags"] &= ~(1 << 31)
    with pytest.raises(ValueError, match="TaskDone"):
        audit_native_protocol(raw, identities)


@pytest.mark.parametrize("field", ("graph_source", "node_map", "close_seal_stride"))
def test_native_graph_v9_requires_graph_and_scheduler_contracts(field):
    raw, _ = native_capture()
    del raw[field]
    with pytest.raises(ValueError, match="missing or unknown fields"):
        read_native_protocol(raw)


def test_legacy_native_graph_rejects_claim_seal_fields():
    raw, _ = native_capture(6)
    raw["task_controls"][0]["claim_seal"] = 0
    with pytest.raises(ValueError, match="missing or unknown fields"):
        read_native_protocol(raw)


@pytest.mark.parametrize("ref,deps", [(1, []), (2, []), (3, [1]), (4, [2]), (5, [1])])
def test_independent_source_rejects_missing_enter_sink_and_external_mapping(ref, deps):
    raw, _ = native_capture()
    raw["tasks"][ref]["dependencies"] = deps
    with pytest.raises(ValueError, match="original source"):
        audit_graph_expansion(raw)


@pytest.mark.parametrize("field", ["kind", "public_task_id", "original_task_id", "local_child_id"])
def test_node_mapping_cannot_alias_a_different_instance_or_control(field):
    raw, _ = native_capture()
    raw["node_map"][2][field] += 1
    with pytest.raises(ValueError, match="identity or node kind"):
        audit_graph_expansion(raw)


def test_original_definition_changes_are_observed_without_changing_planner_output():
    raw, _ = native_capture()
    raw["graph_source"]["definitions"][0]["children"][1]["dependencies"] = []
    with pytest.raises(ValueError, match="original source"):
        audit_graph_expansion(raw)


def test_missing_control_completion_is_not_reported_as_host_done():
    raw, identities = native_capture()
    raw["task_events"] = [event for event in raw["task_events"] if event["index"] != 4]
    with pytest.raises(ValueError):
        audit_native_protocol(raw, identities)


def test_native_schema_requires_explicit_mapping_and_source():
    raw, _ = native_capture()
    for field in ("node_map", "graph_source"):
        incomplete = copy.deepcopy(raw)
        del incomplete[field]
        with pytest.raises(ValueError, match="missing or unknown"):
            read_native_protocol(incomplete)


def test_kernel_dependencies_cross_graph_entry_dummy_and_exit():
    raw, identities = native_capture()
    child = (1 << 62) | (1 << 32)
    result = export_kernel_dependencies(raw, {row["task_id"] for row in identities})
    assert {int(row["task_id"]) for row in result["tasks"]} == {0, child, 2}
    assert {(int(row["pred"]), int(row["succ"])) for row in result["edges"]} == {(0, child), (child, 2)}
    assert result["tasks"][1]["kernel_ids"] == [-1, 0, -1]


def test_kernel_dependencies_forward_through_an_unexecuted_predicated_child():
    raw, _ = native_capture()
    raw["graph_source"]["definitions"][0]["children"][0]["predicate_slot"] = 1
    raw["tasks"][2]["flags"] |= 64
    result = export_kernel_dependencies(raw, {0, 2})
    assert [(row["pred"], row["succ"]) for row in result["edges"]] == [("0", "2")]


def test_kernel_dependencies_reject_unknown_execution_and_unpredicated_holes():
    raw, _ = native_capture()
    with pytest.raises(ValueError, match="execution"):
        export_kernel_dependencies(raw, {999})
    with pytest.raises(ValueError, match="unexecuted"):
        export_kernel_dependencies(raw, {0, 2})


def test_kernel_dependencies_reject_corrupt_source_edges():
    raw, identities = native_capture()
    raw["tasks"][4]["dependencies"] = []
    with pytest.raises(ValueError, match="original source"):
        export_kernel_dependencies(raw, {row["task_id"] for row in identities})


def test_kernel_dependency_projection_gives_outer_consumer_its_child_ready_time():
    raw, identities = native_capture()
    child = (1 << 62) | (1 << 32)
    observed = [
        dict(task_id=0, end_time_us=3, core_type="aiv"),
        dict(task_id=child, end_time_us=11, core_type="aiv"),
        dict(task_id=2, end_time_us=15, core_type="aiv"),
    ]
    outer_only = dict(edges=[dict(pred="0", succ="1"), dict(pred="1", succ="2")])
    assert build_task_graph(observed, outer_only, 0)[0][2] == 0
    expanded = export_kernel_dependencies(raw, {row["task_id"] for row in identities})
    ready, _, _, _, _ = build_task_graph(observed, expanded, 0)
    assert ready == {0: 0, child: 3, 2: 11}


def test_kernel_dependencies_preserve_all_graph_roots_and_sinks():
    raw, _ = native_capture()
    children = raw["graph_source"]["definitions"][0]["children"]
    children[1] = copy.deepcopy(children[0])
    raw["tasks"][3] = dict(raw["tasks"][2], task_ref=3)
    raw["tasks"][4]["dependencies"] = [2, 3]
    raw["node_map"][3]["kind"] = 1
    first = (1 << 62) | (1 << 32)
    result = export_kernel_dependencies(raw, {0, first, first + 1, 2})
    assert {(int(row["pred"]), int(row["succ"])) for row in result["edges"]} == {
        (0, first),
        (0, first + 1),
        (first, 2),
        (first + 1, 2),
    }
