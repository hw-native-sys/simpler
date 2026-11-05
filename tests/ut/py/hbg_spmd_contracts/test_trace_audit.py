# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent trace-oracle checks, including invalid synthetic producers."""

import copy

import pytest

from .helpers import make_protocol, run_task
from .trace_audit import SCHEMA, audit_trace
from .types import ContractViolation, Task


def sample(mask=7, blocks=2):
    model = make_protocol(Task(0, blocks, mask), Task(1, 1, 2, deps=(0,)))
    run_task(model, 0)
    run_task(model, 1)
    payload = {
        "schema": SCHEMA,
        "epoch": model.epoch,
        "producer": "reference",
        "complete": True,
        "dropped": 0,
        "events": copy.deepcopy(model.events),
    }
    return model, payload


def audit(model, payload):
    return audit_trace(
        model.topology,
        tuple(s.spec for s in model.tasks.values()),
        payload,
        expected_epoch=model.epoch,
        expected_producer="reference",
    )


def renumber(payload):
    for tick, event in enumerate(payload["events"], 1):
        event["tick"] = tick


def first_event(payload, kind):
    return next(event for event in payload["events"] if event["kind"] == kind)


@pytest.mark.parametrize("mask", [1, 2, 3, 5, 6, 7])
@pytest.mark.parametrize("blocks", [1, 2, 7, 33])
def test_reference_traces_satisfy_an_independent_oracle(mask, blocks):
    model, payload = sample(mask, blocks)
    assert audit(model, payload) == {"tasks": 2, "blocks": blocks + 1, "executions": blocks * bin(mask).count("1") + 1}


@pytest.mark.parametrize(
    "kind",
    [
        "admit",
        "claim",
        "publish",
        "arrive",
        "start",
        "return",
        "visible",
        "retire",
        "block_done",
        "task_done",
        "cluster_close",
        "local_done",
    ],
)
def test_missing_required_event_never_silently_passes(kind):
    model, payload = sample()
    payload["events"].remove(first_event(payload, kind))
    renumber(payload)
    with pytest.raises(ContractViolation):
        audit(model, payload)


@pytest.mark.parametrize(
    "kind,code",
    [
        ("admit", "TRACE_ADMISSION"),
        ("claim", "TRACE_DUPLICATE_CLAIM"),
        ("publish", "TRACE_PUBLICATION"),
        ("arrive", "TRACE_ARRIVAL"),
        ("start", "TRACE_DUPLICATE_EXECUTION"),
        ("return", "TRACE_NOT_RUNNING"),
        ("visible", "TRACE_VISIBILITY"),
        ("retire", "TRACE_DUPLICATE_COMPLETION"),
        ("block_done", "TRACE_DUPLICATE_BLOCK"),
        ("task_done", "TRACE_DUPLICATE_TASK"),
        ("cluster_close", "TRACE_CLUSTER"),
        ("local_done", "TRACE_LOCAL_CLOSED"),
    ],
)
def test_duplicate_event_is_rejected_for_its_protocol_reason_not_timestamp(kind, code):
    model, payload = sample()
    event = first_event(payload, kind)
    payload["events"].insert(payload["events"].index(event) + 1, copy.deepcopy(event))
    renumber(payload)
    with pytest.raises(ContractViolation, match=code):
        audit(model, payload)


@pytest.mark.parametrize(
    "kind,field,value,code",
    [
        ("claim", "generation", 0, "TRACE_GENERATION"),
        ("claim", "generation", 1 << 32, "TRACE_GENERATION"),
        ("claim", "owner", 999, "TRACE_MEMBER"),
        ("claim", "block", -1, "TRACE_BLOCK"),
        ("claim", "block", 2, "TRACE_BLOCK"),
        ("admit", "order", 1, "TRACE_ADMISSION_GAP"),
        ("admit", "members", [999], "MEMBERS"),
        ("publish", "order", 99, "TRACE_ORDER"),
        ("start", "generation", 999, "TRACE_GENERATION"),
        ("start", "slot", 2, "TRACE_CONTEXT"),
        ("start", "block_num", 1, "TRACE_CONTEXT"),
        ("start", "worker", 999, "TRACE_EXECUTION_MEMBER"),
        ("start", "source", "unknown", "TRACE_SOURCE"),
        ("retire", "communication_done", False, "TRACE_VISIBILITY"),
        ("return", "epoch", 99, "TRACE_EPOCH"),
        ("return", "task", 99, "TRACE_TASK"),
    ],
)
def test_corrupt_identity_context_or_publication_is_rejected(kind, field, value, code):
    model, payload = sample()
    first_event(payload, kind)[field] = value
    with pytest.raises(ContractViolation, match=code):
        audit(model, payload)


@pytest.mark.parametrize(
    "mapping",
    [
        [(0, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 1), (2, 1)],
        [(0, 0), (1, 4), (2, 2)],
        [(0, 0), (1, 2), (2, 1)],
        [(0, 1), (1, 0), (2, 2)],
    ],
)
def test_mix_mapping_corruption_is_rejected(mapping):
    model, payload = sample()
    first_event(payload, "claim")["mapping"] = mapping
    with pytest.raises(ContractViolation, match="TRACE_MAPPING"):
        audit(model, payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "legacy-swimlane"),
        ("complete", False),
        ("dropped", 1),
        ("producer", "aicpu"),
        ("epoch", 2),
    ],
)
def test_header_must_identify_a_complete_current_run(field, value):
    model, payload = sample()
    payload[field] = value
    with pytest.raises(ContractViolation, match="TRACE_SCHEMA|TRACE_PRODUCER|TRACE_EPOCH"):
        audit(model, payload)


def test_default_gate_rejects_reference_data_as_hardware_evidence():
    model, payload = sample()
    with pytest.raises(ContractViolation, match="TRACE_PRODUCER"):
        audit_trace(model.topology, [s.spec for s in model.tasks.values()], payload, model.epoch)


@pytest.mark.parametrize("field", ["schema", "epoch", "producer", "complete", "dropped", "events"])
def test_missing_header_fields_are_not_inferred(field):
    model, payload = sample()
    del payload[field]
    with pytest.raises(ContractViolation, match="TRACE_SCHEMA"):
        audit(model, payload)


@pytest.mark.parametrize("field", ["generation", "block_num", "slot", "order", "worker", "source"])
def test_missing_execution_fields_are_not_inferred(field):
    model, payload = sample()
    del first_event(payload, "start")[field]
    with pytest.raises(ContractViolation, match="TRACE_SCHEMA"):
        audit(model, payload)


def test_audit_uses_external_expected_task_shape_not_a_producers_self_report():
    model, payload = sample()
    wrong_plan = (Task(0, 1, 7), Task(1, 1, 2, deps=(0,)))
    with pytest.raises(ContractViolation, match="TRACE_CONTEXT|TRACE_BLOCK"):
        audit_trace(model.topology, wrong_plan, payload, model.epoch, "reference")


def test_task_done_cannot_move_ahead_of_the_last_block():
    model, payload = sample()
    done = first_event(payload, "task_done")
    payload["events"].remove(done)
    first_block = first_event(payload, "block_done")
    payload["events"].insert(payload["events"].index(first_block) + 1, done)
    renumber(payload)
    with pytest.raises(ContractViolation, match="TRACE_EARLY_DONE"):
        audit(model, payload)


def test_task_done_does_not_replace_all_local_closes():
    model, payload = sample()
    payload["events"] = [e for e in payload["events"] if e["kind"] != "local_done" or e["task"] != 1]
    renumber(payload)
    with pytest.raises(ContractViolation, match="TRACE_INCOMPLETE"):
        audit(model, payload)


def test_unknown_event_and_backward_clock_are_rejected():
    model, payload = sample()
    payload["events"][0]["kind"] = "invented"
    with pytest.raises(ContractViolation, match="TRACE_EVENT"):
        audit(model, payload)
    model, payload = sample()
    payload["events"][1]["tick"] = payload["events"][0]["tick"]
    with pytest.raises(ContractViolation, match="TRACE_TIME"):
        audit(model, payload)


@pytest.mark.parametrize(
    "kind,field,value",
    [
        ("admit", "task", False),
        ("admit", "epoch", True),
        ("admit", "order", False),
        ("admit", "members", [False, 1, 2, 3, 4, 5]),
        ("admit", "members", [0, 1, 2, 3, 4, 5, 5]),
        ("claim", "owner", False),
        ("claim", "resources", [False, 1, 2]),
        ("claim", "resources", [0, 1, 2, 2]),
        ("claim", "mapping", [[False, 0], [1, 1], [2, 2]]),
        ("start", "worker", False),
        ("start", "slot", False),
        ("start", "generation", True),
        ("start", "block_num", 2.0),
        ("cluster_close", "cluster", False),
    ],
)
def test_boolean_numeric_aliases_and_duplicate_members_are_not_valid_trace_fields(kind, field, value):
    model, payload = sample()
    first_event(payload, kind)[field] = value
    with pytest.raises(ContractViolation):
        audit(model, payload)


def test_dropped_count_must_be_an_integer_not_a_boolean():
    model, payload = sample()
    payload["dropped"] = False
    with pytest.raises(ContractViolation, match="TRACE_SCHEMA"):
        audit(model, payload)


def test_service_node_cannot_fabricate_a_kernel_block_completion():
    model = make_protocol(Task(0, 1, 0, service=True))
    run_task(model, 0)
    payload = {
        "schema": SCHEMA,
        "epoch": 1,
        "producer": "reference",
        "complete": True,
        "dropped": 0,
        "events": [{"kind": "block_done", "epoch": 1, "task": 0, "block": 0}, *model.events],
    }
    renumber(payload)
    with pytest.raises(ContractViolation, match="TRACE_UNCLAIMED"):
        audit(model, payload)
