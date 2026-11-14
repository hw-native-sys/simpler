# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Read native HBG protocol facts without inferring missing events or a global clock order.

Sequence integrity proves lossless local event streams, not scheduler causality,
execution coverage, membership, dependency ordering or Mix resource correctness.
"""

SCHEMAS = ("hbg-native-protocol-v22", "hbg-native-protocol-v23")
COMPLETION_FROM_CONTRIBUTION = 1 << 30
DEFERRED_FINISH = 1 << 31

ROOT_NUMBERS = frozenset(
    """
run_epoch task_count worker_count trace_count member_count cooperative_task_count
cluster_count mix_task_count sync_mix_task_count expected_task_count inline_completed_count resolved_task_count
close_seal_stride active_worker_count scheduler_error error_claimed commit_tail commit_lock
""".split()
)
ROOT_TABLES = frozenset(
    "tasks workers admissions admission_log task_controls task_events members blocks executions".split()
)
ERROR_NUMBERS = (
    "task_ref core_id core_type site run_epoch order_key block_idx group_seq worker_index cluster_lane identity_fields"
).split()

# Scalar fields are unsigned wire integers unless listed in SIGNED_FIELDS.
NUMBERS = {
    "task": "task_ref flags active_mask block_num trace_base",
    "worker": (
        "index worker_index context_worker_index run_epoch next_sequence "
        "error_flags finished core_id core_type cluster_index"
    ),
    "admission": (
        "index run_epoch task_ref order_key block_num member_count active_mask flags record_offset close_seals_offset"
    ),
    "control": (
        "task_ref state next_block completed_subtasks admission_state canonical_total pending_compute completion_debt"
    ),
    "claim": "run_epoch worker_index block_idx confirmed_count seal_value",
    "actor": "run_epoch task_ref order_key worker_index flags",
    "member": (
        "index run_epoch task_ref order_key observed_tail closed_cursor "
        "close_seal last_group_seq worker_index flags pending_mask"
    ),
    "owner": (
        "run_epoch task_ref order_key group_seq completion_ticket worker_index "
        "block_idx cluster_index block_num active_mask flags sync_tail"
    ),
    "lane": (
        "index run_epoch task_ref order_key group_seq worker_index block_idx "
        "cluster_index lane active_mask flags block_num resource_mask"
    ),
    "execution": (
        "index run_epoch task_ref order_key permit_tail worker_index block_idx "
        "pending_slot subtask_slot flags ready_source group_seq block_num cluster_index cluster_lane "
        "publication_before_cycles publication_after_cycles publication_count "
        "contribution_ticket contribution_worker"
    ),
}
STAMPS = {
    "claim": "published_before published_after",
    "actor": "before after",
    "member": "observed closed",
    "owner": (
        "staged claim_before claim_after publish_before publish_after "
        "completion_before completion_after releases_observed"
    ),
    "lane": "descriptor_ready arrive_before arrive_after arrivals_observed release_before release_after",
    "execution": (
        "entry returned visible completion_consumed permit trace_published contribution_before contribution_after"
    ),
}
ARRAYS = {
    "task": {"kernel_ids": 3, "dependencies": None},
    "worker": {"cluster_workers": 3},
    "admission": {"close_seals": None},
    "owner": {"released_seq": 3, "sync_frontier": 3, "sync_hold": 3},
    "lane": {"arrived_seq": 3, "lane_subtasks": 3},
}
SIGNED_FIELDS = frozenset(("task_ref", "core_id", "core_type", "state"))


def protocol_version(section):
    schema = section.get("schema") if isinstance(section, dict) else None
    return int(schema.rsplit("v", 1)[-1]) if schema in SCHEMAS else 0


def has_native_graph(section):
    """The native Graph schema includes independent source and node maps."""
    return isinstance(section, dict) and section.get("schema") == "hbg-native-protocol-v23"


def _error(message):
    raise ValueError(f"native protocol: {message}")


def _object(value, fields, path):
    if not isinstance(value, dict) or set(value) != set(fields):
        _error(f"{path} has missing or unknown fields")


def _number(value, path, *, signed=False):
    low, high = (-(1 << 63), 1 << 63) if signed else (0, 1 << 64)
    if type(value) is not int or not low <= value < high:
        _error(f"invalid integer at {path}")


def _array(value, path, count=None, *, signed=False):
    if not isinstance(value, list) or (count is not None and len(value) != count):
        _error(f"invalid array at {path}")
    for index, item in enumerate(value):
        _number(item, f"{path}[{index}]", signed=signed)


def _record(record, kind, path, extra_numbers=(), extra_arrays=None):
    numbers = NUMBERS[kind].split() + list(extra_numbers)
    stamps = STAMPS.get(kind, "").split()
    arrays = ARRAYS.get(kind, {}) | (extra_arrays or {})
    _object(record, {*numbers, *stamps, *arrays}, path)
    for name in numbers:
        _number(record[name], f"{path}.{name}", signed=name in SIGNED_FIELDS)
    for name in stamps:
        _array(record[name], f"{path}.{name}", 2)
    for name, count in arrays.items():
        _array(
            record[name],
            f"{path}.{name}",
            count,
            signed=name == "dependencies",
        )


def _indices(rows, count, path, *, key="index", dense=False):
    if not isinstance(rows, list) or (dense and len(rows) != count):
        _error(f"{path} table length mismatch")
    observed = set()
    for row in rows:
        if not isinstance(row, dict) or key not in row:
            _error(f"{path} row is missing {key}")
        index = row[key]
        _number(index, f"{path}.{key}")
        if index >= count or index in observed:
            _error(f"{path} duplicate or out-of-range {key}")
        observed.add(index)


def read_native_protocol(section):
    """Validate the wire schema; preserve incomplete/error records for diagnosis."""
    if not protocol_version(section):
        _error("unsupported schema")
    native = has_native_graph(section)
    root_numbers = ROOT_NUMBERS
    extra = {"node_map", "graph_source"} if native else set()
    _object(section, root_numbers | ROOT_TABLES | {"schema", "producer", "first_error"} | extra, "root")
    if section["producer"] != "aicore":
        _error("unknown schema or producer")
    for field in root_numbers:
        _number(section[field], field)
    if section["close_seal_stride"] != 64:
        _error("close_seal_stride disagrees with the schema version")
    error_numbers = ERROR_NUMBERS
    _object(section["first_error"], error_numbers, "first_error")
    for field in error_numbers:
        _number(section["first_error"][field], f"first_error.{field}")
    if section["member_count"] != section["worker_count"] * section["cooperative_task_count"]:
        _error("member region capacity disagrees with task/worker counts")
    _array(section["admission_log"], "admission_log", section["cooperative_task_count"], signed=True)
    for table, kind, count, key in (
        ("tasks", "task", section["task_count"], "task_ref"),
        ("workers", "worker", section["worker_count"], "index"),
        ("task_controls", "control", section["task_count"], "task_ref"),
        ("admissions", "admission", section["task_count"] if section["cooperative_task_count"] else 0, "index"),
    ):
        _indices(section[table], count, table, key=key, dense=True)
        for row in section[table]:
            extra = ("claim_tickets", "claim_seal", "parked_count") if kind == "control" else ()
            _record(row, kind, f"{table}[{row[key]}]", extra)
    for row in section["admissions"]:
        if len(row["close_seals"]) not in (0, section["cluster_count"]):
            _error("close seal array length mismatch")
    for table, kind, count in (
        ("members", "member", section["member_count"]),
        ("executions", "execution", section["trace_count"]),
    ):
        _indices(section[table], count, table)
        for row in section[table]:
            extra = ("claim_seal_seen", "claim_cursor_seen") if kind == "member" else ()
            _record(row, kind, f"{table}[{row['index']}]", extra)
    _read_nested_events(section)
    if native:
        from .native_graph import read_graph_source  # noqa: PLC0415

        read_graph_source(section)
    return section


def _read_nested_events(section):
    _indices(section["task_events"], section["task_count"], "task_events")
    for row in section["task_events"]:
        _object(row, {"index", "admission", "completion"}, "task_events row")
        for name in ("admission", "completion"):
            if row[name] is not None:
                _record(row[name], "actor", f"task_events[{row['index']}].{name}")
    _indices(section["blocks"], section["trace_count"], "blocks")
    for row in section["blocks"]:
        fields = {"index", "owner", "lanes", "claim"}
        _object(row, fields, "block row")
        if row["claim"] is not None:
            _record(row["claim"], "claim", f"blocks[{row['index']}].claim")
        if row["owner"] is not None:
            _record(row["owner"], "owner", f"blocks[{row['index']}].owner")
        _indices(row["lanes"], 3, "block lanes")
        for lane in row["lanes"]:
            extra = {"previous_lane_seq": 3}
            _record(lane, "lane", f"blocks[{row['index']}].lanes[{lane['index']}]", extra_arrays=extra)


def _event_records(section):
    for row in section["task_events"]:
        for name in ("admission", "completion"):
            if row[name] is not None:
                yield row[name], "actor", f"task_events[{row['index']}].{name}"
    for row in section["blocks"]:
        if row["claim"] is not None:
            yield row["claim"], "claim", f"blocks[{row['index']}].claim"
        if row["owner"] is not None:
            yield row["owner"], "owner", f"blocks[{row['index']}].owner"
        for lane in row["lanes"]:
            yield lane, "lane", f"blocks[{row['index']}].lanes[{lane['index']}]"
    for table, kind in (("members", "member"), ("executions", "execution")):
        for row in section[table]:
            yield row, kind, f"{table}[{row['index']}]"


def _completion_aliases(section):
    """Validate every shared completion interval before deduplicating its stamps."""
    tasks = {row["task_ref"]: row for row in section["tasks"]}
    controls = {row["task_ref"]: row for row in section["task_controls"]}
    contributions = {}
    for execution in section["executions"]:
        key = execution["task_ref"], execution["contribution_ticket"]
        contributions.setdefault(key, []).append(execution)
    aliases = {}
    for event in section["task_events"]:
        admission, done = event["admission"], event["completion"]
        if admission is not None and admission["flags"] & COMPLETION_FROM_CONTRIBUTION:
            _error("completion alias on admission")
        if done is None or not done["flags"] & COMPLETION_FROM_CONTRIBUTION:
            continue
        ref = event["index"]
        task, control = tasks[ref], controls[ref]
        total = task["block_num"] * task["active_mask"].bit_count()
        if (
            not task["flags"] & 1
            or task["flags"] & 32
            or not 0 < task["active_mask"] < 8
            or not 0 < total <= 32767
            or done["flags"] & DEFERRED_FINISH
            or done["flags"] != task["flags"] | COMPLETION_FROM_CONTRIBUTION
            or done["task_ref"] != ref
            or done["run_epoch"] != section["run_epoch"]
            or control["canonical_total"] != total
            or control["pending_compute"] != 0
            or control["completion_debt"] != 0
            or control["state"] != 2
        ):
            _error(f"invalid physical TaskDone completion alias for task {ref}")
        candidates = contributions.get((ref, total - 1), ())
        if len(candidates) != 1:
            _error(f"TaskDone completion alias lacks a unique last contribution for task {ref}")
        source = candidates[0]
        lane, block = source["subtask_slot"], source["block_idx"]
        if (
            lane >= 3
            or not task["active_mask"] & (1 << lane)
            or block >= task["block_num"]
            or source["block_num"] != task["block_num"]
            or source["index"]
            != task["trace_base"]
            + block * task["active_mask"].bit_count()
            + (task["active_mask"] & ((1 << lane) - 1)).bit_count()
            or source["run_epoch"] != done["run_epoch"]
            or source["worker_index"] != done["worker_index"]
            or source["contribution_worker"] != done["worker_index"]
            or source["order_key"] != done["order_key"]
            or done["before"] != source["contribution_before"]
            or done["after"] != source["contribution_after"]
            or not 0 < done["before"][1] < done["after"][1]
        ):
            _error(f"TaskDone completion alias does not identify its last contribution for task {ref}")
        aliases[ref] = source
    return aliases


def audit_protocol_sequences(section):
    """Check producer terminal status and contiguous per-worker sequences, not full causality."""
    read_native_protocol(section)
    if section["scheduler_error"] != 0 or section["error_claimed"] != 0:
        _error("scheduler reported an error")
    _completion_aliases(section)
    workers = {row["index"]: row for row in section["workers"]}
    streams = {index: {} for index in workers}
    epoch = section["run_epoch"]
    for index, worker in workers.items():
        if (
            worker["worker_index"] != index
            or worker["context_worker_index"] != index
            or worker["run_epoch"] != epoch
            or worker["error_flags"] != 0
            or worker["finished"] != 1
        ):
            _error(f"worker {index} identity, error or terminal status is invalid")
    for record, kind, path in _event_records(section):
        worker = record["worker_index"]
        if worker not in workers or record["run_epoch"] != epoch:
            _error(f"foreign worker or epoch at {path}")
        if kind == "actor" and record["flags"] & COMPLETION_FROM_CONTRIBUTION:
            # The validated physical contribution contributes these events once.
            continue
        stream = streams[worker]
        for name in STAMPS[kind].split():
            cycles, sequence = record[name]
            if sequence == 0:
                if cycles != 0:
                    _error(f"partial stamp at {path}.{name}")
                continue
            if sequence in stream:
                _error(f"worker {worker} duplicate sequence {sequence} at {path}.{name}")
            stream[sequence] = (cycles, f"{path}.{name}")
    for index, stream in streams.items():
        expected = workers[index]["next_sequence"]
        if len(stream) != expected:
            _error(f"worker {index} lost or unexpected events: {len(stream)} != {expected}")
        previous_cycles = 0
        for ordinal, (sequence, (cycles, path)) in enumerate(sorted(stream.items()), 1):
            if sequence != ordinal or cycles < previous_cycles:
                _error(f"worker {index} sequence hole or backwards clock at {path}")
            previous_cycles = cycles
    return {
        "sequence_integrity_verified": True,
        "event_count": sum(map(len, streams.values())),
        "worker_count": len(workers),
    }
