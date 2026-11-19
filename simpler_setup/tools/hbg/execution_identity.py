# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Join device-captured HBG execution identities; never infer them from row order."""

FIELDS = frozenset(
    (
        "run_epoch",
        "core_id",
        "reg_task_id",
        "task_id",
        "block_idx",
        "block_num",
        "subtask_slot",
        "pending_slot",
        "order_key",
        "group_seq",
        "group_kind",
        "cluster_index",
        "cluster_lane",
        "executed",
    )
)
JOIN = ("run_epoch", "core_id", "reg_task_id")


def valid_source_index(index, group_kind):
    if group_kind == 0:
        # The trace reserves slots 0 through 3 for ordinary work and omits the runtime window depth.
        return 0 <= index < 4
    return group_kind == 2 and index in (4, 5)


def _validate_record(record, *, legacy=False):
    if not isinstance(record, dict) or set(record) != FIELDS:
        raise ValueError("execution identity has missing or unknown fields")
    for field in FIELDS - {"order_key", "group_seq", "executed"}:
        if type(record[field]) is not int or not 0 <= record[field] < 1 << 64:
            raise ValueError(f"invalid execution identity {field}")
    for field in ("order_key", "group_seq"):
        value = record[field]
        minimum = 1 if field == "group_seq" else 0
        if value is not None and (type(value) is not int or not minimum <= value < 1 << 64):
            raise ValueError(f"invalid execution identity {field}")
    group_kind = record["group_kind"]
    rank, generation = record["order_key"], record["group_seq"]
    if legacy:
        # The original two pending slots carried unranked ordinary tasks, ordered
        # groups (kind 1), or unranked cluster groups (kind 2).
        valid_source = group_kind == 0 or (group_kind == 1 and rank is not None) or (group_kind == 2 and rank is None)
        valid_slot = record["pending_slot"] < 2
    else:
        valid_source = group_kind in (0, 2) and rank is not None
        valid_slot = valid_source_index(record["pending_slot"], group_kind)
    if not valid_source or (group_kind == 0) != (generation is None) or (group_kind == 2 and generation % 2 != 1):
        raise ValueError("execution identity group source mismatch")
    if (
        type(record["executed"]) is not bool
        or not 0 <= record["block_idx"] < record["block_num"] <= 32767
        or record["subtask_slot"] > 2
        or not valid_slot
        or record["cluster_lane"] > 2
    ):
        raise ValueError("invalid execution identity shape")


def attach_execution_identities(section, tasks, runtime):
    if (
        runtime != "host_build_graph"
        or not isinstance(section, dict)
        or set(section) not in ({"schema", "producer", "records"}, {"schema", "version", "producer", "records"})
        or ("version" in section and (type(section["version"]) is not int or section["version"] != 2))
        or section["schema"] != "hbg-execution-identity"
        or section["producer"] != "aicore"
        or not isinstance(section["records"], list)
    ):
        raise ValueError("invalid execution identity envelope")
    # Before the ranked exporter gained a version tag, both layouts used this
    # envelope. A null rank or an ordered-group source identifies the old layout.
    legacy = "version" not in section and any(
        isinstance(row, dict) and (row.get("order_key") is None or row.get("group_kind") == 1)
        for row in section["records"]
    )
    rows = {tuple(task[field] for field in JOIN): task for task in tasks}
    if len(rows) != len(tasks):
        raise ValueError("duplicate execution identity row join")
    observed, executions = set(), set()
    positions, orders, groups, block_counts = {}, {}, {}, {}
    for record in section["records"]:
        _validate_record(record, legacy=legacy)
        key = tuple(record[field] for field in JOIN)
        task = rows.get(key)
        if key in observed or task is None or task["task_id"] != record["task_id"]:
            raise ValueError("missing, duplicate or mismatched execution identity join")
        is_cube = task["core_type"] == "aic"
        if (record["subtask_slot"] == 0) != is_cube or (record["cluster_lane"] == 0) != is_cube:
            raise ValueError("execution identity lane/type mismatch")
        if record["group_seq"] is not None and record["subtask_slot"] != record["cluster_lane"]:
            raise ValueError("execution identity Mix placement mismatch")
        execution = (record["run_epoch"], record["task_id"], record["block_idx"], record["subtask_slot"])
        if execution in executions:
            raise ValueError("duplicate execution identity")
        position_key = (record["run_epoch"], record["core_id"])
        position = (record["cluster_index"], record["cluster_lane"])
        if positions.setdefault(position_key, position) != position:
            raise ValueError("execution identity worker changed physical position")
        order_key = (record["run_epoch"], record["task_id"])
        if orders.setdefault(order_key, record["order_key"]) != record["order_key"]:
            raise ValueError("execution identity task has inconsistent schedule rank")
        group_key = execution[:3]
        group = (
            record["group_seq"],
            record["cluster_index"] if record["group_seq"] is not None else None,
            record["group_kind"],
            record["pending_slot"] if not legacy and record["group_seq"] is not None else None,
        )
        if groups.setdefault(group_key, group) != group:
            raise ValueError("execution identity Mix block spans different groups")
        block_counts[group_key] = block_counts.get(group_key, 0) + 1
        observed.add(key)
        executions.add(execution)
        task.update({field: record[field] for field in FIELDS - set(JOIN) - {"task_id"}})
    if observed != rows.keys():
        raise ValueError("execution identity coverage is incomplete")
    if any(count > 1 and groups[key][0] is None for key, count in block_counts.items()):
        raise ValueError("execution identity multi-lane block has no group")
