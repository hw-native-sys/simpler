# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Async wire evidence, diagnostic summaries and publication/join causality.

Group pending_count counts PENDING service slices, including budget yields.
FINISH pending_count is the current unpublished batch length. Active cycles sum
work within individual slices; seal/terminal timestamps are not a busy interval.
"""

from collections import Counter, defaultdict

from .native_protocol import _array, _error, _indices, _number, _object, has_async_protocol

GROUP_NUMBERS = """
index run_epoch task_ref block_idx subtask_slot execution_index dispatch_generation lease_word
capacity registered_count observations pending_count slice_count active_cycles failure_code
""".split()
FINISH_NUMBERS = """
index run_epoch task_ref lease_word waiter_count pending_count slice_count active_cycles failure_code
""".split()


def _scalars(row, fields, path):
    for field in fields:
        _number(row[field], f"{path}.{field}", signed=field in ("task_ref", "failure_code"))


def _evidence(value, path, *, interval=False):
    if value is None:
        return
    stamps = ("before", "after") if interval else ("at",)
    _object(value, {"run_epoch", "worker_index", *stamps}, path)
    _scalars(value, ("run_epoch", "worker_index"), path)
    for field in stamps:
        if interval and field == "after" and value[field] is None:
            continue
        _array(value[field], f"{path}.{field}", 2)
        if value[field][1] == 0:
            _error(f"{path}.{field} requires recorded evidence; use null when unavailable")


def read_async_protocol(section):
    """Validate exact shapes while retaining failed and incomplete diagnostic state."""
    _object(section, {"group_count", "finish_count", "groups", "releases", "finishes"}, "async")
    _scalars(section, ("group_count", "finish_count"), "async")
    _indices(section["groups"], section["group_count"], "async.groups", dense=True)
    _indices(section["finishes"], section["finish_count"], "async.finishes", dense=True)
    for row in section["groups"]:
        path = f"async.groups[{row['index']}]"
        _object(row, {*GROUP_NUMBERS, "conditions", "state", "sealed", "terminal"}, path)
        _scalars(row, GROUP_NUMBERS, path)
        if row["state"] not in ("UNSEALED", "EMPTY", "PENDING", "READY", "FAILED"):
            _error(f"invalid state at {path}")
        _indices(row["conditions"], row["registered_count"], f"{path}.conditions", dense=True)
        for condition in row["conditions"]:
            name = f"{path}.conditions[{condition['index']}]"
            _object(condition, {"index", "type", "binding_index", "owner_generation"}, name)
            _scalars(condition, ("index", "type"), name)
            if condition["binding_index"] is not None:
                _number(condition["binding_index"], f"{name}.binding_index")
            generation = condition["owner_generation"]
            if generation is not None:
                _number(generation, f"{name}.owner_generation")
                if generation == 0:
                    _error(f"{name}.owner_generation must be positive or null")
        _evidence(row["sealed"], f"{path}.sealed")
        _evidence(row["terminal"], f"{path}.terminal")
    if not isinstance(section["releases"], list):
        _error("invalid async.releases table")
    for row in section["releases"]:
        _object(row, {"task_ref", "block_idx", "run_epoch", "publication"}, "async.release")
        _scalars(row, ("task_ref", "block_idx", "run_epoch"), "async.release")
        _evidence(row["publication"], "async.release.publication", interval=True)
    for row in section["finishes"]:
        path = f"async.finishes[{row['index']}]"
        _object(row, {*FINISH_NUMBERS, "state", "group_indices", "drained"}, path)
        _scalars(row, FINISH_NUMBERS, path)
        if row["pending_count"] > 8:
            _error(f"{path}.pending_count out of range: {row['pending_count']} (expected 0..8)")
        if row["state"] not in ("PENDING", "DRAINED", "FAILED"):
            _error(f"invalid state at {path}")
        _array(row["group_indices"], f"{path}.group_indices")
        _evidence(row["drained"], f"{path}.drained")
    return section


def async_event_records(section):
    for row in section["groups"]:
        for field in ("sealed", "terminal"):
            if row[field] is not None:
                yield row[field], "async_point", f"async.groups[{row['index']}].{field}"
    for index, row in enumerate(section["releases"]):
        if row["publication"] is not None:
            publication = row["publication"]
            path = f"async.releases[{index}].publication"
            if publication["after"] is None:
                yield (
                    dict(
                        run_epoch=publication["run_epoch"],
                        worker_index=publication["worker_index"],
                        at=publication["before"],
                    ),
                    "async_point",
                    f"{path}.before",
                )
            else:
                yield publication, "async_interval", path
    for row in section["finishes"]:
        if row["drained"] is not None:
            yield row["drained"], "async_point", f"async.finishes[{row['index']}].drained"


def summarize_async_protocol(section):
    """Return observed counters and evidence availability, without elapsed-time inference."""
    from .native_protocol import read_native_protocol  # noqa: PLC0415

    read_native_protocol(section)
    if not has_async_protocol(section):
        return None
    raw = section["async"]
    public = {row["node_ref"]: row["public_task_id"] for row in section.get("node_map", [])}
    groups = []
    for row in raw["groups"]:
        groups.append(
            dict(
                index=row["index"],
                task_ref=row["task_ref"],
                task_id=public.get(row["task_ref"], row["task_ref"]),
                state=row["state"],
                registration_sealed=row["sealed"] is not None,
                terminal_recorded=row["terminal"] is not None,
                observations=row["observations"],
                pending_count=row["pending_count"],
                slice_count=row["slice_count"],
                active_cycles=row["active_cycles"],
                failure_code=row["failure_code"],
            )
        )
    finishes = [
        dict(
            index=row["index"],
            task_ref=row["task_ref"],
            task_id=public.get(row["task_ref"], row["task_ref"]),
            state=row["state"],
            resolve_drained=row["drained"] is not None,
            waiter_count=row["waiter_count"],
            pending_count=row["pending_count"],
            slice_count=row["slice_count"],
            active_cycles=row["active_cycles"],
            failure_code=row["failure_code"],
        )
        for row in raw["finishes"]
    ]
    return dict(
        group_count=raw["group_count"],
        finish_count=raw["finish_count"],
        group_states=dict(Counter(row["state"] for row in raw["groups"])),
        finish_states=dict(Counter(row["state"] for row in raw["finishes"])),
        execution_releases_recorded=sum(
            row["publication"] is not None and row["publication"]["after"] is not None for row in raw["releases"]
        ),
        execution_release_intervals_incomplete=sum(
            row["publication"] is not None and row["publication"]["after"] is None for row in raw["releases"]
        ),
        resolve_drained_recorded=sum(row["drained"] is not None for row in raw["finishes"]),
        groups=groups,
        finishes=finishes,
    )


def _require(condition, message):
    if not condition:
        _error(f"async {message}")


def _point(record):
    _require(record is not None, "missing recorded point")
    return record["worker_index"], *record["at"]


def _lease(row, phase):
    word = row["lease_word"]
    _require(word >> 3 > 0 and word & 7 == phase, "invalid terminal lease")
    if phase == 0:
        _require(word == 8, "invalid dormant lease generation")


def audit_async_protocol(audit):
    """Join READY/EMPTY and actual block contributions before TaskDone and FINISH drain."""
    if not has_async_protocol(audit.raw):
        return set()
    raw = audit.raw["async"]
    by_task = defaultdict(list)
    executions = set()
    for group in raw["groups"]:
        ref, index = group["task_ref"], group["execution_index"]
        _require(ref in audit.tasks and index in audit.executions, "unknown task or physical execution")
        execution = audit.executions[index]
        _require(index not in executions, "duplicate physical execution group")
        executions.add(index)
        _require(
            group["run_epoch"] == audit.raw["run_epoch"]
            and (ref, group["block_idx"], group["subtask_slot"], group["dispatch_generation"])
            == (execution["task_ref"], execution["block_idx"], execution["subtask_slot"], execution["generation"]),
            "stale or inconsistent execution identity",
        )
        _audit_group(audit, group, execution)
        by_task[ref].append(group)
    finishes = {}
    covered = set()
    for finish in raw["finishes"]:
        ref = finish["task_ref"]
        _require(ref in by_task and ref not in finishes, "missing or duplicate task group coverage")
        finishes[ref] = finish
        indices = finish["group_indices"]
        expected = {group["index"] for group in by_task[ref]}
        _require(len(indices) == len(expected) and set(indices) == expected, "FINISH group coverage mismatch")
        _require(not covered.intersection(indices), "group belongs to multiple FINISH records")
        covered.update(indices)
        _require(
            finish["run_epoch"] == audit.raw["run_epoch"]
            and finish["state"] == "DRAINED"
            and finish["failure_code"] == 0,
            "FINISH is incomplete, failed or stale",
        )
        _lease(finish, 3)
        _require(finish["slice_count"] > 0, "drained FINISH has no service slice")
        _require(finish["pending_count"] == 0, "drained FINISH has an unpublished batch")
        audit.causal.add(audit.done_points[ref], _point(finish["drained"]))
    _require(set(finishes) == set(by_task), "missing FINISH coverage")
    _require(len(covered) == raw["group_count"], "missing group coverage")
    _audit_releases(audit, by_task)
    return set(by_task)


def _audit_group(audit, group, execution):
    from .native_causality import point  # noqa: PLC0415

    ref = group["task_ref"]
    _require(0 < group["capacity"] <= 64 and group["registered_count"] <= group["capacity"], "invalid capacity")
    _require(group["state"] in ("EMPTY", "READY") and group["failure_code"] == 0, "group is incomplete or failed")
    _require(execution["flags"] & 2 or group["state"] == "EMPTY", "skipped kernel has registered conditions")
    sealed = _point(group["sealed"])
    _require(sealed[0] == execution["worker_index"], "seal belongs to another execution worker")
    previous = "visible" if execution["flags"] & 2 else "entry"
    audit.before(point(execution, previous), sealed, "async seal before kernel visibility or skipped entry")
    audit.before(sealed, point(execution, "completion_consumed"), "async seal after execution consumption")
    _require(group["pending_count"] <= group["slice_count"], "pending slices exceed service slices")
    for condition in group["conditions"]:
        _require(condition["type"] in (0, 1), "unsupported condition type")
        if condition["binding_index"] is not None:
            _require(condition["binding_index"] < (1 << 64) - 1, "unbound condition")
    if group["state"] == "EMPTY":
        _lease(group, 0)
        _require(
            group["registered_count"]
            == group["observations"]
            == group["pending_count"]
            == group["slice_count"]
            == group["active_cycles"]
            == 0
            and group["terminal"] is None,
            "EMPTY group has CHECK activity or registered conditions",
        )
        audit.causal.add(sealed, audit.done_points[ref])
    else:
        _lease(group, 3)
        _require(group["registered_count"] > 0 and group["slice_count"] > 0, "READY group has no conditions or slice")
        _require(group["observations"] >= group["registered_count"], "READY group lacks condition observations")
        terminal = _point(group["terminal"])
        audit.before(sealed, terminal, "async terminal precedes seal")
        audit.causal.add(terminal, audit.done_points[ref])


def _audit_releases(audit, by_task):
    from .native_causality import point  # noqa: PLC0415

    expected = {(ref, block) for ref in by_task for block in range(audit.tasks[ref]["block_num"])}
    owners = {
        (block["owner"]["task_ref"], block["owner"]["block_idx"]): block["owner"] for block in audit.blocks.values()
    }
    executions = defaultdict(list)
    for execution in audit.executions.values():
        executions[(execution["task_ref"], execution["block_idx"])].append(execution)
    for ref, groups in by_task.items():
        slots = {group["subtask_slot"] for group in groups}
        actual = {(group["block_idx"], group["subtask_slot"]) for group in groups}
        _require(
            actual == {(block, slot) for block in range(audit.tasks[ref]["block_num"]) for slot in slots},
            "immutable task group coverage mismatch",
        )
    seen = set()
    for release in audit.raw["async"]["releases"]:
        key = release["task_ref"], release["block_idx"]
        _require(key in expected and key not in seen, "missing, duplicate or extraneous block release")
        seen.add(key)
        _require(release["run_epoch"] == audit.raw["run_epoch"], "stale block release")
        publication = release["publication"]
        _require(publication is not None and publication["after"] is not None, "incomplete block release publication")
        left, right = point(publication, "before"), point(publication, "after")
        released = audit.causal.bounded(("ExecutionReleased", *key), left, right)
        if key in owners:
            owner = owners[key]
            _require(publication["worker_index"] == owner["worker_index"], "block release has another owner")
            audit.before(point(owner, "completion_after"), left, "async block released before contribution")
        for execution in executions[key]:
            if key not in owners:
                _require(publication["worker_index"] == execution["worker_index"], "ordinary release has another owner")
            audit.before(point(execution, "trace_published"), left, "async block released before trace publication")
        # A remote last contributor can finish before the publisher records its after stamp.
        audit.causal.add(released, audit.done_points[key[0]])
    _require(seen == expected, "missing block release coverage")
