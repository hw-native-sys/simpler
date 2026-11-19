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

CONTROL_NOTIFY_ACCOUNTING = 1 << 26
COUNTER_ACCOUNTING = 1 << 27
COMPLETION_FROM_GROUP = 1 << 29
COMPLETION_FROM_CONTRIBUTION = 1 << 30
COMPLETION_ALIAS_FLAGS = COMPLETION_FROM_GROUP | COMPLETION_FROM_CONTRIBUTION
DEFERRED_FINISH = 1 << 31

ROOT_NUMBERS = frozenset(
    """
version run_epoch task_count worker_count trace_count member_count cooperative_task_count
cluster_count mix_task_count sync_mix_task_count expected_task_count inline_completed_count
active_worker_count scheduler_error error_claimed floor_count
""".split()
)
ROOT_TABLES = frozenset(
    "tasks workers task_controls task_events members floor_publications blocks executions "
    "condition_contributions completion_counters".split()
)
ERROR_NUMBERS = (
    "task_ref core_id core_type site run_epoch order_key block_idx group_seq worker_index cluster_lane identity_fields"
).split()

# Scalar fields are unsigned wire integers unless listed in SIGNED_FIELDS.
NUMBERS = {
    "task": (
        "task_ref flags active_mask block_num trace_base schedule_rank topological_level bottom_level cooperative_index"
    ),
    "worker": (
        "index worker_index context_worker_index run_epoch next_sequence "
        "error_flags finished core_id core_type cluster_index"
    ),
    "control": (
        "task_ref state next_block completed_subtasks canonical_total pending_compute completion_debt "
        "completion_backend counter_index counter_threshold counter_limit async_mask"
    ),
    "counter": "index value valid limit",
    "condition": "index run_epoch task_ref worker_index block_idx subtask_slot source credits",
    "claim": "run_epoch worker_index block_idx confirmed_count seal_value",
    "actor": "run_epoch task_ref order_key worker_index flags",
    "member": "index run_epoch task_ref order_key worker_index observed_mix_floor observed_self_floor",
    "floor": "index run_epoch cluster_index worker_index prior_mix_ref floor",
    "owner": (
        "run_epoch task_ref order_key group_seq completion_ticket worker_index "
        "block_idx cluster_index block_num active_mask flags completion_worker completion_credits"
    ),
    "lane": (
        "index run_epoch task_ref order_key group_seq worker_index block_idx "
        "cluster_index lane active_mask flags block_num resource_mask observed_self_floor"
    ),
    "execution": (
        "index run_epoch task_ref order_key observed_mix_floor observed_self_floor worker_index block_idx "
        "pending_slot subtask_slot flags ready_source group_seq block_num cluster_index cluster_lane "
        "publication_before_cycles publication_after_cycles publication_count "
        "contribution_ticket contribution_worker contribution_credits"
    ),
}
STAMPS = {
    "claim": "published_before published_after",
    "actor": "before after",
    "condition": "before after",
    "member": "committed retired",
    "floor": "before after",
    "owner": (
        "claim_before claim_after publish_before publish_after completion_before completion_after releases_observed"
    ),
    "lane": "descriptor_ready arrive_before arrive_after arrivals_observed release_before release_after",
    "execution": (
        "entry returned visible completion_consumed permit trace_published contribution_before contribution_after"
    ),
}
ARRAYS = {
    "task": {"kernel_ids": 3, "dependencies": None, "required_floor": 3},
    "worker": {"cluster_workers": 3},
    "owner": {"released_seq": 3},
    "lane": {"arrived_seq": 3, "lane_subtasks": 3},
}
SIGNED_FIELDS = frozenset(("task_ref", "core_id", "core_type", "state"))


def has_native_graph(section):
    """Native Graph evidence includes independent source and node maps."""
    return isinstance(section, dict) and ("graph_source" in section or "node_map" in section)


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
    if not isinstance(section, dict) or section.get("schema") != "hbg-native-protocol":
        _error("unsupported schema")
    version = section.get("version")
    if type(version) is not int or version != 2:
        _error(f"unsupported protocol version {version!r}; this reader requires version 2")
    native = has_native_graph(section)
    root_numbers = ROOT_NUMBERS
    extra = {"node_map", "graph_source"} if native else set()
    _object(
        section,
        root_numbers | ROOT_TABLES | {"schema", "producer", "first_error", "resolved_task_count"} | extra,
        "root",
    )
    if section["producer"] != "aicore":
        _error("unknown schema or producer")
    for field in root_numbers:
        _number(section[field], field)
    error_numbers = ERROR_NUMBERS
    _object(section["first_error"], error_numbers, "first_error")
    for field in error_numbers:
        _number(section["first_error"][field], f"first_error.{field}")
    if section["member_count"] != section["worker_count"] * section["cooperative_task_count"]:
        _error("member region capacity disagrees with task/worker counts")
    mix_tasks = section["mix_task_count"]
    if section["floor_count"] != ((mix_tasks + 1) * section["cluster_count"] if mix_tasks else 0):
        _error("floor publication capacity disagrees with cluster/task counts")
    for table, kind, count, key in (
        ("tasks", "task", section["task_count"], "task_ref"),
        ("workers", "worker", section["worker_count"], "index"),
        ("task_controls", "control", section["task_count"], "task_ref"),
    ):
        _indices(section[table], count, table, key=key, dense=True)
        for row in section[table]:
            extra = ("claim_seal",) if kind == "control" else ()
            _record(row, kind, f"{table}[{row[key]}]", extra)
    for table, kind, count in (
        ("members", "member", section["member_count"]),
        ("floor_publications", "floor", section["floor_count"]),
        ("executions", "execution", section["trace_count"]),
        ("condition_contributions", "condition", section["trace_count"]),
    ):
        _indices(section[table], count, table)
        for row in section[table]:
            _record(row, kind, f"{table}[{row['index']}]")
    _read_completion_counters(section)
    _read_nested_events(section)
    if native:
        from .native_graph import read_graph_source  # noqa: PLC0415

        read_graph_source(section)
    return section


def _read_completion_counters(section):
    counters = section["completion_counters"]
    if not isinstance(counters, list) or len(counters) > section["task_count"]:
        _error("invalid completion counter count")
    _indices(counters, len(counters), "completion_counters", dense=True)
    for counter in counters:
        _record(counter, "counter", f"completion_counters[{counter['index']}]")
    counter_chains(section)
    if any(counter["valid"] == 0 for counter in counters):
        if section["resolved_task_count"] is not None:
            _error("resolved_task_count must be null when a counter observation is missing")
    else:
        _number(section["resolved_task_count"], "resolved_task_count")


def _counter_ancestor(tasks, task, ancestor):
    pending, visited = list(tasks[task]["dependencies"]), set()
    while pending:
        current = pending.pop()
        if current == ancestor:
            return True
        if current in visited or current not in tasks:
            continue
        visited.add(current)
        pending.extend(tasks[current]["dependencies"])
    return False


def counter_chains(section):
    """Each counter accumulates complete tasks along a true dependency chain."""
    tasks = {row["task_ref"]: row for row in section["tasks"]}
    counters = {row["index"]: row for row in section["completion_counters"]}
    chains = {index: [] for index in counters}
    for control in section["task_controls"]:
        backend, index = control["completion_backend"], control["counter_index"]
        threshold, limit = control["counter_threshold"], control["counter_limit"]
        task = tasks[control["task_ref"]]
        if control["async_mask"] & ~task["active_mask"]:
            _error("async completion mask contains an inactive lane")
        if backend == 0:
            if index != (1 << 64) - 1 or threshold != 0 or limit != 0:
                _error("GM completion carries counter metadata")
            continue
        if (
            backend != 1
            or index not in counters
            or not task["flags"] & 1
            or task["flags"] & 32
            or not 0 < task["active_mask"] < 8
            or not 0 < threshold <= limit < 1 << 32
            or counters[index]["limit"] != limit
        ):
            _error("invalid completion counter descriptor")
        chains[index].append(control)
    for index, chain in chains.items():
        counter = counters[index]
        if (
            not chain
            or counter["valid"] not in (0, 1)
            or not 0 <= counter["value"] <= counter["limit"]
            or (not counter["valid"] and counter["value"] != 0)
        ):
            _error("invalid completion counter observation")
        previous, threshold = None, 0
        chain.sort(key=lambda row: row["counter_threshold"])
        for control in chain:
            ref = control["task_ref"]
            task = tasks[ref]
            threshold += task["block_num"] * (
                bin(task["active_mask"]).count("1") + bin(control["async_mask"]).count("1")
            )
            if control["counter_threshold"] != threshold or (
                previous is not None and not _counter_ancestor(tasks, ref, previous)
            ):
                _error("counter thresholds do not describe a true ancestor chain")
            previous = ref
        if threshold != counter["limit"]:
            _error("completion counter limit does not cover its chain")
    return chains


def _read_nested_events(section):
    _indices(section["task_events"], section["task_count"], "task_events")
    for row in section["task_events"]:
        _object(row, {"index", "completion"}, "task_events row")
        if row["completion"] is not None:
            _record(row["completion"], "actor", f"task_events[{row['index']}].completion")
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
            _record(lane, "lane", f"blocks[{row['index']}].lanes[{lane['index']}]")


def _event_records(section):
    for row in section["task_events"]:
        if row["completion"] is not None:
            yield row["completion"], "actor", f"task_events[{row['index']}].completion"
    for row in section["blocks"]:
        if row["claim"] is not None:
            yield row["claim"], "claim", f"blocks[{row['index']}].claim"
        if row["owner"] is not None:
            yield row["owner"], "owner", f"blocks[{row['index']}].owner"
        for lane in row["lanes"]:
            yield lane, "lane", f"blocks[{row['index']}].lanes[{lane['index']}]"
    for table, kind in (
        ("members", "member"),
        ("floor_publications", "floor"),
        ("executions", "execution"),
        ("condition_contributions", "condition"),
    ):
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
    groups = {}
    for block in section["blocks"]:
        owner = block["owner"]
        if owner is not None:
            groups.setdefault((owner["task_ref"], owner["completion_ticket"]), []).append(block)
    aliases = {}
    for event in section["task_events"]:
        done = event["completion"]
        if done is None or not done["flags"] & COMPLETION_ALIAS_FLAGS:
            continue
        ref = event["index"]
        task, control = tasks[ref], controls[ref]
        width = bin(task["active_mask"]).count("1")
        total = task["block_num"] * width
        group_alias = bool(done["flags"] & COMPLETION_FROM_GROUP)
        alias_flag = COMPLETION_FROM_GROUP if group_alias else COMPLETION_FROM_CONTRIBUTION
        if (
            not task["flags"] & 1
            or task["flags"] & 32
            or not 0 < task["active_mask"] < 8
            or not 0 < total <= 32767
            or done["flags"] & DEFERRED_FINISH
            or done["flags"] != task["flags"] | alias_flag
            or group_alias != bool(task["flags"] & 4)
            or done["task_ref"] != ref
            or done["run_epoch"] != section["run_epoch"]
            or control["canonical_total"] != total
            or control["pending_compute"] != 0
            or control["completion_debt"] != 0
            or control["state"] != 2
        ):
            _error(f"invalid physical TaskDone completion alias for task {ref}")
        if group_alias:
            candidates = groups.get((ref, total - width), ())
            if len(candidates) != 1:
                _error(f"TaskDone completion alias lacks a unique last group contribution for task {ref}")
            block = candidates[0]
            source = block["owner"]
            if (
                source["block_idx"] >= task["block_num"]
                or source["block_num"] != task["block_num"]
                or source["active_mask"] != task["active_mask"]
                or source["flags"] != task["flags"]
                or block["index"] != task["trace_base"] + source["block_idx"] * width
                or source["run_epoch"] != done["run_epoch"]
                or source["completion_worker"] != done["worker_index"]
                or source["order_key"] != done["order_key"]
                or done["before"] != source["completion_before"]
                or done["after"] != source["completion_after"]
                or not 0 < done["before"][1] < done["after"][1]
            ):
                _error(f"TaskDone completion alias does not identify its last group contribution for task {ref}")
            aliases[ref] = source
            continue
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
            + block * bin(task["active_mask"]).count("1")
            + bin(task["active_mask"] & ((1 << lane) - 1)).count("1")
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


def stamp_worker(record, field):
    if "completion_worker" in record and field in ("completion_before", "completion_after", "releases_observed"):
        return record["completion_worker"]
    return record["worker_index"]


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
        if kind == "actor" and record["flags"] & COMPLETION_ALIAS_FLAGS:
            # The validated physical or group contribution supplies these events once.
            continue
        for name in STAMPS[kind].split():
            worker = stamp_worker(record, name)
            if worker not in workers:
                _error(f"foreign event writer at {path}.{name}")
            stream = streams[worker]
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
