# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Task/block timing lanes for device-side Percounter scheduling."""

from collections import defaultdict

_LABELS = {
    "row_begin": "Calendar row begins",
    "inputs_ready": "Inputs satisfied",
    "mix_release_observed": "Observe local MIX release",
    "ready": "Ready",
    "kernel_end": "Kernel return",
    "completion_issue": "DONE issued",
    "local_retired": "Row retired",
    "defer_registered": "Defer flag issued",
    "defer_poll_begin": "First defer poll",
    "defer_signal_ready": "Defer signal observed",
    "deferred_completion_issue": "Deferred DONE issued",
}

_RANGE_LABELS = {
    "range_enter_begin": "Wait for range",
    "graph_ready_observed": "Graph watermark observed",
    "range_entered": "Range entered",
}

_DEFER_EVENTS = {
    "defer_registered",
    "defer_poll_begin",
    "defer_signal_ready",
    "deferred_completion_issue",
}
_ROLES = ("C", "V0", "V1")


def worker_lane(cluster, role):
    return f"Cluster {cluster:02d} · {_ROLES[role]}", cluster * len(_ROLES) + role


def parse_capture(section):
    if not isinstance(section, dict) or not isinstance(section.get("records"), list):
        raise ValueError("percounter_task_events must contain a records array")
    for field in ("dropped", "unmatched_deferred_updates"):
        if type(section.get(field)) is not int or section[field] < 0:
            raise ValueError(f"invalid percounter_task_events.{field}")
    for field in ("folded_task_ids", "inline_completed_task_ids"):
        ids = section.get(field)
        if not isinstance(ids, list) or any(type(task_id) is not int or task_id < 0 for task_id in ids):
            raise ValueError(f"invalid percounter_task_events.{field}")
    for row in section["records"]:
        _validate_task_event(row)
    ranges = section.get("range_records")
    if not isinstance(ranges, list):
        raise ValueError("percounter_task_events must contain a range_records array")
    for row in ranges:
        if not isinstance(row, dict) or row.get("kind") not in _RANGE_LABELS:
            raise ValueError("invalid percounter range event kind")
        for field in ("cycles", "run_epoch", "worker_id", "cluster_index", "cluster_lane", "range_index"):
            if type(row.get(field)) is not int or row[field] < 0:
                raise ValueError(f"invalid percounter range event {field}")
        if not 0 <= row["cluster_lane"] < len(_ROLES) or row["cycles"] == 0:
            raise ValueError("invalid percounter range coordinates or timestamp")
        if row["kind"] == "graph_ready_observed" and (
            type(row.get("graph_shell_id")) is not int or type(row.get("watermark")) is not int
        ):
            raise ValueError("invalid percounter Graph watermark event")
    return section


def _validate_task_event(row):
    if not isinstance(row, dict) or not isinstance(row.get("kind"), str) or row["kind"] not in _LABELS:
        raise ValueError("invalid percounter task event kind")
    for field in (
        "cycles",
        "run_epoch",
        "task_id",
        "worker_id",
        "cluster_index",
        "cluster_lane",
        "block_index",
        "block_count",
    ):
        if type(row.get(field)) is not int or row[field] < 0:
            raise ValueError(f"invalid percounter task event {field}")
    if not 0 <= row["block_index"] < row["block_count"] or row["cluster_lane"] >= 3 or row["cycles"] == 0:
        raise ValueError("invalid percounter task event coordinates or timestamp")
    if row["kind"] == "kernel_end":
        dispatch, start, finish = (
            row.get(name) for name in ("dispatch_cycles", "kernel_start_cycles", "finish_cycles")
        )
        if (
            type(dispatch) is not int
            or type(start) is not int
            or type(finish) is not int
            or not 0 < dispatch <= start <= row["cycles"] <= finish
        ):
            raise ValueError("invalid percounter execution checkpoints")
    if row["kind"] in _DEFER_EVENTS:
        group = row.get("deferred_group")
        if type(group) is not int or group <= 0:
            raise ValueError("invalid percounter deferred group")
        if row["kind"] != "defer_registered":
            source = row.get("source_worker_id")
            if type(source) is not int or source < 0:
                raise ValueError("invalid percounter deferred producer")
    if row["kind"] in ("completion_issue", "deferred_completion_issue"):
        contribution = row.get("contribution")
        if type(contribution) is not int or contribution != 1:
            raise ValueError("invalid percounter completion contribution")


def decode_capture(section, to_us):
    return {
        **section,
        "range_records": [{**row, "time_us": to_us(row["cycles"])} for row in section["range_records"]],
        "records": [
            {
                **row,
                "time_us": to_us(row["cycles"]),
                **(
                    {"dispatch_us": to_us(row["dispatch_cycles"]), "finish_us": to_us(row["finish_cycles"])}
                    if "dispatch_cycles" in row
                    else {}
                ),
                **({"kernel_start_us": to_us(row["kernel_start_cycles"])} if row["kind"] == "kernel_end" else {}),
            }
            for row in section["records"]
        ],
    }


def _meta(pid, name, args, tid=None):
    return {
        "cat": "__metadata",
        "ph": "M",
        "pid": pid,
        "name": name,
        "args": args,
        **({"tid": tid} if tid is not None else {}),
    }


def _block_completion_issue(rows, shares):
    """Return the last required lane credit for a fully recorded block."""
    immediate = []
    registered = set()
    settled = {}
    for row in rows:
        if row["kind"] == "completion_issue":
            immediate.append(row["time_us"])
        elif row["kind"] in ("defer_registered", "deferred_completion_issue"):
            group = row.get("deferred_group")
            if type(group) is not int or group <= 0:
                return None
            if row["kind"] == "defer_registered":
                if group in registered:
                    return None
                registered.add(group)
            else:
                if group in settled:
                    return None
                settled[group] = row["time_us"]

    if registered != settled.keys() or len(immediate) + len(settled) != shares:
        return None
    return max([*immediate, *settled.values()])


def _task_completion_issues(capture, blocks, producers, deps_kernel_map, deps_block_map):
    if (
        not producers
        or not deps_kernel_map
        or not deps_block_map
        or capture["dropped"]
        or capture["unmatched_deferred_updates"]
    ):
        return {}

    by_task = defaultdict(dict)
    for (epoch, task_id, block), rows in blocks.items():
        if task_id in producers:
            by_task[epoch, task_id][block] = rows

    issues = {}
    for key, task_blocks in by_task.items():
        task_id = key[1]
        block_count = deps_block_map.get(task_id)
        kernels = deps_kernel_map.get(task_id)
        if (
            kernels is None
            or block_count != len(task_blocks)
            or any(row["block_count"] != block_count for rows in task_blocks.values() for row in rows)
        ):
            continue
        # A dependency-only block contributes one share if it has an issue event.
        shares = max(1, sum(kernel >= 0 for kernel in kernels))
        times = [ts for rows in task_blocks.values() if (ts := _block_completion_issue(rows, shares)) is not None]
        if len(times) == len(task_blocks):
            issues[key] = max(times)
    return issues


def control_completions(capture, kernel_ids, block_counts):
    """Yield the recorded issue which satisfied each dependency-only task's last block."""
    controls = {ref for ref, kids in kernel_ids.items() if kids == [-1, -1, -1]}
    if not controls:
        return
    blocks = defaultdict(list)
    for row in capture["records"]:
        if row["task_id"] in controls:
            blocks[row["run_epoch"], row["task_id"], row["block_index"]].append(row)
    issues = _task_completion_issues(capture, blocks, controls, kernel_ids, block_counts)
    for (epoch, task, _), rows in blocks.items():
        for row in rows:
            if (
                row["kind"] in ("completion_issue", "deferred_completion_issue")
                and row["contribution"] > 0
                and issues.get((epoch, task)) == row["time_us"]
            ):
                del issues[epoch, task]
                yield row


def capture_status(capture, pid):
    if capture["dropped"] or capture["unmatched_deferred_updates"]:
        yield _meta(pid, "thread_name", {"name": "Capture status"}, 0)
        yield {
            "pid": pid,
            "tid": 0,
            "cat": "percounter_task_timing",
            "ph": "i",
            "s": "t",
            "name": "Partial task timing capture",
            "ts": min((row["time_us"] for row in capture["records"]), default=0),
            "args": {key: capture[key] for key in ("dropped", "unmatched_deferred_updates")},
        }


def _row_spans(rows):
    """Measured intervals within one physical lane's calendar row."""
    timestamps = {row["kind"]: row["time_us"] for row in rows}
    kernel = next((row for row in rows if row["kind"] == "kernel_end"), None)
    if kernel is not None:
        timestamps["kernel_start"] = kernel["kernel_start_us"]
        timestamps["fence_end"] = kernel["finish_us"]
    stages = (
        ("row_begin", "inputs_ready", "Read row and wait for inputs"),
        ("inputs_ready", "ready", "Finalize row readiness"),
        ("row_begin", "mix_release_observed", "Read row and wait for local MIX release"),
        ("ready", "kernel_start", "Prepare kernel entry"),
        ("kernel_end", "fence_end", "Output fence"),
    )
    for begin, end, label in stages:
        if begin in timestamps and end in timestamps and timestamps[begin] < timestamps[end]:
            yield label, timestamps[begin], timestamps[end]
    finish = timestamps.get("fence_end", timestamps.get("ready"))
    retired = timestamps.get("local_retired")
    if finish is not None and retired is not None and finish < retired:
        label = "Retire row"
        if "defer_registered" in timestamps:
            label = "Publish defer and retire"
        elif "completion_issue" in timestamps:
            label = "Issue DONE and retire"
        yield label, finish, retired


def trace_events(capture, pid, task_name):
    """Draw calendar transitions on the cores executing them."""
    yield _meta(pid, "process_name", {"name": "Scheduler View"})
    yield _meta(pid, "process_sort_index", {"sort_index": 3})
    lanes = {}
    ranges = capture.get("range_records", [])
    for row in capture["records"] + ranges:
        worker = row["worker_id"]
        if worker not in lanes:
            name, order = worker_lane(row["cluster_index"], row["cluster_lane"])
            lanes[worker] = {"pid": pid, "tid": order + 1}
            yield _meta(pid, "thread_name", {"name": name}, order + 1)
            yield _meta(pid, "thread_sort_index", {"sort_index": order}, order + 1)
    yield from _range_events(ranges, lanes)
    rows_by_execution = defaultdict(list)
    deferred_by_group = defaultdict(dict)
    for row in capture["records"]:
        worker = row["worker_id"]
        kind = row["kind"]
        if kind in _DEFER_EVENTS:
            deferred_by_group[row["run_epoch"], row["deferred_group"]][kind] = row
        if kind not in _DEFER_EVENTS or kind == "defer_registered":
            rows_by_execution[row["run_epoch"], row["task_id"], row["block_index"], worker].append(row)
        yield {
            **lanes[worker],
            "cat": "percounter_task_timing",
            "ph": "i",
            "s": "t",
            "ts": row["time_us"],
            "name": f"{task_name(row['task_id'])} [{row['block_index']}] · {_LABELS[kind]}",
            "args": {**row, "role": _ROLES[row["cluster_lane"]]},
            **({"id": _defer_marker_id(row)} if kind in _DEFER_EVENTS else {}),
        }
    for rows in rows_by_execution.values():
        row = rows[0]
        args = {key: row[key] for key in ("run_epoch", "task_id", "block_index", "block_count", "worker_id")}
        args["role"] = _ROLES[row["cluster_lane"]]
        for label, start, end in _row_spans(rows):
            yield {
                **lanes[row["worker_id"]],
                "cat": "percounter_task_timing",
                "ph": "X",
                "ts": start,
                "dur": end - start,
                "name": f"{task_name(row['task_id'])} [{row['block_index']}] · {label}",
                "args": args,
            }
    for (epoch, group), stages in deferred_by_group.items():
        registered = stages.get("defer_registered")
        first_poll = stages.get("defer_poll_begin")
        if registered is None or first_poll is None or first_poll["time_us"] < registered["time_us"]:
            continue
        for row, phase in ((registered, "s"), (first_poll, "f")):
            yield {
                **lanes[row["worker_id"]],
                "cat": "percounter_defer",
                "ph": phase,
                "ts": row["time_us"],
                "name": "Deferred completion ownership",
                "id": f"percounter-defer/{epoch}/{group}",
                "bind_id": _defer_marker_id(row),
                **({"bp": "e"} if phase == "f" else {}),
                "args": {"run_epoch": epoch, "deferred_group": group},
            }
    if capture.get("dropped") or capture.get("unmatched_deferred_updates"):
        yield from capture_status(capture, pid)


def _defer_marker_id(row):
    return f"percounter-defer/{row['run_epoch']}/{row['deferred_group']}/{row['kind']}"


def _range_events(rows, lanes):
    ranges = defaultdict(dict)
    for row in rows:
        worker, index, kind = row["worker_id"], row["range_index"], row["kind"]
        ranges[row["run_epoch"], worker, index][kind] = row
        graph = row.get("graph_shell_id")
        label = f"Graph {graph} · " if graph is not None else ""
        yield {
            **lanes[worker],
            "cat": "percounter_range",
            "ph": "i",
            "s": "t",
            "ts": row["time_us"],
            "name": label + _RANGE_LABELS[kind],
            "args": row,
        }
    for (_epoch, worker, _index), stages in ranges.items():
        begin, end = stages.get("range_enter_begin"), stages.get("range_entered")
        if begin is not None and end is not None and begin["time_us"] < end["time_us"]:
            yield {
                **lanes[worker],
                "cat": "percounter_range",
                "ph": "X",
                "ts": begin["time_us"],
                "dur": end["time_us"] - begin["time_us"],
                "name": "Range entry",
                "args": begin,
            }
