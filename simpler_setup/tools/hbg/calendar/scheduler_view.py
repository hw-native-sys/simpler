# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Task/block timing lanes for device-side Calendar scheduling."""

from collections import defaultdict

INPUT_WAIT = "input_wait"

_LABELS = {
    "row_begin": "Calendar row begins",
    "peer_dmb_write_begin": "Begin peer DMB store",
    "peer_dmb_batch_begin": "Begin DMB batch",
    "peer_dmb_batch_issued": "DMB batch issued",
    "peer_dmb_wait_begin": "Begin local DMB wait",
    "peer_dmb_published": "Write peer DMB",
    "peer_dmb_observed": "Accept DMB token / check completion",
    "peer_dmb_mov_at_entry": "MOV DMB · first sample",
    "peer_dmb_mov_changed": "MOV DMB · changed token",
    "peer_dmb_initial_proof": "Check CntNotify on wait entry",
    "peer_dmb_periodic_ready": "Periodic CntNotify check reached threshold",
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
    "ready_row_scan": "Select a ready row",
}

_DEFER_EVENTS = {
    "defer_registered",
    "defer_poll_begin",
    "defer_signal_ready",
    "deferred_completion_issue",
}
_ROLES = ("C", "V0", "V1")


def calendar_subtask_slot(kernel_ids, cluster_lane):
    """Return the logical C/V role for a calendar row's physical cluster lane."""
    if (
        len(kernel_ids) != 3
        or any(type(kernel) is not int for kernel in kernel_ids)
        or type(cluster_lane) is not int
        or not 0 <= cluster_lane < len(kernel_ids)
    ):
        raise ValueError("invalid calendar task shape or physical lane")
    active = [role for role, kernel in enumerate(kernel_ids) if kernel >= 0]
    if len(active) == 1 and active[0] != 0 and cluster_lane != 0:
        # A standalone AIV task can use either physical Vector lane.
        return active[0]
    if cluster_lane in active:
        return cluster_lane
    raise ValueError(f"calendar lane {cluster_lane} does not execute a kernel in roles {active}")


def worker_lane(cluster, role):
    return f"Cluster {cluster:02d} · {_ROLES[role]}", cluster * len(_ROLES) + role


def parse_capture(section):
    if not isinstance(section, dict) or not isinstance(section.get("records"), list):
        raise ValueError("calendar_task_events must contain a records array")
    for field in ("dropped", "unmatched_deferred_updates"):
        if type(section.get(field)) is not int or section[field] < 0:
            raise ValueError(f"invalid calendar_task_events.{field}")
    for field in ("folded_task_ids", "inline_completed_task_ids"):
        ids = section.get(field)
        if not isinstance(ids, list) or any(type(task_id) is not int or task_id < 0 for task_id in ids):
            raise ValueError(f"invalid calendar_task_events.{field}")
    for row in section["records"]:
        _validate_task_event(row)
    ranges = section.get("range_records")
    if not isinstance(ranges, list):
        raise ValueError("calendar_task_events must contain a range_records array")
    for row in ranges:
        if not isinstance(row, dict) or row.get("kind") not in _RANGE_LABELS:
            raise ValueError("invalid calendar range event kind")
        for field in ("cycles", "run_epoch", "worker_id", "cluster_index", "cluster_lane", "range_index"):
            if type(row.get(field)) is not int or row[field] < 0:
                raise ValueError(f"invalid calendar range event {field}")
        if not 0 <= row["cluster_lane"] < len(_ROLES) or row["cycles"] == 0:
            raise ValueError("invalid calendar range coordinates or timestamp")
        if row["kind"] == "graph_ready_observed" and (
            type(row.get("graph_shell_id")) is not int or type(row.get("watermark")) is not int
        ):
            raise ValueError("invalid calendar Graph watermark event")
        if row["kind"] == "ready_row_scan" and (
            type(row.get("scan_start_cycles")) is not int or not 0 < row["scan_start_cycles"] <= row["cycles"]
        ):
            raise ValueError("invalid calendar ready-row scan interval")
    return section


def _validate_peer_event(row):
    if row["kind"] in ("peer_dmb_batch_begin", "peer_dmb_batch_issued"):
        if row["kind"] == "peer_dmb_batch_issued" and row.get("transport") not in ("Scalar", "MTE3"):
            raise ValueError("invalid percounter DMB batch transport")
        count, sequence = row.get("target_count"), row.get("sequence")
        if type(count) is not int or count < 2 or type(sequence) is not int or not 0 < sequence < 2**32:
            raise ValueError("invalid percounter DMB batch")
        commands = row.get("command_count")
        if commands is not None and (type(commands) is not int or not 1 <= commands <= count):
            raise ValueError("invalid percounter DMB command count")
        return
    if row["kind"] == "peer_dmb_wait_begin":
        threshold = row.get("input_count")
        if type(threshold) is not int or not 1 <= threshold <= 2:
            raise ValueError("invalid percounter DMB input count")
        return
    if row["kind"] == "peer_dmb_periodic_ready":
        interval = row.get("cycles_since_last_check")
        count = row.get("ready_inputs")
        if type(interval) is not int or interval <= 0 or type(count) is not int or not 1 <= count <= 2:
            raise ValueError("invalid periodic DMB completion check")
        return
    token = row.get("token")
    if type(token) is not int or not 0 <= token < 2**32:
        raise ValueError("invalid percounter DMB token")
    if row["kind"] in ("peer_dmb_mov_at_entry", "peer_dmb_mov_changed"):
        seq = row.get("sequence")
        if type(seq) is not int or not 0 <= seq < 2**32:
            raise ValueError("invalid DMB MOV sequence")
        prev = row.get("previous_poll_cycles")
        if prev is not None and (
            row["kind"] == "peer_dmb_mov_at_entry" or type(prev) is not int or not 0 < prev <= row["cycles"]
        ):
            raise ValueError("invalid DMB MOV previous sample time")
        return
    if row["kind"] in ("peer_dmb_write_begin", "peer_dmb_published"):
        target = row.get("target_worker")
        if type(target) is not int or target < 0:
            raise ValueError("invalid percounter DMB target worker")
    elif type(row.get("ready_inputs")) is not int or not 0 <= row["ready_inputs"] <= 2:
        raise ValueError("invalid percounter DMB completion count")


def _validate_task_event(row):
    if not isinstance(row, dict) or not isinstance(row.get("kind"), str) or row["kind"] not in _LABELS:
        raise ValueError("invalid calendar task event kind")
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
            raise ValueError(f"invalid calendar task event {field}")
    if not 0 <= row["block_index"] < row["block_count"] or row["cluster_lane"] >= 3 or row["cycles"] == 0:
        raise ValueError("invalid calendar task event coordinates or timestamp")
    if row["kind"].startswith("peer_dmb_"):
        _validate_peer_event(row)
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
            raise ValueError("invalid calendar execution checkpoints")
    if row["kind"] in _DEFER_EVENTS:
        group = row.get("deferred_group")
        if type(group) is not int or group <= 0:
            raise ValueError("invalid calendar deferred group")
        if row["kind"] != "defer_registered":
            source = row.get("source_worker_id")
            if type(source) is not int or source < 0:
                raise ValueError("invalid calendar deferred producer")
    if row["kind"] in ("completion_issue", "deferred_completion_issue"):
        contribution = row.get("contribution")
        if type(contribution) is not int or contribution != 1:
            raise ValueError("invalid calendar completion contribution")


def decode_capture(section, to_us):
    return {
        **section,
        "range_records": [
            {
                **row,
                "time_us": to_us(row["cycles"]),
                **({"scan_start_us": to_us(row["scan_start_cycles"])} if row["kind"] == "ready_row_scan" else {}),
            }
            for row in section["range_records"]
        ],
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
            "cat": "calendar_task_timing",
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
        (INPUT_WAIT, "row_begin", "inputs_ready", "Read row and wait for inputs"),
        ("finalize_ready", "inputs_ready", "ready", "Finalize row readiness"),
        ("mix_release_wait", "row_begin", "mix_release_observed", "Read row and wait for local MIX release"),
        ("prepare_entry", "ready", "kernel_start", "Prepare kernel entry"),
        ("output_fence", "kernel_end", "fence_end", "Output fence"),
    )
    for kind, begin, end, label in stages:
        if begin in timestamps and end in timestamps and timestamps[begin] < timestamps[end]:
            yield kind, label, timestamps[begin], timestamps[end]
    finish = timestamps.get("fence_end", timestamps.get("ready"))
    retired = timestamps.get("local_retired")
    if finish is not None and retired is not None and finish < retired:
        label = "Retire row"
        if "defer_registered" in timestamps:
            label = "Publish defer and retire"
        elif "completion_issue" in timestamps:
            label = "Issue DONE and retire"
        yield "retire_row", label, finish, retired


def _dmb_spans(rows, waiting_tasks):
    batch = next((r for r in rows if r["kind"] == "peer_dmb_batch_begin"), None)
    if batch is not None:
        end = next((r for r in rows if r["kind"] == "peer_dmb_batch_issued"), None)
        if (
            end is not None
            and end["sequence"] == batch["sequence"]
            and end["target_count"] == batch["target_count"]
            and end["cycles"] >= batch["cycles"]
        ):
            detail = {
                "transport": end["transport"],
                "target_count": batch["target_count"],
                "sequence": batch["sequence"],
            }
            if "command_count" in end:
                detail["command_count"] = end["command_count"]
            yield (
                f"Issue {end['transport']} batch to {batch['target_count']} DMBs",
                batch["time_us"],
                end["time_us"],
                detail,
            )
    writes = {(r["target_worker"], r["token"]): r for r in rows if r["kind"] == "peer_dmb_write_begin"}
    for r in rows:
        if r["kind"] != "peer_dmb_published":
            continue
        target = r["target_worker"]
        start = writes.get((target, r["token"]))
        if start is None or r["cycles"] < start["cycles"]:
            continue
        names = waiting_tasks.get((r["run_epoch"], target), set())
        destination = next(iter(names)) if len(names) == 1 else f"worker {target}"
        mte3 = r.get("transport") == "MTE3"
        yield (
            f"Issue MTE3 DMB write → {destination}" if mte3 else f"Write DMB → {destination}",
            start["time_us"],
            r["time_us"],
            {"target_worker": target, "token": r["token"], "transport": "MTE3" if mte3 else "Scalar"},
        )

    start = next((r for r in rows if r["kind"] == "peer_dmb_wait_begin"), None)
    if start is None:
        return
    threshold = start["input_count"]
    checks = (
        r for r in rows if r["kind"] in ("peer_dmb_initial_proof", "peer_dmb_observed", "peer_dmb_periodic_ready")
    )
    tokens = 0
    for r in sorted(checks, key=lambda row: row["cycles"]):
        tokens += r["kind"] == "peer_dmb_observed"
        if r["cycles"] < start["cycles"] or r["ready_inputs"] < threshold:
            continue
        args = {"observed_tokens": tokens, "input_count": threshold, "ready_inputs": r["ready_inputs"]}
        if "token" in r:
            args["token"] = r["token"]
        label = "Wait on local DMB + CntNotify"
        if r["kind"] == "peer_dmb_initial_proof":
            label = "CntNotify ready on wait entry"
        elif r["kind"] == "peer_dmb_periodic_ready":
            label = "Wait until periodic CntNotify check succeeds"
            args["cycles_since_last_check"] = r["cycles_since_last_check"]
        yield label, start["time_us"], r["time_us"], args
        break


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
    waiting_tasks = defaultdict(set)
    for row in capture["records"]:
        worker = row["worker_id"]
        kind = row["kind"]
        if kind in _DEFER_EVENTS:
            deferred_by_group[row["run_epoch"], row["deferred_group"]][kind] = row
        if kind not in _DEFER_EVENTS or kind == "defer_registered":
            rows_by_execution[row["run_epoch"], row["task_id"], row["block_index"], worker].append(row)
        if kind == "peer_dmb_wait_begin":
            waiting_tasks[row["run_epoch"], worker].add(task_name(row["task_id"]))
        if kind in ("peer_dmb_wait_begin", "peer_dmb_write_begin", "peer_dmb_batch_begin", "peer_dmb_batch_issued"):
            continue
        yield {
            **lanes[worker],
            "cat": "calendar_task_timing",
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
        for kind, label, start, end in _row_spans(rows):
            yield {
                **lanes[row["worker_id"]],
                "cat": "calendar_task_timing",
                "ph": "X",
                "ts": start,
                "dur": end - start,
                "name": f"{task_name(row['task_id'])} [{row['block_index']}] · {label}",
                "args": {**args, "kind": kind},
            }
        for label, start, end, detail in _dmb_spans(rows, waiting_tasks):
            yield {
                **lanes[row["worker_id"]],
                "cat": "calendar_dmb",
                "ph": "X",
                "ts": start,
                "dur": end - start,
                "name": f"{task_name(row['task_id'])} [{row['block_index']}] · {label}",
                "args": {**args, **detail},
            }
    for (epoch, group), stages in deferred_by_group.items():
        registered = stages.get("defer_registered")
        first_poll = stages.get("defer_poll_begin")
        if registered is None or first_poll is None or first_poll["time_us"] < registered["time_us"]:
            continue
        for row, phase in ((registered, "s"), (first_poll, "f")):
            yield {
                **lanes[row["worker_id"]],
                "cat": "calendar_defer",
                "ph": phase,
                "ts": row["time_us"],
                "name": "Deferred completion ownership",
                "id": f"calendar-defer/{epoch}/{group}",
                "bind_id": _defer_marker_id(row),
                **({"bp": "e"} if phase == "f" else {}),
                "args": {"run_epoch": epoch, "deferred_group": group},
            }
    if capture.get("dropped") or capture.get("unmatched_deferred_updates"):
        yield from capture_status(capture, pid)


def _defer_marker_id(row):
    return f"calendar-defer/{row['run_epoch']}/{row['deferred_group']}/{row['kind']}"


def _range_events(rows, lanes):
    ranges = defaultdict(dict)
    for row in rows:
        worker, index, kind = row["worker_id"], row["range_index"], row["kind"]
        graph = row.get("graph_shell_id")
        label = f"Graph {graph} · " if graph is not None else ""
        if kind == "ready_row_scan":
            yield {
                **lanes[worker],
                "cat": "calendar_range",
                "ph": "X",
                "ts": row["scan_start_us"],
                "dur": row["time_us"] - row["scan_start_us"],
                "name": label + _RANGE_LABELS[kind],
                "args": row,
            }
            continue
        ranges[row["run_epoch"], worker, index][kind] = row
        yield {
            **lanes[worker],
            "cat": "calendar_range",
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
                "cat": "calendar_range",
                "ph": "X",
                "ts": begin["time_us"],
                "dur": end["time_us"] - begin["time_us"],
                "name": "Range entry",
                "args": begin,
            }
