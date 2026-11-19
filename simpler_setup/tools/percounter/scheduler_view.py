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
from heapq import heappop, heappush

_LABELS = {
    "claim": "Claim",
    "ready": "Ready",
    "kernel_end": "Kernel end",
    "completion_issue": "Completion issued",
    "defer_registered": "Defer registered",
    "deferred_completion_issue": "Deferred completion issued",
}


def parse_capture(section):
    if not isinstance(section, dict) or not isinstance(section.get("records"), list):
        raise ValueError("percounter_task_events must contain a records array")
    for field in ("dropped", "unmatched_deferred_updates"):
        if type(section.get(field)) is not int or section[field] < 0:
            raise ValueError(f"invalid percounter_task_events.{field}")
    for row in section["records"]:
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
        if row["kind"] == "claim":
            start = row.get("claim_start_cycles")
            if type(start) is not int or not 0 < start <= row["cycles"]:
                raise ValueError("invalid percounter claim interval")
        if row["kind"] == "kernel_end":
            start = row.get("kernel_start_cycles")
            if type(start) is not int or not 0 < start <= row["cycles"]:
                raise ValueError("invalid percounter kernel interval")
        if (
            row["kind"] in ("completion_issue", "deferred_completion_issue")
            and type(row.get("contribution")) is not int
        ):
            raise ValueError("invalid percounter completion contribution")
    return section


def decode_capture(section, to_us):
    return {
        **section,
        "records": [
            {
                **row,
                "time_us": to_us(row["cycles"]),
                **({"claim_start_us": to_us(row["claim_start_cycles"])} if row["kind"] == "claim" else {}),
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
    """Return the last required positive issue for a fully recorded block."""
    ordinary = []
    registered = set()
    settled = {}
    for row in rows:
        if row["kind"] == "completion_issue":
            ordinary.append((row["contribution"], row["time_us"]))
        elif row["kind"] in ("defer_registered", "deferred_completion_issue"):
            group = row.get("deferred_group")
            if type(group) is not int or group <= 0:
                return None
            if row["kind"] == "defer_registered":
                if group in registered:
                    return None
                registered.add(group)
            else:
                if group in settled or row["contribution"] != 1:
                    return None
                settled[group] = row["time_us"]

    if registered != settled.keys() or len(registered) > shares:
        return None
    # MIX withdraws one share per defer, then issues one aggregate per block.
    # Single-lane defer withholds the contribution until settlement.
    if shares > 1:
        expected = [-1] * len(registered) + [shares]
    else:
        expected = [] if registered else [1]
    if sorted(delta for delta, _ in ordinary) != expected:
        return None
    return max([ts for delta, ts in ordinary if delta > 0] + list(settled.values()))


def _task_completion_issues(capture, blocks, deps_edges, deps_kernel_map, deps_block_map):
    if (
        not deps_edges
        or not deps_kernel_map
        or not deps_block_map
        or capture["dropped"]
        or capture["unmatched_deferred_updates"]
    ):
        return {}

    by_task = defaultdict(dict)
    for (epoch, task_id, block), rows in blocks.items():
        if task_id in deps_edges:
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


def _milestones(rows, ideal_ready):
    """Use recorded endpoints for every span, including in partial captures."""
    by_kind = defaultdict(list)
    for row in rows:
        by_kind[row["kind"]].append(row)
    ticks = [row["time_us"] for row in rows]
    starts = [row["kernel_start_us"] for row in by_kind["kernel_end"]]
    completion = [
        row for row in by_kind["completion_issue"] + by_kind["deferred_completion_issue"] if row["contribution"] > 0
    ]
    spans = []
    claim = by_kind["claim"][0] if by_kind["claim"] else None
    if claim:
        ticks.append(claim["claim_start_us"])
        spans.append(("Claim start → claim", claim["claim_start_us"], claim["time_us"], "thread_state_running"))
    if claim and by_kind["ready"]:
        claimed, ready = claim["time_us"], by_kind["ready"][0]["time_us"]
        if ideal_ready is not None and ideal_ready <= ready:
            ticks.append(ideal_ready)
            if ideal_ready < claimed:
                spans.append(("Claim → ready (dependencies already ready)", claimed, ready, "thread_state_runnable"))
            else:
                spans.append(("Claim → ideal ready", claimed, ideal_ready, "thread_state_iowait"))
                spans.append(("Ideal ready → ready", ideal_ready, ready, "thread_state_runnable"))
        else:
            ideal_ready = None
            spans.append(("Claim → ready", claimed, ready, "thread_state_runnable"))
    else:
        ideal_ready = None
    if starts:
        start, end = min(starts), by_kind["kernel_end"][-1]["time_us"]
        if by_kind["ready"]:
            spans.append(("Ready → kernel start", by_kind["ready"][0]["time_us"], start, "thread_state_iowait"))
        spans.append(("Kernel execution", start, end, "good"))
        if completion:
            spans.append(
                (
                    "Kernel end → completion issue",
                    end,
                    max(r["time_us"] for r in completion),
                    "thread_state_uninterruptible",
                )
            )
    return min(ticks + starts), max(ticks), spans, ideal_ready


def _block_events(rows, pid, tid, name, spans, ideal_ready):
    block = rows[0]
    args = {key: block[key] for key in ("run_epoch", "task_id", "block_index", "block_count")}
    location = {"pid": pid, "tid": tid, "cat": "percounter_task_timing"}
    if ideal_ready is not None:
        args["ideal_ready_us"] = ideal_ready
        args["ideal_ready_source"] = "predecessor_completion_issue"
    for label, start, end, color in spans:
        if start < end:
            yield {
                **location,
                "ph": "X",
                "ts": start,
                "dur": end - start,
                "name": f"{label} · {name}",
                "cname": color,
                "args": args,
            }
    if ideal_ready is not None:
        yield {
            **location,
            "ph": "i",
            "s": "t",
            "ts": ideal_ready,
            "name": f"Ideal ready (estimated) · {name}",
            "args": {**args, "kind": "ideal_ready"},
        }
    for row in rows:
        label = _LABELS[row["kind"]]
        role = ("C", "V0", "V1")[row["cluster_lane"]]
        if row["kind"] in ("ready", "kernel_end"):
            label += f" ({role})"
        if "contribution" in row:
            label += f" {row['contribution']:+d}"
        row_args = {
            key: value for key, value in row.items() if key not in ("time_us", "claim_start_us", "kernel_start_us")
        }
        row_args["worker_role"] = role
        if row["kind"] == "claim":
            yield {
                **location,
                "ph": "i",
                "s": "t",
                "ts": row["claim_start_us"],
                "name": f"Claim start · {name}",
                "args": {**row_args, "kind": "claim_start", "cycles": row["claim_start_cycles"]},
            }
        yield {
            **location,
            "ph": "i",
            "s": "t",
            "ts": row["time_us"],
            "name": f"{label} · {name}",
            "args": row_args,
        }


def trace_events(capture, pid, task_name, *, deps_edges=None, deps_kernel_map=None, deps_block_map=None):
    blocks = defaultdict(list)
    for row in capture["records"]:
        blocks[row["run_epoch"], row["task_id"], row["block_index"]].append(row)

    issues = _task_completion_issues(capture, blocks, deps_edges, deps_kernel_map, deps_block_map)
    predecessors = defaultdict(set)
    if issues:
        for pred, successors in (deps_edges or {}).items():
            for succ in successors:
                predecessors[succ].add(pred)

    by_cluster = defaultdict(list)
    for (epoch, task_id, block), rows in blocks.items():
        rows.sort(key=lambda row: row["time_us"])
        placement = next((row for row in rows if row["kind"] == "claim"), rows[0])
        preds = predecessors.get(task_id, ())
        times = [ts for pred in preds if (ts := issues.get((epoch, pred))) is not None]
        ideal_ready = max(times) if times and len(times) == len(preds) else None
        start, end, spans, ideal_ready = _milestones(rows, ideal_ready)
        by_cluster[placement["cluster_index"]].append((start, end, epoch, task_id, block, spans, rows, ideal_ready))

    yield _meta(pid, "process_name", {"name": "Scheduler View"})
    yield _meta(pid, "process_sort_index", {"sort_index": 3})
    next_tid = 1
    for cluster, entries in sorted(by_cluster.items()):
        lanes = []
        lane_count = 0
        for start, end, _epoch, task_id, block, spans, rows, ideal_ready in sorted(
            entries, key=lambda entry: entry[:5]
        ):
            if lanes and lanes[0][0] < start:
                _, tid = heappop(lanes)
            else:
                tid = next_tid
                next_tid += 1
                lane_count += 1
                name = f"Cluster {cluster:02d}"
                if lane_count > 1:
                    name += f" / {lane_count}"
                yield _meta(pid, "thread_name", {"name": name}, tid)
                yield _meta(pid, "thread_sort_index", {"sort_index": tid}, tid)
            heappush(lanes, (end, tid))
            name = f"{task_name(task_id)} [block {block}/{rows[0]['block_count']}]"
            yield from _block_events(rows, pid, tid, name, spans, ideal_ready)
    if capture["dropped"]:
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
