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


def _milestones(rows):
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
    if by_kind["claim"] and by_kind["ready"]:
        spans.append(
            ("Claim → ready", by_kind["claim"][0]["time_us"], by_kind["ready"][0]["time_us"], "thread_state_runnable")
        )
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
    return min(ticks + starts), max(ticks), spans


def _block_events(rows, pid, tid, name, spans):
    block = rows[0]
    args = {key: block[key] for key in ("run_epoch", "task_id", "block_index", "block_count")}
    location = {"pid": pid, "tid": tid, "cat": "percounter_task_timing"}
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
    for row in rows:
        label = _LABELS[row["kind"]]
        role = ("C", "V0", "V1")[row["cluster_lane"]]
        if row["kind"] in ("ready", "kernel_end"):
            label += f" ({role})"
        if "contribution" in row:
            label += f" {row['contribution']:+d}"
        yield {
            **location,
            "ph": "i",
            "s": "t",
            "ts": row["time_us"],
            "name": f"{label} · {name}",
            "args": {
                **{key: value for key, value in row.items() if key not in ("time_us", "kernel_start_us")},
                "worker_role": role,
            },
        }


def trace_events(capture, pid, task_name):
    blocks = defaultdict(list)
    for row in capture["records"]:
        blocks[row["run_epoch"], row["task_id"], row["block_index"]].append(row)

    by_cluster = defaultdict(list)
    for (epoch, task_id, block), rows in blocks.items():
        rows.sort(key=lambda row: row["time_us"])
        placement = next((row for row in rows if row["kind"] == "claim"), rows[0])
        start, end, spans = _milestones(rows)
        by_cluster[placement["cluster_index"]].append((start, end, epoch, task_id, block, spans, rows))

    yield _meta(pid, "process_name", {"name": "Scheduler View"})
    yield _meta(pid, "process_sort_index", {"sort_index": 3})
    next_tid = 1
    for cluster, entries in sorted(by_cluster.items()):
        lanes = []
        lane_count = 0
        for start, end, _epoch, task_id, block, spans, rows in sorted(entries, key=lambda entry: entry[:5]):
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
            yield from _block_events(rows, pid, tid, name, spans)
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
