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


_CHECKPOINTS = {"inputs_pending", "inputs_ready", "mix_publish", "entry_commit"}


def _validate_l2_checkpoints(row):
    if row["kind"] == "claim" and "slot" in row:
        if (
            type(row["slot"]) is not int
            or row["slot"] < 0
            or type(row.get("mix")) is not bool
            or type(row.get("participant_mask")) is not int
            or not 0 <= row["participant_mask"] <= 7
        ):
            raise ValueError("invalid percounter claim location")
    if row["kind"] == "kernel_end" and ("dispatch_cycles" in row or "finish_cycles" in row):
        dispatch, finish = row.get("dispatch_cycles"), row.get("finish_cycles")
        if (
            type(dispatch) is not int
            or type(finish) is not int
            or not 0 < dispatch <= row["kernel_start_cycles"] <= row["cycles"] <= finish
        ):
            raise ValueError("invalid percounter execution checkpoints")


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
    return section


def _validate_task_event(row):
    if (
        not isinstance(row, dict)
        or not isinstance(row.get("kind"), str)
        or (row["kind"] not in _LABELS and row["kind"] not in _CHECKPOINTS)
    ):
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
    _validate_l2_checkpoints(row)
    if row["kind"] in ("completion_issue", "deferred_completion_issue"):
        contribution = row.get("contribution")
        if type(contribution) is not int or contribution != 1:
            raise ValueError("invalid percounter completion contribution")


def decode_capture(section, to_us):
    return {
        **section,
        "records": [
            {
                **row,
                "time_us": to_us(row["cycles"]),
                **(
                    {"dispatch_us": to_us(row["dispatch_cycles"]), "finish_us": to_us(row["finish_cycles"])}
                    if "dispatch_cycles" in row
                    else {}
                ),
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


def _milestones(rows, ideal_ready, include_kernel_execution):
    """Use recorded endpoints for every span, including in partial captures."""
    by_kind = defaultdict(list)
    for row in rows:
        by_kind[row["kind"]].append(row)
    ticks = [row["time_us"] for row in rows]
    starts = [row["kernel_start_us"] for row in by_kind["kernel_end"]]
    completion = by_kind["completion_issue"] + by_kind["deferred_completion_issue"]
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
        if include_kernel_execution:
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


def _block_events(rows, pid, tid, name, spans, ideal_ready, prefix):
    block = rows[0]
    args = {key: block[key] for key in ("run_epoch", "task_id", "block_index", "block_count")}

    def stage_name(stage):
        return f"{prefix}-{stage}" if prefix is not None else f"{stage} · {name}"

    if prefix is not None:
        args["kernel_slice_label"] = name
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
                "name": stage_name(label),
                "cname": color,
                "args": args,
            }
    if ideal_ready is not None:
        yield {
            **location,
            "ph": "i",
            "s": "t",
            "ts": ideal_ready,
            "name": stage_name("Ideal ready (estimated)"),
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
        if prefix is not None:
            row_args["kernel_slice_label"] = name
        if row["kind"] == "claim":
            yield {
                **location,
                "ph": "i",
                "s": "t",
                "ts": row["claim_start_us"],
                "name": stage_name("Claim start"),
                "args": {**row_args, "kind": "claim_start", "cycles": row["claim_start_cycles"]},
            }
        yield {
            **location,
            "ph": "i",
            "s": "t",
            "ts": row["time_us"],
            "name": stage_name(label),
            "args": row_args,
        }


def _execution_predecessors(deps_edges, folded_task_ids, inline_completed_task_ids):
    predecessors = defaultdict(set)
    completed = set(inline_completed_task_ids)
    for pred, successors in deps_edges.items():
        if pred in completed:
            continue
        for succ in successors:
            predecessors[succ].add(pred)

    folded = set(folded_task_ids)
    for task_id, parents in predecessors.items():
        if task_id in folded:
            continue
        pending = parents & folded
        seen = set(pending)
        while pending:
            for pred in predecessors.get(pending.pop(), ()):
                if pred not in folded:
                    parents.add(pred)
                elif pred not in seen:
                    pending.add(pred)
                    seen.add(pred)
        parents.difference_update(folded)
    return predecessors


def trace_events(
    capture,
    pid,
    task_name,
    *,
    include_kernel_execution=True,
    l2_task_display=None,
    deps_edges=None,
    deps_kernel_map=None,
    deps_block_map=None,
):
    blocks = defaultdict(list)
    for row in capture["records"]:
        if row["kind"] not in _CHECKPOINTS:
            blocks[row["run_epoch"], row["task_id"], row["block_index"]].append(row)

    issues = _task_completion_issues(capture, blocks, deps_edges, deps_kernel_map, deps_block_map)
    predecessors = (
        _execution_predecessors(deps_edges, capture["folded_task_ids"], capture["inline_completed_task_ids"])
        if issues
        else {}
    )

    by_cluster = defaultdict(list)
    for (epoch, task_id, block), rows in blocks.items():
        rows.sort(key=lambda row: row["time_us"])
        placement = next((row for row in rows if row["kind"] == "claim"), rows[0])
        preds = predecessors.get(task_id, ())
        times = [ts for pred in preds if (ts := issues.get((epoch, pred))) is not None]
        ideal_ready = max(times) if times and len(times) == len(preds) else None
        start, end, spans, ideal_ready = _milestones(rows, ideal_ready, include_kernel_execution)
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
            prefix = f"{l2_task_display(task_id)}-b{block}" if l2_task_display is not None else None
            yield from _block_events(rows, pid, tid, name, spans, ideal_ready, prefix)
    yield from capture_status(capture, pid)


def capture_status(capture, pid):
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
