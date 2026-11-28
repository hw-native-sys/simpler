# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Convert physical WSE PE records on the simulator monotonic clock to Chrome trace events."""

from collections import defaultdict


def _physical_tracks(cores, pes):
    events = []
    for core in range(cores):
        process = {"ph": "M", "cat": "__metadata", "pid": core + 1}
        events.append({**process, "name": "process_name", "args": {"name": f"Core {core}"}})
        events.append({**process, "name": "process_sort_index", "args": {"sort_index": core}})
        for pe in range(pes):
            lane = {**process, "tid": cores + 2 + core * pes + pe}
            events.append({**lane, "name": "thread_name", "args": {"name": f"PE {pe}"}})
            events.append({**lane, "name": "thread_sort_index", "args": {"sort_index": pe}})
    return events


def _execution_ends(records):
    """A physical lane has at most one live kernel; both ends carry the same Block identity."""
    active, ends = {}, {}
    for row in records:
        lane = row["core_id"], row["pe_id"]
        if row["kind"] == "KERNEL_BEGIN":
            if lane in active:
                raise ValueError(f"overlapping WSE kernels on lane {lane}")
            active[lane] = row
        elif row["kind"] == "KERNEL_END":
            start = active.pop(lane, None)
            if (
                start is None
                or row["task_id"] != start["task_id"]
                or row["block_idx"] != start["block_idx"]
                or row["monotonic_ns"] < start["monotonic_ns"]
            ):
                raise ValueError(f"unmatched WSE kernel end on lane {lane}")
            ends[start["seq"]] = row
    if active:
        raise ValueError("WSE capture contains unfinished kernels")
    return ends


def generate_wse_trace(capture):  # noqa: PLR0912, PLR0915
    metadata = capture["metadata"]
    if metadata["device_clock_domain"] != "host_monotonic_ns":
        raise ValueError("WSE PE records require host monotonic timestamps")
    cores, pes = metadata["core_count"], metadata["pes_per_core"]
    tasks = {task["task_id"]: task for task in capture["tasks"]}
    records = sorted(capture["pe_records"], key=lambda row: row["seq"])
    origin = min((row["monotonic_ns"] for row in records), default=0)
    ends = _execution_ends(records)
    skipped = {(r["task_id"], r["block_idx"]) for r in records if r["kind"] == "BLOCK_SKIPPED"}
    completions = {r["task_id"]: r for r in records if r["kind"] == "TASK_DONE"}
    releases = {r["task_id"]: r for r in records if r["kind"] == "SYNC_RELEASE"}

    # A flow is anchored on the observed event that caused it and the event it enabled.
    # Keep endpoint packets with their slices, including co-timestamped events on a lane.
    flows = defaultdict(list)
    flow_id = 0

    def connect(source, destination, name):
        nonlocal flow_id
        if source["monotonic_ns"] > destination["monotonic_ns"]:
            raise ValueError(f"WSE {name} flow ends before it starts")
        flow_id += 1
        pair = {"cat": "flow", "id": flow_id, "name": name}
        flows[source["seq"]].append({**pair, "ph": "s"})
        flows[destination["seq"]].append({**pair, "ph": "f", "bp": "e"})

    for row in records:
        task_id, kind = row["task_id"], row["kind"]
        task = tasks[task_id]
        if kind == ("PE_ARRIVE" if task["sync_start"] else "KERNEL_BEGIN"):
            for producer in task["waits"]:
                if producer in completions:
                    connect(completions[producer], row, "dependency")
        if kind == "PE_ARRIVE":
            connect(row, releases[task_id], "sync_start")
        elif kind == "KERNEL_BEGIN" and task["sync_start"]:
            connect(releases[task_id], row, "sync_start")

    events = _physical_tracks(cores, pes)
    host_track = False
    for row in records:
        seq, kind = row["seq"], row["kind"]
        if kind in ("KERNEL_END", "BLOCK_SKIPPED") or (kind == "TASK_DONE" and seq not in flows):
            continue
        core, pe = row["core_id"], row["pe_id"]
        task_id = row["task_id"]
        args = {"seq": seq, "epoch": metadata["run_epoch"], "task": task_id}
        if core is None and pe is None and kind == "TASK_DONE":
            pid = tid = cores + 1
            if not host_track:
                events.append({"ph": "M", "name": "process_name", "pid": pid, "args": {"name": "WSE Host"}})
                host_track = True
        elif isinstance(core, int) and isinstance(pe, int) and 0 <= core < cores and 0 <= pe < pes:
            pid, tid = core + 1, cores + 2 + core * pes + pe
            args.update(core_id=core, pe_id=pe)
        else:
            raise ValueError(f"invalid WSE coordinates for event {seq}")
        stamp = row["monotonic_ns"]
        endpoint = {"pid": pid, "tid": tid, "ts": (stamp - origin) / 1000}
        if kind != "TASK_DONE":
            args["block_idx"] = row["block_idx"]
        if kind == "KERNEL_BEGIN":
            end = ends[seq]
            args["end_seq"] = end["seq"]
            label = "Group" if tasks[task_id]["block_num"] > 1 else "SS"
            category = "wse.skipped" if (task_id, row["block_idx"]) in skipped else "wse.kernel"
            event = {
                "ph": "X",
                "name": f"{label} · T{task_id}",
                "cat": category,
                "dur": (end["monotonic_ns"] - stamp) / 1000,
            }
        elif kind in ("PE_ARRIVE", "SYNC_RELEASE", "TASK_DONE"):
            event = {"ph": "i", "s": "t", "cat": "wse.event", "name": kind}
        else:
            raise ValueError(f"unknown WSE PE record kind: {kind}")
        events.append({**event, **endpoint, "args": args})
        events.extend({**flow, **endpoint, "args": {"seq": seq}} for flow in flows.get(seq, ()))
    return {
        "traceEvents": events,
        "displayTimeUnit": "ns",
        "metadata": {**metadata, "timeline_origin_ns": origin, "chip_swimlane_level": capture["chip_swimlane_level"]},
    }
