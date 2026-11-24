# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Render the measured intervals between successive kernels on one HBG worker."""

from bisect import bisect_right
from collections import defaultdict

from .formatter import display
from .gap_evidence import build_mix_publications, latest_predecessor_returns


def _interval_union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _clip(intervals, ends, start, end):
    """Intersect a window with sorted, disjoint (start, end, *data) intervals."""
    for index in range(bisect_right(ends, start), len(intervals)):
        left, right, *data = intervals[index]
        if left >= end:
            break
        yield max(start, left), min(end, right), data


def kernel_gap_events(profiles, streams, tasks, fanin_map):
    lanes = {
        stream["scheduler_id"]: (index, stream)
        for index, stream in enumerate(streams or [])
        if stream.get("producer") == "aicore"
    }
    publications = build_mix_publications(profiles, tasks, streams or [])
    predecessor_returns = latest_predecessor_returns(tasks, fanin_map)
    events = []
    named_lanes = set()
    for profile in profiles:
        if not profile["complete"]:
            continue
        kernels = sorted(
            (row for row in profile["rows"] if row["kind"] == "scalar_kernel"),
            key=lambda row: row["start_cycles"],
        )
        if len(kernels) < 2:
            continue
        worker, epoch = profile["worker_id"], profile["run_epoch"]
        index, stream = lanes.get(profile["scheduler_id"], (profile["scheduler_id"], {}))
        track = {"pid": 2, "tid": 30000 + index * 10 + 2}
        if track["tid"] not in named_lanes:
            core = (stream.get("core_type") or "worker").upper()
            name = f"{core}_{worker} / Kernel gaps"
            events.append({**track, "ph": "M", "cat": "__metadata", "name": "thread_name", "args": {"name": name}})
            named_lanes.add(track["tid"])

        exclusive = [
            (row["start_cycles"], row["end_cycles"], row.get("hbg_stage", "UNATTRIBUTED"))
            for row in profile["exclusive"]
        ]
        idle = _interval_union(profile.get("no_progress_intervals", []))
        exclusive_ends = [end for _, end, _ in exclusive]
        idle_ends = [end for _, end in idle]
        frequency = profile["clock_freq_hz"]

        def to_us(cycles):
            return cycles * 1_000_000 / frequency

        for previous, following in zip(kernels, kernels[1:]):
            start, end = previous["end_cycles"], following["start_cycles"]
            if end <= start:
                continue
            by_stage = defaultdict(int)
            for left, right, (stage,) in _clip(exclusive, exclusive_ends, start, end):
                by_stage[stage] += right - left
            if sum(by_stage.values()) != end - start:
                raise ValueError(f"Kernel gap partition does not cover worker {worker}, run {epoch}")
            overlaps = [(left, right) for left, right, _ in _clip(idle, idle_ends, start, end)]
            before, after = previous["task_id"], following["task_id"]
            args = {
                "worker_id": worker,
                "run_epoch": epoch,
                "previous_task_id": before,
                "previous_reg_task_id": previous["reg_task_id"],
                "next_task_id": after,
                "next_reg_task_id": following["reg_task_id"],
                "no_progress_us": to_us(sum(right - left for left, right in overlaps)),
                "exclusive_us": {stage: to_us(cycles) for stage, cycles in sorted(by_stage.items())},
                **predecessor_returns.get((epoch, after), {}),
                **publications.get((epoch, worker, following["reg_task_id"]), {}),
            }
            ts = previous["end_time_us"]
            events.append(
                {
                    **track,
                    "ph": "X",
                    "cat": "kernel_gap",
                    "name": f"KERNEL GAP · {display(before)} → {display(after)}",
                    "ts": ts,
                    "dur": to_us(end - start),
                    "cname": "generic_work",
                    "args": args,
                }
            )
            for left, right in overlaps:
                events.append(
                    {
                        **track,
                        "ph": "X",
                        "cat": "scheduler_no_progress",
                        "name": "No scheduler progress",
                        "ts": ts + to_us(left - start),
                        "dur": to_us(right - left),
                        "cname": "thread_state_iowait",
                        "args": {"blocking_condition": "Not recorded"},
                    }
                )
    return events
