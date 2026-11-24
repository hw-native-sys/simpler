# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Physical slot lifetimes sampled by the Percounter L2 executor."""

from collections import defaultdict
from heapq import heappop, heappush

from .scheduler_view import _meta, capture_status

_ROLES = ("AIC", "AIV0", "AIV1")


def _block_spans(rows):
    """Pair physical transitions by execution identity, never by thread adjacency."""
    kinds = defaultdict(list)
    for row in rows:
        kinds[row["kind"]].append(row)
    if not kinds["claim"] or "slot" not in kinds["claim"][0]:
        return
    (claim,) = kinds["claim"]
    cluster, slot, mix = claim["cluster_index"], claim["slot"], claim["mix"]
    owner = _ROLES[claim["cluster_lane"]]
    prefix = f"C{cluster:02d} / "
    group = f"{prefix}MIX{slot}"
    proxy = f"{group} (AIV0 proxy)" if mix else f"{prefix}{owner} slot{slot}"

    def member(row):
        return f"{group}/{_ROLES[row['cluster_lane']]}" if mix else proxy

    def interval(track, label, start, end, actor):
        return track, label, start, end, {"worker_id": actor["worker_id"], "slot": slot}

    claimed = claim["time_us"]
    yield interval(proxy, "Task claim", claim["claim_start_us"], claimed, claim)
    bound = claimed
    if mix:
        if not kinds["mix_publish"]:
            return
        bound = kinds["mix_publish"][0]["time_us"]
        yield interval(proxy, "Prepare MIX descriptor → SSBUF publish", claimed, bound, claim)
    proof = kinds["inputs_ready"][0] if kinds["inputs_ready"] else None
    miss = kinds["inputs_pending"][0] if kinds["inputs_pending"] else None
    observer = miss or proof
    if observer:
        lane = member(observer)
        if miss:
            yield interval(lane, "Bound → first pending observation", bound, miss["time_us"], observer)
        if proof:
            label = "Predecessor pending → inputs observed" if miss else "Bound → inputs observed"
            yield interval(lane, label, miss["time_us"] if miss else bound, proof["time_us"], proof)

    yield from _execution_spans(kinds, member, interval, mix, proof)
    yield from _completion_spans(kinds, mix, member, interval)


def _execution_spans(kinds, member, interval, mix, proof):
    commits = {r["worker_id"]: r for r in kinds["entry_commit"]}
    selected = {r["worker_id"]: r for r in kinds["ready"]}
    for worker, row in selected.items():
        track = member(row)
        commit = commits.get(worker)
        if commit:
            if proof:
                yield interval(track, "Inputs proven → entry commitment", proof["time_us"], commit["time_us"], row)
            label = "MIX rendezvous → selection" if mix else "sync_start arrival → selection"
            yield interval(track, label, commit["time_us"], row["time_us"], row)
        elif proof:
            yield interval(track, "Inputs observed → selection", proof["time_us"], row["time_us"], row)

    for row in kinds["kernel_end"]:
        track = member(row)
        if "dispatch_us" not in row:
            continue
        chosen = selected.get(row["worker_id"])
        if chosen:
            yield interval(track, "Selection → executor pickup", chosen["time_us"], row["dispatch_us"], row)
        yield interval(track, "Kernel entry preparation", row["dispatch_us"], row["kernel_start_us"], row)
        yield interval(track, "Kernel return → output fence", row["time_us"], row["finish_us"], row)


def _completion_spans(kinds, mix, member, interval):
    kernels = {row["worker_id"]: row for row in kinds["kernel_end"]}
    for issue in kinds["completion_issue"]:
        worker = issue["worker_id"]
        kernel = kernels.get(worker)
        if kernel and "finish_us" in kernel:
            start, label = kernel["finish_us"], "Output fence → completion issue"
        elif not mix and (chosen := next((r for r in kinds["ready"] if r["worker_id"] == worker), None)):
            start, label = chosen["time_us"], "Selection → completion issue (no kernel recorded)"
        else:
            continue
        yield interval(member(issue), label, start, issue["time_us"], issue)


def schedule_events(capture, pid, task_name):
    blocks = defaultdict(list)
    for row in capture["records"]:
        blocks[row["run_epoch"], row["task_id"], row["block_index"]].append(row)
    tracks = defaultdict(list)
    deferred = defaultdict(list)
    for (epoch, task, block), rows in blocks.items():
        args = {"run_epoch": epoch, "task_id": task, "block_index": block}
        prefix = f"{task_name(task)}-b{block}"
        for track, label, start, end, actor in _block_spans(rows):
            if start < end:
                tracks[track].append((start, end, f"{prefix}-{label}", {**args, **actor}))
        registrations = {r["deferred_group"]: r for r in rows if r["kind"] == "defer_registered"}
        for row in rows:
            if row["kind"] != "deferred_completion_issue":
                continue
            begin = registrations.get(row["deferred_group"])
            if begin and begin["time_us"] < row["time_us"]:
                deferred[begin["cluster_index"]].append(
                    (
                        begin["time_us"],
                        row["time_us"],
                        f"{prefix}-Defer registration → settlement issue",
                        {
                            **args,
                            "group": row["deferred_group"],
                            "registered_by": begin["worker_id"],
                            "settled_by": row["worker_id"],
                        },
                    )
                )
    for cluster, spans in deferred.items():
        available = []
        for span in sorted(spans, key=lambda span: span[:3]):
            index = heappop(available)[1] if available and available[0][0] <= span[0] else len(available)
            tracks[f"C{cluster:02d} / deferred {index}"].append(span)
            heappush(available, (span[1], index))
    if not tracks and not capture.get("dropped"):
        return
    yield _meta(pid, "process_name", {"name": "AICore Schedule (L2)"})
    yield _meta(pid, "process_sort_index", {"sort_index": 2})
    for cookie, (name, spans) in enumerate(sorted(tracks.items())):
        lane = {"pid": pid, "cat": "Percounter L2", "id2": {"local": cookie}}
        spans.sort(key=lambda bar: bar[:3])
        yield {**lane, "ph": "n", "ts": spans[0][0], "name": name}
        for start, end, label, args in spans:
            yield {**lane, "ph": "b", "ts": start, "name": label, "args": args}
            yield {**lane, "ph": "e", "ts": end}

    if capture.get("dropped"):
        yield from capture_status(capture, pid)


def worker_lanes(capture):
    positions = defaultdict(set)
    for row in capture["records"]:
        positions[row["worker_id"]].add((row["cluster_index"], row["cluster_lane"]))
    lanes = {}
    for worker, observed in positions.items():
        if len(observed) == 1:
            cluster, lane = next(iter(observed))
            lanes[worker] = (f"C{cluster:02d} / {_ROLES[lane]}", cluster * 3 + lane)
    return lanes
