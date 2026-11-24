# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Task-bound resident L2 milestones. Slot residence can overlap work in another slot."""

from collections import defaultdict


def attach_checkpoints(task, identity, cycles, to_us):
    if (
        not isinstance(cycles, list)
        or len(cycles) not in (4, 6)
        or any(type(v) is not int or not 0 < v < 1 << 64 for v in cycles)
        or (len(cycles) == 6 and (identity["group_kind"] != 2 or cycles[4] > cycles[5] or cycles[5] > cycles[0]))
    ):
        raise ValueError("invalid HBG L2 schedule checkpoints")
    begin, bound, fanin, retired, *proxy = map(to_us, cycles)
    edges = (
        begin,
        bound,
        fanin,
        task["dispatch_time_us"],
        task["receive_time_us"],
        task["start_time_us"],
        task["end_time_us"],
        task["finish_time_us"],
        retired,
    )
    if any(left > right for left, right in zip(edges, edges[1:])):
        raise ValueError("out-of-order HBG L2 schedule checkpoints")
    task["schedule_checkpoints"] = (begin, bound, fanin, retired)
    if "pending_fanin" in identity:
        miss = identity["pending_fanin"]
        if (
            not isinstance(miss, list)
            or len(miss) != 2
            or type(miss[0]) is not int
            or not 0 <= miss[0] < 1 << 64
            or type(miss[1]) is not int
            or not cycles[1] <= miss[1] <= cycles[2]
        ):
            raise ValueError("invalid HBG L2 pending predecessor observation")
        task["first_pending_fanin"] = (miss[0], to_us(miss[1]))
    if identity["cluster_lane"] == 1:
        task["coordinator_complete_cycles"] = cycles[3]
    if "mix_claim_cycles" in identity:
        stamps = identity["mix_claim_cycles"]
        if (
            not proxy
            or not isinstance(stamps, list)
            or len(stamps) != 4
            or any(type(v) is not int or not 0 <= v < 1 << 64 for v in stamps)
        ):
            raise ValueError("invalid HBG L2 MIX claim checkpoints")
        previous, released, claim, claimed = stamps
        if not (
            bool(previous) == bool(released) and previous <= released <= claim and 0 < claim <= claimed <= cycles[4]
        ):
            raise ValueError("out-of-order HBG L2 MIX claim checkpoints")
        task["mix_claim_timing"] = (previous, to_us(released) if released else None, to_us(claim), to_us(claimed))
    if proxy:
        task["mix_proxy_timing"] = tuple(proxy)


def _lane_name(track):
    cluster, role, slot = track
    member = ("AIC", "AIV0", "AIV1", "AIV0 proxy")[role]
    prefix = f"C{cluster:02d} / "
    if role == 3 and slot == 0:
        return f"{prefix}AIV0 dispatch"
    if slot < 4:
        return f"{prefix}{member} slot{slot}"
    # Perfetto sorts JSON tracks by lower(name). Group by MIX bank, with its
    # publisher (space) immediately before its members (slash).
    bank = f"{prefix}MIX{slot - 4}"
    return f"{bank} ({member})" if role == 3 else f"{bank}/{member}"


def schedule_events(tasks, display):
    pid = 3
    events = [
        {"ph": "M", "pid": pid, "name": "process_name", "args": {"name": "AICore Schedule (L2)"}},
        {"ph": "M", "pid": pid, "name": "process_sort_index", "args": {"sort_index": 3}},
    ]
    intervals = defaultdict(list)
    groups = defaultdict(list)
    coordinator_completions = {}

    def span(track, name, start, end, args):
        if start < end:
            intervals[track].append((start, end, f"{args['task']}-b{args['block_idx']}-{name}", args))

    def pickup_spans(track, mix, fanin, ready, picked, args):
        if mix:
            span(
                track,
                "MIX inputs observed to executor pickup",
                fanin,
                picked,
                {
                    **args,
                    "ready_time_us": ready,
                    "order_and_arrival_us": ready - fanin,
                    "ready_to_pickup_us": picked - ready,
                },
            )
        else:
            span(track, "Predicate and READY publication", fanin, ready, args)
            span(track, "READY to executor pickup", ready, picked, args)

    for task in tasks:
        if "schedule_checkpoints" not in task:
            continue
        begin, bound, fanin, retired = task["schedule_checkpoints"]
        cluster, member, slot = task["cluster_index"], task["cluster_lane"], task["pending_slot"]
        mix = task["group_kind"] == 2
        # Distinct slot tracks retain overlapping waits without nesting them on the executing core.
        track = (cluster, member, slot)
        args = {
            "task": display(task["task_id"]),
            "task_id": task["task_id"],
            "block_idx": task["block_idx"],
            "worker_id": task["core_id"],
            "run_epoch": task["run_epoch"],
            "pending_slot": slot,
        }
        if "coordinator_complete_cycles" in task:
            key = (task["run_epoch"], cluster, task["coordinator_complete_cycles"])
            coordinator_completions[key] = (track, retired, args)
        if mix:
            groups[(task["run_epoch"], task["task_id"], task["block_idx"])].append(task)
        ready, picked = task["dispatch_time_us"], task["receive_time_us"]
        span(track, "MIX member bind" if mix else "Claim and bind", begin, bound, args)
        if "first_pending_fanin" in task:
            predecessor, first_miss = task["first_pending_fanin"]
            span(track, "Bound to first pending observation", bound, first_miss, args)
            span(
                track,
                "Predecessor pending to inputs observed",
                first_miss,
                fanin,
                {**args, "first_pending_predecessor": display(predecessor), "first_pending_task_id": predecessor},
            )
        else:
            span(track, "Bound to predecessors observed", bound, fanin, args)
        pickup_spans(track, mix, fanin, ready, picked, args)
        span(track, "Kernel entry preparation", picked, task["start_time_us"], args)
        span(track, "Kernel return to completion observation", task["end_time_us"], task["finish_time_us"], args)
        span(track, "Local completion and release", task["finish_time_us"], retired, args)

    for (epoch, task_id, block), members in groups.items():
        owners = [row for row in members if "mix_proxy_timing" in row]
        if not owners:
            continue  # A partial export can omit the member carrying the proxy's evidence.
        if len(owners) != 1:
            raise ValueError("duplicate HBG L2 MIX proxy evidence for one block")
        owner = owners[0]
        track = (owner["cluster_index"], 3, owner["pending_slot"])
        start, arm = owner["mix_proxy_timing"]
        observed = {row["subtask_slot"]: row["schedule_checkpoints"][0] for row in members}
        if min(observed.values()) < arm:
            raise ValueError("HBG L2 MIX observation precedes its descriptor")
        args = {
            "run_epoch": epoch,
            "task": display(task_id),
            "task_id": task_id,
            "block_idx": block,
            "generation": owner["group_seq"],
            "member_observation_us_after_arm": {("AIC", "AIV0", "AIV1")[k]: v - arm for k, v in observed.items()},
        }
        if "mix_claim_timing" in owner:
            previous, released, claim, claimed = owner["mix_claim_timing"]
            dispatch = (owner["cluster_index"], 3, 0)
            if released is not None:
                prior = coordinator_completions.get((epoch, owner["cluster_index"], previous))
                if prior is not None:
                    prior_track, complete, prior_args = prior
                    span(prior_track, "L2 trace writeback and retire", complete, released, prior_args)
                span(dispatch, "Dispatch to MIX claim", released, claim, args)
            span(dispatch, "MIX block claim", claim, claimed, args)
            span(dispatch, "Claim to descriptor preparation", claimed, start, args)
        span(track, "MIX descriptor preparation", start, arm, args)
        span(track, "SSBUF arm to last observed member", arm, max(observed.values()), args)

    # Perfetto appends the OS tid to thread tracks. A process-local async track
    # takes its name from its first event, at the lane's first observed checkpoint.
    for label, track in sorted((_lane_name(track), track) for track in intervals):
        cluster, role, slot = track
        spans = intervals[track]
        cookie = cluster * 32 + role * 8 + slot
        lane = {"pid": pid, "cat": "HBG L2", "id2": {"local": cookie}}
        spans.sort(key=lambda row: row[0])
        events.append({**lane, "ph": "n", "ts": spans[0][0], "name": label})
        for start, end, name, args in spans:
            events.append({**lane, "ph": "b", "ts": start, "name": name, "args": args})
            events.append({**lane, "ph": "e", "ts": end})
    return events
