# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Check fixed cluster ownership, real lane returns and non-overlapping descriptor generations."""

from collections import defaultdict

from .native_causality import MAX, MIX, SYNC, point, require


def _group(audit, block):
    owner = block["owner"]
    ref, cluster, proxy, generation = (owner[k] for k in ("task_ref", "cluster_index", "worker_index", "group_seq"))
    mask = owner["active_mask"]
    peers = audit.workers[proxy]["cluster_workers"]
    active = [lane for lane in range(3) if mask & 1 << lane]
    accountant_lane = 1 if 1 in active else active[0]
    require(
        audit.positions[proxy] == (cluster, 1) and 0 < generation < MAX and generation % 2 == 1,
        "invalid MIX publisher or generation",
    )
    require(owner["completion_worker"] == peers[accountant_lane], "MIX completion from the wrong computing member")
    audit.chain(owner, "claim_after", "publish_before", "publish_after")
    audit.chain(owner, "releases_observed", "completion_before", "completion_after")
    publication = audit.causal.bounded(
        ("MIX descriptor", ref, owner["block_idx"]), point(owner, "publish_before"), point(owner, "publish_after")
    )
    lanes = {lane["index"]: lane for lane in block["lanes"]}
    require(set(lanes) == set(active), "MIX members differ from the computing mask")
    values = [generation if lane in active else 0 for lane in range(3)]
    mapping = [lane if lane in active else 255 for lane in range(3)]
    require(owner["released_seq"] == values, "incomplete MIX FIN observation")
    arrivals = []
    for i, lane in lanes.items():
        audit.record_identity(lane, ref)
        require(
            lane["worker_index"] == peers[i]
            and lane["lane"] == i
            and lane["cluster_index"] == cluster
            and lane["block_idx"] == owner["block_idx"]
            and lane["block_num"] == owner["block_num"]
            and lane["group_seq"] == generation
            and lane["active_mask"] == lane["resource_mask"] == mask
            and lane["lane_subtasks"] == mapping,
            "MIX descriptor identity or placement mismatch",
        )
        require(lane["arrived_seq"] == values, "incomplete MIX group arrival observation")
        audit.check_opposite_source(
            ref,
            peers[i],
            {"observed_mix_floor": 0, "observed_self_floor": lane["observed_self_floor"]},
            point(lane, "arrive_before"),
        )
        audit.chain(
            lane,
            "descriptor_ready",
            "arrive_before",
            "arrive_after",
            "arrivals_observed",
            "release_before",
            "release_after",
        )
        audit.causal.add(publication, point(lane, "descriptor_ready"))
        arrival = audit.causal.bounded(
            ("arrive", ref, owner["block_idx"], i), point(lane, "arrive_before"), point(lane, "arrive_after")
        )
        release = audit.causal.bounded(
            ("FIN", ref, owner["block_idx"], i), point(lane, "release_before"), point(lane, "release_after")
        )
        arrivals.append(arrival)
        audit.causal.add(release, point(owner, "releases_observed"))
        if audit.tasks[ref]["flags"] & SYNC:
            member = audit.member(ref, peers[i])
            require(
                member["observed_self_floor"] == lane["observed_self_floor"], "inconsistent MIX self-floor observation"
            )
            audit.before(
                point(lane, "descriptor_ready"), point(member, "committed"), "member committed to unseen descriptor"
            )
            audit.before(point(member, "committed"), point(lane, "arrive_before"), "arrival without a commitment")
            audit.before(point(lane, "release_after"), point(member, "retired"), "member retired before publishing FIN")
        execution = audit.executions[block["index"] + active.index(i)]
        require(
            execution["worker_index"] == peers[i] and execution["group_seq"] == generation,
            "execution belongs to another MIX generation or lane",
        )
        require(
            execution["observed_self_floor"] == lane["observed_self_floor"], "inconsistent MIX self-floor observation"
        )
        require(
            lane["arrivals_observed"][0] <= execution["publication_before_cycles"],
            "MIX payload publication before local group handshake",
        )
        audit.causal.add(point(lane, "arrivals_observed"), audit.payload_points[execution["index"]][0])
        audit.before(
            point(execution, "trace_published"), point(lane, "release_before"), "MIX FIN before local trace/return"
        )
    for lane in lanes.values():
        for arrival in arrivals:
            audit.causal.add(arrival, point(lane, "arrivals_observed"))


def audit_mix(audit):
    clusters = defaultdict(list)
    for block in audit.blocks.values():
        owner = block["owner"]
        if audit.tasks[owner["task_ref"]]["flags"] & MIX:
            _group(audit, block)
            clusters[owner["cluster_index"]].append(owner)
    for groups in clusters.values():
        groups.sort(key=lambda owner: owner["group_seq"])
        require(
            [g["group_seq"] for g in groups] == list(range(1, 2 * len(groups), 2)),
            "missing or duplicated MIX generation",
        )
        for previous, current in zip(groups, groups[1:]):
            require(previous["order_key"] <= current["order_key"], "decreasing MIX rank within a cluster")
            audit.before(
                point(previous, "completion_after"),
                point(current, "publish_before"),
                "MIX descriptor reused before the previous accountant retired it",
            )
