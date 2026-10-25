# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""GM Mix resource handshakes and retained task seals, independent of TaskDone."""

from collections import defaultdict

from .native_causality import MAX, MIX, point, require


def _group(audit, block):
    owner = block["owner"]
    ref, worker, group = owner["task_ref"], owner["worker_index"], owner["group_seq"]
    require(audit.positions[worker][1] == 0 and 0 < group < MAX, "invalid GM group coordinator/sequence")
    audit.chain(owner, "claim_after", "publish_before", "publish_after", "releases_observed", "completion_before")
    publication = audit.causal.bounded(
        ("group", ref, owner["block_idx"]), point(owner, "publish_before"), point(owner, "publish_after")
    )
    lanes = {row["index"]: row for row in block["lanes"]}
    require(set(lanes) == {0, 1, 2}, "missing GM resource lane")
    require(owner["released_seq"] == [group] * 3, "stale GM release observation")
    arrivals, releases = {}, {}
    for index, lane in lanes.items():
        arrivals[index] = audit.causal.bounded(
            ("arrive", ref, owner["block_idx"], index), point(lane, "arrive_before"), point(lane, "arrive_after")
        )
        releases[index] = audit.causal.bounded(
            ("release", ref, owner["block_idx"], index), point(lane, "release_before"), point(lane, "release_after")
        )
    mapping = [lane if owner["active_mask"] & (1 << lane) else 255 for lane in range(3)]
    for index, lane in lanes.items():
        audit.record_identity(lane, ref)
        peer = audit.workers[worker]["cluster_workers"][index]
        require(
            lane["worker_index"] == peer
            and lane["lane"] == index
            and lane["cluster_index"] == owner["cluster_index"]
            and lane["block_idx"] == owner["block_idx"]
            and lane["block_num"] == owner["block_num"]
            and lane["active_mask"] == owner["active_mask"]
            and lane["group_seq"] == group
            and lane["resource_mask"] == 7
            and lane["lane_subtasks"] == mapping,
            "GM descriptor identity/placement mismatch",
        )
        require(lane["arrived_seq"] == [group] * 3, "stale GM arrival observation")
        audit.chain(
            lane,
            "descriptor_ready",
            "arrive_before",
            "arrive_after",
            "arrivals_observed",
            "release_before",
            "release_after",
        )
        member = audit.member(ref, peer)
        audit.before(point(member, "observed"), point(lane, "descriptor_ready"), "GM lane bypassed its local frontier")
        audit.causal.add(publication, point(lane, "descriptor_ready"))
        if index == 0:
            audit.before(
                point(owner, "publish_after"),
                point(lane, "descriptor_ready"),
                "coordinator used its unpublished descriptor",
            )
        for arrival in arrivals.values():
            audit.causal.add(arrival, point(lane, "arrivals_observed"))
        audit.causal.add(releases[index], point(owner, "releases_observed"))
        audit.before(point(lane, "release_after"), point(member, "closed"), "GM member closed before lane release")
        if mapping[index] != 255:
            trace = block["index"] + (owner["active_mask"] & ((1 << index) - 1)).bit_count()
            execution = audit.executions[trace]
            require(
                execution["worker_index"] == peer
                and execution["subtask_slot"] == index
                and execution["group_seq"] == group,
                "GM execution belongs to another group/lane",
            )
            require(
                lane["arrivals_observed"][0] <= execution["publication_before_cycles"],
                "GM payload prepared before resources arrived",
            )
            audit.before(
                point(lane, "arrivals_observed"), point(execution, "permit"), "GM entered before resource handshake"
            )
            audit.before(
                point(execution, "trace_published"),
                point(lane, "release_before"),
                "GM lane released before output/trace retirement",
            )
    audit.before(
        point(lanes[0], "release_after"),
        point(owner, "releases_observed"),
        "coordinator aggregate preceded its release",
    )


def audit_mix(audit):
    clusters = defaultdict(list)
    task_clusters = defaultdict(list)
    for block in audit.blocks.values():
        owner = block["owner"]
        if not audit.tasks[owner["task_ref"]]["flags"] & MIX:
            continue
        _group(audit, block)
        clusters[owner["cluster_index"]].append(owner)
        task_clusters[(owner["task_ref"], owner["cluster_index"])].append(owner)
    for groups in clusters.values():
        groups.sort(key=lambda row: row["group_seq"])
        require(
            [row["group_seq"] for row in groups] == list(range(1, len(groups) + 1)),
            "missing or duplicate cluster group sequence",
        )
        for previous, current in zip(groups, groups[1:], strict=False):
            audit.before(
                point(previous, "completion_after"),
                point(current, "claim_before"),
                "GM mailbox reused before prior group retired",
            )
    for ref, task in audit.tasks.items():
        if not task["flags"] & MIX:
            continue
        for worker in audit.task_members[ref]:
            member = audit.member(ref, worker)
            cluster, lane = audit.positions[worker]
            groups = task_clusters[(ref, cluster)]
            last = max((row["group_seq"] for row in groups), default=0)
            require(
                member["last_group_seq"] == last
                and member["close_seal"] == last + 1
                and audit.admissions[ref]["close_seals"][cluster] == last + 1,
                "missing or stale retained Mix close seal",
            )
            if lane:
                require(member["closed_cursor"] == MAX, "Mix AIV close cursor is not an actual observation")
            seal = ("seal", ref, cluster)
            audit.causal.add(seal, point(member, "closed"))
            if lane == 0:
                audit.causal.add(audit.close_reads[(ref, worker)], seal)
            for group in groups:
                audit.causal.add(point(group, "completion_after"), seal)
