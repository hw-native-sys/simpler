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
    bank = audit.executions[block["index"]]["pending_slot"]
    all_arrivals = any(audit.tasks[ref]["required_floor"][i] for i in active)
    arrivals = {}
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
        expected = values if all_arrivals else [generation if peer == i else 0 for peer in range(3)]
        require(lane["arrived_seq"] == expected, "incomplete MIX group arrival observation")
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
        arrivals[i] = arrival
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
            execution["worker_index"] == peers[i]
            and execution["group_seq"] == generation
            and execution["pending_slot"] == bank,
            "execution belongs to another MIX generation, slot or lane",
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
    for i, lane in lanes.items():
        for arrival in arrivals.values() if all_arrivals else (arrivals[i],):
            audit.causal.add(arrival, point(lane, "arrivals_observed"))
    return bank


def audit_mix(audit):
    clusters = defaultdict(list)
    for block in audit.blocks.values():
        owner = block["owner"]
        if audit.tasks[owner["task_ref"]]["flags"] & MIX:
            bank = _group(audit, block)
            clusters[owner["cluster_index"]].append((owner, bank, block["lanes"]))
    for groups in clusters.values():
        groups.sort(key=lambda group: group[0]["publish_before"][1])
        by_bank = {}
        by_task = {}
        lane_retired = {}
        for number, (owner, bank, lanes) in enumerate(groups):
            require(owner["group_seq"] == 2 * number + 1, "missing or duplicated MIX generation")
            if number:
                require(
                    groups[number - 1][0]["order_key"] <= owner["order_key"], "decreasing MIX rank within a cluster"
                )
            publication = point(owner, "publish_before")
            completion = point(owner, "completion_after")
            if bank in by_bank:
                audit.before(
                    by_bank[bank], publication, "MIX descriptor reused before the previous accountant retired it"
                )
            if owner["task_ref"] in by_task:
                audit.before(by_task[owner["task_ref"]], publication, "MIX task still owns a block on this cluster")
            by_bank[bank] = completion
            by_task[owner["task_ref"]] = completion
            for lane in lanes:
                worker = lane["worker_index"]
                if worker in lane_retired:
                    audit.before(
                        lane_retired[worker], point(lane, "arrive_before"), "MIX lane still owes an earlier group"
                    )
                lane_retired[worker] = (
                    completion if worker == owner["completion_worker"] else point(lane, "release_after")
                )
