# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Locate measured scheduler and kernel checkpoints for the next task in a gap."""

from collections import defaultdict


def latest_predecessor_returns(tasks, fanin_map):
    """Find the latest measured kernel return among each task's direct graph predecessors."""
    if not fanin_map:
        return {}
    returns = {}
    for row in tasks:
        if row.get("executed") is not True:
            continue
        key = row["run_epoch"], row["task_id"]
        if key not in returns or returns[key]["end_time_us"] < row["end_time_us"]:
            returns[key] = row

    evidence = {}
    for epoch, task in returns:
        observed = [returns[epoch, pred] for pred in fanin_map.get(task, ()) if (epoch, pred) in returns]
        if observed:
            latest = max(observed, key=lambda row: row["end_time_us"])
            evidence[epoch, task] = {
                "latest_observed_predecessor_kernel": {
                    "task_id": latest["task_id"],
                    "worker": latest["core_id"],
                    "block": latest["block_idx"],
                    "return_us": latest["end_time_us"],
                }
            }
    return evidence


def build_mix_publications(profiles, tasks, streams):
    """Join only when the cluster ran one generation and recorded one publication for the task."""
    proxies = {
        stream["worker_id"]: stream["cluster_index"]
        for stream in streams or ()
        if stream.get("producer") == "aicore"
        and stream.get("cluster_lane") == 1
        and stream.get("cluster_index") is not None
    }
    publications = defaultdict(list)
    for profile in profiles:
        worker = profile["worker_id"]
        if worker not in proxies or not profile.get("complete"):
            continue
        for row in profile["rows"]:
            if row["kind"] == "scalar_mix_notify_members" and row.get("task_id") is not None:
                key = (profile["run_epoch"], proxies[worker], row["task_id"])
                publications[key].append((worker, row))

    members = defaultdict(list)
    for task in tasks:
        if task.get("group_kind") == 2:
            members[task["run_epoch"], task["cluster_index"], task["task_id"]].append(task)

    evidence = {}
    for key, rows in members.items():
        # The publish scope carries the task identity but no block or generation.
        groups = {(row["block_idx"], row["group_seq"]) for row in rows}
        if len(groups) != 1 or len(publications.get(key, ())) != 1:
            continue
        worker, publication = publications[key][0]
        for row in rows:
            if not row["executed"] or row["start_time_us"] < publication["start_time_us"]:
                continue
            identity = (row["run_epoch"], row["core_id"], row["reg_task_id"])
            evidence[identity] = {
                "mix_publication": {
                    "cluster": row["cluster_index"],
                    "block": row["block_idx"],
                    "generation": row["group_seq"],
                    "proxy_worker": worker,
                    "scope_start_us": publication["start_time_us"],
                    "scope_end_us": publication["end_time_us"],
                }
            }
    return evidence
