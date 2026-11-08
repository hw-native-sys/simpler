# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Recorded async endpoints and logical wait links, never reconstructed busy spans.

The standard converter calls the projection only after its strict full audit.
The pure helper can also preserve partial evidence for isolated diagnostics;
that does not make failed captures acceptable to the standard converter.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from . import native_protocol
from .native_async import async_summary_counts

_CLOCK_NOTE = "Each endpoint uses its recorded worker clock; link lengths are not elapsed or busy time."
_GROUP_PENDING_NOTE = "PENDING service slices, including acquire and budget yields; not a count of unmet counter reads."
_FINISH_PENDING_NOTE = "Current unpublished FINISH batch length, not service slices."
_ACTIVE_NOTE = "Sum of work within individual service slices; no continuous busy interval is recorded."


def project_async_protocol(protocol, to_us, *, protocol_verified=False) -> dict[str, Any] | None:
    """Map each real (worker, cycles) independently; zero cycles can be recorded.

    ``to_us(worker_index, cycles)`` is the capture's actual clock-placement
    callback. This function never subtracts endpoints, invents missing stamps,
    or turns the aggregate pending count into per-observation events.
    ``protocol_verified=True`` requires a successful audit of this unchanged input.
    """
    if not protocol_verified:
        native_protocol.read_native_protocol(protocol)
    if not native_protocol.has_async_protocol(protocol):
        return None
    raw = protocol["async"]
    epoch = protocol["run_epoch"]
    public_ids = {row["node_ref"]: row["public_task_id"] for row in protocol.get("node_map", [])}
    workers = {row["worker_index"]: row for row in protocol["workers"]}
    points, links = [], []

    def attributes(row, *evidence_fields):
        result = {key: deepcopy(value) for key, value in row.items() if key not in evidence_fields}
        result["task_id"] = public_ids.get(row["task_ref"], row["task_ref"])
        if "lease_word" in row:
            result["lease_generation"] = row["lease_word"] >> 3
        if "registered_count" in row:
            result["pending_count_meaning"] = _GROUP_PENDING_NOTE
            result["condition_types"] = [
                {0: "COUNTER", 1: "SDMA"}.get(condition["type"], f"unknown({condition['type']})")
                for condition in row["conditions"]
            ]
        elif "waiter_count" in row:
            result["pending_count_meaning"] = _FINISH_PENDING_NOTE
        if "active_cycles" in row:
            result["active_cycles_meaning"] = _ACTIVE_NOTE
        return result

    def endpoint(name, key, evidence, fields, stamp_field="at"):
        if evidence is None:
            return None
        stamp = evidence[stamp_field]
        if stamp is None or stamp[1] == 0:
            return None
        worker = evidence["worker_index"]
        if worker not in workers:
            raise ValueError(f"async endpoint refers to unknown worker {worker}")
        identity = workers[worker]
        point_id = f"async:{epoch}:{key}"
        args = dict(
            fields,
            run_epoch=evidence["run_epoch"],
            worker_index=worker,
            cycles=stamp[0],
            sequence=stamp[1],
            clock_note=_CLOCK_NOTE,
            snapshot_note=(
                "capture_* fields describe the stopped-capture snapshot and aggregate counters, "
                "not state or counter values at this endpoint."
            ),
        )
        for field in (
            "state",
            "lease_word",
            "lease_generation",
            "registered_count",
            "observations",
            "pending_count",
            "slice_count",
            "active_cycles",
            "waiter_count",
            "failure_code",
        ):
            if field in args:
                args[f"capture_{field}"] = args.pop(field)
        points.append(
            dict(
                id=point_id,
                name=name,
                time_us=to_us(worker, stamp[0]),
                worker_index=worker,
                core_id=identity["core_id"],
                core_type=identity["core_type"],
                args=args,
            )
        )
        return point_id

    def link(name, key, first, last):
        if protocol_verified and first is not None and last is not None:
            links.append(dict(id=f"async:{epoch}:link:{key}", name=name, first=first, last=last))

    terminals = {}
    for row in raw["groups"]:
        index = row["index"]
        fields = attributes(row, "sealed", "terminal")
        sealed = endpoint("registration_sealed", f"group:{index}:sealed", row["sealed"], fields)
        terminal_name = f"condition_{row['state'].lower()}" if row["state"] in ("READY", "FAILED") else "check_terminal"
        terminal = endpoint(terminal_name, f"group:{index}:terminal", row["terminal"], fields)
        terminals[index] = terminal
        link("logical_wait_endpoints", f"wait:{index}", sealed, terminal)

    releases = {}
    for row in raw["releases"]:
        ref, block = row["task_ref"], row["block_idx"]
        fields = attributes(row, "publication")
        fields["meaning"] = (
            "Publication bounds enclose one logical block contribution; neither endpoint is its linearization point."
        )
        before = endpoint(
            "execution_release.before", f"release:{ref}:{block}:before", row["publication"], fields, "before"
        )
        after = endpoint("execution_release.after", f"release:{ref}:{block}:after", row["publication"], fields, "after")
        releases[(ref, block)] = before
        link("release_publication_bounds", f"release:{ref}:{block}", before, after)

    completions = {row["index"]: row["completion"] for row in protocol["task_events"]}
    for row in raw["finishes"]:
        ref = row["task_ref"]
        fields = attributes(row, "drained")
        fields["meaning"] = (
            "logical_done reuses the existing task completion bounds; resolve_drained is a separate endpoint."
        )
        done = completions.get(ref)
        before = endpoint("logical_done.before", f"task:{ref}:done:before", done, fields, "before")
        after = endpoint("logical_done.after", f"task:{ref}:done:after", done, fields, "after")
        drained = endpoint("resolve_drained", f"finish:{row['index']}:drained", row["drained"], fields)
        link("logical_done_bounds", f"done:{ref}", before, after)
        # Use the lower/upper *bounds*, not a guessed publication linearization
        # point. A peer may finish before the release publisher records after.
        link("logical_done_lower_bound_to_drained", f"drain:{ref}", before, drained)
        for group in row["group_indices"]:
            link("terminal_to_logical_done_upper_bound", f"terminal-join:{group}", terminals.get(group), after)
        for (release_ref, block), first in releases.items():
            if release_ref == ref:
                link("release_lower_bound_to_logical_done_upper_bound", f"release-join:{ref}:{block}", first, after)

    summary: dict[str, Any] = async_summary_counts(raw)
    summary.update(
        protocol_verified=bool(protocol_verified),
        clock_note=_CLOCK_NOTE,
        group_pending_meaning=_GROUP_PENDING_NOTE,
        finish_pending_meaning=_FINISH_PENDING_NOTE,
        active_cycles_meaning=_ACTIVE_NOTE,
        groups=[attributes(row) for row in raw["groups"]],
        releases=[attributes(row) for row in raw["releases"]],
        finishes=[attributes(row) for row in raw["finishes"]],
    )
    return dict(points=points, links=links, summary=summary)


def contribute_async_events(view, *, pid, tid_base, display):
    """Draw actual instants on observer/owner tracks, with separate logical links."""
    if view is None:
        return []
    verified = view["summary"]["protocol_verified"]
    title = "Async Completion" if verified else "Async Completion (unverified diagnostics)"
    events = [
        dict(ph="M", pid=pid, name="process_name", cat="__metadata", args=dict(name=title)),
        dict(ph="M", pid=pid, name="process_sort_index", cat="__metadata", args=dict(sort_index=5)),
    ]
    workers = {point["worker_index"]: point for point in view["points"]}
    for worker, point in sorted(workers.items()):
        core = "AIC" if point["core_type"] == 0 else "AIV"
        events.append(
            dict(
                ph="M",
                pid=pid,
                tid=tid_base + worker,
                name="thread_name",
                cat="__metadata",
                args=dict(name=f"Worker {worker} / {core} {point['core_id']}"),
            )
        )
    by_id = {}
    for point in view["points"]:
        args = dict(point["args"], task_label=display(point["args"]["task_id"]))
        event = dict(
            ph="i",
            s="t",
            pid=pid,
            tid=tid_base + point["worker_index"],
            cat="hbg_async",
            name=point["name"],
            ts=point["time_us"],
            args=args,
        )
        events.append(event)
        by_id[point["id"]] = event
    for relation in view["links"]:
        for phase, key in (("s", "first"), ("f", "last")):
            point = by_id[relation[key]]
            event = dict(
                ph=phase,
                pid=pid,
                tid=point["tid"],
                cat="hbg_async_flow",
                name=relation["name"],
                id=relation["id"],
                ts=point["ts"],
                args=dict(meaning="Logical relation between recorded endpoints; no elapsed duration."),
            )
            if phase == "f":
                event["bp"] = "e"
            events.append(event)
    return events
