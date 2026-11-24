# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Account A5 HBG Scalar intervals in integer cycles without overlapping totals."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Map each device record kind to its scheduler stage and device function.
SCALAR_OPERATIONS = {
    "scalar_task_issue_head": ("DISCOVER", "scheduler_find_issuable_task"),
    "scalar_task_issue_fanin": ("DISCOVER", "scheduler_check_predecessor_supply"),
    "scalar_task_supply_closed": ("DISCOVER", "scheduler_task_supply_closed"),
    "scalar_prepared_receive": ("ENTRY_PREP", "scheduler_receive_payload"),
    "scalar_prepared_validate": ("ENTRY_PREP", "scheduler_prepared_function_address"),
    "scalar_dispatch_metadata": ("DISPATCH", "scheduler_load_dispatch_metadata"),
    "scalar_mix_claim_for_execution": ("CLAIM_EXECUTE", "scheduler_claim_for_execute"),
    "scalar_mix_notify_members": ("DISPATCH", "scheduler_notify_mix_members"),
    "scalar_prepared_claim": ("CLAIM_EXECUTE", "scheduler_claim_for_execution"),
    "scalar_execution_token_fanin": ("FANIN_WAIT", "scheduler_poll_fanin"),
    "scalar_fanin_observe": ("FANIN_WAIT", "scheduler_observe_fanin_prefix"),
    "scalar_cluster_mix_dependencies": ("FANIN_WAIT", "scheduler_poll_mix_fanin"),
    "scalar_subtask_contribute": ("FANOUT_PUBLISH", "scheduler_publish_fanout"),
    "scalar_mix_group_contribute": ("FANOUT_PUBLISH", "scheduler_publish_fanout"),
    "scalar_returns_contribute": ("FANOUT_PUBLISH", "scheduler_contribute_returns"),
    "scalar_counter_condition_publish": ("FANOUT_PUBLISH", "scheduler_counter_publish_condition"),
    "scalar_async_finish": ("FANOUT_PUBLISH", "scheduler_async_complete_task"),
    "scalar_can_enter_execution": ("ENTRY_PERMIT", "scheduler_validate_selected_entry"),
    "scalar_self_order_ready": ("ENTRY_PERMIT", "scheduler_self_order_ready"),
    "scalar_mix_lane_order_ready": ("ENTRY_PERMIT", "scheduler_mix_lane_order_ready"),
    "scalar_cluster_owes_before": ("ENTRY_PERMIT", "scheduler_cluster_owes_before"),
    "scalar_cluster_mix_enter": ("ENTRY_PERMIT", "scheduler_validate_mix_entry"),
    "scalar_prepare": ("ENTRY_PREP", "prepare_kernel_entry"),
    "scalar_cache_observe": ("ENTRY_PREP", "scheduler_observe_data_cache"),
    "scalar_async_begin_prebound": ("ENTRY_PREP", "scheduler_async_begin_prebound_execution"),
    "scalar_executor_scan": ("SELECT", "inspect_ready_slot"),
    "scalar_execution_token_select": ("SELECT", "mark_owned_execution_ready"),
    "scalar_select_owned_execution": ("SELECT", "scheduler_select_owned_execution"),
    "scalar_advance_mix_floor": ("MIX_SYNC", "scheduler_advance_mix_floor"),
    "scalar_cluster_mix_observe": ("MIX_SYNC", "scheduler_cluster_mix_observe"),
    "scalar_cluster_mix_all": ("MIX_SYNC", "scheduler_cluster_mix_all"),
    "scalar_prepared_return": ("RETIRE", "scheduler_return_prepared_execution"),
    "scalar_retire_execution": ("RETIRE", "scheduler_retire_execution"),
    "scalar_cluster_mix_release": ("RETIRE", "scheduler_release_mix_lane"),
    "scalar_cluster_mix_recycle": ("RETIRE", "scheduler_cluster_mix_recycle"),
    "scalar_seal": ("RETIRE", "seal_kernel_return"),
    "scalar_async_seal_execution": ("RETIRE", "scheduler_async_seal_execution"),
    "scalar_completions": ("COMPLETE", "scheduler_complete_lane_execution"),
    "scalar_async_close_computation": ("COMPLETE", "scheduler_async_close_computation"),
    "scalar_cluster_mix_account": ("COMPLETE", "scheduler_account_mix_returns"),
    "scalar_flush_completions": ("COMPLETE", "scheduler_flush_completions"),
    "scalar_graph_control": ("CONTROL", "scheduler_progress_graph_controls"),
    "scalar_counter_accounting": ("CONTROL", "scheduler_snapshot_counter_completions"),
    "scalar_loop_control": ("CONTROL", "check_exit_or_error"),
    "scalar_async_service": ("CONTROL", "scheduler_service_async_slice"),
    "scalar_async_check": ("CONTROL", "scheduler_async_check_step"),
    "scalar_execution_token_progress": ("DISPATCH", "scheduler_progress_execution_tokens"),
    "scalar_execution_token_refill": ("DISPATCH", "scheduler_refill_execution_tokens"),
    "scalar_execution_token_receive": ("DISPATCH", "scheduler_bind_ticketed_execution"),
    "scalar_fill_own_slots": ("DISPATCH", "scheduler_find_local_execution"),
    "scalar_cluster_mix_service": ("DISPATCH", "scheduler_cluster_mix_service"),
    "scalar_cluster_mix_discover_next": ("DISPATCH", "scheduler_cluster_mix_discover_next"),
    "scalar_cluster_mix_prepare_member": ("DISPATCH", "scheduler_prepare_mix_lane"),
    "scalar_group_acquire": ("DISPATCH", "scheduler_acquire_group_lane"),
    "scalar_execution_continuation": ("DISPATCH", "scheduler_continue_same_task"),
    "scalar_service_cooperative_slots": ("DISPATCH", "scheduler_progress_mix_groups"),
    "scalar_cluster_mix_service_active": ("DISPATCH", "scheduler_cluster_mix_service_active"),
    "scalar_dispatch_publish": ("DISPATCH", "scheduler_publish_dispatch_slot"),
    "scalar_ordinary_block_claim": ("CLAIM", "scheduler_claim_ticket"),
    "scalar_task_completion_observe": ("OBSERVE", "scheduler_observe_task_completion"),
    "scalar_cached_completion_observe": ("OBSERVE", "scheduler_observe_cached_completion(source)"),
    "scalar_cached_completion_lookup": ("OBSERVE", "scheduler_observe_cached_completion(base)"),
    "scalar_graph_fanin_view": ("OBSERVE", "scheduler_node_fanin_view"),
    "scalar_prepared_index": ("OBSERVE", "scheduler_prepared_execution_index"),
    "scalar_execution_identity_capture": ("TRACE", "scheduler_capture_execution_identity"),
    "scalar_graph_payload_arguments": ("TRACE", "scheduler_parse_task_payload_arguments"),
    "scalar_publish_idle": ("TRACE", "publish_deferred_idle_activity"),
    "scalar_trace_before_execution": ("TRACE", "stage_task_trace_before_execution"),
    "scalar_trace_before_completion": ("TRACE", "stage_task_trace_before_completion"),
    "scalar_trace_append": ("TRACE", "SchedulerScalarTrace::append"),
    "scalar_end_iteration": ("TRACE", "SchedulerScalarTrace::end_iteration"),
    "scalar_trace_finish": ("TRACE", "SchedulerScalarTrace::finish"),
    "scalar_trace_finish_clean": ("TRACE", "SchedulerScalarTrace::finish: DCCI issue"),
    "scalar_trace_finish_barrier": ("TRACE", "SchedulerScalarTrace::finish: DSB"),
    "scalar_bootstrap": ("START", "bootstrap_ready_graph"),
    "scalar_bootstrap_init_wait": ("START", "bootstrap_ready_graph: cluster initialization wait"),
    "scalar_bootstrap_peer_wait": ("START", "bootstrap_ready_graph: cluster first-fill wait"),
    "scalar_bootstrap_topology": ("START", "scheduler_validate_resident_topology"),
    "scalar_launch_wait": ("START", "run_resident_scheduler: launch wait"),
    "scalar_exit_drain": ("EXIT", "run_resident_scheduler: exit drain"),
    "scalar_cluster_mix_drain": ("EXIT", "scheduler_cluster_mix_drain"),
    "scalar_atomic_cas": ("MEM", "atomicCAS: return ready"),
    "scalar_atomic_store_issue": ("MEM", "atomicExch: issue (unused result)"),
    "scalar_atomic_fetch_add": ("MEM", "atomicAdd: return ready"),
    "scalar_atomic_fetch_add_issue": ("MEM", "atomicAdd: issue (unused result)"),
    "scalar_dcci_out": ("MEM", "DCCI: SINGLE CACHELINE_OUT issue"),
    "scalar_dcci_single": ("MEM", "DCCI: SINGLE CACHELINE_ALL issue"),
    "scalar_dcci_entire": ("MEM", "DCCI: ENTIRE_DATA_CACHE issue"),
    "scalar_dsb_all": ("MEM", "DSB: ALL completion wait"),
    "scalar_dsb_ddr": ("MEM", "DSB: DDR completion wait"),
    "scalar_bypass_load": ("MEM", "ld_dev: return ready"),
    "scalar_gm_load": ("MEM", "ld_dev(GM): return ready"),
    "scalar_cnt_notify_load": ("MEM", "ld_dev(CntNotify): return ready"),
    "scalar_bypass_store": ("MEM", "st_dev: issue"),
}

SCALAR_FUNCTIONS = {kind: operation for kind, (_, operation) in SCALAR_OPERATIONS.items()}

STAGE_COLORS = {
    "DISCOVER": "thread_state_runnable",
    "CLAIM_EXECUTE": "rail_response",
    "FANIN_WAIT": "thread_state_iowait",
    "EXECUTE": "good",
    "FANOUT_PUBLISH": "rail_idle",
    "ENTRY_PERMIT": "thread_state_iowait",
    "ENTRY_PREP": "thread_state_runnable",
    "SELECT": "thread_state_runnable",
    "MIX_SYNC": "thread_state_uninterruptible",
    "RETIRE": "rail_idle",
    "COMPLETE": "cq_build_passed",
    "CONTROL": "thread_state_unknown",
    "DISPATCH": "generic_work",
    "CLAIM": "rail_response",
    "OBSERVE": "generic_work",
    "TRACE": "grey",
    "START": "thread_state_runnable",
    "EXIT": "grey",
    "MEM": "generic_work",
    "SKIP": "grey",
}


def _nest_intervals(nodes, filtered):
    """Normalize only declared filtered ancestry; retain strict bounds and ordering."""
    # Zero-duration children have no measured extent at SYS_CNT resolution.
    # Keep them in the diagnostic rows but do not invent an ordering for ties.
    positive = sorted(
        (n for n in nodes if n["end_cycles"] > n["start_cycles"]),
        key=lambda n: (n["start_cycles"], -n["end_cycles"], n["scalar_depth"]),
    )
    stack = []
    for node in positive:
        depth = node["scalar_depth"]
        if not isinstance(depth, int) or depth < 0:
            raise ValueError("Invalid Scalar nesting depth")
        if filtered:
            while stack and node["start_cycles"] >= stack[-1]["end_cycles"]:
                stack.pop()
            if (not stack and node["kind"] != "scalar_window") or (stack and depth <= stack[-1]["raw_scalar_depth"]):
                raise ValueError(f"Invalid filtered Scalar ancestry for {node['kind']}")
            node["raw_scalar_depth"] = depth
            node["scalar_depth"] = len(stack)
        else:
            stack = stack[:depth]
            if depth != len(stack) or (depth == 0 and node["kind"] != "scalar_window"):
                raise ValueError(f"Missing parent for {node['kind']}")
        node["children"] = []
        if stack:
            parent = stack[-1]
            if not (parent["start_cycles"] <= node["start_cycles"] <= node["end_cycles"] <= parent["end_cycles"]):
                raise ValueError(f"Scalar child escapes parent: {node['kind']} in {parent['kind']}")
            if parent["children"] and parent["children"][-1]["end_cycles"] > node["start_cycles"]:
                raise ValueError(f"Overlapping Scalar siblings: {node['kind']}")
            parent["children"].append(node)
        stack.append(node)
    return positive


def partition(records, kernels, *, nested_kernels=False):
    """Return a nested view and an exact, non-overlapping partition of its window."""
    roots = [r for r in records if r["kind"] == "scalar_window"]
    if len(roots) != 1:
        raise ValueError("Expected one supported Scalar window per worker/run")
    root = roots[0]
    minimum = root.get("function_min_cycles", 0)
    if "function_min_cycles" not in root or not isinstance(minimum, int) or minimum < 0:
        raise ValueError("Missing or invalid Scalar function threshold")
    filtered = minimum > 0
    nodes = [dict(record) for record in records]
    for kernel in kernels:
        depth = 1
        if nested_kernels:
            depth += max(
                (
                    r["scalar_depth"]
                    for r in records
                    if r["start_cycles"] <= kernel["start_cycles"] and kernel["end_cycles"] <= r["end_cycles"]
                ),
                default=0,
            )
        nodes.append({"kind": "scalar_kernel", "scalar_depth": depth, "scalar_calls": 1, **kernel})
    for node in nodes:
        if node["end_cycles"] < node["start_cycles"]:
            raise ValueError("Negative Scalar interval")
    positive = _nest_intervals(nodes, filtered)
    segments = []
    totals = defaultdict(int)
    for node in positive:
        cursor = node["start_cycles"]
        kind = "scalar_unattributed" if node["kind"] == "scalar_window" else node["kind"]
        if not nested_kernels:
            # Propagate a stage only to direct callees at the recorded depth. A filtered frame may have its own stage.
            parent_depth = node.get("raw_scalar_depth", node["scalar_depth"])
            parent_stage = node.get("hbg_stage") or _business_stage(node["kind"])
            for child in node["children"]:
                if child.get("raw_scalar_depth", child["scalar_depth"]) == parent_depth + 1:
                    child["hbg_stage"] = _business_stage(child["kind"], node["kind"], parent_stage)
        for child in [*node["children"], {"start_cycles": node["end_cycles"], "end_cycles": node["end_cycles"]}]:
            if child["start_cycles"] > cursor:
                segments.append(
                    {
                        "kind": kind,
                        "start_cycles": cursor,
                        "end_cycles": child["start_cycles"],
                        "task_id": node.get("task_id"),
                        "run_epoch": root["run_epoch"],
                        **({"hbg_stage": node["hbg_stage"]} if "hbg_stage" in node else {}),
                    }
                )
                if not nested_kernels and kind != "scalar_unattributed":
                    segments[-1]["hbg_stage"] = node.get("hbg_stage") or _business_stage(kind)
                totals[kind] += child["start_cycles"] - cursor
            cursor = child["end_cycles"]
    window = root["end_cycles"] - root["start_cycles"]
    if sum(totals.values()) != window:
        raise ValueError("Scalar exclusive partition does not conserve its window")
    return {
        "complete": True,
        "iterations_detailed": root.get("iterations_detailed", False),
        "function_min_cycles": minimum,
        "short_function_intervals_omitted": root.get("short_function_intervals_omitted", 0),
        "window_cycles": window,
        "kernel_cycles": totals["scalar_kernel"],
        "nonkernel_cycles": window - totals["scalar_kernel"],
        "exclusive_cycles": dict(totals),
        "batch_write_cycles": root.get("batch_write_cycles", 0),
        "no_progress_loops": root.get("no_progress_loops", 0),
        "rows": [{k: v for k, v in n.items() if k != "children"} for n in nodes],
        "exclusive": sorted(segments, key=lambda r: r["start_cycles"]),
    }


def analyze_streams(streams, metadata, task_rows, execution_identities=None, *, nested_kernels=False):
    """Join same-worker, same-run L1 kernel intervals to the detailed L3 records."""
    identities = {
        (r["core_id"], r["reg_task_id"], r["run_epoch"]): r["executed"]
        for r in (execution_identities or {}).get("records", [])
    }
    kernels = defaultdict(list)
    for row in task_rows:
        worker, task, registration, begin, end, _, epoch = row[:7]
        kernels[(worker, epoch)].append(
            {
                "task_id": task,
                **({} if nested_kernels else {"reg_task_id": registration}),
                "start_cycles": begin,
                "end_cycles": end,
                "run_epoch": epoch,
                "kind": "scalar_task_skip"
                if identities.get((worker, registration, epoch)) is False
                else "scalar_kernel",
            }
        )
    profiles = []
    if len(streams) != len(metadata):
        raise ValueError("Scalar stream metadata count differs from record streams")
    for records, stream in zip(streams, metadata):
        detailed = [r for r in records if r["kind"].startswith("scalar_")]
        if not detailed:
            continue
        if stream.get("producer") != "aicore":
            raise ValueError("Scalar records require the AICore producer")
        capture = stream["capture"]
        if (
            capture.get("truncated") is not False
            or capture.get("dropped") != 0
            or capture.get("committed") != len(records)
        ):
            profiles.append(
                {
                    "worker_id": stream["worker_id"],
                    "scheduler_id": stream["scheduler_id"],
                    "complete": False,
                    "reason": "incomplete capture",
                    "rows": detailed,
                    "exclusive": [],
                }
            )
            continue
        for epoch in sorted({r["run_epoch"] for r in detailed}):
            result = partition(
                [r for r in detailed if r["run_epoch"] == epoch],
                kernels[(stream["worker_id"], epoch)],
                nested_kernels=nested_kernels,
            )
            result.update(worker_id=stream["worker_id"], scheduler_id=stream["scheduler_id"], run_epoch=epoch)
            profiles.append(result)
    return profiles


def _business_stage(kind, caller=None, caller_stage=None):
    if kind == "scalar_window":
        return None
    if kind == "scalar_kernel":
        return "EXECUTE"
    if kind == "scalar_task_skip":
        return "SKIP"
    stage = SCALAR_OPERATIONS[kind][0]
    if kind == "scalar_ordinary_block_claim":
        if caller in (
            "scalar_mix_claim_for_execution",
            "scalar_execution_token_refill",
            "scalar_execution_continuation",
        ):
            return "CLAIM_EXECUTE"
    if stage in ("MEM", "OBSERVE") and caller_stage not in (None, "DISPATCH", "MEM", "OBSERVE", "CLAIM"):
        return caller_stage
    return stage


def apply_l3_detail_policy(profile):
    """Apply HBG L3 collection semantics to the view and preserve the exclusive-time partition."""

    def visible(row):
        kind = row["kind"]
        if kind in ("scalar_gm_load", "scalar_cnt_notify_load"):
            # The producer samples these only inside an execution claim at L3.
            row["hbg_stage"] = "CLAIM_EXECUTE"
            return True
        if kind == "scalar_bypass_load":
            raise ValueError("HBG L3 requires an explicit GM or CntNotify load source")
        stage = row.get("hbg_stage") or SCALAR_OPERATIONS.get(kind, (None,))[0]
        return stage not in ("TRACE", "CONTROL") and kind not in (
            "scalar_dcci_out",
            "scalar_dcci_single",
            "scalar_dcci_entire",
        )

    profile["rows"] = [row for row in profile["rows"] if visible(row)]
    for row in profile["exclusive"]:
        if not visible(row):
            duration = row["end_cycles"] - row["start_cycles"]
            totals = profile["exclusive_cycles"]
            kind = row["kind"]
            totals[kind] -= duration
            if not totals[kind]:
                del totals[kind]
            totals["scalar_unattributed"] = totals.get("scalar_unattributed", 0) + duration
            row["kind"] = "scalar_unattributed"
            row["task_id"] = None
            row.pop("hbg_stage", None)


def trace_events(
    profiles, *, pid, tid_base, display, streams=None, function_names=None, kernel_labels=None, worker_lane=None
):
    """Render one nested Scalar lane per core using the capture's core identity."""
    lanes = {
        stream.get("scheduler_id", index): (index, stream)
        for index, stream in enumerate(streams or [])
        if stream.get("producer") == "aicore"
    }
    events = []
    for profile in profiles:
        worker = profile["worker_id"]
        index, stream = lanes.get(profile["scheduler_id"], (profile["scheduler_id"], {}))
        tid = tid_base + index * 10
        core_type = stream.get("core_type")
        name = f"{core_type.upper()}_{worker}" if core_type else f"Worker_{worker}"
        sort_index = index
        if (
            worker_lane is not None
            and stream.get("cluster_index") is not None
            and stream.get("cluster_lane") is not None
        ):
            name, sort_index = worker_lane(stream["cluster_index"], stream["cluster_lane"])
        if stream.get("physical_core_id") is not None:
            name += f" (physical {stream['physical_core_id']})"
        if not profile["complete"]:
            name += " [incomplete]"
        events.append(
            {"ph": "M", "cat": "__metadata", "name": "thread_name", "pid": pid, "tid": tid, "args": {"name": name}}
        )
        events.append(
            {
                "ph": "M",
                "cat": "__metadata",
                "name": "thread_sort_index",
                "pid": pid,
                "tid": tid,
                "args": {"sort_index": sort_index},
            }
        )
        rows = sorted(profile["rows"], key=lambda r: (r["start_time_us"], -r["end_time_us"], r.get("scalar_depth", 0)))
        for row in rows:
            kind: str = row["kind"]
            if kind == "scalar_window":
                continue
            names = SCALAR_FUNCTIONS if function_names is None else function_names
            operation = names.get(kind, kind.removeprefix("scalar_"))
            suffix = f"({display(row['task_id'])})" if row.get("task_id") is not None else ""
            args = {k: v for k, v in row.items() if k not in ("start_time_us", "end_time_us")}
            name = operation + suffix
            stage = args.get("hbg_stage")
            if function_names is None:
                stage = stage or _business_stage(kind)
                args["hbg_stage"] = stage
                name = operation.removeprefix("scheduler_") + suffix
                if kernel_labels and kind in ("scalar_kernel", "scalar_task_skip"):
                    key = (row["run_epoch"], worker, row.get("reg_task_id"))
                    name = kernel_labels.get(key, name)
                name = f"{stage} · {name}"
            args["capture_complete"] = profile["complete"]
            # Subtract integer counters before conversion, avoiding cancellation
            # and floating-point tails in the exported per-operation durations.
            duration = (
                (row["end_cycles"] - row["start_cycles"]) * 1_000_000 / profile["clock_freq_hz"]
                if profile.get("clock_freq_hz")
                else row["end_time_us"] - row["start_time_us"]
            )
            events.append(
                {
                    "ph": "X",
                    "cat": "aicore_scalar",
                    "name": name,
                    "pid": pid,
                    "tid": tid,
                    "ts": row["start_time_us"],
                    "dur": duration,
                    "args": args,
                    "cname": STAGE_COLORS[stage]
                    if stage is not None
                    else "grey"
                    if kind == "scalar_unattributed"
                    else "good"
                    if kind == "scalar_kernel"
                    else "generic_work",
                }
            )
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.capture.read_text())
    streams, metadata = [], []
    for stream in raw.get("scheduler_records", {}).get("streams", []):
        rows = [dict(r) for r in stream["records"]]
        for metric in stream.get("metrics", []):
            rows[metric["record_index"]].update({k: v for k, v in metric.items() if k != "record_index"})
        streams.append(rows)
        metadata.append(stream)
    profiles = analyze_streams(streams, metadata, raw["aicore_tasks"], raw.get("aicore_execution_identities"))
    report = {
        "complete": bool(profiles) and all(p["complete"] for p in profiles),
        "clock_freq_hz": raw["metadata"]["clock_freq_hz"],
        "workers": [{k: v for k, v in p.items() if k not in ("rows", "exclusive")} for p in profiles],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["complete"]:
        raise SystemExit("Scalar capture is absent or incomplete")


if __name__ == "__main__":
    main()
