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

SCALAR_FUNCTIONS = {
    "scalar_completions": "scheduler_complete_own_slot",
    "scalar_graph_control": "scheduler_service_controls",
    "scalar_fill_slots": "scheduler_fill_own_slots",
    "scalar_ordered_fill": "scheduler_fill_ordered_slots",
    "scalar_admission": "scheduler_ordered_can_enter_slot",
    "scalar_executor_scan": "run_ready_dispatch_loop: peek ready head",
    "scalar_prepare": "run_ready_dispatch_loop: prepare execution",
    "scalar_seal": "run_ready_dispatch_loop: seal execution",
    "scalar_publish_completion": "run_ready_dispatch_loop: publish completion",
    "scalar_async_service": "scheduler_service_async_slice",
    "scalar_backoff": "local_backoff",
    "scalar_ready_claim": "scheduler_claim_ready_for_slot",
    "scalar_payload": "scheduler_prepare_dispatch_slot",
    "scalar_early_prepare": "scheduler_prepare_early_candidate",
    "scalar_cluster_prepare": "scheduler_prepare_cluster_lane",
    "scalar_execution_token_progress": "scheduler_progress_execution_tokens",
    "scalar_execution_token_refill": "scheduler_refill_execution_tokens",
    "scalar_execution_token_receive": "scheduler_execution_token_receive",
    "scalar_execution_token_dispatch": "scheduler_progress_execution_tokens: transfer token to dispatch slot",
    "scalar_fanin_observe": "scheduler_observe_fanin_prefix",
    "scalar_cooperative_discover": "scheduler_discover_cooperative_task",
    "scalar_fill_own_slots": "scheduler_fill_own_slots",
    "scalar_fill_cooperative_slots": "scheduler_fill_cooperative_slots",
    "scalar_cluster_mix_service": "scheduler_cluster_mix_service",
    "scalar_cluster_mix_discover_task": "scheduler_cluster_mix_discover_task",
    "scalar_cluster_mix_discover_next": "scheduler_cluster_mix_discover_next",
    "scalar_cluster_mix_pending": "scheduler_cluster_mix_progress_pending",
    "scalar_cluster_mix_arm": "scheduler_cluster_mix_try_arm",
    "scalar_cluster_mix_observe": "scheduler_cluster_mix_observe",
    "scalar_cluster_mix_retire": "scheduler_cluster_mix_retire",
    "scalar_cluster_mix_enter": "scheduler_cluster_mix_can_enter",
    "scalar_cluster_mix_release": "scheduler_cluster_mix_release",
    "scalar_cluster_mix_drain": "scheduler_cluster_mix_drain",
    "scalar_cluster_mix_publish_preparation": "scheduler_cluster_mix_publish_preparation",
    "scalar_cluster_mix_conflicts": "scheduler_cluster_mix_conflicts",
    "scalar_cluster_mix_all": "scheduler_cluster_mix_all",
    "scalar_prepared_receive": "scheduler_receive_execution_token",
    "scalar_prepared_acquire": "scheduler_acquire_execution_token",
    "scalar_prepared_park_begin": "scheduler_begin_park_prepared_execution",
    "scalar_prepared_park_publish": "scheduler_publish_parked_execution",
    "scalar_prepared_resume": "scheduler_resume_parked_execution",
    "scalar_prepared_return": "scheduler_return_prepared_execution",
    "scalar_prepared_observe": "scheduler_observe_prepared_call",
    "scalar_prepared_index": "scheduler_prepared_execution_index",
    "scalar_prepared_validate": "scheduler_prepared_call_valid",
    "scalar_group_reserve": "scheduler_reserve_group_execution",
    "scalar_group_park": "scheduler_park_group_execution",
    "scalar_group_resume": "scheduler_resume_group_execution",
    "scalar_group_arm": "scheduler_arm_group_execution",
    "scalar_group_acquire": "scheduler_acquire_group_lane",
    "scalar_cluster_mix_needs_service": "scheduler_cluster_mix_needs_service",
    "scalar_execution_continuation": "scheduler_try_execution_continuation",
    "scalar_execution_token_fanin": "scheduler_execution_token_observe_fanin",
    "scalar_cluster_mix_ready_replacement": "scheduler_cluster_mix_ready_replacement",
    "scalar_resolve": "scheduler_resolve_completion_step",
    "scalar_mix": "scheduler_service_gm_mix",
    "scalar_async_check": "scheduler_async_check_step",
    "scalar_async_finish": "scheduler_async_finish_step",
    "scalar_bootstrap": "bootstrap_ready_graph",
    "scalar_launch_wait": "run_resident_scheduler: launch wait",
    "scalar_exit_drain": "run_resident_scheduler: exit drain",
    "scalar_atomic_claim": "scheduler_gm_compare_exchange (return ready)",
    "scalar_cache_observe": "scheduler_observe_data_cache",
    "scalar_cache_publish": "scheduler_prepare_dispatch_slot: payload writeback",
    "scalar_flush_completions": "scheduler_flush_completions",
    "scalar_loop_control": "run_ready_dispatch_loop: exit/error checks",
    "scalar_ordered_poll": "scheduler_ordered_poll_may_progress",
    "scalar_admission_claim": "scheduler_stage_admission_candidate",
    "scalar_admission_publish": "scheduler_ordered_publish_pending",
    "scalar_admission_commit": "scheduler_try_commit_admission",
    "scalar_admission_refresh": "scheduler_refresh_local_obligation",
    "scalar_ordinary_park": "scheduler_ordered_park_ordinary",
    "scalar_block_claim": "scheduler_claim_ordered_block",
    "scalar_block_cursor_read": "scheduler_read_claim_cursor",
    "scalar_admission_close": "scheduler_close_ordered_obligation",
    "scalar_ordinary_permit": "scheduler_can_enter_ordinary",
    "scalar_dispatch_metadata": "scheduler_load_dispatch_metadata",
    "scalar_dispatch_fill": "scheduler_fill_dispatch_slot",
    "scalar_dispatch_publish": "scheduler_publish_dispatch_slot",
    "scalar_inline_complete": "scheduler_complete_inline_task",
    "scalar_cursor_validate": "scheduler_read_claim_cursor: validate context",
    "scalar_claim_seal_probe": "scheduler_read_claim_cursor: claim_seal hint",
    "scalar_claim_seal_confirm": "scheduler_read_claim_cursor: claim_seal atomic confirm",
    "scalar_next_block_probe": "scheduler_read_claim_cursor: next_block hint",
    "scalar_next_block_confirm": "scheduler_read_claim_cursor: next_block atomic confirm",
    "scalar_poll_local_state": "scheduler_ordered_poll_may_progress: local state",
    "scalar_poll_ready_queue": "scheduler_ordered_poll_may_progress: ready queue",
    "scalar_poll_ready_pool": "scheduler_ready_pool_may_have_work",
    "scalar_poll_admission_tail": "scheduler_ordered_poll_may_progress: admission commit_tail",
    "scalar_poll_mix_state": "scheduler_ordered_poll_may_progress: MIX state",
    "scalar_poll_mix_close_seal": "scheduler_ordered_poll_may_progress: MIX close seal",
    "scalar_poll_mix_assignment": "scheduler_ordered_poll_may_progress: MIX assigned_seq",
    "scalar_poll_mix_arrived": "scheduler_ordered_poll_may_progress: MIX arrived_seq",
    "scalar_poll_mix_released": "scheduler_ordered_poll_may_progress: MIX released_seq",
    "scalar_trace_finish": "SchedulerScalarTrace::finish",
    "scalar_trace_finish_clean": "SchedulerScalarTrace::finish: DCCI issue",
    "scalar_trace_finish_barrier": "SchedulerScalarTrace::finish: DSB",
    "scalar_atomic_load": "atomicAdd(0): return ready",
    "scalar_atomic_cas": "atomicCAS: return ready",
    "scalar_atomic_exchange": "atomicExch: return ready",
    "scalar_atomic_store_issue": "atomicExch: issue (unused result)",
    "scalar_atomic_fetch_add": "atomicAdd: return ready",
    "scalar_atomic_fetch_add_issue": "atomicAdd: issue (unused result)",
    "scalar_dcci_out": "DCCI: SINGLE CACHELINE_OUT issue",
    "scalar_dcci_single": "DCCI: SINGLE CACHELINE_ALL issue",
    "scalar_dcci_entire": "DCCI: ENTIRE_DATA_CACHE issue",
    "scalar_dsb_all": "DSB: ALL completion wait",
    "scalar_dsb_ddr": "DSB: DDR completion wait",
    "scalar_bypass_load": "ld_dev: return ready",
    "scalar_bypass_store": "st_dev: issue",
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
    if len(roots) != 1 or roots[0].get("scalar_schema") != 2:
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
        for child in [*node["children"], {"start_cycles": node["end_cycles"], "end_cycles": node["end_cycles"]}]:
            if child["start_cycles"] > cursor:
                segments.append(
                    {
                        "kind": kind,
                        "start_cycles": cursor,
                        "end_cycles": child["start_cycles"],
                        "task_id": node.get("task_id"),
                        "run_epoch": root["run_epoch"],
                    }
                )
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
        worker, task, registration, begin, end, _, epoch = row
        kernels[(worker, epoch)].append(
            {
                "task_id": task,
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


def trace_events(profiles, *, pid, tid_base, display, streams=None, function_names=None):
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
                "args": {"sort_index": index},
            }
        )
        rows = sorted(profile["rows"], key=lambda r: (r["start_time_us"], -r["end_time_us"], r.get("scalar_depth", 0)))
        for row in rows:
            kind = row["kind"]
            if kind == "scalar_window":
                continue
            name = (SCALAR_FUNCTIONS if function_names is None else function_names).get(
                kind, kind.removeprefix("scalar_")
            )
            if row.get("task_id") is not None:
                name += f"({display(row['task_id'])})"
            args = {k: v for k, v in row.items() if k not in ("start_time_us", "end_time_us")}
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
                    "cname": "grey"
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
