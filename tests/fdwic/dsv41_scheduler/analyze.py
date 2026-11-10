#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Audit production captures and compare original kernel durations, without synthetic timings."""

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from simpler_setup.tools.strace_timing import parse_spans


def load(path):
    return json.loads(path.read_text())


def stats(values):
    ordered = sorted(values)
    if not ordered:
        return None
    return {
        "count": len(values),
        "median_us": statistics.median(values),
        "p95_us": ordered[math.ceil(len(ordered) * 0.95) - 1],
        "max_us": ordered[-1],
        "over_5us": sum(value > 5 for value in values),
    }


def geometry(task):
    return (
        task["kernel_ids"],
        task["block_num"],
        [
            {key: arg[key] for key in ("idx", "type", "dtype", "shape", "start_offset", "strides")}
            for arg in task["args"]
        ],
    )


def scheduler_phases(records, run):
    frequency = records["metadata"]["clock_freq_hz"]
    phases = defaultdict(list)
    for stream in records.get("scheduler_records", {}).get("streams", []):
        capture = stream["capture"]
        if capture["truncated"] or capture["dropped"] or capture["committed"] != len(stream["records"]):
            raise ValueError(f"Truncated scheduler trace: {run}")
        for event in stream["records"]:
            phases[event["kind"]].append((event["end_cycles"] - event["start_cycles"]) * 1e6 / frequency)
    return phases


def device_wall_samples(directory, result):
    spans = []
    for path in directory.glob("host/host.*.log"):
        spans.extend(
            span
            for span in parse_spans(path.read_text().splitlines())
            if span.name == "chip.run.runner_run.device_wall" and span.is_device
        )
    spans.sort(key=lambda span: (span.pid, span.inv))
    warmup, rounds = result["warmup"], result["rounds"]
    if result.get("snapshot_inputs"):
        if len(spans) != warmup + rounds + 2:
            raise ValueError(f"Missing single-device timing spans: {directory}")
        return [span.dur / 1e6 for span in spans[warmup : warmup + rounds]]
    # The driver performs fused-reference, then prefix/case pairs. Require
    # every actual device span before selecting the measured case calls.
    if len(spans) != 1 + 2 * (warmup + rounds + 2):
        raise ValueError(f"Missing device timing spans: {directory}")
    return [spans[2 + 2 * iteration].dur / 1e6 for iteration in range(warmup, warmup + rounds)]


def execution_durations(records, selected, mapping=None):
    identities = {
        (item["core_id"], item["reg_task_id"]): item
        for item in records["aicore_execution_identities"]["records"]
        if item["executed"]
    }
    values = {}
    frequency = records["metadata"]["clock_freq_hz"]
    for row in records["aicore_tasks"]:
        task = mapping[int(row[1])] if mapping is not None else int(row[1])
        if task not in selected:
            continue
        identity = identities[(row[0], row[2])]
        key = task, identity["block_idx"], identity["subtask_slot"]
        if key in values:
            raise ValueError(f"Duplicate physical invocation: {key}")
        values[key] = (row[4] - row[3]) * 1e6 / frequency
    return values


def compare_kernel_durations(records, source_records, source_tasks, selected, mapping):
    actual = execution_durations(records, selected, mapping)
    original = execution_durations(source_records, selected)
    if actual.keys() != original.keys():
        raise ValueError("Original (task, block, subtask) invocation set changed")
    groups = defaultdict(lambda: ([], []))
    for (task, block, slot), duration in actual.items():
        fid = source_tasks[task]["kernel_ids"][slot]
        groups[fid][0].append(duration)
        groups[fid][1].append(original[(task, block, slot)])
    return {
        fid: {
            "isolated": stats(values),
            "original": stats(reference),
            "median_ratio": statistics.median(values) / statistics.median(reference),
        }
        for fid, (values, reference) in groups.items()
    }


def audit(root, cases, original, names, destination, convert):
    destination.mkdir(parents=True, exist_ok=True)
    rows, details = [], []
    for case_dir in sorted(cases.iterdir()):
        case = load(case_dir / "case.json")
        expected = case["source"]
        for rank in range(case.get("device_count", 2)):
            directory = root / case["case"] / f"rank{rank}"
            result = load(directory / "result.json")
            if not result["passed"] or not result["validated_against_fused_prefix"]:
                raise ValueError(f"Case has not passed boundary validation: {directory}")
            wall_samples = device_wall_samples(directory, result)
            source_dir = original / f"rank{rank}/run0"
            source_deps = load(source_dir / "deps.json")
            source_tasks = {int(task["task_id"]): task for task in source_deps["tasks"]}
            source_records = load(source_dir / "chip_swimlane_records.json")
            levels = {}
            for run in directory.glob("run*"):
                path = run / "chip_swimlane_records.json"
                if not path.exists():
                    continue
                records = load(path)
                level = records["chip_swimlane_level"]
                mapping = {
                    int(target): int(source)
                    for source, target in (
                        line.split() for line in (run / "source-task-map.txt").read_text().splitlines()
                    )
                }
                deps = load(run / "deps.json")
                actual_tasks = {int(task["task_id"]): task for task in deps["tasks"]}
                if len(actual_tasks) != expected["tasks"] or len(records["aicore_tasks"]) != expected["executions"]:
                    raise ValueError(f"Task/execution count changed: {run}")
                identities = records["aicore_execution_identities"]["records"]
                if sum(identity["executed"] for identity in identities) != expected["executions"]:
                    raise ValueError(f"Incomplete physical execution identities: {run}")
                for tid, task in actual_tasks.items():
                    if geometry(task) != geometry(source_tasks[mapping[tid]]):
                        raise ValueError(f"Original kernel arguments changed: {run}: {tid}")
                selected = {mapping[tid] for tid in actual_tasks}
                original_edges = {
                    (int(edge["pred"]), int(edge["succ"]))
                    for edge in source_deps["edges"]
                    if int(edge["pred"]) in selected and int(edge["succ"]) in selected
                }
                actual_edges = {
                    (mapping[int(edge["pred"])], mapping[int(edge["succ"])])
                    for edge in deps["edges"]
                    if int(edge["pred"]) in actual_tasks and int(edge["succ"]) in actual_tasks
                }
                if original_edges != actual_edges:
                    raise ValueError(
                        f"Internal dependencies changed: {run}: "
                        f"missing={len(original_edges - actual_edges)} extra={len(actual_edges - original_edges)}"
                    )
                frequency = records["metadata"]["clock_freq_hz"]
                kernel = [(row[4] - row[3]) * 1e6 / frequency for row in records["aicore_tasks"]]
                source_frequency = source_records["metadata"]["clock_freq_hz"]
                reference = [
                    (row[4] - row[3]) * 1e6 / source_frequency
                    for row in source_records["aicore_tasks"]
                    if int(row[1]) in selected
                ]
                if len(reference) != expected["executions"]:
                    raise ValueError(f"Original forward capture is incomplete for this case: {source_dir}")
                kernel_comparison = compare_kernel_durations(records, source_records, source_tasks, selected, mapping)
                phases = scheduler_phases(records, run)
                span_ms = (
                    (max(row[4] for row in records["aicore_tasks"]) - min(row[3] for row in records["aicore_tasks"]))
                    * 1000
                    / frequency
                )
                levels[level] = {
                    "kernel_window_ms": span_ms,
                    "kernel": stats(kernel),
                    "original_kernel": stats(reference),
                    "per_kernel_comparison": kernel_comparison,
                    "scheduler_phases": {key: stats(values) for key, values in sorted(phases.items())},
                    "internal_dependencies": len(actual_edges),
                    "capture": str(path),
                }
                if convert:
                    trace = destination / f"{case['flavor']}-{case['case']}-rank{rank}-L{level}.json"
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "simpler_setup.tools.swimlane_converter",
                            str(path),
                            "--func-names",
                            str(names),
                            "-o",
                            str(trace),
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                    )
                    levels[level]["perfetto"] = str(trace)
            if set(levels) != {1, 3}:
                raise ValueError(f"Missing production L1/L3 captures: {directory}")
            l1, l3 = levels[1], levels[3]
            active = [value for key, value in l3["scheduler_phases"].items() if key != "idle"]
            rows.append(
                {
                    "flavor": case["flavor"],
                    "case": case["case"],
                    "rank": rank,
                    "tasks": expected["tasks"],
                    "executions": expected["executions"],
                    "L0_device_wall_median_ms": statistics.median(wall_samples),
                    "L1_kernel_window_ms": l1["kernel_window_ms"],
                    "L1_kernel_median_us": l1["kernel"]["median_us"],
                    "original_kernel_median_us": l1["original_kernel"]["median_us"],
                    "L3_nonidle_span_max_us": max(value["max_us"] for value in active),
                    "L3_nonidle_spans_over_5us": sum(value["over_5us"] for value in active),
                }
            )
            details.append(
                {
                    "flavor": case["flavor"],
                    "case": case["case"],
                    "rank": rank,
                    "L0_device_wall_samples_ms": wall_samples,
                    "levels": levels,
                }
            )
    (destination / "audit.json").write_text(
        json.dumps(
            {
                "passed": True,
                "cases": details,
                "limits": [
                    "Kernel window is not whole Device wall time.",
                    "L3 phases are producer-defined spans; idle is reported separately, "
                    "not reclassified as dependency wait.",
                    "Original forward and isolated launches can have different cache warmth and worker placement.",
                    "This audit does not certify the 5 us target; it reports measured spans without invented gaps.",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    with (destination / "baseline.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Production audit passed: {len(rows)} case/rank combinations, {len(rows) * 2} L1/L3 captures")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("cases", type=Path)
    parser.add_argument("original", type=Path)
    parser.add_argument("names", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--convert", action="store_true")
    args = parser.parse_args()
    audit(
        args.runs.resolve(),
        args.cases.resolve(),
        args.original.resolve(),
        args.names.resolve(),
        args.output.resolve(),
        args.convert,
    )
