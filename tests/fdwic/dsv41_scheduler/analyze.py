#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Audit measured model replays and standalone scheduler benchmarks."""

import argparse
import csv
import importlib.util
import json
import math
import runpy
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from production.ideal_bound import validate_synthetic_dag
from production.timing import replay_launches, summarize_replay


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


def native_moe_reports(root, case):
    path = Path(__file__).resolve().parent / "production/moe.py"
    spec = importlib.util.spec_from_file_location("dsv41_native_moe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reports = {level: load(root / case["case"] / f"L{level}" / "result.json") for level in (0, 1, 3)}
    for level, report in reports.items():
        module.validate_report(report, case, level)
    return {"flavor": case["flavor"], "case": case["case"], "native_host_control": reports}


def add_perfetto(details, path, names, trace, convert):
    if not convert:
        return
    if names is None:
        raise ValueError("Conversion requires a function name mapping")
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
    details["perfetto"] = str(trace)


def synthetic_report(bundle, case, run, destination, convert):
    rank = run / "rank0"
    result = load(rank / "result.json")
    if (
        result["runtime"] != case["runtime"]
        or result["ranks"] != 1
        or not result["passed"]
        or not result["numerical_validation"]
        or not result["snapshot_inputs"]
        or not result["isolated_case"]
    ):
        raise ValueError(f"Synthetic bundle and run disagree on runtime or rank count: {run}")
    stages = case["source"]["stages"]
    members = {
        (stage["task_id"], block, lane)
        for stage in stages
        for block in range(stage["block_num"])
        for lane, kernel in enumerate(stage["kernel_ids"])
        if kernel >= 0
    }
    diamond = case["execution"] == "mpmd_diamond"
    expected_tasks, expected_members, expected_edges = (4, 4, 4) if diamond else (6, 386, 5)
    if (
        len(stages) != expected_tasks
        or len(members) != expected_members
        or any(not (bundle / "reference" / f"{index}.bin").is_file() for index in range(1, expected_tasks + 1))
    ):
        raise ValueError(f"Incomplete synthetic bundle: {bundle}")
    by_engine = Counter((task, "aic" if lane == 0 else "aiv") for task, _, lane in members)
    launches = replay_launches(rank, result)
    profiled = {i: level for i, _, level, _ in launches if level}
    l0 = summarize_replay(run, analytical_golden=True)["samples_ms"] if launches[0][2] == 0 else []
    captures = []
    for iteration, level in profiled.items():
        path = rank / f"run{iteration}" / "chip_swimlane_records.json"
        records = load(path)
        deps = load(path.with_name("deps.json"))
        validate_synthetic_dag(deps, stages, case["source"].get("edges"))
        types = records["metadata"]["core_types"]
        actual = Counter((row[1], types[row[0]]) for row in records["aicore_tasks"])
        if actual != by_engine or records["chip_swimlane_level"] != level or len(deps["edges"]) != expected_edges:
            raise ValueError(f"Incomplete synthetic capture: {path}")
        if (
            case["runtime"] == "host_build_graph"
            and execution_durations(records, {t for t, _, _ in members}).keys() != members
        ):
            raise ValueError(f"Synthetic physical executions differ from the expected blocks and lanes: {path}")
        capture = {
            "capture": str(path),
            "level": level,
            "tasks": expected_tasks,
            "executions": sum(actual.values()),
            "dependencies": expected_edges,
        }
        trace = destination / f"{case['case']}-{path.parent.name}-L{level}.json"
        add_perfetto(capture, path, bundle / "names.json", trace, convert)
        captures.append(capture)
    return {
        "flavor": case["flavor"],
        "case": case["case"],
        "rank": 0,
        "validation": case["validation"],
        "L0_aicore_wall_samples_ms": l0,
        "captures": captures,
    }


def audit(root, cases, original, names, destination, convert):
    destination.mkdir(parents=True, exist_ok=True)
    rows, details = [], []
    for case_dir in sorted(cases.iterdir()):
        case = load(case_dir / "case.json")
        if case.get("execution") in ("full_wave_mix_chain", "mpmd_diamond"):
            details.append(synthetic_report(case_dir, case, root / case["case"], destination, convert))
            continue
        if case.get("execution") == "native_moe_host_control":
            details.append(native_moe_reports(root, case))
            continue
        expected = case["source"]
        timing = runpy.run_path(str(Path(__file__).with_name("production") / "timing.py"))
        case_timing = timing["summarize_replay"](root / case["case"])
        for rank in range(case.get("device_count", 2)):
            directory = root / case["case"] / f"rank{rank}"
            result = load(directory / "result.json")
            if not result["passed"] or not result["validated_against_fused_prefix"]:
                raise ValueError(f"Case has not passed boundary validation: {directory}")
            wall_samples = case_timing["per_rank_samples_ms"][rank]
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
                trace = destination / f"{case['flavor']}-{case['case']}-rank{rank}-L{level}.json"
                add_perfetto(levels[level], path, names, trace, convert)
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
                    "L0_aicore_wall_mean_ms": case_timing["mean_ms"],
                    "L0_rank_aicore_wall_mean_ms": statistics.mean(wall_samples),
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
                    "L0_aicore_wall_samples_ms": wall_samples,
                    "levels": levels,
                }
            )
    (destination / "audit.json").write_text(
        json.dumps(
            {
                "passed": True,
                "cases": details,
                "limits": [
                    "L0 uses actual AICore kernel endpoints, excluding AICPU startup and teardown.",
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
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Production audit passed for {len(details)} case/rank reports")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("cases", type=Path)
    parser.add_argument("original", nargs="?", type=Path)
    parser.add_argument("names", nargs="?", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--convert", action="store_true")
    args = parser.parse_args()
    audit(
        args.runs.resolve(),
        args.cases.resolve(),
        args.original.resolve() if args.original else None,
        args.names.resolve() if args.names else None,
        args.output.resolve(),
        args.convert,
    )
