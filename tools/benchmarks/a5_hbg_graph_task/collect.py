#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Inventory A5 HBG scenes and retain reproducible, per-case timing evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from simpler_setup.environment import PROJECT_ROOT

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from simpler_setup.tools.strace_timing import (  # noqa: E402
    bucket_by_hid,
    count_record_heads,
    group_invocations,
    invocation_spans,
    parse_drop_summaries,
    parse_spans,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def inventory(output):
    class Inventory:
        def pytest_collection_finish(self, session):
            scenes = []
            functions = []
            seen = set()
            for item in session.items:
                cls = getattr(item, "cls", None)
                if cls is None or not hasattr(cls, "_st_level"):
                    functions.append(item.nodeid)
                    continue
                if cls in seen:
                    continue
                seen.add(cls)
                for case in cls.CASES:
                    platforms = case.get("platforms", [])
                    if platforms and "a5" not in platforms:
                        continue
                    scenes.append(
                        {
                            "path": str(item.path.relative_to(ROOT)),
                            "class": cls.__name__,
                            "case": case["name"],
                            "params": case.get("params", {}),
                            "config": case.get("config", {}),
                            "manual": case.get("manual", False),
                            "markers": [mark.name for mark in item.iter_markers()],
                        }
                    )
            write_json(output, {"scenes": scenes, "functions": functions})

    return pytest.main(
        [
            "tests/st/a5/host_build_graph",
            "examples/a5/host_build_graph",
            "--platform",
            "a5",
            "--collect-only",
            "--manual",
            "include",
            "--exclude-level",
            "4",
            "-q",
        ],
        plugins=[Inventory()],
    )


def extract_metrics(log, warmup):
    lines = log.read_text(errors="replace").splitlines()
    spans = list(parse_spans(lines))
    if count_record_heads(lines) > len(spans) or any(item["total"] for item in parse_drop_summaries(lines).values()):
        return {"error": "incomplete or dropped STRACE records", "rows": []}
    measured_spans = [
        span
        for span in spans
        if not (
            span.inv == 0
            and span.name.startswith(("chip.run.proof.", "chip.run.borrow_release", "chip.run.claim_release"))
        )
    ]
    buckets = bucket_by_hid(group_invocations(invocation_spans(measured_spans)))
    if not buckets:
        return {"error": "no STRACE invocation markers", "rows": []}
    invocations = max(buckets.values(), key=len)
    rows = []
    for invocation in invocations:
        names = invocation.by_name()
        host = names.get("chip.run")
        device = names.get("chip.run.runner_run.device_wall")
        bind = names.get("chip.run.bind")
        rows.append(
            {
                "epoch": invocation.inv,
                "host_us": host.dur / 1000 if host else None,
                "device_us": device.dur / 1000 if device else None,
                "bind_us": bind.dur / 1000 if bind else None,
            }
        )
    summary = {}
    for metric in ("host_us", "device_us", "bind_us"):
        values = [row[metric] for row in rows[warmup:] if row[metric] is not None]
        if not values:
            summary[metric] = None
            continue
        ordered = sorted(values)
        trim = len(values) // 10
        trimmed = ordered[trim:-trim] if trim else ordered
        median = statistics.median(values)
        summary[metric] = {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": median,
            "trimmed_mean": statistics.mean(trimmed),
            "p95": ordered[math.ceil(len(values) * 0.95) - 1],
            "mad": statistics.median(abs(value - median) for value in values),
        }
    return {"warmup": warmup, "rows": rows, "summary": summary}


def runtime_identity(package_root):
    directory = package_root / "build/lib/a5/onboard/host_build_graph"
    files = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("libhost_runtime.so", "libaicpu_kernel.so", "aicore_kernel.o", "aicpu_cpu_topo_fallback.json")
    }
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"revision": "runtime-sha256:" + digest, "files": files}


class ManifestCaseCheck:
    def __init__(self, expected):
        self.expected = expected

    def pytest_collection_finish(self, session):
        expected = self.expected
        items = [
            item
            for item in session.items
            if Path(item.path).resolve() == (ROOT / expected["path"]).resolve()
            and getattr(item, "cls", None) is not None
            and item.cls.__name__ == expected["class"]
        ]
        if len(items) != 1:
            raise pytest.UsageError("manifest must select exactly one scene test")
        selected = items[0].instance._matching_cases("a5", SimpleNamespace(config=session.config))
        if len(selected) != 1 or selected[0]["name"] != expected["case"]:
            raise pytest.UsageError("manifest must select exactly one matching case")
        for field in ("params", "config"):
            actual = json.loads(json.dumps(selected[0].get(field, {}), default=str))
            if actual != expected.get(field, {}):
                raise pytest.UsageError(f"manifest {field} differs from executed case {expected['case']}")


def pytest_command(case):
    entry = (
        "import json, sys, pytest; "
        "from tools.benchmarks.a5_hbg_graph_task.collect import ManifestCaseCheck; "
        "raise SystemExit(pytest.main(sys.argv[2:], plugins=[ManifestCaseCheck(json.loads(sys.argv[1]))]))"
    )
    return [sys.executable, "-c", entry, json.dumps(case, sort_keys=True)]


def run(args):
    if not os.environ.get("TASK_DEVICE"):
        raise SystemExit("Run this command inside a task-submit device allocation")
    args.output = args.output.resolve()
    manifest = json.loads(args.manifest.read_text())
    records = []
    args.output.mkdir(parents=True, exist_ok=True)
    identity = runtime_identity(PROJECT_ROOT)
    for index, case in enumerate(manifest["scenes"]):
        if not case.get("measure", False):
            continue
        label = f"{index:03d}-{case['class']}-{case['case']}"
        node = f"{case['path']}::{case['class']}::test_run"
        command = pytest_command(case) + [
            node,
            "--platform",
            "a5",
            "--device",
            os.environ["TASK_DEVICE"],
            "--case",
            case["case"],
            "--manual",
            "include",
            "--pto-session-timeout",
            "300",
            "-q",
            "-s",
        ]
        record = {
            "case": case,
            "revision": identity["revision"],
            "runtime": identity,
            "device": os.environ["TASK_DEVICE"],
            "batches": [],
        }
        environment = os.environ.copy()
        environment["SIMPLER_HOST_STRACE"] = "1"
        environment["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        logs = (args.output / label / "ascend").resolve()
        logs.mkdir(parents=True, exist_ok=True)
        environment["ASCEND_PROCESS_LOG_PATH"] = str(logs)
        for batch in range(-1, args.batches):
            if runtime_identity(PROJECT_ROOT) != identity:
                raise SystemExit("runtime artifacts changed during collection")
            functional = batch == -1
            batch_command = (
                ["timeout", "--kill-after=15s", str(args.timeout)]
                + command
                + (
                    ["--rounds", "1"]
                    if functional
                    else [
                        "--rounds",
                        str(args.rounds + args.warmup),
                        "--skip-golden",
                    ]
                )
            )
            log = args.output / label / ("functional.log" if functional else f"batch-{batch}.log")
            started = time.monotonic()
            print(f"START {label} {'functional' if functional else batch}", flush=True)
            with log.open("w") as stream:
                result = subprocess.run(
                    batch_command,
                    cwd=ROOT,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                code = result.returncode
            if runtime_identity(PROJECT_ROOT) != identity:
                raise SystemExit("runtime artifacts changed during collection")
            entry = {
                "command": batch_command,
                "log": os.path.relpath(log, ROOT),
                "exit_code": code,
                "wall_seconds": time.monotonic() - started,
            }
            if not functional and code == 0:
                entry["timing"] = extract_metrics(log, args.warmup)
                rows = entry["timing"]["rows"]
                entry["valid"] = len(rows) == args.rounds + args.warmup and all(
                    row["host_us"] is not None and row["device_us"] is not None for row in rows
                )
            if functional:
                record["functional"] = entry
            else:
                record["batches"].append(entry)
            write_json(args.output / label / "result.json", record)
            print(f"DONE {label} {'functional' if functional else batch} exit={code}", flush=True)
            if code in (124, 137):
                records.append(record)
                write_json(args.output / "results.json", records)
                return 1
            if code:
                break
        records.append(record)
        write_json(args.output / "results.json", records)
    if not records:
        raise SystemExit("manifest contains no selected scenes")
    return int(
        any(
            record["functional"]["exit_code"] != 0
            or len(record["batches"]) != args.batches
            or any(batch["exit_code"] or not batch.get("valid") for batch in record["batches"])
            for record in records
        )
    )


def summarize(args):
    manifest = json.loads(args.manifest.read_text())
    records = json.loads(args.results.read_text())

    def key(case):
        return case["path"], case["class"], case["case"]

    expected = {key(case): case for case in manifest["scenes"] if case.get("measure", False)}
    actual = {key(record["case"]): record for record in records}
    if not expected or set(expected) != set(actual) or len(actual) != len(records):
        raise SystemExit("results must contain every selected case exactly once")
    rows = []
    for case_key, case in expected.items():
        record = actual[case_key]
        if any(record["case"].get(field, {}) != case.get(field, {}) for field in ("params", "config")):
            raise SystemExit(f"manifest parameters changed: {case_key}")
        batches = record["batches"]
        if record["functional"]["exit_code"] or len(batches) != 3:
            raise SystemExit(f"incomplete functional/three-batch evidence: {case_key}")
        row = {"case": case, "revision": record["revision"], "device": record["device"], "metrics": {}}
        if "runtime" in record:
            row["runtime"] = record["runtime"]
        for batch in batches:
            if batch["exit_code"] or not batch.get("valid"):
                raise SystemExit(f"invalid timing evidence: {case_key}")
            for metric in ("host_us", "device_us"):
                summary = batch["timing"]["summary"][metric]
                if summary is None or summary["count"] < 100:
                    raise SystemExit(f"fewer than 100 steady rounds: {case_key} {metric}")
        for metric in ("host_us", "device_us", "bind_us"):
            summaries = [batch["timing"]["summary"][metric] for batch in batches]
            if any(summary is None for summary in summaries):
                row["metrics"][metric] = None
                continue
            medians = [summary["median"] for summary in summaries]
            row["metrics"][metric] = {
                "batches": summaries,
                "median": statistics.median(medians),
                "trimmed_mean": statistics.median(summary["trimmed_mean"] for summary in summaries),
                "p95": statistics.median(summary["p95"] for summary in summaries),
                "batch_median_range": [min(medians), max(medians)],
            }
        row["functional_log"] = record["functional"]["log"]
        row["timing_logs"] = [batch["log"] for batch in batches]
        rows.append(row)
    if len({(row["revision"], row["device"]) for row in rows}) != 1:
        raise SystemExit("summary cannot mix revisions or devices")
    write_json(args.output, {"schema": 1, "aggregation": "median of three independent batch statistics", "rows": rows})
    print(f"Validated {len(rows)} cases, each with golden and three batches of at least 100 steady rounds")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inv = commands.add_parser("inventory")
    inv.add_argument("output", type=Path)
    bench = commands.add_parser("run")
    bench.add_argument("manifest", type=Path)
    bench.add_argument("output", type=Path)
    bench.add_argument("--rounds", type=int, default=10)
    bench.add_argument("--warmup", type=int, default=3)
    bench.add_argument("--batches", type=int, default=1)
    bench.add_argument("--timeout", type=int, default=360)
    summary = commands.add_parser("summarize")
    summary.add_argument("manifest", type=Path)
    summary.add_argument("results", type=Path)
    summary.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "inventory":
        return inventory(args.output)
    if args.command == "summarize":
        return summarize(args)
    if args.rounds < 1 or args.warmup < 0 or args.batches < 1:
        parser.error("rounds/batches must be positive and warmup nonnegative")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
