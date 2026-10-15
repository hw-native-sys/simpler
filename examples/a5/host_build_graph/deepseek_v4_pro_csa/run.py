#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""HBG CSA checks and measurements with explicit scheduler provenance."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

if __package__:
    from . import workload
    from .analyze_perf import summarize
else:
    import workload
    from analyze_perf import summarize


def validate_task_capture(raw_path, deps_path, level):
    from simpler_setup.tools.swimlane_converter import read_perf_data  # noqa: PLC0415

    raw = json.loads(raw_path.read_text())
    deps = json.loads(deps_path.read_text())
    parsed = read_perf_data(str(raw_path))
    if raw["chip_swimlane_level"] != level:
        raise ValueError("Wrong chip swimlane collection level")
    expected = Counter(
        {int(task["task_id"]): task["block_num"] * sum(k >= 0 for k in task["kernel_ids"]) for task in deps["tasks"]}
    )
    actual = Counter(task["task_id"] for task in parsed["tasks"])
    if actual != expected or len(parsed["tasks"]) != workload.fixture.manifest()["expected_physical_tasks"]:
        raise ValueError("CSA task capture is incomplete or the workload changed")
    if len(raw["aicore_tasks"]) != len(parsed["tasks"]):
        raise ValueError("AICore rows were lost during parsing")
    summary = {
        "logical_tasks": len(actual),
        "physical_tasks": len(parsed["tasks"]),
        "clock_freq_hz": raw["metadata"]["clock_freq_hz"],
        "level": level,
    }
    return raw, parsed, summary


def validate_trace(raw_path, deps_path, level, *, require_aicore=False):
    raw, parsed, summary = validate_task_capture(raw_path, deps_path, level)
    metadata = raw["metadata"]
    if metadata.get("runtime") != workload.RUNTIME:
        raise ValueError("Expected a host-built graph trace")
    producer = None
    if level == 4:
        if metadata.get("orchestrator_source") != "host":
            raise ValueError("Expected host orchestration phases")
        scheduler = raw.get("scheduler_tasks", {})
        producer = scheduler.get("producer")
        if producer not in ("aicpu", "aicore"):
            raise ValueError("Missing scheduler producer identity")
        if len(scheduler.get("records", [])) != len(parsed["tasks"]):
            raise ValueError("Scheduler/AICore records do not match")
        if any(task["dispatch_time_us"] is None or task["finish_time_us"] is None for task in parsed["tasks"]):
            raise ValueError("Scheduler/AICore record join is incomplete")
        streams = raw.get("scheduler_records", {}).get("streams", [])
        if not streams or any(stream.get("producer") != producer for stream in streams):
            raise ValueError("Scheduler phase producers are missing or inconsistent")
        for stream in streams:
            capture = stream.get("capture", {})
            if (
                capture.get("dropped") != 0
                or capture.get("truncated") is not False
                or capture.get("committed") != len(stream.get("records", []))
            ):
                raise ValueError("Scheduler phases are incomplete")
        host = metadata.get("host_capture", {})
        if (
            host.get("status") != "complete"
            or host.get("dropped_records") != 0
            or host.get("recorded_records") != host.get("expected_records")
            or host.get("recorded_records", 0) < 1
            or not raw.get("host_orchestrator_phases")
            or sum(len(phases) for phases in raw["host_orchestrator_phases"]) != host["recorded_records"]
        ):
            raise ValueError("Host orchestration capture is incomplete")
        print(f"[CSA HBG] scheduler producer: {producer}", flush=True)
        if producer == "aicpu":
            print(
                "[CSA HBG] AICPU scheduling: this run does NOT validate the distributed AICore scheduler.", flush=True
            )
    if require_aicore and producer != "aicore":
        raise ValueError("Distributed scheduler required: need an L4 trace with scheduler producer=aicore")
    timeline = parsed.get("timeline_metadata", {})
    return {
        **summary,
        "runtime": workload.RUNTIME,
        "scheduler_producer": producer,
        "distributed_scheduler_verified": producer == "aicore",
        "require_aicore_scheduler": require_aicore,
        "host_capture": metadata.get("host_capture"),
        "clock_alignment_status": timeline.get("clock_alignment", {}).get("status"),
        "cross_domain_latency_available": timeline.get("cross_domain_latency_available", False),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="HBG CSA correctness, performance and L1/L4 traces.")
    parser.add_argument("mode", choices=("check", "bench", "l1", "l4"))
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "--require-aicore-scheduler", action="store_true", help="L4 only: reject AICPU scheduler fallback"
    )
    args = parser.parse_args(argv)
    if args.require_aicore_scheduler and args.mode != "l4":
        parser.error("--require-aicore-scheduler requires l4; other modes cannot establish scheduler provenance")
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.environ.setdefault("OMP_NUM_THREADS", "16")
    os.environ.setdefault("MKL_NUM_THREADS", "16")
    workload.fixture.verify_snapshot()
    identity = workload.checkout_identity()
    outputs = workload.fixture.REPO_ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    result_dir = Path(tempfile.mkdtemp(prefix=f"csa-hbg-{args.mode}-", dir=outputs))
    ascend = result_dir / "ascend"
    ascend.mkdir()
    env = dict(os.environ, ASCEND_PROCESS_LOG_PATH=str(ascend))
    report = {"mode": args.mode, "device": args.device, "identity": identity, "commands": []}
    report_path = result_dir / "result.json"

    def save():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def execute(label, extra, expected_checks):
        case_glob = "TestDeepSeekV4ProCSAHBG_B4S2Seed0_*"
        before = set(outputs.glob(case_glob))
        command = [
            sys.executable,
            "-u",
            str(workload.CASE_DIR / "test_deepseek_v4_pro_csa.py"),
            "-p",
            "a5",
            "-d",
            str(args.device),
            "--manual",
            "include",
            *extra,
        ]
        report["commands"].append(command)
        save()
        log = result_dir / f"{label}.log"
        print(f"[CSA] {label}: {log}", flush=True)
        with log.open("w") as stream:
            result = subprocess.run(
                command, cwd=workload.fixture.REPO_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False
            )
        if result.returncode:
            raise RuntimeError(f"{label} failed (exit {result.returncode}); see {log}")
        if expected_checks and log.read_text().count("[CSA] ") != expected_checks * 7:
            raise RuntimeError(f"{label}: missing seven-output validation records")
        return log, set(outputs.glob(case_glob)) - before

    try:
        if args.mode in ("check", "bench"):
            execute("check", ["--rounds", "3"], 3)
            report["correctness"] = {"rounds": 3, "outputs_per_round": 7, "passed": True}
            if args.mode == "bench":
                log, _ = execute("bench", ["--rounds", "105", "--skip-golden"], 0)
                report["benchmark"] = summarize(log)
                median = report["benchmark"]["metrics"]["device_us"]["median"]
                print(f"[CSA] device_us median={median:.3f} (5 warmup + 100 measured)")
        else:
            level = int(args.mode[1:])
            _, graph_dirs = execute("deps", ["--enable-dep-gen"], 1)
            _, timing_dirs = execute("timing", ["--enable-chip-swimlane", str(level)], 1)
            if len(graph_dirs) != 1 or len(timing_dirs) != 1:
                raise RuntimeError("Expected one fresh graph directory and one fresh timing directory")
            (graph_dir,) = graph_dirs
            (timing_dir,) = timing_dirs
            raw_path, deps_path = result_dir / "chip_swimlane_records.json", result_dir / "deps.json"
            shutil.copyfile(graph_dir / "deps.json", deps_path)
            shutil.copyfile(timing_dir / "chip_swimlane_records.json", raw_path)
            for pattern in ("name_map_*.json", "host_clock_alignment.*.log", "dispatch_identity.json"):
                for companion in timing_dir.glob(pattern):
                    shutil.copyfile(companion, result_dir / companion.name)
            report["trace"] = validate_trace(raw_path, deps_path, level, require_aicore=args.require_aicore_scheduler)
            trace = result_dir / f"L{level}.json"
            command = [
                sys.executable,
                "-m",
                "simpler_setup.tools.swimlane_converter",
                str(raw_path),
                "--deps-json",
                str(deps_path),
                "--output",
                str(trace),
            ]
            report["commands"].append(command)
            with (result_dir / "convert.log").open("w") as stream:
                subprocess.run(
                    command,
                    cwd=workload.fixture.REPO_ROOT,
                    env=env,
                    check=True,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            if not json.loads(trace.read_text())["traceEvents"]:
                raise RuntimeError("Empty converted swimlane")
            report["trace"]["file"] = str(trace)
            report["correctness"] = {"rounds": 2, "outputs_per_round": 7, "passed": True}
        if workload.checkout_identity() != identity:
            raise RuntimeError("simpler binaries or sources changed during this run; discard the measurement")
        report["passed"] = True
    except Exception as error:
        report["passed"] = False
        report["error"] = str(error)
        raise
    finally:
        save()
        print(f"[CSA] result: {report_path}", flush=True)


if __name__ == "__main__":
    main()
