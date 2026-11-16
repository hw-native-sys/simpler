#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare and run native MoE without the PyPTO/PyPTO-lib execution environment."""

import argparse
import json
import math
import os
import re
import runpy
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def control_reads(program):
    """Require an actual orchestration read of num_tokens in the compiled program."""
    reads = []
    for path in sorted(program.rglob("*.cpp")):
        if "orchestration" not in path.parts:
            continue
        for line, text in enumerate(path.read_text().splitlines(), 1):
            if text.lstrip().startswith("//"):
                continue
            if re.search(r"get_tensor_data\s*<\s*int32_t\s*>\s*\([^,]*num_tokens\b", text):
                reads.append({"file": str(path.relative_to(program)), "line": line, "code": text.strip()})
    if not reads:
        raise ValueError("MoE orchestration does not read num_tokens with get_tensor_data<int32_t>")
    return reads


def validate_report(report, case, level=None):
    """Require a performance-only full-count run and complete timing measurements."""
    checks = report.get("checks", [])
    required = {(72, 72)}
    if (
        report.get("passed") is not True
        or report.get("compile_only") is not False
        or report.get("numerical_validation") is not False
        or len(checks) != 1
        or any(check.get("passed") is not True for check in checks)
        or {tuple(check.get("counts", [])) for check in checks} != required
    ):
        raise ValueError("Native MoE has not completed its performance-only run")
    if report.get("source_commit") != case["source_commit"] or (
        level is not None and report.get("profiling_level") != level
    ):
        raise ValueError("Native MoE report source or profiling level mismatch")
    if not report.get("control_reads"):
        raise ValueError("Native MoE is missing its executable control-read evidence")
    perf = report.get("performance", {})
    rounds = perf.get("rounds", 0)
    if not isinstance(rounds, int) or rounds < 1 or perf.get("warmup", 0) < 1:
        raise ValueError("Native MoE needs warmup and measured rounds")
    samples = perf.get("device_wall_us", [])
    ranks = list(perf.get("per_rank_device_us", {}).values())
    if len(ranks) != 2 or any(len(values) != rounds for values in [samples, *ranks]):
        raise ValueError("Native MoE is missing Device or rank samples")
    if any(not math.isfinite(x) or x <= 0 for values in [samples, *ranks] for x in values):
        raise ValueError("Native MoE has invalid timing samples")
    if any(not math.isclose(samples[i], max(rank[i] for rank in ranks), abs_tol=1e-6) for i in range(rounds)):
        raise ValueError("Native MoE Device samples must use the per-round rank maximum")
    if report.get("profiling_level") == 0:
        core_samples = perf.get("aicore_wall_us", [])
        core_ranks = list(perf.get("per_rank_aicore_us", {}).values())
        if (
            perf.get("metric") != "chip.run.runner_run.aicore_wall"
            or perf.get("endpoint_semantics") != "first_kernel_start_to_last_kernel_end"
            or len(core_ranks) != 2
            or any(len(values) != rounds for values in [core_samples, *core_ranks])
            or any(not math.isfinite(x) or x <= 0 for values in [core_samples, *core_ranks] for x in values)
            or any(
                not math.isclose(core_samples[i], max(rank[i] for rank in core_ranks), abs_tol=1e-6)
                for i in range(rounds)
            )
            or not math.isclose(statistics.mean(core_samples), perf.get("aicore_us_mean", 0), abs_tol=1e-6)
        ):
            raise ValueError("Native MoE requires complete AICore L0 kernel endpoints and per-round rank maxima")


def run_native(args):
    bundle, output, runtime = args.bundle.resolve(), args.output.resolve(), args.runtime_lib.resolve()
    helpers = runpy.run_path(str(Path(__file__).with_name("moe_bundle.py")))
    meta = helpers["validate_bundle"](bundle)
    gate = runpy.run_path(str(Path(__file__).with_name("device_start_gate.py")))
    gate["validate_runtime_gate"](runtime, args.runtime)
    devices = [int(v) for v in os.environ["TASK_DEVICE"].split(",")]
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError("Native MoE requires two allocated devices")
    runner = Path(os.environ.get("DSV41_RUNNER", ROOT / "build/portable/host/native-runner"))
    if not runner.is_file():
        raise ValueError("Build the native Host runner first")
    output.mkdir(parents=True, exist_ok=False)
    processes, streams = [], []
    started = time.monotonic()
    try:
        for rank, device in enumerate(devices):
            directory = output / f"rank{rank}"
            (directory / "ascend").mkdir(parents=True)
            stream = (directory / "console.log").open("w")
            streams.append(stream)
            env = dict(os.environ, ASCEND_PROCESS_LOG_PATH=str(directory / "ascend"))
            command = [
                str(runner),
                str(bundle),
                str(runtime),
                str(device),
                str(rank),
                str(output / "rootinfo"),
                str(directory),
                str(args.level),
                str(args.warmup),
                str(args.rounds),
                f"--runtime={args.runtime}",
            ]
            processes.append(subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT))
        codes = [process.wait() for process in processes]
        if any(codes):
            raise RuntimeError(f"Native MoE rank execution failed {codes}; see {output}")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        for stream in streams:
            stream.close()
    timing = runpy.run_path(str(Path(__file__).with_name("timing.py")))
    trace = "\n".join(path.read_text() for path in sorted(output.glob("rank*/host/host.*.log")))
    console = "\n".join(path.read_text() for path in sorted(output.glob("rank*/console.log")))
    gates = [line for line in console.splitlines() if "[DEVICE_START_GATE]" in line]
    (output / "benchmark.strace.log").write_text(trace + "\n" + "\n".join(gates) + "\n")
    kernels, walls = timing["paired_spans"](trace)
    if set(kernels) != set(devices):
        raise ValueError("Native MoE timing differs from the allocated devices")
    per_rank = {}
    for device in devices:
        rows = walls[device]
        if len(rows) != args.warmup + args.rounds:
            raise ValueError("Incomplete native MoE timing records")
        per_rank[str(rows[0]["pid"])] = [row["duration_ns"] / 1000 for row in rows[args.warmup :]]
    samples = [max(values[i] for values in per_rank.values()) for i in range(args.rounds)]
    performance = {
        "warmup": args.warmup,
        "rounds": args.rounds,
        "device_wall_us": samples,
        "device_us_mean": statistics.mean(samples),
        "per_rank_device_us": per_rank,
    }
    performance.update(timing["native_performance"](trace, performance, devices))
    if args.level == 0:
        measured = timing["summarize_replay"](output)
        (output / "l0-timing.json").write_text(json.dumps(measured, indent=2) + "\n")
    case = json.loads((bundle / "case.json").read_text())
    report = {
        "case": "moe",
        "flavor": "pypto_lib_latest",
        "source_commit": meta["source_commit"],
        "runtime_lib": str(runtime),
        "runtime": args.runtime,
        "devices": devices,
        "profiling_level": args.level,
        "checks": [{"counts": meta["counts"], "passed": True}],
        "numerical_validation": False,
        "control_reads": meta["control_reads"],
        "performance": performance,
        "compile_only": False,
        "passed": True,
        "execution_seconds": time.monotonic() - started,
        "execution_driver": "native-runner",
    }
    validate_report(report, case, args.level)
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"AICore L{args.level} wall: {performance['aicore_us_mean'] / 1000:.6f} ms ({args.rounds} rounds)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/portable/native-moe")
    parser.add_argument("--reuse-bundle", type=Path)
    parser.add_argument("--level", type=int, choices=(0, 1, 3), default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--runtime-lib", type=Path)
    parser.add_argument("--runtime", choices=("host_build_graph", "percounter"), default="host_build_graph")
    args = parser.parse_args()
    if args.warmup < 1 or args.rounds < 1:
        parser.error("Use at least one warmup and one measured round")
    if args.compile_only:
        helpers = runpy.run_path(str(Path(__file__).with_name("moe_bundle.py")))
        helpers["prepare_bundle"](args.bundle, args.build_dir, args.reuse_bundle)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "result.json").write_text(
            json.dumps({"compile_only": True, "passed": False, "compilation_passed": True, "checks": []}, indent=2)
            + "\n"
        )
    else:
        if not os.environ.get("TASK_DEVICE") or args.runtime_lib is None:
            parser.error("Run through production/run.sh with an allocated runtime and two devices")
        run_native(args)


if __name__ == "__main__":
    main()
