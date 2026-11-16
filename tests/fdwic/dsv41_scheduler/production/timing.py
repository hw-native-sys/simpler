#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Strict raw AICore-kernel endpoints, paired with RunWall only for identity/evidence."""

import argparse
import json
import re
import statistics
from pathlib import Path

METRIC = "chip.run.runner_run.aicore_wall"
RUNWALL = "chip.run.runner_run.device_wall"
_FIELD = re.compile(r"\b([A-Za-z_][A-Za-z_0-9]*)=([^\s]+)")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _records(text, name, required):
    result = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        if "[STRACE]" not in line or f"name={name} " not in line:
            continue
        pairs = _FIELD.findall(line)
        fields = dict(pairs)
        for field in required:
            require(sum(key == field for key, _ in pairs) == 1, f"{name}:{line_number}: missing/duplicate {field}")
        require(fields["clk"] == "dev" and fields["ts"] == "0", f"{name}: not a device span")
        key = (int(fields["pid"]), int(fields["inv"]))
        require(key[0] > 0 and key[1] > 0 and key not in result, f"{name}: duplicate/invalid (pid,inv) {key}")
        result[key] = fields
    return result


def paired_spans(text):
    """Reject missing/partial/duplicate endpoints; never substitute RunWall or an older loop span."""
    common = ("pid", "inv", "name", "clk", "ts", "dur")
    walls = _records(text, RUNWALL, (*common, "dev_id", "dev_start_cycle", "dev_end_cycle", "dev_cnt_hz"))
    kernels = _records(
        text, METRIC, (*common, "clock_source", "aicore_start_cycle", "aicore_end_cycle", "aicore_cnt_hz")
    )
    require(
        walls and set(walls) == set(kernels), "AICore endpoints and RunWall must have identical nonempty (pid,inv) sets"
    )
    kernel_by_device, wall_by_device = {}, {}
    for (pid, invocation), wall in walls.items():
        core = kernels[(pid, invocation)]
        device, ws, we, whz = (int(wall[key]) for key in ("dev_id", "dev_start_cycle", "dev_end_cycle", "dev_cnt_hz"))
        ks, ke, khz = (int(core[key]) for key in ("aicore_start_cycle", "aicore_end_cycle", "aicore_cnt_hz"))
        require(device >= 0 and ws > 0 and we > ws and whz == 1_000_000_000, "Invalid A5 RunWall endpoints")
        require(core["clock_source"] == "aicore" and khz > 0, "Wrong AICore endpoint clock")
        require(ks > 0 and ke > ks, "Missing or reversed kernel endpoints")
        wall_ns = (we - ws) * 1_000_000_000 // whz
        kernel_ns = (ke - ks) * 1_000_000_000 // khz
        # The two clocks need not have the same epoch. Compare durations only;
        # the existing gate audit retains its separately validated RunWall domain.
        require(kernel_ns <= wall_ns + 1, "Kernel duration exceeds the associated whole-run duration")
        require(abs(int(wall["dur"]) - wall_ns) <= 1, "RunWall duration disagrees with its endpoints")
        require(abs(int(core["dur"]) - kernel_ns) <= 1, "AICore duration disagrees with strict kernel endpoints")
        identity = {"pid": pid, "invocation": invocation, "device": device}
        wall_by_device.setdefault(device, []).append(
            {
                **identity,
                "duration_ns": int(wall["dur"]),
                "endpoint_duration_ns": wall_ns,
                "start_cycle": ws,
                "end_cycle": we,
            }
        )
        kernel_by_device.setdefault(device, []).append(
            {**identity, "duration_ns": kernel_ns, "start_cycle": ks, "end_cycle": ke, "clock_hz": khz}
        )
    for spans in (*wall_by_device.values(), *kernel_by_device.values()):
        spans.sort(key=lambda span: (span["pid"], span["invocation"]))
        require(len({span["pid"] for span in spans}) == 1, "A rank changed worker pid")
        require(
            [span["invocation"] for span in spans] == list(range(1, len(spans) + 1)), "Incomplete invocation sequence"
        )
    return kernel_by_device, wall_by_device


def summarize(per_rank, devices, warmup, rounds):
    require(len(per_rank) == len(devices) == len(set(devices)), "Invalid rank/device set")
    require(warmup >= 1 and rounds >= 1, "Need warmup and measured rounds")
    require(all(len(values) == rounds for values in per_rank), "Incomplete measured rounds")
    samples = [max(values[index] for values in per_rank) for index in range(rounds)]
    return {
        "metric": METRIC,
        "profiling_level": 0,
        "endpoint_semantics": "first_kernel_start_to_last_kernel_end",
        "devices": devices,
        "warmup": warmup,
        "rounds": rounds,
        "per_rank_samples_ms": per_rank,
        "samples_ms": samples,
        "mean_ms": statistics.mean(samples),
    }


def replay_rank_samples(directory, result):
    native = result.get("execution") == "native_moe_host_control"
    require(result["passed"], "Execution failed")
    if native:
        require(result.get("numerical_validation") is False, "Native MoE is a performance-only run")
    else:
        require(result["validated_against_fused_prefix"], "Replay numerical validation failed")
    text = "\n".join(path.read_text() for path in sorted((directory / "host").glob("host.*.log")))
    kernels, _ = paired_spans(text)
    device = result["device"]
    require(set(kernels) == {device}, "Raw replay device differs from result")
    runs = [
        tuple(map(int, values))
        for values in re.findall(
            r"run=(\d+) scheduler_mode=(\d+) profiling=(\d+) warmup=(\d+)",
            (directory / "console.log").read_text(),
        )
    ]
    warmup, rounds = result["warmup"], result["rounds"]
    mode = {"host_build_graph": 1, "percounter": 5}[result["runtime"]]
    require(len(runs) in (warmup + rounds, warmup + rounds + 2), "Incomplete replay run sequence")
    require(
        runs[: warmup + rounds] == [(index, mode, 0, int(index < warmup)) for index in range(warmup + rounds)],
        "Performance samples must be L0 in the selected AICore scheduler mode",
    )
    if len(runs) > warmup + rounds:
        require(
            runs[-2:] == [(warmup + rounds, mode, 1, 0), (warmup + rounds + 1, mode, 3, 0)],
            "Unexpected profiled runs after L0 samples",
        )
    spans = kernels[device]
    snapshot = result["snapshot_inputs"] or native
    require(len(spans) == (len(runs) if snapshot else 1 + 2 * len(runs)), "Missing replay kernel endpoints")
    return [
        spans[index if snapshot else 2 + 2 * index]["duration_ns"] / 1e6 for index in range(warmup, warmup + rounds)
    ]


def summarize_replay(directory):
    reports = [json.loads(path.read_text()) for path in directory.glob("rank*/result.json")]
    reports.sort(key=lambda result: result["rank"])
    require(
        reports and [result["rank"] for result in reports] == list(range(reports[0]["ranks"])),
        "Incomplete rank results",
    )
    first = reports[0]
    for result in reports:
        require(
            all(result[key] == first[key] for key in ("ranks", "runtime", "warmup", "rounds", "snapshot_inputs")),
            "Ranks disagree on the replay contract",
        )
    return summarize(
        [replay_rank_samples(directory / f"rank{result['rank']}", result) for result in reports],
        [result["device"] for result in reports],
        first["warmup"],
        first["rounds"],
    )


def native_performance(text, performance, devices):
    kernels, walls = paired_spans(text)
    require(set(kernels) == set(devices) and len(devices) == 2, "Native MoE rank/device mismatch")
    warmup, rounds = performance["warmup"], performance["rounds"]
    per_rank, per_pid = [], {}
    for device in devices:
        spans = kernels[device]
        require(len(spans) == warmup + rounds, "Incomplete native kernel endpoints")
        pid = str(spans[0]["pid"])
        observed = [span["duration_ns"] / 1000 for span in walls[device][warmup:]]
        reported = performance["per_rank_device_us"].get(pid, performance["per_rank_device_us"].get(int(pid), []))
        require(
            len(reported) == rounds and all(abs(x - y) < 1e-6 for x, y in zip(observed, reported)),
            "Native Device report differs from the paired raw RunWall",
        )
        values = [span["duration_ns"] / 1e6 for span in spans[warmup:]]
        per_rank.append(values)
        per_pid[pid] = [value * 1000 for value in values]
    result = summarize(per_rank, devices, warmup, rounds)
    return {
        "metric": METRIC,
        "endpoint_semantics": result["endpoint_semantics"],
        "aicore_wall_us": [value * 1000 for value in result["samples_ms"]],
        "aicore_us_mean": result["mean_ms"] * 1000,
        "per_rank_aicore_us": per_pid,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Completed replay output containing rank directories")
    args = parser.parse_args()
    result = summarize_replay(args.run)
    (args.run / "l0-timing.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"AICore L0 wall: {result['mean_ms']:.6f} ms ({result['rounds']} rounds, per-round rank maximum)")
