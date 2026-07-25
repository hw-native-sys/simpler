#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build a compact Host-O / Device-S Perfetto trace from one profiled run."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import TypedDict


class StraceSpan(TypedDict):
    name: str
    ts_ns: int
    dur_ns: int
    attrs: dict[str, str]


class SchedulerLayer(TypedDict):
    layer: int
    start_cycles: int
    end_cycles: int
    start_ns: int
    dur_ns: int
    records: int
    unique_tasks: int


STRACE_RE = re.compile(
    r"\[STRACE\].*?\binv=(?P<inv>\d+).*?\bname=(?P<name>\S+) "
    r"ts=(?P<ts>\d+) dur=(?P<dur>\d+)(?: (?P<attrs>.*))?$"
)
WALL_TIME_RE = re.compile(r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]")


def parse_attrs(raw: str | None) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for token in (raw or "").split():
        if "=" in token:
            key, value = token.split("=", 1)
            attrs[key] = value
    return attrs


def load_strace_spans(log_path: Path) -> dict[int, list[StraceSpan]]:
    rounds: dict[int, list[StraceSpan]] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = STRACE_RE.search(line)
        if match is None:
            continue
        invocation = int(match.group("inv"))
        rounds.setdefault(invocation, []).append(
            {
                "name": match.group("name"),
                "ts_ns": int(match.group("ts")),
                "dur_ns": int(match.group("dur")),
                "attrs": parse_attrs(match.group("attrs")),
            }
        )
    if not rounds:
        raise ValueError(f"no STRACE spans found in {log_path}")
    return rounds


def wall_time_ns(raw: str) -> int:
    stamp = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S.%f")
    epoch = datetime(1970, 1, 1)
    return round((stamp - epoch).total_seconds() * 1_000_000_000)


def load_whole_graph_anchors(log_path: Path) -> dict[str, int]:
    anchors: dict[str, int] = {}
    patterns = {
        "orch_ready": "Device orchestration ready:",
        "profile_ready": "Performance profiling initialized",
        "aicpu_launch": "=== launch_aicpu_kernel simpler_aicpu_exec ===",
        "device_complete": "=== aclrtSynchronizeStreamWithTimeout stream_aicore_ ===",
    }
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = WALL_TIME_RE.match(line)
        if match is None:
            continue
        for name, marker in patterns.items():
            if marker in line:
                anchors[name] = wall_time_ns(match.group("ts"))
    required = {"orch_ready", "aicpu_launch", "device_complete"}
    missing = sorted(required - anchors.keys())
    if missing:
        raise ValueError(f"missing whole-graph Host anchors in {log_path}: {', '.join(missing)}")
    return anchors


def layer_value(span: StraceSpan) -> int:
    return int(span["attrs"].get("layer", "-1"))


def find_span(spans: list[StraceSpan], name: str, layer: int | None = None) -> StraceSpan:
    matches = [span for span in spans if span["name"] == name and (layer is None or layer_value(span) == layer)]
    if len(matches) != 1:
        suffix = "" if layer is None else f" for layer {layer}"
        raise ValueError(f"expected one {name} span{suffix}, found {len(matches)}")
    return matches[0]


def find_optional_span(
    spans: list[StraceSpan],
    name: str,
    layer: int,
) -> StraceSpan | None:
    matches = [span for span in spans if span["name"] == name and layer_value(span) == layer]
    if len(matches) > 1:
        raise ValueError(f"expected at most one {name} span for layer {layer}, found {len(matches)}")
    return matches[0] if matches else None


def metadata_event(pid: int, tid: int, kind: str, value: str) -> dict[str, object]:
    return {"ph": "M", "name": kind, "pid": pid, "tid": tid, "args": {"name": value}}


def complete_event(
    pid: int,
    tid: int,
    name: str,
    category: str,
    ts_ns: int,
    dur_ns: int,
    base_ns: int,
    args: dict[str, object],
) -> dict[str, object]:
    return {
        "ph": "X",
        "pid": pid,
        "tid": tid,
        "name": name,
        "cat": category,
        "ts": (ts_ns - base_ns) / 1000.0,
        "dur": dur_ns / 1000.0,
        "args": args,
    }


def read_l2(l2_path: Path) -> tuple[int, list[list[int]]]:
    data = json.loads(l2_path.read_text(encoding="utf-8"))
    return int(data["metadata"]["clock_freq_hz"]), data["aicore_tasks"]


def build_pipeline_events(  # noqa: PLR0912 -- translates independent trace event kinds
    invocation: int,
    round_index: int,
    spans: list[StraceSpan],
    l2_path: Path,
    title: str,
) -> list[dict[str, object]]:
    frequency, aicore_tasks = read_l2(l2_path)
    task_layers = sorted(
        {
            layer_value(span)
            for span in spans
            if span["name"] == "simpler_run.host_orch.layer_build" and int(span["attrs"].get("tasks", "0")) > 0  # type: ignore[union-attr]
        }
    )
    if not task_layers:
        raise ValueError(f"invocation {invocation} has no task-bearing Host-O layers")

    layer_ranges: dict[int, tuple[int, int]] = {}
    for layer in task_layers:
        build = find_span(spans, "simpler_run.host_orch.layer_build", layer)
        attrs = build["attrs"]
        assert isinstance(attrs, dict)
        layer_ranges[layer] = (int(attrs["task_begin"]), int(attrs["task_end"]))

    s_layers: list[SchedulerLayer] = []
    for layer in task_layers:
        begin, end = layer_ranges[layer]
        records = [record for record in aicore_tasks if begin <= int(record[1]) < end]
        if not records:
            raise ValueError(f"{l2_path} has no AICore records for task range [{begin},{end})")
        start_cycles = min(int(record[3]) for record in records)
        end_cycles = max(int(record[4]) for record in records)
        s_layers.append(
            {
                "layer": layer,
                "start_cycles": start_cycles,
                "end_cycles": end_cycles,
                "start_ns": 0,
                "dur_ns": 0,
                "records": len(records),
                "unique_tasks": len({int(record[1]) for record in records}),
            }
        )

    completion_waits = [
        span
        for span in spans
        if span["name"] == "simpler_run.host_orch.epoch.wait_device"
        and span["attrs"].get("kind") in {"exec-done", "storage-free", "slot-free"}  # type: ignore[union-attr]
    ]
    if not completion_waits:
        raise ValueError(f"invocation {invocation} has no Device completion acknowledgement")
    completion_by_epoch: dict[int, StraceSpan] = {}
    for wait in completion_waits:
        epoch = int(wait["attrs"]["target_epoch"])  # type: ignore[index]
        end_ns = int(wait["ts_ns"]) + int(wait["dur_ns"])
        current = completion_by_epoch.get(epoch)
        if current is None or end_ns < int(current["ts_ns"]) + int(current["dur_ns"]):
            completion_by_epoch[epoch] = wait
    anchor_epoch = max(completion_by_epoch)
    anchor_wait = completion_by_epoch[anchor_epoch]
    anchor_layer = anchor_epoch - 1
    anchor_s = next((layer for layer in s_layers if layer["layer"] == anchor_layer), None)
    if anchor_s is None:
        raise ValueError(f"exec-done epoch {anchor_epoch} has no matching S layer")
    anchor_host_ns = int(anchor_wait["ts_ns"]) + int(anchor_wait["dur_ns"])
    anchor_device_ns = round(int(anchor_s["end_cycles"]) * 1_000_000_000 / frequency)
    device_to_host_offset_ns = anchor_host_ns - anchor_device_ns
    for s_layer in s_layers:
        start_ns = round(int(s_layer["start_cycles"]) * 1_000_000_000 / frequency) + device_to_host_offset_ns
        end_ns = round(int(s_layer["end_cycles"]) * 1_000_000_000 / frequency) + device_to_host_offset_ns
        s_layer["start_ns"] = start_ns
        s_layer["dur_ns"] = end_ns - start_ns

    ack_residuals: dict[int, int] = {}
    for epoch, wait in completion_by_epoch.items():
        layer = epoch - 1
        s_layer = next((item for item in s_layers if item["layer"] == layer), None)
        if s_layer is None:
            continue
        mapped_end = int(s_layer["start_ns"]) + int(s_layer["dur_ns"])
        ack_end = int(wait["ts_ns"]) + int(wait["dur_ns"])
        ack_residuals[layer] = ack_end - mapped_end

    host_starts = [span["ts_ns"] for span in spans if span["name"].startswith("simpler_run.host_orch.")]
    device_starts = [layer["start_ns"] for layer in s_layers]
    base_ns = min(*host_starts, *device_starts)
    pid = round_index + 1
    events: list[dict[str, object]] = [
        metadata_event(pid, 0, "process_name", f"{title} - round {round_index}"),
        metadata_event(pid, 1, "thread_name", "O active (Host build/materialize/stage/commit)"),
        metadata_event(pid, 2, "thread_name", "O synchronization (S release/completion)"),
        metadata_event(pid, 3, "thread_name", "S execution (AICore envelope)"),
    ]

    phase_names = {
        "simpler_run.host_orch.layer_build": "build",
        "simpler_run.host_orch.epoch.materialize": "materialize",
        "simpler_run.host_orch.epoch.stage_upload": "stage upload",
        "simpler_run.host_orch.epoch.commit": "commit",
    }
    for span in spans:
        phase = phase_names.get(str(span["name"]))
        if phase is None:
            continue
        attrs = dict(span["attrs"])
        layer = layer_value(span) if phase == "build" else int(attrs["epoch"]) - 1
        if layer not in task_layers:
            continue
        cache_mode = attrs.get("cache", "none")
        events.append(
            complete_event(
                pid,
                1,
                f"O L{layer} {phase} ({cache_mode})",
                "host_orchestrator",
                int(span["ts_ns"]),
                int(span["dur_ns"]),
                base_ns,
                dict[str, object](layer=layer, phase=phase, **attrs),
            )
        )

    for wait in spans:
        if wait["name"] != "simpler_run.host_orch.epoch.wait_device":
            continue
        attrs = dict(wait["attrs"])
        kind = attrs.get("kind", "device")
        target_epoch = int(attrs.get("target_epoch", "0"))
        events.append(
            complete_event(
                pid,
                2,
                f"wait {kind} epoch {target_epoch}",
                "host_backpressure",
                int(wait["ts_ns"]),
                int(wait["dur_ns"]),
                base_ns,
                dict(attrs),
            )
        )

    for s_layer in s_layers:
        layer = int(s_layer["layer"])
        begin, end = layer_ranges[layer]
        events.append(
            complete_event(
                pid,
                3,
                f"S L{layer} ({s_layer['records']} AICore records)",
                "device_scheduler",
                int(s_layer["start_ns"]),
                int(s_layer["dur_ns"]),
                base_ns,
                {
                    "layer": layer,
                    "task_range": f"[{begin},{end})",
                    "records": s_layer["records"],
                    "unique_task_ids": s_layer["unique_tasks"],
                    "timing": (
                        "actual AICore cycles; clock offset anchored by earliest completion acknowledgement "
                        f"for epoch {anchor_epoch} ({anchor_wait['attrs']['kind']})"
                    ),
                    "ack_residual_ns": ack_residuals.get(layer),
                },
            )
        )

    for layer in task_layers[1:]:
        build = find_span(spans, "simpler_run.host_orch.layer_build", layer)
        prior_s = next(item for item in s_layers if item["layer"] == layer - 1)
        build_start = int(build["ts_ns"])
        build_end = build_start + int(build["dur_ns"])
        s_start = int(prior_s["start_ns"])
        s_end = s_start + int(prior_s["dur_ns"])
        overlap_ns = max(0, min(build_end, s_end) - max(build_start, s_start))
        events.append(
            {
                "ph": "C",
                "pid": pid,
                "tid": 1,
                "name": "O/S overlap ns",
                "ts": (build_start - base_ns) / 1000.0,
                "args": {f"O L{layer} with S L{layer - 1}": overlap_ns},
            }
        )
    return events


def parse_layer_task_counts(raw: str) -> list[int]:
    counts = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("--layer-task-counts must contain positive comma-separated integers")
    return counts


def build_whole_graph_events(
    spans: list[StraceSpan],
    anchors: dict[str, int],
    l2_path: Path,
    layer_task_counts: list[int],
    title: str,
    alignment: str,
) -> list[dict[str, object]]:
    bind = find_span(spans, "simpler_run.bind")
    device_wall = find_span(spans, "simpler_run.runner_run.device_wall")
    frequency, aicore_tasks = read_l2(l2_path)
    if not aicore_tasks:
        raise ValueError(f"{l2_path} has no AICore records")

    layer_ranges: list[tuple[int, int]] = []
    begin = 0
    for count in layer_task_counts:
        layer_ranges.append((begin, begin + count))
        begin += count

    orch_end_ns = anchors["orch_ready"]
    orch_dur_ns = int(bind["dur_ns"])
    orch_start_ns = orch_end_ns - orch_dur_ns
    if alignment == "compact":
        device_wall_start_ns = orch_end_ns
        completion_ns = device_wall_start_ns + int(device_wall["dur_ns"])
        alignment_note = "device wall placed immediately after Host O; profiling setup gap omitted"
    else:
        device_wall_start_ns = anchors["device_complete"] - int(device_wall["dur_ns"])
        completion_ns = anchors["device_complete"]
        alignment_note = "device completion aligned to the profiled Host wall-clock acknowledgement"

    global_end_cycles = max(int(record[4]) for record in aicore_tasks)
    s_layers: list[dict[str, int]] = []
    for layer, (begin, end) in enumerate(layer_ranges):
        records = [record for record in aicore_tasks if begin <= int(record[1]) < end]
        if not records:
            raise ValueError(f"{l2_path} has no AICore records for task range [{begin},{end})")
        start_cycles = min(int(record[3]) for record in records)
        end_cycles = max(int(record[4]) for record in records)
        start_ns = completion_ns - round((global_end_cycles - start_cycles) * 1_000_000_000 / frequency)
        end_ns = completion_ns - round((global_end_cycles - end_cycles) * 1_000_000_000 / frequency)
        s_layers.append(
            {
                "layer": layer,
                "start_ns": start_ns,
                "dur_ns": end_ns - start_ns,
                "records": len(records),
                "unique_tasks": len({int(record[1]) for record in records}),
            }
        )

    first_s_ns = min(layer["start_ns"] for layer in s_layers)
    base_ns = min(orch_start_ns, first_s_ns)
    pid = 1
    events: list[dict[str, object]] = [
        metadata_event(pid, 0, "process_name", title),
        metadata_event(pid, 1, "thread_name", "O active (Host whole-graph bind/build/relocate/upload)"),
        metadata_event(pid, 2, "thread_name", "Host/device setup and synchronization"),
        metadata_event(pid, 3, "thread_name", "S execution (AICore envelope)"),
        complete_event(
            pid,
            1,
            "O whole graph (Host bind/build/relocate/upload)",
            "host_orchestrator",
            orch_start_ns,
            orch_dur_ns,
            base_ns,
            {
                "layers": len(layer_task_counts),
                "tasks": sum(layer_task_counts),
                "timing": "Host bind STRACE duration; end aligned to Device orchestration ready log",
            },
        ),
        complete_event(
            pid,
            2,
            "Device run wall (AICPU + S + teardown)",
            "device_wall",
            device_wall_start_ns,
            int(device_wall["dur_ns"]),
            base_ns,
            {"alignment": alignment_note},
        ),
    ]

    if alignment == "profile-wall" and first_s_ns > orch_end_ns:
        events.append(
            complete_event(
                pid,
                2,
                "Host/device setup before first AICore task",
                "host_device_setup",
                orch_end_ns,
                first_s_ns - orch_end_ns,
                base_ns,
                {
                    "profile_ready_seen": "profile_ready" in anchors,
                    "aicpu_launch_ns": anchors["aicpu_launch"],
                },
            )
        )

    for layer, s_layer in enumerate(s_layers):
        begin, end = layer_ranges[layer]
        events.append(
            complete_event(
                pid,
                3,
                f"S L{layer} ({s_layer['records']} AICore records)",
                "device_scheduler",
                s_layer["start_ns"],
                s_layer["dur_ns"],
                base_ns,
                {
                    "layer": layer,
                    "task_range": f"[{begin},{end})",
                    "records": s_layer["records"],
                    "unique_task_ids": s_layer["unique_tasks"],
                    "timing": f"actual AICore cycles; {alignment_note}",
                },
            )
        )
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="run.log containing Host STRACE spans")
    parser.add_argument("--l2-dir", type=Path, required=True, help="test output containing L2 JSON files")
    parser.add_argument("--output", type=Path, required=True, help="output Perfetto JSON")
    parser.add_argument("--title", default="Host O / Device S", help="Perfetto process name")
    parser.add_argument(
        "--layer-task-counts",
        help="whole-graph mode: comma-separated scheduler task counts for consecutive layers",
    )
    parser.add_argument(
        "--whole-graph-alignment",
        choices=("compact", "profile-wall"),
        default="compact",
        help="whole-graph mode: omit profiling setup or retain its Host wall-clock gap",
    )
    args = parser.parse_args()

    rounds = load_strace_spans(args.log)
    has_pipeline_spans = any(
        span["name"] == "simpler_run.host_orch.layer_build" for spans in rounds.values() for span in spans
    )
    events: list[dict[str, object]] = []
    if has_pipeline_spans:
        for round_index, invocation in enumerate(sorted(rounds)):
            root_l2_path = args.l2_dir / "l2_swimlane_records.json"
            l2_path = (
                root_l2_path
                if len(rounds) == 1 and root_l2_path.is_file()
                else args.l2_dir / f"round_{round_index:03d}" / "l2_swimlane_records.json"
            )
            if not l2_path.is_file():
                raise FileNotFoundError(l2_path)
            events.extend(build_pipeline_events(invocation, round_index, rounds[invocation], l2_path, args.title))
        mode = "pipeline"
    else:
        if args.layer_task_counts is None:
            raise ValueError("whole-graph mode requires --layer-task-counts")
        if len(rounds) != 1:
            raise ValueError("whole-graph mode currently requires exactly one profiled invocation")
        l2_path = args.l2_dir / "l2_swimlane_records.json"
        if not l2_path.is_file():
            raise FileNotFoundError(l2_path)
        invocation = next(iter(rounds))
        events = build_whole_graph_events(
            rounds[invocation],
            load_whole_graph_anchors(args.log),
            l2_path,
            parse_layer_task_counts(args.layer_task_counts),
            args.title,
            args.whole_graph_alignment,
        )
        mode = "whole-graph"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"traceEvents": events}, indent=2) + "\n", encoding="utf-8")
    print(f"mode={mode}")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
