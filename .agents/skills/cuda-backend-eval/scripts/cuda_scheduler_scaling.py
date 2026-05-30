#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Summarize CUDA persistent-device scheduler-block scaling smokes."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def _artifact_label(path: Path) -> str:
    name = path.stem.lower()
    if "a100" in name:
        return "a100"
    if "h200" in name:
        return "h200"
    return name


def _scheduler_blocks(payload: dict[str, Any]) -> int:
    value = payload.get("scheduler_blocks")
    if isinstance(value, int):
        return value
    policy = payload.get("resource_policy")
    if isinstance(policy, dict) and isinstance(policy.get("scheduler_blocks"), int):
        return int(policy["scheduler_blocks"])
    return 0


def _worker_blocks(payload: dict[str, Any]) -> int:
    value = payload.get("worker_blocks")
    if isinstance(value, int):
        return value
    policy = payload.get("resource_policy")
    if isinstance(policy, dict) and isinstance(policy.get("worker_blocks"), int):
        return int(policy["worker_blocks"])
    return 0


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    return default


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            result.append(item)
    return result


def _comma(values: list[int]) -> str:
    if not values:
        return "-"
    return ",".join(str(value) for value in values)


def _baseline_key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row["artifact"]),
        str(row["dag_shape"]),
        int(row["n"]),
        int(row["worker_blocks"]),
    )


def _baseline_times(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int, int], int]:
    baselines = {}
    for row in rows:
        if int(row["scheduler_blocks"]) != 1:
            continue
        device_ns = int(row["device_wall_ns"])
        if device_ns > 0:
            baselines[_baseline_key(row)] = device_ns
    return baselines


def _ratio(row: dict[str, Any], baselines: dict[tuple[str, str, int, int], int]) -> str:
    baseline = baselines.get(_baseline_key(row))
    device_ns = int(row["device_wall_ns"])
    if baseline is None or baseline == 0 or device_ns <= 0:
        return "-"
    return f"{device_ns / baseline:.2f}x"


def _percent(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def load_scaling_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        payload = json.loads(path.read_text())
        scheduler_blocks = _scheduler_blocks(payload)
        processed_by_block = _int_list(payload.get("scheduler_processed_by_block"))
        scheduler_processed_count = _as_int(payload.get("scheduler_processed_count"))
        active_scheduler_count = sum(1 for value in processed_by_block if value > 0)
        scheduler_utilization = active_scheduler_count / scheduler_blocks if scheduler_blocks > 0 else 0.0
        busiest_scheduler_share = (
            max(processed_by_block) / scheduler_processed_count
            if processed_by_block and scheduler_processed_count > 0
            else 0.0
        )
        rows.append(
            {
                "artifact": _artifact_label(path),
                "path": str(path),
                "status": str(payload.get("status", "unknown")),
                "runtime": str(payload.get("runtime", "unknown")),
                "mode": str(payload.get("mode", "unknown")),
                "dag_shape": str(payload.get("dag_shape", "-")),
                "n": _as_int(payload.get("n")),
                "device_wall_ns": _as_int(payload.get("device_wall_ns")),
                "host_wall_ns": _as_int(payload.get("host_wall_ns")),
                "scheduler_blocks": scheduler_blocks,
                "worker_blocks": _worker_blocks(payload),
                "scheduler_loop_count": _as_int(payload.get("scheduler_loop_count")),
                "scheduler_processed_count": scheduler_processed_count,
                "scheduler_processed_by_block": processed_by_block,
                "active_scheduler_count": active_scheduler_count,
                "nonzero_scheduler_blocks": active_scheduler_count,
                "scheduler_utilization": scheduler_utilization,
                "busiest_scheduler_share": busiest_scheduler_share,
                "launch_completed_counts": _int_list(payload.get("launch_completed_counts")),
            }
        )
    return sorted(rows, key=lambda row: (str(row["artifact"]), str(row["dag_shape"]), int(row["scheduler_blocks"])))


def render_markdown_report(rows: list[dict[str, Any]], label: str) -> str:
    baselines = _baseline_times(rows)
    lines = [
        "# CUDA Scheduler Scaling Report",
        "",
        f"- Label: `{label}`",
        (
            "- Ratio column compares device time with the same artifact, DAG shape, vector length, "
            "and worker-block policy at one scheduler block."
        ),
        "- Busiest column is the largest per-scheduler completion share.",
        "",
        (
            "| Artifact | DAG shape | Scheduler blocks | Device ns | Host ns | Processed by block | "
            "Active schedulers | Busiest | Vs sched=1 |"
        ),
        (
            "| -------- | --------- | ---------------- | --------- | ------- | ------------------ | "
            "----------------- | ------- | ---------- |"
        ),
    ]
    for row in rows:
        processed_by_block = _comma(row["scheduler_processed_by_block"])
        active = f"{row['active_scheduler_count']}/{row['scheduler_blocks']}"
        busiest = _percent(float(row["busiest_scheduler_share"]))
        lines.append(
            f"| {row['artifact']} | {row['dag_shape']} | {row['scheduler_blocks']} | {row['device_wall_ns']} | "
            f"{row['host_wall_ns']} | `{processed_by_block}` | `{active}` | `{busiest}` | "
            f"`{_ratio(row, baselines)}` |"
        )
    lines.append("")
    return "\n".join(lines)


def render_svg_report(rows: list[dict[str, Any]], label: str) -> str:
    width = 820
    left = 220
    right = 40
    top = 70
    row_height = 58
    chart_width = width - left - right
    max_value = max((int(row.get("device_wall_ns", 0) or 0) for row in rows), default=1)
    height = top + max(1, len(rows)) * row_height + 40
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="34" font-family="sans-serif" font-size="20" font-weight="600">{html.escape(label)}</text>',
        (
            '<text x="24" y="55" font-family="sans-serif" font-size="12" fill="#555">'
            "CUDA persistent-device scheduler-block scaling smoke summary</text>"
        ),
    ]
    for index, row in enumerate(rows):
        y = top + index * row_height
        value = int(row["device_wall_ns"])
        bar_width = int(chart_width * value / max_value) if max_value else 0
        name = f"{row['artifact']} {row['dag_shape']} sched={row['scheduler_blocks']}"
        by_block = _comma(row["scheduler_processed_by_block"])
        active = f"{row['active_scheduler_count']}/{row['scheduler_blocks']}"
        busiest = _percent(float(row["busiest_scheduler_share"]))
        lines.extend(
            [
                f'<text x="24" y="{y + 17}" font-family="sans-serif" font-size="12">{html.escape(name)}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_width}" height="24" fill="#1d4ed8"/>',
                (
                    f'<text x="{left + bar_width + 8}" y="{y + 17}" '
                    f'font-family="sans-serif" font-size="12">{value} ns</text>'
                ),
                (
                    f'<text x="{left}" y="{y + 42}" font-family="sans-serif" font-size="11" fill="#555">'
                    f"sched={row['scheduler_blocks']}; active={active}; busiest={busiest}; "
                    f"by_block={html.escape(by_block)}</text>"
                ),
            ]
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_scaling_report(rows: list[dict[str, Any]], output_dir: Path, label: str) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "cuda-scheduler-scaling.json"
    markdown_path = output_dir / "cuda-scheduler-scaling.md"
    svg_path = output_dir / "cuda-scheduler-scaling.svg"
    json_path.write_text(json.dumps({"label": label, "rows": rows}, indent=2) + "\n")
    markdown_path.write_text(render_markdown_report(rows, label))
    svg_path.write_text(render_svg_report(rows, label))
    return json_path, markdown_path, svg_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_paths", nargs="+", type=Path)
    parser.add_argument("--label", default="cuda-scheduler-scaling")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_scaling_rows(args.json_paths)
    json_path, markdown_path, svg_path = write_scaling_report(rows, args.output_dir, args.label)
    print(json_path)
    print(markdown_path)
    print(svg_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
