#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Render compact reports from CUDA smoke JSON outputs."""

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


def load_smoke_payloads(paths: list[Path]) -> list[dict[str, Any]]:
    payloads = []
    for path in paths:
        payload = json.loads(path.read_text())
        payload["_artifact"] = _artifact_label(path)
        payload["_path"] = str(path)
        payloads.append(payload)
    return payloads


def _shape(row: dict[str, Any]) -> str:
    tensor = row.get("tensor_tile")
    if not isinstance(tensor, dict):
        return "-"
    rows = tensor.get("rows")
    cols = tensor.get("cols")
    inner = tensor.get("inner")
    if rows is None or cols is None or inner is None:
        return "-"
    return f"{rows}x{cols}x{inner}"


def _tile_count(row: dict[str, Any]) -> str:
    tensor = row.get("tensor_tile")
    if isinstance(tensor, dict) and tensor.get("tile_count") is not None:
        return str(tensor["tile_count"])
    return "-"


def _mode(row: dict[str, Any]) -> str:
    mode = str(row.get("mode", "unknown"))
    dag_shape = row.get("dag_shape")
    if dag_shape:
        return f"{mode}/{dag_shape}"
    return mode


def _dispatch(row: dict[str, Any]) -> str:
    ids = row.get("dispatch_func_ids")
    if not isinstance(ids, list) or not ids:
        return "-"
    return ",".join(str(item) for item in ids)


def _scheduler_errors(row: dict[str, Any]) -> str:
    errors = row.get("device_scheduler_errors")
    if not isinstance(errors, dict):
        return "-"
    count = errors.get("count", 0)
    code = errors.get("code", 0)
    task_id = errors.get("task_id", 0)
    return f"count={count},code={code},task={task_id}"


def _repeat_runs(row: dict[str, Any]) -> str:
    repeat_runs = row.get("repeat_runs")
    if repeat_runs is None:
        return "-"
    return str(repeat_runs)


def _launch_completed_counts(row: dict[str, Any]) -> str:
    counts = row.get("launch_completed_counts")
    if not isinstance(counts, list):
        return "-"
    return ",".join(str(count) for count in counts)


def _lifecycle(row: dict[str, Any]) -> str:
    return f"repeat={_repeat_runs(row)},completed={_launch_completed_counts(row)}"


def _resource_policy(row: dict[str, Any]) -> str:
    policy = row.get("resource_policy")
    if not isinstance(policy, dict):
        return "-"
    return (
        f"sched={policy.get('scheduler_blocks', '-')},"
        f"workers={policy.get('worker_blocks', '-')},"
        f"wp={policy.get('worker_blocks_per_task', '-')},"
        f"stream={policy.get('stream_id', '-')},"
        f"block={policy.get('block_dim', '-')},"
        f"grid={policy.get('grid_dim', '-')}"
    )


def _scalar_args(row: dict[str, Any]) -> str:
    scalars = row.get("scalar_args")
    if not isinstance(scalars, dict) or not scalars:
        return "-"
    return ",".join(f"{key}={scalars[key]}" for key in sorted(scalars))


def _tensor_args(row: dict[str, Any]) -> str:
    tensors = row.get("tensor_args")
    if not isinstance(tensors, dict) or not tensors:
        return "-"
    return ",".join(f"{key}={tensors[key]}" for key in sorted(tensors))


def _graph_task_args(row: dict[str, Any]) -> str:
    task_args = row.get("graph_task_args")
    if not isinstance(task_args, dict) or not task_args:
        return "-"
    return ";".join(f"{key}={task_args[key]}" for key in sorted(task_args))


def _scratch_reuse(row: dict[str, Any]) -> str:
    scratch_reuse = row.get("scratch_reuse")
    if not isinstance(scratch_reuse, dict) or not scratch_reuse:
        return "-"
    keys = [key for key in ("reused_buffer", "reuse_task") if key in scratch_reuse]
    keys.extend(key for key in sorted(scratch_reuse) if key not in keys)
    return ",".join(f"{key}={scratch_reuse[key]}" for key in keys)


def _tensor_core(row: dict[str, Any]) -> str:
    tensor_core = row.get("tensor_core")
    if not isinstance(tensor_core, dict):
        return "-"
    api = tensor_core.get("api")
    mma_shape = tensor_core.get("mma_shape")
    input_type = tensor_core.get("input")
    accumulator = tensor_core.get("accumulator")
    if not all((api, mma_shape, input_type, accumulator)):
        return "-"
    return f"{api}:{mma_shape}:{input_type}->{accumulator}"


def render_markdown_report(payloads: list[dict[str, Any]], label: str) -> str:
    lines = [
        "# CUDA Smoke Report",
        "",
        f"- Label: `{label}`",
        "",
        (
            "| Artifact | Status | Runtime | Mode | N | PTX arch | Device ns | "
            "Host ns | Tensor shape | Tiles | Tensor core | Dispatch | "
            "Scheduler errors | Repeat runs | Launch completions | Resource policy | "
            "Scalar args | Tensor args | Scratch reuse | Graph task args |"
        ),
        (
            "| -------- | ------ | ------- | ---- | - | -------- | --------- | "
            "------- | ------------ | ----- | ----------- | -------- | "
            "---------------- | ----------- | ------------------ | --------------- | "
            "----------- | ----------- | ------------- | --------------- |"
        ),
    ]
    for row in payloads:
        lines.append(
            f"| {row.get('_artifact', 'unknown')} | {row.get('status', 'unknown')} | "
            f"{row.get('runtime', 'unknown')} | {_mode(row)} | {row.get('n', '-')} | "
            f"`{row.get('ptx_arch', 'unknown')}` | {row.get('device_wall_ns', '-')} | "
            f"{row.get('host_wall_ns', '-')} | {_shape(row)} | {_tile_count(row)} | "
            f"`{_tensor_core(row)}` | `{_dispatch(row)}` | `{_scheduler_errors(row)}` | "
            f"`{_repeat_runs(row)}` | `{_launch_completed_counts(row)}` | "
            f"`{_resource_policy(row)}` | "
            f"`{_scalar_args(row)}` | `{_tensor_args(row)}` | "
            f"`{_scratch_reuse(row)}` | `{_graph_task_args(row)}` |"
        )

    lines.extend(["", "## PTX Sources", ""])
    for row in payloads:
        lines.append(f"- `{row.get('_artifact', 'unknown')}`: `{row.get('ptx_source', 'unknown')}`")
    lines.append("")
    return "\n".join(lines)


def render_svg_report(payloads: list[dict[str, Any]], label: str) -> str:
    width = 760
    bar_height = 28
    row_gap = 104
    left = 170
    right = 40
    top = 70
    chart_width = width - left - right
    max_value = max((int(row.get("device_wall_ns", 0) or 0) for row in payloads), default=1)
    height = top + len(payloads) * (bar_height + row_gap) + 40
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="34" font-family="sans-serif" font-size="20" font-weight="600">{html.escape(label)}</text>',
        (
            '<text x="24" y="55" font-family="sans-serif" font-size="12" fill="#555">'
            "Device time from CUDA smoke JSON</text>"
        ),
    ]
    for index, row in enumerate(payloads):
        y = top + index * (bar_height + row_gap)
        value = int(row.get("device_wall_ns", 0) or 0)
        bar_width = int(chart_width * value / max_value) if max_value else 0
        name = str(row.get("_artifact", "unknown"))
        scheduler_errors = _scheduler_errors(row)
        lifecycle = _lifecycle(row)
        resource_policy = _resource_policy(row)
        scalar_args = _scalar_args(row)
        tensor_args = _tensor_args(row)
        scratch_reuse = _scratch_reuse(row)
        graph_task_args = _graph_task_args(row)
        lines.extend(
            [
                f'<text x="24" y="{y + 19}" font-family="sans-serif" font-size="13">{html.escape(name)}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_width}" height="{bar_height}" fill="#2563eb"/>',
                (
                    f'<text x="{left + bar_width + 8}" y="{y + 19}" '
                    f'font-family="sans-serif" font-size="13">{value} ns</text>'
                ),
                (
                    f'<text x="{left}" y="{y + bar_height + 14}" '
                    'font-family="sans-serif" font-size="11" fill="#555">'
                    f"errors: {html.escape(scheduler_errors)}</text>"
                ),
                (
                    f'<text x="{left}" y="{y + bar_height + 28}" '
                    'font-family="sans-serif" font-size="11" fill="#555">'
                    f"lifecycle: {html.escape(lifecycle)}</text>"
                ),
                (
                    f'<text x="{left}" y="{y + bar_height + 42}" '
                    'font-family="sans-serif" font-size="11" fill="#555">'
                    f"policy: {html.escape(resource_policy)}</text>"
                ),
                (
                    f'<text x="{left}" y="{y + bar_height + 56}" '
                    'font-family="sans-serif" font-size="11" fill="#555">'
                    f"scalars: {html.escape(scalar_args)}</text>"
                ),
                (
                    f'<text x="{left}" y="{y + bar_height + 70}" '
                    'font-family="sans-serif" font-size="11" fill="#555">'
                    f"tensors: {html.escape(tensor_args)}</text>"
                ),
                (
                    f'<text x="{left}" y="{y + bar_height + 84}" '
                    'font-family="sans-serif" font-size="11" fill="#555">'
                    f"scratch: {html.escape(scratch_reuse)}</text>"
                ),
                (
                    f'<text x="{left}" y="{y + bar_height + 98}" '
                    'font-family="sans-serif" font-size="11" fill="#555">'
                    f"task args: {html.escape(graph_task_args)}</text>"
                ),
            ]
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_report(paths: list[Path], output_dir: Path, label: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = load_smoke_payloads(paths)
    markdown_path = output_dir / "cuda-smoke-report.md"
    svg_path = output_dir / "cuda-smoke-report.svg"
    markdown_path.write_text(render_markdown_report(payloads, label))
    svg_path.write_text(render_svg_report(payloads, label))
    return markdown_path, svg_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--label", default="cuda-smoke")
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/cuda-backend/smoke-report"))
    args = parser.parse_args()
    markdown_path, svg_path = write_report(args.inputs, args.output_dir, args.label)
    print(markdown_path)
    print(svg_path)


if __name__ == "__main__":
    main()
