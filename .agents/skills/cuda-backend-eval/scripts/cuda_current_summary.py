#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Render compact CUDA current-evaluation summary tables from benchmark JSON."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cuda_benchmark import summarize_results

SummaryKey = tuple[str, str, int, int, int]
Summary = Mapping[SummaryKey, Mapping[str, Any]]
Payload = dict[str, Any]

MACHINE_LABELS = {
    "hina": "A100",
    "dasys-h200x8": "H200",
}

LAUNCH_BASELINES = (
    "pto_host_schedule",
    "pto_host_schedule_compiler",
    "direct_driver",
    "direct_driver_graph",
)

DAG_BASELINES = (
    "pto_persistent_dag",
    "pto_persistent_dag_chain",
    "pto_persistent_dag_reuse",
    "pto_persistent_dag_scalar_axpy",
    "pto_persistent_dag_scalar_affine",
    "pto_persistent_dag_triad",
    "pto_persistent_dag_quad",
    "pto_persistent_dag_unary_square",
    "pto_persistent_dag_tensor",
)


def _machine_label(machine: str) -> str:
    return MACHINE_LABELS.get(machine, machine)


def _machines(summary: Summary) -> list[str]:
    return sorted({key[0] for key in summary}, key=lambda machine: (_machine_label(machine), machine))


def _sizes_for_baselines(summary: Summary, machine: str, baselines: Sequence[str]) -> list[int]:
    sizes = {
        key[2]
        for key in summary
        if key[0] == machine
        and key[1] in baselines
        and all((machine, baseline, key[2], 1, 1) in summary for baseline in baselines)
    }
    return sorted(sizes)


def _dag_sizes(summary: Summary, machine: str) -> list[int]:
    sizes = {
        key[2]
        for key in summary
        if key[0] == machine
        and (machine, "pto_persistent_dag", key[2], 3, 1) in summary
        and (machine, "pto_persistent_dag_chain", key[2], 5, 1) in summary
        and (machine, "pto_persistent_dag_reuse", key[2], 6, 1) in summary
        and (machine, "pto_persistent_dag_scalar_axpy", key[2], 3, 1) in summary
        and (machine, "pto_persistent_dag_tensor", key[2], 4, 1) in summary
    }
    return sorted(sizes)


def _median(summary: Summary, key: SummaryKey) -> int:
    return int(summary[key]["median_device_wall_ns"])


def _ratio(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.2f}x"


def _ratio_for_key(summary: Summary, key: SummaryKey, denominator: int) -> str:
    if key not in summary:
        return "-"
    return _ratio(_median(summary, key), denominator)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str | int]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" if idx == 0 else "-" for idx, _ in enumerate(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def render_launch_table(payload: Payload) -> str:
    summary = summarize_results(payload)
    rows: list[list[str | int]] = []
    for machine in _machines(summary):
        for n in _sizes_for_baselines(summary, machine, LAUNCH_BASELINES):
            host = _median(summary, (machine, "pto_host_schedule", n, 1, 1))
            compiler = _median(summary, (machine, "pto_host_schedule_compiler", n, 1, 1))
            driver = _median(summary, (machine, "direct_driver", n, 1, 1))
            graph = _median(summary, (machine, "direct_driver_graph", n, 1, 1))
            rows.append(
                [
                    _machine_label(machine),
                    n,
                    host,
                    compiler,
                    driver,
                    graph,
                    _ratio(compiler, host),
                    _ratio(graph, host),
                ]
            )
    return _table(
        ["GPU", "N", "PTO host ns", "Compiler ns", "Driver ns", "Graph ns", "Compiler/PTO", "Graph/PTO"],
        rows,
    )


def render_unary_square_table(payload: Payload) -> str:
    summary = summarize_results(payload)
    rows: list[list[str | int]] = []
    for machine in _machines(summary):
        sizes = {
            key[2]
            for key in summary
            if key[0] == machine and key[1] in {"pto_host_schedule_unary_square", "pto_host_schedule_quad"}
        }
        for n in sorted(sizes):
            rows.append(
                [
                    _machine_label(machine),
                    n,
                    _median(summary, (machine, "pto_host_schedule_unary_square", n, 1, 1))
                    if (machine, "pto_host_schedule_unary_square", n, 1, 1) in summary
                    else "-",
                    _median(summary, (machine, "pto_host_schedule_quad", n, 1, 1))
                    if (machine, "pto_host_schedule_quad", n, 1, 1) in summary
                    else "-",
                ]
            )
    return _table(["GPU", "N", "Unary square ns", "Quad ns"], rows)


def render_worker_grid_table(payload: Payload) -> str:
    summary = summarize_results(payload)
    rows: list[list[str | int]] = []
    for machine in _machines(summary):
        host_batch_keys = [key for key in summary if key[0] == machine and key[1] == "pto_host_schedule_batch"]
        for _, _, n, tasks, _ in sorted(host_batch_keys, key=lambda key: (key[2], key[3])):
            candidates = [
                key
                for key in summary
                if key[0] == machine
                and key[1] == "pto_persistent_device_grid_batch"
                and key[2] == n
                and key[3] == tasks
            ]
            if not candidates:
                continue
            best = min(candidates, key=lambda key: _median(summary, key))
            best_ns = _median(summary, best)
            host_ns = _median(summary, (machine, "pto_host_schedule_batch", n, tasks, 1))
            rows.append([_machine_label(machine), n, tasks, best[4], best_ns, _ratio(best_ns, host_ns)])
    return _table(
        ["GPU", "N", "Tasks", "Best worker blocks/task", "Device ns", "Vs host batch"],
        rows,
    )


def render_dag_shape_table(payload: Payload) -> str:
    summary = summarize_results(payload)
    rows: list[list[str | int]] = []
    for machine in _machines(summary):
        for n in _dag_sizes(summary, machine):
            dag = _median(summary, (machine, "pto_persistent_dag", n, 3, 1))
            chain = _median(summary, (machine, "pto_persistent_dag_chain", n, 5, 1))
            reuse = _median(summary, (machine, "pto_persistent_dag_reuse", n, 6, 1))
            scalar = _median(summary, (machine, "pto_persistent_dag_scalar_axpy", n, 3, 1))
            tensor = _median(summary, (machine, "pto_persistent_dag_tensor", n, 4, 1))
            rows.append(
                [
                    _machine_label(machine),
                    n,
                    _ratio(chain, dag),
                    _ratio(reuse, dag),
                    _ratio(scalar, dag),
                    _ratio_for_key(summary, (machine, "pto_persistent_dag_scalar_affine", n, 3, 1), dag),
                    _ratio_for_key(summary, (machine, "pto_persistent_dag_triad", n, 3, 1), dag),
                    _ratio_for_key(summary, (machine, "pto_persistent_dag_quad", n, 3, 1), dag),
                    _ratio_for_key(summary, (machine, "pto_persistent_dag_unary_square", n, 3, 1), dag),
                    _ratio(tensor, dag),
                ]
            )
    return _table(
        [
            "GPU",
            "N",
            "Chain/DAG",
            "Reuse/DAG",
            "Scalar AXPY/DAG",
            "Scalar Affine/DAG",
            "Triad/DAG",
            "Quad/DAG",
            "Unary Square/DAG",
            "Tensor/DAG",
        ],
        rows,
    )


def render_summary(payload: Payload) -> str:
    return "\n\n".join(
        [
            "## Launch Baselines",
            render_launch_table(payload),
            "## Host-Schedule Shape Rows",
            render_unary_square_table(payload),
            "## Worker Grid Rows",
            render_worker_grid_table(payload),
            "## Persistent DAG Shapes",
            render_dag_shape_table(payload),
        ]
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument(
        "--section",
        choices=("all", "launch", "unary-square", "worker-grid", "dag-shapes"),
        default="all",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = json.loads(args.json_path.read_text())
    if args.section == "launch":
        print(render_launch_table(payload))
    elif args.section == "unary-square":
        print(render_unary_square_table(payload))
    elif args.section == "worker-grid":
        print(render_worker_grid_table(payload))
    elif args.section == "dag-shapes":
        print(render_dag_shape_table(payload))
    else:
        print(render_summary(payload))


if __name__ == "__main__":
    main()
