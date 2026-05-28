#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Render compact CUDA current-evaluation summary tables from raw JSON."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cuda_benchmark import summarize_results

SummaryKey = tuple[str, str, int, int, int]
Summary = Mapping[SummaryKey, Mapping[str, Any]]
Payload = dict[str, Any]

MACHINE_LABELS = {
    "a100": "A100",
    "hina": "A100",
    "h200": "H200",
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
    "pto_persistent_dag_scalar_scale",
    "pto_persistent_dag_scalar_affine",
    "pto_persistent_dag_triad",
    "pto_persistent_dag_quad",
    "pto_persistent_dag_generic_args",
    "pto_persistent_dag_graph",
    "pto_persistent_dag_graph_diamond",
    "pto_persistent_dag_unary_square",
    "pto_persistent_dag_tensor",
)
BENCHMARK_TENSOR_BASELINES = (
    "pto_persistent_dag_tensor",
    "pto_persistent_dag_graph_tensor",
    "pto_persistent_dag_tensor_core",
    "cublas_sgemm",
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


def _ratio(numerator: int | float, denominator: int | float) -> str:
    return f"{numerator / denominator:.2f}x"


def _ratio_for_key(summary: Summary, key: SummaryKey, denominator: int) -> str:
    if key not in summary:
        return "-"
    return _ratio(_median(summary, key), denominator)


def _format_number(value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _format_gflops(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str | int]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("-" * max(1, len(header)) for header in headers) + " |",
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
                    _ratio_for_key(summary, (machine, "pto_persistent_dag_scalar_scale", n, 3, 1), dag),
                    _ratio_for_key(summary, (machine, "pto_persistent_dag_scalar_affine", n, 3, 1), dag),
                    _ratio_for_key(summary, (machine, "pto_persistent_dag_triad", n, 3, 1), dag),
                    _ratio_for_key(summary, (machine, "pto_persistent_dag_quad", n, 3, 1), dag),
                    _ratio_for_key(summary, (machine, "pto_persistent_dag_generic_args", n, 3, 1), dag),
                    _ratio_for_key(summary, (machine, "pto_persistent_dag_graph", n, 3, 1), dag),
                    _ratio_for_key(summary, (machine, "pto_persistent_dag_graph_diamond", n, 5, 1), dag),
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
            "Scalar Scale/DAG",
            "Scalar Affine/DAG",
            "Triad/DAG",
            "Quad/DAG",
            "Generic Args/DAG",
            "Graph Descriptor/DAG",
            "Graph Diamond/DAG",
            "Unary Square/DAG",
            "Tensor/DAG",
        ],
        rows,
    )


def _tensor_row_n(row: Mapping[str, Any], payload: Payload) -> str:
    n = row.get("n", payload.get("metadata", {}).get("n", "-"))
    return str(n)


def _tensor_tile_shape(row: Mapping[str, Any]) -> str:
    tile = row.get("tensor_tile")
    if not isinstance(tile, Mapping):
        return str(row.get("shape", "-"))
    rows = tile.get("rows")
    cols = tile.get("cols")
    inner = tile.get("inner")
    if rows is None or cols is None or inner is None:
        return "-"
    return f"{rows}x{cols}x{inner}"


def _tensor_flops(n: str, shape: str) -> int | None:
    parts = shape.split("x")
    if len(parts) != 3:
        return None
    try:
        _, _, inner = (int(part) for part in parts)
        output_elements = int(n)
    except ValueError:
        return None
    return 2 * output_elements * inner


def _tensor_gflops(
    n: str,
    shape: str,
    device_wall_ns: int | float | None,
) -> float | None:
    flops = _tensor_flops(n, shape)
    if flops is None or not device_wall_ns:
        return None
    return float(flops) / float(device_wall_ns)


def _tensor_sweep_medians(payload: Payload) -> dict[tuple[str, str, str, str], int | float]:
    groups: dict[tuple[str, str, str, str], list[int]] = {}
    for row in payload.get("results", []):
        if row.get("status", "pass") != "pass":
            continue
        machine = str(row.get("machine") or row.get("artifact", "unknown"))
        baseline = str(row.get("baseline", "unknown"))
        n = _tensor_row_n(row, payload)
        shape = str(row.get("shape", "unknown"))
        groups.setdefault((machine, baseline, n, shape), []).append(int(row.get("device_wall_ns", 0)))
    return {key: statistics.median(values) for key, values in groups.items()}


def _benchmark_tensor_medians(payload: Payload) -> dict[tuple[str, str, str, str], int | float]:
    groups: dict[tuple[str, str, str, str], list[int]] = {}
    for row in payload.get("results", []):
        if row.get("status", "pass") != "pass":
            continue
        baseline = str(row.get("baseline", "unknown"))
        if baseline not in BENCHMARK_TENSOR_BASELINES:
            continue
        machine = str(row.get("machine") or row.get("artifact", "unknown"))
        n = _tensor_row_n(row, payload)
        shape = _tensor_tile_shape(row)
        if shape == "-":
            continue
        groups.setdefault((machine, baseline, n, shape), []).append(int(row.get("device_wall_ns", 0)))
    return {key: statistics.median(values) for key, values in groups.items()}


def render_benchmark_tensor_throughput_table(payload: Payload) -> str:
    medians = _benchmark_tensor_medians(payload)
    rows: list[list[str | int]] = []
    machine_sizes_shapes = sorted(
        {(machine, n, shape) for machine, _, n, shape in medians},
        key=lambda item: (_machine_label(item[0]), int(item[1]) if item[1].isdigit() else item[1], item[2], item[0]),
    )
    for machine, n, shape in machine_sizes_shapes:
        scalar_key = (machine, "pto_persistent_dag_tensor", n, shape)
        graph_tensor_key = (machine, "pto_persistent_dag_graph_tensor", n, shape)
        tensor_core_key = (machine, "pto_persistent_dag_tensor_core", n, shape)
        cublas_key = (machine, "cublas_sgemm", n, shape)
        if scalar_key not in medians:
            continue
        scalar = medians[scalar_key]
        graph_tensor = medians.get(graph_tensor_key)
        tensor_core = medians.get(tensor_core_key)
        cublas = medians.get(cublas_key)
        rows.append(
            [
                _machine_label(machine),
                n,
                shape,
                _format_number(scalar),
                _format_number(graph_tensor) if graph_tensor is not None else "-",
                _format_number(tensor_core) if tensor_core is not None else "-",
                _format_number(cublas) if cublas is not None else "-",
                _format_gflops(_tensor_gflops(n, shape, scalar)),
                _format_gflops(_tensor_gflops(n, shape, graph_tensor)),
                _format_gflops(_tensor_gflops(n, shape, tensor_core)),
                _format_gflops(_tensor_gflops(n, shape, cublas)),
                _ratio(tensor_core, scalar) if tensor_core is not None else "-",
                _ratio(cublas, scalar) if cublas is not None else "-",
            ]
        )
    return _table(
        [
            "GPU",
            "N",
            "Shape",
            "Scalar ns",
            "Graph ns",
            "Tensor-core ns",
            "cuBLAS ns",
            "Scalar GF/s",
            "Graph GF/s",
            "Tensor-core GF/s",
            "cuBLAS GF/s",
            "Tensor-core/scalar",
            "cuBLAS/scalar",
        ],
        rows,
    )


def render_tensor_sweep_table(payload: Payload) -> str:
    medians = _tensor_sweep_medians(payload)
    rows: list[list[str | int]] = []
    machine_sizes_shapes = sorted(
        {(machine, n, shape) for machine, _, n, shape in medians},
        key=lambda item: (_machine_label(item[0]), int(item[1]) if item[1].isdigit() else item[1], item[2], item[0]),
    )
    for machine, n, shape in machine_sizes_shapes:
        scalar_key = (machine, "pto_persistent_dag_tensor", n, shape)
        graph_tensor_key = (machine, "pto_persistent_dag_graph_tensor", n, shape)
        tensor_core_key = (machine, "pto_persistent_dag_tensor_core", n, shape)
        cublas_key = (machine, "cublas_sgemm", n, shape)
        if scalar_key not in medians:
            continue
        scalar = medians[scalar_key]
        graph_tensor = medians.get(graph_tensor_key)
        tensor_core = medians.get(tensor_core_key)
        cublas = medians.get(cublas_key)
        scalar_gflops = _tensor_gflops(n, shape, scalar)
        graph_tensor_gflops = _tensor_gflops(n, shape, graph_tensor)
        tensor_core_gflops = _tensor_gflops(n, shape, tensor_core)
        cublas_gflops = _tensor_gflops(n, shape, cublas)
        rows.append(
            [
                _machine_label(machine),
                n,
                shape,
                _format_number(scalar),
                _format_number(graph_tensor) if graph_tensor is not None else "-",
                _format_number(tensor_core) if tensor_core is not None else "-",
                _format_number(cublas) if cublas is not None else "-",
                _format_gflops(scalar_gflops),
                _format_gflops(graph_tensor_gflops),
                _format_gflops(tensor_core_gflops),
                _format_gflops(cublas_gflops),
                _ratio(graph_tensor, scalar) if graph_tensor is not None else "-",
                _ratio(tensor_core, scalar) if tensor_core is not None else "-",
                _ratio(cublas, scalar) if cublas is not None else "-",
            ]
        )
    return _table(
        [
            "GPU",
            "N",
            "Shape",
            "Scalar tensor ns",
            "Graph tensor ns",
            "Tensor-core ns",
            "cuBLAS ns",
            "Scalar GF/s",
            "Graph tensor GF/s",
            "Tensor-core GF/s",
            "cuBLAS GF/s",
            "Graph/scalar",
            "Tensor-core/scalar",
            "cuBLAS/scalar",
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
            "## Selected Tensor Throughput",
            render_benchmark_tensor_throughput_table(payload),
        ]
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument(
        "--section",
        choices=(
            "all",
            "launch",
            "unary-square",
            "worker-grid",
            "dag-shapes",
            "tensor-throughput",
            "tensor-sweep",
        ),
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
    elif args.section == "tensor-throughput":
        print(render_benchmark_tensor_throughput_table(payload))
    elif args.section == "tensor-sweep":
        print(render_tensor_sweep_table(payload))
    else:
        print(render_summary(payload))


if __name__ == "__main__":
    main()
