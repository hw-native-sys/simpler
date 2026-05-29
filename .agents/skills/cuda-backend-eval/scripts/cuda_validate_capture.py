#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate CUDA benchmark capture artifacts before publishing summaries."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

PAIRED_CURRENT_MACHINES = ("hina", "dasys-h200x8")
PAIRED_CURRENT_BASELINES = (
    "cublas_sgemm",
    "cublas_sgemm_graph",
    "direct_driver",
    "direct_driver_graph",
    "pto_host_schedule",
    "pto_host_schedule_batch",
    "pto_host_schedule_compiler",
    "pto_host_schedule_generic_args",
    "pto_host_schedule_quad",
    "pto_host_schedule_unary_square",
    "pto_persistent_dag",
    "pto_persistent_dag_chain",
    "pto_persistent_dag_reuse",
    "pto_persistent_dag_scalar_affine",
    "pto_persistent_dag_scalar_axpy",
    "pto_persistent_dag_scalar_scale",
    "pto_persistent_dag_tensor",
    "pto_persistent_dag_tensor_core",
    "pto_persistent_dag_triad",
    "pto_persistent_dag_quad",
    "pto_persistent_dag_generic_args",
    "pto_persistent_dag_graph",
    "pto_persistent_dag_graph_generic_args4",
    "pto_persistent_dag_graph_chain",
    "pto_persistent_dag_graph_scratch_reuse",
    "pto_persistent_dag_graph_diamond",
    "pto_persistent_dag_graph_tagged",
    "pto_persistent_dag_graph_tagged_inout",
    "pto_persistent_dag_graph_role_keyed_inout",
    "pto_persistent_dag_graph_triad",
    "pto_persistent_dag_graph_quad",
    "pto_persistent_dag_graph_unary_square",
    "pto_persistent_dag_graph_tensor",
    "pto_persistent_dag_graph_tensor_core",
    "pto_persistent_dag_unary_square",
    "pto_persistent_device",
    "pto_persistent_device_batch",
    "pto_persistent_device_grid_batch",
    "pto_persistent_queue",
    "pto_persistent_queue_batch",
)
PAIRED_CURRENT_SIZES = (1024, 65536, 1048576)
COMPACT_CURRENT_SIZES = (1024,)
COMPACT_CURRENT_EXPECTED_REPEATS = 1
COMPACT_CURRENT_EXPECTED_RESULT_COUNT = 80
PAIRED_CURRENT_EXPECTED_RESULT_COUNT = 1026
REQUIRED_SOURCE_PAPER_IDS = ("arXiv:2605.03190", "arXiv:2512.22219v1")
REPORT_FILES = (
    "cuda-benchmark.md",
    "cuda-benchmark.svg",
    "cuda-benchmark-ratios.svg",
    "cuda-benchmark-dag-deltas.svg",
    "cuda-benchmark-throughput.svg",
)
PAIRED_CURRENT_DISPATCH = {
    "pto_persistent_dag": "1,2,1",
    "pto_persistent_dag_chain": "1,2,1,2,1",
    "pto_persistent_dag_reuse": "1,2,1,2,1,1",
    "pto_persistent_dag_scalar_axpy": "4,2,1",
    "pto_persistent_dag_scalar_scale": "11,2,1",
    "pto_persistent_dag_scalar_affine": "5,2,1",
    "pto_persistent_dag_triad": "6,2,1",
    "pto_persistent_dag_quad": "8,2,1",
    "pto_persistent_dag_generic_args": "9,2,1",
    "pto_persistent_dag_graph": "9,2,1",
    "pto_persistent_dag_graph_generic_args4": "9,2,1",
    "pto_persistent_dag_graph_chain": "1,2,1,2,1",
    "pto_persistent_dag_graph_scratch_reuse": "1,2,1,2,1,1",
    "pto_persistent_dag_graph_diamond": "9,2,1,2,1",
    "pto_persistent_dag_graph_tagged": "9,2,1",
    "pto_persistent_dag_graph_tagged_inout": "1,1,1",
    "pto_persistent_dag_graph_role_keyed_inout": "1,1,1",
    "pto_persistent_dag_graph_triad": "6,2,1",
    "pto_persistent_dag_graph_quad": "8,2,1",
    "pto_persistent_dag_graph_unary_square": "7,1,1",
    "pto_persistent_dag_unary_square": "7,1,1",
    "pto_persistent_dag_tensor": "3,1,2,1",
    "pto_persistent_dag_graph_tensor": "3,1,2,1",
    "pto_persistent_dag_tensor_core": "10,1,2,1",
    "pto_persistent_dag_graph_tensor_core": "10,1,2,1",
}
PAIRED_CURRENT_TENSOR_TILES = {
    "pto_persistent_dag_tensor": "16x16x16",
    "pto_persistent_dag_graph_tensor": "16x16x16",
    "pto_persistent_dag_tensor_core": "16x16x16",
    "pto_persistent_dag_graph_tensor_core": "16x16x16",
    "cublas_sgemm": "16x16x16",
    "cublas_sgemm_graph": "16x16x16",
}
PAIRED_CURRENT_SCRATCH_REUSE = {
    "pto_persistent_dag_graph_scratch_reuse": "reused_buffer=tmp0,reuse_task=4",
}
PAIRED_CURRENT_GRAPH_TASK_ARGS = {
    "pto_persistent_dag_graph_tagged": (
        "task0=input:a,input:b,output:tmp1,scalar:scalar_args[0],scalar:scalar_args[1];"
        "task1=input:a,input:b,output:tmp2;task2=input:tmp1,input:tmp2,output_existing:out"
    ),
    "pto_persistent_dag_graph_tagged_inout": (
        "task0=input:a,input:b,output:tmp1;task1=inout:tmp1,input:b;task2=input:tmp1,input:a,output_existing:out"
    ),
    "pto_persistent_dag_graph_role_keyed_inout": (
        "task0=input:a,input:b,output:tmp1;task1=inout:tmp1,input:b;task2=input:tmp1,input:a,output_existing:out"
    ),
}
PAIRED_CURRENT_GRAPH_TASK_ARG_KEYS = {
    "pto_persistent_dag_graph_role_keyed_inout": "role",
}
PAIRED_CURRENT_GRAPH_FANIN = {
    "pto_persistent_dag_graph": "0,0,2",
    "pto_persistent_dag_graph_generic_args4": "0,0,2",
    "pto_persistent_dag_graph_chain": "0,0,2,1,1",
    "pto_persistent_dag_graph_scratch_reuse": "0,0,2,1,1,2",
    "pto_persistent_dag_graph_diamond": "0,0,2,2,2",
    "pto_persistent_dag_graph_tagged": "0,0,2",
    "pto_persistent_dag_graph_tagged_inout": "0,1,1",
    "pto_persistent_dag_graph_role_keyed_inout": "0,1,1",
    "pto_persistent_dag_graph_triad": "0,0,2",
    "pto_persistent_dag_graph_quad": "0,0,2",
    "pto_persistent_dag_graph_unary_square": "0,1,1",
    "pto_persistent_dag_graph_tensor": "0,1,1,2",
    "pto_persistent_dag_graph_tensor_core": "0,1,1,2",
}
PAIRED_CURRENT_GRAPH_DEPENDENTS = {
    "pto_persistent_dag_graph": "2,2",
    "pto_persistent_dag_graph_generic_args4": "2,2",
    "pto_persistent_dag_graph_chain": "2,2,3,4",
    "pto_persistent_dag_graph_scratch_reuse": "2,2,3,4,5,5",
    "pto_persistent_dag_graph_diamond": "2,3,2,3,4,4",
    "pto_persistent_dag_graph_tagged": "2,2",
    "pto_persistent_dag_graph_tagged_inout": "1,2",
    "pto_persistent_dag_graph_role_keyed_inout": "1,2",
    "pto_persistent_dag_graph_triad": "2,2",
    "pto_persistent_dag_graph_quad": "2,2",
    "pto_persistent_dag_graph_unary_square": "1,2",
    "pto_persistent_dag_graph_tensor": "1,2,3,3",
    "pto_persistent_dag_graph_tensor_core": "1,2,3,3",
}


def _as_list(values: Sequence[str] | None) -> list[str]:
    if values is None:
        return []
    result: list[str] = []
    for value in values:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    return result


def _as_int_list(values: Sequence[str] | None) -> list[int]:
    return [int(value) for value in _as_list(values)]


def _results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("results")
    if not isinstance(results, list):
        return []
    return [row for row in results if isinstance(row, dict)]


def _row_matches(row: dict[str, Any], *, machine: str, baseline: str, size: int | None = None) -> bool:
    if row.get("machine") != machine or row.get("baseline") != baseline:
        return False
    return size is None or row.get("n") == size


def _unique_repeats(rows: Iterable[dict[str, Any]]) -> set[Any]:
    return {row.get("repeat") for row in rows if "repeat" in row}


def _dispatch_text(row: dict[str, Any]) -> str:
    dispatch = row.get("dispatch_func_ids")
    if not isinstance(dispatch, list):
        return "-"
    return ",".join(str(value) for value in dispatch)


def _tensor_tile_shape(row: dict[str, Any]) -> str:
    tensor_tile = row.get("tensor_tile")
    if not isinstance(tensor_tile, dict):
        return "-"
    rows = tensor_tile.get("rows")
    cols = tensor_tile.get("cols")
    inner = tensor_tile.get("inner")
    if rows is None or cols is None or inner is None:
        return "-"
    return f"{rows}x{cols}x{inner}"


def _scratch_reuse_text(row: dict[str, Any]) -> str:
    scratch_reuse = row.get("scratch_reuse")
    if not isinstance(scratch_reuse, dict):
        return "-"
    keys = [key for key in ("reused_buffer", "reuse_task") if key in scratch_reuse]
    keys.extend(key for key in sorted(scratch_reuse) if key not in keys)
    return ",".join(f"{key}={scratch_reuse[key]}" for key in keys)


def _graph_task_args_text(row: dict[str, Any]) -> str:
    task_args = row.get("graph_task_args")
    if not isinstance(task_args, dict):
        return "-"
    return ";".join(f"{key}={task_args[key]}" for key in sorted(task_args))


def _graph_task_arg_key_text(row: dict[str, Any]) -> str:
    key = row.get("graph_task_arg_key")
    return str(key) if key else "-"


def _graph_descriptor_text(row: dict[str, Any], field_name: str) -> str:
    graph_descriptor = row.get("graph_descriptor")
    if not isinstance(graph_descriptor, dict):
        return "-"
    values = graph_descriptor.get(field_name)
    if not isinstance(values, list):
        return "-"
    return ",".join(str(value) for value in values)


def _validate_required_machines(rows: list[dict[str, Any]], required_machines: Sequence[str]) -> list[str]:
    machines = {str(row.get("machine")) for row in rows if row.get("machine") is not None}
    return [f"missing machine {machine}" for machine in required_machines if machine not in machines]


def _validate_status(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        if row.get("status", "pass") == "pass":
            continue
        baseline = row.get("baseline", "unknown")
        machine = row.get("machine", "unknown")
        size = row.get("n", "unknown")
        errors.append(f"non-pass row machine={machine} baseline={baseline} n={size}")
    return errors


def _validate_required_rows(
    rows: list[dict[str, Any]],
    *,
    required_machines: Sequence[str],
    required_baselines: Sequence[str],
    required_sizes: Sequence[int],
    expected_repeats: int | None,
) -> list[str]:
    errors: list[str] = []
    for machine in required_machines:
        for baseline in required_baselines:
            baseline_rows = [row for row in rows if _row_matches(row, machine=machine, baseline=baseline)]
            if not baseline_rows:
                errors.append(f"missing baseline {baseline} on {machine}")
                continue
            errors.extend(
                _validate_required_sizes(
                    baseline_rows,
                    machine=machine,
                    baseline=baseline,
                    required_sizes=required_sizes,
                    expected_repeats=expected_repeats,
                )
            )
    return errors


def _validate_required_sizes(
    baseline_rows: list[dict[str, Any]],
    *,
    machine: str,
    baseline: str,
    required_sizes: Sequence[int],
    expected_repeats: int | None,
) -> list[str]:
    errors: list[str] = []
    for size in required_sizes:
        size_rows = [row for row in baseline_rows if _row_matches(row, machine=machine, baseline=baseline, size=size)]
        if not size_rows:
            errors.append(f"missing baseline {baseline} on {machine} for n={size}")
        elif expected_repeats is not None and len(_unique_repeats(size_rows)) != expected_repeats:
            found = len(_unique_repeats(size_rows))
            errors.append(f"expected {expected_repeats} repeats for {baseline} on {machine} n={size}, found {found}")
    return errors


def _validate_report_files(artifact_dir: Path | None) -> list[str]:
    if artifact_dir is None:
        return ["missing artifact directory for report-file validation"]
    return [f"missing report file {file_name}" for file_name in REPORT_FILES if not (artifact_dir / file_name).exists()]


def _validate_source_papers(payload: dict[str, Any], *, source_root: Path) -> list[str]:
    metadata = payload.get("metadata")
    paper_setup = metadata.get("paper_setup") if isinstance(metadata, dict) else None
    source_papers = metadata.get("source_papers") if isinstance(metadata, dict) else None
    errors: list[str] = []
    if not isinstance(paper_setup, str) or not paper_setup:
        errors.append("missing metadata.paper_setup")
    if not isinstance(source_papers, list):
        return [
            *errors,
            *[f"missing metadata.source_papers {paper_id}" for paper_id in REQUIRED_SOURCE_PAPER_IDS],
        ]

    papers_by_id = {
        paper.get("id"): paper for paper in source_papers if isinstance(paper, dict) and paper.get("id") is not None
    }
    for paper_id in REQUIRED_SOURCE_PAPER_IDS:
        paper = papers_by_id.get(paper_id)
        if not isinstance(paper, dict):
            errors.append(f"missing metadata.source_papers {paper_id}")
            continue
        path = paper.get("path")
        if not isinstance(path, str) or not path.startswith("tmp/sources/"):
            errors.append(f"metadata.source_papers {paper_id} path must stay under tmp/sources/")
            continue
        if not (source_root / path).is_file():
            errors.append(f"missing metadata.source_papers {paper_id} file {path}")
    return errors


def _validate_command_examples(payload: dict[str, Any]) -> list[str]:
    metadata = payload.get("metadata")
    examples = metadata.get("command_examples") if isinstance(metadata, dict) else None
    errors: list[str] = []
    if not isinstance(examples, dict):
        return [
            "missing metadata.command_examples.local_sample",
            "missing metadata.command_examples.remote_sample",
        ]

    local_sample = examples.get("local_sample")
    remote_sample = examples.get("remote_sample")
    if not isinstance(local_sample, str) or not local_sample:
        errors.append("missing metadata.command_examples.local_sample")
    else:
        if str(Path.cwd()) in local_sample:
            errors.append("metadata.command_examples.local_sample contains local checkout path")
        if "$PWD" not in local_sample:
            errors.append("metadata.command_examples.local_sample must use $PWD")

    if not isinstance(remote_sample, str) or not remote_sample:
        errors.append("missing metadata.command_examples.remote_sample")
    elif "ssh" not in remote_sample.split():
        errors.append("metadata.command_examples.remote_sample must use ssh")

    sync_sample = examples.get("sync_remote_tree")
    if isinstance(sync_sample, str) and str(Path.cwd()) in sync_sample:
        errors.append("metadata.command_examples.sync_remote_tree contains local checkout path")

    return errors


def _validate_zero_scheduler_errors(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        scheduler_errors = row.get("device_scheduler_errors")
        baseline = row.get("baseline", "unknown")
        machine = row.get("machine", "unknown")
        size = row.get("n", "unknown")
        if scheduler_errors is None:
            if isinstance(baseline, str) and baseline.startswith("pto_persistent_dag"):
                errors.append(f"missing scheduler errors machine={machine} baseline={baseline} n={size}")
            continue
        if not isinstance(scheduler_errors, dict):
            errors.append(f"invalid scheduler errors machine={machine} baseline={baseline} n={size}")
            continue
        count = scheduler_errors.get("count", 0)
        code = scheduler_errors.get("code", 0)
        task_id = scheduler_errors.get("task_id", 0)
        if count != 0 or code != 0 or task_id != 0:
            errors.append(
                "scheduler error "
                f"machine={machine} baseline={baseline} n={size} "
                f"count={count} code={code} task_id={task_id}"
            )
    return errors


def _validate_dispatch(rows: list[dict[str, Any]], required_dispatch: dict[str, str]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        baseline = row.get("baseline")
        expected = required_dispatch.get(str(baseline))
        if expected is None:
            continue
        found = _dispatch_text(row)
        if found != expected:
            machine = row.get("machine", "unknown")
            n = row.get("n", "unknown")
            errors.append(
                f"expected dispatch {expected} for machine={machine} baseline={baseline} n={n}, found {found}"
            )
    return errors


def _validate_tensor_tiles(rows: list[dict[str, Any]], required_tensor_tiles: dict[str, str]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        baseline = row.get("baseline")
        expected = required_tensor_tiles.get(str(baseline))
        if expected is None:
            continue
        found = _tensor_tile_shape(row)
        if found != expected:
            machine = row.get("machine", "unknown")
            n = row.get("n", "unknown")
            errors.append(
                f"expected tensor tile {expected} for machine={machine} baseline={baseline} n={n}, found {found}"
            )
    return errors


def _validate_scratch_reuse(rows: list[dict[str, Any]], required_scratch_reuse: dict[str, str]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        baseline = row.get("baseline")
        expected = required_scratch_reuse.get(str(baseline))
        if expected is None:
            continue
        found = _scratch_reuse_text(row)
        if found != expected:
            machine = row.get("machine", "unknown")
            n = row.get("n", "unknown")
            errors.append(
                f"expected scratch_reuse {expected} for machine={machine} baseline={baseline} n={n}, found {found}"
            )
    return errors


def _validate_graph_task_args(rows: list[dict[str, Any]], required_graph_task_args: dict[str, str]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        baseline = row.get("baseline")
        expected = required_graph_task_args.get(str(baseline))
        if expected is None:
            continue
        found = _graph_task_args_text(row)
        if found != expected:
            machine = row.get("machine", "unknown")
            n = row.get("n", "unknown")
            errors.append(
                f"expected graph_task_args {expected} for machine={machine} baseline={baseline} n={n}, found {found}"
            )
    return errors


def _validate_graph_task_arg_keys(
    rows: list[dict[str, Any]],
    required_graph_task_arg_keys: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    for row in rows:
        baseline = row.get("baseline")
        expected = required_graph_task_arg_keys.get(str(baseline))
        if expected is None:
            continue
        found = _graph_task_arg_key_text(row)
        if found != expected:
            machine = row.get("machine", "unknown")
            n = row.get("n", "unknown")
            errors.append(
                f"expected graph_task_arg_key {expected} for machine={machine} baseline={baseline} n={n}, found {found}"
            )
    return errors


def _validate_graph_descriptor(
    rows: list[dict[str, Any]],
    *,
    field_name: str,
    required_values: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    for row in rows:
        baseline = row.get("baseline")
        expected = required_values.get(str(baseline))
        if expected is None:
            continue
        found = _graph_descriptor_text(row, field_name)
        if found != expected:
            machine = row.get("machine", "unknown")
            n = row.get("n", "unknown")
            errors.append(
                f"expected graph_descriptor.{field_name} {expected} "
                f"for machine={machine} baseline={baseline} n={n}, found {found}"
            )
    return errors


def validate_capture(  # noqa: PLR0913
    payload: dict[str, Any],
    *,
    artifact_dir: Path | None = None,
    required_machines: Sequence[str] = (),
    required_baselines: Sequence[str] = (),
    required_sizes: Sequence[int] = (),
    expected_repeats: int | None = None,
    expected_result_count: int | None = None,
    require_report_files: bool = False,
    require_command_examples: bool = False,
    require_zero_scheduler_errors: bool = False,
    required_dispatch: dict[str, str] | None = None,
    required_tensor_tiles: dict[str, str] | None = None,
    required_scratch_reuse: dict[str, str] | None = None,
    required_graph_task_args: dict[str, str] | None = None,
    required_graph_task_arg_keys: dict[str, str] | None = None,
    required_graph_fanin: dict[str, str] | None = None,
    required_graph_dependents: dict[str, str] | None = None,
    source_paper_root: Path | None = None,
) -> list[str]:
    rows = _results(payload)
    errors: list[str] = []
    if not rows:
        errors.append("missing benchmark results")

    if expected_result_count is not None and len(rows) != expected_result_count:
        errors.append(f"expected {expected_result_count} results, found {len(rows)}")

    errors.extend(_validate_required_machines(rows, required_machines))
    errors.extend(_validate_status(rows))
    errors.extend(
        _validate_required_rows(
            rows,
            required_machines=required_machines,
            required_baselines=required_baselines,
            required_sizes=required_sizes,
            expected_repeats=expected_repeats,
        )
    )

    if require_report_files:
        errors.extend(_validate_report_files(artifact_dir))

    if require_command_examples:
        errors.extend(_validate_command_examples(payload))

    if require_zero_scheduler_errors:
        errors.extend(_validate_zero_scheduler_errors(rows))

    errors.extend(_validate_dispatch(rows, required_dispatch or {}))
    errors.extend(_validate_tensor_tiles(rows, required_tensor_tiles or {}))
    errors.extend(_validate_scratch_reuse(rows, required_scratch_reuse or {}))
    errors.extend(_validate_graph_task_args(rows, required_graph_task_args or {}))
    errors.extend(_validate_graph_task_arg_keys(rows, required_graph_task_arg_keys or {}))
    errors.extend(_validate_graph_descriptor(rows, field_name="fanin", required_values=required_graph_fanin or {}))
    errors.extend(
        _validate_graph_descriptor(rows, field_name="dependents", required_values=required_graph_dependents or {})
    )

    if source_paper_root is not None:
        errors.extend(_validate_source_papers(payload, source_root=source_paper_root))

    return errors


def _apply_preset(args: argparse.Namespace) -> None:
    if args.preset not in {"paired-current", "compact-current"}:
        return
    if not args.require_machine:
        args.require_machine = list(PAIRED_CURRENT_MACHINES)
    if not args.require_baseline:
        args.require_baseline = list(PAIRED_CURRENT_BASELINES)
    if not args.require_size:
        sizes = COMPACT_CURRENT_SIZES if args.preset == "compact-current" else PAIRED_CURRENT_SIZES
        args.require_size = [str(size) for size in sizes]
    if args.expected_repeats is None:
        args.expected_repeats = COMPACT_CURRENT_EXPECTED_REPEATS if args.preset == "compact-current" else 3
    if args.expected_result_count is None:
        args.expected_result_count = (
            COMPACT_CURRENT_EXPECTED_RESULT_COUNT
            if args.preset == "compact-current"
            else PAIRED_CURRENT_EXPECTED_RESULT_COUNT
        )
    if not args.require_dispatch:
        args.require_dispatch = [f"{baseline}={dispatch}" for baseline, dispatch in PAIRED_CURRENT_DISPATCH.items()]
    if not args.require_tensor_tile:
        args.require_tensor_tile = [f"{baseline}={shape}" for baseline, shape in PAIRED_CURRENT_TENSOR_TILES.items()]
    if not args.require_scratch_reuse:
        args.require_scratch_reuse = [
            f"{baseline}={metadata}" for baseline, metadata in PAIRED_CURRENT_SCRATCH_REUSE.items()
        ]
    if not args.require_graph_task_args:
        args.require_graph_task_args = [
            f"{baseline}={metadata}" for baseline, metadata in PAIRED_CURRENT_GRAPH_TASK_ARGS.items()
        ]
    if not args.require_graph_task_arg_key:
        args.require_graph_task_arg_key = [
            f"{baseline}={metadata}" for baseline, metadata in PAIRED_CURRENT_GRAPH_TASK_ARG_KEYS.items()
        ]
    if not args.require_graph_fanin:
        args.require_graph_fanin = [
            f"{baseline}={metadata}" for baseline, metadata in PAIRED_CURRENT_GRAPH_FANIN.items()
        ]
    if not args.require_graph_dependents:
        args.require_graph_dependents = [
            f"{baseline}={metadata}" for baseline, metadata in PAIRED_CURRENT_GRAPH_DEPENDENTS.items()
        ]
    args.require_report_files = True
    args.require_zero_scheduler_errors = True
    if args.preset == "compact-current":
        args.require_command_examples = True
        args.require_source_papers = True


def _parse_required_mapping(values: Sequence[str] | None, *, flag: str) -> dict[str, str]:
    required: dict[str, str] = {}
    for value in values or ():
        if "=" not in value:
            raise ValueError(f"invalid {flag} {value!r}; expected BASELINE=VALUE")
        baseline, expected = value.split("=", 1)
        required[baseline.strip()] = expected.strip()
    return required


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--preset", choices=("none", "paired-current", "compact-current"), default="none")
    parser.add_argument("--require-machine", action="append")
    parser.add_argument("--require-baseline", action="append")
    parser.add_argument("--require-size", action="append")
    parser.add_argument("--expected-repeats", type=int)
    parser.add_argument("--expected-result-count", type=int)
    parser.add_argument("--require-report-files", action="store_true")
    parser.add_argument("--require-command-examples", action="store_true")
    parser.add_argument("--require-zero-scheduler-errors", action="store_true")
    parser.add_argument("--require-dispatch", action="append")
    parser.add_argument("--require-tensor-tile", action="append")
    parser.add_argument("--require-scratch-reuse", action="append")
    parser.add_argument("--require-graph-task-args", action="append")
    parser.add_argument("--require-graph-task-arg-key", action="append")
    parser.add_argument("--require-graph-fanin", action="append")
    parser.add_argument("--require-graph-dependents", action="append")
    parser.add_argument("--require-source-papers", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _apply_preset(args)
    payload = json.loads(args.json_path.read_text())
    try:
        required_dispatch = _parse_required_mapping(args.require_dispatch, flag="--require-dispatch")
        required_tensor_tiles = _parse_required_mapping(args.require_tensor_tile, flag="--require-tensor-tile")
        required_scratch_reuse = _parse_required_mapping(
            args.require_scratch_reuse,
            flag="--require-scratch-reuse",
        )
        required_graph_task_args = _parse_required_mapping(
            args.require_graph_task_args,
            flag="--require-graph-task-args",
        )
        required_graph_task_arg_keys = _parse_required_mapping(
            args.require_graph_task_arg_key,
            flag="--require-graph-task-arg-key",
        )
        required_graph_fanin = _parse_required_mapping(args.require_graph_fanin, flag="--require-graph-fanin")
        required_graph_dependents = _parse_required_mapping(
            args.require_graph_dependents,
            flag="--require-graph-dependents",
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    errors = validate_capture(
        payload,
        artifact_dir=args.json_path.parent,
        required_machines=_as_list(args.require_machine),
        required_baselines=_as_list(args.require_baseline),
        required_sizes=_as_int_list(args.require_size),
        expected_repeats=args.expected_repeats,
        expected_result_count=args.expected_result_count,
        require_report_files=args.require_report_files,
        require_command_examples=args.require_command_examples,
        require_zero_scheduler_errors=args.require_zero_scheduler_errors,
        required_dispatch=required_dispatch,
        required_tensor_tiles=required_tensor_tiles,
        required_scratch_reuse=required_scratch_reuse,
        required_graph_task_args=required_graph_task_args,
        required_graph_task_arg_keys=required_graph_task_arg_keys,
        required_graph_fanin=required_graph_fanin,
        required_graph_dependents=required_graph_dependents,
        source_paper_root=Path.cwd() if args.require_source_papers else None,
    )
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"validated {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
