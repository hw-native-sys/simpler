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
    "direct_driver",
    "direct_driver_graph",
    "pto_host_schedule",
    "pto_host_schedule_batch",
    "pto_host_schedule_compiler",
    "pto_host_schedule_quad",
    "pto_host_schedule_unary_square",
    "pto_persistent_dag",
    "pto_persistent_dag_chain",
    "pto_persistent_dag_reuse",
    "pto_persistent_dag_scalar_affine",
    "pto_persistent_dag_scalar_axpy",
    "pto_persistent_dag_tensor",
    "pto_persistent_dag_triad",
    "pto_persistent_dag_quad",
    "pto_persistent_dag_generic_args",
    "pto_persistent_dag_unary_square",
    "pto_persistent_device",
    "pto_persistent_device_batch",
    "pto_persistent_device_grid_batch",
    "pto_persistent_queue",
    "pto_persistent_queue_batch",
)
PAIRED_CURRENT_SIZES = (1024, 65536, 1048576)
REPORT_FILES = (
    "cuda-benchmark.md",
    "cuda-benchmark.svg",
    "cuda-benchmark-ratios.svg",
)


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


def validate_capture(
    payload: dict[str, Any],
    *,
    artifact_dir: Path | None = None,
    required_machines: Sequence[str] = (),
    required_baselines: Sequence[str] = (),
    required_sizes: Sequence[int] = (),
    expected_repeats: int | None = None,
    expected_result_count: int | None = None,
    require_report_files: bool = False,
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

    return errors


def _apply_preset(args: argparse.Namespace) -> None:
    if args.preset != "paired-current":
        return
    if not args.require_machine:
        args.require_machine = list(PAIRED_CURRENT_MACHINES)
    if not args.require_baseline:
        args.require_baseline = list(PAIRED_CURRENT_BASELINES)
    if not args.require_size:
        args.require_size = [str(size) for size in PAIRED_CURRENT_SIZES]
    if args.expected_repeats is None:
        args.expected_repeats = 3
    if args.expected_result_count is None:
        args.expected_result_count = 702
    args.require_report_files = True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--preset", choices=("none", "paired-current"), default="none")
    parser.add_argument("--require-machine", action="append")
    parser.add_argument("--require-baseline", action="append")
    parser.add_argument("--require-size", action="append")
    parser.add_argument("--expected-repeats", type=int)
    parser.add_argument("--expected-result-count", type=int)
    parser.add_argument("--require-report-files", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _apply_preset(args)
    payload = json.loads(args.json_path.read_text())
    errors = validate_capture(
        payload,
        artifact_dir=args.json_path.parent,
        required_machines=_as_list(args.require_machine),
        required_baselines=_as_list(args.require_baseline),
        required_sizes=_as_int_list(args.require_size),
        expected_repeats=args.expected_repeats,
        expected_result_count=args.expected_result_count,
        require_report_files=args.require_report_files,
    )
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"validated {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
