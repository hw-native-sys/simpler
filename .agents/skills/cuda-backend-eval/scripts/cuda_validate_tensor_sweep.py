#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate CUDA tensor shape sweep artifacts before publishing summaries."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

COMPACT_TENSOR_BASELINE_ARTIFACTS = ("a100", "h200")
COMPACT_TENSOR_BASELINE_BASELINES = (
    "pto_persistent_dag_tensor",
    "pto_persistent_dag_tensor_core",
    "cublas_sgemm",
)
COMPACT_TENSOR_BASELINE_SIZES = (256,)
COMPACT_TENSOR_BASELINE_SHAPES = ("16x16x16", "16x16x64")
COMPACT_TENSOR_BASELINE_REPEATS = 3
COMPACT_TENSOR_BASELINE_RESULT_COUNT = (
    len(COMPACT_TENSOR_BASELINE_ARTIFACTS)
    * len(COMPACT_TENSOR_BASELINE_BASELINES)
    * len(COMPACT_TENSOR_BASELINE_SHAPES)
    * COMPACT_TENSOR_BASELINE_REPEATS
)
COMPACT_TENSOR_BASELINE_DISPATCH = {
    "pto_persistent_dag_tensor": "3,1,2,1",
    "pto_persistent_dag_tensor_core": "10,1,2,1",
}
REPORT_FILES = (
    "cuda-tensor-shape-sweep.json",
    "cuda-tensor-shape-sweep.md",
    "cuda-tensor-shape-sweep.svg",
    "cuda-tensor-shape-throughput.svg",
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


def _unique_repeats(rows: Sequence[dict[str, Any]]) -> set[Any]:
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


def _validate_required_artifacts(rows: list[dict[str, Any]], required_artifacts: Sequence[str]) -> list[str]:
    artifacts = {str(row.get("artifact")) for row in rows if row.get("artifact") is not None}
    return [f"missing artifact {artifact}" for artifact in required_artifacts if artifact not in artifacts]


def _validate_status(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        if row.get("status", "pass") == "pass":
            continue
        artifact = row.get("artifact", "unknown")
        baseline = row.get("baseline", "unknown")
        n = row.get("n", "unknown")
        shape = row.get("shape", "unknown")
        errors.append(f"non-pass row artifact={artifact} baseline={baseline} n={n} shape={shape}")
    return errors


def _validate_required_rows(
    rows: list[dict[str, Any]],
    *,
    required_artifacts: Sequence[str],
    required_baselines: Sequence[str],
    required_sizes: Sequence[int],
    required_shapes: Sequence[str],
    expected_repeats: int | None,
) -> list[str]:
    errors: list[str] = []
    sizes: Sequence[int | None] = required_sizes or (None,)
    for artifact in required_artifacts:
        for baseline in required_baselines:
            for size in sizes:
                for shape in required_shapes:
                    matching = [
                        row
                        for row in rows
                        if row.get("artifact") == artifact
                        and row.get("baseline") == baseline
                        and (size is None or row.get("n") == size)
                        and row.get("shape") == shape
                    ]
                    if not matching:
                        size_text = "" if size is None else f" n={size}"
                        errors.append(f"missing baseline {baseline} artifact={artifact}{size_text} shape={shape}")
                        continue
                    if expected_repeats is not None and len(_unique_repeats(matching)) != expected_repeats:
                        found = len(_unique_repeats(matching))
                        size_text = "" if size is None else f" n={size}"
                        errors.append(
                            f"expected {expected_repeats} repeats for artifact={artifact} "
                            f"baseline={baseline}{size_text} shape={shape}, found {found}"
                        )
    return errors


def _validate_tensor_tile_shapes(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        shape = row.get("shape")
        tensor_tile = row.get("tensor_tile")
        if shape is None or tensor_tile is None:
            continue
        found = _tensor_tile_shape(row)
        if found != shape:
            artifact = row.get("artifact", "unknown")
            baseline = row.get("baseline", "unknown")
            n = row.get("n", "unknown")
            errors.append(
                f"expected tensor tile {shape} for artifact={artifact} baseline={baseline} n={n}, found {found}"
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
            artifact = row.get("artifact", "unknown")
            n = row.get("n", "unknown")
            errors.append(f"expected dispatch {expected} for artifact={artifact} baseline={baseline} n={n}, found {found}")
    return errors


def _validate_report_files(artifact_dir: Path | None) -> list[str]:
    if artifact_dir is None:
        return ["missing artifact directory for report-file validation"]
    return [f"missing report file {file_name}" for file_name in REPORT_FILES if not (artifact_dir / file_name).exists()]


def _validate_command_examples(payload: dict[str, Any]) -> list[str]:
    metadata = payload.get("metadata")
    examples = (
        metadata.get("command_examples")
        if isinstance(metadata, dict)
        else None
    )
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
        errors.append(
            "metadata.command_examples.sync_remote_tree contains local checkout path"
        )

    return errors


def validate_tensor_sweep(
    payload: dict[str, Any],
    *,
    artifact_dir: Path | None = None,
    required_artifacts: Sequence[str] = (),
    required_baselines: Sequence[str] = (),
    required_sizes: Sequence[int] = (),
    required_shapes: Sequence[str] = (),
    expected_repeats: int | None = None,
    expected_result_count: int | None = None,
    required_dispatch: dict[str, str] | None = None,
    require_report_files: bool = False,
    require_command_examples: bool = False,
) -> list[str]:
    rows = _results(payload)
    errors: list[str] = []
    if not rows:
        errors.append("missing tensor sweep results")

    if expected_result_count is not None and len(rows) != expected_result_count:
        errors.append(f"expected {expected_result_count} results, found {len(rows)}")

    errors.extend(_validate_required_artifacts(rows, required_artifacts))
    errors.extend(_validate_status(rows))
    errors.extend(
        _validate_required_rows(
            rows,
            required_artifacts=required_artifacts,
            required_baselines=required_baselines,
            required_sizes=required_sizes,
            required_shapes=required_shapes,
            expected_repeats=expected_repeats,
        )
    )
    errors.extend(_validate_tensor_tile_shapes(rows))
    errors.extend(_validate_dispatch(rows, required_dispatch or {}))

    if require_report_files:
        errors.extend(_validate_report_files(artifact_dir))

    if require_command_examples:
        errors.extend(_validate_command_examples(payload))

    return errors


def _parse_required_dispatch(values: Sequence[str] | None) -> dict[str, str]:
    required: dict[str, str] = {}
    for value in values or ():
        if "=" not in value:
            raise ValueError(f"invalid --require-dispatch {value!r}; expected BASELINE=ID,ID")
        baseline, dispatch = value.split("=", 1)
        required[baseline.strip()] = dispatch.strip()
    return required


def _apply_preset(args: argparse.Namespace) -> None:
    if args.preset != "compact-tensor-baselines":
        return
    if not args.require_artifact:
        args.require_artifact = list(COMPACT_TENSOR_BASELINE_ARTIFACTS)
    if not args.require_baseline:
        args.require_baseline = list(COMPACT_TENSOR_BASELINE_BASELINES)
    if not args.require_size:
        args.require_size = [str(size) for size in COMPACT_TENSOR_BASELINE_SIZES]
    if not args.require_shape:
        args.require_shape = list(COMPACT_TENSOR_BASELINE_SHAPES)
    if args.expected_repeats is None:
        args.expected_repeats = COMPACT_TENSOR_BASELINE_REPEATS
    if args.expected_result_count is None:
        args.expected_result_count = COMPACT_TENSOR_BASELINE_RESULT_COUNT
    if not args.require_dispatch:
        args.require_dispatch = [
            f"{baseline}={dispatch}" for baseline, dispatch in COMPACT_TENSOR_BASELINE_DISPATCH.items()
        ]
    args.require_report_files = True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--preset", choices=("none", "compact-tensor-baselines"), default="none")
    parser.add_argument("--require-artifact", action="append")
    parser.add_argument("--require-baseline", action="append")
    parser.add_argument("--require-size", action="append")
    parser.add_argument("--require-shape", action="append")
    parser.add_argument("--expected-repeats", type=int)
    parser.add_argument("--expected-result-count", type=int)
    parser.add_argument("--require-dispatch", action="append")
    parser.add_argument("--require-report-files", action="store_true")
    parser.add_argument("--require-command-examples", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _apply_preset(args)
    payload = json.loads(args.json_path.read_text())
    try:
        required_dispatch = _parse_required_dispatch(args.require_dispatch)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    errors = validate_tensor_sweep(
        payload,
        artifact_dir=args.json_path.parent,
        required_artifacts=_as_list(args.require_artifact),
        required_baselines=_as_list(args.require_baseline),
        required_sizes=_as_int_list(args.require_size),
        required_shapes=_as_list(args.require_shape),
        expected_repeats=args.expected_repeats,
        expected_result_count=args.expected_result_count,
        required_dispatch=required_dispatch,
        require_report_files=args.require_report_files,
        require_command_examples=args.require_command_examples,
    )
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"validated {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
