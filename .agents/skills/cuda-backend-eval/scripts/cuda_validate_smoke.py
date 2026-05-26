#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate CUDA smoke artifacts before using them as A100/H200 evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPORT_FILES = (
    "cuda-smoke-report.md",
    "cuda-smoke-report.svg",
)


def _artifact_label(path: Path) -> str:
    name = path.stem.lower()
    if "a100" in name:
        return "a100"
    if "h200" in name:
        return "h200"
    return name


def _as_list(values: Sequence[str] | None) -> list[str]:
    if values is None:
        return []
    result: list[str] = []
    for value in values:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    return result


def _dispatch(row: dict[str, Any]) -> str | None:
    dispatch = row.get("dispatch_func_ids")
    if not isinstance(dispatch, list):
        return None
    return ",".join(str(func_id) for func_id in dispatch)


def load_smoke_payloads(paths: Sequence[Path]) -> list[dict[str, Any]]:
    payloads = []
    for path in paths:
        payload = json.loads(path.read_text())
        payload["_artifact"] = _artifact_label(path)
        payload["_path"] = str(path)
        payloads.append(payload)
    return payloads


def _validate_required_artifacts(payloads: list[dict[str, Any]], required_artifacts: Sequence[str]) -> list[str]:
    artifacts = {str(payload.get("_artifact")) for payload in payloads}
    return [f"missing artifact {artifact}" for artifact in required_artifacts if artifact not in artifacts]


def _validate_status(payloads: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for payload in payloads:
        if payload.get("status") == "pass":
            continue
        errors.append(
            f"non-pass artifact={payload.get('_artifact', 'unknown')} "
            f"status={payload.get('status', 'unknown')}"
        )
    return errors


def _validate_scheduler_errors(payloads: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for payload in payloads:
        scheduler_errors = payload.get("device_scheduler_errors")
        if not isinstance(scheduler_errors, dict):
            continue
        count = scheduler_errors.get("count", 0)
        if count == 0:
            continue
        errors.append(
            f"scheduler error artifact={payload.get('_artifact', 'unknown')} "
            f"count={count} code={scheduler_errors.get('code', 0)} task={scheduler_errors.get('task_id', 0)}"
        )
    return errors


def _validate_common_fields(
    payloads: list[dict[str, Any]],
    *,
    expected_runtime: str | None,
    expected_mode: str | None,
    expected_dag_shape: str | None,
    expected_dispatch: str | None,
) -> list[str]:
    errors: list[str] = []
    for payload in payloads:
        artifact = payload.get("_artifact", "unknown")
        for field_name, expected in (
            ("runtime", expected_runtime),
            ("mode", expected_mode),
            ("dag_shape", expected_dag_shape),
        ):
            if expected is not None and payload.get(field_name) != expected:
                errors.append(
                    f"expected {field_name}={expected} for artifact={artifact}, "
                    f"found {payload.get(field_name, 'unknown')}"
                )
        if expected_dispatch is not None and _dispatch(payload) != expected_dispatch:
            errors.append(f"expected dispatch {expected_dispatch} for artifact={artifact}, found {_dispatch(payload)}")
    return errors


def _validate_lifecycle(
    payloads: list[dict[str, Any]],
    *,
    expected_repeat_runs: int | None,
    expected_completed_count: int | None,
) -> list[str]:
    errors: list[str] = []
    for payload in payloads:
        artifact = payload.get("_artifact", "unknown")
        repeat_runs = payload.get("repeat_runs")
        if expected_repeat_runs is not None and repeat_runs != expected_repeat_runs:
            errors.append(f"expected repeat_runs {expected_repeat_runs} for artifact={artifact}, found {repeat_runs}")
        counts = payload.get("launch_completed_counts")
        if expected_repeat_runs is not None and isinstance(counts, list) and len(counts) != expected_repeat_runs:
            errors.append(
                f"expected {expected_repeat_runs} completion entries for artifact={artifact}, found {len(counts)}"
            )
        if expected_completed_count is None:
            continue
        if not isinstance(counts, list):
            errors.append(f"missing launch_completed_counts for artifact={artifact}")
            continue
        for launch_idx, count in enumerate(counts):
            if count != expected_completed_count:
                errors.append(
                    f"expected completed count {expected_completed_count} for artifact={artifact} "
                    f"launch={launch_idx}, found {count}"
                )
    return errors


def _validate_report_files(artifact_dir: Path | None) -> list[str]:
    if artifact_dir is None:
        return ["missing artifact directory for report-file validation"]
    return [f"missing report file {file_name}" for file_name in REPORT_FILES if not (artifact_dir / file_name).exists()]


def validate_smoke(
    payloads: list[dict[str, Any]],
    *,
    artifact_dir: Path | None = None,
    required_artifacts: Sequence[str] = (),
    expected_runtime: str | None = None,
    expected_mode: str | None = None,
    expected_dag_shape: str | None = None,
    expected_repeat_runs: int | None = None,
    expected_completed_count: int | None = None,
    expected_dispatch: str | None = None,
    require_report_files: bool = False,
) -> list[str]:
    errors: list[str] = []
    if not payloads:
        errors.append("missing smoke payloads")
    errors.extend(_validate_required_artifacts(payloads, required_artifacts))
    errors.extend(_validate_status(payloads))
    errors.extend(_validate_scheduler_errors(payloads))
    errors.extend(
        _validate_common_fields(
            payloads,
            expected_runtime=expected_runtime,
            expected_mode=expected_mode,
            expected_dag_shape=expected_dag_shape,
            expected_dispatch=expected_dispatch,
        )
    )
    errors.extend(
        _validate_lifecycle(
            payloads,
            expected_repeat_runs=expected_repeat_runs,
            expected_completed_count=expected_completed_count,
        )
    )
    if require_report_files:
        errors.extend(_validate_report_files(artifact_dir))
    return errors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_paths", nargs="+", type=Path)
    parser.add_argument("--require-artifact", action="append")
    parser.add_argument("--expected-runtime")
    parser.add_argument("--expected-mode")
    parser.add_argument("--expected-dag-shape")
    parser.add_argument("--expected-repeat-runs", type=int)
    parser.add_argument("--expected-completed-count", type=int)
    parser.add_argument("--expected-dispatch")
    parser.add_argument("--require-report-files", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payloads = load_smoke_payloads(args.json_paths)
    artifact_dir = args.json_paths[0].parent if args.json_paths else None
    errors = validate_smoke(
        payloads,
        artifact_dir=artifact_dir,
        required_artifacts=_as_list(args.require_artifact),
        expected_runtime=args.expected_runtime,
        expected_mode=args.expected_mode,
        expected_dag_shape=args.expected_dag_shape,
        expected_repeat_runs=args.expected_repeat_runs,
        expected_completed_count=args.expected_completed_count,
        expected_dispatch=args.expected_dispatch,
        require_report_files=args.require_report_files,
    )
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("validated " + ", ".join(str(path) for path in args.json_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
