#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate CUDA scheduler-error matrix artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from cuda_scheduler_error_matrix import DEFAULT_SCHEDULER_ERROR_CASES  # noqa: E402
from cuda_scheduler_errors import scheduler_error_code_label  # noqa: E402

REQUIRED_SOURCE_PAPER_IDS = ("arXiv:2605.03190", "arXiv:2512.22219v1")
DEFAULT_MACHINES = ("a100", "h200")
REPORT_FILES = (
    "cuda-scheduler-error-matrix.md",
    "cuda-scheduler-error-matrix.svg",
)
COMMAND_EXAMPLE_SCRIPTS = ("cuda_scheduler_error_matrix.py", "cuda_persistent_smoke.py")


def load_scheduler_error_matrix(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("results")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _validate_required_machines(rows: list[dict[str, Any]], required_machines: list[str]) -> list[str]:
    machines = {str(row.get("machine")) for row in rows if row.get("machine") is not None}
    return [f"missing machine {machine}" for machine in required_machines if machine not in machines]


def _validate_required_cases(rows: list[dict[str, Any]], required_cases: list[str]) -> list[str]:
    cases = {str(row.get("case")) for row in rows if row.get("case") is not None}
    return [f"missing case {case}" for case in required_cases if case not in cases]


def _validate_case_machine_coverage(
    rows: list[dict[str, Any]],
    *,
    required_cases: list[str],
    required_machines: list[str],
) -> list[str]:
    pairs = {(str(row.get("machine")), str(row.get("case"))) for row in rows}
    return [
        f"missing row machine={machine} case={case}"
        for machine in required_machines
        for case in required_cases
        if (machine, case) not in pairs
    ]


def _expected_case_contracts() -> dict[str, tuple[str, int, int]]:
    return {
        case.name: (case.dag_shape, case.expected_code, case.expected_task_id) for case in DEFAULT_SCHEDULER_ERROR_CASES
    }


def _validate_rows(rows: list[dict[str, Any]]) -> list[str]:
    contracts = _expected_case_contracts()
    errors: list[str] = []
    for row in rows:
        machine = row.get("machine", "unknown")
        case = str(row.get("case", "unknown"))
        expected = contracts.get(case)
        if expected is None:
            errors.append(f"unknown case machine={machine} case={case}")
            continue
        expected_shape, expected_code, expected_task_id = expected
        if row.get("dag_shape") != expected_shape:
            errors.append(
                f"wrong dag shape machine={machine} case={case} expected {expected_shape}, found {row.get('dag_shape')}"
            )
        if row.get("status") != "pass":
            errors.append(f"non-pass row machine={machine} case={case} status={row.get('status')}")
        if row.get("expected_code") != expected_code or row.get("expected_task_id") != expected_task_id:
            errors.append(
                f"wrong expected diagnostic machine={machine} case={case} "
                f"expected code={scheduler_error_code_label(expected_code)} task={expected_task_id}, "
                f"found code={scheduler_error_code_label(row.get('expected_code'))} "
                f"task={row.get('expected_task_id')}"
            )
        observed_code = row.get("observed_code")
        observed_task_id = row.get("observed_task_id")
        observed_count = row.get("observed_count")
        if observed_code != expected_code or observed_task_id != expected_task_id or observed_count != 1:
            errors.append(
                f"wrong observed diagnostic machine={machine} case={case} "
                f"expected code={scheduler_error_code_label(expected_code)} task={expected_task_id} count=1, "
                f"observed code={scheduler_error_code_label(observed_code)} "
                f"task={observed_task_id} count={observed_count}"
            )
    return errors


def _validate_report_files(artifact_dir: Path | None) -> list[str]:
    if artifact_dir is None:
        return ["missing artifact directory for report-file validation"]
    return [f"missing report file {file_name}" for file_name in REPORT_FILES if not (artifact_dir / file_name).exists()]


def _validate_source_papers(payload: dict[str, Any], *, source_root: Path) -> list[str]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return [f"missing metadata.source_papers {paper_id}" for paper_id in REQUIRED_SOURCE_PAPER_IDS]
    papers = metadata.get("source_papers")
    if not isinstance(papers, list):
        return [f"missing metadata.source_papers {paper_id}" for paper_id in REQUIRED_SOURCE_PAPER_IDS]
    errors: list[str] = []
    by_id = {paper.get("id"): paper for paper in papers if isinstance(paper, dict)}
    for paper_id in REQUIRED_SOURCE_PAPER_IDS:
        paper = by_id.get(paper_id)
        if paper is None:
            errors.append(f"missing metadata.source_papers {paper_id}")
            continue
        path = paper.get("path")
        if not isinstance(path, str) or not path.startswith("tmp/sources/"):
            errors.append(f"missing metadata.source_papers {paper_id} file {path}")
        elif not (source_root / path).exists():
            errors.append(f"missing metadata.source_papers {paper_id} file {path}")
    return errors


def _validate_command_examples(payload: dict[str, Any]) -> list[str]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return [
            "missing metadata.command_examples.local_sample",
            "missing metadata.command_examples.remote_sample",
        ]
    examples = metadata.get("command_examples")
    if not isinstance(examples, dict):
        return [
            "missing metadata.command_examples.local_sample",
            "missing metadata.command_examples.remote_sample",
        ]
    errors: list[str] = []
    local = examples.get("local_sample")
    remote = examples.get("remote_sample")
    if not isinstance(local, str) or not any(script in local for script in COMMAND_EXAMPLE_SCRIPTS):
        errors.append("missing metadata.command_examples.local_sample")
    if not isinstance(remote, str) or not any(script in remote for script in COMMAND_EXAMPLE_SCRIPTS):
        errors.append("missing metadata.command_examples.remote_sample")
    return errors


def validate_scheduler_error_matrix(
    payload: dict[str, Any],
    *,
    artifact_dir: Path | None = None,
    required_cases: list[str] | None = None,
    required_machines: list[str] | None = None,
    require_report_files: bool = False,
    require_source_papers: bool = False,
    require_command_examples: bool = False,
    source_paper_root: Path | None = None,
) -> list[str]:
    rows = _rows(payload)
    errors: list[str] = []
    if not rows:
        errors.append("missing scheduler error matrix rows")
    required_cases = required_cases or [case.name for case in DEFAULT_SCHEDULER_ERROR_CASES]
    required_machines = required_machines or list(DEFAULT_MACHINES)
    errors.extend(_validate_required_cases(rows, required_cases))
    errors.extend(_validate_required_machines(rows, required_machines))
    errors.extend(
        _validate_case_machine_coverage(rows, required_cases=required_cases, required_machines=required_machines)
    )
    errors.extend(_validate_rows(rows))
    if require_report_files:
        errors.extend(_validate_report_files(artifact_dir))
    if require_source_papers:
        errors.extend(_validate_source_papers(payload, source_root=source_paper_root or Path.cwd()))
    if require_command_examples:
        errors.extend(_validate_command_examples(payload))
    return errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--preset", choices=("none", "default"), default="none")
    parser.add_argument("--require-case", action="append")
    parser.add_argument("--require-machine", action="append")
    parser.add_argument("--require-report-files", action="store_true")
    parser.add_argument("--require-source-papers", action="store_true")
    parser.add_argument("--require-command-examples", action="store_true")
    return parser.parse_args(argv)


def _apply_preset(args: argparse.Namespace) -> None:
    if args.preset != "default":
        return
    if not args.require_case:
        args.require_case = [case.name for case in DEFAULT_SCHEDULER_ERROR_CASES]
    if not args.require_machine:
        args.require_machine = list(DEFAULT_MACHINES)
    args.require_report_files = True
    args.require_source_papers = True
    args.require_command_examples = True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _apply_preset(args)
    errors = validate_scheduler_error_matrix(
        load_scheduler_error_matrix(args.json_path),
        artifact_dir=args.json_path.parent,
        required_cases=args.require_case,
        required_machines=args.require_machine,
        require_report_files=args.require_report_files,
        require_source_papers=args.require_source_papers,
        require_command_examples=args.require_command_examples,
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
