#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate CUDA persistent lifecycle matrix artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPORT_FILES = (
    "cuda-lifecycle-matrix.md",
    "cuda-lifecycle-matrix.svg",
)
SCHEDULER_ERROR_NAMES = {
    0: "none",
    1: "unsupported_func_id",
    2: "invalid_dependent_id",
    3: "invalid_dependent_range",
    4: "fanin_underflow",
    5: "initial_fanin_mismatch",
    6: "no_root_task",
    7: "unreachable_task",
    8: "duplicate_dependent",
}
REQUIRED_SOURCE_PAPER_IDS = ("arXiv:2605.03190", "arXiv:2512.22219v1")
DEFAULT_SCENARIOS = ("direct", "queue", "dag-chain", "graph-scratch-reuse")
DEFAULT_ARTIFACTS = ("a100", "h200")
DEFAULT_DISPATCH = {
    "dag-chain": "1,2,1,2,1",
    "graph-scratch-reuse": "1,2,1,2,1,1",
}


def scheduler_error_code_label(code: Any) -> str:
    if not isinstance(code, int):
        return str(code)
    name = SCHEDULER_ERROR_NAMES.get(code)
    if name is None:
        return str(code)
    if code == 0:
        return "0"
    return f"{code}({name})"


def _as_list(values: Sequence[str] | None) -> list[str]:
    if values is None:
        return []
    result: list[str] = []
    for value in values:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    return result


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _dispatch(row: dict[str, Any]) -> str | None:
    dispatch = row.get("dispatch_func_ids")
    if not isinstance(dispatch, list):
        return None
    return ",".join(str(func_id) for func_id in dispatch)


def load_lifecycle_matrix(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _validate_required_scenarios(rows: list[dict[str, Any]], required_scenarios: Sequence[str]) -> list[str]:
    scenarios = {str(row.get("scenario")) for row in rows}
    return [f"missing scenario {scenario}" for scenario in required_scenarios if scenario not in scenarios]


def _validate_required_artifacts(rows: list[dict[str, Any]], required_artifacts: Sequence[str]) -> list[str]:
    artifacts = {str(row.get("artifact")) for row in rows}
    return [f"missing artifact {artifact}" for artifact in required_artifacts if artifact not in artifacts]


def _validate_status(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        if row.get("status") == "pass":
            continue
        errors.append(
            "non-pass "
            f"scenario={row.get('scenario', 'unknown')} "
            f"artifact={row.get('artifact', 'unknown')} "
            f"status={row.get('status', 'unknown')}"
        )
    return errors


def _validate_scheduler_errors(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        scheduler_errors = row.get("device_scheduler_errors")
        if not isinstance(scheduler_errors, dict):
            continue
        count = scheduler_errors.get("count", 0)
        code = scheduler_errors.get("code", 0)
        task_id = scheduler_errors.get("task_id", 0)
        if count == 0 and code == 0 and task_id == 0:
            continue
        errors.append(
            "scheduler error "
            f"scenario={row.get('scenario', 'unknown')} "
            f"artifact={row.get('artifact', 'unknown')} "
            f"count={count} code={scheduler_error_code_label(code)} task={task_id}"
        )
    return errors


def _validate_lifecycle(rows: list[dict[str, Any]], expected_repeat_runs: int | None) -> list[str]:
    errors: list[str] = []
    for row in rows:
        scenario = row.get("scenario", "unknown")
        artifact = row.get("artifact", "unknown")
        repeat_runs = row.get("repeat_runs")
        if expected_repeat_runs is not None and repeat_runs != expected_repeat_runs:
            errors.append(
                f"expected repeat_runs {expected_repeat_runs} for scenario={scenario} "
                f"artifact={artifact}, found {repeat_runs}"
            )
        counts = row.get("launch_completed_counts")
        if expected_repeat_runs is not None and isinstance(counts, list) and len(counts) != expected_repeat_runs:
            errors.append(
                f"expected {expected_repeat_runs} completion entries for scenario={scenario} "
                f"artifact={artifact}, found {len(counts)}"
            )
        expected_completed = row.get("completed_count")
        if expected_completed is None:
            continue
        if not isinstance(counts, list):
            errors.append(f"missing launch_completed_counts for scenario={scenario} artifact={artifact}")
            continue
        for launch_idx, count in enumerate(counts):
            if count != expected_completed:
                errors.append(
                    f"expected completed count {expected_completed} for scenario={scenario} "
                    f"artifact={artifact} launch={launch_idx}, found {count}"
                )
    return errors


def _validate_dispatch(rows: list[dict[str, Any]], required_dispatch: dict[str, str]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        scenario = str(row.get("scenario"))
        expected = required_dispatch.get(scenario)
        if expected is None:
            continue
        found = _dispatch(row)
        if found != expected:
            errors.append(
                f"expected dispatch {expected} for scenario={scenario} "
                f"artifact={row.get('artifact', 'unknown')}, found {found}"
            )
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
        if (
            isinstance(metadata, dict)
            and metadata.get("collection_mode") == "existing"
            and "--collect-existing-suffix" not in local_sample
        ):
            errors.append("metadata.command_examples.local_sample missing --collect-existing-suffix")

    if not isinstance(remote_sample, str) or not remote_sample:
        errors.append("missing metadata.command_examples.remote_sample")
    elif "ssh" not in remote_sample.split():
        errors.append("metadata.command_examples.remote_sample must use ssh")

    sync_sample = examples.get("sync_remote_tree")
    if isinstance(sync_sample, str) and str(Path.cwd()) in sync_sample:
        errors.append("metadata.command_examples.sync_remote_tree contains local checkout path")

    return errors


def validate_lifecycle_matrix(
    payload: dict[str, Any],
    *,
    artifact_dir: Path | None = None,
    expected_repeat_runs: int | None = None,
    required_scenarios: Sequence[str] = (),
    require_artifacts: Sequence[str] = (),
    required_dispatch: dict[str, str] | None = None,
    require_report_files: bool = False,
    require_source_papers: bool = False,
    require_command_examples: bool = False,
    source_paper_root: Path | None = None,
) -> list[str]:
    rows = _rows(payload)
    errors: list[str] = []
    if not rows:
        errors.append("missing lifecycle matrix rows")

    errors.extend(_validate_required_scenarios(rows, required_scenarios))
    errors.extend(_validate_required_artifacts(rows, require_artifacts))
    errors.extend(_validate_status(rows))
    errors.extend(_validate_scheduler_errors(rows))
    errors.extend(_validate_lifecycle(rows, expected_repeat_runs))
    errors.extend(_validate_dispatch(rows, required_dispatch or DEFAULT_DISPATCH))
    if require_report_files:
        errors.extend(_validate_report_files(artifact_dir))
    if require_source_papers:
        errors.extend(_validate_source_papers(payload, source_root=source_paper_root or Path.cwd()))
    if require_command_examples:
        errors.extend(_validate_command_examples(payload))
    return errors


def _parse_required_dispatch(values: Sequence[str] | None) -> dict[str, str]:
    required: dict[str, str] = {}
    for value in values or ():
        if "=" not in value:
            raise ValueError(f"invalid --require-dispatch {value!r}; expected SCENARIO=VALUE")
        scenario, dispatch = value.split("=", 1)
        required[scenario.strip()] = dispatch.strip()
    return required


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--preset", choices=("none", "default"), default="none")
    parser.add_argument("--expected-repeat-runs", type=int)
    parser.add_argument("--require-scenario", action="append")
    parser.add_argument("--require-artifact", action="append")
    parser.add_argument("--require-dispatch", action="append")
    parser.add_argument("--require-report-files", action="store_true")
    parser.add_argument("--require-source-papers", action="store_true")
    parser.add_argument("--require-command-examples", action="store_true")
    return parser.parse_args(argv)


def _apply_preset(args: argparse.Namespace) -> None:
    if args.preset != "default":
        return
    if args.expected_repeat_runs is None:
        args.expected_repeat_runs = 2
    if not args.require_scenario:
        args.require_scenario = list(DEFAULT_SCENARIOS)
    if not args.require_artifact:
        args.require_artifact = list(DEFAULT_ARTIFACTS)
    if not args.require_dispatch:
        args.require_dispatch = [f"{scenario}={dispatch}" for scenario, dispatch in DEFAULT_DISPATCH.items()]
    args.require_report_files = True


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _apply_preset(args)
    try:
        required_dispatch = _parse_required_dispatch(args.require_dispatch)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    errors = validate_lifecycle_matrix(
        load_lifecycle_matrix(args.json_path),
        artifact_dir=args.json_path.parent,
        expected_repeat_runs=args.expected_repeat_runs,
        required_scenarios=_as_list(args.require_scenario),
        require_artifacts=_as_list(args.require_artifact),
        required_dispatch=required_dispatch,
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
