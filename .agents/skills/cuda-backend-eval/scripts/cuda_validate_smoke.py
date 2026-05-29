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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPORT_FILES = (
    "cuda-smoke-report.md",
    "cuda-smoke-report.svg",
)


@dataclass(frozen=True)
class ResourcePolicyExpectation:
    scheduler_blocks: int | None = None
    worker_blocks: int | None = None
    worker_blocks_per_task: int | None = None
    stream_id: int | None = None
    block_dim: int | None = None
    grid_dim: int | None = None

    def fields(self) -> tuple[tuple[str, int | None], ...]:
        return (
            ("scheduler_blocks", self.scheduler_blocks),
            ("worker_blocks", self.worker_blocks),
            ("worker_blocks_per_task", self.worker_blocks_per_task),
            ("stream_id", self.stream_id),
            ("block_dim", self.block_dim),
            ("grid_dim", self.grid_dim),
        )

    def is_empty(self) -> bool:
        return all(expected is None for _, expected in self.fields())


@dataclass(frozen=True)
class SmokeValidationExpectation:
    artifact_dir: Path | None = None
    required_artifacts: Sequence[str] = ()
    runtime: str | None = None
    mode: str | None = None
    dag_shape: str | None = None
    repeat_runs: int | None = None
    completed_count: int | None = None
    dispatch: str | None = None
    tensor_tile: str | None = None
    graph_fanin: str | None = None
    graph_dependents: str | None = None
    graph_task_arg_key: str | None = None
    graph_task_args: str | None = None
    scratch_reuse: str | None = None
    resource_policy: ResourcePolicyExpectation | None = None
    require_report_files: bool = False
    require_report_graph_topology: bool = False
    require_report_graph_task_args: bool = False


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


def _tensor_tile_shape(row: dict[str, Any]) -> str | None:
    tensor_tile = row.get("tensor_tile")
    if not isinstance(tensor_tile, dict):
        return None
    rows = tensor_tile.get("rows")
    cols = tensor_tile.get("cols")
    inner = tensor_tile.get("inner")
    if rows is None or cols is None or inner is None:
        return None
    return f"{rows}x{cols}x{inner}"


def _int_list(row: dict[str, Any], field_name: str) -> str | None:
    values = row.get(field_name)
    if not isinstance(values, list):
        return None
    return ",".join(str(value) for value in values)


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
            f"non-pass artifact={payload.get('_artifact', 'unknown')} status={payload.get('status', 'unknown')}"
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
    expected_tensor_tile: str | None,
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
        if expected_tensor_tile is not None and _tensor_tile_shape(payload) != expected_tensor_tile:
            errors.append(
                f"expected tensor tile {expected_tensor_tile} for artifact={artifact}, "
                f"found {_tensor_tile_shape(payload)}"
            )
    return errors


def _validate_resource_policy(
    payloads: list[dict[str, Any]],
    *,
    expected_policy: ResourcePolicyExpectation | None,
) -> list[str]:
    errors: list[str] = []
    if expected_policy is None or expected_policy.is_empty():
        return errors
    for payload in payloads:
        artifact = payload.get("_artifact", "unknown")
        policy = payload.get("resource_policy")
        if not isinstance(policy, dict):
            errors.append(f"missing resource_policy for artifact={artifact}")
            continue
        for field_name, expected in expected_policy.fields():
            if expected is not None and policy.get(field_name) != expected:
                errors.append(
                    f"expected resource_policy.{field_name} {expected} for artifact={artifact}, "
                    f"found {policy.get(field_name, 'unknown')}"
                )
    return errors


def _validate_graph_descriptor(
    payloads: list[dict[str, Any]],
    *,
    expected_fanin: str | None,
    expected_dependents: str | None,
) -> list[str]:
    errors: list[str] = []
    if expected_fanin is None and expected_dependents is None:
        return errors
    for payload in payloads:
        artifact = payload.get("_artifact", "unknown")
        graph_descriptor = payload.get("graph_descriptor")
        if not isinstance(graph_descriptor, dict):
            errors.append(f"missing graph_descriptor for artifact={artifact}")
            continue
        for field_name, expected in (
            ("fanin", expected_fanin),
            ("dependents", expected_dependents),
        ):
            if expected is None:
                continue
            actual = _int_list(graph_descriptor, field_name)
            if actual != expected:
                errors.append(
                    f"expected graph_descriptor.{field_name} {expected} for artifact={artifact}, found {actual}"
                )
    return errors


def _graph_task_args(payload: dict[str, Any]) -> str | None:
    task_args = payload.get("graph_task_args")
    if not isinstance(task_args, dict):
        return None
    return ";".join(f"{key}={task_args[key]}" for key in sorted(task_args))


def _scratch_reuse(payload: dict[str, Any]) -> str | None:
    scratch_reuse = payload.get("scratch_reuse")
    if not isinstance(scratch_reuse, dict):
        return None
    keys = [key for key in ("reused_buffer", "reuse_task") if key in scratch_reuse]
    keys.extend(key for key in sorted(scratch_reuse) if key not in keys)
    return ",".join(f"{key}={scratch_reuse[key]}" for key in keys)


def _validate_graph_task_args(
    payloads: list[dict[str, Any]],
    *,
    expected_task_args: str | None,
) -> list[str]:
    errors: list[str] = []
    if expected_task_args is None:
        return errors
    for payload in payloads:
        artifact = payload.get("_artifact", "unknown")
        actual = _graph_task_args(payload)
        if actual != expected_task_args:
            errors.append(f"expected graph_task_args {expected_task_args} for artifact={artifact}, found {actual}")
    return errors


def _validate_graph_task_arg_key(
    payloads: list[dict[str, Any]],
    *,
    expected_key: str | None,
) -> list[str]:
    errors: list[str] = []
    if expected_key is None:
        return errors
    for payload in payloads:
        artifact = payload.get("_artifact", "unknown")
        actual = payload.get("graph_task_arg_key")
        if actual != expected_key:
            errors.append(f"expected graph_task_arg_key {expected_key} for artifact={artifact}, found {actual}")
    return errors


def _validate_scratch_reuse(
    payloads: list[dict[str, Any]],
    *,
    expected_scratch_reuse: str | None,
) -> list[str]:
    errors: list[str] = []
    if expected_scratch_reuse is None:
        return errors
    for payload in payloads:
        artifact = payload.get("_artifact", "unknown")
        actual = _scratch_reuse(payload)
        if actual != expected_scratch_reuse:
            errors.append(f"expected scratch_reuse {expected_scratch_reuse} for artifact={artifact}, found {actual}")
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


def _validate_report_graph_topology(
    artifact_dir: Path | None,
    *,
    expected_fanin: str | None,
    expected_dependents: str | None,
) -> list[str]:
    if artifact_dir is None:
        return ["missing artifact directory for report graph topology validation"]

    checks = {
        "cuda-smoke-report.md": [
            "Graph fan-in",
            "Graph dependents",
            f"`{expected_fanin}`" if expected_fanin is not None else None,
            f"`{expected_dependents}`" if expected_dependents is not None else None,
        ],
        "cuda-smoke-report.svg": [
            "graph:",
            f"fanin={expected_fanin}" if expected_fanin is not None else None,
            f"dependents={expected_dependents}" if expected_dependents is not None else None,
        ],
    }

    errors: list[str] = []
    for file_name, needles in checks.items():
        path = artifact_dir / file_name
        if not path.exists():
            errors.append(f"missing report graph topology in {file_name}")
            continue
        content = path.read_text()
        if any(needle is not None and needle not in content for needle in needles):
            errors.append(f"missing report graph topology in {file_name}")
    return errors


def _validate_report_graph_task_args(
    artifact_dir: Path | None,
    *,
    expected_key: str | None,
    expected_task_args: str | None,
) -> list[str]:
    if artifact_dir is None:
        return ["missing artifact directory for report graph task args validation"]

    checks = {
        "cuda-smoke-report.md": [
            "Graph task arg key" if expected_key is not None else None,
            "Graph task args",
            f"`{expected_key}`" if expected_key is not None else None,
            f"`{expected_task_args}`" if expected_task_args is not None else None,
        ],
        "cuda-smoke-report.svg": [
            f"task arg key: {expected_key}" if expected_key is not None else None,
            f"task args: {expected_task_args}" if expected_task_args is not None else None,
        ],
    }

    errors: list[str] = []
    for file_name, needles in checks.items():
        path = artifact_dir / file_name
        if not path.exists():
            errors.append(f"missing report graph task args in {file_name}")
            continue
        content = path.read_text()
        if any(needle is not None and needle not in content for needle in needles):
            errors.append(f"missing report graph task args in {file_name}")
    return errors


def validate_smoke(
    payloads: list[dict[str, Any]],
    *,
    expectation: SmokeValidationExpectation,
) -> list[str]:
    errors: list[str] = []
    if not payloads:
        errors.append("missing smoke payloads")
    errors.extend(_validate_required_artifacts(payloads, expectation.required_artifacts))
    errors.extend(_validate_status(payloads))
    errors.extend(_validate_scheduler_errors(payloads))
    errors.extend(
        _validate_common_fields(
            payloads,
            expected_runtime=expectation.runtime,
            expected_mode=expectation.mode,
            expected_dag_shape=expectation.dag_shape,
            expected_dispatch=expectation.dispatch,
            expected_tensor_tile=expectation.tensor_tile,
        )
    )
    errors.extend(
        _validate_resource_policy(
            payloads,
            expected_policy=expectation.resource_policy,
        )
    )
    errors.extend(
        _validate_graph_descriptor(
            payloads,
            expected_fanin=expectation.graph_fanin,
            expected_dependents=expectation.graph_dependents,
        )
    )
    errors.extend(_validate_graph_task_arg_key(payloads, expected_key=expectation.graph_task_arg_key))
    errors.extend(_validate_graph_task_args(payloads, expected_task_args=expectation.graph_task_args))
    errors.extend(_validate_scratch_reuse(payloads, expected_scratch_reuse=expectation.scratch_reuse))
    errors.extend(
        _validate_lifecycle(
            payloads,
            expected_repeat_runs=expectation.repeat_runs,
            expected_completed_count=expectation.completed_count,
        )
    )
    if expectation.require_report_files:
        errors.extend(_validate_report_files(expectation.artifact_dir))
    if expectation.require_report_graph_topology:
        errors.extend(
            _validate_report_graph_topology(
                expectation.artifact_dir,
                expected_fanin=expectation.graph_fanin,
                expected_dependents=expectation.graph_dependents,
            )
        )
    if expectation.require_report_graph_task_args:
        errors.extend(
            _validate_report_graph_task_args(
                expectation.artifact_dir,
                expected_key=expectation.graph_task_arg_key,
                expected_task_args=expectation.graph_task_args,
            )
        )
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
    parser.add_argument("--expected-tensor-tile")
    parser.add_argument("--expected-graph-fanin")
    parser.add_argument("--expected-graph-dependents")
    parser.add_argument("--expected-graph-task-arg-key")
    parser.add_argument("--expected-graph-task-args")
    parser.add_argument("--expected-scratch-reuse")
    parser.add_argument("--expected-scheduler-blocks", type=int)
    parser.add_argument("--expected-worker-blocks", type=int)
    parser.add_argument("--expected-worker-blocks-per-task", type=int)
    parser.add_argument("--expected-stream-id", type=int)
    parser.add_argument("--expected-block-dim", type=int)
    parser.add_argument("--expected-grid-dim", type=int)
    parser.add_argument("--require-report-files", action="store_true")
    parser.add_argument("--require-report-graph-topology", action="store_true")
    parser.add_argument("--require-report-graph-task-args", action="store_true")
    return parser.parse_args(argv)


def _resource_policy_expectation(args: argparse.Namespace) -> ResourcePolicyExpectation:
    return ResourcePolicyExpectation(
        scheduler_blocks=args.expected_scheduler_blocks,
        worker_blocks=args.expected_worker_blocks,
        worker_blocks_per_task=args.expected_worker_blocks_per_task,
        stream_id=args.expected_stream_id,
        block_dim=args.expected_block_dim,
        grid_dim=args.expected_grid_dim,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payloads = load_smoke_payloads(args.json_paths)
    artifact_dir = args.json_paths[0].parent if args.json_paths else None
    errors = validate_smoke(
        payloads,
        expectation=SmokeValidationExpectation(
            artifact_dir=artifact_dir,
            required_artifacts=_as_list(args.require_artifact),
            runtime=args.expected_runtime,
            mode=args.expected_mode,
            dag_shape=args.expected_dag_shape,
            repeat_runs=args.expected_repeat_runs,
            completed_count=args.expected_completed_count,
            dispatch=args.expected_dispatch,
            tensor_tile=args.expected_tensor_tile,
            graph_fanin=args.expected_graph_fanin,
            graph_dependents=args.expected_graph_dependents,
            graph_task_arg_key=args.expected_graph_task_arg_key,
            graph_task_args=args.expected_graph_task_args,
            scratch_reuse=args.expected_scratch_reuse,
            resource_policy=_resource_policy_expectation(args),
            require_report_files=args.require_report_files,
            require_report_graph_topology=args.require_report_graph_topology,
            require_report_graph_task_args=args.require_report_graph_task_args,
        ),
    )
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("validated " + ", ".join(str(path) for path in args.json_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
