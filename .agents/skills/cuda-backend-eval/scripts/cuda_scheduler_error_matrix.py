#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Capture paired CUDA persistent-device scheduler-error diagnostics."""

from __future__ import annotations

import argparse
import html
import json
import re
import shlex
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cuda_scheduler_errors import scheduler_error_code_label

Runner = Callable[..., subprocess.CompletedProcess]
_ERROR_RE = re.compile(r"persistent dag scheduler error code=(\d+) task_id=(\d+) count=(\d+)")
_SMOKE_SCRIPT = ".agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py"
SOURCE_PAPERS = (
    {
        "id": "arXiv:2605.03190",
        "title": "VDCores",
        "path": "tmp/sources/arxiv-2605.03190-vdcores.pdf",
    },
    {
        "id": "arXiv:2512.22219v1",
        "title": "Mirage persistent-kernel evaluation",
        "path": "tmp/sources/arxiv-2512.22219v1-mirage-persistent-kernel.pdf",
    },
)


@dataclass(frozen=True)
class SchedulerErrorCase:
    name: str
    dag_shape: str
    expected_code: int
    expected_task_id: int
    task_count: int
    queue_capacity: int
    worker_blocks: int | None = None


@dataclass(frozen=True)
class SchedulerErrorMatrixConfig:
    output_root: Path = Path("tmp/cuda-backend")
    local_device: int = 0
    remote_device: int = 0
    n: int = 1024
    local_arch: str = "compute_80"
    remote_arch: str = "compute_90"
    local_python: str = ".venv/bin/python"
    remote_python: str = ".venv/bin/python"
    remote: str = "bizhaoh200"
    remote_workdir: str = "/data/shibizhao/pto-cu"
    ssh_connect_timeout: int = 8
    sync_remote_tree: bool = False
    cases: Sequence[SchedulerErrorCase] = ()


DEFAULT_SCHEDULER_ERROR_CASES = (
    SchedulerErrorCase("invalid-dispatch", "bad_func_id", 1, 0, 1, 1),
    SchedulerErrorCase("invalid-dependent", "bad_dependent", 2, 7, 1, 1),
    SchedulerErrorCase("dependent-range", "bad_dependent_range", 3, 0, 1, 1),
    SchedulerErrorCase("fanin-underflow", "bad_fanin_underflow", 4, 2, 3, 2),
    SchedulerErrorCase("duplicate-dependent", "bad_duplicate_dependent", 8, 1, 2, 2),
    SchedulerErrorCase("self-dependent", "bad_self_dependent", 9, 0, 1, 1),
    SchedulerErrorCase("initial-fanin", "bad_initial_fanin", 5, 0, 1, 1),
    SchedulerErrorCase("no-root", "bad_no_root", 6, 0, 1, 1),
    SchedulerErrorCase("unreachable", "bad_unreachable", 7, 1, 2, 1, worker_blocks=2),
)
SCHEDULER_ERROR_CASES_BY_NAME = {case.name: case for case in DEFAULT_SCHEDULER_ERROR_CASES}


def parse_scheduler_error(text: str) -> dict[str, int] | None:
    match = _ERROR_RE.search(text)
    if match is None:
        return None
    code, task_id, count = match.groups()
    return {"code": int(code), "task_id": int(task_id), "count": int(count)}


def _git_commit(runner: Runner) -> str:
    result = runner(["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True)
    return str(result.stdout).strip()


def _artifact_label(commit: str) -> str:
    return f"scheduler-error-matrix-{commit}"


def build_sync_command(config: SchedulerErrorMatrixConfig) -> list[str]:
    return [
        "rsync",
        "-a",
        "--delete",
        "--exclude=.venv",
        "--exclude=build",
        "--exclude=tmp",
        "--exclude=__pycache__",
        "--exclude=.pytest_cache",
        f"{Path.cwd()}/",
        f"{config.remote}:{config.remote_workdir}/",
    ]


def _base_smoke_args(
    *,
    python: str,
    device: int,
    n: int,
    arch: str,
    case: SchedulerErrorCase,
) -> list[str]:
    command = [
        python,
        _SMOKE_SCRIPT,
        "--device",
        str(device),
        "--task-count",
        str(case.task_count),
        "--n",
        str(n),
        "--arch",
        arch,
        "--mode",
        "dag",
        "--queue-capacity",
        str(case.queue_capacity),
        "--dag-shape",
        case.dag_shape,
    ]
    if case.worker_blocks is not None:
        command.extend(["--worker-blocks", str(case.worker_blocks)])
    return command


def build_local_smoke_command(config: SchedulerErrorMatrixConfig, case: SchedulerErrorCase) -> list[str]:
    return _base_smoke_args(
        python=config.local_python,
        device=config.local_device,
        n=config.n,
        arch=config.local_arch,
        case=case,
    )


def build_remote_smoke_command(config: SchedulerErrorMatrixConfig, case: SchedulerErrorCase) -> list[str]:
    remote_args = _base_smoke_args(
        python=config.remote_python,
        device=config.remote_device,
        n=config.n,
        arch=config.remote_arch,
        case=case,
    )
    remote_command = (
        f"cd {shlex.quote(config.remote_workdir)} && "
        "CUDA_HOME=/usr/local/cuda-12.8 "
        "PATH=/usr/local/cuda-12.8/bin:$PATH "
        "PYTHONPATH=$PWD:$PWD/python " + " ".join(shlex.quote(part) for part in remote_args)
    )
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={config.ssh_connect_timeout}",
        config.remote,
        remote_command,
    ]


def _run_case(
    command: list[str],
    *,
    machine: str,
    arch: str,
    case: SchedulerErrorCase,
    runner: Runner,
) -> dict[str, Any]:
    result = runner(command, check=False, capture_output=True, text=True)
    stdout = str(result.stdout or "")
    stderr = str(result.stderr or "")
    parsed = parse_scheduler_error(stdout + "\n" + stderr)
    observed_code = parsed["code"] if parsed else None
    observed_task_id = parsed["task_id"] if parsed else None
    observed_count = parsed["count"] if parsed else None
    status = (
        "pass"
        if result.returncode != 0
        and observed_code == case.expected_code
        and observed_task_id == case.expected_task_id
        and isinstance(observed_count, int)
        and observed_count > 0
        else "fail"
    )
    return {
        "machine": machine,
        "arch": arch,
        "case": case.name,
        "dag_shape": case.dag_shape,
        "expected_code": case.expected_code,
        "expected_task_id": case.expected_task_id,
        "observed_code": observed_code,
        "observed_task_id": observed_task_id,
        "observed_count": observed_count,
        "status": status,
        "returncode": result.returncode,
        "command": command,
        "stdout": stdout,
        "stderr": stderr,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata", {})
    lines = [
        "# CUDA Scheduler Error Matrix",
        "",
        f"- Label: `{metadata.get('label', 'unknown')}`",
        f"- Git commit: `{metadata.get('git_commit', 'unknown')}`",
        f"- Paper setup: {metadata.get('paper_setup', 'unknown')}",
        "",
        "| Machine | Case | DAG shape | Status | Expected | Observed |",
        "| ------- | ---- | --------- | ------ | -------- | -------- |",
    ]
    for row in payload.get("results", []):
        expected = f"code={scheduler_error_code_label(row.get('expected_code'))},task={row.get('expected_task_id')}"
        observed = (
            f"code={scheduler_error_code_label(row.get('observed_code'))},"
            f"task={row.get('observed_task_id')},"
            f"count={row.get('observed_count')}"
        )
        lines.append(
            f"| {row.get('machine')} | {row.get('case')} | {row.get('dag_shape')} | "
            f"{row.get('status')} | {expected} | {observed} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_svg(payload: dict[str, Any]) -> str:
    rows = list(payload.get("results", []))
    width = 1120
    row_height = 24
    height = max(160, 92 + row_height * len(rows))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="34" font-family="monospace" font-size="20" fill="#111827">CUDA scheduler error matrix</text>',
        '<text x="24" y="60" font-family="monospace" font-size="13" fill="#4b5563">'
        "Machine / case / expected / observed</text>",
    ]
    y = 92
    for row in rows:
        color = "#dcfce7" if row.get("status") == "pass" else "#fee2e2"
        observed = scheduler_error_code_label(row.get("observed_code"))
        expected = scheduler_error_code_label(row.get("expected_code"))
        text = (
            f"{row.get('machine')}  {row.get('case')}  {row.get('dag_shape')}  "
            f"{row.get('status')}  expected={expected}/task={row.get('expected_task_id')}  "
            f"observed={observed}/task={row.get('observed_task_id')}/count={row.get('observed_count')}"
        )
        lines.extend(
            [
                f'<rect x="20" y="{y - 16}" width="{width - 40}" height="22" rx="3" fill="{color}"/>',
                f'<text x="28" y="{y}" font-family="monospace" font-size="12" '
                f'fill="#111827">{html.escape(text)}</text>',
            ]
        )
        y += row_height
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_report(payload: dict[str, Any], output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "cuda-scheduler-error-matrix.json"
    markdown_path = output_dir / "cuda-scheduler-error-matrix.md"
    svg_path = output_dir / "cuda-scheduler-error-matrix.svg"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    markdown_path.write_text(render_markdown(payload))
    svg_path.write_text(render_svg(payload))
    return json_path, markdown_path, svg_path


def run_scheduler_error_matrix(
    config: SchedulerErrorMatrixConfig,
    *,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    cases = tuple(config.cases) if config.cases else DEFAULT_SCHEDULER_ERROR_CASES
    commit = _git_commit(runner)
    label = _artifact_label(commit)
    if config.sync_remote_tree:
        runner(build_sync_command(config), check=True)
    results = []
    local_example: list[str] | None = None
    remote_example: list[str] | None = None
    for case in cases:
        local_command = build_local_smoke_command(config, case)
        remote_command = build_remote_smoke_command(config, case)
        local_example = local_example or local_command
        remote_example = remote_example or remote_command
        results.append(_run_case(local_command, machine="a100", arch=config.local_arch, case=case, runner=runner))
        results.append(_run_case(remote_command, machine="h200", arch=config.remote_arch, case=case, runner=runner))
    payload = {
        "metadata": {
            "label": label,
            "git_commit": commit,
            "paper_setup": "VDCores/MPK persistent-kernel scheduler diagnostics on paired A100/H200.",
            "source_papers": list(SOURCE_PAPERS),
            "command_examples": {
                "local_sample": " ".join(local_example or []),
                "remote_sample": " ".join(remote_example or []),
            },
        },
        "results": results,
    }
    write_report(payload, config.output_root / label)
    return payload


def _parse_cases(case_names: Sequence[str]) -> tuple[SchedulerErrorCase, ...]:
    if not case_names:
        return DEFAULT_SCHEDULER_ERROR_CASES
    missing = [name for name in case_names if name not in SCHEDULER_ERROR_CASES_BY_NAME]
    if missing:
        raise ValueError(f"unknown scheduler error case(s): {', '.join(missing)}")
    return tuple(SCHEDULER_ERROR_CASES_BY_NAME[name] for name in case_names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("tmp/cuda-backend"))
    parser.add_argument("--local-device", type=int, default=0)
    parser.add_argument("--remote-device", type=int, default=0)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--local-arch", default="compute_80")
    parser.add_argument("--remote-arch", default="compute_90")
    parser.add_argument("--local-python", default=".venv/bin/python")
    parser.add_argument("--remote-python", default=".venv/bin/python")
    parser.add_argument("--remote", default="bizhaoh200")
    parser.add_argument("--remote-workdir", default="/data/shibizhao/pto-cu")
    parser.add_argument("--sync-remote-tree", action="store_true")
    parser.add_argument("--case", action="append", choices=sorted(SCHEDULER_ERROR_CASES_BY_NAME))
    args = parser.parse_args()

    config = SchedulerErrorMatrixConfig(
        output_root=args.output_root,
        local_device=args.local_device,
        remote_device=args.remote_device,
        n=args.n,
        local_arch=args.local_arch,
        remote_arch=args.remote_arch,
        local_python=args.local_python,
        remote_python=args.remote_python,
        remote=args.remote,
        remote_workdir=args.remote_workdir,
        sync_remote_tree=args.sync_remote_tree,
        cases=_parse_cases(args.case or []),
    )
    payload = run_scheduler_error_matrix(config)
    label = payload["metadata"]["label"]
    print(config.output_root / label / "cuda-scheduler-error-matrix.json")
    print(config.output_root / label / "cuda-scheduler-error-matrix.md")
    print(config.output_root / label / "cuda-scheduler-error-matrix.svg")


if __name__ == "__main__":
    main()
