#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run a paired A100/H200 lifecycle matrix for CUDA persistent-device smokes."""

from __future__ import annotations

import argparse
import html
import json
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cuda_pair_persistent_smoke as paired_smoke

Runner = Callable[..., subprocess.CompletedProcess]


@dataclass(frozen=True)
class LifecycleScenario:
    name: str
    mode: str
    dag_shape: str = "fork_join"
    task_count: int = 3
    queue_capacity: int = 2
    worker_blocks_per_task: int = 1
    worker_blocks: int | None = None


SCENARIOS: dict[str, LifecycleScenario] = {
    "direct": LifecycleScenario(
        name="direct",
        mode="direct",
        task_count=2,
        worker_blocks_per_task=2,
    ),
    "queue": LifecycleScenario(
        name="queue",
        mode="queue",
        task_count=4,
        queue_capacity=2,
        worker_blocks=2,
    ),
    "dag-chain": LifecycleScenario(
        name="dag-chain",
        mode="dag",
        dag_shape="chain",
        task_count=5,
        queue_capacity=3,
        worker_blocks=2,
    ),
}


@dataclass(frozen=True)
class LifecycleMatrixConfig:
    remote: str = "bizhaoh200"
    remote_workdir: str = "/data/shibizhao/pto-cu"
    branch: str = "design/nvidia-backend"
    output_root: Path = Path("tmp/cuda-backend")
    local_device: int = 0
    remote_device: int = 0
    n: int = 1024
    repeat_runs: int = 2
    stream_id: int = 1
    scenario_names: tuple[str, ...] = ("direct", "queue", "dag-chain")
    local_arch: str = "compute_80"
    remote_arch: str = "compute_90"
    local_python: str = sys.executable
    remote_python: str = ".venv/bin/python"
    ssh_connect_timeout: int = 8
    remote_git_low_speed_limit: int = 1
    remote_git_low_speed_time: int = 30
    remote_git_fetch_timeout: int = 60
    refresh_remote: bool = True
    sync_remote_tree: bool = False


def _scenario_names(raw: Sequence[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ("direct", "queue", "dag-chain")
    names: list[str] = []
    for value in raw:
        names.extend(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted({name for name in names if name not in SCENARIOS})
    if unknown:
        raise ValueError(f"unknown lifecycle scenario(s): {', '.join(unknown)}")
    return tuple(names)


def _scenario_config(
    config: LifecycleMatrixConfig,
    scenario: LifecycleScenario,
    *,
    sync_remote_tree: bool,
    refresh_remote: bool,
) -> paired_smoke.PairedPersistentSmokeConfig:
    return paired_smoke.PairedPersistentSmokeConfig(
        remote=config.remote,
        remote_workdir=config.remote_workdir,
        branch=config.branch,
        output_root=config.output_root,
        local_device=config.local_device,
        remote_device=config.remote_device,
        n=config.n,
        mode=scenario.mode,
        dag_shape=scenario.dag_shape,
        task_count=scenario.task_count,
        queue_capacity=scenario.queue_capacity,
        worker_blocks_per_task=scenario.worker_blocks_per_task,
        worker_blocks=scenario.worker_blocks,
        stream_id=config.stream_id,
        repeat_runs=config.repeat_runs,
        local_arch=config.local_arch,
        remote_arch=config.remote_arch,
        local_python=config.local_python,
        remote_python=config.remote_python,
        ssh_connect_timeout=config.ssh_connect_timeout,
        remote_git_low_speed_limit=config.remote_git_low_speed_limit,
        remote_git_low_speed_time=config.remote_git_low_speed_time,
        remote_git_fetch_timeout=config.remote_git_fetch_timeout,
        refresh_remote=refresh_remote,
        sync_remote_tree=sync_remote_tree,
        validate_smoke=True,
    )


def _matrix_suffix(config: LifecycleMatrixConfig, runner: Runner) -> str:
    local_commit = paired_smoke._git_commit(runner)
    remote_commit = local_commit
    if not config.refresh_remote and not config.sync_remote_tree:
        probe_config = paired_smoke.PairedPersistentSmokeConfig(
            remote=config.remote,
            remote_workdir=config.remote_workdir,
            ssh_connect_timeout=config.ssh_connect_timeout,
        )
        remote_commit = paired_smoke._remote_git_commit(probe_config, runner)
    return local_commit if local_commit == remote_commit else f"{local_commit}-{remote_commit}"


def _load_json(path: Path, scenario: LifecycleScenario, artifact: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    payload["scenario"] = scenario.name
    payload["artifact"] = artifact
    return payload


def collect_lifecycle_rows(
    config: LifecycleMatrixConfig,
    suffix: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_name in config.scenario_names:
        scenario = SCENARIOS[scenario_name]
        paired_config = _scenario_config(
            config,
            scenario,
            sync_remote_tree=False,
            refresh_remote=config.refresh_remote,
        )
        output_dir = paired_smoke._output_dir(paired_config, suffix)
        rows.append(_load_json(output_dir / "a100.json", scenario, "a100"))
        rows.append(_load_json(output_dir / "h200.json", scenario, "h200"))
    return rows


def _mode(row: dict[str, Any]) -> str:
    mode = str(row.get("mode", "unknown"))
    dag_shape = row.get("dag_shape")
    if dag_shape:
        return f"{mode}/{dag_shape}"
    return mode


def _dispatch(row: dict[str, Any]) -> str:
    dispatch = row.get("dispatch_func_ids")
    if not isinstance(dispatch, list) or not dispatch:
        return "-"
    return ",".join(str(item) for item in dispatch)


def _scheduler_errors(row: dict[str, Any]) -> str:
    errors = row.get("device_scheduler_errors")
    if not isinstance(errors, dict):
        return "-"
    return f"count={errors.get('count', 0)},code={errors.get('code', 0)},task={errors.get('task_id', 0)}"


def _completed(row: dict[str, Any]) -> str:
    counts = row.get("launch_completed_counts")
    if not isinstance(counts, list):
        return "-"
    return ",".join(str(count) for count in counts)


def _policy(row: dict[str, Any]) -> str:
    policy = row.get("resource_policy")
    if not isinstance(policy, dict):
        return "-"
    return (
        f"sched={policy.get('scheduler_blocks', '-')},"
        f"workers={policy.get('worker_blocks', '-')},"
        f"wp={policy.get('worker_blocks_per_task', '-')},"
        f"stream={policy.get('stream_id', '-')},"
        f"block={policy.get('block_dim', '-')},"
        f"grid={policy.get('grid_dim', '-')}"
    )


def render_lifecycle_markdown(rows: list[dict[str, Any]], label: str) -> str:
    lines = [
        "# CUDA Persistent Lifecycle Matrix",
        "",
        f"- Label: `{label}`",
        "",
        (
            "| Scenario | Artifact | Status | Runtime | Mode | N | Device ns | "
            "Host ns | Repeat runs | Completions | Dispatch | Scheduler errors | "
            "Resource policy |"
        ),
        (
            "| -------- | -------- | ------ | ------- | ---- | - | --------- | "
            "------- | ----------- | ----------- | -------- | ---------------- | "
            "--------------- |"
        ),
    ]
    for row in rows:
        lines.append(
            f"| {row.get('scenario', '-')} | {row.get('artifact', '-')} | "
            f"{row.get('status', 'unknown')} | {row.get('runtime', 'unknown')} | "
            f"{_mode(row)} | {row.get('n', '-')} | {row.get('device_wall_ns', '-')} | "
            f"{row.get('host_wall_ns', '-')} | `{row.get('repeat_runs', '-')}` | "
            f"`{_completed(row)}` | `{_dispatch(row)}` | `{_scheduler_errors(row)}` | "
            f"`{_policy(row)}` |"
        )
    lines.append("")
    return "\n".join(lines)


def render_lifecycle_svg(rows: list[dict[str, Any]], label: str) -> str:
    width = 820
    left = 190
    right = 40
    top = 70
    row_height = 64
    bar_height = 24
    chart_width = width - left - right
    max_value = max((int(row.get("device_wall_ns", 0) or 0) for row in rows), default=1)
    height = top + max(1, len(rows)) * row_height + 40
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="34" font-family="sans-serif" font-size="20" font-weight="600">{html.escape(label)}</text>',
        (
            '<text x="24" y="55" font-family="sans-serif" font-size="12" fill="#555">'
            "Persistent-device repeat-run lifecycle smoke matrix</text>"
        ),
    ]
    for index, row in enumerate(rows):
        y = top + index * row_height
        value = int(row.get("device_wall_ns", 0) or 0)
        bar_width = int(chart_width * value / max_value) if max_value else 0
        name = f"{row.get('scenario', '-')}/{row.get('artifact', '-')}"
        lines.extend(
            [
                f'<text x="24" y="{y + 17}" font-family="sans-serif" font-size="12">{html.escape(name)}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_width}" height="{bar_height}" fill="#0f766e"/>',
                (
                    f'<text x="{left + bar_width + 8}" y="{y + 17}" '
                    f'font-family="sans-serif" font-size="12">{value} ns</text>'
                ),
                (
                    f'<text x="{left}" y="{y + 40}" font-family="sans-serif" font-size="11" fill="#555">'
                    f"completed={html.escape(_completed(row))}; policy={html.escape(_policy(row))}</text>"
                ),
            ]
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_lifecycle_report(
    rows: list[dict[str, Any]],
    output_dir: Path,
    label: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "cuda-lifecycle-matrix.md"
    svg_path = output_dir / "cuda-lifecycle-matrix.svg"
    json_path = output_dir / "cuda-lifecycle-matrix.json"
    json_path.write_text(json.dumps({"label": label, "rows": rows}, indent=2) + "\n")
    markdown_path.write_text(render_lifecycle_markdown(rows, label))
    svg_path.write_text(render_lifecycle_svg(rows, label))
    return markdown_path, svg_path


def build_validate_command(config: LifecycleMatrixConfig, suffix: str) -> list[str]:
    output_dir = config.output_root / f"persistent-lifecycle-matrix-{suffix}"
    return [
        config.local_python,
        ".agents/skills/cuda-backend-eval/scripts/cuda_validate_lifecycle_matrix.py",
        str(output_dir / "cuda-lifecycle-matrix.json"),
        "--preset",
        "default",
    ]


def run_lifecycle_matrix(
    config: LifecycleMatrixConfig,
    *,
    runner: Runner = subprocess.run,
    dry_run: bool = False,
) -> list[list[str]]:
    suffix = _matrix_suffix(config, runner)
    commands: list[list[str]] = []
    for index, scenario_name in enumerate(config.scenario_names):
        scenario = SCENARIOS[scenario_name]
        sync_remote_tree = config.sync_remote_tree and index == 0
        refresh_remote = config.refresh_remote and not config.sync_remote_tree
        paired_config = _scenario_config(
            config,
            scenario,
            sync_remote_tree=sync_remote_tree,
            refresh_remote=refresh_remote,
        )
        commands.extend(paired_smoke.run_paired_persistent_smoke(paired_config, runner=runner, dry_run=dry_run))
    output_dir = config.output_root / f"persistent-lifecycle-matrix-{suffix}"
    validate_command = build_validate_command(config, suffix)
    index_command = paired_smoke.build_index_command(
        paired_smoke.PairedPersistentSmokeConfig(
            output_root=config.output_root,
            local_python=config.local_python,
        )
    )
    if not dry_run:
        rows = collect_lifecycle_rows(config, suffix)
        markdown_path, svg_path = write_lifecycle_report(rows, output_dir, f"persistent-lifecycle-matrix-{suffix}")
        print(markdown_path)
        print(svg_path)
    for command in (validate_command, index_command):
        print(" ".join(shlex.quote(part) for part in command), flush=True)
        if not dry_run:
            runner(command, check=True)
        commands.append(command)
    return commands


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", default="bizhaoh200")
    parser.add_argument("--remote-workdir", default="/data/shibizhao/pto-cu")
    parser.add_argument("--branch", default="design/nvidia-backend")
    parser.add_argument("--output-root", type=Path, default=Path("tmp/cuda-backend"))
    parser.add_argument("--local-device", type=int, default=0)
    parser.add_argument("--remote-device", type=int, default=0)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--repeat-runs", type=int, default=2)
    parser.add_argument("--stream-id", type=int, default=1)
    parser.add_argument("--scenario", action="append")
    parser.add_argument("--local-arch", default="compute_80")
    parser.add_argument("--remote-arch", default="compute_90")
    parser.add_argument("--local-python", default=sys.executable)
    parser.add_argument("--remote-python", default=".venv/bin/python")
    parser.add_argument("--ssh-connect-timeout", type=int, default=8)
    parser.add_argument("--remote-git-low-speed-limit", type=int, default=1)
    parser.add_argument("--remote-git-low-speed-time", type=int, default=30)
    parser.add_argument("--remote-git-fetch-timeout", type=int, default=60)
    parser.add_argument("--skip-remote-refresh", action="store_true")
    parser.add_argument("--sync-remote-tree", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = LifecycleMatrixConfig(
        remote=args.remote,
        remote_workdir=args.remote_workdir,
        branch=args.branch,
        output_root=args.output_root,
        local_device=args.local_device,
        remote_device=args.remote_device,
        n=args.n,
        repeat_runs=args.repeat_runs,
        stream_id=args.stream_id,
        scenario_names=_scenario_names(args.scenario),
        local_arch=args.local_arch,
        remote_arch=args.remote_arch,
        local_python=args.local_python,
        remote_python=args.remote_python,
        ssh_connect_timeout=args.ssh_connect_timeout,
        remote_git_low_speed_limit=args.remote_git_low_speed_limit,
        remote_git_low_speed_time=args.remote_git_low_speed_time,
        remote_git_fetch_timeout=args.remote_git_fetch_timeout,
        refresh_remote=not args.skip_remote_refresh and not args.sync_remote_tree,
        sync_remote_tree=args.sync_remote_tree,
    )
    run_lifecycle_matrix(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
