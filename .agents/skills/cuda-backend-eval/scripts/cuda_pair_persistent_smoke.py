#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run paired local A100 / remote H200 CUDA persistent-device smoke captures."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

Runner = Callable[..., subprocess.CompletedProcess]


def _is_tensor_tile_shape(dag_shape: str) -> bool:
    return dag_shape in {"graph_tensor_tile", "tensor_core_tile", "tensor_tile"}


@dataclass(frozen=True)
class PairedPersistentSmokeConfig:
    remote: str = "bizhaoh200"
    remote_workdir: str = "/data/shibizhao/pto-cu"
    branch: str = "design/nvidia-backend"
    output_root: Path = Path("tmp/cuda-backend")
    local_device: int = 0
    remote_device: int = 0
    n: int = 1024
    mode: str = "dag"
    dag_shape: str = "fork_join"
    task_count: int = 3
    queue_capacity: int = 2
    worker_blocks_per_task: int = 1
    worker_blocks: int | None = None
    stream_id: int = 0
    repeat_runs: int = 1
    tensor_rows: int = 16
    tensor_cols: int = 16
    tensor_inner: int = 16
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
    validate_smoke: bool = True


def _git_commit(runner: Runner = subprocess.run) -> str:
    result = runner(["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _artifact_label(config: PairedPersistentSmokeConfig, suffix: str) -> str:
    repeat = f"-repeat{config.repeat_runs}" if config.repeat_runs > 1 else ""
    if config.mode != "dag":
        return f"persistent-{config.mode}{repeat}-smoke-{suffix}"
    if _is_tensor_tile_shape(config.dag_shape):
        return (
            f"persistent-{config.dag_shape}-{config.tensor_rows}x"
            f"{config.tensor_cols}x{config.tensor_inner}{repeat}-smoke-{suffix}"
        )
    return f"persistent-{config.dag_shape}{repeat}-smoke-{suffix}"


def _output_dir(config: PairedPersistentSmokeConfig, suffix: str) -> Path:
    return config.output_root / _artifact_label(config, suffix)


def build_remote_git_commit_command(config: PairedPersistentSmokeConfig) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={config.ssh_connect_timeout}",
        config.remote,
        f"cd {shlex.quote(config.remote_workdir)} && git rev-parse --short HEAD",
    ]


def build_remote_sync_command(config: PairedPersistentSmokeConfig) -> list[str]:
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


def _remote_git_commit(config: PairedPersistentSmokeConfig, runner: Runner = subprocess.run) -> str:
    result = runner(build_remote_git_commit_command(config), check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _smoke_args(*, device: int, arch: str, output_json: Path, config: PairedPersistentSmokeConfig) -> list[str]:
    args = [
        "--device",
        str(device),
        "--task-count",
        str(config.task_count),
        "--n",
        str(config.n),
        "--arch",
        arch,
        "--mode",
        config.mode,
        "--worker-blocks-per-task",
        str(config.worker_blocks_per_task),
        "--stream-id",
        str(config.stream_id),
        "--repeat-runs",
        str(config.repeat_runs),
        "--output-json",
        str(output_json),
    ]
    if config.worker_blocks is not None:
        args.extend(["--worker-blocks", str(config.worker_blocks)])
    if config.mode == "dag":
        args.extend(
            [
                "--queue-capacity",
                str(config.queue_capacity),
                "--dag-shape",
                config.dag_shape,
            ]
        )
        if _is_tensor_tile_shape(config.dag_shape):
            args.extend(
                [
                    "--tensor-rows",
                    str(config.tensor_rows),
                    "--tensor-cols",
                    str(config.tensor_cols),
                    "--tensor-inner",
                    str(config.tensor_inner),
                ]
            )
    elif config.mode == "queue":
        args.extend(["--queue-capacity", str(config.queue_capacity)])
    return args


def build_local_smoke_command(config: PairedPersistentSmokeConfig, suffix: str) -> list[str]:
    output_dir = _output_dir(config, suffix)
    return [
        "env",
        f"PYTHONPATH={Path.cwd()}:{Path.cwd() / 'python'}",
        config.local_python,
        ".agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py",
        *_smoke_args(
            device=config.local_device,
            arch=config.local_arch,
            output_json=output_dir / "a100.json",
            config=config,
        ),
    ]


def _remote_shell_command(config: PairedPersistentSmokeConfig, suffix: str) -> str:
    output_dir = _output_dir(config, suffix)
    smoke = [
        config.remote_python,
        ".agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py",
        *_smoke_args(
            device=config.remote_device,
            arch=config.remote_arch,
            output_json=output_dir / "h200.json",
            config=config,
        ),
    ]
    remote_env = "CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH PYTHONPATH=$PWD:$PWD/python"
    fetch_command = (
        f"timeout {config.remote_git_fetch_timeout} "
        "git "
        f"-c http.lowSpeedLimit={config.remote_git_low_speed_limit} "
        f"-c http.lowSpeedTime={config.remote_git_low_speed_time} "
        f"fetch origin {shlex.quote(config.branch)} >/dev/null"
    )
    commands = [f"cd {shlex.quote(config.remote_workdir)}"]
    if config.refresh_remote and not config.sync_remote_tree:
        commands.extend(
            [
                fetch_command,
                f"git checkout -B {shlex.quote(config.branch)} FETCH_HEAD >/dev/null",
            ]
        )
    commands.append(f"{remote_env} {' '.join(shlex.quote(part) for part in smoke)}")
    return " && ".join(commands)


def build_remote_smoke_command(config: PairedPersistentSmokeConfig, suffix: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={config.ssh_connect_timeout}",
        config.remote,
        _remote_shell_command(config, suffix),
    ]


def build_scp_command(config: PairedPersistentSmokeConfig, suffix: str) -> list[str]:
    output_dir = _output_dir(config, suffix)
    return [
        "scp",
        f"{config.remote}:{config.remote_workdir}/{output_dir / 'h200.json'}",
        str(output_dir / "h200.json"),
    ]


def build_report_command(config: PairedPersistentSmokeConfig, suffix: str) -> list[str]:
    output_dir = _output_dir(config, suffix)
    return [
        "env",
        f"PYTHONPATH={Path.cwd()}:{Path.cwd() / 'python'}",
        config.local_python,
        ".agents/skills/cuda-backend-eval/scripts/cuda_smoke_report.py",
        str(output_dir / "a100.json"),
        str(output_dir / "h200.json"),
        "--label",
        _artifact_label(config, suffix),
        "--output-dir",
        str(output_dir),
    ]


def _expected_completed_count(config: PairedPersistentSmokeConfig) -> int:
    if config.mode != "dag":
        return config.task_count
    return {
        "chain": 5,
        "graph_tensor_tile": 4,
        "scratch_reuse": 6,
        "tensor_core_tile": 4,
        "tensor_tile": 4,
    }.get(config.dag_shape, 3)


def _expected_dispatch(config: PairedPersistentSmokeConfig) -> str | None:
    if config.mode != "dag":
        return None
    return {
        "chain": "1,2,1,2,1",
        "fork_join": "1,2,1",
        "generic_args": "9,2,1",
        "graph_descriptor": "9,2,1",
        "graph_descriptor_reordered": "1,9,2",
        "graph_tensor_tile": "3,1,2,1",
        "quad": "8,2,1",
        "scalar_affine": "5,2,1",
        "scalar_axpy": "4,2,1",
        "scalar_scale": "11,2,1",
        "scratch_reuse": "1,2,1,2,1,1",
        "tensor_tile": "3,1,2,1",
        "tensor_core_tile": "10,1,2,1",
        "triad": "6,2,1",
        "unary_square": "7,1,1",
    }.get(config.dag_shape)


def _expected_tensor_tile(config: PairedPersistentSmokeConfig) -> str | None:
    if config.mode == "dag" and _is_tensor_tile_shape(config.dag_shape):
        return f"{config.tensor_rows}x{config.tensor_cols}x{config.tensor_inner}"
    return None


def _expected_scheduler_blocks(config: PairedPersistentSmokeConfig) -> int:
    if config.mode == "direct":
        return 0
    return 1


def _expected_worker_blocks(config: PairedPersistentSmokeConfig) -> int:
    if config.mode == "direct":
        return max(1, config.task_count * config.worker_blocks_per_task)
    if config.worker_blocks is not None:
        return config.worker_blocks
    return max(1, config.task_count)


def build_validate_command(config: PairedPersistentSmokeConfig, suffix: str) -> list[str]:
    output_dir = _output_dir(config, suffix)
    expected_scheduler_blocks = _expected_scheduler_blocks(config)
    expected_worker_blocks = _expected_worker_blocks(config)
    command = [
        "env",
        f"PYTHONPATH={Path.cwd()}:{Path.cwd() / 'python'}",
        config.local_python,
        ".agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py",
        str(output_dir / "a100.json"),
        str(output_dir / "h200.json"),
        "--require-artifact",
        "a100",
        "--require-artifact",
        "h200",
        "--expected-runtime",
        "persistent_device",
        "--expected-mode",
        config.mode,
        "--expected-repeat-runs",
        str(config.repeat_runs),
        "--expected-completed-count",
        str(_expected_completed_count(config)),
        "--expected-scheduler-blocks",
        str(expected_scheduler_blocks),
        "--expected-worker-blocks",
        str(expected_worker_blocks),
        "--expected-worker-blocks-per-task",
        str(config.worker_blocks_per_task),
        "--expected-stream-id",
        str(config.stream_id),
        "--expected-block-dim",
        "256",
        "--expected-grid-dim",
        str(expected_scheduler_blocks + expected_worker_blocks),
        "--require-report-files",
    ]
    if config.mode == "dag":
        command.extend(["--expected-dag-shape", config.dag_shape])
    expected_dispatch = _expected_dispatch(config)
    if expected_dispatch is not None:
        command.extend(["--expected-dispatch", expected_dispatch])
    expected_tensor_tile = _expected_tensor_tile(config)
    if expected_tensor_tile is not None:
        command.extend(["--expected-tensor-tile", expected_tensor_tile])
    return command


def build_index_command(config: PairedPersistentSmokeConfig) -> list[str]:
    return [
        "env",
        f"PYTHONPATH={Path.cwd()}:{Path.cwd() / 'python'}",
        config.local_python,
        ".agents/skills/cuda-backend-eval/scripts/cuda_artifact_index.py",
        "--root",
        str(config.output_root),
    ]


def run_paired_persistent_smoke(
    config: PairedPersistentSmokeConfig,
    *,
    runner: Runner = subprocess.run,
    dry_run: bool = False,
) -> list[list[str]]:
    local_commit = _git_commit(runner)
    remote_commit = local_commit
    if not config.refresh_remote and not config.sync_remote_tree:
        remote_commit = _remote_git_commit(config, runner)
    suffix = local_commit if remote_commit == local_commit else f"{local_commit}-{remote_commit}"

    commands = [build_local_smoke_command(config, suffix)]
    if config.sync_remote_tree:
        commands.append(build_remote_sync_command(config))
    commands.extend(
        [
            build_remote_smoke_command(config, suffix),
            build_scp_command(config, suffix),
            build_report_command(config, suffix),
        ]
    )
    if config.validate_smoke:
        commands.append(build_validate_command(config, suffix))
    commands.append(build_index_command(config))
    for command in commands:
        print(" ".join(shlex.quote(part) for part in command), flush=True)
        if not dry_run:
            runner(command, check=True)
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
    parser.add_argument("--mode", choices=("dag", "direct", "queue"), default="dag")
    parser.add_argument(
        "--dag-shape",
        choices=(
            "bad_dependent",
            "bad_dependent_range",
            "bad_fanin_underflow",
            "bad_func_id",
            "bad_initial_fanin",
            "bad_no_root",
            "chain",
            "fork_join",
            "generic_args",
            "graph_descriptor",
            "graph_descriptor_reordered",
            "graph_tensor_tile",
            "quad",
            "scalar_affine",
            "scalar_axpy",
            "scalar_scale",
            "scratch_reuse",
            "tensor_core_tile",
            "tensor_tile",
            "triad",
            "unary_square",
        ),
        default="fork_join",
    )
    parser.add_argument("--task-count", type=int, default=3)
    parser.add_argument("--queue-capacity", type=int, default=2)
    parser.add_argument("--worker-blocks-per-task", type=int, default=1)
    parser.add_argument("--worker-blocks", type=int, default=None)
    parser.add_argument("--stream-id", type=int, default=0)
    parser.add_argument("--repeat-runs", type=int, default=1)
    parser.add_argument("--tensor-rows", type=int, default=16)
    parser.add_argument("--tensor-cols", type=int, default=16)
    parser.add_argument("--tensor-inner", type=int, default=16)
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
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = PairedPersistentSmokeConfig(
        remote=args.remote,
        remote_workdir=args.remote_workdir,
        branch=args.branch,
        output_root=args.output_root,
        local_device=args.local_device,
        remote_device=args.remote_device,
        n=args.n,
        mode=args.mode,
        dag_shape=args.dag_shape,
        task_count=args.task_count,
        queue_capacity=args.queue_capacity,
        worker_blocks_per_task=args.worker_blocks_per_task,
        worker_blocks=args.worker_blocks,
        stream_id=args.stream_id,
        repeat_runs=args.repeat_runs,
        tensor_rows=args.tensor_rows,
        tensor_cols=args.tensor_cols,
        tensor_inner=args.tensor_inner,
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
        validate_smoke=not args.skip_validation,
    )
    run_paired_persistent_smoke(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
