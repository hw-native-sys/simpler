#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the paired local A100 / remote H200 CUDA benchmark workflow."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

Runner = Callable[..., subprocess.CompletedProcess]


@dataclass(frozen=True)
class PairedBenchmarkConfig:
    remote: str = "bizhaoh200"
    remote_workdir: str = "/data/shibizhao/pto-cu"
    branch: str = "design/nvidia-backend"
    output_root: Path = Path("tmp/cuda-backend")
    local_device: int = 0
    remote_device: int = 0
    sizes: tuple[int, ...] = (1024, 65536, 1048576)
    repeats: int = 3
    local_arch: str = "compute_80"
    remote_arch: str = "compute_90"
    batch_tasks: tuple[int, ...] = (2, 6, 12)
    worker_blocks_per_task: tuple[int, ...] = (32, 64, 128, 256)
    tensor_rows: int = 8
    tensor_cols: int = 4
    tensor_inner: int = 12
    local_python: str = sys.executable
    remote_python: str = ".venv/bin/python"
    ssh_connect_timeout: int = 8
    remote_git_low_speed_limit: int = 1
    remote_git_low_speed_time: int = 30
    remote_git_fetch_timeout: int = 60


def _csv(values: Sequence[int]) -> str:
    return ",".join(str(value) for value in values)


def _git_commit(runner: Runner = subprocess.run) -> str:
    result = runner(["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _benchmark_args(
    *,
    device: int,
    arch: str,
    label: str,
    output_dir: Path,
    config: PairedBenchmarkConfig,
) -> list[str]:
    return [
        "--device",
        str(device),
        "--sizes",
        _csv(config.sizes),
        "--repeats",
        str(config.repeats),
        "--arch",
        arch,
        "--include-persistent",
        "--batch-tasks",
        _csv(config.batch_tasks),
        "--worker-blocks-per-task",
        _csv(config.worker_blocks_per_task),
        "--tensor-rows",
        str(config.tensor_rows),
        "--tensor-cols",
        str(config.tensor_cols),
        "--tensor-inner",
        str(config.tensor_inner),
        "--label",
        label,
        "--output-dir",
        str(output_dir),
    ]


def build_local_benchmark_command(config: PairedBenchmarkConfig, commit: str) -> list[str]:
    label = f"a100-current-{commit}"
    output_dir = config.output_root / label
    return [
        "env",
        f"PYTHONPATH={Path.cwd()}:{Path.cwd() / 'python'}",
        config.local_python,
        ".agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py",
        *_benchmark_args(
            device=config.local_device,
            arch=config.local_arch,
            label=label,
            output_dir=output_dir,
            config=config,
        ),
    ]


def _remote_shell_command(config: PairedBenchmarkConfig, commit: str) -> str:
    label = f"h200-current-{commit}"
    output_dir = config.output_root / label
    benchmark = [
        config.remote_python,
        ".agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py",
        *_benchmark_args(
            device=config.remote_device,
            arch=config.remote_arch,
            label=label,
            output_dir=output_dir,
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
    commands = [
        f"cd {shlex.quote(config.remote_workdir)}",
        fetch_command,
        f"git checkout -B {shlex.quote(config.branch)} FETCH_HEAD >/dev/null",
        f"{remote_env} {' '.join(shlex.quote(part) for part in benchmark)}",
    ]
    return " && ".join(commands)


def build_remote_benchmark_command(config: PairedBenchmarkConfig, commit: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={config.ssh_connect_timeout}",
        config.remote,
        _remote_shell_command(config, commit),
    ]


def build_scp_command(config: PairedBenchmarkConfig, commit: str) -> list[str]:
    label = f"h200-current-{commit}"
    return [
        "scp",
        "-r",
        f"{config.remote}:{config.remote_workdir}/{config.output_root / label}",
        str(config.output_root),
    ]


def build_merge_command(config: PairedBenchmarkConfig, commit: str) -> list[str]:
    return [
        "env",
        f"PYTHONPATH={Path.cwd()}:{Path.cwd() / 'python'}",
        config.local_python,
        ".agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py",
        "--merge-json",
        str(config.output_root / f"a100-current-{commit}" / "cuda-benchmark.json"),
        str(config.output_root / f"h200-current-{commit}" / "cuda-benchmark.json"),
        "--label",
        f"combined-current-{commit}",
        "--output-dir",
        str(config.output_root / f"combined-current-{commit}"),
    ]


def build_index_command(config: PairedBenchmarkConfig) -> list[str]:
    return [
        "env",
        f"PYTHONPATH={Path.cwd()}:{Path.cwd() / 'python'}",
        config.local_python,
        ".agents/skills/cuda-backend-eval/scripts/cuda_artifact_index.py",
        "--root",
        str(config.output_root),
    ]


def run_paired_benchmark(
    config: PairedBenchmarkConfig,
    *,
    runner: Runner = subprocess.run,
    dry_run: bool = False,
) -> list[list[str]]:
    commit = _git_commit(runner)
    commands = [
        build_local_benchmark_command(config, commit),
        build_remote_benchmark_command(config, commit),
        build_scp_command(config, commit),
        build_merge_command(config, commit),
        build_index_command(config),
    ]
    for command in commands:
        print(" ".join(shlex.quote(part) for part in command))
        if not dry_run:
            runner(command, check=True)
    return commands


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", default="bizhaoh200")
    parser.add_argument("--remote-workdir", default="/data/shibizhao/pto-cu")
    parser.add_argument("--branch", default="design/nvidia-backend")
    parser.add_argument("--output-root", type=Path, default=Path("tmp/cuda-backend"))
    parser.add_argument("--local-device", type=int, default=0)
    parser.add_argument("--remote-device", type=int, default=0)
    parser.add_argument("--sizes", type=_parse_int_tuple, default=(1024, 65536, 1048576))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--local-arch", default="compute_80")
    parser.add_argument("--remote-arch", default="compute_90")
    parser.add_argument("--batch-tasks", type=_parse_int_tuple, default=(2, 6, 12))
    parser.add_argument("--worker-blocks-per-task", type=_parse_int_tuple, default=(32, 64, 128, 256))
    parser.add_argument("--tensor-rows", type=int, default=8)
    parser.add_argument("--tensor-cols", type=int, default=4)
    parser.add_argument("--tensor-inner", type=int, default=12)
    parser.add_argument("--local-python", default=sys.executable)
    parser.add_argument("--remote-python", default=".venv/bin/python")
    parser.add_argument("--ssh-connect-timeout", type=int, default=8)
    parser.add_argument("--remote-git-low-speed-limit", type=int, default=1)
    parser.add_argument("--remote-git-low-speed-time", type=int, default=30)
    parser.add_argument("--remote-git-fetch-timeout", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = PairedBenchmarkConfig(
        remote=args.remote,
        remote_workdir=args.remote_workdir,
        branch=args.branch,
        output_root=args.output_root,
        local_device=args.local_device,
        remote_device=args.remote_device,
        sizes=args.sizes,
        repeats=args.repeats,
        local_arch=args.local_arch,
        remote_arch=args.remote_arch,
        batch_tasks=args.batch_tasks,
        worker_blocks_per_task=args.worker_blocks_per_task,
        tensor_rows=args.tensor_rows,
        tensor_cols=args.tensor_cols,
        tensor_inner=args.tensor_inner,
        local_python=args.local_python,
        remote_python=args.remote_python,
        ssh_connect_timeout=args.ssh_connect_timeout,
        remote_git_low_speed_limit=args.remote_git_low_speed_limit,
        remote_git_low_speed_time=args.remote_git_low_speed_time,
        remote_git_fetch_timeout=args.remote_git_fetch_timeout,
    )
    run_paired_benchmark(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
