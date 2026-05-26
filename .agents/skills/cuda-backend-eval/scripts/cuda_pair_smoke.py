#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run paired local A100 / remote H200 CUDA smoke captures."""

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
class PairedSmokeConfig:
    remote: str = "bizhaoh200"
    remote_workdir: str = "/data/shibizhao/pto-cu"
    branch: str = "design/nvidia-backend"
    output_root: Path = Path("tmp/cuda-backend")
    local_device: int = 0
    remote_device: int = 0
    n: int = 1024
    block_dim: int = 256
    op: str = "add"
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
    build_runtime: bool = False


def _git_commit(runner: Runner = subprocess.run) -> str:
    result = runner(["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _artifact_label(config: PairedSmokeConfig, suffix: str) -> str:
    return f"worker-{config.op}-smoke-{suffix}"


def _output_dir(config: PairedSmokeConfig, suffix: str) -> Path:
    return config.output_root / _artifact_label(config, suffix)


def build_remote_git_commit_command(config: PairedSmokeConfig) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={config.ssh_connect_timeout}",
        config.remote,
        f"cd {shlex.quote(config.remote_workdir)} && git rev-parse --short HEAD",
    ]


def build_remote_sync_command(config: PairedSmokeConfig) -> list[str]:
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


def _remote_git_commit(config: PairedSmokeConfig, runner: Runner = subprocess.run) -> str:
    result = runner(build_remote_git_commit_command(config), check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _smoke_args(*, device: int, arch: str, output_json: Path, config: PairedSmokeConfig) -> list[str]:
    args = [
        "--runner",
        "worker",
        "--op",
        config.op,
        "--device",
        str(device),
        "--n",
        str(config.n),
        "--block-dim",
        str(config.block_dim),
        "--arch",
        arch,
        "--output-json",
        str(output_json),
    ]
    if not config.build_runtime:
        args.insert(-2, "--no-build")
    return args


def build_local_smoke_command(config: PairedSmokeConfig, suffix: str) -> list[str]:
    output_dir = _output_dir(config, suffix)
    return [
        "env",
        f"PYTHONPATH={Path.cwd()}:{Path.cwd() / 'python'}",
        config.local_python,
        ".agents/skills/cuda-backend-eval/scripts/cuda_smoke.py",
        *_smoke_args(
            device=config.local_device,
            arch=config.local_arch,
            output_json=output_dir / "a100.json",
            config=config,
        ),
    ]


def _remote_shell_command(config: PairedSmokeConfig, suffix: str) -> str:
    output_dir = _output_dir(config, suffix)
    smoke = [
        config.remote_python,
        ".agents/skills/cuda-backend-eval/scripts/cuda_smoke.py",
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


def build_remote_smoke_command(config: PairedSmokeConfig, suffix: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={config.ssh_connect_timeout}",
        config.remote,
        _remote_shell_command(config, suffix),
    ]


def build_scp_command(config: PairedSmokeConfig, suffix: str) -> list[str]:
    output_dir = _output_dir(config, suffix)
    return [
        "scp",
        f"{config.remote}:{config.remote_workdir}/{output_dir / 'h200.json'}",
        str(output_dir / "h200.json"),
    ]


def build_report_command(config: PairedSmokeConfig, suffix: str) -> list[str]:
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


def build_index_command(config: PairedSmokeConfig) -> list[str]:
    return [
        "env",
        f"PYTHONPATH={Path.cwd()}:{Path.cwd() / 'python'}",
        config.local_python,
        ".agents/skills/cuda-backend-eval/scripts/cuda_artifact_index.py",
        "--root",
        str(config.output_root),
    ]


def run_paired_smoke(
    config: PairedSmokeConfig,
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
            build_index_command(config),
        ]
    )
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
    parser.add_argument("--block-dim", type=int, default=256)
    parser.add_argument(
        "--op",
        choices=("add", "mul", "scale", "square", "axpy", "affine", "triad", "quad"),
        default="add",
    )
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
    parser.add_argument("--build-runtime", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = PairedSmokeConfig(
        remote=args.remote,
        remote_workdir=args.remote_workdir,
        branch=args.branch,
        output_root=args.output_root,
        local_device=args.local_device,
        remote_device=args.remote_device,
        n=args.n,
        block_dim=args.block_dim,
        op=args.op,
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
        build_runtime=args.build_runtime,
    )
    run_paired_smoke(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
