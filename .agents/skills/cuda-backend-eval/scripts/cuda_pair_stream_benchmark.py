#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run paired A100/H200 CUDA stream-concurrency benchmarks."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

Runner = Callable[..., subprocess.CompletedProcess]

STREAM_BASELINES: tuple[str, ...] = ("pto_stream_serial", "pto_stream_parallel")
STREAM_SIZE = 2


@dataclass(frozen=True)
class PairedStreamBenchmarkConfig:
    remote: str = "bizhaoh200"
    remote_workdir: str = "/data/shibizhao/pto-cu"
    branch: str = "design/nvidia-backend"
    output_root: Path = Path("tmp/cuda-backend")
    local_device: int = 0
    remote_device: int = 0
    repeats: int = 2
    stream_pool_size: int = 6
    local_arch: str = "compute_80"
    remote_arch: str = "compute_90"
    local_machine: str = "hina"
    remote_machine: str = "dasys-h200x8"
    local_python: str = sys.executable
    remote_python: str = ".venv/bin/python"
    ssh_connect_timeout: int = 8
    remote_git_low_speed_limit: int = 1
    remote_git_low_speed_time: int = 30
    remote_git_fetch_timeout: int = 60
    refresh_remote: bool = True
    sync_remote_tree: bool = False


def _git_commit(runner: Runner = subprocess.run) -> str:
    result = runner(["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def build_remote_git_commit_command(config: PairedStreamBenchmarkConfig) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={config.ssh_connect_timeout}",
        config.remote,
        f"cd {shlex.quote(config.remote_workdir)} && git rev-parse --short HEAD",
    ]


def build_remote_sync_command(config: PairedStreamBenchmarkConfig) -> list[str]:
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


def _remote_git_commit(config: PairedStreamBenchmarkConfig, runner: Runner = subprocess.run) -> str:
    result = runner(build_remote_git_commit_command(config), check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _label(prefix: str, config: PairedStreamBenchmarkConfig, commit: str) -> str:
    return f"{prefix}-stream-pool{config.stream_pool_size}-{commit}"


def _combined_label(
    config: PairedStreamBenchmarkConfig,
    local_commit: str,
    remote_commit: str | None = None,
) -> str:
    if remote_commit is None:
        remote_commit = local_commit
    combined_label = f"combined-stream-pool{config.stream_pool_size}-{local_commit}"
    if remote_commit != local_commit:
        combined_label = f"{combined_label}-{remote_commit}"
    return combined_label


def _benchmark_args(
    *,
    device: int,
    arch: str,
    label: str,
    output_dir: Path,
    config: PairedStreamBenchmarkConfig,
) -> list[str]:
    return [
        "--stream-concurrency",
        "--device",
        str(device),
        "--repeats",
        str(config.repeats),
        "--arch",
        arch,
        "--stream-pool-size",
        str(config.stream_pool_size),
        "--label",
        label,
        "--output-dir",
        str(output_dir),
    ]


def build_local_benchmark_command(config: PairedStreamBenchmarkConfig, commit: str) -> list[str]:
    label = _label("a100", config, commit)
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


def _remote_shell_command(config: PairedStreamBenchmarkConfig, commit: str) -> str:
    label = _label("h200", config, commit)
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
    commands = [f"cd {shlex.quote(config.remote_workdir)}"]
    if config.refresh_remote and not config.sync_remote_tree:
        commands.extend(
            [
                fetch_command,
                f"git checkout -B {shlex.quote(config.branch)} FETCH_HEAD >/dev/null",
            ]
        )
    commands.append(f"{remote_env} {' '.join(shlex.quote(part) for part in benchmark)}")
    return " && ".join(commands)


def build_remote_benchmark_command(config: PairedStreamBenchmarkConfig, commit: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={config.ssh_connect_timeout}",
        config.remote,
        _remote_shell_command(config, commit),
    ]


def build_scp_command(config: PairedStreamBenchmarkConfig, remote_commit: str) -> list[str]:
    label = _label("h200", config, remote_commit)
    return [
        "scp",
        "-r",
        f"{config.remote}:{config.remote_workdir}/{config.output_root / label}",
        str(config.output_root),
    ]


def _display_command(command: Sequence[str]) -> str:
    return shlex.join(command).replace(str(Path.cwd()), "$PWD")


def build_command_examples(
    config: PairedStreamBenchmarkConfig,
    local_commit: str,
    remote_commit: str | None = None,
) -> dict[str, str]:
    if remote_commit is None:
        remote_commit = local_commit
    examples = {
        "local_sample": _display_command(build_local_benchmark_command(config, local_commit)),
        "remote_sample": _display_command(build_remote_benchmark_command(config, remote_commit)),
    }
    if config.sync_remote_tree:
        examples["sync_remote_tree"] = _display_command(build_remote_sync_command(config))
    return examples


def build_merge_command(
    config: PairedStreamBenchmarkConfig,
    local_commit: str,
    remote_commit: str | None = None,
) -> list[str]:
    if remote_commit is None:
        remote_commit = local_commit
    command_examples = build_command_examples(config, local_commit, remote_commit)
    command_example_args = [
        part for key, value in command_examples.items() for part in ("--command-example", f"{key}={value}")
    ]
    return [
        "env",
        f"PYTHONPATH={Path.cwd()}:{Path.cwd() / 'python'}",
        config.local_python,
        ".agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py",
        "--merge-json",
        str(config.output_root / _label("a100", config, local_commit) / "cuda-benchmark.json"),
        str(config.output_root / _label("h200", config, remote_commit) / "cuda-benchmark.json"),
        *command_example_args,
        "--label",
        _combined_label(config, local_commit, remote_commit),
        "--output-dir",
        str(config.output_root / _combined_label(config, local_commit, remote_commit)),
    ]


def build_validate_command(
    config: PairedStreamBenchmarkConfig,
    local_commit: str,
    remote_commit: str | None = None,
) -> list[str]:
    combined_label = _combined_label(config, local_commit, remote_commit)
    baseline_args = [part for baseline in STREAM_BASELINES for part in ("--require-baseline", baseline)]
    return [
        "env",
        f"PYTHONPATH={Path.cwd()}:{Path.cwd() / 'python'}",
        config.local_python,
        ".agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py",
        str(config.output_root / combined_label / "cuda-benchmark.json"),
        "--require-machine",
        config.local_machine,
        "--require-machine",
        config.remote_machine,
        "--require-size",
        str(STREAM_SIZE),
        "--expected-repeats",
        str(config.repeats),
        "--expected-result-count",
        str(len((config.local_machine, config.remote_machine)) * len(STREAM_BASELINES) * config.repeats),
        *baseline_args,
        "--require-report-files",
        "--require-command-examples",
        "--require-source-papers",
    ]


def build_index_command(config: PairedStreamBenchmarkConfig) -> list[str]:
    return [
        "env",
        f"PYTHONPATH={Path.cwd()}:{Path.cwd() / 'python'}",
        config.local_python,
        ".agents/skills/cuda-backend-eval/scripts/cuda_artifact_index.py",
        "--root",
        str(config.output_root),
    ]


def run_paired_stream_benchmark(
    config: PairedStreamBenchmarkConfig,
    *,
    runner: Runner = subprocess.run,
    dry_run: bool = False,
) -> list[list[str]]:
    local_commit = _git_commit(runner)
    remote_commit = local_commit
    if not config.refresh_remote and not config.sync_remote_tree:
        remote_commit = _remote_git_commit(config, runner)
    commands = [build_local_benchmark_command(config, local_commit)]
    if config.sync_remote_tree:
        commands.append(build_remote_sync_command(config))
    commands.extend(
        [
            build_remote_benchmark_command(config, remote_commit),
            build_scp_command(config, remote_commit),
            build_merge_command(config, local_commit, remote_commit),
            build_validate_command(config, local_commit, remote_commit),
            build_index_command(config),
        ]
    )
    for command in commands:
        print(" ".join(shlex.quote(part) for part in command))
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
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--stream-pool-size", type=int, default=6)
    parser.add_argument("--local-arch", default="compute_80")
    parser.add_argument("--remote-arch", default="compute_90")
    parser.add_argument("--local-machine", default="hina")
    parser.add_argument("--remote-machine", default="dasys-h200x8")
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
    config = PairedStreamBenchmarkConfig(
        remote=args.remote,
        remote_workdir=args.remote_workdir,
        branch=args.branch,
        output_root=args.output_root,
        local_device=args.local_device,
        remote_device=args.remote_device,
        repeats=args.repeats,
        stream_pool_size=args.stream_pool_size,
        local_arch=args.local_arch,
        remote_arch=args.remote_arch,
        local_machine=args.local_machine,
        remote_machine=args.remote_machine,
        local_python=args.local_python,
        remote_python=args.remote_python,
        ssh_connect_timeout=args.ssh_connect_timeout,
        remote_git_low_speed_limit=args.remote_git_low_speed_limit,
        remote_git_low_speed_time=args.remote_git_low_speed_time,
        remote_git_fetch_timeout=args.remote_git_fetch_timeout,
        refresh_remote=not args.skip_remote_refresh and not args.sync_remote_tree,
        sync_remote_tree=args.sync_remote_tree,
    )
    run_paired_stream_benchmark(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
