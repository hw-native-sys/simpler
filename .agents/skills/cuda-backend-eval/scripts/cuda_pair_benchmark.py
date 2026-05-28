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

BASELINE_ROWS: tuple[str, ...] = (
    "pto_host_schedule",
    "pto_host_schedule_compiler",
    "pto_host_schedule_unary_square",
    "pto_host_schedule_quad",
    "pto_host_schedule_generic_args",
    "direct_driver",
    "direct_driver_graph",
    "pto_persistent_device",
    "pto_persistent_queue",
    "pto_persistent_dag",
    "pto_persistent_dag_chain",
    "pto_persistent_dag_reuse",
    "pto_persistent_dag_scalar_axpy",
    "pto_persistent_dag_scalar_scale",
    "pto_persistent_dag_scalar_affine",
    "pto_persistent_dag_triad",
    "pto_persistent_dag_quad",
    "pto_persistent_dag_generic_args",
    "pto_persistent_dag_graph",
    "pto_persistent_dag_graph_generic_args4",
    "pto_persistent_dag_graph_chain",
    "pto_persistent_dag_graph_scratch_reuse",
    "pto_persistent_dag_graph_diamond",
    "pto_persistent_dag_graph_tagged_inout",
    "pto_persistent_dag_unary_square",
    "pto_persistent_dag_tensor",
    "pto_persistent_dag_graph_tensor",
    "pto_persistent_dag_tensor_core",
    "cublas_sgemm",
    "cublas_sgemm_graph",
)
BATCH_ROWS: tuple[str, ...] = (
    "pto_host_schedule_batch",
    "pto_persistent_device_batch",
    "pto_persistent_queue_batch",
)
GRID_BATCH_ROW: str = "pto_persistent_device_grid_batch"
EXPECTED_DISPATCH_BY_BASELINE: dict[str, str] = {
    "pto_persistent_dag": "1,2,1",
    "pto_persistent_dag_chain": "1,2,1,2,1",
    "pto_persistent_dag_reuse": "1,2,1,2,1,1",
    "pto_persistent_dag_scalar_axpy": "4,2,1",
    "pto_persistent_dag_scalar_scale": "11,2,1",
    "pto_persistent_dag_scalar_affine": "5,2,1",
    "pto_persistent_dag_triad": "6,2,1",
    "pto_persistent_dag_quad": "8,2,1",
    "pto_persistent_dag_generic_args": "9,2,1",
    "pto_persistent_dag_graph": "9,2,1",
    "pto_persistent_dag_graph_generic_args4": "9,2,1",
    "pto_persistent_dag_graph_chain": "1,2,1,2,1",
    "pto_persistent_dag_graph_scratch_reuse": "1,2,1,2,1,1",
    "pto_persistent_dag_graph_diamond": "9,2,1,2,1",
    "pto_persistent_dag_graph_tagged_inout": "1,1,1",
    "pto_persistent_dag_unary_square": "7,1,1",
    "pto_persistent_dag_tensor": "3,1,2,1",
    "pto_persistent_dag_graph_tensor": "3,1,2,1",
    "pto_persistent_dag_tensor_core": "10,1,2,1",
}
TENSOR_TILE_BASELINES: tuple[str, ...] = (
    "pto_persistent_dag_tensor",
    "pto_persistent_dag_graph_tensor",
    "pto_persistent_dag_tensor_core",
    "cublas_sgemm",
    "cublas_sgemm_graph",
)
EXPECTED_SCRATCH_REUSE_BY_BASELINE: dict[str, str] = {
    "pto_persistent_dag_graph_scratch_reuse": "reused_buffer=tmp0,reuse_task=4",
}
EXPECTED_GRAPH_TASK_ARGS_BY_BASELINE: dict[str, str] = {
    "pto_persistent_dag_graph_tagged_inout": (
        "task0=input:a,input:b,output:tmp1;task1=inout:tmp1,input:b;task2=input:tmp1,input:a,output_existing:out"
    ),
}


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
    tensor_rows: int = 16
    tensor_cols: int = 16
    tensor_inner: int = 16
    local_python: str = sys.executable
    remote_python: str = ".venv/bin/python"
    ssh_connect_timeout: int = 8
    remote_git_low_speed_limit: int = 1
    remote_git_low_speed_time: int = 30
    remote_git_fetch_timeout: int = 60
    refresh_remote: bool = True
    sync_remote_tree: bool = False


def _csv(values: Sequence[int]) -> str:
    return ",".join(str(value) for value in values)


def _git_commit(runner: Runner = subprocess.run) -> str:
    result = runner(["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def build_remote_git_commit_command(config: PairedBenchmarkConfig) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={config.ssh_connect_timeout}",
        config.remote,
        f"cd {shlex.quote(config.remote_workdir)} && git rev-parse --short HEAD",
    ]


def build_remote_sync_command(config: PairedBenchmarkConfig) -> list[str]:
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


def _remote_git_commit(config: PairedBenchmarkConfig, runner: Runner = subprocess.run) -> str:
    result = runner(build_remote_git_commit_command(config), check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _benchmark_args(
    *,
    device: int,
    arch: str,
    label: str,
    output_dir: Path,
    config: PairedBenchmarkConfig,
) -> list[str]:
    args = [
        "--device",
        str(device),
        "--sizes",
        _csv(config.sizes),
        "--repeats",
        str(config.repeats),
        "--arch",
        arch,
        "--include-persistent",
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
    if config.batch_tasks:
        args.extend(["--batch-tasks", _csv(config.batch_tasks)])
        if config.worker_blocks_per_task:
            args.extend(["--worker-blocks-per-task", _csv(config.worker_blocks_per_task)])
    return args


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


def build_scp_command(config: PairedBenchmarkConfig, remote_commit: str) -> list[str]:
    label = f"h200-current-{remote_commit}"
    return [
        "scp",
        "-r",
        f"{config.remote}:{config.remote_workdir}/{config.output_root / label}",
        str(config.output_root),
    ]


def _display_command(command: Sequence[str]) -> str:
    return shlex.join(command).replace(str(Path.cwd()), "$PWD")


def build_command_examples(
    config: PairedBenchmarkConfig,
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
    config: PairedBenchmarkConfig, local_commit: str, remote_commit: str | None = None
) -> list[str]:
    if remote_commit is None:
        remote_commit = local_commit
    combined_label = f"combined-current-{local_commit}"
    if remote_commit != local_commit:
        combined_label = f"{combined_label}-{remote_commit}"
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
        str(config.output_root / f"a100-current-{local_commit}" / "cuda-benchmark.json"),
        str(config.output_root / f"h200-current-{remote_commit}" / "cuda-benchmark.json"),
        *command_example_args,
        "--label",
        combined_label,
        "--output-dir",
        str(config.output_root / combined_label),
    ]


def _combined_label(local_commit: str, remote_commit: str | None = None) -> str:
    if remote_commit is None:
        remote_commit = local_commit
    combined_label = f"combined-current-{local_commit}"
    if remote_commit != local_commit:
        combined_label = f"{combined_label}-{remote_commit}"
    return combined_label


def _selected_baselines(config: PairedBenchmarkConfig) -> list[str]:
    baselines = list(BASELINE_ROWS)
    if config.batch_tasks:
        baselines.extend(BATCH_ROWS)
        if config.worker_blocks_per_task:
            baselines.append(GRID_BATCH_ROW)
    return baselines


def _expected_result_count(config: PairedBenchmarkConfig) -> int:
    rows_per_size_repeat = len(BASELINE_ROWS)
    rows_per_size_repeat += len(config.batch_tasks) * len(BATCH_ROWS)
    rows_per_size_repeat += len(config.batch_tasks) * len(config.worker_blocks_per_task)
    return 2 * len(config.sizes) * config.repeats * rows_per_size_repeat


def build_validate_command(
    config: PairedBenchmarkConfig,
    local_commit: str,
    remote_commit: str | None = None,
) -> list[str]:
    combined_label = _combined_label(local_commit, remote_commit)
    baseline_args = [part for baseline in _selected_baselines(config) for part in ("--require-baseline", baseline)]
    dispatch_args = [
        part
        for baseline in _selected_baselines(config)
        if baseline in EXPECTED_DISPATCH_BY_BASELINE
        for part in ("--require-dispatch", f"{baseline}={EXPECTED_DISPATCH_BY_BASELINE[baseline]}")
    ]
    tensor_shape = f"{config.tensor_rows}x{config.tensor_cols}x{config.tensor_inner}"
    tensor_tile_args = [
        part
        for baseline in _selected_baselines(config)
        if baseline in TENSOR_TILE_BASELINES
        for part in ("--require-tensor-tile", f"{baseline}={tensor_shape}")
    ]
    scratch_reuse_args = [
        part
        for baseline in _selected_baselines(config)
        if baseline in EXPECTED_SCRATCH_REUSE_BY_BASELINE
        for part in ("--require-scratch-reuse", f"{baseline}={EXPECTED_SCRATCH_REUSE_BY_BASELINE[baseline]}")
    ]
    graph_task_args = [
        part
        for baseline in _selected_baselines(config)
        if baseline in EXPECTED_GRAPH_TASK_ARGS_BY_BASELINE
        for part in ("--require-graph-task-args", f"{baseline}={EXPECTED_GRAPH_TASK_ARGS_BY_BASELINE[baseline]}")
    ]
    return [
        "env",
        f"PYTHONPATH={Path.cwd()}:{Path.cwd() / 'python'}",
        config.local_python,
        ".agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py",
        str(config.output_root / combined_label / "cuda-benchmark.json"),
        "--require-size",
        _csv(config.sizes),
        "--expected-repeats",
        str(config.repeats),
        "--expected-result-count",
        str(_expected_result_count(config)),
        *baseline_args,
        *dispatch_args,
        *tensor_tile_args,
        *scratch_reuse_args,
        *graph_task_args,
        "--require-report-files",
        "--require-command-examples",
        "--require-zero-scheduler-errors",
        "--require-source-papers",
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
    local_commit = _git_commit(runner)
    remote_commit = local_commit
    if not config.refresh_remote and not config.sync_remote_tree:
        remote_commit = _remote_git_commit(config, runner)
    commands = [
        build_local_benchmark_command(config, local_commit),
    ]
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
    parser.add_argument("--tensor-rows", type=int, default=16)
    parser.add_argument("--tensor-cols", type=int, default=16)
    parser.add_argument("--tensor-inner", type=int, default=16)
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
        refresh_remote=not args.skip_remote_refresh and not args.sync_remote_tree,
        sync_remote_tree=args.sync_remote_tree,
    )
    run_paired_benchmark(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
