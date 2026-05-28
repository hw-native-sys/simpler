#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run paired A100/H200 tensor-tile shape sweeps for CUDA persistent DAGs."""

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

Runner = Callable[..., subprocess.CompletedProcess]


@dataclass(frozen=True)
class TensorShape:
    rows: int
    cols: int
    inner: int

    @property
    def label(self) -> str:
        return f"{self.rows}x{self.cols}x{self.inner}"


@dataclass(frozen=True)
class TensorShapeSweepConfig:
    remote: str = "bizhaoh200"
    remote_workdir: str = "/data/shibizhao/pto-cu"
    branch: str = "design/nvidia-backend"
    output_root: Path = Path("tmp/cuda-backend")
    local_device: int = 0
    remote_device: int = 0
    n: int = 4096
    repeats: int = 3
    shapes: tuple[TensorShape, ...] = (
        TensorShape(rows=8, cols=4, inner=12),
        TensorShape(rows=16, cols=16, inner=64),
        TensorShape(rows=32, cols=16, inner=64),
    )
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


def parse_shapes(raw: str) -> tuple[TensorShape, ...]:
    shapes: list[TensorShape] = []
    for part in raw.split(","):
        fields = [field.strip() for field in part.lower().split("x")]
        if len(fields) != 3 or not all(fields):
            raise ValueError(f"invalid tensor shape {part!r}; expected ROWSxCOLSxINNER")
        rows, cols, inner = (int(field) for field in fields)
        if rows <= 0 or cols <= 0 or inner <= 0:
            raise ValueError(f"invalid tensor shape {part!r}; dimensions must be positive")
        shapes.append(TensorShape(rows=rows, cols=cols, inner=inner))
    if not shapes:
        raise ValueError("at least one tensor shape is required")
    return tuple(shapes)


def _git_commit(runner: Runner = subprocess.run) -> str:
    result = runner(["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _sample_args(*, device: int, arch: str, n: int, shape: TensorShape) -> list[str]:
    return [
        "--device",
        str(device),
        "--single-baseline",
        "pto_persistent_dag_tensor",
        "--sizes",
        str(n),
        "--arch",
        arch,
        "--tensor-rows",
        str(shape.rows),
        "--tensor-cols",
        str(shape.cols),
        "--tensor-inner",
        str(shape.inner),
    ]


def build_local_sample_command(config: TensorShapeSweepConfig, shape: TensorShape) -> list[str]:
    return [
        "env",
        f"PYTHONPATH={Path.cwd()}:{Path.cwd() / 'python'}",
        config.local_python,
        ".agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py",
        *_sample_args(device=config.local_device, arch=config.local_arch, n=config.n, shape=shape),
    ]


def build_remote_sync_command(config: TensorShapeSweepConfig) -> list[str]:
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


def _remote_prefix(config: TensorShapeSweepConfig) -> list[str]:
    commands = [f"cd {shlex.quote(config.remote_workdir)}"]
    if config.refresh_remote and not config.sync_remote_tree:
        fetch_command = (
            f"timeout {config.remote_git_fetch_timeout} "
            "git "
            f"-c http.lowSpeedLimit={config.remote_git_low_speed_limit} "
            f"-c http.lowSpeedTime={config.remote_git_low_speed_time} "
            f"fetch origin {shlex.quote(config.branch)} >/dev/null"
        )
        commands.extend(
            [
                fetch_command,
                f"git checkout -B {shlex.quote(config.branch)} FETCH_HEAD >/dev/null",
            ]
        )
    return commands


def build_remote_sample_command(config: TensorShapeSweepConfig, shape: TensorShape) -> list[str]:
    benchmark = [
        config.remote_python,
        ".agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py",
        *_sample_args(device=config.remote_device, arch=config.remote_arch, n=config.n, shape=shape),
    ]
    remote_env = "CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH PYTHONPATH=$PWD:$PWD/python"
    commands = _remote_prefix(config)
    commands.append(f"{remote_env} {' '.join(shlex.quote(part) for part in benchmark)}")
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={config.ssh_connect_timeout}",
        config.remote,
        " && ".join(commands),
    ]


def _sample_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise ValueError("benchmark command did not print a JSON sample")


def _run_sample(
    command: list[str],
    *,
    artifact: str,
    shape: TensorShape,
    repeat: int,
    runner: Runner,
    dry_run: bool,
) -> dict[str, Any]:
    print(" ".join(shlex.quote(part) for part in command), flush=True)
    if dry_run:
        return {
            "artifact": artifact,
            "shape": shape.label,
            "repeat": repeat,
            "status": "dry-run",
            "device_wall_ns": 0,
            "host_wall_ns": 0,
            "dispatch_func_ids": [],
            "tensor_tile": {"rows": shape.rows, "cols": shape.cols, "inner": shape.inner},
        }
    result = runner(command, check=True, capture_output=True, text=True)
    sample = _sample_from_stdout(result.stdout)
    sample["artifact"] = artifact
    sample.setdefault("machine", artifact)
    sample["shape"] = shape.label
    sample["repeat"] = repeat
    return sample


def _dispatch(row: dict[str, Any]) -> str:
    dispatch = row.get("dispatch_func_ids")
    if not isinstance(dispatch, list):
        return "-"
    return ",".join(str(value) for value in dispatch)


def _tile_count(row: dict[str, Any]) -> str:
    tensor_tile = row.get("tensor_tile")
    if not isinstance(tensor_tile, dict):
        return "-"
    tile_count = tensor_tile.get("tile_count")
    return str(tile_count) if tile_count is not None else "-"


def render_markdown(payload: dict[str, Any]) -> str:
    metadata = payload["metadata"]
    lines = [
        "# CUDA Tensor Shape Sweep",
        "",
        f"- Label: `{metadata['label']}`",
        f"- Commit: `{metadata['git_commit']}`",
        f"- N: `{metadata['n']}`",
        f"- Repeats: `{metadata['repeats']}`",
        "- Workload: `pto_persistent_dag_tensor` scalar tiled GEMM DAG.",
        "- Scope: early model-shaped tile sweep, not a tuned tensor-core result.",
        "",
        "| Artifact | Machine | Shape | Repeat | Status | Device ns | Host ns | Tiles | Dispatch |",
        "| -------- | ------- | ----- | ------ | ------ | --------- | ------- | ----- | -------- |",
    ]
    results = sorted(payload["results"], key=lambda row: (row["shape"], row["artifact"], row["repeat"]))
    for row in results:
        machine = row.get("machine") or row.get("artifact", "-")
        lines.append(
            f"| {row.get('artifact', '-')} | {machine} | {row['shape']} | "
            f"{row['repeat']} | {row.get('status', '-')} | {row.get('device_wall_ns', '-')} | "
            f"{row.get('host_wall_ns', '-')} | {_tile_count(row)} | `{_dispatch(row)}` |"
        )
    lines.append("")
    return "\n".join(lines)


def render_svg(payload: dict[str, Any]) -> str:
    results = list(payload["results"])
    max_device_ns = max((int(row.get("device_wall_ns", 0)) for row in results), default=1)
    width = 900
    row_height = 28
    height = 80 + max(1, len(results)) * row_height
    title = html.escape(payload["metadata"]["label"])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="32" font-family="monospace" font-size="16">{title}</text>',
    ]
    bar_x = 250
    bar_max = 560
    for idx, row in enumerate(results):
        y = 64 + idx * row_height
        value = int(row.get("device_wall_ns", 0))
        bar_width = int(bar_max * value / max_device_ns) if max_device_ns else 0
        label = f'{row.get("artifact", "-")} {row.get("shape", "-")} r{row.get("repeat", "-")}'
        lines.append(
            f'<text x="24" y="{y + 16}" font-family="monospace" font-size="12">{html.escape(label)}</text>'
        )
        lines.append(f'<rect x="{bar_x}" y="{y}" width="{bar_width}" height="18" fill="#2a9d8f"/>')
        lines.append(
            f'<text x="{bar_x + bar_width + 8}" y="{y + 14}" font-family="monospace" font-size="12">{value}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_report(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cuda-tensor-shape-sweep.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (output_dir / "cuda-tensor-shape-sweep.md").write_text(render_markdown(payload))
    (output_dir / "cuda-tensor-shape-sweep.svg").write_text(render_svg(payload))


def run_tensor_shape_sweep(
    config: TensorShapeSweepConfig,
    *,
    runner: Runner = subprocess.run,
    dry_run: bool = False,
) -> dict[str, Any]:
    commit = _git_commit(runner)
    output_dir = config.output_root / f"tensor-shape-sweep-{commit}"
    if config.sync_remote_tree:
        sync_command = build_remote_sync_command(config)
        print(" ".join(shlex.quote(part) for part in sync_command), flush=True)
        if not dry_run:
            runner(sync_command, check=True)
    results: list[dict[str, Any]] = []
    for shape in config.shapes:
        for repeat in range(config.repeats):
            results.append(
                _run_sample(
                    build_local_sample_command(config, shape),
                    artifact="a100",
                    shape=shape,
                    repeat=repeat,
                    runner=runner,
                    dry_run=dry_run,
                )
            )
            results.append(
                _run_sample(
                    build_remote_sample_command(config, shape),
                    artifact="h200",
                    shape=shape,
                    repeat=repeat,
                    runner=runner,
                    dry_run=dry_run,
                )
            )
    payload = {
        "metadata": {
            "label": f"tensor-shape-sweep-{commit}",
            "git_commit": commit,
            "n": config.n,
            "repeats": config.repeats,
            "shapes": [shape.label for shape in config.shapes],
            "paper_setup": "Model-shaped tensor tile sweep inspired by VDCores/MPK evaluation shapes.",
        },
        "results": results,
    }
    write_report(payload, output_dir)
    print(output_dir / "cuda-tensor-shape-sweep.json")
    print(output_dir / "cuda-tensor-shape-sweep.md")
    print(output_dir / "cuda-tensor-shape-sweep.svg")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", default="bizhaoh200")
    parser.add_argument("--remote-workdir", default="/data/shibizhao/pto-cu")
    parser.add_argument("--branch", default="design/nvidia-backend")
    parser.add_argument("--output-root", type=Path, default=Path("tmp/cuda-backend"))
    parser.add_argument("--local-device", type=int, default=0)
    parser.add_argument("--remote-device", type=int, default=0)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--shapes", default="8x4x12,16x16x64,32x16x64")
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
    config = TensorShapeSweepConfig(
        remote=args.remote,
        remote_workdir=args.remote_workdir,
        branch=args.branch,
        output_root=args.output_root,
        local_device=args.local_device,
        remote_device=args.remote_device,
        n=args.n,
        repeats=args.repeats,
        shapes=parse_shapes(args.shapes),
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
    run_tensor_shape_sweep(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
