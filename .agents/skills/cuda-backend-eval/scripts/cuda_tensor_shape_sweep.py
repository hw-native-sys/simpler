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
import statistics
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Runner = Callable[..., subprocess.CompletedProcess]

_ALLOWED_BASELINES = frozenset(
    {
        "pto_persistent_dag_tensor",
        "pto_persistent_dag_tensor_core",
        "cublas_sgemm",
    }
)
SOURCE_PAPERS = (
    {
        "id": "arXiv:2605.03190",
        "label": "VDCores",
        "path": "tmp/sources/arxiv-2605.03190-vdcores.txt",
    },
    {
        "id": "arXiv:2512.22219v1",
        "label": "MPK persistent kernel",
        "path": "tmp/sources/arxiv-2512.22219v1-mirage-persistent-kernel.txt",
    },
)
PAPER_SETUP = (
    "Model-shaped tensor tile sweep using fixed GPU work, repeated samples, "
    "selected launch/library baselines, local A100, and remote H200."
)
WORKLOAD_DESCRIPTIONS = {
    "pto_persistent_dag_tensor": "PTO persistent DAG with scalar tiled GEMM work.",
    "pto_persistent_dag_tensor_core": "PTO persistent DAG with a block-wide WMMA TF32/F32 task.",
    "cublas_sgemm": "CUDA Runtime API plus cuBLAS SGEMM over the same descriptor.",
}


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
    sizes: tuple[int, ...] = ()
    repeats: int = 3
    baselines: tuple[str, ...] = ("pto_persistent_dag_tensor",)
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


def parse_sizes(raw: str) -> tuple[int, ...]:
    sizes: list[int] = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        size = int(value)
        if size <= 0:
            raise ValueError(f"invalid tensor sweep size {part!r}; size must be positive")
        sizes.append(size)
    if not sizes:
        raise ValueError("at least one tensor sweep size is required")
    return tuple(sizes)


def parse_baselines(raw: str) -> tuple[str, ...]:
    baselines = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not baselines:
        raise ValueError("at least one tensor shape sweep baseline is required")
    unknown = [baseline for baseline in baselines if baseline not in _ALLOWED_BASELINES]
    if unknown:
        allowed = ", ".join(sorted(_ALLOWED_BASELINES))
        raise ValueError(f"unknown tensor shape sweep baseline(s): {', '.join(unknown)}; allowed: {allowed}")
    return baselines


def _git_commit(runner: Runner = subprocess.run) -> str:
    result = runner(["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _sample_args(*, device: int, arch: str, n: int, shape: TensorShape, baseline: str) -> list[str]:
    return [
        "--device",
        str(device),
        "--single-baseline",
        baseline,
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


def _default_baseline(config: TensorShapeSweepConfig) -> str:
    return config.baselines[0]


def build_local_sample_command(
    config: TensorShapeSweepConfig,
    shape: TensorShape,
    baseline: str | None = None,
    n: int | None = None,
) -> list[str]:
    return [
        "env",
        f"PYTHONPATH={Path.cwd()}:{Path.cwd() / 'python'}",
        config.local_python,
        ".agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py",
        *_sample_args(
            device=config.local_device,
            arch=config.local_arch,
            n=n if n is not None else config.n,
            shape=shape,
            baseline=baseline or _default_baseline(config),
        ),
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


def build_remote_sample_command(
    config: TensorShapeSweepConfig,
    shape: TensorShape,
    baseline: str | None = None,
    n: int | None = None,
) -> list[str]:
    benchmark = [
        config.remote_python,
        ".agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py",
        *_sample_args(
            device=config.remote_device,
            arch=config.remote_arch,
            n=n if n is not None else config.n,
            shape=shape,
            baseline=baseline or _default_baseline(config),
        ),
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
    n: int,
    shape: TensorShape,
    baseline: str,
    repeat: int,
    runner: Runner,
    dry_run: bool,
) -> dict[str, Any]:
    print(" ".join(shlex.quote(part) for part in command), flush=True)
    if dry_run:
        return {
            "artifact": artifact,
            "baseline": baseline,
            "n": n,
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
    sample.setdefault("baseline", baseline)
    sample.setdefault("n", n)
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


def _format_number(value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _format_gflops(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _shape_dims(row: dict[str, Any], shape: str) -> tuple[int, int, int] | None:
    tensor_tile = row.get("tensor_tile")
    if isinstance(tensor_tile, dict):
        rows = tensor_tile.get("rows")
        cols = tensor_tile.get("cols")
        inner = tensor_tile.get("inner")
        if (
            isinstance(rows, int)
            and isinstance(cols, int)
            and isinstance(inner, int)
        ):
            return rows, cols, inner
    parts = shape.split("x")
    if len(parts) != 3:
        return None
    try:
        rows, cols, inner = (int(part) for part in parts)
    except ValueError:
        return None
    return rows, cols, inner


def _tensor_flops(
    row: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> int | None:
    dims = _shape_dims(row, str(row.get("shape", "-")))
    if dims is None:
        return None
    rows, cols, inner = dims
    tensor_tile = row.get("tensor_tile")
    tile_count = tensor_tile.get("tile_count") if isinstance(tensor_tile, dict) else None
    if not isinstance(tile_count, int):
        n = _row_n(row, metadata)
        if not isinstance(n, int):
            return None
        output_elements = rows * cols
        if output_elements <= 0 or n % output_elements != 0:
            return None
        tile_count = n // output_elements
    return 2 * tile_count * rows * cols * inner


def _source_paper_summary(metadata: dict[str, Any]) -> str:
    papers = metadata.get("source_papers") or SOURCE_PAPERS
    parts = []
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        paper_id = paper.get("id")
        label = paper.get("label")
        if paper_id and label:
            parts.append(f"`{paper_id}` {label}")
        elif paper_id:
            parts.append(f"`{paper_id}`")
    return "; ".join(parts)


def _row_n(row: dict[str, Any], metadata: dict[str, Any] | None = None) -> int | str:
    n = row.get("n")
    if n is None and metadata is not None:
        n = metadata.get("n")
    try:
        return int(n)
    except (TypeError, ValueError):
        return "-"


def _median_summary_rows(
    results: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for row in results:
        if row.get("status") != "pass":
            continue
        key = (
            str(row.get("artifact", "-")),
            str(row.get("machine") or row.get("artifact", "-")),
            str(row.get("baseline", "-")),
            str(_row_n(row, metadata)),
            str(row.get("shape", "-")),
            _tile_count(row),
        )
        group = groups.setdefault(key, {"device": [], "host": [], "flops": []})
        group["device"].append(int(row.get("device_wall_ns", 0)))
        group["host"].append(int(row.get("host_wall_ns", 0)))
        flops = _tensor_flops(row, metadata)
        if flops is not None:
            group["flops"].append(flops)
    summary: list[dict[str, Any]] = []
    for (artifact, machine, baseline, n, shape, tiles), values in groups.items():
        median_device = statistics.median(values["device"])
        flops = values["flops"][0] if values["flops"] else None
        summary.append(
            {
                "artifact": artifact,
                "machine": machine,
                "baseline": baseline,
                "n": n,
                "shape": shape,
                "median_device_wall_ns": median_device,
                "median_host_wall_ns": statistics.median(values["host"]),
                "median_gflops": (
                    float(flops) / float(median_device)
                    if flops is not None and median_device
                    else None
                ),
                "samples": len(values["device"]),
                "tiles": tiles,
            }
        )
    return sorted(summary, key=lambda row: (row["n"], row["shape"], row["baseline"], row["artifact"]))


def render_markdown(payload: dict[str, Any]) -> str:
    metadata = payload["metadata"]
    lines = [
        "# CUDA Tensor Shape Sweep",
        "",
        f"- Label: `{metadata['label']}`",
        f"- Commit: `{metadata['git_commit']}`",
        f"- N: `{metadata['n']}`",
        f"- Sizes: {', '.join(f'`{size}`' for size in metadata.get('sizes', [metadata['n']]))}",
        f"- Repeats: `{metadata['repeats']}`",
        f"- Baselines: {', '.join(f'`{baseline}`' for baseline in metadata.get('baselines', []))}.",
        f"- Source setup: {_source_paper_summary(metadata)}.",
        f"- Paper alignment: {metadata.get('paper_setup', PAPER_SETUP)}",
        "- Workloads:",
        *[
            f"  - `{baseline}`: {WORKLOAD_DESCRIPTIONS.get(baseline, 'custom tensor sweep baseline.')}"
            for baseline in metadata.get("baselines", [])
        ],
        "- Scope: early model-shaped tile sweep, not an end-to-end model benchmark.",
        "",
        "| Artifact | Machine | Baseline | N | Shape | Repeat | Status | Device ns | Host ns | Tiles | Dispatch |",
        "| -------- | ------- | -------- | - | ----- | ------ | ------ | --------- | ------- | ----- | -------- |",
    ]
    results = sorted(
        payload["results"],
        key=lambda row: (_row_n(row, metadata), row["shape"], row.get("baseline", ""), row["artifact"], row["repeat"]),
    )
    for row in results:
        machine = row.get("machine") or row.get("artifact", "-")
        lines.append(
            f"| {row.get('artifact', '-')} | {machine} | {row.get('baseline', '-')} | "
            f"{_row_n(row, metadata)} | {row['shape']} | "
            f"{row['repeat']} | {row.get('status', '-')} | {row.get('device_wall_ns', '-')} | "
            f"{row.get('host_wall_ns', '-')} | {_tile_count(row)} | `{_dispatch(row)}` |"
        )
    summary_rows = _median_summary_rows(results, metadata)
    if summary_rows:
        lines.extend(
            [
                "",
                "## Median Summary",
                "",
                "| Artifact | Machine | Baseline | N | Shape | Median device ns | Median host ns | Median GF/s | Samples |",
                "| -------- | ------- | -------- | - | ----- | ---------------- | -------------- | ----------- | ------- |",
            ]
        )
        for row in summary_rows:
            lines.append(
                f"| {row['artifact']} | {row['machine']} | {row['baseline']} | {row['n']} | {row['shape']} | "
                f"{_format_number(row['median_device_wall_ns'])} | {_format_number(row['median_host_wall_ns'])} | "
                f"{_format_gflops(row['median_gflops'])} | {row['samples']} |"
            )
    lines.append("")
    return "\n".join(lines)


def render_svg(payload: dict[str, Any]) -> str:
    summary_rows = _median_summary_rows(list(payload["results"]), payload.get("metadata"))
    max_device_ns = max((float(row.get("median_device_wall_ns", 0)) for row in summary_rows), default=1)
    width = 900
    row_height = 28
    height = 96 + max(1, len(summary_rows)) * row_height
    title = html.escape(payload["metadata"]["label"])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="32" font-family="monospace" font-size="16">{title}</text>',
        '<text x="24" y="54" font-family="monospace" font-size="12">Median device ns</text>',
    ]
    bar_x = 340
    bar_max = 480
    colors = {
        "pto_persistent_dag_tensor": "#2a9d8f",
        "pto_persistent_dag_tensor_core": "#d1495b",
        "cublas_sgemm": "#006d77",
    }
    for idx, row in enumerate(summary_rows):
        y = 76 + idx * row_height
        value = float(row.get("median_device_wall_ns", 0))
        bar_width = int(bar_max * value / max_device_ns) if max_device_ns else 0
        baseline = row.get("baseline", "-")
        color = colors.get(baseline, "#2a9d8f")
        label = (
            f'{row.get("artifact", "-")} {baseline} n={row.get("n", "-")} {row.get("shape", "-")} '
            f'samples={row.get("samples", "-")}'
        )
        lines.append(
            f'<text x="24" y="{y + 16}" font-family="monospace" font-size="12">{html.escape(label)}</text>'
        )
        lines.append(f'<rect x="{bar_x}" y="{y}" width="{bar_width}" height="18" fill="{color}"/>')
        lines.append(
            f'<text x="{bar_x + bar_width + 8}" y="{y + 14}" font-family="monospace" font-size="12">'
            f'{_format_number(value)}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_throughput_svg(payload: dict[str, Any]) -> str:
    summary_rows = _median_summary_rows(list(payload["results"]), payload.get("metadata"))
    max_gflops = max((float(row.get("median_gflops") or 0) for row in summary_rows), default=1)
    width = 900
    row_height = 28
    height = 96 + max(1, len(summary_rows)) * row_height
    title = html.escape(payload["metadata"]["label"])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="32" font-family="monospace" font-size="16">{title}</text>',
        '<text x="24" y="54" font-family="monospace" font-size="12">Median GF/s</text>',
    ]
    bar_x = 340
    bar_max = 480
    colors = {
        "pto_persistent_dag_tensor": "#2a9d8f",
        "pto_persistent_dag_tensor_core": "#d1495b",
        "cublas_sgemm": "#006d77",
    }
    for idx, row in enumerate(summary_rows):
        y = 76 + idx * row_height
        value = row.get("median_gflops")
        numeric_value = float(value or 0)
        bar_width = int(bar_max * numeric_value / max_gflops) if max_gflops else 0
        baseline = row.get("baseline", "-")
        color = colors.get(baseline, "#2a9d8f")
        label = (
            f'{row.get("artifact", "-")} {baseline} n={row.get("n", "-")} {row.get("shape", "-")} '
            f'samples={row.get("samples", "-")}'
        )
        lines.append(
            f'<text x="24" y="{y + 16}" font-family="monospace" font-size="12">{html.escape(label)}</text>'
        )
        lines.append(f'<rect x="{bar_x}" y="{y}" width="{bar_width}" height="18" fill="{color}"/>')
        lines.append(
            f'<text x="{bar_x + bar_width + 8}" y="{y + 14}" font-family="monospace" font-size="12">'
            f'{_format_gflops(value)}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_report(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cuda-tensor-shape-sweep.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (output_dir / "cuda-tensor-shape-sweep.md").write_text(render_markdown(payload))
    (output_dir / "cuda-tensor-shape-sweep.svg").write_text(render_svg(payload))
    (output_dir / "cuda-tensor-shape-throughput.svg").write_text(render_throughput_svg(payload))


def print_report_paths(output_dir: Path) -> None:
    print(output_dir / "cuda-tensor-shape-sweep.json")
    print(output_dir / "cuda-tensor-shape-sweep.md")
    print(output_dir / "cuda-tensor-shape-sweep.svg")
    print(output_dir / "cuda-tensor-shape-throughput.svg")


def render_existing_report(input_json: Path, output_dir: Path | None = None) -> dict[str, Any]:
    payload = json.loads(input_json.read_text())
    target_dir = output_dir or input_json.parent
    write_report(payload, target_dir)
    print_report_paths(target_dir)
    return payload


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
    sizes = config.sizes or (config.n,)
    for baseline in config.baselines:
        for shape in config.shapes:
            for n in sizes:
                for repeat in range(config.repeats):
                    results.append(
                        _run_sample(
                            build_local_sample_command(config, shape, baseline, n=n),
                            artifact="a100",
                            n=n,
                            shape=shape,
                            baseline=baseline,
                            repeat=repeat,
                            runner=runner,
                            dry_run=dry_run,
                        )
                    )
                    results.append(
                        _run_sample(
                            build_remote_sample_command(config, shape, baseline, n=n),
                            artifact="h200",
                            n=n,
                            shape=shape,
                            baseline=baseline,
                            repeat=repeat,
                            runner=runner,
                            dry_run=dry_run,
                        )
                    )
    payload = {
        "metadata": {
            "label": f"tensor-shape-sweep-{commit}",
            "git_commit": commit,
            "n": sizes[0],
            "sizes": list(sizes),
            "repeats": config.repeats,
            "baselines": list(config.baselines),
            "shapes": [shape.label for shape in config.shapes],
            "paper_setup": PAPER_SETUP,
            "source_papers": list(SOURCE_PAPERS),
        },
        "results": results,
    }
    write_report(payload, output_dir)
    print_report_paths(output_dir)
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
    parser.add_argument("--sizes")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--baselines", default="pto_persistent_dag_tensor")
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
    parser.add_argument("--render-json", type=Path)
    parser.add_argument("--render-output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.render_json is not None:
        render_existing_report(args.render_json, args.render_output_dir)
        return
    config = TensorShapeSweepConfig(
        remote=args.remote,
        remote_workdir=args.remote_workdir,
        branch=args.branch,
        output_root=args.output_root,
        local_device=args.local_device,
        remote_device=args.remote_device,
        n=(parse_sizes(args.sizes)[0] if args.sizes else args.n),
        sizes=parse_sizes(args.sizes) if args.sizes else (args.n,),
        repeats=args.repeats,
        baselines=parse_baselines(args.baselines),
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
