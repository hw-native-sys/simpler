#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CUDA host_schedule microbenchmark and report generator."""

from __future__ import annotations

import argparse
import ctypes
import json
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cuda_smoke import _compile_ptx, run_smoke


def _git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _nvidia_smi_summary() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "nvidia-smi unavailable"


def _check_cuda(rc: int, op: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{op} failed with CUDA driver code {rc}")


class DirectCudaDriver:
    """Minimal CUDA Driver API wrapper used as a launch baseline."""

    def __init__(self, device: int, ptx: bytes):
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.ctx = ctypes.c_void_p()
        self.module = ctypes.c_void_p()
        self.function = ctypes.c_void_p()
        self.start_event = ctypes.c_void_p()
        self.stop_event = ctypes.c_void_p()
        self.ptx_buf = ctypes.create_string_buffer(ptx + b"\0")
        self._bind()
        self._init(device)

    def _bind(self) -> None:
        cu_device = ctypes.c_int
        cu_deviceptr = ctypes.c_uint64
        self.lib.cuInit.argtypes = [ctypes.c_uint]
        self.lib.cuDeviceGet.argtypes = [ctypes.POINTER(cu_device), ctypes.c_int]
        self.lib.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, cu_device]
        self.lib.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
        self.lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(cu_deviceptr), ctypes.c_size_t]
        self.lib.cuMemFree_v2.argtypes = [cu_deviceptr]
        self.lib.cuMemcpyHtoD_v2.argtypes = [cu_deviceptr, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, cu_deviceptr, ctypes.c_size_t]
        self.lib.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        self.lib.cuModuleUnload.argtypes = [ctypes.c_void_p]
        self.lib.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        self.lib.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        self.lib.cuCtxSynchronize.argtypes = []
        self.lib.cuEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        self.lib.cuEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.cuEventSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cuEventElapsedTime.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.cuEventDestroy_v2.argtypes = [ctypes.c_void_p]

    def _init(self, device: int) -> None:
        dev = ctypes.c_int()
        _check_cuda(self.lib.cuInit(0), "cuInit")
        _check_cuda(self.lib.cuDeviceGet(ctypes.byref(dev), device), "cuDeviceGet")
        _check_cuda(self.lib.cuCtxCreate_v2(ctypes.byref(self.ctx), 0, dev), "cuCtxCreate")
        _check_cuda(self.lib.cuModuleLoadData(ctypes.byref(self.module), self.ptx_buf), "cuModuleLoadData")
        _check_cuda(
            self.lib.cuModuleGetFunction(ctypes.byref(self.function), self.module, b"pto_vector_add_f32"),
            "cuModuleGetFunction",
        )
        _check_cuda(self.lib.cuEventCreate(ctypes.byref(self.start_event), 0), "cuEventCreate(start)")
        _check_cuda(self.lib.cuEventCreate(ctypes.byref(self.stop_event), 0), "cuEventCreate(stop)")

    def close(self) -> None:
        if self.start_event:
            self.lib.cuEventDestroy_v2(self.start_event)
            self.start_event = ctypes.c_void_p()
        if self.stop_event:
            self.lib.cuEventDestroy_v2(self.stop_event)
            self.stop_event = ctypes.c_void_p()
        if self.module:
            self.lib.cuModuleUnload(self.module)
            self.module = ctypes.c_void_p()
        if self.ctx:
            self.lib.cuCtxDestroy_v2(self.ctx)
            self.ctx = ctypes.c_void_p()

    def __enter__(self) -> DirectCudaDriver:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def run_vector_add(self, n: int, block_dim: int) -> dict[str, Any]:
        array_t = ctypes.c_float * n
        host_a = array_t(*[float(i) for i in range(n)])
        host_b = array_t(*[float(2 * i) for i in range(n)])
        host_out = array_t()
        nbytes = ctypes.sizeof(host_a)

        dev_a = ctypes.c_uint64()
        dev_b = ctypes.c_uint64()
        dev_out = ctypes.c_uint64()
        _check_cuda(self.lib.cuMemAlloc_v2(ctypes.byref(dev_a), nbytes), "cuMemAlloc(a)")
        _check_cuda(self.lib.cuMemAlloc_v2(ctypes.byref(dev_b), nbytes), "cuMemAlloc(b)")
        _check_cuda(self.lib.cuMemAlloc_v2(ctypes.byref(dev_out), nbytes), "cuMemAlloc(out)")
        try:
            _check_cuda(self.lib.cuMemcpyHtoD_v2(dev_a, ctypes.byref(host_a), nbytes), "cuMemcpyHtoD(a)")
            _check_cuda(self.lib.cuMemcpyHtoD_v2(dev_b, ctypes.byref(host_b), nbytes), "cuMemcpyHtoD(b)")

            a_arg = ctypes.c_uint64(dev_a.value)
            b_arg = ctypes.c_uint64(dev_b.value)
            out_arg = ctypes.c_uint64(dev_out.value)
            n_arg = ctypes.c_uint64(n)
            kernel_args = (ctypes.c_void_p * 4)(
                ctypes.cast(ctypes.byref(a_arg), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(b_arg), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(out_arg), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(n_arg), ctypes.c_void_p),
            )

            grid_dim = (n + block_dim - 1) // block_dim
            host_start = time.time_ns()
            _check_cuda(self.lib.cuEventRecord(self.start_event, None), "cuEventRecord(start)")
            _check_cuda(
                self.lib.cuLaunchKernel(self.function, grid_dim, 1, 1, block_dim, 1, 1, 0, None, kernel_args, None),
                "cuLaunchKernel",
            )
            _check_cuda(self.lib.cuEventRecord(self.stop_event, None), "cuEventRecord(stop)")
            _check_cuda(self.lib.cuEventSynchronize(self.stop_event), "cuEventSynchronize")
            host_wall_ns = time.time_ns() - host_start

            elapsed_ms = ctypes.c_float()
            _check_cuda(
                self.lib.cuEventElapsedTime(ctypes.byref(elapsed_ms), self.start_event, self.stop_event),
                "cuEventElapsedTime",
            )
            _check_cuda(self.lib.cuMemcpyDtoH_v2(ctypes.byref(host_out), dev_out, nbytes), "cuMemcpyDtoH")
            if list(host_out) != [float(3 * i) for i in range(n)]:
                raise RuntimeError("direct-driver output mismatch")
        finally:
            self.lib.cuMemFree_v2(dev_a)
            self.lib.cuMemFree_v2(dev_b)
            self.lib.cuMemFree_v2(dev_out)

        return {
            "baseline": "direct_driver",
            "n": n,
            "block_dim": block_dim,
            "host_wall_ns": host_wall_ns,
            "device_wall_ns": int(elapsed_ms.value * 1_000_000),
            "status": "pass",
        }


def run_pto_sample(device: int, n: int, block_dim: int, arch: str) -> dict[str, Any]:
    result = run_smoke(device=device, n=n, block_dim=block_dim, arch=arch)
    result["baseline"] = "pto_host_schedule"
    return result


def run_direct_sample(device: int, n: int, block_dim: int, ptx: bytes) -> dict[str, Any]:
    with DirectCudaDriver(device=device, ptx=ptx) as driver:
        return driver.run_vector_add(n=n, block_dim=block_dim)


def run_single_sample(baseline: str, device: int, n: int, block_dim: int, arch: str) -> dict[str, Any]:
    if baseline == "pto_host_schedule":
        return run_pto_sample(device=device, n=n, block_dim=block_dim, arch=arch)
    if baseline == "direct_driver":
        with tempfile.TemporaryDirectory(prefix="pto_cuda_bench_") as td:
            ptx, ptx_source = _compile_ptx(Path(td), arch)
        result = run_direct_sample(device=device, n=n, block_dim=block_dim, ptx=ptx)
        result["ptx_source"] = ptx_source
        return result
    raise ValueError(f"unknown baseline: {baseline}")


def run_benchmark(device: int, sizes: list[int], repeats: int, block_dim: int, arch: str, label: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="pto_cuda_bench_") as td:
        ptx, ptx_source = _compile_ptx(Path(td), arch)

    metadata = {
        "label": label,
        "machine": socket.gethostname(),
        "git_commit": _git_commit(),
        "timestamp_unix": time.time(),
        "device": device,
        "block_dim": block_dim,
        "ptx_arch": arch,
        "ptx_source": ptx_source,
        "nvidia_smi": _nvidia_smi_summary(),
        "paper_setup": (
            "Microbenchmark slice inspired by VDCores/MPK persistent-kernel evaluation; "
            "not an end-to-end LLM serving result."
        ),
    }
    results: list[dict[str, Any]] = []
    for n in sizes:
        for repeat in range(repeats):
            pto = run_single_sample(baseline="pto_host_schedule", device=device, n=n, block_dim=block_dim, arch=arch)
            pto.update({"machine": metadata["machine"], "repeat": repeat})
            results.append(pto)

            direct = run_single_sample(baseline="direct_driver", device=device, n=n, block_dim=block_dim, arch=arch)
            direct.update({"machine": metadata["machine"], "repeat": repeat})
            results.append(direct)

    return {"metadata": metadata, "results": results}


def summarize_results(payload: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in payload.get("results", []):
        key = (row["machine"], row["baseline"], int(row["n"]))
        grouped[key].append(row)

    summary: dict[tuple[str, str, int], dict[str, Any]] = {}
    for key, rows in grouped.items():
        host_values = [int(row.get("host_wall_ns", row["device_wall_ns"])) for row in rows]
        device_values = [int(row["device_wall_ns"]) for row in rows]
        summary[key] = {
            "machine": key[0],
            "baseline": key[1],
            "n": key[2],
            "samples": len(rows),
            "median_host_wall_ns": int(statistics.median(host_values)),
            "median_device_wall_ns": int(statistics.median(device_values)),
        }
    return summary


def _sorted_summary_rows(summary: dict[tuple[str, str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(summary.values(), key=lambda row: (row["machine"], row["n"], row["baseline"]))


def merge_payloads(payloads: list[dict[str, Any]], label: str) -> dict[str, Any]:
    source_labels = [payload.get("metadata", {}).get("label", "unknown") for payload in payloads]
    git_commits = sorted(
        {
            payload.get("metadata", {}).get("git_commit", "unknown")
            for payload in payloads
            if payload.get("metadata", {}).get("git_commit")
        }
    )
    results: list[dict[str, Any]] = []
    for payload in payloads:
        results.extend(payload.get("results", []))

    return {
        "metadata": {
            "label": label,
            "git_commit": ",".join(git_commits) if git_commits else "unknown",
            "git_commits": git_commits,
            "machine": "combined",
            "nvidia_smi": "combined report",
            "paper_setup": (
                "Combined microbenchmark slice inspired by VDCores/MPK "
                "persistent-kernel evaluation; not an end-to-end LLM serving "
                "result."
            ),
            "source_labels": source_labels,
            "timestamp_unix": time.time(),
        },
        "results": results,
    }


def render_svg(summary: dict[tuple[str, str, int], dict[str, Any]]) -> str:
    rows = _sorted_summary_rows(summary)
    if not rows:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="80"></svg>\n'

    max_ns = max(row["median_device_wall_ns"] for row in rows) or 1
    bar_height = 18
    row_gap = 10
    left = 230
    chart_width = 520
    height = 50 + len(rows) * (bar_height + row_gap)
    width = left + chart_width + 160
    colors = {"pto_host_schedule": "#2f6fbb", "direct_driver": "#2a9d65"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" font-family="sans-serif" font-size="18" font-weight="700">'
        "Median device time by baseline</text>",
    ]
    for idx, row in enumerate(rows):
        y = 50 + idx * (bar_height + row_gap)
        label = f"{row['machine']} n={row['n']} {row['baseline']}"
        bar_width = int(chart_width * row["median_device_wall_ns"] / max_ns)
        color = colors.get(row["baseline"], "#777777")
        lines.append(f'<text x="20" y="{y + 14}" font-family="sans-serif" font-size="12">{label}</text>')
        lines.append(f'<rect x="{left}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{color}"/>')
        lines.append(
            f'<text x="{left + bar_width + 8}" y="{y + 14}" font-family="sans-serif" font-size="12">'
            f"{row['median_device_wall_ns']} ns</text>"
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_markdown_report(payload: dict[str, Any]) -> str:
    summary = summarize_results(payload)
    metadata = payload.get("metadata", {})
    lines = [
        "# CUDA Backend Microbenchmark Report",
        "",
        "This report is an early PTO CUDA runtime microbenchmark. It follows the",
        "VDCores and MPK papers only at the evaluation-shape level: fixed offline",
        "GPU work, repeated batch-size-like problem sizes, and selected baselines.",
        "It is not an end-to-end LLM serving result.",
        "",
        f"- Label: `{metadata.get('label', 'unknown')}`",
        f"- Git commit: `{metadata.get('git_commit', 'unknown')}`",
        f"- Machine: `{metadata.get('machine', 'unknown')}`",
        f"- NVIDIA: `{metadata.get('nvidia_smi', 'unknown')}`",
        f"- Setup note: {metadata.get('paper_setup', 'not provided')}",
    ]
    if metadata.get("source_labels"):
        lines.append(f"- Source reports: `{', '.join(metadata['source_labels'])}`")
    lines.extend(
        [
            "",
            "| Machine | Baseline | N | Samples | Median device ns | Median host ns |",
            "| ------- | -------- | - | ------- | ---------------- | -------------- |",
        ]
    )
    for row in _sorted_summary_rows(summary):
        lines.append(
            f"| {row['machine']} | {row['baseline']} | {row['n']} | {row['samples']} | "
            f"{row['median_device_wall_ns']} | {row['median_host_wall_ns']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `direct_driver` measures a thin CUDA Driver API launch path for the same",
            "  vector-add PTX kernel.",
            "- `pto_host_schedule` measures the current PTO CUDA host runtime path,",
            "  including the runtime C API boundary and manifest lookup.",
            "- Small `n` values are dominated by launch overhead; larger `n` values start",
            "  to include more device work but are still a microbenchmark.",
            "",
            "![Median device time](cuda-benchmark.svg)",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cuda-benchmark.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (output_dir / "cuda-benchmark.md").write_text(render_markdown_report(payload))
    (output_dir / "cuda-benchmark.svg").write_text(render_svg(summarize_results(payload)))


def _parse_sizes(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sizes", default="1024,1048576")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--block-dim", type=int, default=256)
    parser.add_argument("--arch", default="compute_80")
    parser.add_argument("--label", default="local")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cuda-backend/latest"))
    parser.add_argument("--single-baseline", choices=["pto_host_schedule", "direct_driver"], default=None)
    parser.add_argument("--merge-json", type=Path, nargs="*", default=None)
    args = parser.parse_args()

    if args.merge_json:
        payloads = [json.loads(path.read_text()) for path in args.merge_json]
        write_report(merge_payloads(payloads, label=args.label), args.output_dir)
        print(f"merged {len(payloads)} reports into {args.output_dir}")
        return

    if args.single_baseline is not None:
        sizes = _parse_sizes(args.sizes)
        if len(sizes) != 1:
            raise SystemExit("--single-baseline requires exactly one --sizes value")
        print(
            json.dumps(
                run_single_sample(args.single_baseline, args.device, sizes[0], args.block_dim, args.arch),
                sort_keys=True,
            )
        )
        return

    payload = run_benchmark(
        device=args.device,
        sizes=_parse_sizes(args.sizes),
        repeats=args.repeats,
        block_dim=args.block_dim,
        arch=args.arch,
        label=args.label,
    )
    write_report(payload, args.output_dir)
    print(json.dumps(payload["metadata"], indent=2, sort_keys=True))
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
