#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import importlib.util
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import torch
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy
from pypto.runtime import RunConfig, run

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
HIGHPERF_REL_DIR = Path("tests/st/a2a3/tensormap_and_ringbuffer/spmd_paged_attention_highperf")
DEFAULT_PYTPO_EXAMPLE = SCRIPT_DIR / "pypto_paged_attention_spmd_fp16_gqa.py"
AVG_HOST_RE = re.compile(r"Avg Host:\s*([0-9.]+)\s*us")
AVG_DEVICE_RE = re.compile(r"Avg Device:\s*([0-9.]+)\s*us")
PYTPO_ROUND_RE = re.compile(r"PYTPO_ROUND_RESULT round=\d+ host_us=([0-9.]+)(?: npu_event_us=([0-9.]+))?")
DEVICE_LOG_AVG_RE = re.compile(
    r"Avg\s+Total:\s*([0-9.]+)\s*us\s*\|\s*Orch:\s*([0-9.]+)\s*us\s*\|\s*Sched:\s*([0-9.]+)\s*us"
)
DEVICE_LOG_TRIMMED_RE = re.compile(
    r"Trimmed Avg\s+Total:\s*([0-9.]+)\s*us\s*\|\s*Orch:\s*([0-9.]+)\s*us\s*\|\s*Sched:\s*([0-9.]+)\s*us"
)


def _default_repo_root() -> Path:
    candidates = [REPO_ROOT, Path.cwd(), Path("/mounted_home/simpler")]
    for candidate in candidates:
        if (candidate / HIGHPERF_REL_DIR).is_dir():
            return candidate
    return REPO_ROOT


@dataclass(frozen=True)
class TimingResult:
    host_us: float
    device_us: float | None
    total_us: float | None = None
    orch_us: float | None = None
    sched_us: float | None = None


@dataclass(frozen=True)
class ShapePreset:
    highperf_case: str
    batch: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    context_len: int
    block_dim: int | None = None
    include_in_suite: bool = True
    run_highperf: bool = True


SHAPE_PRESETS = {
    "b1_h32_kv8_s128_bs128_fp16": ShapePreset(
        highperf_case="b1_h32_kv8_s128_bs128_fp16",
        batch=1,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        block_size=128,
        context_len=128,
    ),
    "b4_h32_kv8_s512_bs128_fp16": ShapePreset(
        highperf_case="b4_h32_kv8_s512_bs128_fp16",
        batch=4,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        block_size=128,
        context_len=512,
    ),
    "b1_h32_kv8_s4096_bs128_fp16": ShapePreset(
        highperf_case="b1_h32_kv8_s4096_bs128_fp16",
        batch=1,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        block_size=128,
        context_len=4096,
    ),
    "b1_h32_kv8_s6144_bs128_fp16": ShapePreset(
        highperf_case="b1_h32_kv8_s6144_bs128_fp16",
        batch=1,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        block_size=128,
        context_len=6144,
        include_in_suite=False,
    ),
    "b1_h32_kv8_s8192_bs128_fp16": ShapePreset(
        highperf_case="b1_h32_kv8_s8192_bs128_fp16",
        batch=1,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        block_size=128,
        context_len=8192,
        include_in_suite=False,
    ),
    "b1_h32_kv8_s16384_bs128_fp16": ShapePreset(
        highperf_case="b1_h32_kv8_s16384_bs128_fp16",
        batch=1,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        block_size=128,
        context_len=16384,
        include_in_suite=False,
    ),
    "b2_h32_kv8_s4096_bs128_fp16": ShapePreset(
        highperf_case="b2_h32_kv8_s4096_bs128_fp16",
        batch=2,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        block_size=128,
        context_len=4096,
        include_in_suite=False,
    ),
    "b2_h32_kv8_s8192_bs128_fp16": ShapePreset(
        highperf_case="b2_h32_kv8_s8192_bs128_fp16",
        batch=2,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        block_size=128,
        context_len=8192,
        include_in_suite=False,
    ),
}


def _backend_for(platform: str) -> BackendType:
    return BackendType.Ascend950 if platform.startswith("a5") else BackendType.Ascend910B


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("pypto_paged_attention_spmd_example", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import PyPTO example from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sync_torch() -> None:
    npu = getattr(torch, "npu", None)
    if npu is not None and hasattr(npu, "synchronize"):
        npu.synchronize()


def _new_npu_event_pair():
    npu = getattr(torch, "npu", None)
    event_type = getattr(npu, "Event", None) if npu is not None else None
    if event_type is None:
        return None
    try:
        return event_type(enable_timing=True), event_type(enable_timing=True)
    except TypeError:
        return event_type(), event_type()


def _overwrite_pypto_inputs_like_highperf(tensors, args: argparse.Namespace) -> None:
    active_blocks = (args.context_len + args.block_size - 1) // args.block_size
    max_blocks = args.max_model_len // args.block_size
    heads_per_kv = args.num_heads // args.num_kv_heads
    pypto_num_heads = args.num_kv_heads * args.q_tile
    dtype = torch.float16

    torch.manual_seed(42)
    query = torch.randn(args.batch, args.num_heads, args.head_dim, dtype=dtype)
    key_dense = torch.randn(args.batch, args.context_len, args.num_kv_heads, args.head_dim, dtype=dtype)
    value_dense = torch.randn(args.batch, args.context_len, args.num_kv_heads, args.head_dim, dtype=dtype)

    tensors["query"].zero_()
    for batch_idx in range(args.batch):
        for kv_idx in range(args.num_kv_heads):
            logical_start = kv_idx * heads_per_kv
            logical_end = logical_start + heads_per_kv
            padded_start = batch_idx * pypto_num_heads + kv_idx * args.q_tile
            padded_end = padded_start + heads_per_kv
            tensors["query"][padded_start:padded_end, :] = query[batch_idx, logical_start:logical_end, :]

    tensors["key_cache"].zero_()
    tensors["value_cache"].zero_()
    for batch_idx in range(args.batch):
        for block_idx in range(active_blocks):
            token_start = block_idx * args.block_size
            token_end = min(token_start + args.block_size, args.context_len)
            valid = token_end - token_start
            phys_block = batch_idx * max_blocks + block_idx
            for kv_idx in range(args.num_kv_heads):
                cache_start = (phys_block * args.num_kv_heads + kv_idx) * args.block_size
                cache_end = cache_start + valid
                tensors["key_cache"][cache_start:cache_end, :] = key_dense[batch_idx, token_start:token_end, kv_idx]
                tensors["value_cache"][cache_start:cache_end, :] = value_dense[batch_idx, token_start:token_end, kv_idx]

    block_table = torch.arange(max_blocks, dtype=torch.int32).unsqueeze(0).expand(args.batch, -1).clone()
    block_table += torch.arange(args.batch, dtype=torch.int32).unsqueeze(1) * max_blocks
    tensors["block_table"][:] = block_table.flatten()
    tensors["context_lens"][:] = torch.full((args.batch,), args.context_len, dtype=torch.int32)
    tensors["scale"][:] = torch.tensor([args.scale], dtype=torch.float32)
    tensors["config"][:] = torch.tensor(
        [args.batch, args.num_heads, args.num_kv_heads, args.head_dim, args.block_size, max_blocks, active_blocks],
        dtype=torch.int64,
    )


def _parse_device_log_summary(output: str) -> tuple[float, float, float] | None:
    matches = list(DEVICE_LOG_TRIMMED_RE.finditer(output)) or list(DEVICE_LOG_AVG_RE.finditer(output))
    if not matches:
        return None
    match = matches[-1]
    return float(match.group(1)), float(match.group(2)), float(match.group(3))


def _parse_highperf_timing(output: str) -> TimingResult:
    host_match = AVG_HOST_RE.search(output)
    if host_match is None:
        raise RuntimeError("Could not parse highperf Avg Host timing from scene-test output")
    device_match = AVG_DEVICE_RE.search(output)
    total_us = orch_us = sched_us = None
    device_log_summary = _parse_device_log_summary(output)
    if device_log_summary is not None:
        total_us, orch_us, sched_us = device_log_summary
    return TimingResult(
        host_us=float(host_match.group(1)),
        device_us=float(device_match.group(1)) if device_match is not None else None,
        total_us=total_us,
        orch_us=orch_us,
        sched_us=sched_us,
    )


def _parse_pypto_child_timing(output: str, args: argparse.Namespace, device_rounds=None) -> TimingResult:
    matches = list(PYTPO_ROUND_RE.finditer(output))
    host_samples = [float(match.group(1)) for match in matches]
    if not host_samples:
        raise RuntimeError("Could not parse PyPTO host timing from child output")
    event_samples = [float(match.group(2)) for match in matches if match.group(2) is not None]
    total_us = orch_us = sched_us = None
    if device_rounds:
        measured = device_rounds[-args.rounds :]
        total_us = mean(round.total_us for round in measured)
        orch_us = mean(round.orch_us for round in measured)
        sched_us = mean(round.sched_us for round in measured)
    return TimingResult(
        host_us=mean(host_samples),
        device_us=mean(event_samples) if event_samples else None,
        total_us=total_us,
        orch_us=orch_us,
        sched_us=sched_us,
    )


def _apply_shape_preset(args: argparse.Namespace, preset: ShapePreset) -> argparse.Namespace:
    shaped_args = argparse.Namespace(**vars(args))
    shaped_args.highperf_case = preset.highperf_case
    shaped_args.batch = preset.batch
    shaped_args.num_heads = preset.num_heads
    shaped_args.num_kv_heads = preset.num_kv_heads
    shaped_args.head_dim = preset.head_dim
    shaped_args.block_size = preset.block_size
    shaped_args.context_len = preset.context_len
    shaped_args.max_model_len = preset.context_len
    if preset.block_dim is not None:
        shaped_args.block_dim = preset.block_dim
    shaped_args.scale = 1.0 / math.sqrt(float(preset.head_dim))
    if not preset.run_highperf:
        shaped_args.skip_highperf = True
    if shaped_args.q_tile is None:
        shaped_args.q_tile = 16
    return shaped_args


def _selected_shape_names(args: argparse.Namespace) -> list[str]:
    if args.shape_suite:
        return [name for name, preset in SHAPE_PRESETS.items() if preset.include_in_suite]
    return args.shape or []


def _load_device_log_tools(repo_root: Path):
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from simpler_setup.tools.device_log_resolver import get_log_root
    from simpler_setup.tools.device_log_timing import parse_device_log_timing

    return get_log_root, parse_device_log_timing


def _device_log_paths(args: argparse.Namespace) -> list[Path]:
    get_log_root, _ = _load_device_log_tools(args.repo_root)
    log_dir = get_log_root() / f"device-{args.device}"
    paths_with_mtime = []
    for log_path in log_dir.glob("*.log"):
        try:
            paths_with_mtime.append((log_path.stat().st_mtime_ns, log_path))
        except OSError:
            continue
    paths_with_mtime.sort()
    return [path for _, path in paths_with_mtime]


def _snapshot_device_logs(args: argparse.Namespace) -> tuple[list[Path], dict[str, int]]:
    logs = _device_log_paths(args)
    offsets: dict[str, int] = {}
    for log_path in logs:
        try:
            offsets[str(log_path)] = log_path.stat().st_size
        except OSError:
            continue
    return logs, offsets


def _parse_pypto_device_log_rounds(args: argparse.Namespace, offsets: dict[str, int], timeout: float = 20.0):
    _, parse_device_log_timing = _load_device_log_tools(args.repo_root)
    deadline = time.monotonic() + timeout
    rounds = []
    while time.monotonic() < deadline:
        rounds = parse_device_log_timing(_device_log_paths(args), offsets=offsets)
        if len(rounds) >= args.rounds:
            return rounds
        time.sleep(0.5)
    return rounds


def _run_highperf(args: argparse.Namespace) -> TimingResult:
    rounds = max(args.rounds, 2)
    if rounds != args.rounds:
        print("highperf scene-test timing requires --rounds >= 2; using 2 rounds")
    highperf_dir = args.repo_root / HIGHPERF_REL_DIR
    if not highperf_dir.is_dir():
        raise SystemExit(
            f"highperf scene test not found at {highperf_dir}; pass --repo-root pointing to a full simpler checkout"
        )

    command = [
        sys.executable,
        "test_spmd_paged_attention_highperf.py",
        "-p",
        args.platform,
        "-d",
        str(args.device),
        "--case",
        f"TestSpmdPagedAttentionHighPerf::{args.highperf_case}",
        "--manual",
        "include",
        "--rounds",
        str(rounds),
    ]
    command.append("--skip-golden")
    command.append("--enable-device-log-timing")
    if args.enable_l2_swimlane:
        command.append("--enable-l2-swimlane")

    print("\n== highperf scene test ==")
    print(" ".join(command))
    proc = subprocess.run(command, cwd=highperf_dir, text=True, capture_output=True, check=False)
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"highperf benchmark failed with exit code {proc.returncode}")
    return _parse_highperf_timing(proc.stdout)


def _pypto_child_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--pypto-child",
        "-p",
        args.platform,
        "-d",
        str(args.device),
        "--rounds",
        str(args.rounds),
        "--warmup",
        str(args.warmup),
        "--pypto-example",
        str(args.pypto_example),
        "--batch",
        str(args.batch),
        "--num-heads",
        str(args.num_heads),
        "--num-kv-heads",
        str(args.num_kv_heads),
        "--head-dim",
        str(args.head_dim),
        "--block-size",
        str(args.block_size),
        "--context-len",
        str(args.context_len),
        "--max-model-len",
        str(args.max_model_len),
        "--q-tile",
        str(args.q_tile),
        "--scale",
        str(args.scale),
        "--block-dim",
        str(args.block_dim),
        "--aicpu-thread-num",
        str(args.aicpu_thread_num),
    ]
    if args.enable_l2_swimlane:
        command.append("--enable-l2-swimlane")
    return command


def _run_pypto(args: argparse.Namespace) -> TimingResult:
    command = _pypto_child_command(args)
    print("\n== PyPTO SPMD example ==")
    print(" ".join(command))
    _logs, offsets = _snapshot_device_logs(args)
    proc = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    device_rounds = _parse_pypto_device_log_rounds(args, offsets)
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"PyPTO benchmark failed with exit code {proc.returncode}")
    return _parse_pypto_child_timing(proc.stdout + proc.stderr, args, device_rounds=device_rounds)


def _run_pypto_child(args: argparse.Namespace) -> None:
    example_path = args.pypto_example
    module = _load_module(example_path)
    active_blocks = (args.context_len + args.block_size - 1) // args.block_size
    max_blocks = args.max_model_len // args.block_size

    pypto_num_heads = args.num_kv_heads * args.q_tile
    program = module.build_paged_attention_spmd_program(
        batch=args.batch,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        block_size=args.block_size,
        max_num_blocks_per_req=max_blocks,
        q_tile=args.q_tile,
    )
    tensor_specs = module.build_tensor_specs(
        batch=args.batch,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        block_size=args.block_size,
        max_num_blocks_per_req=max_blocks,
        active_num_blocks=active_blocks,
        context_len=args.context_len,
        scale=args.scale,
        q_tile=args.q_tile,
    )
    tensors, input_tensors = module._build_runtime_tensors(tensor_specs)
    _overwrite_pypto_inputs_like_highperf(tensors, args)
    input_tensors = tuple(tensors[spec.name] for spec in tensor_specs if not spec.is_output)
    run_config = RunConfig(
        platform=args.platform,
        device_id=args.device,
        strategy=OptimizationStrategy.Default,
        backend_type=_backend_for(args.platform),
        dump_passes=False,
        enable_l2_swimlane=args.enable_l2_swimlane,
        block_dim=args.block_dim,
        aicpu_thread_num=args.aicpu_thread_num,
    )

    print(f"example={example_path}", flush=True)
    print(
        "shape="
        f"batch={args.batch}, logical_heads={args.num_heads}, padded_heads={pypto_num_heads}, "
        f"kv_heads={args.num_kv_heads}, head_dim={args.head_dim}, context_len={args.context_len}, "
        f"max_model_len={args.max_model_len}, block_size={args.block_size}, q_tile={args.q_tile}, "
        f"dtype=fp16, scale={args.scale}",
        flush=True,
    )
    compiled = run(program, config=run_config)

    compiled(*input_tensors, config=run_config)
    _sync_torch()

    for warmup_idx in range(args.warmup):
        print(f"PYTPO_WARMUP_BEGIN {warmup_idx}", flush=True)
        compiled(*input_tensors, config=run_config)
        _sync_torch()
        print(f"PYTPO_WARMUP_END {warmup_idx}", flush=True)

    for round_idx in range(args.rounds):
        print(f"PYTPO_ROUND_BEGIN {round_idx}", flush=True)
        events = _new_npu_event_pair()
        start = time.perf_counter()
        if events is not None:
            events[0].record()
        compiled(*input_tensors, config=run_config)
        if events is not None:
            events[1].record()
            events[1].synchronize()
        else:
            _sync_torch()
        elapsed_us = (time.perf_counter() - start) * 1_000_000.0
        event_us = events[0].elapsed_time(events[1]) * 1_000.0 if events is not None else None
        print(f"PYTPO_ROUND_END {round_idx}", flush=True)
        if event_us is None:
            print(f"PYTPO_ROUND_RESULT round={round_idx} host_us={elapsed_us:.1f}", flush=True)
        else:
            print(f"PYTPO_ROUND_RESULT round={round_idx} host_us={elapsed_us:.1f} npu_event_us={event_us:.1f}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare highperf SPMD PA with the PyPTO SPMD PA example.")
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--enable-l2-swimlane", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument("--pypto-example", type=Path, default=DEFAULT_PYTPO_EXAMPLE)
    parser.add_argument("--shape", action="append", choices=sorted(SHAPE_PRESETS))
    parser.add_argument("--shape-suite", action="store_true")
    parser.add_argument("--highperf-case", default="b4_h32_kv8_s512_bs128_fp16")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--q-tile", type=int, default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--block-dim", type=int, default=24)
    parser.add_argument("--aicpu-thread-num", type=int, default=4)
    parser.add_argument("--skip-highperf", action="store_true")
    parser.add_argument("--skip-pypto", action="store_true")
    parser.add_argument("--pypto-child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _print_highperf_result(result: TimingResult) -> None:
    print(f"highperf host e2e avg:        {result.host_us:.1f} us")
    if result.device_us is not None:
        print(f"highperf runtime device avg:  {result.device_us:.1f} us")
    if result.total_us is not None:
        print(f"highperf device-log total:    {result.total_us:.1f} us")
        print(f"highperf device-log orch:     {result.orch_us:.1f} us")
        print(f"highperf device-log sched:    {result.sched_us:.1f} us")


def _print_pypto_result(result: TimingResult) -> None:
    print(f"PyPTO host e2e avg:           {result.host_us:.1f} us")
    if result.device_us is not None:
        print(f"PyPTO torch.npu.Event avg:    {result.device_us:.1f} us")
    if result.total_us is not None:
        print(f"PyPTO device-log total:       {result.total_us:.1f} us")
        print(f"PyPTO device-log orch:        {result.orch_us:.1f} us")
        print(f"PyPTO device-log sched:       {result.sched_us:.1f} us")


def _run_one_shape(args: argparse.Namespace) -> tuple[TimingResult | None, TimingResult | None]:
    print(f"Note: PyPTO runs from {args.pypto_example} with deterministic highperf-style inputs.")
    print("Note: device-log Total/Orch/Sched is the primary on-device comparison.")
    highperf = None if args.skip_highperf else _run_highperf(args)
    pypto = None if args.skip_pypto else _run_pypto(args)

    print("\n== summary ==")
    if highperf is not None:
        _print_highperf_result(highperf)
    if pypto is not None:
        _print_pypto_result(pypto)
    if highperf is not None and pypto is not None:
        if highperf.host_us > 0:
            print(f"PyPTO / highperf host e2e:    {pypto.host_us / highperf.host_us:.2f}x")
        if highperf.device_us is not None and pypto.total_us is not None and highperf.device_us > 0:
            print(f"PyPTO device-log / highperf runtime device: {pypto.total_us / highperf.device_us:.2f}x")
    return highperf, pypto


def _run_shape_suite(args: argparse.Namespace, shape_names: list[str]) -> None:
    rows = []
    for shape_name in shape_names:
        shape_args = _apply_shape_preset(args, SHAPE_PRESETS[shape_name])
        print(f"\n######## shape: {shape_name} ########")
        highperf, pypto = _run_one_shape(shape_args)
        rows.append((shape_name, highperf, pypto))

    print("\n== shape suite summary ==")
    for shape_name, highperf, pypto in rows:
        highperf_host = f"{highperf.host_us:.1f}" if highperf is not None else "n/a"
        pypto_host = f"{pypto.host_us:.1f}" if pypto is not None else "n/a"
        pypto_event = "n/a"
        if pypto is not None and pypto.device_us is not None:
            pypto_event = f"{pypto.device_us:.1f}"
        host_ratio = "n/a"
        if highperf is not None and pypto is not None and highperf.host_us > 0:
            host_ratio = f"{pypto.host_us / highperf.host_us:.2f}x"
        runtime_device_ratio = "n/a"
        if (
            highperf is not None
            and pypto is not None
            and highperf.device_us is not None
            and pypto.total_us is not None
            and highperf.device_us > 0
        ):
            runtime_device_ratio = f"{pypto.total_us / highperf.device_us:.2f}x"
        pypto_total = "n/a"
        pypto_orch = "n/a"
        pypto_sched = "n/a"
        if pypto is not None and pypto.total_us is not None:
            pypto_total = f"{pypto.total_us:.1f}"
            pypto_orch = f"{pypto.orch_us:.1f}"
            pypto_sched = f"{pypto.sched_us:.1f}"
        highperf_device = "n/a"
        if highperf is not None and highperf.device_us is not None:
            highperf_device = f"{highperf.device_us:.1f}"
        highperf_total = "n/a"
        highperf_orch = "n/a"
        highperf_sched = "n/a"
        if highperf is not None and highperf.total_us is not None:
            highperf_total = f"{highperf.total_us:.1f}"
            highperf_orch = f"{highperf.orch_us:.1f}"
            highperf_sched = f"{highperf.sched_us:.1f}"
        print(
            f"{shape_name}: highperf_host_us={highperf_host} "
            f"highperf_device_us={highperf_device} highperf_total_us={highperf_total} "
            f"highperf_orch_us={highperf_orch} highperf_sched_us={highperf_sched} "
            f"pypto_host_us={pypto_host} pypto_npu_event_us={pypto_event} "
            f"pypto_total_us={pypto_total} pypto_orch_us={pypto_orch} pypto_sched_us={pypto_sched} "
            f"pypto/highperf_host={host_ratio} pypto/highperf_runtime_device={runtime_device_ratio}"
        )


def main() -> None:
    args = _parse_args()
    if args.scale is None:
        args.scale = 1.0 / math.sqrt(float(args.head_dim))
    if args.max_model_len is None:
        args.max_model_len = args.context_len
    if args.num_heads % args.num_kv_heads != 0:
        raise SystemExit("--num-heads must be divisible by --num-kv-heads")
    heads_per_kv = args.num_heads // args.num_kv_heads
    if args.q_tile is None:
        args.q_tile = 16
    if args.q_tile < heads_per_kv:
        raise SystemExit("GQA PyPTO baseline requires --q-tile >= num_heads // num_kv_heads")
    if args.pypto_child:
        _run_pypto_child(args)
        return
    if args.skip_highperf and args.skip_pypto:
        raise SystemExit("nothing to benchmark")
    shape_names = _selected_shape_names(args)
    if shape_names:
        _run_shape_suite(args, shape_names)
    else:
        _run_one_shape(args)


if __name__ == "__main__":
    main()
