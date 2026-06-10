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
SCHED_COST_RE = re.compile(r"sched_cost=([0-9.]+)us")
PYTPO_ROUND_RE = re.compile(r"PYTPO_ROUND_RESULT round=\d+ host_us=([0-9.]+)(?: npu_event_us=([0-9.]+))?")


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
    scheduler_log_us: float | None = None


@dataclass(frozen=True)
class ShapePreset:
    highperf_case: str
    batch: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    context_len: int
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
    "b1_h32_kv8_s8192_bs128_fp16": ShapePreset(
        highperf_case="b1_h32_kv8_s8192_bs128_fp16",
        batch=1,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        block_size=128,
        context_len=8192,
        include_in_suite=False,
        run_highperf=False,
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
        run_highperf=False,
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
        [args.batch, pypto_num_heads, args.num_kv_heads, args.head_dim, args.block_size, max_blocks, active_blocks],
        dtype=torch.int64,
    )


def _extract_logical_heads(padded: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    heads_per_kv = args.num_heads // args.num_kv_heads
    pypto_num_heads = args.num_kv_heads * args.q_tile
    logical = torch.empty(args.batch, args.num_heads, args.head_dim, dtype=padded.dtype, device=padded.device)
    for batch_idx in range(args.batch):
        for kv_idx in range(args.num_kv_heads):
            logical_start = kv_idx * heads_per_kv
            logical_end = logical_start + heads_per_kv
            padded_start = batch_idx * pypto_num_heads + kv_idx * args.q_tile
            padded_end = padded_start + heads_per_kv
            logical[batch_idx, logical_start:logical_end, :] = padded[padded_start:padded_end, :]
    return logical


def _compute_highperf_reference_from_pypto_tensors(tensors, args: argparse.Namespace) -> torch.Tensor:
    query = _extract_logical_heads(tensors["query"], args)
    max_blocks = args.max_model_len // args.block_size
    total_blocks = args.batch * max_blocks
    key_cache = tensors["key_cache"].reshape(total_blocks, args.num_kv_heads, args.block_size, args.head_dim)
    value_cache = tensors["value_cache"].reshape(total_blocks, args.num_kv_heads, args.block_size, args.head_dim)
    key_page = key_cache.permute(0, 2, 1, 3).contiguous()
    value_page = value_cache.permute(0, 2, 1, 3).contiguous()
    block_table = tensors["block_table"].reshape(args.batch, max_blocks)
    context_lens = tensors["context_lens"]

    heads_per_kv = args.num_heads // args.num_kv_heads
    reference = torch.empty(args.batch, args.num_heads, args.head_dim, dtype=torch.float32)
    for batch_idx in range(args.batch):
        seq_len = int(context_lens[batch_idx].item())
        block_count = (seq_len + args.block_size - 1) // args.block_size
        blocks = block_table[batch_idx, :block_count]
        for head_idx in range(args.num_heads):
            kv_head = head_idx // heads_per_kv
            keys = []
            values = []
            remaining = seq_len
            for block in blocks:
                valid = min(args.block_size, remaining)
                block_id = int(block.item())
                keys.append(key_page[block_id, :valid, kv_head, :])
                values.append(value_page[block_id, :valid, kv_head, :])
                remaining -= valid
            key = torch.cat(keys, dim=0).float()
            value = torch.cat(values, dim=0).float()
            scores = torch.mv(key, query[batch_idx, head_idx].float()) * args.scale
            probs = torch.softmax(scores, dim=0)
            reference[batch_idx, head_idx] = torch.mv(value.t(), probs)
    return reference


def _validate_pypto_reference_parity(module, tensors, args: argparse.Namespace) -> None:
    expected = torch.zeros_like(tensors["out"])
    tensors["out"] = expected
    module.golden(tensors)
    pypto_reference = _extract_logical_heads(expected, args).float()
    highperf_reference = _compute_highperf_reference_from_pypto_tensors(tensors, args)
    if not torch.allclose(pypto_reference, highperf_reference, rtol=1e-3, atol=1e-3):
        max_diff = (pypto_reference - highperf_reference).abs().max().item()
        raise RuntimeError(f"PyPTO/highperf reference parity failed: max diff={max_diff}")
    print("PYTPO_HIGHPERF_REFERENCE_PARITY passed", flush=True)


def _parse_highperf_timing(output: str) -> TimingResult:
    host_match = AVG_HOST_RE.search(output)
    if host_match is None:
        raise RuntimeError("Could not parse highperf Avg Host timing from scene-test output")
    device_match = AVG_DEVICE_RE.search(output)
    return TimingResult(
        host_us=float(host_match.group(1)),
        device_us=float(device_match.group(1)) if device_match is not None else None,
    )


def _parse_pypto_device_timings(output: str, rounds: int, scheduler_threads: int) -> list[float]:
    costs = [float(match.group(1)) for match in SCHED_COST_RE.finditer(output)]
    if not costs:
        return []

    group_size = max(1, scheduler_threads)
    groups = [costs[idx : idx + group_size] for idx in range(0, len(costs), group_size)]
    complete_groups = [group for group in groups if len(group) == group_size]
    if not complete_groups:
        return []

    measured_groups = complete_groups[-rounds:]
    return [max(group) for group in measured_groups]


def _parse_pypto_child_timing(output: str, args: argparse.Namespace) -> TimingResult:
    matches = list(PYTPO_ROUND_RE.finditer(output))
    host_samples = [float(match.group(1)) for match in matches]
    if not host_samples:
        raise RuntimeError("Could not parse PyPTO host timing from child output")
    event_samples = [float(match.group(2)) for match in matches if match.group(2) is not None]
    log_samples = _parse_pypto_device_timings(output, args.rounds, args.aicpu_thread_num - 1)
    return TimingResult(
        host_us=mean(host_samples),
        device_us=mean(event_samples) if event_samples else None,
        scheduler_log_us=mean(log_samples) if log_samples else None,
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


def _attention_matmul_flops(args: argparse.Namespace | ShapePreset) -> int:
    return 4 * args.batch * args.num_heads * args.context_len * args.head_dim


def _attention_memory_bytes(args: argparse.Namespace | ShapePreset) -> int:
    fp16_bytes = 2
    query_bytes = args.batch * args.num_heads * args.head_dim * fp16_bytes
    kv_cache_bytes = 2 * args.batch * args.context_len * args.num_kv_heads * args.head_dim * fp16_bytes
    output_bytes = args.batch * args.num_heads * args.head_dim * fp16_bytes
    return query_bytes + kv_cache_bytes + output_bytes


def _device_gflops(args: argparse.Namespace | ShapePreset, result: TimingResult) -> float | None:
    if result.device_us is None or result.device_us <= 0:
        return None
    return _attention_matmul_flops(args) / (result.device_us * 1_000.0)


def _device_bandwidth_gbs(args: argparse.Namespace | ShapePreset, result: TimingResult) -> float | None:
    if result.device_us is None or result.device_us <= 0:
        return None
    return _attention_memory_bytes(args) / (result.device_us * 1_000.0)


def _device_log_snapshot(device: int) -> dict[Path, int]:
    log_dir = Path(f"/root/ascend/log/debug/device-{device}")
    if not log_dir.is_dir():
        return {}

    snapshot: dict[Path, int] = {}
    for log_path in log_dir.glob("device-*.log"):
        try:
            if log_path.is_file():
                snapshot[log_path] = log_path.stat().st_size
        except OSError:
            continue
    return snapshot


def _read_device_log_delta(device: int, snapshot: dict[Path, int]) -> str:
    log_dir = Path(f"/root/ascend/log/debug/device-{device}")
    if not log_dir.is_dir():
        return ""

    chunks = []
    for log_path in sorted(log_dir.glob("device-*.log")):
        try:
            old_size = snapshot.get(log_path, 0)
            new_size = log_path.stat().st_size
            if new_size <= old_size:
                continue
            with log_path.open("rb") as log_file:
                log_file.seek(old_size)
                chunks.append(log_file.read().decode(errors="ignore"))
        except OSError:
            continue
    return "".join(chunks)


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
    if not args.validate_results:
        command.append("--skip-golden")
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
    if args.validate_pypto:
        command.append("--validate-pypto")
    if args.validate_results:
        command.append("--validate-results")
    return command


def _run_pypto(args: argparse.Namespace) -> TimingResult:
    command = _pypto_child_command(args)
    print("\n== PyPTO SPMD example ==")
    print(" ".join(command))
    log_snapshot = _device_log_snapshot(args.device)
    proc = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    device_log_delta = _read_device_log_delta(args.device, log_snapshot)
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"PyPTO benchmark failed with exit code {proc.returncode}")
    return _parse_pypto_child_timing(proc.stdout + proc.stderr + device_log_delta, args)


def _run_pypto_child(args: argparse.Namespace) -> None:
    example_path = args.pypto_example
    module = _load_module(example_path)
    active_blocks = (args.context_len + args.block_size - 1) // args.block_size
    max_blocks = args.max_model_len // args.block_size

    pypto_num_heads = args.num_kv_heads * args.q_tile
    program = module.build_paged_attention_spmd_program(
        batch=args.batch,
        num_heads=pypto_num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        block_size=args.block_size,
        max_num_blocks_per_req=max_blocks,
        q_tile=args.q_tile,
    )
    tensor_specs = module.build_tensor_specs(
        batch=args.batch,
        num_heads=pypto_num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        block_size=args.block_size,
        max_num_blocks_per_req=max_blocks,
        active_num_blocks=active_blocks,
        context_len=args.context_len,
        scale=args.scale,
    )
    tensors, input_tensors = module._build_runtime_tensors(tensor_specs)
    _overwrite_pypto_inputs_like_highperf(tensors, args)
    if args.validate_results:
        _validate_pypto_reference_parity(module, tensors, args)
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

    output = compiled(*input_tensors, config=run_config)
    _sync_torch()
    if args.validate_pypto or args.validate_results:
        expected = torch.zeros_like(output)
        tensors["out"] = expected
        module.golden(tensors)
        if not torch.allclose(output, expected, rtol=2e-2, atol=2e-2):
            max_diff = (output - expected).abs().max().item()
            raise RuntimeError(f"PyPTO validation failed: max diff={max_diff}")

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
    parser.add_argument("--validate-pypto", action="store_true")
    parser.add_argument("--validate-results", action="store_true")
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


def _print_highperf_result(args: argparse.Namespace, result: TimingResult) -> None:
    print(f"highperf host e2e avg:        {result.host_us:.1f} us")
    device_gflops = _device_gflops(args, result)
    device_bandwidth_gbs = _device_bandwidth_gbs(args, result)
    if result.device_us is not None:
        print(f"highperf runtime device avg:  {result.device_us:.1f} us")
    if device_gflops is not None:
        print(f"highperf device throughput:   {device_gflops:.2f} GFLOP/s")
    if device_bandwidth_gbs is not None:
        print(
            f"highperf est. bandwidth:      {device_bandwidth_gbs:.2f} GB/s "
            f"({device_bandwidth_gbs / 1_000.0:.4f} TB/s)"
        )


def _print_pypto_result(args: argparse.Namespace, result: TimingResult) -> None:
    print(f"PyPTO host e2e avg:           {result.host_us:.1f} us")
    device_gflops = _device_gflops(args, result)
    device_bandwidth_gbs = _device_bandwidth_gbs(args, result)
    if result.device_us is not None:
        print(f"PyPTO torch.npu.Event avg:    {result.device_us:.1f} us")
    if device_gflops is not None:
        print(f"PyPTO device throughput:      {device_gflops:.2f} GFLOP/s")
    if device_bandwidth_gbs is not None:
        print(
            f"PyPTO est. bandwidth:         {device_bandwidth_gbs:.2f} GB/s "
            f"({device_bandwidth_gbs / 1_000.0:.4f} TB/s)"
        )


def _run_one_shape(args: argparse.Namespace) -> tuple[TimingResult | None, TimingResult | None]:
    print(f"Note: PyPTO runs from {args.pypto_example} with deterministic highperf-style inputs.")
    print("Note: host e2e is the primary comparison; PyPTO torch.npu.Event timing is secondary context.")
    highperf = None if args.skip_highperf else _run_highperf(args)
    pypto = None if args.skip_pypto else _run_pypto(args)

    print("\n== summary ==")
    if highperf is not None:
        _print_highperf_result(args, highperf)
    if pypto is not None:
        _print_pypto_result(args, pypto)
    if highperf is not None and pypto is not None:
        if highperf.host_us > 0:
            print(f"PyPTO / highperf host e2e:    {pypto.host_us / highperf.host_us:.2f}x")
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
        highperf_gflops = "n/a"
        if highperf is not None:
            highperf_gflops_value = _device_gflops(SHAPE_PRESETS[shape_name], highperf)
            if highperf_gflops_value is not None:
                highperf_gflops = f"{highperf_gflops_value:.2f}"
        highperf_gbs = "n/a"
        if highperf is not None:
            highperf_gbs_value = _device_bandwidth_gbs(SHAPE_PRESETS[shape_name], highperf)
            if highperf_gbs_value is not None:
                highperf_gbs = f"{highperf_gbs_value:.2f}"
        pypto_gflops = "n/a"
        if pypto is not None:
            pypto_gflops_value = _device_gflops(SHAPE_PRESETS[shape_name], pypto)
            if pypto_gflops_value is not None:
                pypto_gflops = f"{pypto_gflops_value:.2f}"
        pypto_gbs = "n/a"
        if pypto is not None:
            pypto_gbs_value = _device_bandwidth_gbs(SHAPE_PRESETS[shape_name], pypto)
            if pypto_gbs_value is not None:
                pypto_gbs = f"{pypto_gbs_value:.2f}"
        ratio = "n/a"
        if highperf is not None and pypto is not None and highperf.host_us > 0:
            ratio = f"{pypto.host_us / highperf.host_us:.2f}x"
        print(
            f"{shape_name}: highperf_host_us={highperf_host} "
            f"pypto_host_us={pypto_host} pypto_npu_event_us={pypto_event} "
            f"highperf_device_gflops={highperf_gflops} pypto_device_gflops={pypto_gflops} "
            f"highperf_est_gbs={highperf_gbs} pypto_est_gbs={pypto_gbs} "
            f"pypto/highperf={ratio}"
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
    if args.validate_results:
        args.validate_pypto = True
    shape_names = _selected_shape_names(args)
    if shape_names:
        _run_shape_suite(args, shape_names)
    else:
        _run_one_shape(args)


if __name__ == "__main__":
    main()
