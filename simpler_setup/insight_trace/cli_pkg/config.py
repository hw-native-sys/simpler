# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

from simpler_setup.environment import PROJECT_ROOT
from simpler_setup.pto_isa import ensure_pto_isa_root

from ..models import (
    KernelShape,
    PlatformArch,
    PlatformFamily,
    SPMDDispatch,
    SPMDReplayMeta,
    TraceBackend,
    TraceConfig,
)


def build_trace_config(args, context, kernel, trace_args):
    output_dir = args.output_dir or default_output_dir(args.case, kernel.name, datetime.now())

    platform = resolve_platform_arch(args.platform, context)
    if args.soc_version:
        platform = with_soc_version_override(platform, args.soc_version)

    return TraceConfig(
        backend=TraceBackend.SIMPLER,
        test_module=args.test_module.resolve() if args.test_module else None,
        case_name=args.case,
        kernel_spec=kernel,
        args=trace_args,
        output_dir=output_dir,
        repo_root=PROJECT_ROOT,
        cann_home=args.cann_home,
        pto_isa_root=pto_isa_root(args.pto_isa_root),
        platform_arch=platform,
        device=args.device,
        launch_count=args.launch_count,
        timeout=args.timeout,
        hw_block_num=hw_block_num(args, kernel),
        dry_run=args.dry_run,
        spmd_meta=spmd_meta(kernel, platform),
    )


def default_output_dir(case_name: str, kernel_name: str, now: datetime) -> Path:
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    safe_case = case_name.replace("/", "_")
    safe_kernel = kernel_name.replace("/", "_")
    return PROJECT_ROOT / "outputs" / f"insight_trace_{safe_case}_{safe_kernel}_{timestamp}"


def resolve_platform_arch(platform_arg: str, context) -> PlatformArch:
    if platform_arg not in ("a2a3", "a5"):
        raise ValueError(f"Unknown platform family: {platform_arg!r}")
    family = PlatformFamily.A5 if platform_arg == "a5" else PlatformFamily.A2A3

    if context is not None:
        case_platforms = context.case.get("platforms", [])
        if case_platforms and not any(p.startswith(platform_arg) for p in case_platforms):
            raise ValueError(f"Platform family {platform_arg!r} does not match case platforms {case_platforms}")

    return PlatformArch.for_family(family)


def with_soc_version_override(platform: PlatformArch, soc_version: str) -> PlatformArch:
    return replace(platform, soc_version=soc_version)


def hw_block_num(args, kernel) -> int:
    if kernel.shape == KernelShape.SPMD_MIX:
        if args.platform == PlatformFamily.A5.value and kernel.source_path.name == "kernel_spmd_mix.cpp":
            return 2 + 8 + 12 + 24 + 48
        return 24
    return args.hw_block_num


def spmd_meta(kernel, platform: PlatformArch | None = None):
    if kernel.shape != KernelShape.SPMD_MIX:
        return None
    if platform is not None and platform.family == PlatformFamily.A5:
        if kernel.source_path.name != "kernel_spmd_mix.cpp":
            return None
        return SPMDReplayMeta(
            hw_block_dim=24,
            aiv_lanes_per_core=2,
            dispatches=(
                SPMDDispatch(logical_block_num=2, scalar_overrides=((1, 0),)),
                SPMDDispatch(logical_block_num=8, scalar_overrides=((1, 6),)),
                SPMDDispatch(logical_block_num=12, scalar_overrides=((1, 30),)),
                SPMDDispatch(logical_block_num=24, scalar_overrides=((1, 66),)),
                SPMDDispatch(logical_block_num=48, scalar_overrides=((1, 138),)),
            ),
        )
    if kernel.source_path.name != "paged_attention_parallel.cpp":
        raise ValueError(f"No SPMD replay metadata for kernel {kernel.name}")
    # FIFO element sizes (bytes): sij_fifo=float32, pij_fifo=bfloat16, oi_fifo=float32
    SIJ_FIFO_ELEMENT_SIZE = 4
    PIJ_FIFO_ELEMENT_SIZE = 2
    OI_FIFO_ELEMENT_SIZE = 4
    fifo_depth = 2
    max_q_tile = 64
    max_block_size = 128
    head_dim = 128
    hw_block_dim = 24
    return SPMDReplayMeta(
        hw_block_dim=hw_block_dim,
        aiv_lanes_per_core=2,
        fifo_sizes=(
            max_q_tile * max_block_size * SIJ_FIFO_ELEMENT_SIZE * fifo_depth,
            max_q_tile * max_block_size * PIJ_FIFO_ELEMENT_SIZE * fifo_depth,
            max_q_tile * head_dim * OI_FIFO_ELEMENT_SIZE * fifo_depth,
        ),
    )


def pto_isa_root(path: Path | None) -> Path:
    if path is not None:
        return path.resolve()
    return Path(ensure_pto_isa_root()).resolve()
