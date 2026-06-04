# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Union


class PlatformFamily(str, Enum):
    A2A3 = "a2a3"
    A5 = "a5"


@dataclass(frozen=True)
class PlatformArch:
    family: PlatformFamily = PlatformFamily.A2A3
    soc_version: str = "dav_2201"
    cce_aicore_number: int = 220
    pto_arch_macro: str = "PTO_NPU_ARCH_A2A3"
    cce_aicore_arch: str = "dav-c220"
    prologue_event_id7: str = "((event_t)7)"
    prologue_pipe_fix: str = "((pipe_t)10)"
    runtime_include_roots: tuple[str, ...] = (
        "src/a2a3/runtime/tensormap_and_ringbuffer/runtime",
        "src/a2a3/runtime/tensormap_and_ringbuffer/common",
        "src/a2a3/runtime/tensormap_and_ringbuffer/orchestration",
    )
    platform_include_roots: tuple[str, ...] = ("src/a2a3/platform/include",)

    @staticmethod
    def for_family(family: PlatformFamily) -> PlatformArch:
        if family == PlatformFamily.A5:
            return _A5_PLATFORM
        return _A2A3_PLATFORM


_A2A3_PLATFORM = PlatformArch(
    family=PlatformFamily.A2A3,
    soc_version="dav_2201",
    cce_aicore_number=220,
    pto_arch_macro="PTO_NPU_ARCH_A2A3",
    cce_aicore_arch="dav-c220",
)

_A5_PLATFORM = PlatformArch(
    family=PlatformFamily.A5,
    soc_version="dav_3510",
    cce_aicore_number=310,
    pto_arch_macro="PTO_NPU_ARCH_A5",
    cce_aicore_arch="dav-c310",
    prologue_event_id7="((::event_t)7)",
    prologue_pipe_fix="((::pipe_t)10)",
    runtime_include_roots=(
        "src/a5/runtime/tensormap_and_ringbuffer/runtime",
        "src/a5/runtime/tensormap_and_ringbuffer/common",
        "src/a5/runtime/tensormap_and_ringbuffer/orchestration",
    ),
    platform_include_roots=("src/a5/platform/include",),
)


class KernelShape(str, Enum):
    AIC_ONLY = "aic-only"
    AIV_ONLY = "aiv-only"
    SPMD_MIX = "spmd-mix"


class TraceBackend(str, Enum):
    SIMPLER = "simpler"
    PTOAS = "ptoas"


@dataclass(frozen=True)
class KernelSpec:
    name: str
    func_id: int
    core_type: str
    source_path: Path
    shape: KernelShape


@dataclass(frozen=True)
class TraceTensorArg:
    index: int
    name: str
    dtype: str
    shape: tuple[int, ...]
    role: str = "input"
    fill: str = "zero"


@dataclass(frozen=True)
class TraceScalarArg:
    index: int
    name: str
    dtype: str
    value: int | float
    pack_mode: str = "value"


TraceArg = Union[TraceTensorArg, TraceScalarArg]


@dataclass(frozen=True)
class SPMDDispatch:
    logical_block_num: int
    scalar_overrides: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class SPMDReplayMeta:
    hw_block_dim: int = 24
    aiv_lanes_per_core: int = 2
    fifo_sizes: tuple[int, int, int] = (0, 0, 0)  # (sij_bytes, pij_bytes, oi_bytes) per hw block
    dispatches: tuple[SPMDDispatch, ...] = ()


@dataclass(frozen=True)
class TraceConfig:
    backend: TraceBackend
    test_module: Path | None
    case_name: str | None
    kernel_spec: KernelSpec | None
    args: tuple[TraceArg, ...]
    output_dir: Path
    repo_root: Path
    cann_home: Path | None
    pto_isa_root: Path | None
    platform_arch: PlatformArch = _A2A3_PLATFORM
    device: int = 0
    launch_count: int = 1
    timeout: int = 120
    hw_block_num: int = 1
    dry_run: bool = False
    spmd_meta: SPMDReplayMeta | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceResult:
    workspace_dir: Path
    simulator_dir: Path | None
    collect_log: Path | None
    export_log: Path | None


@dataclass(frozen=True)
class SceneCaseContext:
    test_class: type
    case: dict[str, Any]
    callable_spec: dict[str, Any]
    test_module: Path
    module_dir: Path
    runtime: str


@dataclass(frozen=True)
class PtoasTraceConfig:
    ptoas_root: Path
    source_cpp: Path
    testcase_name: str
    kernel_base_name: str
    aicore_arch: str
    output_dir: Path
    cann_home: Path | None
    pto_isa_root: Path | None
    soc_version: str = "dav_2201"
    timeout: int = 120
    launch_count: int = 1
    kernel_symbol: str | None = None


@dataclass(frozen=True)
class PtoasWorkspace:
    run_root: Path
    case_root: Path
    case_dir: Path
    build_dir: Path
    application: Path
    kernel_lib: Path
    kernel_symbol: str
