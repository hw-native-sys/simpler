# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

import re
from functools import cache
from pathlib import Path

from .models import KernelShape, KernelSpec, SceneCaseContext

_ARG_READ_RE = re.compile(r"args\[(\d+)\]")


def select_kernel(
    context: SceneCaseContext,
    kernel_name: str | None = None,
    func_id: int | None = None,
    kernel_source: str | None = None,
) -> KernelSpec:
    selectors = [kernel_name is not None, func_id is not None, kernel_source is not None]
    if sum(selectors) != 1:
        raise ValueError("Exactly one of --kernel, --func-id, or --kernel-source is required")

    incores = context.callable_spec.get("incores", [])
    selected = None
    for entry in incores:
        source = Path(entry.get("source", ""))
        if kernel_name is not None and entry.get("name") == kernel_name:
            selected = entry
            break
        if func_id is not None and entry.get("func_id") == func_id:
            selected = entry
            break
        if kernel_source is not None:
            requested = Path(kernel_source)
            if source == requested or source.name == requested.name or source.as_posix().endswith(requested.as_posix()):
                selected = entry
                break
    if selected is None:
        raise ValueError("Kernel selector did not match any CALLABLE['incores'] entry")

    source_path = Path(selected["source"]).resolve()
    shape = classify_kernel(selected.get("core_type", ""), source_path)
    return KernelSpec(
        name=selected.get("name", source_path.stem),
        func_id=int(selected["func_id"]),
        core_type=selected.get("core_type", ""),
        source_path=source_path,
        shape=shape,
    )


def classify_kernel(core_type: str, source_path: Path) -> KernelShape:
    source = _read_source(source_path)
    if "/kernels/mix/" in source_path.as_posix():
        return KernelShape.SPMD_MIX
    if "SPMD_LOCAL_CONTEXT_INDEX" in source or "SPMD_GLOBAL_CONTEXT_INDEX" in source:
        return KernelShape.SPMD_MIX
    if "get_block_idx(args)" in source or "get_sub_block_id(args)" in source or "get_block_num(args)" in source:
        return KernelShape.SPMD_MIX
    if "args[48]" in source or "args[49]" in source:
        return KernelShape.SPMD_MIX
    if core_type == "aic":
        return KernelShape.AIC_ONLY
    if core_type == "aiv":
        return KernelShape.AIV_ONLY
    if "/kernels/aic/" in source_path.as_posix():
        return KernelShape.AIC_ONLY
    if "/kernels/aiv/" in source_path.as_posix():
        return KernelShape.AIV_ONLY
    raise ValueError(f"Cannot classify kernel core type: {core_type!r}")


def read_arg_indices(source_path: Path) -> set[int]:
    return {int(match.group(1)) for match in _ARG_READ_RE.finditer(_read_source(source_path))}


def validate_single_task_kernel(kernel: KernelSpec) -> None:
    source = _read_source(kernel.source_path)
    if "kernel_entry" not in source or "int64_t *args" not in source:
        raise ValueError(f"Kernel does not look like kernel_entry(args): {kernel.source_path}")


@cache
def _read_source(source_path: Path) -> str:
    return source_path.resolve().read_text()
