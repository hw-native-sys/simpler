# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

from .args_pkg import _load_arg_spec, load_kernel_dump_args, resolve_builtin_args, resolve_scene_test_args
from .models import KernelSpec, SceneCaseContext, TraceArg


def resolve_args(
    context: SceneCaseContext | None,
    kernel: KernelSpec | None,
    arg_spec: Path | None = None,
    dump_dir: Path | None = None,
    task_id: str | None = None,
) -> tuple[TraceArg, ...]:
    """
    Resolve kernel args for insight trace.

    Priority:
    1. arg_spec file
    2. dump_dir (PR792 tensor_dump.json format)
    3. scene_test args
    4. builtin recipes (paged_attention only)
    """
    if arg_spec is not None:
        return _load_arg_spec(arg_spec)
    if dump_dir is not None:
        return load_kernel_dump_args(dump_dir, task_id)
    if context is None or kernel is None:
        return ()
    if _should_use_builtin_fallback(context):
        return resolve_builtin_args(context, kernel)
    try:
        return resolve_scene_test_args(context)
    except Exception:
        return resolve_builtin_args(context, kernel)


def _should_use_builtin_fallback(context: SceneCaseContext) -> bool:
    module_path = context.module_dir.as_posix()
    return "paged_attention" in module_path or "spmd_multiblock_mix" in module_path
