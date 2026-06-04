# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from ..models import KernelShape, TraceConfig
from .common import _prologue, _require_kernel


def render_kernel(config: TraceConfig) -> str:
    kernel = _require_kernel(config)
    prologue = _prologue(config)
    if kernel.shape == KernelShape.SPMD_MIX:
        return f"""#include <stdint.h>

#ifndef AICORE
#define AICORE [aicore]
#endif

#if defined(__DAV_CUBE__) || defined(__DAV_VEC__)
{prologue}#include \"{kernel.source_path}\"
#endif

extern \"C\" __global__ AICORE void replay_entry(
    __gm__ int64_t *aic_args, __gm__ int64_t *aiv_args
) {{
#if defined(__DAV_CUBE__)
    int32_t hw_idx = get_block_idx();
    kernel_entry(aic_args + static_cast<uint64_t>(hw_idx) * 50);
#endif
#if defined(__DAV_VEC__)
    int32_t lane_idx = static_cast<int32_t>(
        get_block_idx() * get_subblockdim() + get_subblockid());
    kernel_entry(aiv_args + static_cast<uint64_t>(lane_idx) * 50);
#endif
}}
"""

    include_guard = "__DAV_CUBE__" if kernel.shape == KernelShape.AIC_ONLY else "__DAV_VEC__"
    return f"""#include <stdint.h>

#ifndef AICORE
#define AICORE [aicore]
#endif

#if defined({include_guard})
{prologue}#include \"{kernel.source_path}\"
#endif

extern \"C\" __global__ AICORE void replay_entry(__gm__ int64_t *args) {{
#if defined({include_guard})
    kernel_entry(args);
#endif
}}
"""


def render_launch(config: TraceConfig) -> str:
    kernel = _require_kernel(config)
    if kernel.shape == KernelShape.SPMD_MIX:
        return f"""#include <stdint.h>
#ifndef AICORE
#define AICORE [aicore]
#endif

extern \"C\" __global__ AICORE void replay_entry(
    __gm__ int64_t *aic_args, __gm__ int64_t *aiv_args);

extern \"C\" void launch_replay(void *aic_args, void *aiv_args, void *stream) {{
    replay_entry<<<{config.hw_block_num}, nullptr, stream>>>(
        (__gm__ int64_t *)aic_args, (__gm__ int64_t *)aiv_args);
}}
"""

    return f"""#include <stdint.h>
#ifndef AICORE
#define AICORE [aicore]
#endif

extern \"C\" __global__ AICORE void replay_entry(__gm__ int64_t *args);

extern \"C\" void launch_replay(void *args, void *stream) {{
    replay_entry<<<{config.hw_block_num}, nullptr, stream>>>((__gm__ int64_t *)args);
}}
"""
