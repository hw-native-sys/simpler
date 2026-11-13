# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Callable construction and deterministic FFN inputs for WSE-owned scenes."""

import torch
from simpler.task_interface import ChipCallable, CoreCallable

from simpler_setup.kernel_compiler import KernelCompiler


def build_callable(platform, directory, orchestration, signature, kernels, extra_include_dirs=()):
    compiler = KernelCompiler(platform)
    includes = compiler.get_orchestration_include_dirs("host_build_graph") + list(extra_include_dirs)
    children = []
    for index, (source, arguments) in enumerate(kernels):
        binary = compiler.compile_incore(str(directory / source), core_type="", extra_include_dirs=includes)
        children.append((index, CoreCallable.build(signature=arguments, binary=binary)))
    return ChipCallable.build(
        signature=signature,
        func_name="aicpu_orchestration_entry",
        binary=compiler.compile_orchestration("host_build_graph", str(directory / orchestration)),
        children=children,
    )


def ffn_inputs(seed, shards):
    width = shards * 32
    row = torch.arange(128).reshape(-1, 1)
    k = torch.arange(32).reshape(1, -1)
    x = ((row + 2 * k + seed) % 5 - 2).float()
    w1 = ((2 * torch.arange(32).reshape(-1, 1) + 3 * torch.arange(width).reshape(1, -1) + seed) % 7 - 3).float()
    w2 = ((3 * torch.arange(width).reshape(-1, 1) + torch.arange(width).reshape(1, -1) + seed) % 5 - 2).float()
    h = x @ w1
    a = torch.relu(h)
    y = a @ w2
    inputs = [
        x,
        w1.reshape(32, shards, 32).permute(1, 2, 0).contiguous(),
        w2.reshape(shards, 32, shards, 32).permute(2, 0, 3, 1).contiguous(),
    ]
    return inputs, [t.reshape(128, shards, 32).permute(1, 0, 2).contiguous() for t in (h, a, y)]
