# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Scene test setup — compilation toolchain, test framework, and platform discovery."""

from .elf_parser import extract_text_section
from .kernel_compiler import KernelCompiler
from .platform_info import parse_platform
from .pto_isa import ensure_pto_isa_root
from .runtime_builder import RuntimeBuilder
from .scene_test import CallableNamespace, Scalar, SceneTestCase, TaskArgsBuilder, Tensor, scene_test
from .torch_interop import make_tensor_arg, torch_dtype_to_datatype

__all__ = [
    "CallableNamespace",
    "KernelCompiler",
    "RuntimeBuilder",
    "Scalar",
    "SceneTestCase",
    "Tensor",
    "TaskArgsBuilder",
    "ensure_pto_isa_root",
    "extract_text_section",
    "make_tensor_arg",
    "parse_platform",
    "scene_test",
    "torch_dtype_to_datatype",
]
