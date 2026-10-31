# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile the shared vector-add example without importing its Torch runner."""

import ctypes
from pathlib import Path

import pytest
from simpler.task_interface import ArgDirection, ChipCallable, CoreCallable

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

HOST_RUNTIMES = (
    pytest.param("tensormap_and_ringbuffer", marks=pytest.mark.runtime("tensormap_and_ringbuffer")),
    pytest.param("host_build_graph", marks=pytest.mark.runtime("host_build_graph")),
)


def require_a3_host_fabric(platform):
    if platform == "a2a3":
        acl = ctypes.CDLL("libascendcl.so")
        acl.aclrtGetSocName.restype = ctypes.c_char_p
        soc = acl.aclrtGetSocName()
        assert soc is not None, "aclrtGetSocName failed"
        if soc.startswith(b"Ascend910B"):
            pytest.skip(f"A3 Host Fabric is not qualified on {soc.decode()}")


def build_chip_callable(platform, runtime="tensormap_and_ringbuffer", *, kernel_source=None):
    root = Path(__file__).resolve().parents[4]
    kernels = root / "examples/workers/l3/multi_chip_dispatch/kernels"
    if runtime == "host_build_graph":
        orchestration_source = (
            root / "examples/workers/l4/compute_then_tload_mixed_l3/kernels/orchestration/local_add_orch.cpp"
        )
        func_name = "local_add_orchestration"
    else:
        orchestration_source = kernels / "orchestration/vector_add_orch.cpp"
        func_name = "vector_add_orchestration"
    compiler = KernelCompiler(platform=platform)
    binary = compiler.compile_incore(
        source_path=str(kernel_source if kernel_source is not None else kernels / "aiv/vector_add_kernel.cpp"),
        core_type="aiv",
        pto_isa_root=ensure_pto_isa_root(),
        extra_include_dirs=compiler.get_orchestration_include_dirs(runtime),
    )
    if not platform.endswith("sim"):
        binary = extract_text_section(binary)
    signature = [ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT]
    orchestration = compiler.compile_orchestration(runtime_name=runtime, source_path=str(orchestration_source))
    return ChipCallable.build(
        signature=signature,
        func_name=func_name,
        binary=orchestration,
        children=[(0, CoreCallable.build(signature=signature, binary=binary))],
    )
