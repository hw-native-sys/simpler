# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""What each built host runtime declares as its run-resource capacity.

Read from the loaded component rather than from the source, because the number a
context grants comes from the component the worker actually opens.

Two declarations per runtime, and they are independent. `get_pipeline_contract`
is the program path's maximum, clamped against the caller's request in
`ChipWorker::init`. `build_kernel_pipeline_contract_impl` is what
`simpler_kernel_mode_init` validates and discards; a kernel context's capacity
is not the program path's, so raising one must not move the other.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: Mirrors `PTO_PIPELINE_MAX_DEPTH` / `PTO_PIPELINE_MAX_RESOURCES` in runtime_c_api.h. The
#: assertions below name the depth each runtime declares, so a ceiling change shows up here as a
#: failing expectation rather than as a silently rescaled one.
_MAX_DEPTH = 3
_MAX_RESOURCES = 8
_CONTRACT_ABI_VERSION = 1

# RUNTIME_ENV_FIELD_GROUPS(3) * RUNTIME_ENV_RING_COUNT(4).
_RUNTIME_ENV_UINT64_FIELDS = 12
_OUTPUT_PREFIX_BYTES = 1024


class _PipelineResource(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_uint32),
        ("resource_class", ctypes.c_uint32),
        ("bytes_per_copy", ctypes.c_uint64),
    ]


class _PipelineContract(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("resource_count", ctypes.c_uint32),
        ("pipeline_depth", ctypes.c_uint32),
        ("resources", _PipelineResource * _MAX_RESOURCES),
    ]


class _CallConfig(ctypes.Structure):
    """Mirror of the packed CallConfig the C ABI takes by pointer."""

    _pack_ = 1
    _fields_ = [
        ("aicpu_thread_num", ctypes.c_int32),
        ("enable_chip_swimlane", ctypes.c_int32),
        ("enable_dump_args", ctypes.c_int32),
        ("enable_pmu", ctypes.c_int32),
        ("enable_dep_gen", ctypes.c_int32),
        ("enable_scope_stats", ctypes.c_int32),
        ("runtime_env", ctypes.c_uint64 * _RUNTIME_ENV_UINT64_FIELDS),
        ("output_prefix", ctypes.c_char * _OUTPUT_PREFIX_BYTES),
    ]


def _load(arch: str, runtime: str) -> ctypes.CDLL:
    path = _PROJECT_ROOT / "build" / "lib" / arch / "onboard" / runtime / "libhost_runtime.so"
    if not path.exists():
        pytest.skip(f"{path} not built")
    # RTLD_LOCAL, as the worker loads it: two runtimes export the same entry names, so a globally
    # scoped load would make the second one's calls land in the first.
    lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
    lib.get_pipeline_contract.restype = ctypes.POINTER(_PipelineContract)
    lib.get_pipeline_contract.argtypes = []
    lib.build_kernel_pipeline_contract_impl.restype = ctypes.c_int
    lib.build_kernel_pipeline_contract_impl.argtypes = [ctypes.POINTER(_CallConfig), ctypes.POINTER(_PipelineContract)]
    return lib


def _program_depth(lib: ctypes.CDLL) -> int:
    contract = lib.get_pipeline_contract()
    assert contract, "get_pipeline_contract returned null"
    assert contract.contents.abi_version == _CONTRACT_ABI_VERSION
    return int(contract.contents.pipeline_depth)


def _kernel_contract(lib: ctypes.CDLL) -> _PipelineContract:
    config = _CallConfig()
    out = _PipelineContract()
    rc = lib.build_kernel_pipeline_contract_impl(ctypes.byref(config), ctypes.byref(out))
    assert rc == 0, f"build_kernel_pipeline_contract_impl reported {rc}"
    return out


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a5"])
def test_a5_host_build_graph_declares_three_program_sets():
    """The program declaration that lets a Worker be granted a third run-resource set."""
    assert _program_depth(_load("a5", "host_build_graph")) == _MAX_DEPTH


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a5"])
def test_a5_host_build_graph_keeps_its_kernel_capacity_at_two():
    """The kernel declaration did not move with the program one.

    `simpler_kernel_mode_init` validates this contract before latching kernel mode, so a capacity
    it never asked for must not arrive through the program path's maximum.
    """
    kernel = _kernel_contract(_load("a5", "host_build_graph"))
    assert kernel.pipeline_depth == 2
    # Same bank topology as the program declaration, which is what the kernel path validates.
    program = _load("a5", "host_build_graph").get_pipeline_contract()
    assert kernel.resource_count == program.contents.resource_count
    for index in range(kernel.resource_count):
        assert kernel.resources[index].kind == program.contents.resources[index].kind
        assert kernel.resources[index].resource_class == program.contents.resources[index].resource_class


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a5"])
def test_a5_tensormap_and_ringbuffer_is_unchanged():
    """Nothing spilled onto the other a5 runtime, whose own capacity is not in this scope."""
    assert _program_depth(_load("a5", "tensormap_and_ringbuffer")) == 2


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
def test_a2a3_host_build_graph_still_declares_three():
    """The platform this capacity shipped on first keeps its declaration."""
    assert _program_depth(_load("a2a3", "host_build_graph")) == _MAX_DEPTH
