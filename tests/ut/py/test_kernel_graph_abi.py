# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Graph ABI structural contract across every architecture/runtime/backend."""

import ctypes

import pytest
from test_kernel_mode_c_api import _load


@pytest.mark.parametrize("arch", ["a2a3", "a5"])
@pytest.mark.parametrize("runtime", ["host_build_graph", "tensormap_and_ringbuffer"])
@pytest.mark.parametrize("backend", ["sim", "onboard"])
def test_graph_abi_structural_and_uninitialized_contract(arch, runtime, backend):
    lib = _load(arch, backend, runtime)
    retain, release = lib.simpler_kernel_mode_retain_graph, lib.simpler_kernel_mode_release_graph
    retain.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
    release.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    retain.restype = release.restype = ctypes.c_int
    token = ctypes.c_uint64(999)
    assert retain(None, ctypes.byref(token)) == -1004
    assert token.value == 0
    assert retain(None, None) == -1004
    assert release(None, 0) == -1004
    ctx = lib.create_device_context()
    try:
        assert retain(ctx, None) == -1004
        assert release(ctx, 0) == -1004
        supported = backend == "onboard" and arch == "a2a3" and runtime == "host_build_graph"
        assert lib.simpler_kernel_mode_supported(None) == int(supported)
        expected = -1003 if supported else -1001
        assert retain(ctx, ctypes.byref(token)) == expected
        assert token.value == 0
        assert release(ctx, 1) == expected
    finally:
        lib.destroy_device_context(ctx)
