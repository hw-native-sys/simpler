# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The Host loader must only request symbols exported by its device image."""

import ctypes
import subprocess
from pathlib import Path

import pytest
from test_kernel_mode_c_api import _load


@pytest.mark.parametrize("arch", ["a2a3", "a5"])
def test_hbg_requested_device_entries_exist(arch):
    lib = _load(arch, "onboard", "host_build_graph")
    query = lib.runtime_extra_aicpu_symbols
    query.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
    query.restype = ctypes.POINTER(ctypes.c_char_p)
    count = ctypes.c_size_t()
    entries = query(ctypes.byref(count))
    image = Path(__file__).resolve().parents[3] / "build/lib" / arch / "onboard/host_build_graph/libaicpu_kernel.so"
    symbols = subprocess.check_output(["nm", "-D", "--defined-only", str(image)], text=True)
    exported = {line.split()[-1] for line in symbols.splitlines() if line.split()}
    missing = [entries[i].decode() for i in range(count.value) if entries[i].decode() not in exported]
    assert not missing, f"Host loader requests absent AICPU entries: {missing}"
