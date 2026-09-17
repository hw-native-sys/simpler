# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real TMR ACLGraph numerics inside the ST's isolated process.

The caller owns the prepared context, callable and tensor allocations. Exceptions
must terminate that process without unwinding graph-visible device resources.
"""

import ctypes

_COUNT = 128 * 128  # Maximum tile size; shorter views leave a checked sentinel suffix.


def _check(code, operation):
    assert code == 0, f"kernel_capture {operation}: rc={code}"


class _TensorIO:
    """Test tensor storage is owned and released by the subprocess caller."""

    def __init__(self, lib, allocations):
        self.lib = lib
        self.allocations = allocations
        self.array = ctypes.c_float * _COUNT
        self.nbytes = ctypes.sizeof(self.array)

    def allocate(self):
        address = ctypes.c_void_p()
        _check(self.lib.aclrtMalloc(ctypes.byref(address), self.nbytes, 0), "allocate tensor")
        self.allocations.append(address)
        return address

    def write(self, address, values):
        host_input = self.array(*values)
        _check(self.lib.aclrtMemcpy(address, self.nbytes, host_input, self.nbytes, 1), "seed tensor")

    def verify(self, address, expected):
        host_output = self.array()
        _check(self.lib.aclrtMemcpy(host_output, self.nbytes, address, self.nbytes, 2), "read output")
        actual = list(host_output)
        if actual != expected:
            mismatch = next(i for i in range(_COUNT) if actual[i] != expected[i])
            raise AssertionError(
                f"kernel_capture address={address.value:#x} element={mismatch}: "
                f"actual={actual[mismatch]} expected={expected[mismatch]}"
            )


def _bind_capture_functions(lib):
    observer = ctypes.CDLL(None)
    for symbol, return_type in {
        "capture_observer_begin": None,
        "capture_observer_status": ctypes.c_int,
        "capture_observer_check_resident": ctypes.c_int,
        "capture_observer_core_launches": ctypes.c_uint64,
        "capture_observer_cpu_launches": ctypes.c_uint64,
        "capture_observer_kernel_args": ctypes.c_uint64,
        "capture_observer_runtime_args": ctypes.c_uint64,
        "capture_observer_regs": ctypes.c_uint64,
    }.items():
        function = getattr(observer, symbol)
        function.argtypes = []
        function.restype = return_type
    observer.capture_observer_begin()
    signatures = {
        "aclmdlRICaptureBegin": [ctypes.c_void_p, ctypes.c_int],
        "aclmdlRICaptureEnd": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
        "aclmdlRIExecuteAsync": [ctypes.c_void_p, ctypes.c_void_p],
        "aclmdlRIDestroy": [ctypes.c_void_p],
    }
    for symbol, argtypes in signatures.items():
        function = getattr(lib, symbol)
        function.argtypes = argtypes
        function.restype = ctypes.c_int
    return observer


def _check_resident(observer, host_launches):
    _check(observer.capture_observer_status(), "native launch observation")
    assert observer.capture_observer_core_launches() == host_launches
    assert observer.capture_observer_cpu_launches() == host_launches
    _check(observer.capture_observer_check_resident(), "resident address stability")
