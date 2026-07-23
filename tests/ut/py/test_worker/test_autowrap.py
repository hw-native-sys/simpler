# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Submit-layer auto-wrap: Tensor args -> self-describing BufferRefs (P1-B 3c-2b-B4, dormant).

Exercises Worker._bufferref_for_tensor's backing selection without a real fork/run, using the same
FakeWorker harness as the host-buffer registration tests.
"""

import ctypes

import pytest
import simpler.worker as worker_mod
from simpler.buffer_handle import BackendKind
from simpler.task_interface import DataType, TensorArgType
from simpler.worker import Worker

_Tensor = worker_mod.Tensor


class _FakeWorker:
    def broadcast_control_all(self, *args, **kwargs):
        return []


def _l3_worker() -> Worker:
    w = Worker(level=3)
    w._lifecycle = worker_mod._Lifecycle.READY
    w._chip_shms = [None]
    w._worker = _FakeWorker()
    w._start_hierarchical = lambda: None
    return w


def _addr_of(buffer) -> int:
    return ctypes.addressof(ctypes.c_char.from_buffer(buffer))


def test_autowrap_host_buffer_input_is_posix_shm():
    w = _l3_worker()
    hb = w.create_host_buffer(256)
    try:
        addr = _addr_of(hb.buffer)
        t = _Tensor.make(addr + 16, (2, 4), DataType.FLOAT32)  # a sub-view of the host buffer
        ref = w._bufferref_for_tensor(t, TensorArgType.INPUT)
        assert ref.handle.backend_kind == BackendKind.POSIX_SHM
        assert ref.byte_offset == 16
        assert ref.shapes == (2, 4)
        # A second arg over the same host buffer reuses one canonical identity (dep-stable).
        t2 = _Tensor.make(addr, (8,), DataType.FLOAT32)
        assert w._bufferref_for_tensor(t2, TensorArgType.OUTPUT_EXISTING).handle.identity == ref.handle.identity
    finally:
        w._release_all_buffer_handles()
        hb.buffer.release()
        w.free_host_buffer(hb)


def test_autowrap_prefork_raw_input_is_fork_shm_memoized():
    w = _l3_worker()
    backing = ctypes.create_string_buffer(64)
    addr = _addr_of(backing)
    try:
        ref = w._bufferref_for_tensor(_Tensor.make(addr, (16,), DataType.INT32), TensorArgType.INPUT)
        assert ref.handle.backend_kind == BackendKind.FORK_SHM
        assert ref.byte_offset == 0
        # Same addr on a later submit -> same identity (dependency inference keys on identity).
        ref2 = w._bufferref_for_tensor(_Tensor.make(addr, (16,), DataType.INT32), TensorArgType.INPUT)
        assert ref2.handle.identity == ref.handle.identity
    finally:
        w._release_all_buffer_handles()


def test_autowrap_output_null_addr_creates_buffer():
    w = _l3_worker()
    try:
        t = _Tensor.make(0, (16,), DataType.FLOAT32)
        ref = w._bufferref_for_tensor(t, TensorArgType.OUTPUT)
        assert ref.handle.backend_kind == BackendKind.POSIX_SHM  # a fresh create_buffer'd intermediate
        assert ref.handle.nbytes == 64
    finally:
        w._release_all_buffer_handles()


def test_autowrap_rejects_writable_raw_tensor():
    w = _l3_worker()
    backing = ctypes.create_string_buffer(64)
    t = _Tensor.make(_addr_of(backing), (16,), DataType.INT32)
    try:
        with pytest.raises(ValueError, match="writable raw"):
            w._bufferref_for_tensor(t, TensorArgType.OUTPUT)
    finally:
        w._release_all_buffer_handles()
