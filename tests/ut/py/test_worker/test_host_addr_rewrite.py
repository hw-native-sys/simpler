# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-pointer blob rewrite must not touch child_memory (device) tensors.

``_rewrite_blob_host_addrs`` redirects registered host pointers in a task-args
blob into a forked child's own mapping. A tensor whose ``buffer.addr`` is a
child-owned device pointer (``child_memory == 1``) carries a device VA, which
may numerically fall inside a registered host range yet must never be rewritten
— doing so corrupts the device pointer. The rewrite therefore keys on the
per-tensor ``child_memory`` flag, not on the numeric address alone.
"""

import struct

from _task_interface import TENSOR_ADDRESS_SPACE_OFFSET, TENSOR_STRIDE_BYTES
from simpler.worker import _BLOB_HEADER_BYTES, _rewrite_blob_host_addrs

_PARENT_LO = 0x7F00_0000_0000
_PARENT_HI = 0x7F00_0010_0000
_CHILD_BASE = 0x7E00_0000_0000


def _make_blob(tensors: list[tuple[int, int]]) -> bytearray:
    """Build a task-args blob: [int32 T][int32 S][Tensor*T].

    ``tensors`` is a list of ``(addr, child_memory)``. Only the two fields the
    rewrite reads (buffer.addr at offset 0, child_memory at its struct offset)
    are populated; the rest of each 128-byte tensor stays zero.
    """
    n = len(tensors)
    buf = bytearray(_BLOB_HEADER_BYTES + n * TENSOR_STRIDE_BYTES)
    struct.pack_into("<i", buf, 0, n)  # tensor count
    struct.pack_into("<i", buf, 4, 0)  # scalar count
    for i, (addr, child_mem) in enumerate(tensors):
        off = _BLOB_HEADER_BYTES + i * TENSOR_STRIDE_BYTES
        struct.pack_into("<Q", buf, off, addr)
        struct.pack_into("<B", buf, off + TENSOR_ADDRESS_SPACE_OFFSET, child_mem)
    return buf


def _tensor_addr(buf: bytearray, i: int) -> int:
    off = _BLOB_HEADER_BYTES + i * TENSOR_STRIDE_BYTES
    return struct.unpack_from("<Q", buf, off)[0]


def test_host_tensor_in_range_is_rewritten():
    host_addr = _PARENT_LO + 0x100
    buf = _make_blob([(host_addr, 0)])
    _rewrite_blob_host_addrs(memoryview(buf), 0, [(_PARENT_LO, _PARENT_HI, _CHILD_BASE)])
    assert _tensor_addr(buf, 0) == _CHILD_BASE + 0x100


def test_child_memory_tensor_in_range_is_not_rewritten():
    """A device pointer numerically inside a host range must stay untouched."""
    device_addr = _PARENT_LO + 0x200
    buf = _make_blob([(device_addr, 1)])
    _rewrite_blob_host_addrs(memoryview(buf), 0, [(_PARENT_LO, _PARENT_HI, _CHILD_BASE)])
    assert _tensor_addr(buf, 0) == device_addr


def test_mixed_blob_rewrites_only_host_tensors():
    host_addr = _PARENT_LO + 0x100
    device_addr = _PARENT_LO + 0x200
    buf = _make_blob([(host_addr, 0), (device_addr, 1)])
    _rewrite_blob_host_addrs(memoryview(buf), 0, [(_PARENT_LO, _PARENT_HI, _CHILD_BASE)])
    assert _tensor_addr(buf, 0) == _CHILD_BASE + 0x100
    assert _tensor_addr(buf, 1) == device_addr
