# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host mapped-region counter Add uses the same uint32 wrap as the core view."""

import struct
import uuid
from multiprocessing.shared_memory import SharedMemory

import pytest
from _task_interface import (  # pyright: ignore[reportMissingImports]
    _region_counter_notify,
    _region_counter_test,
    _region_import_sim,
)
from simpler.comm_region import NotifyOp, WaitCmp


def _observed(handle: int) -> int:
    _matched, observed = _region_counter_test(handle, 0, 0, int(WaitCmp.EQ))
    return int(observed)


def _add(handle: int, value: int) -> int:
    _region_counter_notify(handle, 0, value, int(NotifyOp.Add))
    return _observed(handle)


@pytest.fixture
def sim_counter():
    token = "w8add" + uuid.uuid4().hex[:8]
    shm = SharedMemory(name=token, create=True, size=64)
    # The owner object closes the mapping when it is destroyed. The integer
    # alone does not keep that owner alive.
    owner = _region_import_sim(token, 64, "region-binding-test")
    try:
        yield shm, int(owner)
    finally:
        del owner
        shm.close()
        shm.unlink()


def _store(shm: SharedMemory, value: int) -> None:
    buf = shm.buf
    assert buf is not None
    struct.pack_into("<i", buf, 0, value)


def test_host_counter_add_keeps_small_positive_and_negative_deltas(sim_counter):
    shm, handle = sim_counter
    _store(shm, 5)
    assert _add(handle, 3) == 8
    assert _add(handle, -2) == 6


def test_host_counter_add_wraps_through_int32_limits(sim_counter):
    shm, handle = sim_counter
    _store(shm, 0x7FFFFFFF)
    assert _add(handle, 1) == -2147483648
    assert _add(handle, -1) == 0x7FFFFFFF
    _store(shm, -1)
    assert _add(handle, 1) == 0
    assert _add(handle, -1) == -1
