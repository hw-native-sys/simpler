# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side staging for post-fork zero-copy host buffers.

``submit_next_level`` calls ``_stage_host_buffers_for_chip_submit`` before any
dispatch. A ``create_host_buffer`` buffer is born-shared — its bytes already
live in the child-visible shm and the child writes results back into the same
pages — so staging copies in neither direction; it only validates that each
in-range view fits inside its buffer. These are pure host-side unit tests: they
inject the staging state directly and never fork, compile a kernel, or touch a
device. The end-to-end round-trip lives in the a2a3sim scene test.
"""

from __future__ import annotations

import ctypes

import pytest
import torch
from simpler.task_interface import TaskArgs, TensorArgType
from simpler.worker import Worker, _HostBufEntry

from simpler_setup import make_tensor_arg

_SIZE = 128 * 128


def _staging_worker(entry):
    """A ``Worker`` with only the host-buffer staging state populated.

    ``_stage_host_buffers_for_chip_submit`` / ``_find_host_buf_entry`` read only
    ``_host_buf_registry`` and ``_host_buf_sorted_ptrs``.
    """
    w = Worker.__new__(Worker)
    w._host_buf_registry = {entry.data_ptr: entry}
    w._host_buf_sorted_ptrs = [entry.data_ptr]
    return w


def _entry_for(parent, shm_buf):
    """A born-shared entry mapping ``parent``'s VA onto a caller-owned ``shm_buf``
    (a ctypes buffer standing in for the child-visible shm)."""
    return _HostBufEntry(
        token=1,
        data_ptr=parent.data_ptr(),
        nbytes=parent.numel() * parent.element_size(),
        shm=None,  # type: ignore[arg-type]  # staging reads only shm_base
        shm_name="",
        shm_base=ctypes.addressof(shm_buf),
    )


def _arg(tensor, tag):
    ta = TaskArgs()
    ta.add_tensor(make_tensor_arg(tensor), tag)
    return ta


class TestZeroCopyStagingSkip:
    """A born-shared buffer (``create_host_buffer``) copies in neither direction.

    The user builds the tensor over the child-visible shm itself (via
    ``frombuffer`` on ``HostBuffer.buffer``), so its bytes are already where the
    child reads them and the child writes results back into the same pages.
    Staging must skip both the H2D copy-in and the D2H copy-out.
    """

    def test_input_is_not_copied_in(self):
        parent = torch.full((_SIZE,), 7.0, dtype=torch.float32)
        shm_buf = (ctypes.c_float * _SIZE)(*([3.0] * _SIZE))
        w = _staging_worker(_entry_for(parent, shm_buf))
        w._stage_host_buffers_for_chip_submit(_arg(parent, TensorArgType.INPUT))
        # No copy-in: the shm keeps its own bytes, nothing is mirrored in.
        assert list(shm_buf[:4]) == [3.0, 3.0, 3.0, 3.0]

    def test_output_leaves_shm_untouched(self):
        parent = torch.zeros(_SIZE, dtype=torch.float32)
        shm_buf = (ctypes.c_float * _SIZE)(*([5.0] * _SIZE))
        w = _staging_worker(_entry_for(parent, shm_buf))
        w._stage_host_buffers_for_chip_submit(_arg(parent, TensorArgType.OUTPUT))
        # No copy-out queued: staging does not touch the shm at all.
        assert list(shm_buf[:4]) == [5.0, 5.0, 5.0, 5.0]

    def test_view_overrun_raises(self):
        # A view running past the buffer would read past the child's shm mapping,
        # so the overrun guard in _find_host_buf_entry must still fire.
        parent = torch.zeros(_SIZE, dtype=torch.float32)
        shm_buf = (ctypes.c_float * (_SIZE // 2))()  # buffer half the tensor's size
        entry = _entry_for(parent, shm_buf)
        entry.nbytes = (_SIZE // 2) * parent.element_size()
        w = _staging_worker(entry)
        with pytest.raises(RuntimeError, match="overruns its host buffer"):
            w._stage_host_buffers_for_chip_submit(_arg(parent, TensorArgType.INPUT))

    def test_subview_inside_buffer_is_accepted(self):
        # A sub-view (addr = base + offset) that fits inside the buffer must not
        # raise, and — being zero-copy — must not touch the shm.
        parent = torch.zeros(256, dtype=torch.float32)
        shm_buf = (ctypes.c_float * 256)(*([9.0] * 256))
        w = _staging_worker(_entry_for(parent, shm_buf))
        w._stage_host_buffers_for_chip_submit(_arg(parent[64:128], TensorArgType.INPUT))
        assert list(shm_buf[:4]) == [9.0, 9.0, 9.0, 9.0]
