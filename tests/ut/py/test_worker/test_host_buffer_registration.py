# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side handling for post-fork zero-copy host buffers.

``submit_next_level`` calls ``_stage_host_buffers_for_chip_submit`` before any
dispatch. A ``create_host_buffer`` buffer is born-shared — its bytes already
live in the child-visible shm and the child writes results back into the same
pages — so staging copies in neither direction; it only validates that each
in-range view fits inside its buffer. The SubWorker tests also exercise
MAP/UNMAP control handling and parent-to-child address rewriting directly
against the child loop. These tests never compile a kernel or touch a device;
the end-to-end chip round-trip lives in the a2a3sim scene test.
"""

from __future__ import annotations

import ctypes
import struct
from multiprocessing.shared_memory import SharedMemory

import pytest
import simpler.worker as worker_mod
import torch
from simpler.task_interface import MAILBOX_SIZE, TaskArgs, TensorArgType, WorkerType
from simpler.worker import Worker, _HostBufEntry

from simpler_setup import make_tensor_arg

_SIZE = 128 * 128


def _staging_worker(entry):
    """A ``Worker`` with only the host-buffer staging state populated.

    ``_stage_host_buffers_for_chip_submit`` / ``_find_host_buf_entry`` read only
    ``_host_buf_registry`` (write side) and ``_host_buf_snapshot`` (the lock-free
    read side).
    """
    w = Worker.__new__(Worker)
    w._host_buf_registry = {entry.data_ptr: entry}
    w._host_buf_snapshot = ((entry.data_ptr,), {entry.data_ptr: entry})
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


def test_host_buffer_control_reaches_l3_sub_workers():
    calls = []

    class FakeWorker:
        def broadcast_control_all(self, worker_type, sub_cmd, payload, digest, *, timeout_s):
            calls.append((worker_type, sub_cmd, bytes(payload), digest, timeout_s))
            return []

    worker = Worker(level=3)
    worker._lifecycle = worker_mod._Lifecycle.READY
    worker._chip_shms = [None]
    worker._worker = FakeWorker()
    worker._start_hierarchical = lambda: None

    handle = worker.create_host_buffer(8)
    try:
        assert [(call[0], call[1]) for call in calls] == [
            (WorkerType.NEXT_LEVEL, worker_mod._CTRL_MAP_HOST),
            (WorkerType.SUB, worker_mod._CTRL_MAP_HOST),
        ]
    finally:
        handle.buffer.release()
        worker.free_host_buffer(handle)

    assert [(call[0], call[1]) for call in calls[2:]] == [
        (WorkerType.NEXT_LEVEL, worker_mod._CTRL_UNMAP_HOST),
        (WorkerType.SUB, worker_mod._CTRL_UNMAP_HOST),
    ]


def test_l3_sub_worker_maps_rewrites_and_unmaps_host_buffer(monkeypatch):
    mailbox = SharedMemory(create=True, size=MAILBOX_SIZE)
    host = SharedMemory(create=True, size=8)
    staged = None
    mailbox_buf = mailbox.buf
    host_buf = host.buf
    assert mailbox_buf is not None
    assert host_buf is not None

    try:
        struct.pack_into("<Q", host_buf, 0, 41)
        parent_addr = worker_mod._shm_base_addr(host)
        payload = worker_mod._HOST_BUF_MAP_HEADER.pack(7, parent_addr, 8) + host.name.encode("utf-8")
        staged = SharedMemory(create=True, size=len(payload))
        assert staged.buf is not None
        staged.buf[: len(payload)] = payload

        struct.pack_into("<Q", mailbox_buf, worker_mod._OFF_CALLABLE, worker_mod._CTRL_MAP_HOST)
        struct.pack_into("<Q", mailbox_buf, worker_mod._CTRL_OFF_ARG0, len(payload))
        shm_name = staged.name.encode("utf-8")
        mailbox_buf[worker_mod._OFF_ARGS : worker_mod._OFF_ARGS + len(shm_name)] = shm_name
        struct.pack_into("<Q", mailbox_buf, worker_mod.MAILBOX_OFF_PROTOCOL, worker_mod.MAILBOX_PROTOCOL_MAGIC_VERSION)

        digest = bytes(range(32))
        # MAP, use the child mapping, UNMAP, verify rewriting stops, then exit.
        states = iter(
            (
                worker_mod._CONTROL_REQUEST,
                worker_mod._TASK_READY,
                worker_mod._CONTROL_REQUEST,
                worker_mod._TASK_READY,
                worker_mod._SHUTDOWN,
            )
        )
        called = []

        def load_state(_state_addr):
            state = next(states)
            if state == worker_mod._TASK_READY:
                start = worker_mod._OFF_TASK_CALLABLE_HASH
                mailbox_buf[start : start + len(digest)] = digest
                struct.pack_into("<ii", mailbox_buf, worker_mod._OFF_TASK_ARGS_BLOB, 1, 0)
                struct.pack_into("<Q", mailbox_buf, worker_mod._OFF_TASK_ARGS_BLOB + 8, parent_addr)
            elif state == worker_mod._CONTROL_REQUEST and called:
                unmap_payload = worker_mod._HOST_BUF_UNMAP.pack(7)
                assert staged is not None
                staged_buf = staged.buf
                assert staged_buf is not None
                staged_buf[: len(unmap_payload)] = unmap_payload
                staged_buf.release()
                struct.pack_into("<Q", mailbox_buf, worker_mod._OFF_CALLABLE, worker_mod._CTRL_UNMAP_HOST)
                struct.pack_into("<Q", mailbox_buf, worker_mod._CTRL_OFF_ARG0, len(unmap_payload))
                mailbox_buf[worker_mod._OFF_ARGS : worker_mod._OFF_ARGS + worker_mod._CTRL_SHM_NAME_BYTES] = (
                    b"\x00" * worker_mod._CTRL_SHM_NAME_BYTES
                )
                mailbox_buf[worker_mod._OFF_ARGS : worker_mod._OFF_ARGS + len(shm_name)] = shm_name
            return state

        def read_args(buf):
            child_addr = struct.unpack_from("<Q", buf, worker_mod._OFF_TASK_ARGS_BLOB + 8)[0]
            if called:
                assert child_addr == parent_addr
                return object()
            assert child_addr != parent_addr
            value = ctypes.c_uint64.from_address(child_addr)
            assert value.value == 41
            value.value = 42
            return object()

        monkeypatch.setattr(worker_mod, "_mailbox_load_i32", load_state)
        monkeypatch.setattr(worker_mod, "_mailbox_store_i32", lambda *_args: None)
        monkeypatch.setattr(worker_mod, "_read_args_from_mailbox", read_args)

        worker_mod._sub_worker_loop(
            mailbox_buf,
            {0: lambda _args: called.append(True)},
            {digest: 0},
            {digest: 1},
        )

        assert called == [True, True], worker_mod._read_error_msg(mailbox_buf)
        assert struct.unpack_from("<Q", host_buf, 0)[0] == 42
    finally:
        mailbox_buf.release()
        host_buf.release()
        mailbox.close()
        mailbox.unlink()
        host.close()
        host.unlink()
        if staged is not None:
            staged.close()
            staged.unlink()


class TestCreateHostBufferChildPrecondition:
    """A born-shared buffer needs a process child to attach into — chip OR sub.

    The buffer is broadcast to both NEXT_LEVEL and SUB children, so a sub-only
    L3 (no device_ids) is a valid host; only a truly childless L3 is rejected.
    """

    def test_sub_only_l3_allows_create_host_buffer(self):
        w = Worker(level=3, num_sub_workers=1)
        w.register(lambda args: None)
        w.init()
        try:
            buf = w.create_host_buffer(64)
            try:
                # SharedMemory may round the mapping up to a page.
                assert buf.buffer.nbytes >= 64
            finally:
                buf.buffer.release()
                w.free_host_buffer(buf)
        finally:
            w.close()

    def test_childless_l3_rejects_create_host_buffer(self):
        # No callable registered: a childless L3 with a pre-registered callable
        # is rejected earlier, at init() (see TestEligibleTargetPrecheck).
        w = Worker(level=3, num_sub_workers=0)
        w.init()
        try:
            with pytest.raises(RuntimeError, match="at least one forked chip or sub child"):
                w.create_host_buffer(64)
        finally:
            w.close()
