# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device-free tests for the SceneTest host-tensor rehost adapter.

Eager ``Worker.init()`` forks the L3 chip/sub children before ``generate_args``
runs, so the harness rehosts each host tensor into a born-shared child buffer.
These tests fake the buffer allocator to check the adapter's transactional
contract without any worker/hardware: value/shape fidelity, LIFO release,
partial-construction rollback, and non-contiguous rejection.
"""

from __future__ import annotations

from multiprocessing.shared_memory import SharedMemory

import pytest
import torch

from simpler_setup.scene_test import TaskArgsBuilder, Tensor, _RehostedTaskArgs


class _FakeHostBuffer:
    def __init__(self, nbytes: int):
        self.shm = SharedMemory(create=True, size=nbytes)
        self.buffer = self.shm.buf


class _FakeWorker:
    """Stands in for a started L3 Worker's host-buffer allocator."""

    def __init__(self, fail_on_create: int | None = None):
        self.created: list[_FakeHostBuffer] = []
        self.freed: list[_FakeHostBuffer] = []
        self._fail_on_create = fail_on_create

    def create_host_buffer(self, nbytes: int) -> _FakeHostBuffer:
        if self._fail_on_create is not None and len(self.created) >= self._fail_on_create:
            raise RuntimeError("injected create_host_buffer failure")
        buf = _FakeHostBuffer(nbytes)
        self.created.append(buf)
        return buf

    def free_host_buffer(self, buf: _FakeHostBuffer) -> None:
        self.freed.append(buf)
        buf.shm.close()
        buf.shm.unlink()


def test_rehost_preserves_values_and_frees_lifo():
    ta = TaskArgsBuilder(
        Tensor("a", torch.arange(4, dtype=torch.float32)),
        Tensor("b", torch.zeros(4, dtype=torch.float32)),
    )
    w = _FakeWorker()
    rehosted = _RehostedTaskArgs(w, ta)
    try:
        # Values preserved and the builder now points at born-shared views.
        assert torch.equal(ta.a, torch.arange(4, dtype=torch.float32))
        assert len(w.created) == 2
        # A write lands in the born-shared shm (what a child would read).
        ta.b.copy_(torch.full((4,), 5.0))
        shared = w.created[1].buffer
        assert shared is not None
        # SharedMemory may round its mapping up to a page, so read only the
        # tensor's own elements.
        assert list(memoryview(shared).cast("f"))[:4] == [5.0] * 4
    finally:
        rehosted.release()
    # Both buffers freed, and the builder is restored to plain host tensors.
    assert len(w.freed) == 2
    assert torch.equal(ta.a, torch.arange(4, dtype=torch.float32))


def test_rehost_partial_failure_rolls_back():
    orig_a = torch.zeros(4, dtype=torch.float32)
    orig_b = torch.ones(4, dtype=torch.float32)
    ta = TaskArgsBuilder(
        Tensor("a", orig_a),
        Tensor("b", orig_b),
        Tensor("c", torch.zeros(4, dtype=torch.float32)),
    )
    w = _FakeWorker(fail_on_create=2)  # third allocation fails
    with pytest.raises(RuntimeError, match="injected create_host_buffer failure"):
        _RehostedTaskArgs(w, ta)
    # The two successfully-created buffers are freed, and the builder is
    # restored to its original tensors (no half-rehosted state).
    assert len(w.freed) == 2
    assert ta.a is orig_a
    assert ta.b is orig_b
    assert [s.value for s in ta.specs if isinstance(s, Tensor)][:2] == [orig_a, orig_b]


def test_rehost_rejects_aliased_tensors():
    base = torch.zeros(8, dtype=torch.float32)
    ta = TaskArgsBuilder(Tensor("a", base[:4]), Tensor("b", base[2:6]))
    w = _FakeWorker()
    with pytest.raises(ValueError, match="alias overlapping storage"):
        _RehostedTaskArgs(w, ta)
    # Rejected before any allocation.
    assert w.created == []
    assert w.freed == []


def test_rehost_skips_empty_tensor():
    empty = torch.zeros(0, dtype=torch.float32)
    ta = TaskArgsBuilder(Tensor("a", torch.arange(4, dtype=torch.float32)), Tensor("e", empty))
    w = _FakeWorker()
    rehosted = _RehostedTaskArgs(w, ta)
    try:
        # Only the non-empty tensor is rehosted; the empty one is left untouched.
        assert len(w.created) == 1
        assert ta.e is empty
    finally:
        rehosted.release()


def test_rehost_rejects_noncontiguous():
    noncontig = torch.zeros(4, 4, dtype=torch.float32)[:, ::2]
    assert not noncontig.is_contiguous()
    ta = TaskArgsBuilder(Tensor("a", noncontig))
    w = _FakeWorker()
    with pytest.raises(ValueError, match="contiguous"):
        _RehostedTaskArgs(w, ta)
    # Rejected before any allocation — nothing to leak.
    assert w.created == []
    assert w.freed == []
