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

import ctypes
from multiprocessing.shared_memory import SharedMemory

import pytest
import torch

from simpler_setup.scene_test import Scalar, TaskArgsBuilder, Tensor, _RehostedTaskArgs


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


# ---------------------------------------------------------------------------
# Characterization (partial) — SceneTest host-tensor rehost adapter.
#
# These pin CURRENT behavior of the Python rehost / arg-build layer as input to
# a planned typed-BufferHandle change (that planning context lives in the PR
# description). They assert what is, not what should be:
#   - a size-1 dimension's stride is normalized away, and
#   - non-overlapping views of one storage are split into independent buffers.
#
# SCOPE: they cover the rehost adapter and `make_tensor_arg` output only. They
# do NOT exercise the C++ orchestrator's dependency-key derivation
# (`TensorKey::local_host` in src/common/hierarchical/orchestrator.cpp), which
# runs inside a real submit and is out of device-free reach here. The
# canonical-identity / dependency-key behavior is therefore NOT characterized by
# this file — that remains open work.
# ---------------------------------------------------------------------------


def _make_tensor_arg(t):
    # Imported lazily: torch_interop pulls in the task_interface binding, which
    # the builder-only tests above do not need.
    from simpler_setup.torch_interop import make_tensor_arg  # noqa: PLC0415

    return make_tensor_arg(t)


def test_char_singleton_stride_dropped_by_make_tensor_arg():
    # The wire Tensor's strides are a pure function of shape (row-major), so a
    # size-1 dimension's stride value never survives into the Tensor. Two tensors
    # equal in shape+values but differing ONLY in their singleton-dim stride get
    # identical consumer-visible geometry (shape / stride / start_offset); the
    # inactive-dim and padding bytes are unspecified, so this is a geometry
    # equality, not a byte-for-byte one.
    base = torch.arange(4, dtype=torch.float32)
    weird = base.as_strided((4, 1), (1, 7))  # singleton-dim stride = 7
    canon = base.as_strided((4, 1), (1, 1))  # singleton-dim stride = 1
    # Pin the fixture strides so a future torch that normalizes them at
    # construction fails here loudly instead of silently voiding the coverage.
    assert weird.stride() == (1, 7)
    assert canon.stride() == (1, 1)
    # Both are contiguous per torch (a size-1 dim's stride is free), so
    # make_tensor_arg accepts them rather than rejecting as non-contiguous.
    assert weird.is_contiguous() and canon.is_contiguous()
    assert torch.equal(weird, canon)

    w_weird = _make_tensor_arg(weird)
    w_canon = _make_tensor_arg(canon)
    # The 7 is normalized to the row-major 1; consumer-visible geometry matches.
    assert tuple(w_weird.strides) == (1, 1)
    assert tuple(w_weird.strides) == tuple(w_canon.strides)
    assert tuple(w_weird.shapes) == tuple(w_canon.shapes)
    assert w_weird.start_offset == w_canon.start_offset == 0


def test_char_singleton_stride_dropped_by_rehost():
    # The rehost `.view(shape)` re-canonicalizes strides, so a non-canonical
    # singleton-dim stride on the input host tensor is lost after rehosting into
    # the born-shared buffer.
    base = torch.arange(4, dtype=torch.float32)
    weird = base.as_strided((4, 1), (1, 7))
    assert weird.stride() == (1, 7)  # pin fixture (see note above)
    ta = TaskArgsBuilder(Tensor("a", weird))
    w = _FakeWorker()
    rehosted = _RehostedTaskArgs(w, ta)
    try:
        assert ta.a.stride() == (1, 1)  # input singleton stride 7 → normalized
        assert torch.equal(ta.a, weird)
    finally:
        rehosted.release()


def test_char_nonoverlapping_shared_storage_not_rejected():
    # Two non-overlapping views of ONE backing storage are not rejected (the
    # alias guard tests byte-range overlap, not shared backing) and rehost into
    # INDEPENDENT born-shared buffers — the shared-backing relationship is not
    # preserved across the rehost.
    base = torch.zeros(8, dtype=torch.float32)
    a, b = base[:4], base[4:]  # non-overlapping byte ranges, same storage
    assert a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
    ta = TaskArgsBuilder(Tensor("a", a), Tensor("b", b))
    w = _FakeWorker()
    rehosted = _RehostedTaskArgs(w, ta)
    try:
        # Not rejected; split into two independent born-shared buffers.
        assert len(w.created) == 2
        assert ta.a.untyped_storage().data_ptr() != ta.b.untyped_storage().data_ptr()
    finally:
        rehosted.release()


def test_char_make_tensor_arg_carries_backing_as_addr_only():
    # `make_tensor_arg` records backing location solely as the raw address
    # (`data` == buffer.addr); there is no separate backing-identity field. Two
    # non-overlapping views of one storage therefore get DISTINCT `data` values,
    # offset by the view's byte offset.
    #
    # NOTE: this characterizes make_tensor_arg's output only, NOT the
    # orchestrator's dependency key. It does not, on its own, pin what a future
    # typed handle must change — a handle could add a canonical identity while
    # leaving these addresses as-is. The dependency-key path is not exercised
    # here (see SCOPE above).
    base = torch.zeros(8, dtype=torch.float32)
    a, b = base[:4], base[4:]  # both 1-D contiguous slices
    w_a = _make_tensor_arg(a)
    w_b = _make_tensor_arg(b)
    assert w_b.data - w_a.data == 4 * a.element_size()
    assert w_a.data != w_b.data


# ---------------------------------------------------------------------------
# TaskArgsBuilder duplicate-name fail-fast
# ---------------------------------------------------------------------------


def test_builder_constructor_rejects_duplicate_tensor():
    with pytest.raises(ValueError, match="duplicate argument name 'a'"):
        TaskArgsBuilder(
            Tensor("a", torch.zeros(4)),
            Tensor("a", torch.ones(4)),
        )


def test_builder_constructor_rejects_tensor_scalar_name_clash():
    with pytest.raises(ValueError, match="duplicate argument name 'x'"):
        TaskArgsBuilder(
            Tensor("x", torch.zeros(4)),
            Scalar("x", ctypes.c_float(1.0)),
        )


def test_builder_rejects_name_shadowing_builder_attribute():
    # A name that resolves to a real attribute/method would shadow it, so
    # `args.specs` returns the property instead of the argument. Reject it.
    with pytest.raises(ValueError, match="conflicts with builder attributes/methods"):
        TaskArgsBuilder(Tensor("specs", torch.zeros(4)))
    with pytest.raises(ValueError, match="conflicts with builder attributes/methods"):
        TaskArgsBuilder(Scalar("clone", ctypes.c_int64(1)))
    # A name that is not a builder member is still accepted.
    ta = TaskArgsBuilder(Tensor("value", torch.zeros(4)))
    assert torch.equal(ta.value, torch.zeros(4))


def test_builder_incremental_add_rejects_duplicate():
    ta = TaskArgsBuilder(Tensor("a", torch.zeros(4)))
    with pytest.raises(ValueError, match="duplicate argument name 'a'"):
        ta.add_tensor("a", torch.ones(4))


def test_builder_duplicate_scalar_leaves_state_unchanged():
    # A rejected duplicate scalar must not flip `_has_scalar`, so a legal tensor
    # can still be added afterwards (tensor-before-scalar ordering intact).
    ta = TaskArgsBuilder(Tensor("a", torch.zeros(4)))
    ta.add_scalar("s", ctypes.c_int64(7))
    with pytest.raises(ValueError, match="duplicate argument name 's'"):
        ta.add_scalar("s", ctypes.c_int64(9))
    # State unchanged: names, order, and stored values are exactly as before.
    assert [s.name for s in ta.specs] == ["a", "s"]
    assert ta.s.value == 7

    # A rejected duplicate scalar must not set `_has_scalar` before the check,
    # or the tensor-before-scalar guard would spuriously block a later legal
    # tensor. Fresh tensor-only builder → reject a name-clashing scalar → a
    # subsequent add_tensor must still succeed. (Catches moving the
    # `_has_scalar = True` assignment ahead of the duplicate check.)
    tb = TaskArgsBuilder(Tensor("a", torch.zeros(4)))
    orig_a = tb.a
    with pytest.raises(ValueError, match="duplicate argument name 'a'"):
        tb.add_scalar("a", ctypes.c_int64(3))
    tb.add_tensor("b", torch.ones(4))
    assert [s.name for s in tb.specs] == ["a", "b"]
    assert tb.a is orig_a


def test_builder_valid_args_order_named_access_and_clone():
    ta = TaskArgsBuilder(
        Tensor("a", torch.arange(4, dtype=torch.float32)),
        Tensor("b", torch.ones(4, dtype=torch.float32)),
        Scalar("scale", ctypes.c_float(1.5)),
    )
    assert [s.name for s in ta.specs] == ["a", "b", "scale"]
    assert torch.equal(ta.a, torch.arange(4, dtype=torch.float32))
    assert ta.scale.value == 1.5

    clone = ta.clone()
    assert [s.name for s in clone.specs] == ["a", "b", "scale"]
    assert torch.equal(clone.a, torch.arange(4, dtype=torch.float32))
    # Clone is deep: mutating the clone does not touch the original.
    clone.a.copy_(torch.full((4,), 9.0))
    assert torch.equal(ta.a, torch.arange(4, dtype=torch.float32))
