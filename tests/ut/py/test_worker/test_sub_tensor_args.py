# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Sub-worker task args over the Tensor wire (P1-B B3).

A Python sub callable is a compute leaf: it receives its args as MappedArgs (each backing mapped into
the sub child, map-once) and computes with torch.frombuffer(arg.buffer, ...). Writes land in the
shared backing the owner sees — no C++ Tensor involved. This is exactly the case a closure cannot
serve: a post-init create_buffer shm is not mapped in the pre-forked sub child, so the buffer must
arrive via args and be mapped from its Ref.
"""

import ctypes

import pytest
import torch
from simpler.buffer import (
    AccessMode,
    BackendKind,
    ImportContext,
    ImportRegistry,
    mint_owner_instance_id,
    wrap_fork_inherited,
)
from simpler.comm_endpoints import HOST_CPU
from simpler.task_interface import (
    CallConfig,
    ChipStorageTaskArgs,
    DataType,
    TaskArgs,
    TensorArgType,
)
from simpler.worker import Worker

from ._wire_blob import encode_blob

_F32 = 0  # DataType.FLOAT32 value


def test_alloc_shared_tensor_returns_managed_fork_handle():
    hw = Worker(level=3, num_sub_workers=1)
    hw.init()
    captured = {}
    try:
        # alloc_shared_tensor allocates from the orchestrator's HeapRing, so it must run inside
        # an active building run — drive it through a no-submit orch fn.
        def orch(o, _args, cfg):
            captured["h"] = hw.alloc_shared_tensor((4, 8), DataType.FLOAT32)

        hw.run(orch, args=None, config=CallConfig())
        h = captured["h"]
        assert h.nbytes == 4 * 8 * 4  # prod(shape) * element_size
        # Backed by the orchestrator's HeapRing VA (FORK_SHM, no POSIX shm), runtime-managed.
        assert h.backend_kind == BackendKind.FORK_SHM
        assert h.shm is None and h.base != 0
    finally:
        hw.close()


def test_create_buffer_at_l2_needs_no_child():
    # An L2 leaf has no forked children — it materializes the ref in-process itself — so create_buffer
    # must not require a child. The handle is a usable POSIX-shm backing.
    w = Worker(level=2)
    h = w._create_buffer_locked(64)
    try:
        assert h.nbytes == 64
        assert h.shm is not None
        t = torch.frombuffer(h.shm.buf, dtype=torch.float32, count=4)
        t.fill_(3.0)
        assert t.tolist() == [3.0, 3.0, 3.0, 3.0]
        t = None
        # The level guard now admits L2: the public create_buffer no longer TypeErrors on level, it
        # reaches the READY check (this Worker is uninitialized).
        with pytest.raises(RuntimeError, match="READY"):
            w.create_buffer(64)
    finally:
        h.close()


def test_l2_run_materializes_tensor_to_chip_args():
    # An L2 leaf consumes its own Tensor args: _run_l2_materialized resolves each ref to a local base
    # and hands the runtime the ChipStorageTaskArgs POD, exactly like a chip child, minus the mailbox.
    # Capture that POD via a fake ChipWorker and read its ChipTensors to prove the materialization.
    captured = {}

    class _FakeImpl:
        def run_materialized(self, cid, args, cfg):
            captured["cid"] = cid
            captured["args"] = args

    w = Worker(level=2)
    w._chip_worker = _FakeImpl()  # type: ignore[assignment]
    h = w._create_buffer_locked(16)  # 4 x f32
    try:
        shm = h.shm
        assert shm is not None
        torch.frombuffer(shm.buf, dtype=torch.float32, count=4).fill_(7.0)
        ta = TaskArgs()
        ta.add_tensor(h.tensor(shapes=(4,), dtype=_F32), TensorArgType.INPUT)
        w._run_l2_materialized(3, ta, CallConfig())

        assert captured["cid"] == 3
        args = captured["args"]
        assert isinstance(args, ChipStorageTaskArgs)
        assert args.tensor_count() == 1
        t = args.tensor(0)
        assert tuple(t.shapes[: t.ndims]) == (4,)
        assert t.data != 0  # resolved to a real local base
        # The materialized base maps the same physical pages the owner wrote through.
        mapped = torch.frombuffer((ctypes.c_float * 4).from_address(t.data), dtype=torch.float32, count=4)
        assert mapped.tolist() == [7.0, 7.0, 7.0, 7.0]
    finally:
        h.close()
        w._close_chip_import_registry()


def test_mapped_args_from_blob_delivers_tensors_and_scalars():
    # A sub callable's args expose both the mapped tensors (args[i].buffer) and the blob's scalars
    # (args.scalar_count() / args.scalar(i)) — a sub task built with add_tensor + add_scalar reaches both.
    backing = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
    va = ctypes.addressof(backing)
    ref = wrap_fork_inherited(va, 16, mint_owner_instance_id(), 1, "L3").tensor((4,), _F32)
    blob = encode_blob([ref], (17, 99))
    buf = ctypes.create_string_buffer(blob, len(blob))
    args = ImportRegistry(ImportContext(deployment=HOST_CPU)).mapped_args_from_blob(ctypes.addressof(buf), len(blob))
    assert len(args) == 1 and args.tensor_count() == 1
    assert args.scalar_count() == 2
    assert args.scalar(0) == 17 and args.scalar(1) == 99
    assert torch.frombuffer(args[0].buffer, dtype=torch.float32, count=4).tolist() == [1.0, 2.0, 3.0, 4.0]


def test_make_tensor_arg_memoizes_handle_per_storage():
    # Every ref over the same torch storage shares one handle/identity (so deps key on it); a view's
    # byte_offset places it within that backing. A plain tensor is copy-on-write across the fork, so
    # the handle is FORK_COW and read-only — only share_memory_() earns a writable grant.
    w = Worker(level=3, num_sub_workers=0)  # no init needed: make_tensor_arg only reads owner-side state
    t = torch.zeros(16, dtype=torch.float32)
    r1 = w.make_tensor_arg(t, shapes=(16,), dtype=_F32)
    r2 = w.make_tensor_arg(t[4:12], shapes=(8,), dtype=_F32)  # slice of same storage
    assert r1.buffer.backend_kind == BackendKind.FORK_COW
    assert r1.buffer.identity == r2.buffer.identity  # one memoized handle
    assert r1.byte_offset == 0
    assert r2.byte_offset == 16  # t[4:12] starts 4 float32 = 16 B into the storage
    assert len(w._fork_tensor_handles) == 1


def test_make_tensor_arg_share_memory_output_across_fork():
    # A pre-fork share_memory_() tensor is MAP_SHARED: a forked child's writes land in the same
    # physical pages the parent reads. make_tensor_arg names it (FORK_SHM), the sub child writes through
    # the inherited VA, and the parent sees the result — the fork-inherited read-write path.
    t = torch.zeros(4, dtype=torch.float32).share_memory_()  # allocated + shared BEFORE init/fork

    def sub_fn(args):
        a = torch.frombuffer(args[0].buffer, dtype=torch.float32, count=4)
        a.add_(1.0)

    hw = Worker(level=3, num_sub_workers=1)
    handle = hw.register(sub_fn)
    hw.init()  # forks the sub child, inheriting t's MAP_SHARED mapping at the same VA
    try:
        t.fill_(5.0)

        def orch(o, _args, cfg):
            sa = TaskArgs()
            sa.add_tensor(hw.make_tensor_arg(t, shapes=(4,), dtype=_F32), TensorArgType.INOUT)
            o.submit_sub(handle, sa)

        hw.run(orch, args=None, config=CallConfig())
        assert torch.allclose(t, torch.full((4,), 6.0)), t.tolist()
    finally:
        hw.close()


def test_alloc_shared_tensor_managed_intermediate_deps_and_share():
    # A runtime-managed HeapRing intermediate (alloc_shared_tensor) named as a Tensor: sub-A writes
    # it (INOUT → depends on the alloc slot + becomes producer), sub-B reads it (INPUT → depends on the
    # producer) and copies it into a create_buffer output the parent reads. The dep wires via the ref's
    # canonical identity; the ring VA is MAP_SHARED so B sees A's write; the slot auto-reclaims at scope
    # end (drain completes, no hang). Result crosses back via the shared `out` buffer, not a dict (the
    # subs run in forked processes).
    def producer(args):
        torch.frombuffer(args[0].buffer, dtype=torch.float32, count=4).fill_(9.0)

    def consumer(args):
        src = torch.frombuffer(args[0].buffer, dtype=torch.float32, count=4)
        torch.frombuffer(args[1].buffer, dtype=torch.float32, count=4).copy_(src)

    hw = Worker(level=3, num_sub_workers=2)
    ph = hw.register(producer)
    ch = hw.register(consumer)
    hw.init()
    out_t = None
    try:
        out_h = hw.create_buffer(16)  # 4 x f32 — the parent-observable output
        shm = out_h.shm
        assert shm is not None
        out_t = torch.frombuffer(shm.buf, dtype=torch.float32, count=4)
        out_t.zero_()

        def orch(o, _args, cfg):
            inter = hw.alloc_shared_tensor((4,), DataType.FLOAT32)  # worker captured in the closure
            pa = TaskArgs()
            pa.add_tensor(inter.tensor(shapes=(4,), dtype=_F32), TensorArgType.INOUT)
            o.submit_sub(ph, pa)
            ca = TaskArgs()
            ca.add_tensor(inter.tensor(shapes=(4,), dtype=_F32), TensorArgType.INPUT)
            ca.add_tensor(out_h.tensor(shapes=(4,), dtype=_F32), TensorArgType.OUTPUT_EXISTING)
            o.submit_sub(ch, ca)

        hw.run(orch, args=None, config=CallConfig())
        assert torch.allclose(out_t, torch.full((4,), 9.0)), out_t.tolist()
    finally:
        out_t = None
        hw.close()


def test_sub_worker_mapped_arg_readwrite():
    def sub_fn(args):
        a = torch.frombuffer(args[0].buffer, dtype=torch.float32, count=4)
        a.add_(1.0)  # write through the mapped shared buffer

    hw = Worker(level=3, num_sub_workers=1)
    handle = hw.register(sub_fn)
    hw.init()
    t = None
    try:
        buf_h = hw.create_buffer(16)  # 4 x float32, POSIX shm allocated post-init
        shm = buf_h.shm
        assert shm is not None
        t = torch.frombuffer(shm.buf, dtype=torch.float32, count=4)
        t.fill_(5.0)

        def orch(o, args, cfg):
            sa = TaskArgs()
            sa.add_tensor(buf_h.tensor(shapes=(4,), dtype=_F32), TensorArgType.INOUT)
            o.submit_sub(handle, sa)

        hw.run(orch, args=None, config=CallConfig())
        assert torch.allclose(t, torch.full((4,), 6.0)), t.tolist()
    finally:
        t = None
        hw.close()


def test_make_tensor_arg_remints_when_an_address_is_reused_at_a_different_size():
    """A recycled storage address is a different backing, so it must not inherit the old identity.

    The memo is keyed by address and the allocator reuses addresses. Without a size check, a later
    tensor landing where an earlier one died would borrow its handle: views sized for the new
    storage would overrun the recorded nbytes, and two unrelated buffers would key to one node in
    the dependency graph.
    """
    w = Worker(level=3, num_sub_workers=0)
    t = torch.zeros(64, dtype=torch.float32)
    base = int(t.untyped_storage().data_ptr())

    # A handle left behind by a smaller, now-dead storage that occupied this same address.
    w._fork_tensor_handles[base] = wrap_fork_inherited(
        base, 32, w._owner_instance_id, buffer_id=999, access=AccessMode.READ
    )

    ref = w.make_tensor_arg(t, shapes=(64,), dtype=_F32)
    assert ref.buffer.nbytes == 256  # the live storage, not the stale 32
    assert ref.buffer.identity.buffer_id != 999  # and a fresh identity, not the stale one
