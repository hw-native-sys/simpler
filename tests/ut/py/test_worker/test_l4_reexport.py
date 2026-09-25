# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L4 -> L3 -> sub re-export end-to-end (P1-B B3).

An L4-owned create_buffer backing reaches L3 as a Tensor; L3's orch forwards it re-exported to a
handle H' that keeps L4's canonical identity (no Tensor pass-through, no map on the forwarding hop),
to its sub; the sub (a compute leaf) maps H' and writes through it. The write lands in the shared
backing L4 owns.
No NPU device — L3 uses a SubWorker.
"""

import pytest
import torch
from simpler.buffer import (
    AccessMode,
    AddressSpace,
    BackendKind,
    BufferDescriptor,
    CanonicalIdentity,
    TensorTransfer,
)
from simpler.task_interface import CallConfig, TaskArgs, TensorArgType
from simpler.worker import Worker

_F32 = 0  # DataType.FLOAT32 value


def test_l4_l3_reexport_to_sub():
    def l3_sub(args):
        a = torch.frombuffer(args[0].buffer, dtype=torch.float32, count=4)
        a.add_(1.0)  # compute leaf writes through the mapped, re-exported backing

    l3 = Worker(level=3, num_sub_workers=1)
    l3_sub_handle = l3.register(l3_sub)

    def l3_orch(orch, args, config):
        # A TaskArgs whose tensors are re-exported onto L3 (each level sees only its own handles).
        sa = TaskArgs()
        sa.add_tensor(args.tensor(0), TensorArgType.INOUT)
        orch.submit_sub(l3_sub_handle, sa)

    w4 = Worker(level=4, num_sub_workers=1)
    w4.register(lambda args: None)  # a sub child so create_buffer has a forked child to satisfy
    l3_orch_handle = w4.register(l3_orch)
    w4.add_worker(l3)
    w4.init()

    t = None
    try:
        buf_h = w4.create_buffer(16)  # L4-owned POSIX shm, allocated post-init
        shm = buf_h.shm
        assert shm is not None
        t = torch.frombuffer(shm.buf, dtype=torch.float32, count=4)
        t.fill_(5.0)

        def l4_orch(orch, args, config):
            ta = TaskArgs()
            ta.add_tensor(buf_h.tensor(shapes=(4,), dtype=_F32), TensorArgType.INOUT)
            orch.submit_next_level(l3_orch_handle, ta, CallConfig(), worker=0)

        w4.run(l4_orch)
        assert torch.allclose(t, torch.full((4,), 6.0)), t.tolist()
    finally:
        t = None
        w4.close()


@pytest.mark.parametrize("transfer", [TensorTransfer.NONE, TensorTransfer.H2D])
def test_l4_l3_reexport_carries_scalars(transfer):
    # Re-export replaces each tensor's backing handle and nothing else: the container a nested
    # next-level child's orch fn receives is the TaskArgs the submitter built, scalars included.
    def l3_sub(args):
        a = torch.frombuffer(args[0].buffer, dtype=torch.float32, count=4)
        a[0] = float(args.scalar(0))
        a[1] = float(args.scalar(1))

    l3 = Worker(level=3, num_sub_workers=1)
    l3_sub_handle = l3.register(l3_sub)

    def l3_orch(orch, args, config):
        if args.scalar_count() != 2:
            raise AssertionError(f"nested child lost scalars: scalar_count={args.scalar_count()}")
        sa = TaskArgs()
        assert args.transfer(0) == transfer
        sa.add_tensor(args.tensor(0), TensorArgType.INOUT, transfer=args.transfer(0))
        for i in range(args.scalar_count()):
            sa.add_scalar(args.scalar(i))
        orch.submit_sub(l3_sub_handle, sa)

    w4 = Worker(level=4, num_sub_workers=1)
    w4.register(lambda args: None)
    l3_orch_handle = w4.register(l3_orch)
    w4.add_worker(l3)
    w4.init()

    t = None
    try:
        buf_h = w4.create_buffer(16)
        shm = buf_h.shm
        assert shm is not None
        t = torch.frombuffer(shm.buf, dtype=torch.float32, count=4)
        t.fill_(0.0)

        def l4_orch(orch, args, config):
            ta = TaskArgs()
            ta.add_tensor(buf_h.tensor(shapes=(4,), dtype=_F32), TensorArgType.INOUT, transfer=transfer)
            ta.add_scalar(7)
            ta.add_scalar(11)
            orch.submit_next_level(l3_orch_handle, ta, CallConfig(), worker=0)

        w4.run(l4_orch)
        assert t.tolist() == [7.0, 11.0, 0.0, 0.0], t.tolist()
    finally:
        t = None
        w4.close()


def test_reexport_preserves_canonical_identity():
    # Frozen model §5/§8: the canonical identity is invariant across every edge. Re-exporting a
    # forwarded upper-level backing keeps the SOURCE (owner_instance_id, owner_worker_path, buffer_id,
    # generation) — only the mapping is stripped (base=0, shm=None) on the forwarding hop. Dependency
    # inference keys on the identity, so an alias / retain-release must not split across L4→L3→L2.
    src_identity = CanonicalIdentity(b"\x9f" * 8, 42, 1)
    source = BufferDescriptor(
        identity=src_identity,
        address_space=AddressSpace.HOST,
        access=AccessMode.READWRITE,
        backend_kind=BackendKind.FORK_SHM,
        nbytes=1024,
        body=(0xDEAD0000).to_bytes(8, "little"),
    )
    w = Worker(level=3, num_sub_workers=0)
    h_prime = w._reexport(source)
    assert h_prime.identity == src_identity  # identity invariant across the edge
    assert h_prime.base == 0 and h_prime.shm is None  # no map on the forwarding hop
    assert h_prime.backend_kind == source.backend_kind and h_prime.body == source.body
    # A ref built over H' carries the same identity a downstream consumer keys deps on.
    assert h_prime.tensor((4,), _F32).buffer.identity == src_identity
