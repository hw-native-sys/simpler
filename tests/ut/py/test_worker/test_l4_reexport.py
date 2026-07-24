# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L4 -> L3 -> sub re-export end-to-end (P1-B B3).

An L4-owned create_buffer backing reaches L3 as a BufferRef; L3's orch sees it re-exported to an
L3-owned handle H' (no BufferRef pass-through, no map on the forwarding hop), forwards it to its sub;
the sub (a compute leaf) maps H' and writes through it. The write lands in the shared backing L4 owns.
No NPU device — L3 uses a SubWorker.
"""

import torch
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
        # args are re-exported BufferRefs owned by L3 (each level sees only its own handles).
        sa = TaskArgs()
        sa.add_ref(args[0], TensorArgType.INOUT)
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
            ta.add_ref(buf_h.ref(shapes=(4,), dtype=_F32), TensorArgType.INOUT)
            orch.submit_next_level(l3_orch_handle, ta, CallConfig(), worker=0)

        w4.run(l4_orch)
        assert torch.allclose(t, torch.full((4,), 6.0)), t.tolist()
    finally:
        t = None
        w4.close()
