# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Sub-worker task args over the BufferRef wire (P1-B B3).

A Python sub callable is a compute leaf: it receives its args as MappedArgs (each backing mapped into
the sub child, map-once) and computes with torch.frombuffer(arg.buffer, ...). Writes land in the
shared backing the owner sees — no C++ Tensor involved. This is exactly the case a closure cannot
serve: a post-init create_buffer shm is not mapped in the pre-forked sub child, so the buffer must
arrive via args and be mapped from its Ref.
"""

import torch
from simpler.task_interface import CallConfig, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

_F32 = 0  # DataType.FLOAT32 value


def test_alloc_shared_tensor_sizes_by_shape():
    hw = Worker(level=3, num_sub_workers=1)
    hw.init()
    try:
        h = hw.alloc_shared_tensor((4, 8), DataType.FLOAT32)
        assert h.nbytes == 4 * 8 * 4  # prod(shape) * element_size
        assert h.shm is not None  # a shared, born-attached backing (kind3)
    finally:
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
            sa.add_ref(buf_h.ref(shapes=(4,), dtype=_F32), TensorArgType.INOUT)
            o.submit_sub(handle, sa)

        hw.run(orch, args=None, config=CallConfig())
        assert torch.allclose(t, torch.full((4,), 6.0)), t.tolist()
    finally:
        t = None
        hw.close()
