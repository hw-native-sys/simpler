# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Cross-process consumers of allocation-bounded shared HeapRing slabs."""

import ctypes
import struct

from simpler.task_interface import DataType, TaskArgs, TensorArgType
from simpler.worker import Worker


def test_sub_consumes_nonzero_heap_slices_without_touching_peer_guards():
    def consume(args):
        first, second, output = (args[index].buffer for index in range(3))
        assert len(first) == len(second) == 16
        assert bytes(first) == b"a" * 16
        assert bytes(second) == b"b" * 16
        struct.pack_into("4i", second, 0, 42, 43, 44, 45)
        output[:16] = first
        output[16:32] = second

    worker = Worker(level=3, num_sub_workers=1, heap_ring_size=4096)
    handle = worker.register(consume)
    worker.init()
    try:
        output = worker.create_buffer(32)
        assert output.shm is not None and output.shm.buf is not None

        def build(orch, _args, _config):
            orch.alloc((4,), DataType.INT32)
            first = orch.alloc((4,), DataType.INT32)
            second = orch.alloc((4,), DataType.INT32)
            assert int.from_bytes(first.body[8:16], "little") > 0
            assert int.from_bytes(second.body[8:16], "little") > int.from_bytes(first.body[8:16], "little")
            ctypes.memmove(first.base, b"a" * 16, 16)
            ctypes.memmove(second.base, b"b" * 16, 16)
            args = TaskArgs()
            args.add_tensor(first.tensor((4,), DataType.INT32), TensorArgType.INPUT)
            args.add_tensor(second.tensor((4,), DataType.INT32), TensorArgType.INOUT)
            args.add_tensor(output.tensor((8,), DataType.INT32), TensorArgType.OUTPUT_EXISTING)
            orch.submit_sub(handle, args)

        for _ in range(3):
            worker.run(build)
            assert bytes(output.shm.buf[:16]) == b"a" * 16
            assert struct.unpack_from("4i", output.shm.buf, 16) == (42, 43, 44, 45)
    finally:
        worker.close()


def test_ancestor_heap_slice_survives_recursive_forwarding():
    def consume(args):
        assert len(args[0].buffer) == 16
        assert struct.unpack_from("4i", args[0].buffer) == (7, 8, 9, 10)
        args[1].buffer[:16] = args[0].buffer

    child = Worker(level=3, num_sub_workers=1, heap_ring_size=4096)
    consume_handle = child.register(consume)

    def forward(orch, args, config):
        orch.submit_sub(consume_handle, args)

    parent = Worker(level=4, num_sub_workers=0, heap_ring_size=4096)
    forward_handle = parent.register(forward)
    parent.add_worker(child)
    parent.init()
    try:
        output = parent.create_buffer(16)
        assert output.shm is not None and output.shm.buf is not None

        def build(orch, _args, config):
            orch.alloc((4,), DataType.INT32)
            slab = orch.alloc((4,), DataType.INT32)
            ctypes.memmove(slab.base, struct.pack("4i", 7, 8, 9, 10), 16)
            args = TaskArgs()
            args.add_tensor(slab.tensor((4,), DataType.INT32), TensorArgType.INPUT)
            args.add_tensor(output.tensor((4,), DataType.INT32), TensorArgType.OUTPUT_EXISTING)
            orch.submit_next_level(forward_handle, args, config, worker=0)

        parent.run(build)
        assert struct.unpack_from("4i", output.shm.buf) == (7, 8, 9, 10)
    finally:
        parent.close()
