#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3->L2 chip dispatch over the BufferRef wire (P1-B B3, minimal end-to-end).

The L3 orch is a pure DAG builder over views: it names task args as BufferRefs built from handles
(create_buffer + handle.ref), never touching data. torch is used only OUTSIDE run() to fill inputs
and read the output. The owner writes a BufferRef blob to the chip mailbox; the chip child
materializes it back to Tensors (ImportRegistry -> materialize_bufferref_blob) and runs the kernel;
the result lands in the shared output buffer with no per-run copy.
"""

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, TaskArgs, TensorArgType

from simpler_setup import SceneTestCase, scene_test

KERNELS_BASE = "../../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels"

SIZE = 128 * 128
DTYPE = torch.float32
_F32 = 0  # DataType.FLOAT32 value


def _golden(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    s = a + b
    return (s + 1) * (s + 2) + s


def _one_task_orch(chip_handle, a_ref, b_ref, out_ref):
    def orch_fn(orch, _args, cfg):
        ta = TaskArgs()
        ta.add_ref(a_ref, TensorArgType.INPUT)
        ta.add_ref(b_ref, TensorArgType.INPUT)
        ta.add_ref(out_ref, TensorArgType.OUTPUT_EXISTING)
        orch.submit_next_level(chip_handle, ta, cfg, worker=0)

    return orch_fn


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestL3BufferRefDispatch(SceneTestCase):
    """A single chip task dispatched over the BufferRef wire on one L3 worker."""

    CALLABLE = {
        "callables": [
            {
                "name": "vector",
                "orchestration": {
                    "source": f"{KERNELS_BASE}/orchestration/example_orchestration.cpp",
                    "function_name": "aicpu_orchestration_entry",
                    "signature": [D.IN, D.IN, D.OUT],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": f"{KERNELS_BASE}/aiv/kernel_add.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.IN, D.OUT],
                    },
                    {
                        "func_id": 1,
                        "source": f"{KERNELS_BASE}/aiv/kernel_add_scalar.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT],
                    },
                    {
                        "func_id": 2,
                        "source": f"{KERNELS_BASE}/aiv/kernel_mul.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.IN, D.OUT],
                    },
                ],
            },
        ],
    }

    CASES = [
        {"name": "bufferref_dispatch", "platforms": ["a2a3sim"]},
    ]

    def test_run(self, st_worker):
        worker = st_worker
        chip_handle = type(self)._st_chip_handles["vector"]

        nbytes = SIZE * DTYPE.itemsize
        a_h = worker.create_buffer(nbytes)
        b_h = worker.create_buffer(nbytes)
        out_h = worker.create_buffer(nbytes)
        a = b = out = None
        result = False
        try:
            # torch is used only outside run(): fill inputs before, read output after.
            a = torch.frombuffer(a_h.shm.buf, dtype=DTYPE, count=SIZE)
            b = torch.frombuffer(b_h.shm.buf, dtype=DTYPE, count=SIZE)
            out = torch.frombuffer(out_h.shm.buf, dtype=DTYPE, count=SIZE)
            a.fill_(5.0)
            b.fill_(7.0)
            out.zero_()
            # The orch names args purely as BufferRef views built from the handles.
            a_ref = a_h.ref(shapes=(SIZE,), dtype=_F32)
            b_ref = b_h.ref(shapes=(SIZE,), dtype=_F32)
            out_ref = out_h.ref(shapes=(SIZE,), dtype=_F32)
            worker.run(_one_task_orch(chip_handle, a_ref, b_ref, out_ref), args=None, config=CallConfig())
            result = torch.allclose(out, _golden(a, b), rtol=self.RTOL, atol=self.ATOL)
        finally:
            # Drop views before the framework-driven close unlinks the shm.
            a = b = out = None
        assert result, "BufferRef-dispatched chip task did not produce the golden result"
