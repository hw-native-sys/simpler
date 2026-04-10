#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3 group task — 2 ChipWorkers (process-isolated) on 1 DAG node.

Each chip runs the same kernel with its own args (different tensors).
A downstream SubTask depends on the group output.
Verifies: fork+shm process isolation, 2-chip concurrent execution,
group completion aggregation, downstream dependency waits for group.
"""

import struct
from multiprocessing.shared_memory import SharedMemory

import pytest
import torch
from task_interface import (
    ArgDirection as D,
)
from task_interface import (
    ChipStorageTaskArgs,
    WorkerPayload,
    WorkerType,
    make_tensor_arg,
)
from worker import Task, Worker

from simpler_setup import SceneTestCase, scene_test

KERNELS_BASE = "../../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels"


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class _VectorKernels(SceneTestCase):
    """Shared kernel definition — not a test itself, used for _compile()."""

    __test__ = False
    CALLABLE = {
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
    }
    RUNTIME_CONFIG = {"aicpu_thread_num": 4, "block_dim": 3}

    def generate_inputs(self, params):
        return []

    def compute_golden(self, tensors, params):
        pass


def _make_shared_tensors():
    SIZE = 128 * 128
    a = torch.full((SIZE,), 2.0, dtype=torch.float32).share_memory_()
    b = torch.full((SIZE,), 3.0, dtype=torch.float32).share_memory_()
    f = torch.zeros(SIZE, dtype=torch.float32).share_memory_()
    args = ChipStorageTaskArgs()
    for t in [a, b, f]:
        args.add_tensor(make_tensor_arg(t))
    return a, b, f, args


@pytest.mark.platforms(["a2a3sim", "a2a3"])
@pytest.mark.device_count(2)
def test_l3_group_subtask(st_platform, st_device_ids):
    """L3: Group of 2 ChipWorkers (fork+shm) as 1 DAG node, SubTask depends on group."""
    chip_callable = _VectorKernels.compile_chip_callable(st_platform)
    a0, b0, f0, args0 = _make_shared_tensors()
    a1, b1, f1, args1 = _make_shared_tensors()

    result_shm = SharedMemory(create=True, size=16)
    result_buf = result_shm.buf
    assert result_buf is not None
    struct.pack_into("dd", result_buf, 0, -999.0, -999.0)

    def sub_fn():
        import ctypes  # noqa: PLC0415

        p0 = ctypes.cast(f0.data_ptr(), ctypes.POINTER(ctypes.c_float))
        p1 = ctypes.cast(f1.data_ptr(), ctypes.POINTER(ctypes.c_float))
        struct.pack_into("dd", result_buf, 0, float(p0[0]), float(p1[0]))

    w = Worker(
        level=3, device_ids=st_device_ids, num_sub_workers=1, platform=st_platform, runtime="tensormap_and_ringbuffer"
    )
    sub_cid = w.register(sub_fn)
    w.init()

    def my_orch(w, _args):
        chip_p = WorkerPayload()
        chip_p.worker_type = WorkerType.CHIP
        chip_p.callable = chip_callable.buffer_ptr()
        chip_p.block_dim = 3
        chip_p.aicpu_thread_num = 4
        group_result = w.submit(WorkerType.CHIP, chip_p, args_list=[args0.__ptr__(), args1.__ptr__()], outputs=[4])
        sub_p = WorkerPayload()
        sub_p.worker_type = WorkerType.SUB
        sub_p.callable_id = sub_cid
        w.submit(WorkerType.SUB, sub_p, inputs=[group_result.outputs[0].ptr])

    w.run(Task(orch=my_orch))
    w.close()

    v0, v1 = struct.unpack_from("dd", result_buf, 0)
    result_shm.close()
    result_shm.unlink()

    assert abs(f0[0].item() - 47.0) < 0.01, f"Chip 0 wrong: {f0[0].item()}"
    assert abs(f1[0].item() - 47.0) < 0.01, f"Chip 1 wrong: {f1[0].item()}"
    assert v0 != -999.0 and v1 != -999.0, "SubTask never ran"
    assert abs(v0 - 47.0) < 0.01, f"SubTask saw wrong f0: {v0}"
    assert abs(v1 - 47.0) < 0.01, f"SubTask saw wrong f1: {v1}"
