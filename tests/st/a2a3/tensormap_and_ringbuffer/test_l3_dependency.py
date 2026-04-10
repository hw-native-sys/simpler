#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3 ChipTask → SubTask dependency via TensorMap.

Worker(level=3) submits a ChipTask then a SubTask that depends on it.
Verifies: TensorMap dependency inference, cross-fork data visibility,
SubWorker reads result produced by ChipWorker.
"""

import struct
from multiprocessing.shared_memory import SharedMemory

import pytest
import torch
from simpler.task_interface import (
    ArgDirection as D,
)
from simpler.task_interface import (
    ChipStorageTaskArgs,
    WorkerPayload,
    WorkerType,
    make_tensor_arg,
)
from simpler.worker import Task, Worker

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
@pytest.mark.device_count(1)
def test_l3_chiptask_subtask(st_platform, st_device_ids):
    """L3: ChipTask produces output, SubTask depends on it via TensorMap."""
    chip_callable = _VectorKernels.compile_chip_callable(st_platform)
    a, b, f, orch_args = _make_shared_tensors()
    SIZE = f.numel()

    result_shm = SharedMemory(create=True, size=8)
    result_buf = result_shm.buf
    assert result_buf is not None
    struct.pack_into("d", result_buf, 0, -999.0)

    def sub_fn():
        import ctypes  # noqa: PLC0415

        ptr = ctypes.cast(f.data_ptr(), ctypes.POINTER(ctypes.c_float))
        struct.pack_into("d", result_buf, 0, float(ptr[0]))

    chip_callable_ptr = chip_callable.buffer_ptr()
    orch_args_ptr = orch_args.__ptr__()

    w = Worker(
        level=3, device_ids=st_device_ids, num_sub_workers=1, platform=st_platform, runtime="tensormap_and_ringbuffer"
    )
    sub_cid = w.register(sub_fn)
    w.init()

    def my_orch(w, _args):
        chip_p = WorkerPayload()
        chip_p.worker_type = WorkerType.CHIP
        chip_p.callable = chip_callable_ptr
        chip_p.args = orch_args_ptr
        chip_p.block_dim = 3
        chip_p.aicpu_thread_num = 4
        chip_result = w.submit(WorkerType.CHIP, chip_p, inputs=[], outputs=[SIZE * 4])
        sub_p = WorkerPayload()
        sub_p.worker_type = WorkerType.SUB
        sub_p.callable_id = sub_cid
        w.submit(WorkerType.SUB, sub_p, inputs=[chip_result.outputs[0].ptr])

    w.run(Task(orch=my_orch))
    w.close()

    result_val = struct.unpack_from("d", result_buf, 0)[0]
    result_shm.close()
    result_shm.unlink()

    assert abs(f[0].item() - 47.0) < 0.01, f"ChipTask wrong: {f[0].item()}"
    assert result_val != -999.0, "SubTask never ran"
    assert abs(result_val - 47.0) < 0.01, f"SubTask saw wrong value: {result_val}"
