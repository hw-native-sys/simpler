# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Batched eager execution and recoverable host admission failures."""

import ctypes

from simpler.task_interface import ChipStorageTaskArgs, ChipTensor, DataType

from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.kernel_capture_values import _COUNT


def _rejections(io, reject, recover, source, destination, cid):
    def packet(tensors=2, scalar=True, device=True):
        args = ChipStorageTaskArgs()
        for address in (source, destination)[:tensors]:
            args.add_tensor(ChipTensor.make(address.value, (_COUNT,), DataType.FLOAT32, child_memory=device))
        if scalar:
            args.add_scalar(ctypes.c_float(1.25))
        return args

    cases = (
        (-1, packet(), False, -1004),
        (8191, packet(), False, -1007),
        (cid, packet(tensors=1), False, -1000),
        (cid, packet(scalar=False), False, -1000),
        (cid, packet(device=False), False, -1000),
        (cid, packet(), True, -1004),
    )
    for bad_cid, args, null_stream, expected in cases:
        io.write(destination, [-999.0] * _COUNT)
        reject(bad_cid, args, null_stream, expected)
        io.verify(destination, [-999.0] * _COUNT)
        recover()
    return len(cases)


def run_eager_case(scenario, io, launch, sync, pairs, initial, reject, cid):
    x, y = pairs[0]
    if scenario == "eager_rejections":

        def recover():
            launch(0)
            sync()
            io.verify(y, [value + 1.25 for value in initial])

        recovered = _rejections(io, reject, recover, x, y, cid)
        for iteration in range(100 - recovered):
            launch(0, scalar=float(iteration))
            sync()
            io.verify(y, [value + iteration for value in initial])
        return

    if scenario == "eager_dag":
        for iteration in range(100):
            values = [value + iteration for value in initial]
            io.write(x, values)
            launch(0, (x, y), scalar=float(iteration % 7 + 1))
            sync()
            io.verify(y, [2 * value + 3 * (iteration % 7 + 1) for value in values])
        return
    # Alternating snapshots form a feedback chain. There is only one final sync.
    expected = 0.0
    sign = 1
    for iteration in range(100):
        index = (0, 1, 0)[iteration % 3] if scenario == "eager_multi_callable" else 0
        scalar = float(iteration % 7 + 1)
        launch(index, (x, y), scalar=scalar)
        if index == 1:
            sign = -sign
            expected = -expected - scalar
        else:
            expected += scalar
        x, y = y, x
    sync()
    io.verify(x, [sign * value + expected for value in initial])
