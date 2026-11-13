# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared L3 Host MXFP8 rows gathered by local MTE into a Device MX consumer."""

import argparse
from functools import partial

import numpy as np
from run import ROOT, RUNTIME, close_preserving_error, device_array, diagnostics, make_worker, read_device
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import DataType, TaskArgs, TensorArgType, TensorTransfer

from simpler_setup.scene_test import compile_chip_callable_spec


def build_callable():
    project = ROOT.parents[3]
    spec = {
        "orchestration": {
            "source": str(ROOT.parent / "host_ddr_gather/kernels/orchestration/mx_consumer.cpp"),
            "function_name": "host_ddr_mx",
            "signature": [D.IN, D.IN, D.OUT, D.OUT, D.IN, D.IN, D.IN, D.OUT, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "source": str(ROOT.parent / "host_ddr_gather/kernels/aiv/gather.cpp"),
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT, D.OUT, D.IN],
                "extra_include_dirs": ["$ASCEND_HOME_PATH/asc"],
            },
            {
                "func_id": 1,
                "source": str(
                    project / "tests/st/a5/tensormap_and_ringbuffer/mx_fp_gemm/kernels/aic/kernel_fp_gemm.cpp"
                ),
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.IN, D.IN, D.OUT, D.IN],
            },
        ],
    }
    return compile_chip_callable_spec(spec, "a5", RUNTIME, ("host_ddr_l3_mx", "a5", RUNTIME))


def run(devices=(0, 1), *, staged=False, output_prefix=""):
    compiled = build_callable()
    worker = make_worker(devices, staged)
    entry = worker.register(compiled)
    try:
        worker.init()
        table = worker.create_buffer(4097 * 64)
        native = table.tensor((4096, 64), DataType.FP8E4M3FN, byte_offset=64)
        outer = TaskArgs()
        outer.add_tensor(native, transfer=TensorTransfer.MAP)
        indices = np.array([0, 4095, 7, 7] * 32, np.int32)
        b = np.zeros((64, 64), np.uint8)
        np.fill_diagonal(b, 0x38)
        rank_buffers = [[] for _ in range(2 if staged else 1)]
        output_indices = (1, 2, 6)
        for rank in range(len(devices)):
            data = [
                (indices, DataType.INT32),
                (np.zeros((128, 64), np.uint8), DataType.FP8E4M3FN),
                (np.zeros(1, np.int32), DataType.INT32),
                (np.full(256, 128, np.uint8), DataType.FP8E8M0),
                (b, DataType.FP8E4M3FN),
                (np.full(128, 127, np.uint8), DataType.FP8E8M0),
                (np.zeros((128, 64), np.float32), DataType.FLOAT32),
            ]
            first = [(device_array(worker, rank, array, dtype), array.shape, dtype) for array, dtype in data]
            rank_buffers[0].append(first)
            if staged:
                following = first.copy()
                for index in output_indices:
                    array, dtype = data[index]
                    following[index] = (device_array(worker, rank, array, dtype), array.shape, dtype)
                rank_buffers[1].append(following)
        for round_id in range(2):
            with table.host_view() as memory:
                host = np.ndarray((4097, 64), np.uint8, buffer=memory)
                host[:] = np.where((np.arange(4097)[:, None] + np.arange(64)[None, :] + round_id) % 3 == 0, 0xB8, 0x38)
                expected_bytes = host[indices + 1].copy()
                expected = np.where(expected_bytes == 0x38, 1.0, -1.0).astype(np.float32) * 2
                del host

            def orchestrate(orch, _args, config, *, ranks):
                args_by_rank = []
                for buffers in ranks:
                    args = TaskArgs()
                    args.add_tensor(native, transfer=TensorTransfer.MAP)
                    for index, (buffer, shape, dtype) in enumerate(buffers):
                        args.add_tensor(
                            buffer.tensor(shape, dtype),
                            TensorArgType.OUTPUT_EXISTING if index in output_indices else TensorArgType.INPUT,
                        )
                    args.add_scalar(128)
                    args_by_rank.append(args)
                orch.submit_next_level_group(entry, args_by_rank, config, workers=list(range(len(devices))))

            pending = [
                worker.submit(
                    partial(orchestrate, ranks=ranks),
                    outer,
                    diagnostics(output_prefix, round_id if submission == 0 else round_id * 2 + 100),
                )
                for submission, ranks in enumerate(rank_buffers)
            ]
            for handle in pending:
                handle.wait()
            for submission, ranks in enumerate(rank_buffers):
                for rank, buffers in enumerate(ranks):
                    gathered = read_device(worker, buffers[1][0], (128, 64), np.uint8)
                    status = read_device(worker, buffers[2][0], (1,), np.int32)
                    result = read_device(worker, buffers[6][0], (128, 64), np.float32)
                    exact_rows = np.all(gathered == expected_bytes, axis=1)
                    if not exact_rows.all() or np.any(status):
                        print(
                            f"MX diagnostic rank={rank} round={round_id} submission={submission} "
                            f"status={status.tolist()} "
                            f"exact_rows_per_chunk={exact_rows.reshape(4, 32).sum(axis=1).tolist()} "
                            f"bad_rows={np.flatnonzero(~exact_rows).tolist()} "
                            f"zero_rows={np.flatnonzero(np.all(gathered == 0, axis=1)).tolist()}",
                            flush=True,
                        )
                    np.testing.assert_array_equal(
                        gathered, expected_bytes, err_msg=f"rank={rank} submission={submission}"
                    )
                    np.testing.assert_array_equal(status, 0, err_msg=f"rank={rank} submission={submission}")
                    np.testing.assert_array_equal(result, expected, err_msg=f"rank={rank} submission={submission}")
                    print(
                        f"PASS rank={rank} round={round_id} submission={submission}: "
                        "Host MXFP8 via local MAP, Device E8M0 scales, TMATMUL_MX",
                        flush=True,
                    )
        worker.release_buffer(table)
    finally:
        close_preserving_error(worker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--prepare", action="store_true")
    options = parser.parse_args()
    if options.prepare:
        build_callable()
    else:
        run(
            tuple(int(item) for item in options.devices.split(",")),
            staged=options.staged,
            output_prefix=options.output_prefix,
        )
