# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host MXFP8 payload → Device GM → existing TMATMUL_MX with Device scales."""

import argparse
import sys
from pathlib import Path

import numpy as np
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.scene_test import compile_chip_callable_spec


def run(device=0):
    root = Path(__file__).resolve().parent
    project = root.parents[3]
    spec = {
        "orchestration": {
            "source": str(root / "kernels/orchestration/mx_consumer.cpp"),
            "function_name": "host_ddr_mx",
            "signature": [D.IN, D.IN, D.OUT, D.OUT, D.IN, D.IN, D.IN, D.OUT, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "source": str(root / "kernels/aiv/gather.cpp"),
                "core_type": "aiv",
                "extra_include_dirs": ["$ASCEND_HOME_PATH/asc"],
                "signature": [D.IN, D.IN, D.OUT, D.OUT, D.IN],
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
    compiled = compile_chip_callable_spec(spec, "a5", "tensormap_and_ringbuffer", ("host_ddr_mx",))
    worker = Worker(level=2, platform="a5", runtime="tensormap_and_ringbuffer", device_id=device)
    handles = []
    table = None
    try:
        worker.init()
        entry = worker.register(compiled)
        table = worker.create_buffer(4096 * 64, device_readable=True)
        assert table.shm is not None and table.shm.buf is not None
        host = np.ndarray((4096, 64), np.uint8, buffer=table.shm.buf)
        # Exact E4M3FN +/-1, varying by both row and column.
        host[:] = np.where((np.arange(4096)[:, None] + np.arange(64)[None, :]) % 3 == 0, 0xB8, 0x38)
        ids = np.array(([0, 4095, 7, 7] * 32), dtype=np.int32)
        expected = np.where(host[ids] == 0x38, 1.0, -1.0).astype(np.float32)
        del host
        b = np.zeros((64, 64), np.uint8)
        np.fill_diagonal(b, 0x38)
        data = [
            ids,
            np.zeros((128, 64), np.uint8),
            np.zeros(1, np.int32),
            np.full(256, 128, np.uint8),
            b,
            np.full(128, 127, np.uint8),
            np.zeros((128, 64), np.float32),
        ]
        dtypes = [
            DataType.INT32,
            DataType.FP8E4M3FN,
            DataType.INT32,
            DataType.FP8E8M0,
            DataType.FP8E4M3FN,
            DataType.FP8E8M0,
            DataType.FLOAT32,
        ]
        args = TaskArgs()
        args.add_tensor(table.tensor((4096, 64), DataType.FP8E4M3FN))
        assert len(data) == len(dtypes)
        for i, value in enumerate(data):
            dtype = dtypes[i]
            handle = worker.malloc(value.nbytes)
            handles.append(handle)
            worker.copy_to(handle, value)
            args.add_tensor(
                handle.tensor(value.shape, dtype),
                TensorArgType.OUTPUT_EXISTING if i in (1, 2, 6) else TensorArgType.INPUT,
            )
        args.add_scalar(128)
        worker.run(entry, args)
        got = np.empty_like(expected)
        worker.copy_from(got, handles[-1])
        # A's Device E8M0 scale is 2, B's Device scale is 1.
        np.testing.assert_array_equal(got, expected * 2)
        status = np.empty(1, np.int32)
        worker.copy_from(status, handles[2])
        assert status[0] == 0
        print("PASS Host MXFP8 payload, Device E8M0 scales, existing TMATMUL_MX consumer")
    finally:
        original_error = sys.exc_info()[1]
        try:
            worker.close()
        except BaseException as cleanup_error:
            if original_error is not None:
                raise original_error from cleanup_error
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    run(parser.parse_args().device)
