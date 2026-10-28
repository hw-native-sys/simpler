# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Direct L2 Host DDR gather followed by a dependent Device consumer."""

import argparse
import sys
from pathlib import Path

import numpy as np
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.scene_test import compile_chip_callable_spec


def build_callable(runtime="tensormap_and_ringbuffer"):
    root = Path(__file__).resolve().parent / "kernels"
    spec = {
        "orchestration": {
            "source": str(root / "orchestration/gather.cpp"),
            "function_name": "host_ddr_gather",
            "signature": [D.IN, D.IN, D.OUT, D.OUT, D.OUT, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "source": str(root / "aiv/gather.cpp"),
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT, D.OUT, D.IN],
            },
            {"func_id": 1, "source": str(root / "aiv/consumer.cpp"), "core_type": "aiv", "signature": [D.IN, D.OUT]},
        ],
    }
    for kernel in spec["incores"]:
        kernel["extra_include_dirs"] = ["$ASCEND_HOME_PATH/asc"]
    return compile_chip_callable_spec(spec, "a5", runtime, ("host_ddr_gather", "a5", runtime))


def run(device=0, runtime="tensormap_and_ringbuffer"):
    compiled = build_callable(runtime)
    worker = Worker(level=2, platform="a5", runtime=runtime, device_id=device)
    tables = []
    try:
        worker.init()
        callable_handle = worker.register(compiled)
        ids = np.array([0, 4095, 7, 7, -1, 4096], dtype=np.int32)
        device_ids = worker.malloc(ids.nbytes)
        gathered = worker.malloc(6 * 256 * 2)
        result = worker.malloc(6 * 256 * 4)
        status = worker.malloc(4)
        worker.copy_to(device_ids, ids)
        tables.append(worker.create_buffer(4097 * 256 * 2, device_readable=True))
        for round_id, (table_index, count) in enumerate(((0, 6), (1, 6), (0, 0), (0, 7), (0, 6))):
            if round_id == 1:
                # A larger table created after the first run reuses all Device payload buffers.
                tables.append(worker.create_buffer(16385 * 256 * 2, device_readable=True))
            table = tables[table_index]
            rows = table.nbytes // 512
            host_view = np.ndarray((rows, 256), dtype=np.uint16, buffer=table.shm.buf)
            host_view[:] = 0x3F00 + (np.arange(rows)[:, None] + np.arange(256)[None, :] + round_id * 17) % 128
            expected = np.zeros((6, 256), dtype=np.uint16)
            if count == 6:
                for i, row in enumerate(ids):
                    if 0 <= row < 4096:
                        expected[i] = host_view[row + 1]
            del host_view
            worker.copy_to(gathered, np.zeros((6, 256), dtype=np.uint16))
            args = TaskArgs()
            args.add_tensor(table.tensor((4096, 256), DataType.BFLOAT16, byte_offset=512))
            args.add_tensor(device_ids.tensor((6,), DataType.INT32))
            args.add_tensor(gathered.tensor((6, 256), DataType.BFLOAT16), TensorArgType.OUTPUT_EXISTING)
            args.add_tensor(result.tensor((6, 256), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
            args.add_tensor(status.tensor((1,), DataType.INT32), TensorArgType.OUTPUT_EXISTING)
            args.add_scalar(count)
            pending = worker.submit(callable_handle, args, CallConfig())
            for release in (table.close, table.shm.close, table.shm.unlink, lambda: worker.release_buffer(table)):
                try:
                    release()
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("accepted run must retain its Host mapping until wait")
            pending.wait()
            got = np.empty_like(expected)
            consumed = np.empty((6, 256), dtype=np.float32)
            code = np.empty((1,), dtype=np.int32)
            worker.copy_from(got, gathered)
            worker.copy_from(consumed, result)
            worker.copy_from(code, status)
            np.testing.assert_array_equal(got, expected)
            golden = (expected.astype(np.uint32) << 16).view(np.float32) + 1.0
            np.testing.assert_array_equal(consumed, golden)
            assert code[0] == (1 if count == 7 else 2 if count == 6 else 0)
            print(f"PASS runtime={runtime} round={round_id} table={table_index} requests={count} status={code[0]}")
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
    parser.add_argument(
        "--runtime", choices=("tensormap_and_ringbuffer", "host_build_graph"), default="tensormap_and_ringbuffer"
    )
    arguments = parser.parse_args()
    run(arguments.device, arguments.runtime)
