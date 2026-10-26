# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Direct L2 OUT and INOUT accesses to two disjoint views of one Host Buffer."""

import argparse
from pathlib import Path

import numpy as np
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.scene_test import compile_chip_callable_spec


def run(device=0, runtime="tensormap_and_ringbuffer"):
    root = Path(__file__).resolve().parent / "kernels"
    spec = {
        "orchestration": {
            "source": str(root / "orchestration/write.cpp"),
            "function_name": "host_ddr_write",
            "signature": [D.IN, D.OUT, D.INOUT],
        },
        "incores": [
            {"func_id": 0, "source": str(root / "aiv/consumer.cpp"), "core_type": "aiv", "signature": [D.IN, D.OUT]},
            {"func_id": 1, "source": str(root / "aiv/update.cpp"), "core_type": "aiv", "signature": [D.INOUT]},
        ],
    }
    for kernel in spec["incores"]:
        kernel["extra_include_dirs"] = ["$ASCEND_HOME_PATH/asc"]
    compiled = compile_chip_callable_spec(spec, "a5", runtime, ("host_ddr_write", "a5", runtime))
    worker = Worker(level=2, platform="a5", runtime=runtime, device_id=device)
    try:
        worker.init()
        handle = worker.register(compiled)
        source = worker.malloc(512)
        worker.copy_to(source, np.full((1, 256), 0x3F80, dtype=np.uint16))  # BF16 1.0
        buffer = worker.create_buffer(4096)
        output = worker.map_view(buffer.tensor((1, 256), DataType.FLOAT32, byte_offset=512))
        update = worker.map_view(buffer.tensor((256,), DataType.FLOAT32, byte_offset=2048))
        args = TaskArgs()
        args.add_tensor(source.tensor((1, 256), DataType.BFLOAT16))
        args.add_tensor(output, TensorArgType.OUTPUT_EXISTING)
        args.add_tensor(update, TensorArgType.INOUT)
        for iteration in range(2):
            expected = np.full(1024, -99.0, dtype=np.float32)
            expected[512:768] = np.arange(256, dtype=np.float32) + iteration
            with buffer.host_view() as host:
                host[:] = expected.tobytes()
            expected[128:384] = 2.0
            expected[512:768] += 1.0
            pending = worker.submit(handle, args, CallConfig())
            for view in (output, update):
                try:
                    worker.unmap_view(view)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("accepted writer must retain its mapping until wait")
            pending.wait()
            # Read Host DDR directly: there is no Device-to-Host staging copy.
            with buffer.host_view() as host:
                actual = np.frombuffer(host, dtype=np.float32).copy()
            np.testing.assert_array_equal(actual, expected)
        worker.unmap_view(output)
        worker.unmap_view(update)
        print(f"PASS runtime={runtime}: mapped OUT/INOUT, offset guards and CPU visibility")
    finally:
        worker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--runtime", choices=("tensormap_and_ringbuffer", "host_build_graph"), default="tensormap_and_ringbuffer"
    )
    arguments = parser.parse_args()
    run(arguments.device, arguments.runtime)
