# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One producer posts through independent Host and Device URMA workspaces."""

import argparse

import numpy as np
from run import ROOT, RUNTIME, close_preserving_error, device_array, diagnostics, host_array, make_worker, read_device
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs, TensorArgType

from simpler_setup.scene_test import compile_chip_callable_spec


def build_callable():
    spec = {
        "orchestration": {
            "source": str(ROOT / "kernels/orchestration/coexist.cpp"),
            "function_name": "host_device_urma",
            "signature": [D.IN, D.IN, D.OUT, D.OUT, D.OUT, D.OUT, D.OUT, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "source": str(ROOT / "kernels/aiv/coexist.cpp"),
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT, D.OUT, D.OUT, D.IN],
            },
            {
                "func_id": 1,
                "source": str(ROOT.parent / "host_ddr_gather/kernels/aiv/consumer.cpp"),
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
                "extra_include_dirs": ["$ASCEND_HOME_PATH/asc"],
            },
            {
                "func_id": 2,
                "source": str(ROOT.parent / "urma_deferred_completion_demo/kernels/aiv/kernel_consumer.cpp"),
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
        ],
    }
    return compile_chip_callable_spec(spec, "a5", RUNTIME, ("host_device_urma", "a5", RUNTIME))


def run(devices=(0, 1), *, staged=False, output_prefix=""):
    if len(devices) != 2:
        raise ValueError("the workspace coexistence example uses exactly two ranks")
    compiled = build_callable()
    worker = make_worker(devices, staged)
    entry = worker.register(compiled)
    try:
        worker.init()
        table = host_array(worker, np.full((4097, 256), 0x3F00, np.uint16))
        mapped = worker.map_view(table.tensor((4096, 256), DataType.BFLOAT16, byte_offset=512))
        device_inputs = [host_array(worker, np.arange(16384, dtype=np.float32) + rank * 20000) for rank in range(2)]
        rank_buffers = []
        for rank in range(2):
            data = [
                (np.zeros((1, 256), np.uint16), DataType.BFLOAT16),
                (np.zeros(16384, np.float32), DataType.FLOAT32),
                (np.zeros((1, 256), np.float32), DataType.FLOAT32),
                (np.zeros(16384, np.float32), DataType.FLOAT32),
                (np.zeros(1, np.int32), DataType.INT32),
            ]
            rank_buffers.append(
                [(device_array(worker, rank, value, dtype), value.shape, dtype) for value, dtype in data]
            )
        domain_owner = []

        def orchestrate(orch, _args, config):
            domain = orch.get_or_allocate_domain(
                name="device_urma_alongside_host",
                members=[0, 1],
                window_size=4 * 1024 * 1024,
                buffers=[CommBufferSpec("input", "float32", 16384, 16384 * 4)],
                retain_after_run=True,
                workspace_engines=("urma",),
            )
            if not domain_owner:
                domain_owner.append(domain)
                for rank in range(2):
                    orch.copy_to(domain[rank].buffers["input"], device_inputs[rank])
            args_by_rank = []
            for rank, buffers in enumerate(rank_buffers):
                args = TaskArgs()
                args.add_tensor(mapped)
                args.add_tensor(domain[rank].buffers["input"].tensor((16384,), DataType.FLOAT32))
                for buffer, shape, dtype in buffers:
                    args.add_tensor(buffer.tensor(shape, dtype), TensorArgType.OUTPUT_EXISTING)
                args.add_scalar(domain.device_context(rank, workspace_engines=("urma",)))
                args_by_rank.append(args)
            orch.submit_next_level_group(entry, args_by_rank, config, workers=[0, 1])

        pending = worker.submit(orchestrate, None, diagnostics(output_prefix, 0))
        following = worker.submit(orchestrate, None, diagnostics(output_prefix, 1)) if staged else None
        pending.wait()
        if following is not None:
            following.wait()
        for rank, buffers in enumerate(rank_buffers):
            np.testing.assert_array_equal(read_device(worker, buffers[0][0], (1, 256), np.uint16), 0x3F00)
            expected = np.arange(16384, dtype=np.float32) + (1 - rank) * 20000
            np.testing.assert_array_equal(read_device(worker, buffers[1][0], (16384,), np.float32), expected)
            np.testing.assert_array_equal(read_device(worker, buffers[2][0], (1, 256), np.float32), 1.5)
            np.testing.assert_array_equal(read_device(worker, buffers[3][0], (16384,), np.float32), expected + 1)
            np.testing.assert_array_equal(read_device(worker, buffers[4][0], (1,), np.int32), 0)
            print(f"PASS rank={rank}: Host and Device URMA completion in one producer", flush=True)
        worker.unmap_view(mapped)
        domain_owner[0].release()
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
