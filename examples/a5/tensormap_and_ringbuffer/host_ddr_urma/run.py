# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One L3 Host table, nonself URMA reads and local MTE reads on every A5 child."""

import argparse
import sys
from pathlib import Path

import numpy as np
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, DataType, TaskArgs, TensorArgType, TensorTransfer
from simpler.worker import Worker

from simpler_setup.scene_test import compile_chip_callable_spec

ROOT = Path(__file__).resolve().parent
RUNTIME = "tensormap_and_ringbuffer"
ROWS = 4096
WIDTH = 256
REQUESTS = 64
INDEPENDENT_ROWS = 1024


def build_callable():
    local = ROOT.parent / "host_ddr_gather/kernels/aiv"
    spec = {
        "orchestration": {
            "source": str(ROOT / "kernels/orchestration/gather.cpp"),
            "function_name": "host_ddr_urma",
            "signature": [D.IN, D.IN, D.OUT, D.OUT, D.OUT, D.IN, D.OUT, D.OUT, D.OUT, D.IN, D.IN, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "source": str(ROOT / "kernels/aiv/gather.cpp"),
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT, D.OUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 1,
                "source": str(local / "consumer.cpp"),
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
                "extra_include_dirs": ["$ASCEND_HOME_PATH/asc"],
            },
            {
                "func_id": 2,
                "source": str(local / "gather.cpp"),
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT, D.OUT, D.IN],
                "extra_include_dirs": ["$ASCEND_HOME_PATH/asc"],
            },
        ],
    }
    return compile_chip_callable_spec(spec, "a5", RUNTIME, ("host_ddr_urma", "a5", RUNTIME))


def make_worker(devices, staged):
    if len(devices) < 2 or len(set(devices)) != len(devices):
        raise ValueError("at least two distinct A5 devices are required")
    return Worker(
        level=3,
        platform="a5",
        runtime=RUNTIME,
        device_ids=list(devices),
        num_sub_workers=0,
        pipeline_depth=2 if staged else 1,
        pending_run_depth=2 if staged else 1,
        launch_depth=1,
    )


def host_array(worker, value):
    value = np.ascontiguousarray(value)
    buffer = worker.create_buffer(value.nbytes)
    with buffer.host_view() as memory:
        memory[:] = value.tobytes()
    return buffer


def device_array(worker, rank, value, dtype):
    buffer = worker.alloc_child_tensor(rank, value.shape, dtype)
    source = host_array(worker, value)
    worker.copy_to(buffer, source)
    return buffer


def read_device(worker, buffer, shape, dtype):
    host = worker.create_buffer(buffer.nbytes)
    worker.copy_from(host, buffer)
    with host.host_view() as memory:
        result = np.frombuffer(memory, dtype=dtype).reshape(shape).copy()
    worker.release_buffer(host)
    return result


def close_preserving_error(worker):
    original = sys.exc_info()[1]
    try:
        worker.close()
    except BaseException as cleanup:
        if original is not None:
            raise original from cleanup
        raise


def diagnostics(prefix, round_id):
    config = CallConfig()
    if prefix:
        config.output_prefix = str(Path(prefix).resolve() / f"round{round_id}")
        config.enable_chip_swimlane = True
        config.enable_dep_gen = True
    return config


def remap_views(worker, native, mapped):
    stale = mapped[0]
    for view in mapped:
        worker.unmap_view(view)
    replacement = [worker.map_view(view) for view in native]
    try:
        worker.unmap_view(stale)
    except ValueError:
        pass
    else:
        raise AssertionError("remapping must not revive a revoked Host view")
    return replacement


def run(devices=(0, 1), *, staged=False, output_prefix=""):
    compiled = build_callable()
    worker = make_worker(devices, staged)
    entry = worker.register(compiled)
    try:
        worker.init()
        table = worker.create_buffer((ROWS + 1) * WIDTH * 2)
        native = (
            table.tensor((ROWS, WIDTH), DataType.BFLOAT16, byte_offset=512),
            table.tensor((ROWS - 15, WIDTH), DataType.BFLOAT16, byte_offset=16 * 512),
        )
        mapped = [worker.map_view(view) for view in native]
        independent_bits = np.full((INDEPENDENT_ROWS, WIDTH), 0x3F00, np.uint16)
        independent = host_array(worker, independent_bits)
        rank_buffers = []
        for rank in range(len(devices)):
            rank_buffers.append(
                [
                    device_array(worker, rank, np.zeros(REQUESTS, np.int32), DataType.INT32),
                    device_array(worker, rank, np.zeros((REQUESTS, WIDTH), np.uint16), DataType.BFLOAT16),
                    device_array(worker, rank, np.zeros((REQUESTS, WIDTH), np.float32), DataType.FLOAT32),
                    device_array(worker, rank, np.zeros(1, np.int32), DataType.INT32),
                    device_array(worker, rank, np.zeros((INDEPENDENT_ROWS, WIDTH), np.float32), DataType.FLOAT32),
                    device_array(worker, rank, np.zeros((REQUESTS, WIDTH), np.uint16), DataType.BFLOAT16),
                    device_array(worker, rank, np.zeros(1, np.int32), DataType.INT32),
                ]
            )

        rounds = ((0, REQUESTS), (1, REQUESTS), (0, 0), (1, REQUESTS + 1), (0, REQUESTS), (0, REQUESTS))
        for round_id, (view_index, count) in enumerate(rounds):
            if round_id == 4:
                mapped = remap_views(worker, native, mapped)
            source = native[view_index]
            row_count = source.shapes[0]
            indices = np.array([0, row_count - 1, 7, 7] * (REQUESTS // 4), np.int32)
            indices[-2:] = (-1, row_count)
            with table.host_view() as memory:
                host = np.ndarray((ROWS + 1, WIDTH), np.uint16, buffer=memory)
                # The last two rounds keep identical offsets and bytes while selecting different peers.
                if round_id != 5:
                    host[:] = 0x3F00 + (np.arange(ROWS + 1)[:, None] + np.arange(WIDTH)[None, :] + round_id * 17) % 128
                expected = np.zeros((REQUESTS, WIDTH), np.uint16)
                if count == REQUESTS:
                    for index, row in enumerate(indices):
                        if 0 <= row < row_count:
                            expected[index] = host[row + (1 if view_index == 0 else 16)]
                del host
            index_source = host_array(worker, indices)
            zero_source = host_array(worker, np.zeros_like(expected))
            args_by_rank = []
            for rank, buffers in enumerate(rank_buffers):
                ids, gathered, result, status, independent_result, local_result, local_status = buffers
                worker.copy_to(ids, index_source)
                worker.copy_to(gathered, zero_source)
                worker.copy_to(local_result, zero_source)
                args = TaskArgs()
                args.add_tensor(mapped[view_index])
                args.add_tensor(ids.tensor((REQUESTS,), DataType.INT32))
                args.add_tensor(gathered.tensor(expected.shape, DataType.BFLOAT16), TensorArgType.OUTPUT_EXISTING)
                args.add_tensor(result.tensor(expected.shape, DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
                args.add_tensor(status.tensor((1,), DataType.INT32), TensorArgType.OUTPUT_EXISTING)
                args.add_tensor(
                    independent.tensor((INDEPENDENT_ROWS, WIDTH), DataType.BFLOAT16), transfer=TensorTransfer.H2D
                )
                args.add_tensor(
                    independent_result.tensor((INDEPENDENT_ROWS, WIDTH), DataType.FLOAT32),
                    TensorArgType.OUTPUT_EXISTING,
                )
                args.add_tensor(local_result.tensor(expected.shape, DataType.BFLOAT16), TensorArgType.OUTPUT_EXISTING)
                args.add_tensor(local_status.tensor((1,), DataType.INT32), TensorArgType.OUTPUT_EXISTING)
                args.add_scalar(count)
                # Three or more ranks exercise the same table offsets through different peers.
                args.add_scalar((rank + 1 + round_id % (len(devices) - 1)) % len(devices))
                args.add_scalar(rank)
                args_by_rank.append(args)

            def orchestrate(orch, _args, config):
                orch.submit_next_level_group(entry, args_by_rank, config, workers=list(range(len(devices))))

            pending = worker.submit(orchestrate, None, diagnostics(output_prefix, round_id))
            following = (
                worker.submit(orchestrate, None, diagnostics(output_prefix, round_id * 2 + 100)) if staged else None
            )
            try:
                worker.unmap_view(mapped[view_index])
            except RuntimeError:
                pass
            else:
                raise AssertionError("an accepted L3 run must retain the Host registration")
            pending.wait()
            if following is not None:
                following.wait()
            expected_status = 2 if count == REQUESTS else 1 if count > REQUESTS else 0
            for rank, buffers in enumerate(rank_buffers):
                _, gathered, result, status, independent_result, local_result, local_status = buffers
                np.testing.assert_array_equal(read_device(worker, gathered, expected.shape, np.uint16), expected)
                np.testing.assert_array_equal(read_device(worker, local_result, expected.shape, np.uint16), expected)
                golden = (expected.astype(np.uint32) << 16).view(np.float32) + 1.0
                np.testing.assert_array_equal(read_device(worker, result, expected.shape, np.float32), golden)
                np.testing.assert_array_equal(
                    read_device(worker, independent_result, (INDEPENDENT_ROWS, WIDTH), np.float32), 1.5
                )
                assert read_device(worker, status, (1,), np.int32)[0] == expected_status
                assert read_device(worker, local_status, (1,), np.int32)[0] == expected_status
                peer = (rank + 1 + round_id % (len(devices) - 1)) % len(devices)
                print(f"PASS rank={rank} peer={peer} round={round_id} view={view_index} count={count}", flush=True)
        worker.unmap_view(mapped[0])
        worker.unmap_view(mapped[1])
        print("PASS shared L3 Host table: peer URMA, local MTE, native H2D, dependent consumer", flush=True)
    finally:
        close_preserving_error(worker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1", help="Comma-separated devices already reserved by task-submit")
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--prepare", action="store_true", help="Compile only; does not initialize devices")
    options = parser.parse_args()
    if options.prepare:
        build_callable()
    else:
        run(
            tuple(int(item) for item in options.devices.split(",")),
            staged=options.staged,
            output_prefix=options.output_prefix,
        )
