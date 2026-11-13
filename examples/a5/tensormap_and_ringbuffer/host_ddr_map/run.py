# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Two shared L3 Host tables mapped locally by every A5 child."""

import argparse
import socket
import sys
import tempfile
from contextlib import ExitStack
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


def build_callable(runtime="tensormap_and_ringbuffer"):
    root = ROOT.parent / "host_ddr_gather/kernels"
    orchestration = "gather_host_control.cpp" if runtime == "host_build_graph" else "gather.cpp"
    signature = [D.IN, D.IN, D.OUT, D.OUT, D.OUT]
    if runtime == "host_build_graph":
        signature.append(D.IN)
    signature.append(D.IN)
    spec = {
        "orchestration": {
            "source": str(root / "orchestration" / orchestration),
            "function_name": "host_ddr_gather",
            "signature": signature,
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


def make_worker(devices, staged, *, num_sub_workers=0):
    if len(devices) < 2 or len(set(devices)) != len(devices):
        raise ValueError("at least two distinct A5 devices are required")
    return Worker(
        level=3,
        platform="a5",
        runtime=RUNTIME,
        device_ids=list(devices),
        num_sub_workers=num_sub_workers,
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


def hold_run_at_gate(path):
    def hold(_args):
        with socket.socket(socket.AF_UNIX) as connection:
            connection.settimeout(60)
            connection.connect(path)
            connection.sendall(b"E")
            assert connection.recv(1) == b"R", "staged run gate was not released"

    return hold


def run(devices=(0, 1), *, staged=False, output_prefix=""):
    compiled = build_callable()
    with ExitStack() as resources:
        gate_dir = resources.enter_context(tempfile.TemporaryDirectory(prefix="host-map-"))
        listener = resources.enter_context(socket.socket(socket.AF_UNIX))
        listener.bind(str(Path(gate_dir) / "gate"))
        listener.listen(1)
        listener.settimeout(60)
        worker = make_worker(devices, staged, num_sub_workers=int(staged))
        entry = worker.register(compiled)
        gate = worker.register(hold_run_at_gate(listener.getsockname())) if staged else None
        try:
            worker.init()
            tables = [worker.create_buffer((ROWS + 1) * WIDTH * 2) for _ in range(2)]
            ids = np.array([0, ROWS - 1, 7, 7, -1, ROWS], np.int32)
            ids_host = host_array(worker, ids)
            outputs = [
                [
                    [
                        device_array(worker, rank, np.zeros((6, WIDTH), np.uint16), DataType.BFLOAT16),
                        device_array(worker, rank, np.zeros((6, WIDTH), np.float32), DataType.FLOAT32),
                        device_array(worker, rank, np.zeros(1, np.int32), DataType.INT32),
                    ]
                    for rank in range(len(devices))
                ]
                for _ in range(2 if staged else 1)
            ]

            def outer_args(source):
                args = TaskArgs()
                args.add_tensor(
                    source.tensor((ROWS, WIDTH), DataType.BFLOAT16, byte_offset=512), transfer=TensorTransfer.NONE
                )
                return args

            for round_id, (table_index, count) in enumerate(((0, 6), (1, 6), (0, 0), (1, 7), (0, 6))):
                expected_by_backing = []
                for backing, source in enumerate(tables):
                    with source.host_view() as memory:
                        host = np.ndarray((ROWS + 1, WIDTH), np.uint16, buffer=memory)
                        host[:] = (
                            0x3F00
                            + backing * 0x100
                            + (np.arange(ROWS + 1)[:, None] + np.arange(WIDTH)[None, :] + round_id * 17) % 128
                        )
                        expected = np.zeros((6, WIDTH), np.uint16)
                        if count == 6:
                            for index, row in enumerate(ids):
                                if 0 <= row < ROWS:
                                    expected[index] = host[row + 1]
                        expected_by_backing.append(expected)
                        del host
                zero = host_array(worker, np.zeros_like(expected))
                for ranks in outputs:
                    for gathered, _, _ in ranks:
                        worker.copy_to(gathered, zero)
                worker.release_buffer(zero)

                def orchestrate(invocation):
                    def dispatch(orch, outer, config):
                        view = outer.tensor(0)
                        args_by_rank = []
                        for gathered, result, status in outputs[invocation]:
                            args = TaskArgs()
                            args.add_tensor(view, transfer=TensorTransfer.MAP)
                            args.add_tensor(ids_host.tensor((6,), DataType.INT32), transfer=TensorTransfer.H2D)
                            args.add_tensor(
                                gathered.tensor((6, WIDTH), DataType.BFLOAT16), TensorArgType.OUTPUT_EXISTING
                            )
                            args.add_tensor(result.tensor((6, WIDTH), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
                            args.add_tensor(status.tensor((1,), DataType.INT32), TensorArgType.OUTPUT_EXISTING)
                            args.add_scalar(count)
                            args_by_rank.append(args)
                        orch.submit_next_level_group(entry, args_by_rank, config, workers=list(range(len(devices))))
                        if invocation == 0 and gate is not None:
                            orch.submit_sub(gate)

                    return dispatch

                pending = worker.submit(
                    orchestrate(0), outer_args(tables[table_index]), diagnostics(output_prefix, round_id)
                )
                try:
                    worker.release_buffer(tables[table_index])
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("accepted calls must retain shared Host backing")
                if staged:
                    connection, _ = listener.accept()
                    with connection:
                        connection.settimeout(60)
                        try:
                            assert connection.recv(1) == b"E", "first run did not enter its gate"
                            assert not pending.done, "first run completed before the second submission"
                            following = worker.submit(
                                orchestrate(1),
                                outer_args(tables[1 - table_index]),
                                diagnostics(output_prefix, round_id + 100),
                            )
                            assert not pending.done, "first run escaped its gate during the second submission"
                            print(
                                f"PASS staged gate round={round_id}: second graph submitted with first L3 run active",
                                flush=True,
                            )
                        finally:
                            connection.sendall(b"R")
                    following.wait()
                pending.wait()
                for invocation, ranks in enumerate(outputs):
                    backing = table_index if invocation == 0 else 1 - table_index
                    expected = expected_by_backing[backing]
                    for rank, (gathered, result, status) in enumerate(ranks):
                        label = (
                            f"rank={rank} round={round_id} invocation={invocation} backing={backing} requests={count}"
                        )
                        np.testing.assert_array_equal(
                            read_device(worker, gathered, expected.shape, np.uint16), expected, err_msg=label
                        )
                        golden = (expected.astype(np.uint32) << 16).view(np.float32) + 1.0
                        np.testing.assert_array_equal(
                            read_device(worker, result, expected.shape, np.float32), golden, err_msg=label
                        )
                        assert read_device(worker, status, (1,), np.int32)[0] == (
                            1 if count == 7 else 2 if count == 6 else 0
                        ), label
                        print(f"PASS {label}", flush=True)
            for table in tables:
                worker.release_buffer(table)
            print("PASS shared L3 Host MAP: local MTE, native H2D, dependent consumer, CPU updates", flush=True)
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
