# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Delayed native handoff preserves HOST control values before queue consumption."""

import argparse
import gc
import importlib.util
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    args = parser.parse_args()

    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415, F401
    from simpler.buffer import Buffer  # noqa: PLC0415
    from simpler.task_interface import CallConfig, DataType, Tensor  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    root = Path(__file__).resolve().parents[4]
    path = root / "examples/workers/l2/framework_eager/value_arguments.py"
    spec = importlib.util.spec_from_file_location("host_value_helpers", path)
    assert spec is not None and spec.loader is not None
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    bridge = helpers._reference()._load_bridge(args.bridge)
    torch.npu.set_device(args.device)
    stream = torch.npu.Stream(device=args.device)
    worker = Worker(level=2, device_id=args.device, platform=args.platform, runtime="host_build_graph")
    tickets = []
    try:
        callable = worker.register(helpers._compile("host_build_graph", args.platform, host_control=True))
        worker.init(borrowed_context_generation=1)
        with torch.npu.stream(stream):
            source = torch.full(helpers.SHAPE, 2.0, device="npu")
            output = torch.empty_like(source)
            buffers = tuple(
                Buffer.wrap(address=t.data_ptr(), nbytes=t.numel() * 4, location=worker.device_location)
                for t in (source, output)
            )
            tensors = tuple(Tensor(buffer, shapes=helpers.SHAPE, dtype=DataType.FLOAT32) for buffer in buffers)
            control = torch.tensor([99, 7, 11, 13], dtype=torch.int32)
            control_arg = worker.make_tensor_arg(control[1::2], shapes=(2,), strides=(2,), dtype=DataType.INT32)
            call_args, scalar_wrappers = helpers._arguments(tensors, (1.0, 0, 0, 0.0), host_control=control_arg)
            config = CallConfig()
            ticket = worker.prepare_submission(
                callable,
                call_args,
                config,
                caller_stream=stream.npu_stream,
                keepalive=(source, output, stream, *buffers),
            )
            tickets.append(ticket)
            config.aicpu_thread_num = 9999
            control[1] = 19
            scalar_wrappers[0].value = 5.0
            call_args.clear()
            del control_arg, control, scalar_wrappers, call_args, config
            gc.collect()
            queued = bridge.enqueue_native(ticket.native_handle)
            successor = output + 0.5
            stream.synchronize()
            queued.check()
            ticket.wait()
            assert ticket.safe_to_release
            torch.testing.assert_close(successor.cpu(), torch.full(helpers.SHAPE, 9.5))
        worker.close()
        print("NATIVE_HOST_VALUES_PASS", flush=True)
    finally:
        helpers._reference()._close_with_tickets(worker, tickets)


if __name__ == "__main__":
    main()
