# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Hardware GIL and native-reference checks, separate from the numerical example."""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


def _check_gil_drains(torch, worker, callable, stream, frame, probe, tickets, shape):
    raw_stream = stream.npu_stream
    drain_calls = 0
    for drain in (stream.query, torch.npu.empty_cache):
        ticket = worker.prepare_submission(
            callable,
            frame["args"],
            caller_stream=raw_stream,
            keepalive=(stream, *frame["tensors"], *frame["buffers"]),
        )
        tickets.append(ticket)
        gate = probe.enqueue_submission(ticket.native_handle)
        probe.wait_ready(gate)
        previous_interval = sys.getswitchinterval()
        try:
            sys.setswitchinterval(60)
            probe.drain(gate, drain)
        finally:
            sys.setswitchinterval(previous_interval)
        stream.synchronize()
        ticket.wait()
        assert ticket.safe_to_release
        torch.testing.assert_close(frame["tensors"][2].cpu(), torch.full(shape, 4.0, dtype=torch.float32))
        drain_calls += 1

    return drain_calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location(
        "native_submit_helpers", root / "examples/workers/l2/framework_eager/main.py"
    )
    assert spec is not None and spec.loader is not None
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415, F401
    from simpler.worker import Worker  # noqa: PLC0415

    probe_spec = importlib.util.spec_from_file_location("_framework_queue_gil_probe", args.probe)
    assert probe_spec is not None and probe_spec.loader is not None
    probe = importlib.util.module_from_spec(probe_spec)
    probe_spec.loader.exec_module(probe)

    torch.npu.set_device(args.device)
    stream = torch.npu.Stream(device=args.device)
    worker = Worker(level=2, device_id=args.device, platform=args.platform, runtime=args.runtime)
    callable = worker.register(helpers._compile(args.runtime, args.platform))
    worker.init(borrowed_context_generation=1)
    tickets = []
    try:
        with torch.npu.stream(stream):
            frames, _ = helpers._prepare_frames(torch, worker, callable, stream, stream.npu_stream, 1)
            frame = frames[0]
            frame["tensors"][0].fill_(1)
            frame["tensors"][1].fill_(3)
            ticket = worker.prepare_submission(
                callable,
                frame["args"],
                caller_stream=stream.npu_stream,
                keepalive=(stream, *frame["tensors"], *frame["buffers"]),
            )
            tickets.append(ticket)
            reference = probe.retain_submission(ticket.native_handle)
            reference.release_in_thread()
            assert ticket.cancel() and ticket.safe_to_release
            probe.warmup()
            drain_calls = (
                _check_gil_drains(torch, worker, callable, stream, frame, probe, tickets, helpers.SHAPE)
                if os.environ.get("TASK_QUEUE_ENABLE", "1") == "1"
                else 0
            )
        worker.close()
        print("NATIVE_GIL_PASS " + json.dumps({"gil_drain_calls": drain_calls, "native_reference_released": True}))
    finally:
        for ticket in tickets:
            ticket.cancel()
        worker.close()


if __name__ == "__main__":
    main()
