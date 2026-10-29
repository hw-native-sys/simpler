# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Numerical native submission through the real framework queue."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _helpers():
    return load("native_submit_helpers", Path(__file__).with_name("main.py"))


def run(runtime, device, bridge_path, *, platform="a2a3"):
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415, F401
    from simpler.worker import Worker  # noqa: PLC0415

    helpers = _helpers()
    bridge = helpers._load_bridge(bridge_path)
    torch.npu.set_device(device)
    stream = torch.npu.Stream(device=device)
    worker = Worker(level=2, device_id=device, platform=platform, runtime=runtime)
    callable = worker.register(helpers._compile(runtime, platform))
    worker.init(borrowed_context_generation=1)
    tickets, queued = [], []
    queue = os.environ.get("TASK_QUEUE_ENABLE", "1")
    try:
        with torch.npu.stream(stream):
            raw_stream = stream.npu_stream
            frames, _ = helpers._prepare_frames(torch, worker, callable, stream, raw_stream, 3)
            expected = 0
            for index, frame in enumerate(frames):
                a, b, output = frame["tensors"]
                if index:
                    a.copy_(frames[index - 1]["result"])
                else:
                    a.fill_(1)
                    expected = 1
                b.fill_(3)
                output.fill_(-999)
                expected = (expected + 3) * 2
                frame["expected"] = expected
                ticket = worker.prepare_submission(
                    callable,
                    frame["args"],
                    caller_stream=raw_stream,
                    keepalive=(stream, *frame["tensors"], *frame["buffers"]),
                )
                tickets.append(ticket)
                try:
                    queued.append(bridge.enqueue_native(ticket.native_handle))
                except BaseException:
                    # A queue error does not prove that its callback never started.
                    # Even if cancel returns false, tickets retains this invocation
                    # until the finally block drains it through Worker.close().
                    ticket.cancel()
                    raise
                torch.add(output, output, out=frame["result"])
            stream.synchronize()
            for call in queued:
                call.check()
            for ticket in tickets:
                ticket.wait()
                assert ticket.safe_to_release
            for frame in frames:
                torch.testing.assert_close(
                    frame["result"].cpu(), torch.full(helpers.SHAPE, frame["expected"], dtype=torch.float32)
                )

            cancelled = worker.prepare_submission(callable, frames[0]["args"], caller_stream=raw_stream)
            assert cancelled.cancel()
            assert cancelled.safe_to_release
            failed = bridge.enqueue_native(cancelled.native_handle)
            stream.synchronize()
            try:
                failed.check()
            except RuntimeError as error:
                assert "cancelled" in str(error)
            else:
                raise AssertionError("cancelled native token was accepted")
            duplicate = bridge.enqueue_native(tickets[0].native_handle)
            stream.synchronize()
            try:
                duplicate.check()
            except RuntimeError as error:
                assert "consumed" in str(error)
            else:
                raise AssertionError("native token was submitted twice")
        worker.close()
        assert stream.query()
        print(
            "NATIVE_SUBMISSION_PASS "
            + json.dumps(
                {
                    "runtime": runtime,
                    "platform": platform,
                    "task_queue": queue,
                    "calls": 3,
                    "expected": [frame["expected"] for frame in frames],
                    "cancelled_refused": True,
                    "duplicate_refused": True,
                }
            ),
            flush=True,
        )
    finally:
        for ticket in tickets:
            ticket.cancel()
        worker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, choices=("host_build_graph", "tensormap_and_ringbuffer"))
    parser.add_argument("--platform", default="a2a3", choices=("a2a3", "a5"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--bridge", type=Path)
    args = parser.parse_args()
    bridge = args.bridge
    if bridge is None:
        if args.build_dir is None:
            parser.error("provide --build-dir or --bridge")
        modules = list(args.build_dir.glob("_simpler_framework_queue*.so"))
        if not modules:
            modules = [_helpers().build_bridge(args.build_dir)]
        if len(modules) != 1:
            parser.error("--build-dir must contain exactly one framework queue bridge")
        bridge = modules[0]
    run(args.runtime, args.device, bridge, platform=args.platform)


if __name__ == "__main__":
    main()
