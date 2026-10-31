# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Numerical native submission with real framework queue/GIL drain boundaries."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
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


def _load_probe(helpers, bridge_path, probe_path):
    if probe_path is None:
        builder = load("native_submit_probe_builder", helpers.ROOT / "tools/framework_queue_gil/probe.py")
        probe_path = builder.build(bridge_path.parent / "native-submit-probe")
    return load("_framework_queue_gil_probe", probe_path)


def run(runtime, device, bridge_path, *, platform="a2a3", probe_path=None):
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
        probe = _load_probe(helpers, bridge_path, probe_path)
        with torch.npu.stream(stream):
            raw_stream = stream.npu_stream
            frames, _ = helpers._prepare_frames(torch, worker, callable, stream, raw_stream, 3)
            probe.warmup()
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
                native_reference = probe.retain_submission(ticket.native_handle)
                native_reference.release_in_thread()
                if queue == "1":
                    gate = probe.enqueue_submission(ticket.native_handle)
                    probe.wait_ready(gate)
                    # The gated native callback starts while this thread holds GIL.
                    previous_interval = sys.getswitchinterval()
                    try:
                        sys.setswitchinterval(60)
                        probe.drain(gate, stream.query if index % 2 == 0 else torch.npu.empty_cache)
                    finally:
                        sys.setswitchinterval(previous_interval)
                else:
                    queued.append(bridge.enqueue_native(ticket.native_handle))
                torch.add(output, output, out=frame["result"])
            stream.synchronize()
            for call in queued:
                call.check()
            for ticket in tickets:
                ticket.wait()
            for frame in frames:
                torch.testing.assert_close(
                    frame["result"].cpu(), torch.full(helpers.SHAPE, frame["expected"], dtype=torch.float32)
                )

            cancelled = worker.prepare_submission(callable, frames[0]["args"], caller_stream=raw_stream)
            assert cancelled.cancel()
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
    parser.add_argument("--probe", type=Path)
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
    run(args.runtime, args.device, bridge, platform=args.platform, probe_path=args.probe)


if __name__ == "__main__":
    main()
