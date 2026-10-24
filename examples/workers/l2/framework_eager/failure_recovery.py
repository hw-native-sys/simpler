# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Recover queued requests after a Python callback fails following acceptance.

This ordinary host exception does not cancel the accepted device run. The
framework boundary precedes init-owner-thread close, which retires every run
without individual RunHandle.wait calls. Device errors and unknown drains are
outside this example's recovery contract.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import threading
from pathlib import Path

import main as reference

_ERROR = "request callback failed after Worker.submit accepted its run"
_UNPROVEN = []


def _submit_frame(frame, runs, submissions):
    frame["callback_thread"] = threading.get_native_id()
    # Publication to the caller's retained list precedes any later host error.
    # The exception is not a device-completion or cancellation proof.
    runs.append(frame["worker"].submit(frame["callable"], frame["args"], caller_stream=frame["raw_stream"]))
    submissions.append(frame["index"])
    if frame["index"] == 0:
        raise RuntimeError(_ERROR)


def run(runtime: str, device: int, bridge_path: Path):
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415, F401 -- registers the NPU framework backend
    from simpler.worker import Worker  # noqa: PLC0415

    bridge = reference._load_bridge(bridge_path)
    torch.npu.set_device(device)
    stream = torch.npu.Stream(device=device)
    # Access may flush host emission, so resolve before queuing any work.
    raw_stream = stream.npu_stream
    caller_thread = threading.get_native_id()
    task_queue = os.environ.get("TASK_QUEUE_ENABLE", "1")
    worker, handle = None, None
    frames, queued, runs, submissions = [], [], [], []
    retired = False
    try:
        worker = Worker(level=2, device_id=device, platform="a2a3", runtime=runtime)
        handle = worker.register(reference._compile(runtime))
        worker.init(borrowed_context_generation=1)
        frames, weak_tensors = reference._prepare_frames(torch, worker, handle, stream, raw_stream, 3)
        gc.collect()
        assert all(ref() is not None for ref in weak_tensors)
        with torch.npu.stream(stream):
            for index, frame in enumerate(frames):
                a, b, output = frame["tensors"]
                a.fill_(index + 1)
                b.fill_(index + 3)
                output.fill_(-999)
                queued.append(bridge.enqueue(lambda frame=frame: _submit_frame(frame, runs, submissions)))
                torch.add(output, output, out=frame["result"])
            # All callbacks and framework successors precede this one boundary.
            # Native admission may independently apply capacity backpressure.
            bridge.finish_current_stream()
        try:
            queued[0].check()
        except RuntimeError as error:
            assert _ERROR in str(error)
        else:
            raise AssertionError("the accepted request's callback error was lost")
        for call in queued[1:]:
            call.check()
        assert submissions == [0, 1, 2] and len(runs) == 3
        assert [call.thread_id for call in queued[1:]] == [frame["callback_thread"] for frame in frames[1:]]
        assert all((frame["callback_thread"] != caller_thread) == (task_queue == "1") for frame in frames)
        observed = []
        for index, frame in enumerate(frames):
            value = frame["result"].cpu()
            assert torch.equal(value, torch.full(reference.SHAPE, 8 + 4 * index, dtype=value.dtype))
            observed.append(value[0, 0].item())
        retained_before_close = all(ref() is not None for ref in weak_tensors)
        assert retained_before_close
        # close belongs to the init thread, not the framework's queue thread.
        # It also retires the first run whose callback reported an exception.
        worker.close()
        for frame in frames:
            for buffer in frame["buffers"]:
                buffer.close()
                assert buffer.closed
        worker.close()
        with torch.npu.stream(stream):
            torch.add(frames[-1]["result"], frames[-1]["result"], out=frames[-1]["result"])
            bridge.finish_current_stream()
        value = frames[-1]["result"].cpu()
        assert torch.equal(value, torch.full(reference.SHAPE, 32, dtype=value.dtype))
        retired = True
        summary = {
            "runtime": runtime,
            "task_queue": task_queue,
            "calls": len(runs),
            "submissions": submissions,
            "callback_errors": [0],
            "caller_thread": caller_thread,
            "callback_threads": [frame["callback_thread"] for frame in frames],
            "observed": observed,
            "caller_tensors_retained": retained_before_close,
            "close_without_individual_wait": True,
            "buffers_closed_after_worker": True,
            "caller_stream_after_close": True,
            "verified": True,
        }
        print("FRAMEWORK_FAILURE_RECOVERY_PASS " + json.dumps(summary, sort_keys=True), flush=True)
        return summary
    finally:
        if not retired:
            # Callback failure alone does not establish that any backing is idle.
            _UNPROVEN.append((worker, handle, frames, queued, runs, stream))
            try:
                os.write(2, b"FRAMEWORK_FAILURE_RECOVERY_UNPROVEN: caller and runtime owners retained.\n")
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["tensormap_and_ringbuffer", "host_build_graph"], required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--build-dir", type=Path, default=reference.ROOT / "build/framework_eager")
    options = parser.parse_args()
    modules = list(options.build_dir.glob("_simpler_framework_queue*.so"))
    if len(modules) > 1:
        parser.error("--build-dir contains multiple framework queue modules")
    path = modules[0] if modules else reference.build_bridge(options.build_dir)
    print("FRAMEWORK_QUEUE_MODULE " + str(path), flush=True)
    run(options.runtime, options.device, path)


if __name__ == "__main__":
    main()
