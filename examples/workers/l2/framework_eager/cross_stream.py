# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Producer → caller A/B/A → consumer on one borrowed public Worker.

Framework events carry the cross-stream dependencies. Only the final consumer
boundary is explicitly drained; native admission may still apply backpressure.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import threading
from pathlib import Path

import main as reference

_UNPROVEN = []
_CALLERS = ("caller_A", "caller_B", "caller_A")


def run(runtime: str, device: int, bridge_path: Path, *, platform="a2a3"):
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415, F401 -- registers the NPU framework backend
    from simpler.worker import Worker  # noqa: PLC0415

    bridge = reference._load_bridge(bridge_path)
    torch.npu.set_device(device)
    streams = {role: torch.npu.Stream(device=device) for role in ("producer", "caller_A", "caller_B", "consumer")}
    # Raw handle access can flush host emission. All handles are resolved before
    # any producer/callback/consumer chain, never from inside its callbacks.
    raw_streams = {role: stream.npu_stream for role, stream in streams.items()}
    assert len(set(raw_streams.values())) == len(streams)
    caller_thread = threading.get_native_id()
    task_queue = os.environ.get("TASK_QUEUE_ENABLE", "1")
    worker, handle = None, None
    frames, queued, runs, events, edges, submissions, weak_tensors = [], [], [], [], [], [], []
    retired = False
    try:
        worker = Worker(level=2, device_id=device, platform=platform, runtime=runtime)
        handle = worker.register(reference._compile(runtime, platform=platform))
        worker.init(borrowed_context_generation=1)
        for index, role in enumerate(_CALLERS):
            prepared, references = reference._prepare_frames(torch, worker, handle, streams[role], raw_streams[role], 1)
            frame = prepared[0]
            frame["index"], frame["caller_role"] = index, role
            for tensor in frame["tensors"]:
                tensor.record_stream(streams["producer"])
            frame["tensors"][2].record_stream(streams["consumer"])
            frame["result"].record_stream(streams["consumer"])
            frames.append(frame)
            weak_tensors.extend(references)
            events.append((torch.npu.Event(), torch.npu.Event()))
        # The helper's construction scope is over. Strong frame references,
        # rather than record_stream alone, retain all backing and argument owners.
        del prepared, references, frame, tensor
        gc.collect()
        assert all(reference() is not None for reference in weak_tensors)
        print("FRAMEWORK_CROSS_STREAM_BEGIN " + json.dumps(raw_streams, sort_keys=True), flush=True)
        for index, frame in enumerate(frames):
            role = frame["caller_role"]
            caller = streams[role]
            ready, done = events[index]
            a, b, output = frame["tensors"]
            with torch.npu.stream(streams["producer"]):
                a.fill_(index + 1)
                b.fill_(index + 3)
                output.fill_(-999)
                ready.record(streams["producer"])
            with torch.npu.stream(caller):
                caller.wait_event(ready)
                edges.append({"call": index, "source": "producer", "target": role, "event": "ready"})

                def submit(frame=frame):
                    caller_stream = frame["raw_stream"]
                    runs.append(frame["worker"].submit(frame["callable"], frame["args"], caller_stream=caller_stream))
                    submissions.append(
                        {"call": frame["index"], "caller": frame["caller_role"], "raw_stream": caller_stream}
                    )

                queued.append(bridge.enqueue(submit))
                done.record(caller)
            with torch.npu.stream(streams["consumer"]):
                streams["consumer"].wait_event(done)
                edges.append({"call": index, "source": role, "target": "consumer", "event": "done"})
                torch.add(output, output, out=frame["result"])
            frame["expected"] = (2 * index + 4) * 2
        with torch.npu.stream(streams["consumer"]):
            # This is the first explicit completion boundary. Each consumer
            # waits on the public Worker's joined caller-stream completion.
            bridge.finish_current_stream()
        for call in queued:
            call.check()
            assert (call.thread_id != caller_thread) == (task_queue == "1")
        assert [submission["call"] for submission in submissions] == list(range(len(_CALLERS)))
        assert len(runs) == len(_CALLERS)
        observed = []
        for frame in frames:
            value = frame["result"].cpu()
            assert torch.equal(value, torch.full(reference.SHAPE, frame["expected"], dtype=value.dtype))
            observed.append(value[0, 0].item())
        for run_handle in runs:
            run_handle.wait()
        worker.close()
        for frame in frames:
            for buffer in frame["buffers"]:
                buffer.close()
        retired = True
        summary = {
            "platform": platform,
            "runtime": runtime,
            "task_queue": task_queue,
            "calls": len(_CALLERS),
            "caller_thread": caller_thread,
            "callback_threads": [call.thread_id for call in queued],
            "raw_streams": raw_streams,
            "submissions": submissions,
            "edges": edges,
            "expected": [8, 12, 16],
            "observed": observed,
            "caller_tensors_retained": all(reference() is not None for reference in weak_tensors),
            "consumer_drained_before_retirement": True,
            "verified": True,
        }
        print("FRAMEWORK_CROSS_STREAM_PASS " + json.dumps(summary, sort_keys=True), flush=True)
        return summary
    finally:
        if not retired:
            _UNPROVEN.append((worker, handle, frames, queued, runs, streams, events))
            try:
                os.write(2, b"FRAMEWORK_CROSS_STREAM_UNPROVEN: caller and runtime owners retained after failure.\n")
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["tensormap_and_ringbuffer", "host_build_graph"], required=True)
    parser.add_argument("--platform", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--build-dir", type=Path, default=reference.ROOT / "build/framework_eager")
    options = parser.parse_args()
    modules = list(options.build_dir.glob("_simpler_framework_queue*.so"))
    if len(modules) > 1:
        parser.error("--build-dir contains multiple framework queue modules")
    path = modules[0] if modules else reference.build_bridge(options.build_dir)
    print("FRAMEWORK_QUEUE_MODULE " + str(path), flush=True)
    run(options.runtime, options.device, path, platform=options.platform)


if __name__ == "__main__":
    main()
