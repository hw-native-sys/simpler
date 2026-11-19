# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cold Kernel capture through public preparation and the original Torch graph owner.

Uses stock torch_npu and requires ACL model destroy callback support.
Compilation and Worker initialization precede Begin; no business kernel is warmed up.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path

POISON = -999.0
INPUTS = (2, 7, 19)


def _helpers():
    spec = importlib.util.spec_from_file_location("cold_capture_helpers", Path(__file__).with_name("main.py"))
    if spec is None or spec.loader is None:
        raise ImportError("cannot load framework consumer helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_value(torch, tensor, value):
    actual = tensor.cpu()
    torch.testing.assert_close(actual, torch.full(actual.shape, value, dtype=actual.dtype), rtol=0, atol=0)


def _task_queue():
    mode = os.environ.get("TASK_QUEUE_ENABLE", "1")
    if mode not in ("0", "1"):
        raise ValueError("cold capture requires TASK_QUEUE_ENABLE=0 or 1")
    return mode


def _exercise(torch, worker, bridge, callable, config, stream, graph, frame, ticket_slot, *, task_queue=None):
    """Use the same public token/queue path inside the framework's Begin/End scope."""
    a, b, output = frame["tensors"]
    frame["task_queue"] = _task_queue() if task_queue is None else task_queue
    frame["rhs_probe"] = torch._C._WeakTensorRef(b)
    frame["rhs_roots_removed"] = False
    result = frame["result"]
    raw_stream = stream.npu_stream  # Resolve before Begin: the getter can drain the host queue.
    addresses = tuple(tensor.data_ptr() for tensor in (a, output, result))
    a.fill_(INPUTS[0])
    b.fill_(3)
    output.fill_(POISON)
    result.fill_(POISON)
    stream.synchronize()
    recording_error = None
    try:
        with torch.npu.graph(graph, stream=stream):
            try:
                # The existing Worker root holds the actual tensors as well as
                # their Buffer borrows, including on partial registration.
                ticket_slot[0] = worker.prepare_submission(
                    callable,
                    frame["args"],
                    config,
                    caller_stream=raw_stream,
                    keepalive=(stream, *frame["buffers"], *frame["tensors"]),
                )
                # This is the first native node in this exact graph scope.
                ticket_slot[0]._native_submission._retain_model_resources()
                call = bridge.enqueue_native(ticket_slot[0].native_handle, frame["tensors"])
                call.check()
                torch.add(output, output, out=result)
            except BaseException as error:
                recording_error = error
                raise
    except BaseException as error:
        if recording_error is not None and error is not recording_error:
            error.__context__ = None
            raise recording_error from error
        raise
    stream.synchronize()
    # Buffer borrows do not retain Torch tensors. Queue 0 observes the ticket's
    # TensorImpl roots; queue 1 can retain asynchronous
    # ReleaseQueue roots after the local Call is dropped.
    del call
    frame["tensors"] = (a, output)
    del b
    frame["rhs_roots_removed"] = True
    gc.collect()
    if frame["task_queue"] == "0":
        assert not frame["rhs_probe"].expired(), "graph did not retain the original RHS TensorImpl"
    _assert_value(torch, output, POISON)
    _assert_value(torch, result, POISON)
    ticket = ticket_slot[0]
    assert not ticket.safe_to_release, "Host carrier retirement released the captured model's owners"
    try:
        ticket.wait(timeout=0)
    except TimeoutError:
        pass
    else:
        raise AssertionError("captured resources retired before graph destruction")
    try:
        worker.unregister(callable)
    except RuntimeError as error:
        assert "unretired L2 use" in str(error)
    else:
        raise AssertionError("callable unregistered while its graph still retained the run")
    expected = []
    for value in INPUTS:
        with torch.npu.stream(stream):
            a.fill_(value)
            graph.replay()
        stream.synchronize()
        _assert_value(torch, output, value + 3)
        _assert_value(torch, result, (value + 3) * 2)
        assert tuple(tensor.data_ptr() for tensor in (a, output, result)) == addresses
        assert not ticket.safe_to_release
        expected.append((value + 3) * 2)
    return expected


def _retire_ticket(ticket):
    from simpler.task_interface import RunPreparationRefused  # noqa: PLC0415

    # Stock reset can return without a destruction receipt. Native release
    # requires the registered ACL notice and checked carrier retirement.
    cancelled = ticket._native_submission._release_model_resources()
    try:
        ticket.wait(timeout=0)
    except BaseException as error:
        if not ticket.safe_to_release:
            raise
        if not cancelled or not isinstance(error, RunPreparationRefused):
            return error
    assert ticket.safe_to_release
    return None


def _close(worker, graph, ticket_slot, frames):
    ticket = ticket_slot[0]
    reset_error = None
    try:
        if graph is not None:
            graph.reset()
    except BaseException as error:
        if ticket is None or not ticket._native_submission._model_destroyed():
            raise
        reset_error = error

    try:
        ticket_error = _retire_ticket(ticket) if ticket is not None else None
        try:
            for frame in frames:
                if frame.get("task_queue") == "0" and frame.get("rhs_roots_removed"):
                    gc.collect()
                    assert frame["rhs_probe"].expired(), "checked release did not drop the original RHS TensorImpl"
            worker.close()
            for frame in frames:
                for buffer in frame["buffers"]:
                    buffer.close()
        except BaseException as error:
            if ticket_error is None:
                raise
            raise ticket_error from error
        if ticket_error is not None:
            raise ticket_error
    except BaseException as error:
        if reset_error is None:
            raise
        raise reset_error from error
    if reset_error is not None:
        raise reset_error


def run(runtime, device, bridge_path, *, platform="a2a3"):
    task_queue = _task_queue()  # Resolve before importing the backend that caches the option.
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415, F401 -- registers the actual framework backend
    from simpler.task_interface import CallConfig  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    helpers = _helpers()
    bridge = helpers._load_bridge(bridge_path)
    torch.npu.set_device(device)
    stream = torch.npu.Stream(device=device)
    worker = Worker(level=2, device_id=device, platform=platform, runtime=runtime)
    frames, ticket_slot, graph = [], [None], None
    try:
        callable = worker.register(helpers._compile(runtime, platform))
        worker.init(borrowed_context_generation=1)
        config = CallConfig()
        config.enable_dump_args = config.enable_chip_swimlane = config.enable_pmu = 0
        config.enable_scope_stats = config.enable_dep_gen = False
        with torch.npu.stream(stream):
            frames, _ = helpers._prepare_frames(torch, worker, callable, stream, stream.npu_stream, 1)
            graph = torch.npu.NPUGraph()
            expected = _exercise(
                torch, worker, bridge, callable, config, stream, graph, frames[0], ticket_slot, task_queue=task_queue
            )
    except BaseException as primary:
        try:
            _close(worker, graph, ticket_slot, frames)
        except BaseException as cleanup:
            raise primary from cleanup
        raise
    else:
        _close(worker, graph, ticket_slot, frames)
    print(
        "NATIVE_COLD_CAPTURE_PASS "
        + json.dumps(
            {
                "runtime": runtime,
                "platform": platform,
                "record_computed": False,
                "replay_inputs": list(INPUTS),
                "expected": expected,
                "model_retained_until_reset": True,
                "checked_close": True,
                "task_queue": task_queue,
                "tensor_impl_lifetime": "observed" if task_queue == "0" else "not_observed",
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, choices=("host_build_graph", "tensormap_and_ringbuffer"))
    parser.add_argument("--platform", default="a2a3", choices=("a2a3", "a5"))
    parser.add_argument("--device", type=int, default=0)
    bridge_source = parser.add_mutually_exclusive_group(required=True)
    bridge_source.add_argument("--bridge", type=Path)
    bridge_source.add_argument("--build-dir", type=Path)
    args = parser.parse_args()
    bridge = args.bridge if args.bridge is not None else _helpers().build_bridge(args.build_dir)
    run(args.runtime, args.device, bridge, platform=args.platform)


if __name__ == "__main__":
    main()
