# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Caller-owned persistent DEVICE cache with device-side acceptance controls.

A fixed task consumes a proposal prefix and adds its weighted rows to persistent
cache. DEVICE cursor/count values change between calls without host readback.
This is a bounded state/acceptance reference, not a complete MTP model.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import threading
import weakref
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE_SHAPE = (8, 32)
CACHE_STRIDES = (64, 1)
CACHE_OFFSET = 64
BACKING_ELEMENTS = 10 * 64
PROPOSAL_SHAPE = (4, 32)
STEPS = ((0, 0), (1, 2), (6, 1), (0, 4), (4, 0), (3, 3), (2, 2), (7, 1), (4, 4), (0, 0), (5, 2), (1, 3))
_UNPROVEN = []


def _helpers():
    spec = importlib.util.spec_from_file_location("framework_persistent_helpers", HERE / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load framework bridge helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compile(runtime):
    from simpler.task_interface import ArgDirection as D  # noqa: PLC0415

    from simpler_setup.scene_test import compile_chip_callable_spec  # noqa: PLC0415

    signature = [D.IN, D.INOUT, D.IN, D.IN]
    return compile_chip_callable_spec(
        {
            "orchestration": {
                "source": str(HERE / "kernels/persistent_state_orchestration.cpp"),
                "function_name": "framework_persistent_state",
                "signature": signature,
            },
            "incores": [
                {
                    "func_id": 0,
                    "source": str(HERE / "kernels/aiv/persistent_state.cpp"),
                    "core_type": "aiv",
                    "signature": signature,
                }
            ],
        },
        "a2a3",
        runtime,
        f"framework-persistent-state-{runtime}",
    )


def _validate_storage(weights, backing, cache, proposals, control):
    import torch  # noqa: PLC0415

    for tensor, shape, dtype in (
        (weights, (32,), torch.float32),
        (backing, (BACKING_ELEMENTS,), torch.float32),
        (proposals, PROPOSAL_SHAPE, torch.float32),
        (control, (2,), torch.int32),
    ):
        if tensor.dtype != dtype or tuple(tensor.shape) != shape or not tensor.is_contiguous():
            raise ValueError("Persistent inputs require their declared dtype and contiguous backing geometry")
        if tensor.storage_offset() != 0:
            raise ValueError("Persistent input backings must start at their allocation origin")
        if tensor.device != backing.device:
            raise ValueError("Persistent inputs must reside on one device")
    if cache.dtype != torch.float32 or tuple(cache.shape) != CACHE_SHAPE or cache.stride() != CACHE_STRIDES:
        raise ValueError("Cache view requires FLOAT32 shape (8, 32) and strides (64, 1)")
    if cache.untyped_storage().data_ptr() != backing.untyped_storage().data_ptr():
        raise ValueError("Cache view must use the persistent caller backing")
    if cache.storage_offset() != CACHE_OFFSET or cache.data_ptr() != backing.data_ptr() + CACHE_OFFSET * 4:
        raise ValueError("Cache view has the wrong origin")


def _task_args(worker, weights, backing, cache, proposals, control):
    from simpler.buffer import AccessMode, Buffer  # noqa: PLC0415
    from simpler.task_interface import DataType, TaskArgs, Tensor, TensorArgType, TensorTransfer  # noqa: PLC0415

    _validate_storage(weights, backing, cache, proposals, control)
    buffers = tuple(
        Buffer.wrap(
            address=tensor.data_ptr(),
            nbytes=tensor.numel() * tensor.element_size(),
            location=worker.device_location,
            access=access,
        )
        for tensor, access in (
            (weights, AccessMode.READ),
            (backing, AccessMode.READWRITE),
            (proposals, AccessMode.READ),
            (control, AccessMode.READ),
        )
    )
    tensors = (
        Tensor(buffers[0], shapes=(32,), dtype=DataType.FLOAT32),
        Tensor(
            buffers[1], shapes=CACHE_SHAPE, strides=CACHE_STRIDES, byte_offset=CACHE_OFFSET * 4, dtype=DataType.FLOAT32
        ),
        Tensor(buffers[2], shapes=PROPOSAL_SHAPE, dtype=DataType.FLOAT32),
        Tensor(buffers[3], shapes=(2,), dtype=DataType.INT32),
    )
    args = TaskArgs()
    for tensor, tag in zip(
        tensors, (TensorArgType.INPUT, TensorArgType.INOUT, TensorArgType.INPUT, TensorArgType.INPUT)
    ):
        args.add_tensor(tensor, tag, transfer=TensorTransfer.NONE)
    return args, buffers, tensors


def _proposal_values(index):
    return tuple((index + 1) * (row + 1) * (-1 if (index + row) % 2 else 1) for row in range(4))


def _initial(torch):
    return torch.arange(BACKING_ELEMENTS, dtype=torch.float32).remainder_(23).sub_(11), torch.arange(1, 33).float()


def _reference_snapshots(torch):
    backing, weights = _initial(torch)
    cache = backing.as_strided(CACHE_SHAPE, CACHE_STRIDES, CACHE_OFFSET)
    snapshots = []
    for index, (cursor, accepted) in enumerate(STEPS):
        proposals = torch.tensor(_proposal_values(index), dtype=torch.float32).reshape(4, 1)
        cache[cursor : cursor + accepted].add_(proposals[:accepted] * weights)
        snapshots.append(backing.clone())
    return snapshots


def _verify_snapshots(torch, frame):
    untouched = torch.ones(BACKING_ELEMENTS, dtype=torch.bool)
    untouched.as_strided(CACHE_SHAPE, CACHE_STRIDES, CACHE_OFFSET).fill_(False)
    checksums = []
    previous_actual = frame["initial"]
    for index, (snapshot, expected) in enumerate(zip(frame["snapshots"], frame["expected"])):
        actual = snapshot.cpu()
        if STEPS[index][1] == 0:
            assert torch.equal(actual, previous_actual), "Zero acceptance changed persistent state"
        assert torch.equal(actual, expected), f"Persistent cache mismatch after call {index} controls={STEPS[index]}"
        assert torch.equal(actual[untouched], frame["initial"][untouched]), "Cache guard or padding was modified"
        checksums.append(actual.sum().item())
        previous_actual = actual
    assert torch.equal(frame["weights"].cpu(), frame["host_weights"]), "Read-only weights changed"
    return checksums


def _prepare_frame(torch, worker, handle, stream, raw_stream) -> tuple[dict, list]:
    initial, host_weights = _initial(torch)
    weights = torch.empty(32, dtype=torch.float32, device="npu")
    backing = torch.empty(BACKING_ELEMENTS, dtype=torch.float32, device="npu")
    cache = backing.as_strided(CACHE_SHAPE, CACHE_STRIDES, CACHE_OFFSET)
    proposals = torch.empty(PROPOSAL_SHAPE, dtype=torch.float32, device="npu")
    control = torch.empty(2, dtype=torch.int32, device="npu")
    snapshots = [torch.empty_like(backing) for _ in STEPS]
    args, buffers, tensors = _task_args(worker, weights, backing, cache, proposals, control)
    owned = (weights, backing, cache, proposals, control, *snapshots)
    for tensor in owned:
        tensor.record_stream(stream)
    return dict(
        weights=weights,
        host_weights=host_weights,
        initial=initial,
        backing=backing,
        cache=cache,
        proposals=proposals,
        control=control,
        snapshots=snapshots,
        expected=_reference_snapshots(torch),
        args=args,
        buffers=buffers,
        tensors=tensors,
        worker=worker,
        callable=handle,
        stream=stream,
        raw_stream=raw_stream,
    ), [weakref.ref(tensor) for tensor in owned]


def run(runtime, device, bridge_path):
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415, F401 -- registers the framework backend
    from simpler.worker import Worker  # noqa: PLC0415

    bridge = _helpers()._load_bridge(bridge_path)
    torch.npu.set_device(device)
    stream = torch.npu.Stream(device=device)
    raw_stream = stream.npu_stream
    worker = Worker(level=2, device_id=device, platform="a2a3", runtime=runtime)
    handle = worker.register(_compile(runtime))
    worker.init(borrowed_context_generation=1)
    frame, queued, runs = {}, [], []
    caller_thread, task_queue = threading.get_native_id(), os.environ.get("TASK_QUEUE_ENABLE", "1")
    retired = False
    try:
        with torch.npu.stream(stream):
            frame, references = _prepare_frame(torch, worker, handle, stream, raw_stream)
            gc.collect()
            assert all(reference() is not None for reference in references)
            # All host-to-device initialization precedes the first queued callback.
            frame["weights"].copy_(frame["host_weights"])
            frame["backing"].copy_(frame["initial"])
            proposal_rows = tuple(frame["proposals"].unbind())
            control_fields = tuple(frame["control"].unbind())
            for index, (cursor, accepted) in enumerate(STEPS):
                for row, value in zip(proposal_rows, _proposal_values(index)):
                    row.fill_(value)
                control_fields[0].fill_(cursor)
                control_fields[1].fill_(accepted)

                def submit():
                    runs.append(
                        frame["worker"].submit(frame["callable"], frame["args"], caller_stream=frame["raw_stream"])
                    )

                queued.append(bridge.enqueue(submit))
                torch.add(frame["backing"], 0, out=frame["snapshots"][index])
            bridge.finish_current_stream()
            assert len(runs) == len(STEPS)
            for call in queued:
                call.check()
                assert (call.thread_id != caller_thread) == (task_queue == "1")
            checksums = _verify_snapshots(torch, frame)
            assert all(reference() is not None for reference in references)
        for run_handle in runs:
            run_handle.wait()
        worker.close()
        for buffer in frame["buffers"]:
            buffer.close()
        retired = True
        summary = dict(
            runtime=runtime,
            task_queue=task_queue,
            calls=len(runs),
            controls=[list(step) for step in STEPS],
            cache_shape=list(CACHE_SHAPE),
            cache_strides=list(CACHE_STRIDES),
            cache_byte_offset=CACHE_OFFSET * 4,
            checksums=checksums,
            caller_thread=caller_thread,
            callback_threads=[call.thread_id for call in queued],
            caller_frame_retained=all(reference() is not None for reference in references),
            guard_verified=True,
            zero_accept_verified=True,
            weights_verified=True,
            persistent_state_verified=True,
            verified=True,
        )
        print("FRAMEWORK_PERSISTENT_STATE_PASS " + json.dumps(summary, sort_keys=True), flush=True)
        return summary
    finally:
        if not retired:
            _UNPROVEN.append((worker, handle, frame, queued, runs, stream))
            try:
                os.write(
                    2, b"FRAMEWORK_PERSISTENT_UNPROVEN_RETAINED: caller owners retained after unproven completion.\n"
                )
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["tensormap_and_ringbuffer", "host_build_graph"], required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--build-dir", type=Path, required=True)
    options = parser.parse_args()
    modules = list(options.build_dir.glob("_simpler_framework_queue*.so"))
    if not modules:
        modules = [_helpers().build_bridge(options.build_dir)]
    if len(modules) != 1:
        parser.error("--build-dir must contain exactly one framework queue bridge")
    run(options.runtime, options.device, modules[0])


if __name__ == "__main__":
    main()
