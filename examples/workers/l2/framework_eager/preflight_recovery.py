# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pre-accept source/context refusals recover inside real framework callbacks.

Caller allocations survive registration revocation. A caller-owned alternate RTS
context is used only for a rejected submit, then restored before healthy work.
The original context and stream remain owned by Torch throughout.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import importlib.util
import json
import os
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
_UNPROVEN = {}


def _helpers():
    spec = importlib.util.spec_from_file_location("framework_preflight_helpers", HERE / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load framework bridge helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _context_api():
    # The framework has already loaded this runtime; no other CANN library is loaded.
    lib = ctypes.CDLL("libascendcl.so", mode=os.RTLD_LOCAL | os.RTLD_NOLOAD)
    lib.aclrtGetCurrentContext.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.aclrtCreateContext.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    lib.aclrtSetCurrentContext.argtypes = [ctypes.c_void_p]
    lib.aclrtDestroyContext.argtypes = [ctypes.c_void_p]
    for name in ("aclrtGetCurrentContext", "aclrtCreateContext", "aclrtSetCurrentContext", "aclrtDestroyContext"):
        getattr(lib, name).restype = ctypes.c_int
    return lib


class _CallerContexts:
    """Borrow the framework base context and own only a cold-created alternate."""

    def __init__(self, lib, device):
        self.lib, self.device = lib, device
        self.base = None
        self.alternate = ctypes.c_void_p()
        self.restored = False
        self.destroyed = False

    def current(self):
        value = ctypes.c_void_p()
        rc = self.lib.aclrtGetCurrentContext(ctypes.byref(value))
        if rc != 0 or value.value is None:
            raise RuntimeError(f"Cannot query caller RTS context: rc={rc}")
        return value.value

    def restore(self, context):
        self.restored = False
        rc = self.lib.aclrtSetCurrentContext(context)
        actual = self.current()
        if rc != 0 or actual != context:
            raise RuntimeError(f"Caller RTS context restoration failed: rc={rc}")
        self.restored = True

    def create(self):
        self.base = self.current()
        try:
            rc = self.lib.aclrtCreateContext(ctypes.byref(self.alternate), self.device)
            if rc != 0 or self.alternate.value is None or self.alternate.value == self.base:
                raise RuntimeError(f"Cannot create a distinct caller RTS context: rc={rc}")
        finally:
            self.restore(self.base)

    def refuse(self, frame, runs):
        from simpler.task_interface import ExecutionContextRefused  # noqa: PLC0415

        entry = None
        try:
            entry = self.current()
            if entry != self.base:
                raise RuntimeError("Framework callback entered with an unexpected RTS context")
            rc = self.lib.aclrtSetCurrentContext(self.alternate)
            if rc != 0 or self.current() != self.alternate.value:
                raise RuntimeError(f"Cannot select alternate caller RTS context: rc={rc}")
            try:
                unexpected = frame["worker"].submit(frame["callable"], frame["args"], caller_stream=frame["raw_stream"])
            except ExecutionContextRefused:
                pass
            else:
                runs.append(unexpected)
                raise AssertionError("Wrong-context submit was accepted")
        finally:
            self.restore(self.base if entry is None else entry)

    def check_base(self):
        if self.current() != self.base:
            raise RuntimeError("The caller's original RTS context is not current")

    def close(self):
        self.check_base()
        try:
            rc = self.lib.aclrtDestroyContext(self.alternate)
            if rc != 0:
                raise RuntimeError(f"Caller alternate RTS context destruction failed: rc={rc}")
            self.destroyed = True
        finally:
            self.restore(self.base)


def _renew_registrations(frame):
    from simpler.buffer import Buffer  # noqa: PLC0415
    from simpler.task_interface import DataType, TaskArgs, Tensor, TensorArgType  # noqa: PLC0415

    old = frame["buffers"]
    for buffer in old:
        buffer.close()
    # These were never accepted; successful close proves no source use leaked.
    frame["retired_buffers"] = old
    frame["buffers"] = []
    frame["args"] = TaskArgs()
    for tensor, direction in zip(
        frame["tensors"], (TensorArgType.INPUT, TensorArgType.INPUT, TensorArgType.OUTPUT_EXISTING)
    ):
        buffer = Buffer.wrap(
            address=tensor.data_ptr(),
            nbytes=tensor.numel() * tensor.element_size(),
            location=frame["worker"].device_location,
        )
        frame["buffers"].append(buffer)
        frame["args"].add_tensor(Tensor(buffer, shapes=tuple(tensor.shape), dtype=DataType.FLOAT32), direction)
    assert not {buffer.identity for buffer in old} & {buffer.identity for buffer in frame["buffers"]}
    frame["registration_replaced"] = True


def _submit_frame(frame, contexts, runs, refusals):
    if frame["index"] == 0:
        try:
            unexpected = frame["worker"].submit(
                frame["callable"], frame["stale_args"], caller_stream=frame["raw_stream"]
            )
        except ValueError as error:
            identity = frame["stale_args"].tensor(0).buffer.identity
            assert str(error) == (
                f"orch.submit: device argument (arg 0, {identity}) is not a live "
                "allocation on target worker 0 (wrong worker, or stale)"
            ), "Unexpected source preflight refusal"
        else:
            runs.append(unexpected)
            raise AssertionError("Revoked source registration was accepted")
        _renew_registrations(frame)
        refusals.append("source")
    elif frame["index"] == 1:
        contexts.refuse(frame, runs)
        refusals.append("context")
    runs.append(frame["worker"].submit(frame["callable"], frame["args"], caller_stream=frame["raw_stream"]))


def run(runtime, device, bridge_path, *, platform="a2a3"):
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415, F401 -- registers the framework backend
    from simpler.worker import Worker  # noqa: PLC0415

    reference = _helpers()
    bridge = reference._load_bridge(bridge_path)
    torch.npu.set_device(device)
    stream = torch.npu.Stream(device=device)
    raw_stream = stream.npu_stream
    contexts = _CallerContexts(_context_api(), device)
    capsule: dict = dict(contexts=contexts, stream=stream, worker=None, callable=None, frames=[], queued=[], runs=[])
    # Retention precedes the first context-changing call, including failed creation.
    _UNPROVEN[id(capsule)] = capsule
    retired = False
    try:
        contexts.create()
        worker = Worker(level=2, device_id=device, platform=platform, runtime=runtime)
        capsule["worker"] = worker
        handle = worker.register(reference._compile(runtime, platform=platform))
        capsule["callable"] = handle
        worker.init(borrowed_context_generation=1)
        frames, references = reference._prepare_frames(torch, worker, handle, stream, raw_stream, 3)
        capsule["frames"] = frames
        frames[0]["stale_args"] = frames[0]["args"]
        frames[0]["buffers"][0].close()
        queued, runs, refusals = capsule["queued"], capsule["runs"], []
        gc.collect()
        assert all(reference() is not None for reference in references)
        caller_thread, task_queue = threading.get_native_id(), os.environ.get("TASK_QUEUE_ENABLE", "1")
        with torch.npu.stream(stream):
            for index, frame in enumerate(frames):
                a, b, output = frame["tensors"]
                a.fill_(index + 1)
                b.fill_(index + 3)
                output.fill_(-999)
                queued.append(bridge.enqueue(lambda frame=frame: _submit_frame(frame, contexts, runs, refusals)))
                torch.add(output, output, out=frame["result"])
            bridge.finish_current_stream()
        for call in queued:
            call.check()
            assert (call.thread_id != caller_thread) == (task_queue == "1")
        assert refusals == ["source", "context"] and len(runs) == 3
        observed = []
        for index, frame in enumerate(frames):
            actual = frame["result"].cpu()
            assert torch.equal(actual, torch.full(reference.SHAPE, 8 + 4 * index, dtype=actual.dtype))
            observed.append(actual[0, 0].item())
        for run_handle in runs:
            run_handle.wait()
        worker.close()
        for frame in frames:
            for buffer in frame["buffers"]:
                buffer.close()
        contexts.check_base()
        with torch.npu.stream(stream):
            frames[-1]["result"].add_(1)
            bridge.finish_current_stream()
        assert torch.equal(frames[-1]["result"].cpu(), torch.full(reference.SHAPE, 17.0))
        contexts.check_base()
        contexts.close()
        retired = True
        del _UNPROVEN[id(capsule)]
        summary = dict(
            platform=platform,
            runtime=runtime,
            task_queue=task_queue,
            calls=len(runs),
            observed=observed,
            refusals=refusals,
            source_registration_replaced=frames[0]["registration_replaced"],
            caller_context_preserved=contexts.restored,
            alternate_destroyed=contexts.destroyed,
            caller_stream_after_close=True,
            caller_tensors_retained=all(reference() is not None for reference in references),
            caller_thread=caller_thread,
            callback_threads=[call.thread_id for call in queued],
            verified=True,
        )
        print("FRAMEWORK_PREFLIGHT_RECOVERY_PASS " + json.dumps(summary, sort_keys=True), flush=True)
        return summary
    finally:
        if not retired:
            if contexts.base is not None:
                try:
                    contexts.restore(contexts.base)
                except Exception as error:  # noqa: BLE001 -- preserve the primary failure and all unresolved owners
                    capsule["failure_restore_error"] = error
            try:
                os.write(2, b"FRAMEWORK_PREFLIGHT_UNPROVEN_RETAINED: caller contexts and backing retained.\n")
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["tensormap_and_ringbuffer", "host_build_graph"], required=True)
    parser.add_argument("--platform", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--build-dir", type=Path, required=True)
    options = parser.parse_args()
    modules = list(options.build_dir.glob("_simpler_framework_queue*.so"))
    if not modules:
        modules = [_helpers().build_bridge(options.build_dir)]
    if len(modules) != 1:
        parser.error("--build-dir must contain exactly one framework queue bridge")
    run(options.runtime, options.device, modules[0], platform=options.platform)


if __name__ == "__main__":
    main()
