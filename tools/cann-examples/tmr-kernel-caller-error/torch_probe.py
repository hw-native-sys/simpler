# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Observe native AICPU errors through an equivalent torch_npu multi-stream topology."""

from __future__ import annotations

import argparse
import ctypes
import os

import torch_npu


def _stream_address(stream) -> int:
    return int(stream.npu_stream)


def _load_bridge(path: str):
    bridge = ctypes.CDLL(path)
    bridge.rts_error_probe_init.argtypes = [ctypes.c_int, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_char_p]
    bridge.rts_error_probe_init.restype = ctypes.c_int
    bridge.rts_error_probe_launch.argtypes = [ctypes.c_size_t, ctypes.c_int32, ctypes.c_int32]
    bridge.rts_error_probe_launch.restype = ctypes.c_int
    return bridge


def _enqueue_chain(bridge, caller, producer, status: int, aicpu_num: int) -> None:
    start = torch_npu.npu.Event()
    done = torch_npu.npu.Event()
    start.record(caller)
    producer.wait_event(start)
    rc = bridge.rts_error_probe_launch(_stream_address(producer), status, aicpu_num)
    if rc != 0:
        raise RuntimeError(f"rts_error_probe_launch enqueue failed: {rc}")
    done.record(producer)
    caller.wait_event(done)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device", type=int)
    parser.add_argument("dispatcher_so")
    parser.add_argument("probe_so")
    parser.add_argument("host_bridge")
    parser.add_argument("mode", choices=("eager", "replay"))
    parser.add_argument("observation", choices=("caller", "device"))
    parser.add_argument("result", choices=("success", "error"))
    parser.add_argument("--aicpu-num", type=int, default=1)
    parser.add_argument("--expect-error", type=int, choices=(0, 1))
    args = parser.parse_args()

    torch_npu.npu.set_device(args.device)
    caller = torch_npu.npu.Stream(device=args.device)
    producer = torch_npu.npu.Stream(device=args.device)
    bridge = _load_bridge(args.host_bridge)
    rc = bridge.rts_error_probe_init(
        args.device,
        _stream_address(producer),
        os.fsencode(args.dispatcher_so),
        os.fsencode(args.probe_so),
    )
    if rc != 0:
        raise RuntimeError(f"rts_error_probe_init failed: {rc}")

    status = 0 if args.result == "success" else -47
    if args.mode == "eager":
        _enqueue_chain(bridge, caller, producer, status, args.aicpu_num)
    else:
        graph = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph, stream=caller):
            _enqueue_chain(bridge, caller, producer, status, args.aicpu_num)
        graph.replay()

    observed = None
    try:
        if args.observation == "caller":
            caller.synchronize()
        else:
            torch_npu.npu.synchronize()
    except Exception as exc:  # The concrete exception is the result under test.
        observed = f"{type(exc).__name__}: {exc}"

    has_error = observed is not None
    print(
        f"rts_error_probe mode={args.mode} observation={args.observation} "
        f"result={args.result} aicpu_num={args.aicpu_num} "
        f"observed_error={int(has_error)} detail={observed or 'none'}",
        flush=True,
    )
    os._exit(0 if args.expect_error is None or has_error == bool(args.expect_error) else 1)


if __name__ == "__main__":
    main()
