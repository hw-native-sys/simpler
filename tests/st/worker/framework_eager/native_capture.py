# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Matched-vendor cold capture payload and strict numerical/cleanup report validator."""

import argparse
import ctypes
import gc
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path


def _check_report(reports, runtime, platform, task_queue):
    if task_queue not in ("0", "1"):
        raise ValueError("cold capture requires TASK_QUEUE_ENABLE=0 or 1")
    assert reports == [
        {
            "runtime": runtime,
            "platform": platform,
            "record_computed": False,
            "replay_inputs": [2, 7, 19],
            "expected": [10, 20, 44],
            "model_retained_until_reset": True,
            "checked_close": True,
            "task_queue": task_queue,
            "tensor_impl_lifetime": "observed" if task_queue == "0" else "not_observed",
        }
    ], reports


def _check_retained_report(reports, runtime, task_queue):
    assert len(reports) == 1, reports
    (report,) = reports
    grants = report["grants"]
    assert type(grants) is int and grants >= 2
    count = grants + 1
    assert report == {
        "runtime": runtime,
        "platform": "a2a3",
        "task_queue": task_queue,
        "grants": grants,
        "models": count,
        "record_computed": False,
        "replays": [[base + 202 * index for base in (10, 20, 44)] for index in range(count)],
        "release_order": list(reversed(range(count))),
        "survivor_replays": [1006 + 2 * index for index in reversed(range(1, count))],
        "checked_close": True,
        "tensor_impl_lifetime": "observed" if task_queue == "0" else "not_observed",
    }, reports


def _load_example(path):
    spec = importlib.util.spec_from_file_location("stock_cold_capture_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_live(graph_ticket, frame, queue):
    assert not graph_ticket.safe_to_release and not graph_ticket._native_submission._model_destroyed()
    if queue == "0":
        gc.collect()
        assert not frame["rhs_probe"].expired()
    try:
        graph_ticket.wait(timeout=0)
    except TimeoutError:
        return
    raise AssertionError("model ticket retired before destruction")


@dataclass
class _Model:
    graph: object
    frame: dict
    rhs_value: int
    ticket: object = None


def _record(torch, worker, handle, config, stream, raw_stream, bridge, example, model, queue):
    frame = model.frame
    a, rhs, output = frame["tensors"]
    result = frame["result"]
    frame["rhs_probe"] = torch._C._WeakTensorRef(rhs)
    frame["pointers"] = tuple(tensor.data_ptr() for tensor in (a, output, result))
    a.fill_(example.INPUTS[0])
    rhs.fill_(model.rhs_value)
    output.fill_(example.POISON)
    result.fill_(example.POISON)
    stream.synchronize()

    recording_error = None
    try:
        with torch.npu.graph(model.graph, stream=stream, capture_error_mode="global"):
            try:
                model.ticket = worker.prepare_submission(
                    handle,
                    frame["args"],
                    config,
                    caller_stream=raw_stream,
                    keepalive=(stream, *frame["buffers"], *frame["tensors"]),
                )
                model.ticket._native_submission._retain_model_resources()
                queued = bridge.enqueue_native(model.ticket.native_handle, frame["tensors"])
                queued.check()
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
    del queued
    frame["tensors"] = (a, output)
    del rhs
    _require_live(model.ticket, frame, queue)


def _replay(torch, stream, model, example, value, queue):
    frame = model.frame
    a, output = frame["tensors"]
    result = frame["result"]
    a.fill_(value)
    model.graph.replay()
    stream.synchronize()
    example._assert_value(torch, output, value + model.rhs_value)
    expected = 2 * (value + model.rhs_value)
    example._assert_value(torch, result, expected)
    assert tuple(tensor.data_ptr() for tensor in (a, output, result)) == frame["pointers"]
    _require_live(model.ticket, frame, queue)
    return expected


def _require_callable_retained(worker, handle):
    try:
        worker.unregister(handle)
    except RuntimeError as error:
        assert "unretired L2 use" in str(error), str(error)
    else:
        raise AssertionError("callable unregistered with a live graph")


def _retained_child(args, example_path, roots, shim):
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: F401, PLC0415
    from simpler import worker as worker_module  # noqa: PLC0415
    from simpler.task_interface import CallConfig, RunPreparationRefused  # noqa: PLC0415

    example = _load_example(example_path)
    queue = example._task_queue()
    helpers = example._helpers()
    bridge = helpers._load_bridge(args.bridge)
    callable = helpers._compile(args.runtime, "a2a3")
    torch.npu.set_device(args.device)
    stream = torch.npu.Stream(device=args.device)
    worker = worker_module.Worker(level=2, device_id=args.device, platform="a2a3", runtime=args.runtime)
    roots.extend((worker, stream))
    handle = worker.register(callable)
    worker.init(borrowed_context_generation=1)
    grants = worker._chip_worker.pipeline_depth
    assert grants >= 2 and grants == worker._chip_worker.runtime_slot_count
    config = CallConfig()
    config.enable_dump_args = config.enable_chip_swimlane = config.enable_pmu = 0
    config.enable_scope_stats = config.enable_dep_gen = False
    with torch.npu.stream(stream):
        raw_stream = stream.npu_stream
        if shim is not None:
            assert queue == "0" and args.runtime == "tensormap_and_ringbuffer"
            frames, _ = helpers._prepare_frames(torch, worker, handle, stream, raw_stream, 1)
            graph, slot = torch.npu.NPUGraph(), [None]
            roots.extend((frames, graph, slot))
            replays = example._exercise(
                torch, worker, bridge, handle, config, stream, graph, frames[0], slot, task_queue=queue
            )
            assert replays == [10, 20, 44]
            _destroy_refusal(worker_module, worker, graph, slot, frames, example, shim, RunPreparationRefused)
            return {
                "runtime": args.runtime,
                "platform": "a2a3",
                "task_queue": queue,
                "expected": replays,
                "real_destroy_calls": 1,
                "acl_notice": True,
                "checked_close": True,
            }
        frames, _ = helpers._prepare_frames(torch, worker, handle, stream, raw_stream, grants + 1)
        models = [_Model(torch.npu.NPUGraph(), frame, 3 + index) for index, frame in enumerate(frames)]
        roots.append(models)
        for index, model in enumerate(models):
            _record(torch, worker, handle, config, stream, raw_stream, bridge, example, model, queue)
            for retained in models[: index + 1]:
                _require_live(retained.ticket, retained.frame, queue)

        assert len({id(model.ticket) for model in models}) == grants + 1
        _require_callable_retained(worker, handle)
        for frame in frames:
            example._assert_value(torch, frame["tensors"][1], example.POISON)
            example._assert_value(torch, frame["result"], example.POISON)
        print(f"COLD_MODELS_RECORDED grants={grants} live={len(models)} replays=0", flush=True)
        replays = [
            [_replay(torch, stream, model, example, value + 100 * index, queue) for value in example.INPUTS]
            for index, model in enumerate(models)
        ]

        release_order = []
        survivor_replays = []
        for index in reversed(range(len(models))):
            model = models[index]
            ticket, frame = model.ticket, model.frame
            model.graph.reset()
            assert ticket._native_submission._model_destroyed()
            assert not ticket.safe_to_release
            if queue == "0":
                assert not frame["rhs_probe"].expired()
            error = example._retire_ticket(ticket)
            if error is not None:
                raise error
            assert ticket.safe_to_release
            if queue == "0":
                gc.collect()
                assert frame["rhs_probe"].expired()
            release_order.append(index)
            for survivor in models[:index]:
                _require_live(survivor.ticket, survivor.frame, queue)
            if index:
                _require_callable_retained(worker, handle)
                survivor_replays.append(_replay(torch, stream, models[0], example, 500 + index, queue))
        worker.unregister(handle)
        worker.close()
        for frame in frames:
            for buffer in frame["buffers"]:
                buffer.close()

    return {
        "runtime": args.runtime,
        "platform": "a2a3",
        "task_queue": queue,
        "grants": grants,
        "models": len(models),
        "record_computed": False,
        "replays": replays,
        "release_order": release_order,
        "survivor_replays": survivor_replays,
        "checked_close": True,
        "tensor_impl_lifetime": "observed" if queue == "0" else "not_observed",
    }


class _DestroyShim:
    def __init__(self):
        lib = ctypes.CDLL(None)
        self.configure = lib.stock_capture_configure
        self.configure.argtypes, self.configure.restype = [ctypes.c_char_p], ctypes.c_int
        self.arm = lib.stock_capture_arm
        self.arm.argtypes, self.arm.restype = [], ctypes.c_int
        self.disarm = lib.stock_capture_disarm
        self.disarm.argtypes, self.disarm.restype = [], None
        self.stat = lib.stock_capture_stat
        self.stat.argtypes, self.stat.restype = [ctypes.c_uint], ctypes.c_uint64
        self.recover = lib.stock_capture_recover
        self.recover.argtypes, self.recover.restype = [ctypes.c_uint64], ctypes.c_int
        provider = (Path(os.environ["ASCEND_HOME_PATH"]) / "lib64/libascendcl.so").resolve(strict=True)
        assert self.configure(os.fsencode(provider)) == 0, provider

    def stats(self):
        keys = ("lookups", "injected", "real_calls", "saved_model", "last_real_rc", "blocked_calls")
        return {key: self.stat(index) for index, key in enumerate(keys)}


def _destroy_refusal(worker_module, worker, graph, slot, frames, example, shim, refusal):
    (frame,) = frames
    ticket = slot[0]
    native = ticket._native_submission
    original_root = worker_module._native_submission_roots[ticket]
    assert original_root[0] is worker and native._owner_recovery_required
    _require_live(ticket, frame, "0")
    assert not any(shim.stats()[key] for key in ("injected", "real_calls", "blocked_calls"))
    assert shim.arm() == 0
    try:
        try:
            example._close(worker, graph, slot, frames)
        except refusal as error:
            assert "native resource release requires the ACL model destroy notice" in str(error), str(error)
        else:
            raise AssertionError("checked close succeeded without the destroy notice")
    finally:
        shim.disarm()

    failed = shim.stats()
    assert failed["lookups"] >= 1 and failed["injected"] == 1 and failed["saved_model"] != 0, failed
    assert failed["real_calls"] == failed["blocked_calls"] == 0, failed
    try:
        graph.pool()
    except RuntimeError as error:
        assert "without a preceding successful capture" in str(error), str(error)
    else:
        raise AssertionError("stock reset did not clear the replay flag")
    assert worker_module._native_submission_roots[ticket] is original_root
    assert slot[0] is ticket and ticket._native_submission is native and native._owner_recovery_required
    _require_live(ticket, frame, "0")

    assert shim.recover(failed["saved_model"]) == 0, "same-handle SDK Destroy failed"
    rescued = shim.stats()
    assert rescued == {**failed, "real_calls": 1, "last_real_rc": 0}, rescued
    assert native._model_destroyed() and not frame["rhs_probe"].expired() and not ticket.safe_to_release
    # Drop the fixture's own reference to the exact root tuple before checking expiry.
    del original_root
    example._close(worker, graph, slot, frames)
    assert ticket.safe_to_release and frame["rhs_probe"].expired()
    assert worker._lifecycle is worker_module._Lifecycle.CLOSED
    assert shim.stats() == rescued


def _check_destroy_report(reports):
    assert reports == [
        {
            "runtime": "tensormap_and_ringbuffer",
            "platform": "a2a3",
            "task_queue": "0",
            "expected": [10, 20, 44],
            "real_destroy_calls": 1,
            "acl_notice": True,
            "checked_close": True,
        }
    ], reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, choices=("host_build_graph", "tensormap_and_ringbuffer"))
    parser.add_argument("--platform", required=True, choices=("a2a3", "a5"))
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--destroy-refusal", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--retained-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    example = Path(__file__).resolve().parents[4] / "examples/workers/l2/framework_eager/native_capture.py"
    if args.retained_child:
        assert args.platform == "a2a3"
        roots = []
        try:
            shim = _DestroyShim() if args.destroy_refusal else None
            report = _retained_child(args, example, roots, shim)
        except BaseException:
            # A failed replay or reset does not prove the outstanding models
            # can be destroyed. Keep their owners through subprocess exit.
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        print("NATIVE_COLD_CAPTURE_PASS " + json.dumps(report), flush=True)
        return

    args.output.mkdir(parents=True, exist_ok=False)
    ascend = args.output / "ascend"
    ascend.mkdir()
    argv = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve() if args.platform == "a2a3" else example),
        "--runtime",
        args.runtime,
        "--platform",
        args.platform,
        "--device",
        str(args.device),
        "--bridge",
        str(args.bridge.resolve()),
    ]
    if args.platform == "a2a3":
        argv += ["--output", str(args.output), "--retained-child"]
    env = {**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "ASCEND_PROCESS_LOG_PATH": str(ascend.resolve())}
    if args.destroy_refusal:
        assert args.platform == "a2a3" and args.runtime == "tensormap_and_ringbuffer"
        assert env.get("TASK_QUEUE_ENABLE") == "0" and not env.get("LD_PRELOAD")
        shim_path = args.output / "native_capture_destroy_shim.so"
        command = [
            "cc",
            "-std=c11",
            "-O2",
            "-fPIC",
            "-shared",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(Path(__file__).with_name("native_capture_destroy_shim.c")),
            "-o",
            str(shim_path),
            "-ldl",
        ]
        with (args.output / "shim-build.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=30, check=True)
        env["LD_PRELOAD"] = str(shim_path.resolve())
        argv.append("--destroy-refusal")
    (args.output / "command.json").write_text(json.dumps(argv, indent=2) + "\n")
    # Open the raw files before launch. They survive a timeout or a native crash
    # without a fallible post-error copy/chmod replacing the original failure.
    with (args.output / "stdout.log").open("w") as stdout, (args.output / "stderr.log").open("w") as stderr:
        completed = subprocess.run(
            argv,
            env=env,
            stdout=stdout,
            stderr=stderr,
            timeout=180,
            check=False,
        )
    completed.check_returncode()
    reports = [
        json.loads(line.removeprefix("NATIVE_COLD_CAPTURE_PASS "))
        for line in (args.output / "stdout.log").read_text().splitlines()
        if line.startswith("NATIVE_COLD_CAPTURE_PASS ")
    ]
    queue = os.environ.get("TASK_QUEUE_ENABLE", "1")
    if args.destroy_refusal:
        _check_destroy_report(reports)
    elif args.platform == "a2a3":
        _check_retained_report(reports, args.runtime, queue)
    else:
        _check_report(reports, args.runtime, args.platform, queue)
    print("COLD_CAPTURE_PAYLOAD_PASS " + json.dumps(reports[0]), flush=True)


if __name__ == "__main__":
    main()
