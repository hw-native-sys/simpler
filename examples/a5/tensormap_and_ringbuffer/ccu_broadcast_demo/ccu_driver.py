# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host rendezvous and native resource ownership for the CCU Broadcast example."""

from __future__ import annotations

import ctypes
import json
import math
import multiprocessing as mp
import os
import subprocess
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNTIME = "tensormap_and_ringbuffer"
MAX_COUNT = 1024
MAX_REPEAT = 100


class PeerInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ("input", "output", "token", "trigger_status")]


def validate_request(devices, counts, roots, repeat, timeout):
    if (
        len(devices) != 2
        or any(isinstance(d, bool) or not isinstance(d, int) or not 0 <= d <= 0x7FFFFFFF for d in devices)
        or len(set(devices)) != 2
    ):
        raise ValueError("CCU Broadcast needs exactly two distinct nonnegative device IDs")
    if not counts or any(isinstance(n, bool) or not isinstance(n, int) or not 0 < n <= MAX_COUNT for n in counts):
        raise ValueError(f"Element counts must be integers in [1, {MAX_COUNT}]")
    if not roots or any(isinstance(r, bool) or not isinstance(r, int) or r not in (0, 1) for r in roots):
        raise ValueError("Broadcast roots must be rank 0 or rank 1")
    if not isinstance(repeat, int) or isinstance(repeat, bool) or not 1 <= repeat <= MAX_REPEAT:
        raise ValueError(f"Repeat must be an integer in [1, {MAX_REPEAT}]")
    if isinstance(timeout, bool) or not math.isfinite(timeout) or not 0 < timeout <= 600:
        raise ValueError("Control timeout must be in (0, 600] seconds")


def receive_stage(connection, stage, timeout, rank):
    if timeout <= 0 or not connection.poll(timeout):
        raise TimeoutError(f"CCU rank {rank} did not reach {stage!r} within the control deadline")
    try:
        message = connection.recv()
    except EOFError as error:
        raise RuntimeError(f"CCU rank {rank} exited before {stage!r}") from error
    if not isinstance(message, dict) or message.get("rank") != rank:
        raise RuntimeError(f"Invalid CCU control reply for rank {rank}")
    if message.get("stage") == "error":
        raise RuntimeError(f"CCU rank {rank} failed:\n{message.get('error', 'no error detail')}")
    if message.get("stage") != stage:
        raise RuntimeError(f"CCU rank {rank}: expected {stage!r}, got {message.get('stage')!r}")
    return message


def receive_all(connections, stage, timeout):
    deadline = time.monotonic() + timeout
    return [receive_stage(conn, stage, deadline - time.monotonic(), rank) for rank, conn in enumerate(connections)]


class NativeSession:
    def __init__(self, library):
        self.lib = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
        self.handle = ctypes.c_void_p()
        self.peer = PeerInfo()
        signatures = {
            "ccu_demo_last_error": ([], ctypes.c_char_p),
            "ccu_demo_root_info_size": ([], ctypes.c_size_t),
            "ccu_demo_get_root_info": ([ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
            "ccu_demo_open": (
                [
                    ctypes.c_int,
                    ctypes.c_uint32,
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_uint32,
                    ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.POINTER(PeerInfo),
                ],
                ctypes.c_int,
            ),
            "ccu_demo_prepare": (
                [
                    ctypes.c_void_p,
                    ctypes.POINTER(PeerInfo),
                    ctypes.c_size_t,
                    ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.POINTER(ctypes.c_uint32),
                ],
                ctypes.c_int,
            ),
            "ccu_demo_wait_and_verify": ([ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int], ctypes.c_int),
            "ccu_demo_close": ([ctypes.c_void_p], ctypes.c_int),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = arguments, result

    def check(self, rc):
        if rc != 0:
            raw = self.lib.ccu_demo_last_error()
            raise RuntimeError(raw.decode("utf-8", errors="replace") if raw else f"CCU helper failed: {rc}")

    def root_info(self, device):
        size = self.lib.ccu_demo_root_info_size()
        buffer = ctypes.create_string_buffer(size)
        self.check(self.lib.ccu_demo_get_root_info(device, buffer, size))
        return buffer.raw

    def open(self, device, rank, root_info, count, iteration):
        if self.handle:
            raise RuntimeError("CCU session is already open")
        expected_size = self.lib.ccu_demo_root_info_size()
        if len(root_info) != expected_size:
            raise ValueError("HCCL root-info size differs from the native SDK ABI")
        buffer = ctypes.create_string_buffer(root_info, expected_size)
        self.check(
            self.lib.ccu_demo_open(
                device,
                rank,
                buffer,
                expected_size,
                count,
                iteration,
                ctypes.byref(self.handle),
                ctypes.byref(self.peer),
            )
        )
        return tuple(getattr(self.peer, field) for field, _ in PeerInfo._fields_)

    def prepare(self, peers, root):
        if len(peers) != 2 or any(len(peer) != 4 for peer in peers):
            raise ValueError("CCU prepare requires both ranks' complete memory metadata")
        metadata = (PeerInfo * 2)(*(PeerInfo(*peer) for peer in peers))
        gate, mask = ctypes.c_uint64(), ctypes.c_uint32()
        self.check(self.lib.ccu_demo_prepare(self.handle, metadata, 2, root, ctypes.byref(gate), ctypes.byref(mask)))
        if not gate.value or not mask.value:
            raise RuntimeError("CCU prepare returned an empty gate")
        return gate.value, mask.value

    def verify(self, root):
        self.check(self.lib.ccu_demo_wait_and_verify(self.handle, root, 30000))

    def close(self):
        if self.handle:
            self.check(self.lib.ccu_demo_close(self.handle))
            self.handle = ctypes.c_void_p()


def build_helper():
    from simpler_setup.environment import PROJECT_ROOT  # noqa: PLC0415

    output = PROJECT_ROOT / "build" / "examples" / "a5_ccu_broadcast"
    subprocess.run(
        ["cmake", "-S", str(HERE), "-B", str(output), "-DCMAKE_BUILD_TYPE=RelWithDebInfo"],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(output), "--parallel", "2"], check=True)
    return output / "libccu_broadcast_session.so"


def build_callable():
    from simpler.task_interface import ArgDirection  # noqa: PLC0415

    from simpler_setup.scene_test import compile_chip_callable_spec  # noqa: PLC0415

    signature = [ArgDirection.SCALAR] * 5
    return compile_chip_callable_spec(
        {
            "orchestration": {
                "source": str(HERE / "kernels/orchestration/ccu_broadcast_orch.cpp"),
                "function_name": "ccu_broadcast_orchestration",
                "config_name": "ccu_broadcast_config",
                "signature": signature,
            },
            "incores": [
                {
                    "func_id": 0,
                    "source": str(HERE / "kernels/aiv/ccu_broadcast_trigger.cpp"),
                    "core_type": "aiv",
                    "signature": signature,
                }
            ],
        },
        "a5",
        RUNTIME,
        "a5-ccu-broadcast-trigger",
    )


def _command(connection, expected):
    command = connection.recv()
    if not isinstance(command, dict) or command.get("command") != expected:
        raise RuntimeError(f"CCU child expected command {expected!r}")
    return command


def _rank_entry(rank, device, count, root, iteration, library, bins, target, connection):
    os.environ["HCCL_OP_EXPANSION_MODE"] = "CCU_SCHED"
    worker, native = None, None
    initialized, failed = False, False
    try:
        from simpler.task_interface import CallConfig, ChipWorker, TaskArgs  # noqa: PLC0415

        worker = ChipWorker()
        worker.init(device, bins)
        initialized = True
        target_handle = worker.register_callable(target)
        native = NativeSession(library)
        connection.send({"stage": "ready", "rank": rank, "root_info": native.root_info(device) if rank == 0 else None})
        command = _command(connection, "open")
        peer = native.open(device, rank, command["root_info"], count, iteration)
        connection.send({"stage": "window", "rank": rank, "peer": peer})
        command = _command(connection, "prepare")
        gate, mask = native.prepare(command["peers"], root)
        connection.send({"stage": "armed", "rank": rank})
        _command(connection, "trigger")
        args = TaskArgs()
        for value in (gate, mask, native.peer.trigger_status, root, rank):
            args.add_scalar(value)
        worker.run(target_handle, args, config=CallConfig())
        native.verify(root)
        connection.send({"stage": "verified", "rank": rank})
        _command(connection, "close")
        native.close()
        worker.finalize()
        initialized = False
        connection.send({"stage": "closed", "rank": rank})
    except BaseException:  # noqa: BLE001 -- failures must cross the process boundary
        failed = True
        try:
            connection.send({"stage": "error", "rank": rank, "error": traceback.format_exc()})
        except (OSError, EOFError):
            pass
    finally:
        safe = True
        if native is not None:
            try:
                native.close()
            except RuntimeError:
                safe = False
        if worker is not None and initialized and safe:
            try:
                worker.finalize()
            except Exception:  # noqa: BLE001 -- retain an unproven native context until process exit
                safe = False
        connection.close()
        if failed or not safe:
            os._exit(1)


def run_case(devices, count, root, iteration, library, bins, target, timeout):
    context = mp.get_context("fork")
    connections, processes = [], []
    try:
        for rank, device in enumerate(devices):
            parent, child = context.Pipe()
            process = context.Process(
                target=_rank_entry, args=(rank, device, count, root, iteration, library, bins, target, child)
            )
            process.start()
            child.close()
            connections.append(parent)
            processes.append(process)
        ready = receive_all(connections, "ready", timeout)
        for connection in connections:
            connection.send({"command": "open", "root_info": ready[0]["root_info"]})
        windows = receive_all(connections, "window", timeout)
        peers = [message["peer"] for message in windows]
        for connection in connections:
            connection.send({"command": "prepare", "peers": peers})
        receive_all(connections, "armed", timeout)
        for connection in connections:
            connection.send({"command": "trigger"})
        receive_all(connections, "verified", timeout)
        for connection in connections:
            connection.send({"command": "close"})
        receive_all(connections, "closed", timeout)
        for process in processes:
            process.join(5)
            if process.exitcode != 0:
                raise RuntimeError(f"CCU rank process exited with status {process.exitcode}")
    finally:
        for connection in connections:
            connection.close()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)


def run_suite(devices, *, counts=(64, 1024), roots=(0, 1), repeat=2, timeout=180):
    validate_request(devices, counts, roots, repeat, timeout)
    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError("The real A5 CCU example requires Linux fork-based rank isolation")
    from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

    library = build_helper()
    target = build_callable()
    bins = RuntimeBuilder("a5").get_binaries(RUNTIME)
    cases = []
    for iteration in range(repeat):
        for root in roots:
            for count in counts:
                run_case(devices, count, root, iteration, library, bins, target, timeout)
                result = {
                    "root": root,
                    "count": count,
                    "iteration": iteration,
                    "ranks": 2,
                    "verified": True,
                    "guards": True,
                    "aiv_trigger": True,
                }
                cases.append(result)
                print("CCU_BROADCAST_CASE_PASS " + json.dumps(result, sort_keys=True), flush=True)
    summary = {
        "platform": "a5",
        "runtime": RUNTIME,
        "cases": len(cases),
        "verified": True,
        "completion": "host-stream-sync",
        "fresh_session_per_case": True,
    }
    print("CCU_BROADCAST_PASS " + json.dumps(summary, sort_keys=True), flush=True)
    return summary
