#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Retained A5 Host-source communication windows with local MTE and peer URMA reads."""

import argparse
import sys
from pathlib import Path

import numpy as np
from simpler.task_interface import AddressSpace, CommBufferSpec, DataType, TaskArgs, TensorArgType
from simpler.task_interface import ArgDirection as D
from simpler.worker import Worker

from simpler_setup.scene_test import compile_chip_callable_spec

ROOT = Path(__file__).resolve().parent
RUNTIME = "tensormap_and_ringbuffer"
ROWS = 64
WIDTH = 128
INDICES = (0, 63, 7, 7, 1, 62, 31, 32)
TABLE_BYTES = ROWS * WIDTH * 4
OUTPUT_BYTES = len(INDICES) * WIDTH * 4
PREFIX_BYTES = 256


def build_callable():
    spec = {
        "orchestration": {
            "source": str(ROOT / "kernels/orchestration/gather.cpp"),
            "function_name": "host_comm_domain",
            "config_name": "host_comm_domain_config",
            "signature": [D.IN, D.OUT, D.OUT, D.OUT, D.IN, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "source": str(ROOT / "kernels/aiv/gather.cpp"),
                "core_type": "aiv",
                "signature": [D.IN, D.OUT, D.OUT, D.IN],
            },
            {
                "func_id": 1,
                "source": str(ROOT / "kernels/aiv/consume.cpp"),
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
        ],
    }
    return compile_chip_callable_spec(spec, "a5", RUNTIME, ("host_comm_domain", "a5", RUNTIME))


def _table(rank, generation, *, device=False):
    values = np.arange(ROWS * WIDTH, dtype=np.float32).reshape(ROWS, WIDTH)
    return values + rank * 10000 + generation * 100000 + (500000 if device else 0)


def _specs(*, outputs=False):
    specs = [
        CommBufferSpec("reserved", "uint8", PREFIX_BYTES, PREFIX_BYTES),
        CommBufferSpec("table", "float32", ROWS * WIDTH, TABLE_BYTES),
    ]
    if outputs:
        specs.extend(
            CommBufferSpec(f"{source}_{name}", "float32", len(INDICES) * WIDTH, OUTPUT_BYTES)
            for source in ("host", "device")
            for name in ("local", "peer", "result")
        )
    else:
        specs.append(CommBufferSpec("local_echo", "float32", len(INDICES) * WIDTH, OUTPUT_BYTES))
    return specs


def _allocate(orch, source):
    host = source == "host"
    return orch.get_or_allocate_domain(
        name=f"engram_{source}",
        members=[0, 1],
        window_size=PREFIX_BYTES + TABLE_BYTES + (OUTPUT_BYTES if host else 6 * OUTPUT_BYTES),
        buffers=_specs(outputs=not host),
        memory_source=AddressSpace.HOST if host else AddressSpace.DEVICE,
        workspace_engines=("urma",),
        retain_after_run=True,
    )


def _write_host(domain, generation):
    offset, nbytes = domain.buffer_range("table")
    for rank in range(2):
        with domain.member(rank).buffers["table"].shm.buf[offset : offset + nbytes] as memory:
            memory[:] = _table(rank, generation).tobytes()


def _read_device(worker, buffer, scratch):
    worker.copy_from(scratch, buffer)
    with scratch.shm.buf[:] as memory:
        return np.frombuffer(memory, dtype=np.float32).reshape(len(INDICES), WIDTH).copy()


def run(devices=(0, 1), *, staged=False, aiv_blocks=1):
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError("this example requires two distinct A5 devices")
    if aiv_blocks not in (1, 4):
        raise ValueError("aiv_blocks must be 1 or 4")
    compiled = build_callable()
    worker = Worker(
        level=3,
        platform="a5",
        runtime=RUNTIME,
        device_ids=list(devices),
        num_sub_workers=0,
        pipeline_depth=2 if staged else 1,
        pending_run_depth=2 if staged else 1,
        launch_depth=1,
    )
    entry = worker.register(compiled)
    domains = {}
    try:
        worker.init()
        scratch = worker.create_buffer(OUTPUT_BYTES)

        def prepare(orch, _args, _config):
            domains["host"] = _allocate(orch, "host")
            domains["device"] = _allocate(orch, "device")
            for rank in range(2):
                orch.copy_to_domain(domains["device"], rank, _table(rank, 1, device=True).tobytes(), buffer="table")

        worker.run(prepare)
        host, device = domains["host"], domains["device"]
        original_host_id = host.domain_id
        original_host_bases = tuple(host.local_context(rank).local_window_base for rank in range(2))
        assert host.buffer_range("table") == (PREFIX_BYTES, TABLE_BYTES)
        _write_host(host, 1)

        def submit(orch, _args, config, *, sources=("host", "device")):
            target = orch.get_domain("engram_device")
            for source in sources:
                domain = _allocate(orch, source)
                assert domain is domains[source]
                for rank in range(2):
                    args = TaskArgs()
                    args.add_tensor(domain.member(rank).buffers["table"].tensor((ROWS, WIDTH), DataType.FLOAT32))
                    for name in ("local", "peer", "result"):
                        output = (
                            domain.member(rank).buffers["local_echo"]
                            if source == "host" and name == "local"
                            else target.member(rank).buffers[f"{source}_{name}"]
                        )
                        args.add_tensor(
                            output.tensor((len(INDICES), WIDTH), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING
                        )
                    args.add_scalar(domain.device_context(rank, workspace_engines=("urma",)))
                    args.add_scalar(aiv_blocks)
                    orch.submit_next_level(entry, args, config, worker=rank)

        def verify(generation, *, sources=("host", "device")):
            for source in sources:
                is_device = source == "device"
                source_generation = 1 if is_device else generation
                for rank in range(2):
                    local = _table(rank, source_generation, device=is_device)[list(INDICES)]
                    peer = _table(1 - rank, source_generation, device=is_device)[list(INDICES)]
                    for name, expected in (("local", local), ("peer", peer), ("result", peer + 1.0)):
                        if source == "host" and name == "local":
                            offset, nbytes = host.buffer_range("local_echo")
                            with host.member(rank).buffers["local_echo"].shm.buf[offset : offset + nbytes] as memory:
                                actual = np.frombuffer(memory, dtype=np.float32).reshape(len(INDICES), WIDTH).copy()
                        else:
                            actual = _read_device(worker, device.member(rank).buffers[f"{source}_{name}"], scratch)
                        np.testing.assert_array_equal(actual, expected)
            print(f"PASS generation={generation} sources={sources} staged={staged} aiv_blocks={aiv_blocks}", flush=True)

        for generation in (1, 1, 2):
            if generation == 2:
                _write_host(host, generation)
            offset, nbytes = host.buffer_range("local_echo")
            for rank in range(2):
                with host.member(rank).buffers["local_echo"].shm.buf[offset : offset + nbytes] as memory:
                    memory[:] = bytes(OUTPUT_BYTES)
            pending = worker.submit(submit)
            following = worker.submit(submit) if staged else None
            pending.wait()
            if following is not None:
                following.wait()
            verify(generation)
            assert host.domain_id == original_host_id and not host.freed
            assert tuple(host.local_context(rank).local_window_base for rank in range(2)) == original_host_bases

        saved_buffer = host.member(0).buffers["table"]
        host.release()
        assert host.released and host.freed
        assert saved_buffer.shm.buf is None

        def device_only(orch, args, config):
            submit(orch, args, config, sources=("device",))

        worker.run(device_only)
        verify(2, sources=("device",))
        assert not device.released
        device.release()
        assert device.freed and worker.live_domains == {}
        print("PASS Host-domain retention, CPU updates, local MTE, peer URMA and Device-domain coexistence", flush=True)
    finally:
        original = sys.exc_info()[1]
        try:
            worker.close()
        except BaseException as cleanup:
            if original is not None:
                raise original from cleanup
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1", help="Comma-separated devices reserved by task-submit")
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--aiv-blocks", type=int, choices=(1, 4), default=1)
    parser.add_argument("--prepare", action="store_true", help="Compile kernels without initializing devices")
    options = parser.parse_args()
    if options.prepare:
        build_callable()
    else:
        run(
            tuple(int(item) for item in options.devices.split(",")),
            staged=options.staged,
            aiv_blocks=options.aiv_blocks,
        )
