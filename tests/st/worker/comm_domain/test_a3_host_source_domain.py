# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A3 retained Host windows with local MTE and cross-rank SDMA."""

import re
import struct
import sys
from collections import deque
from functools import partial
from pathlib import Path

import pytest
from simpler.task_interface import AddressSpace, ArgDirection, CommBufferSpec, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from examples.workers.l4.global_tload_mixed_l3.main import COUNT, _build_tload_callable
from simpler_setup.scene_test import compile_chip_callable_spec

_RUNTIME = "tensormap_and_ringbuffer"
_SDMA_DIAGNOSTIC = re.compile(
    r"TIMEOUT_EXIT|\[ASYNC_WAIT|ProcessStarsSdmaErrorInfo|SMMU|\bCQE\b|(?:sched|orch)_error_code=", re.IGNORECASE
)


def _print_sdma_diagnostics(log_dir):
    found = False
    for path in sorted(log_dir.rglob("*.log")):
        try:
            with path.open(errors="replace") as stream:
                records = deque((line.rstrip() for line in stream if _SDMA_DIAGNOSTIC.search(line)), maxlen=64)
        except OSError as exc:
            print(f"Cannot read SDMA diagnostic log {path.name}: {exc}", file=sys.stderr)
            continue
        if records:
            found = True
            print(f"{path.name}:\n" + "\n".join(records), file=sys.stderr)
    if not found:
        print("No SDMA timeout or device fault records captured", file=sys.stderr)


@pytest.mark.platforms(["a2a3"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(1)
def test_a3_device_window_baseline(st_platform, st_device_ids):
    nbytes = COUNT * 4
    worker = Worker(level=3, platform=st_platform, runtime=_RUNTIME, device_ids=[st_device_ids[0]], num_sub_workers=0)
    chip = worker.register(_build_tload_callable(st_platform, _RUNTIME))
    try:
        worker.init()
        scratch = worker.create_buffer(nbytes)

        def run(orch, _args, cfg):
            domain = orch.allocate_domain(
                name="device",
                members=[0],
                window_size=2 * nbytes,
                buffers=[CommBufferSpec(name, "float32", COUNT, nbytes) for name in ("table", "out")],
                retain_after_run=True,
            )
            orch.copy_to_domain(domain, 0, struct.pack(f"<{COUNT}f", *([2.0] * COUNT)), buffer="table")
            args = TaskArgs()
            buffers = domain.member(0).buffers
            args.add_tensor(buffers["table"].tensor((COUNT,), DataType.FLOAT32), TensorArgType.INPUT)
            args.add_tensor(buffers["out"].tensor((COUNT,), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
            args.add_scalar(1)
            args.add_scalar(domain.device_context(0))
            orch.submit_next_level(chip, args, cfg, worker=0)

        worker.run(run)
        worker.copy_from(scratch, worker.live_domains["device"].member(0).buffers["out"])
        assert struct.unpack_from(f"<{COUNT}f", scratch.shm.buf) == (2.0,) * COUNT
    finally:
        worker.close()


def _build_sdma_callable(platform):
    root = Path(__file__).resolve().parents[4]
    kernels = root / "examples/a2a3/tensormap_and_ringbuffer/sdma_async_completion_demo/kernels"
    direction = ArgDirection
    return compile_chip_callable_spec(
        {
            "name": "a3_host_domain_sdma",
            "orchestration": {
                "source": str(kernels / "orchestration/sdma_async_completion_orch.cpp"),
                "function_name": "sdma_async_completion_orchestration",
                "signature": [direction.IN, direction.OUT, direction.OUT, direction.IN],
            },
            "incores": [
                {
                    "func_id": index,
                    "source": str(kernels / "aiv" / name),
                    "core_type": "aiv",
                    "signature": [direction.IN, direction.OUT],
                }
                for index, name in enumerate(("kernel_sdma_tget_async.cpp", "kernel_consumer.cpp"))
            ],
        },
        platform,
        _RUNTIME,
        ("a3_host_domain_sdma", platform, _RUNTIME),
    )


@pytest.mark.platforms(["a2a3"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(2)
@pytest.mark.parametrize(
    ("sdma", "depth"),
    [
        pytest.param(False, 1, id="local-mte"),
        pytest.param(True, 1, id="sdma-serial", marks=pytest.mark.sdma),
        pytest.param(True, 2, id="sdma-staged", marks=pytest.mark.sdma),
    ],
)
def test_a3_host_source_domain(st_platform, st_device_ids, sdma, depth, tmp_path, monkeypatch):
    if sdma:
        log_dir = tmp_path / "ascend"
        log_dir.mkdir()
        monkeypatch.setenv("ASCEND_PROCESS_LOG_PATH", str(log_dir))
    count = 128 * 128 if sdma else COUNT
    nbytes = count * 4
    prefix = 64
    engines = ("sdma",) if sdma else ()
    worker = Worker(
        level=3,
        platform=st_platform,
        runtime=_RUNTIME,
        device_ids=list(st_device_ids[:2]),
        num_sub_workers=0,
        enable_sdma=sdma,
        pipeline_depth=depth,
        pending_run_depth=depth,
        launch_depth=1,
    )
    chip = worker.register(_build_sdma_callable(st_platform) if sdma else _build_tload_callable(st_platform, _RUNTIME))
    request = dict(
        name="host",
        members=[1, 0],
        window_size=prefix + 2 * nbytes,
        buffers=[
            CommBufferSpec("guard", "uint8", prefix, prefix),
            CommBufferSpec("table", "float32", count, nbytes),
            CommBufferSpec("out", "float32", count, nbytes),
        ],
        memory_source=AddressSpace.HOST,
        workspace_engines=engines,
        retain_after_run=True,
    )
    state = {}
    try:
        worker.init()
        scratch = worker.create_buffer(nbytes)

        def prepare(orch, _args, _cfg):
            state["host"] = orch.get_or_allocate_domain(**request)
            state["device"] = orch.get_or_allocate_domain(
                name="device",
                members=[1, 0],
                window_size=2 * depth * nbytes,
                buffers=[CommBufferSpec(name, "float32", depth * count, depth * nbytes) for name in ("out", "result")],
                retain_after_run=True,
            )
            if sdma:
                for rank in range(2):
                    for name in ("out", "result"):
                        orch.copy_to_domain(state["device"], rank, bytes(depth * nbytes), buffer=name)

        worker.run(prepare)
        host, device = state["host"], state["device"]
        windows = [host.member(rank).buffers["table"].shm for rank in range(2)]
        with pytest.raises(RuntimeError, match="belongs to the domain"):
            windows[0].unlink()
        with pytest.raises(ValueError, match="Host domain windows use CPU access"):
            worker.copy_to(host.member(0).buffers["table"], scratch)
        with pytest.raises(ValueError, match="Host domain windows use CPU access"):
            worker.copy_from(scratch, host.member(0).buffers["table"])

        def run_gather(orch, _args, cfg, *, slot=0):
            assert orch.get_or_allocate_domain(**request) is host
            byte_offset = slot * nbytes
            for rank in range(2 if sdma else 1):
                member = host.member(rank)
                outputs = device.member(rank).buffers
                args = TaskArgs()
                args.add_tensor(member.buffers["table"].tensor((count,), DataType.FLOAT32), TensorArgType.INPUT)
                out = outputs["out"] if sdma else member.buffers["out"]
                args.add_tensor(
                    out.tensor((count,), DataType.FLOAT32, byte_offset=byte_offset), TensorArgType.OUTPUT_EXISTING
                )
                if sdma:
                    args.add_tensor(
                        outputs["result"].tensor((count,), DataType.FLOAT32, byte_offset=byte_offset),
                        TensorArgType.OUTPUT_EXISTING,
                    )
                else:
                    args.add_scalar(1)
                args.add_scalar(host.device_context(member.local_worker_id, workspace_engines=engines))
                orch.submit_next_level(chip, args, cfg, worker=member.local_worker_id)

        offset, _ = host.buffer_range("out")
        for value in (1.0, 5.0):
            for rank, data in enumerate((value, value + 1.0)):
                with windows[rank].buf[prefix : prefix + nbytes] as table:
                    table[:] = struct.pack(f"<{count}f", *([data] * count))
            pending = worker.submit(run_gather)
            following = worker.submit(partial(run_gather, slot=1)) if depth == 2 else None
            pending.wait()
            if following is not None:
                following.wait()
            assert all(bytes(window.buf[:prefix]) == bytes(prefix) for window in windows)
            if sdma:
                for rank, data in enumerate((value + 1.0, value)):
                    for name, expected in (("out", data), ("result", data + 1.0)):
                        for slot in range(depth):
                            worker.copy_from(scratch, device.member(rank).buffers[name], src_offset=slot * nbytes)
                            assert struct.unpack_from(f"<{count}f", scratch.shm.buf) == (expected,) * count, (
                                rank,
                                name,
                                slot,
                            )
            else:
                assert struct.unpack_from(f"<{count}f", windows[0].buf, offset) == (value,) * count
        with windows[0].buf[prefix : prefix + nbytes] as exported:
            with pytest.raises(BufferError, match="live CPU exports"):
                host.release()
            assert not host.released
            assert struct.unpack_from(f"<{count}f", exported) == (5.0,) * count
        host.release()
        assert host.freed and all(window.buf is None for window in windows)
        if sdma:
            worker.copy_from(scratch, device.member(1).buffers["out"])
            assert struct.unpack_from(f"<{count}f", scratch.shm.buf) == (5.0,) * count
        device.release()
    except BaseException:
        try:
            worker.close()
        except BaseException as exc:
            print(f"Worker.close after run failure: {exc}", file=sys.stderr)
        if sdma:
            try:
                _print_sdma_diagnostics(log_dir)
            except Exception as exc:
                print(f"Cannot collect SDMA diagnostics: {exc}", file=sys.stderr)
        raise
    else:
        worker.close()
