# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A3 retained Host windows with local MTE and cross-rank SDMA."""

import json
import re
import struct
import subprocess
import sys
from collections import deque
from functools import partial
from pathlib import Path

import pytest
from _task_interface import _host_log_directory  # pyright: ignore[reportMissingImports]
from simpler.task_interface import AddressSpace, ArgDirection, CommBufferSpec, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from examples.workers.l4.global_tload_mixed_l3.main import COUNT, _build_tload_callable
from simpler_setup.scene_test import compile_chip_callable_spec

_RUNTIME = "tensormap_and_ringbuffer"
_SDMA_DIAGNOSTIC = re.compile(
    r"TIMEOUT_EXIT|\[ASYNC_WAIT|SdmaErrorInfo|SMMU|CQE|(?:sched|orch)_error_code=", re.IGNORECASE
)
_HOST_MAPPING_PROBE = """
import ctypes
import json
import sys

acl = ctypes.CDLL("libascendcl.so")
acl.aclInit.argtypes = [ctypes.c_char_p]
acl.aclrtSetDevice.argtypes = [ctypes.c_int32]
acl.aclrtHostMemMapCapabilities.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
assert acl.aclInit(None) == 0
capabilities = []
try:
    for device in map(int, sys.argv[1:]):
        assert acl.aclrtSetDevice(device) == 0
        capability = ctypes.c_int(-1)
        # ACL_RT_HAC_TYPE_SDMA = 6; capability is NOT_SUPPORTED=0 or SUPPORTED=1.
        assert acl.aclrtHostMemMapCapabilities(device, 6, ctypes.byref(capability)) == 0
        assert capability.value in (0, 1)
        capabilities.append(capability.value)
        assert acl.aclrtResetDevice(device) == 0
finally:
    assert acl.aclFinalize() == 0
print(json.dumps(capabilities))
"""


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


def _assert_host_sdma_rejected(worker, request, device_ids, capabilities):
    directory = _host_log_directory()
    assert directory
    paths = [Path(directory) / f"host.{pid}.log" for pid in worker._chip_pids]
    offsets = [path.stat().st_size if path.exists() else 0 for path in paths]

    def prepare(orch, _args, _cfg):
        orch.get_or_allocate_domain(**request)

    with pytest.raises(RuntimeError, match=r"\(code -7\)") as failure:
        worker.run(prepare)
    assert "failed on 2/2 chips" in str(failure.value)
    assert not worker.live_domains
    worker.close()
    assert worker._cleanup_journal.empty
    assert len(device_ids) == len(capabilities) == len(paths) == len(offsets)
    for device, capability, path, offset in zip(device_ids, capabilities, paths, offsets):
        text = path.read_bytes()[offset:].decode()
        if capability == 0:
            assert f"Host domain SDMA unsupported: device={device} query_status=0 capability=0" in text
        assert "timed out waiting for window announcement" not in text


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
        pytest.param(True, 2, id="sdma-depth2", marks=pytest.mark.sdma),
    ],
)
def test_a3_host_source_domain(  # noqa: PLR0912 -- covers supported and rejected domain lifecycles
    st_platform, st_device_ids, sdma, depth, tmp_path, monkeypatch, drain_host_log
):
    if sdma:
        log_dir = tmp_path / "ascend"
        log_dir.mkdir()
        monkeypatch.setenv("ASCEND_PROCESS_LOG_PATH", str(log_dir))
        # Keep ACL initialization out of the parent process that forks L2 workers.
        probe = subprocess.run(
            [sys.executable, "-c", _HOST_MAPPING_PROBE, *map(str, st_device_ids[:2])],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert probe.returncode == 0, probe.stdout + probe.stderr
        capabilities = json.loads(probe.stdout.splitlines()[-1])
        assert len(capabilities) == 2 and all(capability in (0, 1) for capability in capabilities)
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
    initializer_calls = 0
    try:
        worker.init()
        if sdma and 0 in capabilities:
            _assert_host_sdma_rejected(worker, request, st_device_ids[:2], capabilities)
            return
        scratch = worker.create_buffer(nbytes) if sdma else None

        def fill_tables(domain, value):
            offset, size = domain.buffer_range("table")
            for rank, data in enumerate((value, value + 1.0)):
                window = domain.member(rank).buffers["table"].shm
                with window.buf[offset : offset + size] as table:
                    table[:] = struct.pack(f"<{count}f", *([data] * count))

        def initialize_host(domain):
            nonlocal initializer_calls
            initializer_calls += 1
            fill_tables(domain, 1.0)

        def run_gather(orch, _args, cfg, *, slot=0):
            host = orch.get_or_allocate_domain(**request, initializer=initialize_host)
            assert state.setdefault("host", host) is host
            device = state.get("device")
            if sdma and device is None:
                device = orch.get_or_allocate_domain(
                    name="device",
                    members=[1, 0],
                    window_size=2 * depth * nbytes,
                    buffers=[
                        CommBufferSpec(name, "float32", depth * count, depth * nbytes) for name in ("out", "result")
                    ],
                    retain_after_run=True,
                )
                for rank in range(2):
                    for name in ("out", "result"):
                        orch.copy_to_domain(device, rank, bytes(depth * nbytes), buffer=name)
                state["device"] = device

            byte_offset = slot * nbytes
            for rank in range(2 if sdma else 1):
                member = host.member(rank)
                outputs = device.member(rank).buffers if device is not None else member.buffers
                args = TaskArgs()
                args.add_tensor(member.buffers["table"].tensor((count,), DataType.FLOAT32), TensorArgType.INPUT)
                args.add_tensor(
                    outputs["out"].tensor((count,), DataType.FLOAT32, byte_offset=byte_offset),
                    TensorArgType.OUTPUT_EXISTING,
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

        for value in (1.0, 5.0):
            if "host" in state:
                fill_tables(state["host"], value)
            pending = worker.submit(run_gather)
            following = worker.submit(partial(run_gather, slot=1)) if depth == 2 else None
            pending.wait()
            if following is not None:
                following.wait()
            assert initializer_calls == 1
            host, device = state["host"], state.get("device")
            windows = [host.member(rank).buffers["table"].shm for rank in range(2)]
            assert all(bytes(window.buf[:prefix]) == bytes(prefix) for window in windows)
            if device is not None:
                assert scratch is not None
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
                offset, _ = host.buffer_range("out")
                assert struct.unpack_from(f"<{count}f", windows[0].buf, offset) == (value,) * count
        with pytest.raises(RuntimeError, match="belongs to L2"):
            windows[0].unlink()
        with windows[0].buf[prefix : prefix + nbytes] as exported:
            with pytest.raises(BufferError, match="live CPU exports"):
                host.release()
            assert not host.released
            assert struct.unpack_from(f"<{count}f", exported) == (5.0,) * count
        host.release()
        assert host.freed and all(window.buf is None for window in windows)
        if device is not None:
            assert scratch is not None
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
