#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Graph Calendar PGO cache reuse through independent level-2 runs."""

from __future__ import annotations

import ctypes
import logging
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import DataType, TaskArgs, TensorArgType, _flush_host_log
from simpler.worker import Worker

from simpler_setup.scene_test import compile_chip_callable_spec

pytestmark = [
    pytest.mark.platforms(["a5", "a5sim"]),
    pytest.mark.runtime("host_build_graph"),
    pytest.mark.device_count(1),
]

_EVENT = re.compile(r"A5 calendar: (MISS|REUSE|HIT|STORE) [^\n]*kind=(graph_bucket|graph_args) key=([0-9a-f]{16})\b")
_CORE_STRIDE = 16
_OUT_ELEMENTS = 32 * _CORE_STRIDE
_PRIMARY_GRAPH = 1
_OTHER_GRAPH = 2
_RETRY_MARKER = "retry graph after failed image publication"


@dataclass(frozen=True)
class _Invocation:
    mark: int
    bucket: int | None = 0
    graph: int = _PRIMARY_GRAPH


def _build_callable(platform, entry="aicpu_orchestration_entry"):
    test_dir = Path(__file__).resolve().parent
    hbg = test_dir.parents[1] / "host_build_graph"
    spec = {
        "orchestration": {
            "source": str(test_dir / "kernels/orchestration/graph_calendar_cache.cpp"),
            "function_name": entry,
            "signature": [D.INOUT, D.INOUT, *([D.IN] * 7)],
        },
        "incores": [
            {
                "func_id": 0,
                "source": str(hbg / "spmd_multiblock_aic/kernels/aic/kernel_stamp_block.cpp"),
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 1,
                "source": str(hbg / "spmd_mixed_waves/kernels/aiv/kernel_mark_aiv.cpp"),
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
        ],
    }
    return compile_chip_callable_spec(spec, platform, "host_build_graph", (__file__, platform, entry))


def _expected_output(blocks, invocation):
    expected = torch.full((_OUT_ELEMENTS,), -1.0, dtype=torch.float32)
    if invocation is not None:
        expected[: blocks * _CORE_STRIDE : _CORE_STRIDE] = torch.arange(blocks, dtype=expected.dtype)
        expected[1 : blocks * _CORE_STRIDE : _CORE_STRIDE] = blocks
        expected[(4 * invocation.mark + 1) * _CORE_STRIDE] = invocation.mark
    return expected


def _run(worker, handle, buffers, invocation, *, blocks=2, prefix=None, shape_blocks=False):
    sentinel = torch.full((_OUT_ELEMENTS,), -1.0, dtype=torch.float32)
    args = TaskArgs()
    shape = (blocks * 64,) if shape_blocks else sentinel.shape
    for buffer in buffers:
        worker.copy_to(buffer, sentinel)
        args.add_tensor(buffer.tensor(shape, DataType.FLOAT32), TensorArgType.INOUT)
    args.add_scalar(0 if shape_blocks else blocks)
    for item in (invocation, prefix):
        args.add_scalar(item.graph if item else 0)
        args.add_scalar(item.mark if item else 0)
        args.add_scalar(item.bucket if item and item.bucket is not None else -1)
    worker.run(handle, args)
    assert worker.scheduler_runtime_mode == 5
    for buffer, item in zip(buffers, (invocation, prefix)):
        result = torch.empty_like(sentinel)
        worker.copy_from(result, buffer)
        torch.testing.assert_close(result, _expected_output(blocks, item), atol=0, rtol=0)


def _expect_events(log, kind, *actions):
    observed = _EVENT.findall(log)
    assert [(action, observed_kind) for action, observed_kind, _ in observed] == [
        (action, kind) for action in actions
    ], log
    return tuple(key for _, _, key in observed)


@contextmanager
def _calendar_worker(platform, device_id):
    worker = Worker(level=2, platform=platform, runtime="host_build_graph", device_id=device_id)
    buffers = []
    try:
        worker.init()
        for _ in range(3):
            buffers.append(worker.malloc(_OUT_ELEMENTS * 4))
        yield worker, buffers
    finally:
        try:
            for buffer in buffers:
                worker.free(buffer)
        finally:
            worker.close()


@pytest.fixture
def calendar_worker(st_platform, st_device_ids, caplog):
    caplog.set_level(logging.INFO, logger="simpler")
    with _calendar_worker(st_platform, st_device_ids[0]) as resources:
        yield resources


def test_graph_calendar_reuses_plan_across_runs(calendar_worker, st_platform, capfd, drain_host_log):
    worker, (out_a, out_b, prefix_out) = calendar_worker
    target = _build_callable(st_platform)
    handle = worker.register(target)
    alternate = worker.register(_build_callable(st_platform, "aicpu_orchestration_entry_alternate"))
    assert handle.digest != alternate.digest

    _run(worker, handle, (out_a, prefix_out), _Invocation(mark=1))
    first, stored = _expect_events(drain_host_log(capfd), "graph_bucket", "MISS", "STORE")
    assert first == stored

    _run(worker, handle, (out_b, prefix_out), _Invocation(mark=2))
    assert _expect_events(drain_host_log(capfd), "graph_bucket", "HIT") == (stored,)

    _run(worker, handle, (out_b, prefix_out), _Invocation(mark=3, bucket=77))
    other_bucket, stored_bucket = _expect_events(drain_host_log(capfd), "graph_bucket", "MISS", "STORE")
    assert other_bucket == stored_bucket != stored

    _run(worker, handle, (out_b, prefix_out), _Invocation(mark=3), blocks=4)
    other_shape, stored_shape = _expect_events(drain_host_log(capfd), "graph_bucket", "MISS", "STORE")
    assert other_shape == stored_shape != stored

    _run(worker, handle, (out_a, prefix_out), _Invocation(mark=4), prefix=_Invocation(mark=5, graph=_OTHER_GRAPH))
    other_graph, hit, stored_graph = _expect_events(drain_host_log(capfd), "graph_bucket", "MISS", "HIT", "STORE")
    assert other_graph == stored_graph != stored
    assert hit == stored

    _run(worker, handle, (out_b, prefix_out), _Invocation(mark=6), blocks=4)
    assert _expect_events(drain_host_log(capfd), "graph_bucket", "HIT") == (stored_shape,)

    _run(worker, alternate, (out_a, prefix_out), _Invocation(mark=4))
    miss, store = _expect_events(drain_host_log(capfd), "graph_bucket", "MISS", "STORE")
    assert miss == store

    worker.unregister(handle)
    handle = worker.register(target)
    _run(worker, handle, (out_a, prefix_out), _Invocation(mark=4))
    assert _expect_events(drain_host_log(capfd), "graph_bucket", "MISS", "STORE") == (stored, stored)


def test_graph_calendar_keeps_exact_argument_lookup(calendar_worker, st_platform, capfd, drain_host_log):
    worker, (out_a, out_b, prefix_out) = calendar_worker
    handle = worker.register(_build_callable(st_platform))

    _run(worker, handle, (out_a, prefix_out), _Invocation(mark=1, bucket=None))
    first, stored = _expect_events(drain_host_log(capfd), "graph_args", "MISS", "STORE")
    assert first == stored

    _run(worker, handle, (out_b, prefix_out), _Invocation(mark=2, bucket=None))
    second, stored_second = _expect_events(drain_host_log(capfd), "graph_args", "MISS", "STORE")
    assert second == stored_second != stored

    _run(worker, handle, (out_b, prefix_out), _Invocation(mark=1))
    _expect_events(drain_host_log(capfd), "graph_bucket", "MISS", "STORE")

    _run(worker, handle, (out_b, prefix_out), _Invocation(mark=1, bucket=None))
    assert _expect_events(drain_host_log(capfd), "graph_args", "HIT") == (stored,)


def test_graph_calendar_rejects_structure_change_for_equal_config(calendar_worker, st_platform, capfd, drain_host_log):
    worker, (out_a, out_b, prefix_out) = calendar_worker
    handle = worker.register(_build_callable(st_platform))
    _run(worker, handle, (out_a, prefix_out), _Invocation(mark=1), blocks=4, shape_blocks=True)
    _, four = _expect_events(drain_host_log(capfd), "graph_bucket", "MISS", "STORE")

    for bucket in (0, 77, None):
        prefix = _Invocation(mark=1, graph=_OTHER_GRAPH) if bucket == 77 else None
        with pytest.raises(RuntimeError, match=r"prepare_native_run failed"):
            _run(
                worker,
                handle,
                (out_b, prefix_out),
                _Invocation(mark=1, bucket=bucket),
                blocks=8,
                prefix=prefix,
                shape_blocks=True,
            )
        log = drain_host_log(capfd)
        assert re.search(r"Graph .*task=0 field=block_num expected=4 actual=8", log), log
        assert not _EVENT.search(log), log

    _run(worker, handle, (out_b, prefix_out), _Invocation(mark=1), blocks=8)
    miss, eight = _expect_events(drain_host_log(capfd), "graph_bucket", "MISS", "STORE")
    assert miss == eight != four
    _run(worker, handle, (out_b, prefix_out), _Invocation(mark=2), blocks=8)
    assert _expect_events(drain_host_log(capfd), "graph_bucket", "HIT") == (eight,)
    _run(worker, handle, (out_a, prefix_out), _Invocation(mark=2), blocks=4, shape_blocks=True)
    assert _expect_events(drain_host_log(capfd), "graph_bucket", "HIT") == (four,)


def test_graph_calendar_samples_one_invocation_per_bucket(calendar_worker, st_platform, capfd, drain_host_log):
    worker, (out_a, out_b, prefix_out) = calendar_worker
    handle = worker.register(_build_callable(st_platform))

    _run(worker, handle, (out_a, prefix_out), _Invocation(mark=1), prefix=_Invocation(mark=2))
    reused, miss, stored = _expect_events(drain_host_log(capfd), "graph_bucket", "REUSE", "MISS", "STORE")
    assert reused == miss == stored

    _run(worker, handle, (out_b, prefix_out), _Invocation(mark=3, bucket=42), prefix=_Invocation(mark=4))
    hit, miss, stored_other = _expect_events(drain_host_log(capfd), "graph_bucket", "HIT", "MISS", "STORE")
    assert hit == stored
    assert miss == stored_other != stored


def _exercise_retry_after_publish_failure(library, device_id):
    fault = ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
    fault.arm_calendar_publish_fault.argtypes = []
    fault.arm_calendar_publish_fault.restype = None
    logging.getLogger("simpler").setLevel(logging.INFO)

    with _calendar_worker("a5sim", device_id) as (worker, (out_a, out_b, prefix_out)):
        handle = worker.register(_build_callable("a5sim"))
        fault.arm_calendar_publish_fault()
        with pytest.raises(RuntimeError, match=r"prepare_native_run failed with code -1000\b"):
            _run(worker, handle, (out_a, prefix_out), _Invocation(mark=1))
        assert _flush_host_log()
        print(_RETRY_MARKER, file=sys.stderr, flush=True)

        _run(worker, handle, (out_a, prefix_out), _Invocation(mark=2))
        _run(worker, handle, (out_b, prefix_out), _Invocation(mark=3))
        assert _flush_host_log()


@pytest.mark.platforms(["a5sim"])
@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux ELF symbol interposition")
def test_graph_calendar_retries_after_failed_bind(st_device_ids, tmp_path):
    test_dir = Path(__file__).resolve().parent
    common_include = test_dir.parents[4] / "src" / "common"
    source = test_dir / "support" / "fail_publish_image.cpp"
    library = tmp_path / "fail_publish_image.so"
    subprocess.run(
        ["c++", "-std=c++17", "-shared", "-fPIC", "-I", str(common_include), str(source), "-o", str(library), "-ldl"],
        check=True,
        timeout=60,
    )
    # The interposer must be loaded before the simulator runtime and stay local to this test process.
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(library), str(st_device_ids[0])],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    failed, marker, retried = result.stderr.partition(_RETRY_MARKER)
    assert marker, result.stderr
    assert "publishing this run's image failed" in failed, failed
    (miss,) = _expect_events(failed, "graph_bucket", "MISS")
    assert _expect_events(retried, "graph_bucket", "MISS", "STORE", "HIT") == (miss, miss, miss)


if __name__ == "__main__":
    _exercise_retry_after_publish_failure(sys.argv[1], int(sys.argv[2]))
