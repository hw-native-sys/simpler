# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fixed SRAM ownership, protected runtime windows and cross-Core access."""

import json
from pathlib import Path

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.scene_test import _bound_host_logs
from tests.st.wse.host_build_graph.counter import test_counter as counter
from tests.st.wse.host_build_graph.fixed_sram_layout import copy_from, copy_to, layout_view, release_view
from tests.st.wse.host_build_graph.scene_helpers import build_callable
from tests.st.wse.host_build_graph.test_fixed_sram import fixed_layout

pytestmark = [pytest.mark.platforms(["wsesim"]), pytest.mark.runtime("host_build_graph"), pytest.mark.device_count(1)]


def test_fixed_sram_bounds_and_backing_lifetime(st_platform):
    capacities, sizes, offsets = fixed_layout()
    total = sum(sizes)
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    worker.init()
    try:
        buffer = worker.malloc(total)
        base = buffer.base
        value = torch.tensor([123], dtype=torch.uint8)
        worker.copy_to(buffer, value, dst_offset=total - 1)
        out = torch.zeros_like(value)
        worker.copy_from(out, buffer, src_offset=total - 1)
        assert out.item() == 123
        for offset in (total, total + 1):
            with pytest.raises((ValueError, RuntimeError)):
                worker.copy_to(buffer, value, dst_offset=offset)
            with pytest.raises(RuntimeError):
                worker._chip_worker.copy_to(base + offset, value.data_ptr(), 1)
            with pytest.raises(RuntimeError):
                worker._chip_worker.copy_from(out.data_ptr(), base + offset, 1)
        worker.free(buffer)
        assert worker._chip_worker.committed_device_memory == sum(capacities)
        with pytest.raises((ValueError, RuntimeError)):
            worker.copy_from(out, buffer)
        with pytest.raises(RuntimeError):
            worker._chip_worker.copy_to(base, value.data_ptr(), 1)
        again = worker.malloc(total)
        assert again.base == base and again.identity != buffer.identity
        worker.free(again)
    finally:
        worker.close()


def test_sram_counter_replay_and_residency(tmp_path, st_platform):
    counter.test_counter_graph_rebind_and_reuse(tmp_path, st_platform, 40, True)
    records = list(tmp_path.rglob("wse_sram.json"))
    assert len(records) == 4
    for path in records:
        record = json.loads(path.read_text())
        assert record["configuration"][:2] == [40, 4]
        kinds = {section["kind"] for section in record["sections"]}
        assert {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} <= kinds
        for owner, core in enumerate(record["cores"]):
            sections = sorted((v for v in record["sections"] if v["owner"] == owner), key=lambda v: v["address"])
            end = core["runtime_base"]
            for section in sections:
                assert section["address"] >= end
                end = section["address"] + section["bytes"]
                assert end <= core["runtime_base"] + core["reserve"]
            assert sum(v["bytes"] for v in sections) <= core["runtime_used"] <= core["reserve"]
            assert core["runtime_base"] <= core["control_address"] < core["runtime_base"] + core["reserve"]


def build_memory_probe(platform):
    signature = [D.INOUT] * 3 + [D.IN]
    return build_callable(
        platform,
        Path(__file__).parent / "kernels",
        "orchestration.cpp",
        signature,
        [("probe.cpp", signature)],
    )


def memory_args(worker, blocks, owner, stride=1, *, native=True, skew=False):
    args = TaskArgs()
    buffers = []
    tensors = []
    for index, size in enumerate((blocks * stride + 2, blocks * stride + 2, blocks * 8)):
        host = torch.full((size,), 999, dtype=torch.uint64)
        buffer = layout_view(worker, host.nbytes, core=owner if index == 0 else 0)
        copy_to(worker, buffer, host)
        args.add_tensor(
            buffer.tensor(
                shapes=(blocks,) if index < 2 else (size,),
                dtype=DataType.UINT64,
                strides=(stride,) if index < 2 else (1,),
                byte_offset=8 if index < 2 else 0,
            ),
            TensorArgType.INOUT,
        )
        buffers.append(buffer)
        tensors.append(host)
    args.add_scalar(int(native) | (int(skew) << 1))
    return args, buffers, tensors


@pytest.mark.parametrize("blocks", [5, 40, 160, 163])
def test_sram_remote_owner_and_cross_kernel(tmp_path, st_platform, blocks):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    worker.init()
    try:
        handle = worker.register(build_memory_probe(st_platform))
        stride = 2 if blocks == 160 else 1
        args, windows, tensors = memory_args(worker, blocks, 39, stride)
        worker.run(handle, args, counter.config_for(tmp_path))
        for window, tensor in zip(windows, tensors, strict=True):
            copy_from(worker, tensor, window)
        for index in range(2):
            expected = [999] * (blocks * stride + 2)
            for block in range(blocks):
                expected[1 + block * stride] = 100 + index + block
            assert tensors[index].tolist() == expected
        rows = tensors[2].reshape(blocks, 2, 4).tolist()
        assert rows == [[[39, 0, block % 40, 0]] * 2 for block in range(blocks)]
        for window in windows:
            release_view(worker, window)
        worker.unregister(handle)
    finally:
        worker.close()


def test_sram_runtime_capacity_rejects_before_go_and_recovers(tmp_path, st_platform):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    worker.init()
    try:
        handle = worker.register(build_memory_probe(st_platform))
        args, windows, _ = memory_args(worker, 4096, 1, native=False, skew=True)
        rejected = tmp_path / "rejected"
        with pytest.raises(RuntimeError):
            worker.run(handle, args, counter.config_for(rejected))
        report = rejected / "wse_prepare_error.txt"
        assert report.is_file(), "\n".join(log.read_text() for log in _bound_host_logs())
        assert "WSE Runtime SRAM capacity exhausted" in report.read_text()
        assert not (rejected / "wse_events.jsonl").exists()
        for window in windows:
            release_view(worker, window)
        args, windows, tensors = memory_args(worker, 1, 1)
        worker.run(handle, args, counter.config_for(tmp_path / "valid"))
        copy_from(worker, tensors[1], windows[1])
        assert tensors[1].tolist() == [999, 101, 999]
        for window in windows:
            release_view(worker, window)
    finally:
        worker.close()
