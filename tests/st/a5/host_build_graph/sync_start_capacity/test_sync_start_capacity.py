# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public sync_start normalization and capacity refusal through the real Host submit path."""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from simpler.task_interface import CallConfig, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.tools.swimlane_converter import read_perf_data

pytestmark = [
    pytest.mark.platforms(["a5sim", "a5"]),
    pytest.mark.device_count(1),
    pytest.mark.runtime("host_build_graph"),
    pytest.mark.scheduler_mode("resident"),
]

# Each mask holds a device for a process of its own. A simulator run carries
# the single-AIV-lane mask, which the capacity accounting special-cases, and
# all three lanes together; the silicon keeps the full sweep.
_SIM_MASKS = (4, 7)


def _mask_params():
    return [
        pytest.param(mask, marks=[] if mask in _SIM_MASKS else [pytest.mark.platforms(["a5"])])
        for mask in (1, 2, 4, 3, 5, 6, 7)
    ]


@pytest.fixture(scope="module")
def sync_probe():
    source = Path(__file__).resolve().parents[4] / "fdwic/pa_hbg/spmd/test_spmd_contract.py"
    spec = importlib.util.spec_from_file_location("_sync_capacity_probe", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_args(worker, buffers, values, blocks, mask, sync_mode):
    args = TaskArgs()
    for buffer, value in zip(buffers, values, strict=True):
        value.zero_()
        worker.copy_to(buffer, value)
        args.add_tensor(buffer.tensor(shapes=tuple(value.shape), dtype=DataType.INT64), TensorArgType.INOUT)
    for scalar in (blocks, mask, 2, 0, sync_mode):
        args.add_scalar(scalar)
    return args


def _close_worker(worker, buffers):
    for buffer in buffers:
        try:
            worker.free(buffer)
        except RuntimeError:
            # A failed run may retain its memory until close().
            pass
    worker.close()


def _check_normalized_capture(path, mask):
    parsed = read_perf_data(str(path))
    assert parsed["protocol_audit"]["protocol_verified"], parsed["protocol_audit"]
    raw = json.loads(path.read_text())
    protocol = raw["aicore_protocol"]
    assert protocol["producer"] == "aicore"
    tasks = [
        task
        for task in protocol["tasks"]
        if any(kernel in (0, 1, 2) for kernel in task["kernel_ids"]) and task["active_mask"]
    ]
    assert len(tasks) == 2
    normalized_mask = 2 if mask == 4 else mask
    mixed = mask.bit_count() > 1
    kernel_ids = [slot if mask & (1 << slot) else 65535 for slot in range(3)]
    if mask == 4:
        kernel_ids[1:3] = [2, 65535]
    for task in tasks:
        assert task["block_num"] == 1
        assert task["active_mask"] == normalized_mask
        assert task["kernel_ids"] == kernel_ids
        assert task["flags"] & (8 | 16) == 0, "N=1 must not carry SPMD or SYNC_START metadata"
        assert bool(task["flags"] & 4) == mixed
    refs = {task["task_ref"] for task in tasks}
    executions = [entry for entry in protocol["executions"] if entry["task_ref"] in refs]
    expected = {(ref, 0, slot) for ref in refs for slot in range(3) if normalized_mask & (1 << slot)}
    actual = [(entry["task_ref"], entry["block_idx"], entry["subtask_slot"]) for entry in executions]
    assert len(actual) == len(expected) and set(actual) == expected
    assert all(entry["block_num"] == 1 for entry in executions)
    assert len(raw["aicore_tasks"]) == len(expected) + 1  # The final kernel checks real fanin.
    assert protocol["cooperative_task_count"] == (2 if mixed else 0)
    assert protocol["mix_task_count"] == (2 if mixed else 0)
    groups = [block for block in protocol["blocks"] if block["owner"]["task_ref"] in refs]
    if mixed:
        assert len(groups) == 2
        assert {group["owner"]["task_ref"] for group in groups} == refs
        for group in groups:
            assert len(group["lanes"]) == 3
            assert {lane["lane"] for lane in group["lanes"]} == {0, 1, 2}
            assert all(lane["resource_mask"] == 7 for lane in group["lanes"])
            assert all(lane["active_mask"] == mask for lane in group["lanes"])
    else:
        assert not groups


@pytest.mark.parametrize("mask", _mask_params())
def test_single_block_sync_request_preserves_execution(st_platform, st_device_ids, mask, sync_probe):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=int(st_device_ids[0]))
    buffers = []
    probe = sync_probe.TestHbgSpmdContract()
    output_root = Path(__file__).resolve().parents[5] / "outputs"
    output_root.mkdir(exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix=f"sync-start-n1-{st_platform}-mask{mask}-", dir=output_root))
    try:
        handle = worker.register(probe.compile_chip_callable(st_platform))
        worker.init()
        values = [torch.zeros(elements, dtype=torch.int64) for elements in (2 * 4 * 16, 16)]
        for value in values:
            buffers.append(worker.malloc(value.nbytes))
        for iteration, sync_mode in enumerate((1, 0, 1)):
            params = {"blocks": 1, "mask": mask, "tasks": 2, "tail_delay": 0, "sync_mode": sync_mode}
            expected = SimpleNamespace(records=torch.zeros_like(values[0]), summary=torch.zeros_like(values[1]))
            probe.compute_golden(expected, params)
            args = _make_args(worker, buffers, values, 1, mask, sync_mode)
            prefix = artifacts / f"round{iteration}_sync{sync_mode}"
            prefix.mkdir()
            config = CallConfig()
            config.enable_chip_swimlane = 4
            config.output_prefix = str(prefix)
            before = worker.scheduler_mode_publish_count
            worker.run(handle, args, config)
            sync_probe.require_fresh_resident(worker, before)
            for buffer, value in zip(buffers, values, strict=True):
                worker.copy_from(value, buffer)
            torch.testing.assert_close(values[0], expected.records, rtol=0, atol=0)
            torch.testing.assert_close(values[1], expected.summary, rtol=0, atol=0)
            captures = list(prefix.rglob("chip_swimlane_records.json"))
            assert len(captures) == 1, captures
            _check_normalized_capture(captures[0], mask)
    finally:
        _close_worker(worker, buffers)


@pytest.mark.parametrize("mask", _mask_params())
def test_sync_start_excess_capacity_is_refused(st_platform, st_device_ids, mask, sync_probe, capfd, drain_host_log):
    # The oversized shapes exceed every supported A5 topology, including 36 C / 72 V.
    blocks = 73 if mask in (2, 4) else 37
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=int(st_device_ids[0]))
    buffers = []
    try:
        handle = worker.register(sync_probe.TestHbgSpmdContract.compile_chip_callable(st_platform))
        worker.init()
        values = [torch.zeros(elements, dtype=torch.int64) for elements in (2 * (blocks * 3 + 1) * 16, 16)]
        for value in values:
            buffers.append(worker.malloc(value.nbytes))
        args = _make_args(worker, buffers, values, blocks, mask, 1)
        with pytest.raises(RuntimeError, match=r"prepare_native_run failed with code -7\b"):
            worker.run(handle, args, CallConfig())
        log = drain_host_log(capfd)
        assert "REQUIRE_SYNC_START_INVALID" in log, log
    finally:
        _close_worker(worker, buffers)
