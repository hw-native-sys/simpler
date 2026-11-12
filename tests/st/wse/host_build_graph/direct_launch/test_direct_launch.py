# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real PE coverage and mixed matrix/vector computation through the public Worker."""

import json
from functools import lru_cache
from pathlib import Path

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, ChipCallable, CoreCallable, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.kernel_compiler import KernelCompiler
from tests.st.wse.host_build_graph.fixed_sram_layout import copy_from, copy_to, layout_view, release_view

pytestmark = [pytest.mark.platforms(["wsesim"]), pytest.mark.runtime("host_build_graph"), pytest.mark.device_count(1)]
HERE = Path(__file__).parent


@lru_cache
def build_mixed(platform):
    compiler = KernelCompiler(platform)
    kernel = compiler.compile_incore(
        str(HERE / "kernels/mixed.cpp"),
        core_type="",
        extra_include_dirs=compiler.get_orchestration_include_dirs("host_build_graph"),
    )
    return ChipCallable.build(
        signature=[D.IN, D.IN, D.OUT, D.OUT, D.IN],
        func_name="aicpu_orchestration_entry",
        binary=compiler.compile_orchestration("host_build_graph", str(HERE / "kernels/mixed_orch.cpp")),
        children=[(0, CoreCallable.build(signature=[D.IN, D.IN, D.OUT, D.OUT], binary=kernel))],
    )


def audit_events(path, cores, blocks):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    header, events = rows[0], sorted(rows[1:], key=lambda event: event["seq"])
    assert header["core_count"] == cores and header["expected_tasks"] == 1
    assert len({event["seq"] for event in events}) == len(events)
    by_kind = {kind: [event for event in events if event["kind"] == kind] for kind in range(11)}
    assert len(by_kind[1]) == cores
    assert {(e["core"], e["pe"]) for e in by_kind[2]} == {(c, p) for c in range(cores) for p in range(4)}
    assert max(e["seq"] for e in by_kind[2]) < by_kind[3][0]["seq"]
    assert by_kind[3][0]["seq"] < min(e["seq"] for e in by_kind[4])
    assert sorted(e["block"] for e in by_kind[4]) == list(range(blocks))
    assert sorted(e["block"] for e in by_kind[5]) == list(range(blocks))
    assert len(by_kind[6]) == 1 and by_kind[6][0]["value"] == 1
    assert max(e["seq"] for e in by_kind[5]) < by_kind[6][0]["seq"]
    assert by_kind[6][0]["seq"] < by_kind[10][0]["seq"]
    assert by_kind[10][0]["seq"] < min(e["seq"] for e in by_kind[7])
    assert len(by_kind[7]) == 4 * cores and len(by_kind[8]) == cores
    assert max(e["seq"] for e in by_kind[8]) < by_kind[9][0]["seq"]
    if blocks >= cores:
        assert {(e["core"], e["pe"]) for e in by_kind[4]} == {(c, 0) for c in range(cores)}


@pytest.mark.parametrize("blocks", [1, 4, 9, 39, 40, 159, 160, 161, 323])
def test_mixed_entry_and_independent_blocks(tmp_path, blocks, st_platform):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    buffers = []
    try:
        worker.init()
        handle = worker.register(build_mixed(st_platform))
        a = (torch.arange(1024, dtype=torch.float32).reshape(32, 32) % 7) - 3
        b = torch.eye(32, dtype=torch.float32) * 2
        tensors = [a, b, torch.full((blocks, 32, 32), -1.0), torch.full((blocks, 2), -1.0)]
        args = TaskArgs()
        for index, tensor in enumerate(tensors):
            buffer = layout_view(worker, tensor.nbytes, core=len(buffers) % 40)
            buffers.append(buffer)
            copy_to(worker, buffer, tensor)
            args.add_tensor(
                buffer.tensor(shapes=tuple(tensor.shape), dtype=DataType.FLOAT32),
                TensorArgType.INPUT if index < 2 else TensorArgType.OUTPUT_EXISTING,
            )
        args.add_scalar(blocks)
        config = CallConfig()
        config.output_prefix = str(tmp_path)
        worker.run(handle, args, config)
        for index in (2, 3):
            copy_from(worker, tensors[index], buffers[index])
        expected = torch.stack([a @ b + float(block + 1) for block in range(blocks)])
        torch.testing.assert_close(tensors[2], expected, rtol=0, atol=0)
        actual_cores = 40
        identities = tensors[3].to(torch.int64)
        assert torch.all((identities[:, 0] >= 0) & (identities[:, 0] < actual_cores))
        assert torch.all((identities[:, 1] >= 0) & (identities[:, 1] < 4))
        assert identities.tolist() == [[b % actual_cores, 0] for b in range(blocks)]
        audit_events(tmp_path / "wse_events.jsonl", actual_cores, blocks)
        for buffer in buffers:
            release_view(worker, buffer)
        buffers.clear()
        worker.unregister(handle)
    finally:
        worker.close()


@lru_cache
def build_graph(platform):
    compiler = KernelCompiler(platform)
    kernel = compiler.compile_incore(
        str(HERE / "kernels/stamp.cpp"),
        core_type="",
        extra_include_dirs=compiler.get_orchestration_include_dirs("host_build_graph"),
    )
    return ChipCallable.build(
        signature=[D.INOUT] * 3,
        func_name="aicpu_orchestration_entry",
        binary=compiler.compile_orchestration("host_build_graph", str(HERE / "kernels/graph_orch.cpp")),
        children=[(0, CoreCallable.build(signature=[D.INOUT], binary=kernel))],
    )


@pytest.mark.parametrize("cores", [40])
def test_graph_record_replay_and_rebinding(tmp_path, cores, st_platform):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    try:
        worker.init()
        handle = worker.register(build_graph(st_platform))
        epochs = []
        for repeat in range(2):
            tensors = [torch.full((cores * 3 * 16,), float(-7 - repeat)) for _ in range(3)]
            buffers = []
            args = TaskArgs()
            for tensor in tensors:
                buffer = layout_view(worker, tensor.nbytes, core=len(buffers) % 40)
                buffers.append(buffer)
                copy_to(worker, buffer, tensor)
                args.add_tensor(buffer.tensor(shapes=tuple(tensor.shape), dtype=DataType.FLOAT32), TensorArgType.INOUT)
            config = CallConfig()
            config.output_prefix = str(tmp_path / str(repeat))
            config.enable_dep_gen = True
            worker.run(handle, args, config)
            expected = torch.full_like(tensors[0], float(-7 - repeat))
            for block in range(cores):
                for region in range(3):
                    expected[(block * 3 + region) * 16] = block
            for tensor, buffer in zip(tensors, buffers, strict=True):
                copy_from(worker, tensor, buffer)
                torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
                release_view(worker, buffer)
            rows = [json.loads(line) for line in (tmp_path / str(repeat) / "wse_events.jsonl").read_text().splitlines()]
            epochs.append(rows[0]["epoch"])
            assert rows[0]["expected_tasks"] == 9  # Three enter/kernel/exit instances.
            completed = [row for row in rows[1:] if row["kind"] == 6]
            assert len(completed) == 9 and len({row["task"] for row in completed}) == 9
            assert sorted(row["value"] for row in completed) == list(range(1, 10))
            assert (tmp_path / str(repeat) / "deps.json").is_file()
            done = {row["task"]: row["seq"] for row in completed}
            for shell in range(3):
                enter, child, exit_node = shell * 3, shell * 3 + 1, shell * 3 + 2
                child_begins = [row["seq"] for row in rows[1:] if row["kind"] == 4 and row["task"] == child]
                assert done[enter] < min(child_begins)
                assert done[child] < done[exit_node]
        assert epochs[0] != epochs[1]
        worker.unregister(handle)
    finally:
        worker.close()


def test_empty_graph_retires_all_members(tmp_path, st_platform):
    compiler = KernelCompiler(st_platform)
    chip = ChipCallable.build(
        signature=[],
        func_name="aicpu_orchestration_entry",
        binary=compiler.compile_orchestration("host_build_graph", str(HERE / "kernels/empty_orch.cpp")),
        children=[],
    )
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    try:
        worker.init()
        handle = worker.register(chip)
        config = CallConfig()
        config.output_prefix = str(tmp_path)
        worker.run(handle, TaskArgs(), config)
        rows = [json.loads(line) for line in (tmp_path / "wse_events.jsonl").read_text().splitlines()]
        assert rows[0]["expected_tasks"] == 0
        assert len([row for row in rows[1:] if row["kind"] == 7]) == 160
        assert not [row for row in rows[1:] if row["kind"] in (4, 5, 6)]
        assert len([row for row in rows[1:] if row["kind"] == 9]) == 1
        worker.unregister(handle)
    finally:
        worker.close()


def test_fixed_device_width_survives_reverse_wait(tmp_path, st_platform):
    compiler = KernelCompiler(st_platform)
    chip = ChipCallable.build(
        signature=[],
        func_name="aicpu_orchestration_entry",
        binary=compiler.compile_orchestration("host_build_graph", str(HERE / "kernels/empty_orch.cpp")),
        children=[],
    )
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    try:
        worker.init()
        handle = worker.register(chip)
        pending = []
        for index in range(4):
            config = CallConfig()
            config.output_prefix = str(tmp_path / str(index))
            pending.append((worker.submit(handle, TaskArgs(), config), index, 40))
        for run, index, cores in reversed(pending):
            run.wait()
            rows = [json.loads(line) for line in (tmp_path / str(index) / "wse_events.jsonl").read_text().splitlines()]
            assert rows[0]["core_count"] == cores
            assert {row["core"] for row in rows[1:] if row["kind"] == 8} == set(range(cores))
            assert sum(row["kind"] == 7 for row in rows[1:]) == cores * 4
        worker.run(handle, TaskArgs())
        worker.unregister(handle)
    finally:
        worker.close()
