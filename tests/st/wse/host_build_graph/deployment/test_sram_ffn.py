# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Forty AICore Blocks with local SRAM shards and four cooperating Threads."""

import json
from functools import lru_cache
from pathlib import Path

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import ChipCallable, CoreCallable, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.kernel_compiler import KernelCompiler
from tests.st.wse.host_build_graph.counter.test_counter import config_for
from tests.st.wse.host_build_graph.deployment.test_deployment import assert_no_launch
from tests.st.wse.host_build_graph.fixed_sram_layout import copy_from, copy_to, layout_view, release_view

pytestmark = [pytest.mark.platforms(["wsesim"]), pytest.mark.runtime("host_build_graph"), pytest.mark.device_count(1)]


def ffn_inputs(seed):
    row = torch.arange(128).reshape(-1, 1)
    k = torch.arange(32).reshape(1, -1)
    x = ((row + 2 * k + seed) % 5 - 2).float()
    w1 = ((2 * torch.arange(32).reshape(-1, 1) + 3 * torch.arange(128).reshape(1, -1) + seed) % 7 - 3).float()
    w2 = ((3 * row + torch.arange(128).reshape(1, -1) + seed) % 5 - 2).float()
    h = x @ w1
    a = torch.relu(h)
    y = a @ w2

    def shard(t):
        return t.reshape(128, 4, 32).permute(1, 0, 2).contiguous()

    inputs = [
        x,
        w1.reshape(32, 4, 32).permute(1, 2, 0).contiguous(),
        w2.reshape(4, 32, 4, 32).permute(2, 0, 3, 1).contiguous(),
    ]
    return inputs, [shard(t) for t in (h, a, y)]


@lru_cache
def build_sram_ffn(platform, instances):
    compiler = KernelCompiler(platform)
    here = Path(__file__).parent / "kernels"
    children = []
    for stage in range(3):
        signature = [D.IN] * (1 if stage == 1 else 5) + [D.OUT, D.INOUT] + [D.IN] * 2
        binary = compiler.compile_incore(
            str(here / f"ffn_{stage}.cpp"),
            core_type="",
            extra_include_dirs=compiler.get_orchestration_include_dirs("host_build_graph"),
        )
        children.append((stage, CoreCallable.build(signature=signature, binary=binary)))
    return ChipCallable.build(
        signature=[D.INOUT] * (80 * instances) + [D.IN] * 3,
        func_name="aicpu_orchestration_entry",
        binary=compiler.compile_orchestration("host_build_graph", str(here / "ffn_orch.cpp")),
        children=children,
    )


def prepare_ffn(worker, seed, mode, instances=1, *, native=True):
    args, buffers, tensors, goldens = TaskArgs(), [], [], []
    for instance in range(instances):
        grouped = [ffn_inputs(seed + instance * 10 + group) for group in range(10)]
        goldens.append([golden for _, golden in grouped])
        for core in range(40):
            incoming, _ = grouped[core // 4]
            column = core % 4
            main = torch.full((18432,), -12345.0)
            main[:1024] = incoming[0].reshape(4, 32, 32)[column].flatten()
            main[1024:2048] = incoming[1][column].flatten()
            main[2048:6144] = incoming[2][column].flatten()
            diagnostics = torch.full((256,), 999, dtype=torch.uint64)
            for tensor, dtype in ((main, DataType.FLOAT32), (diagnostics, DataType.UINT64)):
                buffer = layout_view(worker, tensor.nbytes, core=core)
                copy_to(worker, buffer, tensor)
                args.add_tensor(buffer.tensor(shapes=tuple(tensor.shape), dtype=dtype), TensorArgType.INOUT)
                buffers.append(buffer)
                tensors.append(tensor)
    args.add_scalar(instances)
    args.add_scalar(mode)
    args.add_scalar(int(native))
    return args, buffers, tensors, goldens


def target(block, mode, thread=0):
    return ((block + 1) % 40 if mode in (1, 2) else block, (thread + int(mode == 2)) % 4)


def check_ffn(worker, path, buffers, tensors, goldens, mode, *, native=True):
    instances = len(goldens)
    for tensor, buffer in zip(tensors, buffers):
        copy_from(worker, tensor, buffer)
    records = []
    for instance, golden in enumerate(goldens):
        base = instance * 80
        for stage, offset in enumerate((6144, 10240, 14336)):
            actual = torch.stack(
                [tensors[base + core * 2][offset : offset + 4096].reshape(128, 32) for core in range(40)]
            ).reshape(10, 4, 128, 32)
            expected = torch.stack([group[stage] for group in golden])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            (path / f"{instance}-{'HAY'[stage]}.bin").write_bytes(actual.numpy().tobytes())
            (path / f"{instance}-{'HAY'[stage]}.golden.bin").write_bytes(expected.numpy().tobytes())
        for block in range(40):
            owner_base = base + block * 2
            logs = tensors[owner_base + 1][:240].reshape(3, 4, 20).tolist()
            records.append(logs)
            assert tensors[owner_base + 1][240:].tolist() == [999] * 16
            for stage, offset in enumerate((0x6000, 0xA000, 0xE000)):
                for thread, log in enumerate(logs[stage]):
                    input_core = block // 4 * 4 + thread if stage == 0 else block if stage == 1 else block // 4 * 4
                    input_offset = 0 if stage == 0 else (0x6000 if stage == 1 else 0xA000) + thread * 4096
                    assert log[:4] == [
                        *target(block, mode, thread),
                        block,
                        buffers[owner_base].base + offset + thread * 4096,
                    ]
                    assert log[4:8] == [
                        input_core,
                        buffers[base + input_core * 2].base + input_offset,
                        int(mode == 2),
                        1,
                    ]
                    assert log[10:12] == [block, instance]
                    if stage != 1:
                        assert log[8:10] == [block, buffers[owner_base].base + (0x1000 if stage == 0 else 0x2000)]
                    if stage == 2:
                        for k in range(4):
                            source = block // 4 * 4 + k
                            assert log[12 + 2 * k : 14 + 2 * k] == [
                                source,
                                buffers[base + source * 2].base + 0xA000 + thread * 4096,
                            ]
    epoch = audit_ffn_events(path, instances, mode, native=native)
    memory = json.loads((path / "wse_sram.json").read_text())
    allocations = []
    for index, buffer in enumerate(buffers):
        owner = (index % 80) // 2
        info = memory["cores"][owner]
        assert (
            info["base"]
            <= buffer.base
            < buffer.base + buffer.nbytes
            <= info["base"] + info["capacity"] - info["reserve"]
        )
        allocations.append(
            {"identity": str(buffer.identity), "base": buffer.base, "bytes": buffer.nbytes, "owner": owner}
        )
    for core in range(40):
        assert sum(a["bytes"] for a in allocations if a["owner"] == core) == instances * 74 * 1024
        assert any(section["owner"] == core and section["kind"] == 5 for section in memory["sections"])
    (path / "sram-ffn-evidence.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "instances": instances,
                "mode": mode,
                "allocations": allocations,
                "records": records,
                "main_bytes_per_owner_per_instance": 72 * 1024,
                "diagnostic_bytes_per_owner_per_instance": 2048,
                "runtime": memory,
            },
            indent=2,
        )
    )
    return epoch


def audit_ffn_events(path, instances, mode, *, native=True):
    nodes = 5 if native else 3
    events = [json.loads(line) for line in (path / "wse_events.jsonl").read_text().splitlines()]
    assert events[0]["expected_tasks"] == nodes * instances
    assert events[0]["core_count"] == 40
    done = [event for event in events[1:] if event["kind"] == 6]
    assert sorted(event["value"] for event in done) == list(range(1, nodes * instances + 1))
    for instance in range(instances):
        previous = -1
        for stage in range(3):
            task = instance * nodes + stage + int(native)
            by_kind = {
                kind: [e for e in events[1:] if e["kind"] == kind and e["task"] == task] for kind in (4, 5, 16, 17)
            }
            assert sorted(e["block"] for e in by_kind[4]) == sorted(e["block"] for e in by_kind[5]) == list(range(40))
            assert previous < min(e["seq"] for e in by_kind[4])
            for kind in (16, 17):
                assert sorted((e["block"], e["value"]) for e in by_kind[kind]) == [
                    (b, t) for b in range(40) for t in range(4)
                ]
                for e in by_kind[kind]:
                    assert (e["core"], e["pe"]) == target(e["block"], mode, e["value"])
            for block in range(40):
                begin = next(e["seq"] for e in by_kind[4] if e["block"] == block)
                end = next(e["seq"] for e in by_kind[5] if e["block"] == block)
                assert all(begin < e["seq"] < end for kind in (16, 17) for e in by_kind[kind] if e["block"] == block)
            previous = next(e["seq"] for e in done if e["task"] == task)
            if stage == 1:
                continue
            for core in range(40):
                consumed_before = -1
                for generation in range(1, 2 if stage == 0 else 5):
                    group = [
                        e
                        for e in events[1:]
                        if e["task"] == task and e["core"] == core and e.get("generation") == generation
                    ]
                    publish = [e for e in group if e["kind"] == 12]
                    reads = [e for e in group if e["kind"] == 14]
                    consumed = [e for e in group if e["kind"] == 15]
                    assert len(publish) == 1 and publish[0]["pe"] == int(mode == 2)
                    assert len(reads) == len(consumed) == 4
                    assert {e["pe"] for e in reads} == {e["pe"] for e in consumed} == set(range(4))
                    assert consumed_before < publish[0]["seq"] < min(e["seq"] for e in reads)
                    consumed_before = max(e["seq"] for e in consumed)
    for core in range(40):
        active = None
        for e in sorted(events[1:], key=lambda event: event["seq"]):
            if e["core"] != core:
                continue
            if e["kind"] == 4:
                assert active is None
                active = (e["task"], e["block"])
            elif e["kind"] == 5:
                assert active == (e["task"], e["block"])
                active = None
        assert active is None
    return events[0]["epoch"]


def ffn_worker(platform):
    worker = Worker(level=2, platform=platform, runtime="host_build_graph", device_id=0)
    worker.init()
    return worker


def test_sram_ffn_placement_rebind_and_generations(tmp_path, st_platform):
    worker = ffn_worker(st_platform)
    try:
        handles = {n: worker.register(build_sram_ffn(st_platform, n)) for n in (1, 2)}
        for seed in (0, 1):
            for instances in (1, 2):
                for iteration, mode in enumerate((-1, 0, 1, 2, 0)):
                    args, buffers, tensors, golden = prepare_ffn(worker, seed, mode, instances)
                    path = tmp_path / f"{seed}-{instances}-{iteration}"
                    worker.run(handles[instances], args, config_for(path))
                    check_ffn(worker, path, buffers, tensors, golden, mode)
                    for buffer in buffers:
                        release_view(worker, buffer)
        for handle in handles.values():
            worker.unregister(handle)
    finally:
        worker.close()


def test_sram_ffn_pending_reverse_wait(tmp_path, st_platform):
    worker = ffn_worker(st_platform)
    pending = []
    try:
        handle = worker.register(build_sram_ffn(st_platform, 2))
        for seed, mode in ((0, 1), (1, 2)):
            args, buffers, tensors, golden = prepare_ffn(worker, seed, mode, 2)
            path = tmp_path / str(seed)
            run = worker.submit(handle, args, config_for(path))
            pending.append((run, args, buffers, tensors, golden, path, mode))
        epochs = []
        for run, _, buffers, tensors, golden, path, mode in reversed(pending):
            run.wait()
            epochs.append(check_ffn(worker, path, buffers, tensors, golden, mode))
        assert len(set(epochs)) == 2
        for _, _, buffers, *_ in pending:
            for buffer in buffers:
                release_view(worker, buffer)
        worker.unregister(handle)
    finally:
        worker.close()


def test_sram_ffn_rejects_invalid_collaboration(tmp_path, st_platform):
    worker = ffn_worker(st_platform)
    try:
        handle = worker.register(build_sram_ffn(st_platform, 1))
        for mode in (3, 4):
            args, buffers, _, _ = prepare_ffn(worker, 0, mode)
            path = tmp_path / str(mode)
            with pytest.raises(RuntimeError):
                worker.run(handle, args, config_for(path))
            assert_no_launch(path)
            for buffer in buffers:
                release_view(worker, buffer)
        args, buffers, tensors, golden = prepare_ffn(worker, 0, 2)
        path = tmp_path / "recovery"
        worker.run(handle, args, config_for(path))
        check_ffn(worker, path, buffers, tensors, golden, 2)
        for buffer in buffers:
            release_view(worker, buffer)
        worker.unregister(handle)
    finally:
        worker.close()


@pytest.mark.parametrize("native", [False, True])
def test_sram_ffn_fixed_device_capacity(tmp_path, st_platform, native):
    worker = ffn_worker(st_platform)
    try:
        handle = worker.register(build_sram_ffn(st_platform, 2))
        args, buffers, tensors, golden = prepare_ffn(worker, 1, 2, 2, native=native)
        worker.run(handle, args, config_for(tmp_path))
        check_ffn(worker, tmp_path, buffers, tensors, golden, 2, native=native)
        for buffer in buffers:
            release_view(worker, buffer)
        worker.unregister(handle)
    finally:
        worker.close()
