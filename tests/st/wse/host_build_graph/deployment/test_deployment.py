# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exact block placement through ordinary/native Graphs and finite SRAM."""

import json
from functools import lru_cache
from pathlib import Path

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from tests.st.wse.host_build_graph.counter import test_counter as counter
from tests.st.wse.host_build_graph.counter.test_counter import config_for
from tests.st.wse.host_build_graph.fixed_sram_layout import copy_from, copy_to, layout_view, release_view
from tests.st.wse.host_build_graph.scene_helpers import build_callable

pytestmark = [pytest.mark.platforms(["wsesim"]), pytest.mark.runtime("host_build_graph"), pytest.mark.device_count(1)]


@lru_cache
def build_probe(platform, orchestration="probe_orch.cpp"):
    return build_callable(
        platform,
        Path(__file__).parent / "kernels",
        orchestration,
        [D.INOUT] * 3 + [D.IN] * 4,
        [("probe.cpp", [D.INOUT] * 3 + [D.IN] * 2)],
    )


def probe_args(worker, blocks, mode, native, token, owner):
    tensors = [
        (torch.arange(1024).float() - 513).reshape(32, 32),
        torch.full((2, blocks, 32, 32), -999.0),
        torch.full((2, blocks, 8), 999, dtype=torch.uint64),
    ]
    buffers = []
    args = TaskArgs()
    for index, tensor in enumerate(tensors):
        buffer = layout_view(worker, tensor.nbytes, core=owner if index < 2 else 0)
        copy_to(worker, buffer, tensor)
        args.add_tensor(
            buffer.tensor(shapes=tuple(tensor.shape), dtype=DataType.UINT64 if index == 2 else DataType.FLOAT32),
            TensorArgType.INOUT,
        )
        buffers.append(buffer)
    for value in (blocks, mode, int(native), token):
        args.add_scalar(value)
    return args, buffers, tensors


def expected_target(block, cores, mode, stage):
    if mode == 3:
        return cores - 1, 3 - stage
    core, pe = block % cores, 0
    return ((core + 1) % cores, (pe + 1 + stage) % 4) if mode == 2 else (core, pe)


def check_probe(worker, path, buffers, tensors, cores, mode, native, token, owner):
    for index in (1, 2):
        copy_from(worker, tensors[index], buffers[index])
    blocks = tensors[1].shape[1]
    golden = torch.relu(tensors[0]).expand_as(tensors[1])
    torch.testing.assert_close(tensors[1], golden, rtol=0, atol=0)
    events = [json.loads(line) for line in (path / "wse_events.jsonl").read_text().splitlines()]
    expected_tasks = 4 if native else 2
    assert events[0]["expected_tasks"] == expected_tasks
    done = [event for event in events[1:] if event["kind"] == 6]
    assert sorted(event["value"] for event in done) == list(range(1, expected_tasks + 1))
    for stage in range(2):
        task = stage + int(native)
        for kind in (16, 17):
            emitted = [event for event in events[1:] if event["kind"] == kind and event["task"] == task]
            assert sorted(event["block"] for event in emitted) == list(range(blocks))
            for event in emitted:
                assert (event["core"], event["pe"]) == expected_target(event["block"], cores, mode, stage)
        for block in range(blocks):
            core, pe = expected_target(block, cores, mode, stage)
            assert tensors[2][stage, block].tolist() == [
                block,
                core,
                pe,
                owner,
                owner,
                buffers[0].base,
                buffers[1].base + (stage * blocks + block) * 4096,
                token,
            ]
    memory = json.loads((path / "wse_sram.json").read_text())
    queues = [section for section in memory["sections"] if section["kind"] == 9]
    assert sum(section["bytes"] for section in queues) == 2 * blocks * 56
    for section in memory["sections"]:
        core = memory["cores"][section["owner"]]
        assert core["runtime_base"] <= section["address"]
        assert section["address"] + section["bytes"] <= core["runtime_base"] + core["reserve"]
    (path / "deployment-evidence.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "cores": cores,
                "blocks": blocks,
                "mode": mode,
                "native": native,
                "owner": owner,
                "records": tensors[2].tolist(),
                "allocations": [
                    {"identity": str(buf.identity), "base": buf.base, "bytes": buf.nbytes} for buf in buffers
                ],
            },
            indent=2,
        )
    )


def configured_worker(platform, cores):
    worker = Worker(level=2, platform=platform, runtime="host_build_graph", device_id=0)
    worker.init()
    return worker


@pytest.mark.parametrize("native", [False, True])
def test_deployment_block_counts_and_replay(tmp_path, st_platform, native):
    worker = configured_worker(st_platform, 40)
    try:
        handle = worker.register(build_probe(st_platform))
        for blocks in (39, 40, 41, 159, 160, 161):
            for iteration, mode in enumerate((0, 1, 2, 3, 2)):
                path = tmp_path / f"{blocks}-{iteration}"
                token = blocks * 10 + iteration
                args, buffers, tensors = probe_args(worker, blocks, mode, native, token, 3)
                worker.run(handle, args, config_for(path))
                check_probe(worker, path, buffers, tensors, 40, mode, native, token, 3)
                for buffer in buffers:
                    release_view(worker, buffer)
        worker.unregister(handle)
    finally:
        worker.close()


@pytest.mark.parametrize("cores", [40])
def test_deployment_topology_and_remote_owner(tmp_path, st_platform, cores):
    device_cores = cores
    worker = configured_worker(st_platform, device_cores)
    try:
        handle = worker.register(build_probe(st_platform))
        for mode in (0, 2):
            args, buffers, tensors = probe_args(worker, cores * 4 + 1, mode, True, 123, device_cores - 1)
            path = tmp_path / str(mode)
            worker.run(handle, args, config_for(path))
            check_probe(worker, path, buffers, tensors, cores, mode, True, 123, device_cores - 1)
            for buffer in buffers:
                release_view(worker, buffer)
        worker.unregister(handle)
    finally:
        worker.close()


def assert_no_launch(path):
    events = path / "wse_events.jsonl"
    if events.exists():
        rows = [json.loads(line) for line in events.read_text().splitlines()]
        assert not any(row.get("kind") in (3, 4) for row in rows)


@pytest.mark.parametrize("native", [False, True])
def test_deployment_rejects_invalid_and_recovers(tmp_path, st_platform, native):
    worker = configured_worker(st_platform, 40)
    try:
        handle = worker.register(build_probe(st_platform))
        for mode in (4, 5, 6, 7, 8, 10, 11, *([9] if native else [])):
            path = tmp_path / f"invalid-{mode}"
            args, buffers, _ = probe_args(worker, 5, mode, native, 0, 3)
            with pytest.raises(RuntimeError):
                worker.run(handle, args, config_for(path))
            assert_no_launch(path)
            for buffer in buffers:
                release_view(worker, buffer)
            args, buffers, tensors = probe_args(worker, 5, 2, native, mode, 3)
            path = tmp_path / f"recovery-{mode}"
            worker.run(handle, args, config_for(path))
            check_probe(worker, path, buffers, tensors, 40, 2, native, mode, 3)
            for buffer in buffers:
                release_view(worker, buffer)
        worker.unregister(handle)
    finally:
        worker.close()


def test_deployment_fixed_capacity_refuses_before_go(tmp_path, st_platform):
    worker = configured_worker(st_platform, 40)
    try:
        handle = worker.register(build_probe(st_platform))
        for mode in (12, 3):
            args, buffers, _ = probe_args(worker, 4096, mode, True, 2, 3)
            path = tmp_path / f"reject-{mode}"
            with pytest.raises(RuntimeError):
                worker.run(handle, args, config_for(path))
            assert "WSE Runtime SRAM capacity exhausted" in (path / "wse_prepare_error.txt").read_text()
            assert_no_launch(path)
            for buffer in buffers:
                release_view(worker, buffer)
            args, buffers, tensors = probe_args(worker, 1, 0, True, 3, 3)
            path = tmp_path / f"small-{mode}"
            worker.run(handle, args, config_for(path))
            check_probe(worker, path, buffers, tensors, 40, 0, True, 3, 3)
            for buffer in buffers:
                release_view(worker, buffer)
        worker.unregister(handle)
    finally:
        worker.close()


@lru_cache
def build_deployed_counter(platform):
    here = Path(__file__).parent / "kernels"
    return build_callable(
        platform,
        here,
        "counter_orch.cpp",
        [D.INOUT] * 3 + [D.IN] * 6,
        [("counter.cpp", [D.INOUT] * 3 + [D.IN] * 5)],
        extra_include_dirs=[str(here.parent.parent / "counter/kernels")],
    )


def test_deployment_sram_counter(tmp_path, st_platform):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    try:
        worker.init()
        handle = worker.register(build_deployed_counter(st_platform))
        for iteration, variant in enumerate((0, 1, 0)):
            base = iteration * 17
            args, buffers, tensors = counter.prepare(worker, base, variant, owners=(3, 2, 0))
            path = tmp_path / str(iteration)
            worker.run(handle, args, config_for(path))
            placements = [[0, 0], [0, 1], [1, 2], [1, 3]]
            if variant:
                placements = [[1 - core, (pe + 1) % 4] for core, pe in placements]
            counter.check(worker, buffers, tensors, base, 40, path, False, placements=placements)
            memory = json.loads((path / "wse_sram.json").read_text())
            ownership = []
            for buffer in buffers:
                owners = [
                    core
                    for core in range(40)
                    if memory["cores"][core]["base"] <= buffer.base
                    and buffer.base + buffer.nbytes
                    <= memory["cores"][core]["base"]
                    + memory["cores"][core]["capacity"]
                    - memory["cores"][core]["reserve"]
                ]
                assert len(owners) == 1
                ownership.append(owners[0])
            assert ownership == [3, 2, 0]
            (path / "counter-deployment.json").write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "placements": placements,
                        "owners": ownership,
                        "allocations": [{"base": buffer.base, "bytes": buffer.nbytes} for buffer in buffers],
                    },
                    indent=2,
                )
            )
            for buffer in buffers:
                release_view(worker, buffer)
        worker.unregister(handle)
    finally:
        worker.close()


def test_deployment_timeout_ownership_and_close(tmp_path, st_platform):
    gate = counter.load_gate(tmp_path, "wse_counter_probe", 12)
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    try:
        worker.init()
        handle = worker.register(build_deployed_counter(st_platform))
        args, buffers, tensors = counter.prepare(worker, 29, 1, gated=True)
        path = tmp_path / "wait"
        run = worker.submit(handle, args, config_for(path))
        assert gate.wait_probe()
        with pytest.raises(TimeoutError):
            run.wait(timeout=0.01)
        for buffer in buffers:
            with pytest.raises(RuntimeError, match="in.flight"):
                worker.free(buffer.root)
        gate.release_probe()
        run.wait()
        placements = [[1, 1], [1, 2], [0, 3], [0, 0]]
        counter.check(worker, buffers, tensors, 29, 40, path, False, placements=placements)
        for buffer in buffers:
            release_view(worker, buffer)
        args2, _, _ = counter.prepare(worker, 41, 0)
        worker.submit(handle, args2, config_for(tmp_path / "close"))
        worker.close()
        events = [json.loads(line) for line in (tmp_path / "close/wse_events.jsonl").read_text().splitlines()][1:]
        assert sum(event["kind"] == 9 for event in events) == 1
        assert sum(event["kind"] == 8 for event in events) == 40
        assert sum(event["kind"] == 7 for event in events) == 160
    finally:
        gate.release_probe()
        worker.close()


@pytest.mark.parametrize("native", [False, True])
def test_deployment_cached_binding_and_predicate(tmp_path, st_platform, native):
    worker = configured_worker(st_platform, 40)
    try:
        handle = worker.register(build_probe(st_platform, "binding_orch.cpp"))
        for enabled in (False, True):
            args, buffers, tensors = probe_args(worker, 1, int(enabled), native, 91, 0)
            path = tmp_path / str(enabled)
            worker.run(handle, args, config_for(path))
            if enabled:
                check_probe(worker, path, buffers, tensors, 40, 3, native, 91, 0)
                for buffer, tensor in zip(buffers, tensors, strict=True):
                    if tensor.dtype == torch.uint64:
                        tensor.fill_(999)
                    copy_to(worker, buffer, tensor)
                restored = tmp_path / "restore"
                worker.run(handle, args, config_for(restored))
                check_probe(worker, restored, buffers, tensors, 40, 3, native, 91, 0)
            else:
                for buffer, tensor in zip(buffers[1:], tensors[1:], strict=True):
                    copy_from(worker, tensor, buffer)
                assert bool(torch.all(tensors[1] == -999))
                assert bool(torch.all(tensors[2] == 999))
                events = [json.loads(line) for line in (path / "wse_events.jsonl").read_text().splitlines()]
                skipped = sorted(
                    (event["task"], event["core"], event["pe"]) for event in events[1:] if event["kind"] == 11
                )
                assert skipped == [(int(native), 39, 0), (int(native) + 1, 39, 0)]
                expected = 4 if native else 2
                assert sorted(event["value"] for event in events[1:] if event["kind"] == 6) == list(
                    range(1, expected + 1)
                )
            for buffer in buffers:
                release_view(worker, buffer)
        worker.unregister(handle)
    finally:
        worker.close()
