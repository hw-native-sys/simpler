# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Application counter publication through independent PE tasks and graph replay."""

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
from tests.st.wse.host_build_graph.tile_operations.test_execution import load_gate

pytestmark = [pytest.mark.platforms(["wsesim"]), pytest.mark.runtime("host_build_graph"), pytest.mark.device_count(1)]
HERE = Path(__file__).parent
PRODUCERS, CONSUMERS, ROUNDS = 2, 2, 3


@lru_cache
def build_counter(platform):
    compiler = KernelCompiler(platform)
    children = []
    for index, name in enumerate(("mixed", "producer", "consumer")):
        binary = compiler.compile_incore(
            str(HERE / f"kernels/{name}.cpp"),
            core_type="",
            extra_include_dirs=compiler.get_orchestration_include_dirs("host_build_graph"),
        )
        children.append((index, CoreCallable.build(signature=[D.INOUT] * 3 + [D.IN] * 5, binary=binary)))
    return ChipCallable.build(
        signature=[D.INOUT] * 3 + [D.IN] * 6,
        func_name="aicpu_orchestration_entry",
        binary=compiler.compile_orchestration("host_build_graph", str(HERE / "kernels/orchestration.cpp")),
        children=children,
    )


def prepare(worker, base, split, gated=False, *, owners=None):
    tensors = [
        torch.tensor([991, 992, base, 0, 0, 993], dtype=torch.uint64),
        torch.zeros(PRODUCERS, dtype=torch.uint64),
        torch.full((PRODUCERS + CONSUMERS, ROUNDS, 6), 99999, dtype=torch.uint64),
    ]
    args, buffers = TaskArgs(), []
    for index, tensor in enumerate(tensors):
        buffer = layout_view(worker, tensor.nbytes, core=owners[index] if owners is not None else len(buffers) % 40)
        copy_to(worker, buffer, tensor)
        buffers.append(buffer)
        view = buffer.tensor(
            shapes=(3,) if index == 0 else tuple(tensor.shape),
            dtype=DataType.UINT64,
            byte_offset=16 if index == 0 else 0,
        )
        args.add_tensor(view, TensorArgType.INOUT)
    base_bits = base if base < (1 << 63) else base - (1 << 64)
    for value in (PRODUCERS, CONSUMERS, ROUNDS, base_bits, int(gated), int(split)):
        args.add_scalar(value)
    return args, buffers, tensors


def check(worker, buffers, tensors, base, cores, path, split, *, placements=None):
    for buffer, tensor in zip(buffers, tensors, strict=True):
        copy_from(worker, tensor, buffer)
    assert tensors[0].tolist() == [991, 992, base + ROUNDS * PRODUCERS, ROUNDS * CONSUMERS, CONSUMERS, 993]
    assert tensors[1].tolist() == [base + ROUNDS * 100 + p for p in range(PRODUCERS)]
    (path / "counter-results.json").write_text(
        json.dumps(
            {
                "base": base,
                "cores": cores,
                "split": split,
                "counter": tensors[0].tolist(),
                "payload": tensors[1].tolist(),
                "records": tensors[2].tolist(),
            },
            indent=2,
        )
    )
    rows = tensors[2].tolist()
    for g in range(1, ROUNDS + 1):
        for b, rounds in enumerate(rows):
            row = rounds[g - 1]
            assert row[:2] == (placements[b] if placements is not None else [b // 2, b % 2])
            assert row[4] == row[3] + 1
            expected = base + g * 100 + b if b < PRODUCERS else sum(base + g * 100 + p for p in range(PRODUCERS))
            assert row[5] == expected
        assert sorted(rows[p][g - 1][4] for p in range(PRODUCERS)) == list(
            range(base + (g - 1) * PRODUCERS + 1, base + g * PRODUCERS + 1)
        )
        assert sorted(rows[c][g - 1][4] for c in range(PRODUCERS, PRODUCERS + CONSUMERS)) == list(
            range((g - 1) * CONSUMERS + 1, g * CONSUMERS + 1)
        )
    assert all(rows[c][0][2] == 1 for c in range(PRODUCERS, PRODUCERS + CONSUMERS))
    events = [json.loads(line) for line in (path / "wse_events.jsonl").read_text().splitlines()]
    completed = [e for e in events[1:] if e["kind"] == 6]
    assert len(completed) == events[0]["expected_tasks"] == (4 if split else 3)
    assert sorted(e["value"] for e in completed) == list(range(1, len(completed) + 1))
    starts = [(e["task"], e["block"]) for e in events[1:] if e["kind"] == 4]
    ends = [(e["task"], e["block"]) for e in events[1:] if e["kind"] == 5]
    assert sorted(starts) == sorted(ends) and len(starts) == len(set(starts))
    assert (path / "deps.json").is_file()
    return events[0]["epoch"]


def config_for(path):
    config = CallConfig()
    config.output_prefix = str(path)
    config.enable_dep_gen = True
    return config


@pytest.mark.parametrize("cores,split", [(40, False), (0, False), (40, True)])
def test_counter_graph_rebind_and_reuse(tmp_path, st_platform, cores, split):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    try:
        worker.init()
        handle = worker.register(build_counter(st_platform))
        epochs = []
        for repeat, base in enumerate((0, 17)):
            args, buffers, tensors = prepare(worker, base, split)
            for reuse in range(2):
                if reuse:
                    for buffer, tensor in zip(buffers, prepare_values(base), strict=True):
                        copy_to(worker, buffer, tensor)
                path = tmp_path / f"{repeat}-{reuse}"
                worker.run(handle, args, config_for(path))
                epochs.append(check(worker, buffers, tensors, base, cores or 40, path, split))
            for buffer in buffers:
                release_view(worker, buffer)
            with pytest.raises((ValueError, RuntimeError), match="released|closed|stale|freed"):
                worker.submit(handle, args)
        assert len(set(epochs)) == 4
        worker.unregister(handle)
    finally:
        worker.close()


def prepare_values(base):
    return [
        torch.tensor([991, 992, base, 0, 0, 993], dtype=torch.uint64),
        torch.zeros(PRODUCERS, dtype=torch.uint64),
        torch.full((PRODUCERS + CONSUMERS, ROUNDS, 6), 99999, dtype=torch.uint64),
    ]


def test_counter_wait_timeout_and_close(tmp_path, st_platform):
    gate = load_gate(tmp_path, "wse_counter_probe", 12)
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    try:
        worker.init()
        handle = worker.register(build_counter(st_platform))
        args, buffers, tensors = prepare(worker, 29, True, gated=True)
        path = tmp_path / "run"
        run = worker.submit(handle, args, config_for(path))
        assert gate.wait_probe(), "producer and consumers must be simultaneously active"
        with pytest.raises(TimeoutError):
            run.wait(timeout=0.01)
        for buffer in buffers:
            with pytest.raises(RuntimeError, match="in.flight"):
                worker.free(buffer.root)
        gate.release_probe()
        run.wait()
        check(worker, buffers, tensors, 29, 40, path, True)
        args2, _, _ = prepare(worker, 41, False)
        worker.submit(handle, args2, config_for(tmp_path / "close"))
        worker.close()
        closed = [json.loads(line) for line in (tmp_path / "close/wse_events.jsonl").read_text().splitlines()][1:]
        assert len([e for e in closed if e["kind"] == 9]) == 1
        assert len([e for e in closed if e["kind"] == 8]) == 40
        assert len([e for e in closed if e["kind"] == 7]) == 160
    finally:
        gate.release_probe()
        worker.close()


def test_counter_pending_runs_and_device_isolation(tmp_path, st_platform):
    for device in (0, 1):
        worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=device)
        try:
            worker.init()
            handle = worker.register(build_counter(st_platform))
            pending = []
            for index in range(2):
                base = 50 * device + index * 7
                args, buffers, tensors = prepare(worker, base, True)
                path = tmp_path / f"{device}-{index}"
                pending.append(
                    (
                        worker.submit(handle, args, config_for(path)),
                        buffers,
                        tensors,
                        base,
                        path,
                    )
                )
            for run, buffers, tensors, base, path in reversed(pending):
                run.wait()
                check(worker, buffers, tensors, base, 40, path, True)
                for buffer in buffers:
                    release_view(worker, buffer)
            worker.unregister(handle)
        finally:
            worker.close()


def test_counter_rejects_round_target_overflow(tmp_path, st_platform):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    try:
        worker.init()
        handle = worker.register(build_counter(st_platform))
        base = (1 << 64) - 2
        args, buffers, tensors = prepare(worker, base, False)
        worker.run(handle, args, config_for(tmp_path))
        for buffer, tensor in zip(buffers, tensors, strict=True):
            copy_from(worker, tensor, buffer)
        assert tensors[0].tolist() == [991, 992, base, 0, 0, 993]
        assert tensors[1].tolist() == [0, 0]
        assert all(row[0][0] == (1 << 64) - 4 for row in tensors[2].tolist())
        for buffer in buffers:
            release_view(worker, buffer)
        worker.unregister(handle)
    finally:
        worker.close()
