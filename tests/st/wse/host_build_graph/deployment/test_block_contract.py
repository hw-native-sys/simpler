# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Static Block queues, Thread identity and Block-local operation lifetime."""

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
from tests.st.wse.host_build_graph.fixed_sram_layout import copy_from, copy_to, layout_view, release_view
from tests.st.wse.host_build_graph.tile_operations.test_execution import _GATE, load_gate

pytestmark = [pytest.mark.platforms(["wsesim"]), pytest.mark.runtime("host_build_graph"), pytest.mark.device_count(1)]


@lru_cache
def build(platform, scope=False):
    compiler = KernelCompiler(platform)
    here = Path(__file__).parent / "kernels"
    stem = "block_scope" if scope else "block_contract"
    child = CoreCallable.build(
        signature=[D.INOUT] + ([] if scope else [D.IN] * 2),
        binary=compiler.compile_incore(
            str(here / f"{stem}.cpp"),
            core_type="",
            extra_include_dirs=compiler.get_orchestration_include_dirs("host_build_graph"),
        ),
    )
    return ChipCallable.build(
        signature=[D.INOUT] + ([] if scope else [D.IN] * 3),
        func_name="aicpu_orchestration_entry",
        binary=compiler.compile_orchestration("host_build_graph", str(here / f"{stem}_orch.cpp")),
        children=[(0, child)],
    )


def start(platform):
    worker = Worker(level=2, platform=platform, runtime="host_build_graph", device_id=0)
    worker.init()
    return worker


@pytest.mark.parametrize("native", [False, True])
def test_one_to_four_threads_and_block_binding(tmp_path, st_platform, native):
    worker = start(st_platform)
    try:
        handle = worker.register(build(st_platform))
        for threads in range(1, 5):
            for token in (19, 37):
                data = torch.full((160,), 999, dtype=torch.uint64)
                buffer = layout_view(worker, data.nbytes, core=1)
                copy_to(worker, buffer, data)
                args = TaskArgs()
                args.add_tensor(buffer.tensor(shapes=(160,), dtype=DataType.UINT64), TensorArgType.INOUT)
                for value in (threads, int(native), token):
                    args.add_scalar(value)
                path = tmp_path / f"{threads}-{token}"
                worker.run(handle, args, config_for(path))
                copy_from(worker, data, buffer)
                rows = data.reshape(5, 4, 8).tolist()
                for b in range(5):
                    for t in range(4):
                        assert rows[b][t] == (
                            [1, b, t, threads, b % 2, (t + 3) % 4, token, buffer.base + (b * 32 + t * 8) * 8]
                            if t < threads
                            else [999] * 8
                        )
                events = [json.loads(line) for line in (path / "wse_events.jsonl").read_text().splitlines()][1:]
                ends = {e["task"]: e["seq"] for e in events if e["kind"] == 6}
                first = int(native)
                assert ends[first] < min(e["seq"] for e in events if e["kind"] == 4 and e["task"] == first + 1)
                for core in range(2):
                    active = None
                    order = []
                    for e in sorted(events, key=lambda e: e["seq"]):
                        if e["core"] != core:
                            continue
                        if e["kind"] == 4:
                            assert active is None
                            active = (e["task"], e["block"])
                            order.append(active)
                        if e["kind"] == 5:
                            assert active == (e["task"], e["block"])
                            active = None
                    assert active is None
                    assert order == [(task, b) for task in (first, first + 1) for b in range(5) if b % 2 == core]
                release_view(worker, buffer)
        worker.unregister(handle)
    finally:
        worker.close()


def test_block_end_invalidates_only_its_own_shared_tiles(tmp_path, st_platform):
    scope_gate = _GATE.replace(
        "else {mask|=", "else if(block==5){changed.wait(lock,[]{return entered && (mask&14)==14;});} else {mask|="
    )
    gate = load_gate(tmp_path, "wse_scope_probe", 30, source_code=scope_gate)
    worker = start(st_platform)
    try:
        handle = worker.register(build(st_platform, True))
        data = torch.full((48,), 999, dtype=torch.uint64)
        buffer = layout_view(worker, data.nbytes, core=1)
        copy_to(worker, buffer, data)
        args = TaskArgs()
        args.add_tensor(buffer.tensor(shapes=(48,), dtype=DataType.UINT64), TensorArgType.INOUT)
        run = worker.submit(handle, args, config_for(tmp_path))
        assert gate.wait_probe(), "Core 0's second Block and Core 1's waiting consumers must coexist"
        with pytest.raises(TimeoutError):
            run.wait(timeout=0.01)
        with pytest.raises(RuntimeError, match="in.flight"):
            worker.free(buffer.root)
        gate.release_probe()
        run.wait()
        copy_from(worker, data, buffer)
        assert data.reshape(3, 4, 4).tolist() == [[[100 + b, int(b == 2), t, t] for t in range(4)] for b in range(3)]
        events = sorted(
            [json.loads(line) for line in (tmp_path / "wse_events.jsonl").read_text().splitlines()][1:],
            key=lambda e: e["seq"],
        )
        ends = {e["block"]: e["seq"] for e in events if e["kind"] == 5}
        begins = {e["block"]: e["seq"] for e in events if e["kind"] == 4}
        assert ends[0] < begins[2]
        assert begins[1] < ends[2]
        assert len([e for e in events if e["kind"] == 6]) == 1
        release_view(worker, buffer)
        worker.unregister(handle)
    finally:
        gate.release_probe()
        worker.close()
