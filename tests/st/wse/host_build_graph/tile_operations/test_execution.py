# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Observe Block/Thread progress and operation sharing within a Kernel Block."""

import ctypes
import json
import subprocess
from functools import lru_cache
from pathlib import Path

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, ChipCallable, CoreCallable, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.kernel_compiler import KernelCompiler
from tests.st.wse.host_build_graph.direct_launch.test_direct_launch import audit_events
from tests.st.wse.host_build_graph.fixed_sram_layout import copy_from, copy_to, layout_view, release_view

pytestmark = [pytest.mark.platforms(["wsesim"]), pytest.mark.runtime("host_build_graph"), pytest.mark.device_count(1)]
HERE = Path(__file__).parent
_GATE = r"""
#include <chrono>
#include <condition_variable>
#include <mutex>
namespace {std::mutex mutex;std::condition_variable changed;bool entered=false,released=false;unsigned mask=0;}
extern "C" void HOOK(int block){
 std::unique_lock lock(mutex);
 if(block==0){entered=true;changed.notify_all();changed.wait(lock,[]{return released;});}
 else {mask|=1u<<block;changed.notify_all();}
}
extern "C" bool wait_probe(){
 std::unique_lock lock(mutex);
 return changed.wait_for(lock,std::chrono::seconds(15),[]{return entered && (mask & MASK)==MASK;});
}
extern "C" void release_probe(){std::lock_guard lock(mutex);released=true;changed.notify_all();}
"""


def load_gate(tmp_path, hook, mask, *, source_code=_GATE):
    source = tmp_path / "gate.cpp"
    source.write_text(source_code.replace("HOOK", hook).replace("MASK", str(mask)))
    library = tmp_path / "gate.so"
    subprocess.run(
        ["c++", "-std=c++17", "-shared", "-fPIC", "-pthread", str(source), "-o", str(library)], check=True, timeout=60
    )
    gate = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    gate.wait_probe.restype = ctypes.c_bool
    return gate


@lru_cache
def build_probe(platform, shared=False):
    compiler = KernelCompiler(platform)
    sources = ["shared_block"] if shared else ["probe"]
    signatures = [[D.IN, D.OUT]] if shared else [[D.IN, D.OUT, D.OUT, D.IN]]
    children = []
    for index, (name, sig) in enumerate(zip(sources, signatures, strict=True)):
        binary = compiler.compile_incore(
            str(HERE / f"kernels/{name}.cpp"),
            core_type="",
            extra_include_dirs=compiler.get_orchestration_include_dirs("host_build_graph"),
        )
        children.append((index, CoreCallable.build(signature=sig, binary=binary)))
    return ChipCallable.build(
        signature=[D.IN, D.OUT] if shared else [D.IN, D.OUT, D.OUT, D.IN, D.IN],
        func_name="aicpu_orchestration_entry",
        binary=compiler.compile_orchestration(
            "host_build_graph", str(HERE / ("kernels/shared_block_orch.cpp" if shared else "kernels/probe_orch.cpp"))
        ),
        children=children,
    )


def prepare(worker, tensors):
    args = TaskArgs()
    buffers = []
    for i, t in enumerate(tensors):
        b = layout_view(worker, t.nbytes, core=i % 40)
        buffers.append(b)
        copy_to(worker, b, t)
        args.add_tensor(
            b.tensor(shapes=tuple(t.shape), dtype=DataType.FLOAT32),
            TensorArgType.INPUT if i == 0 else TensorArgType.OUTPUT_EXISTING,
        )
    return args, buffers


@pytest.mark.parametrize(
    "cores,blocks", [(40, 1), (40, 5), (40, 39), (40, 40), (40, 159), (0, 160), (40, 161), (0, 323)]
)
def test_temporary_operations_keep_independent_blocks(tmp_path, st_platform, cores, blocks):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    try:
        worker.init()
        handle = worker.register(build_probe(st_platform))
        tensors = [
            torch.tensor([[-2.0, 1.0], [3.0, -4.0]]),
            torch.full((blocks, 2, 2), -99.0),
            torch.full((blocks, 2), -1.0),
        ]
        args, buffers = prepare(worker, tensors)
        args.add_scalar(blocks)
        args.add_scalar(0)
        config = CallConfig()
        config.output_prefix = str(tmp_path)
        worker.run(handle, args, config)
        copy_from(worker, tensors[1], buffers[1])
        copy_from(worker, tensors[2], buffers[2])
        expected = torch.relu(tensors[0] @ tensors[0]).expand(blocks, 2, 2)
        torch.testing.assert_close(tensors[1], expected, rtol=0, atol=0)
        audit_events(tmp_path / "wse_events.jsonl", cores or 40, blocks)
        for b in buffers:
            release_view(worker, b)
        worker.unregister(handle)
    finally:
        worker.close()


def test_waiting_thread_holds_its_block_but_not_other_cores(tmp_path, st_platform):
    gate = load_gate(tmp_path, "wse_independent_probe", 30)
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    try:
        worker.init()
        handle = worker.register(build_probe(st_platform))
        tensors = [torch.tensor([[-2.0, 1.0], [3.0, -4.0]]), torch.full((320, 2, 2), -99.0), torch.full((320, 2), -1.0)]
        args, buffers = prepare(worker, tensors)
        args.add_scalar(320)
        args.add_scalar(1)
        config = CallConfig()
        config.output_prefix = str(tmp_path)
        run = worker.submit(handle, args, config)
        assert gate.wait_probe(), "sibling Threads and another Core did not progress while Thread 0 was held"
        with pytest.raises(TimeoutError):
            run.wait(timeout=0.01)
        gate.release_probe()
        run.wait()
        copy_from(worker, tensors[1], buffers[1])
        torch.testing.assert_close(tensors[1], torch.relu(tensors[0] @ tensors[0]).expand(320, 2, 2), rtol=0, atol=0)
        events = [json.loads(line) for line in (tmp_path / "wse_events.jsonl").read_text().splitlines()][1:]
        end = {e["block"]: e["seq"] for e in events if e["kind"] == 5}
        first_thread_end = next(e["seq"] for e in events if e["kind"] == 17 and e["block"] == 0 and e["value"] == 0)
        assert all(
            next(e["seq"] for e in events if e["kind"] == 16 and e["block"] == 0 and e["value"] == thread)
            < first_thread_end
            for thread in (1, 2, 3)
        )
        assert (
            next(e["seq"] for e in events if e["kind"] == 16 and e["block"] == 1 and e["value"] == 0) < first_thread_end
        )
        assert next(e["seq"] for e in events if e["kind"] == 4 and e["block"] == 40) > end[0]
        for b in buffers:
            release_view(worker, b)
        worker.unregister(handle)
    finally:
        gate.release_probe()
        worker.close()


def test_shared_state_between_threads_of_one_block(tmp_path, st_platform):
    gate = load_gate(tmp_path, "wse_shared_probe", 14)
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    try:
        worker.init()
        handle = worker.register(build_probe(st_platform, True))
        tensors = [torch.tensor([[11.0, 12.0], [13.0, 14.0]]), torch.full((4, 2, 2), -99.0)]
        args, buffers = prepare(worker, tensors)
        config = CallConfig()
        config.output_prefix = str(tmp_path)
        run = worker.submit(handle, args, config)
        assert gate.wait_probe(), "consumers did not acknowledge real waits before producer publication"
        with pytest.raises(TimeoutError):
            run.wait(timeout=0.01)
        with pytest.raises(RuntimeError, match="in.flight"):
            worker.free(buffers[1].root)
        gate.release_probe()
        run.wait()
        copy_from(worker, tensors[1], buffers[1])
        torch.testing.assert_close(tensors[1][[1, 2, 3]], tensors[0].expand(3, 2, 2), rtol=0, atol=0)
        assert torch.all(tensors[1][0] == -99)
        events = [json.loads(line) for line in (tmp_path / "wse_events.jsonl").read_text().splitlines()][1:]
        publication = next(e["seq"] for e in events if e["kind"] == 12)
        assert {e["pe"] for e in events if e["kind"] == 13 and e["seq"] < publication} == {1, 2, 3}
        assert {e["slot"] for e in events if e["kind"] in (12, 13, 14, 15)} == {2}
        assert {e["pe"] for e in events if e["kind"] == 15} == {1, 2, 3}
        assert len([e for e in events if e["kind"] == 6]) == 1
        for b in buffers:
            release_view(worker, b)
        worker.unregister(handle)
    finally:
        gate.release_probe()
        worker.close()
