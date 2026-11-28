# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fixed device SRAM through the ordinary Worker and binding surface."""

import json
from pathlib import Path

import _task_interface
import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, ChipCallable, CoreCallable, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.environment import PROJECT_ROOT
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.runtime_builder import RuntimeBuilder

pytestmark = [pytest.mark.platforms(["wsesim"]), pytest.mark.runtime("host_build_graph"), pytest.mark.device_count(1)]


def fixed_layout():
    spec = json.loads((PROJECT_ROOT / "src/wse/platform/config/default_simulation.json").read_text())
    cores = spec["hardware"]["core_count"]

    def expand(value):
        return [value] * cores if isinstance(value, int) else value

    capacities = expand(spec["hardware"]["sram_bytes_per_core"])
    reserves = expand(spec["runtime"]["sram_reserve_bytes_per_core"])
    sizes = [capacity - reserve for capacity, reserve in zip(capacities, reserves, strict=True)]
    return capacities, sizes, spec["hardware"]["global_memory_bytes"]


def test_standard_worker_fixed_sram_and_empty_run(st_platform):
    assert not any(
        hasattr(_task_interface, name) for name in ("_wse_malloc", "_wse_submit", "_wse_query", "_wse_configure")
    )
    assert RuntimeBuilder(st_platform).get_binaries("host_build_graph").aicpu_path is None
    capacities, sizes, global_bytes = fixed_layout()
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    worker.init()
    try:
        with pytest.raises(NotImplementedError):
            worker.malloc(128)
        with pytest.raises(NotImplementedError):
            worker.malloc(sum(sizes))
        for core, size in enumerate(sizes):
            local = worker.get_base_addr(coreid=core)
            assert local.nbytes == size
            worker.copy_to(local, torch.tensor([core, core + 1], dtype=torch.int32))
        for core in range(len(sizes)):
            local = worker.get_base_addr(memory="local", coreid=core)
            out = torch.empty(2, dtype=torch.int32)
            worker.copy_from(out, local)
            assert out.tolist() == [core, core + 1]
        assert worker._chip_worker.committed_device_memory == sum(capacities) + global_bytes
        compiler = KernelCompiler(st_platform)
        source = Path(__file__).parent / "direct_launch/kernels/empty_orch.cpp"
        callable_ = ChipCallable.build(
            signature=[],
            func_name="aicpu_orchestration_entry",
            binary=compiler.compile_orchestration("host_build_graph", str(source)),
            children=[],
        )
        handle = worker.register(callable_)
        worker.submit(handle, TaskArgs()).wait()
    finally:
        worker.close()


def test_l3_child_fixed_memory_windows(st_platform, st_device_ids):
    worker = Worker(
        level=3, num_sub_workers=0, platform=st_platform, runtime="host_build_graph", device_ids=st_device_ids[:1]
    )
    try:
        worker.init()
        with pytest.raises(NotImplementedError):
            worker.alloc_child_tensor(0, (1,), DataType.FLOAT32)
        host = worker.create_buffer(4)
        local = worker.get_base_addr(coreid=0, worker_id=0)
        global_mem = worker.get_base_addr(memory="global", worker_id=0)
        for value, target in ((b"abcd", local), (b"wxyz", global_mem)):
            host.shm.buf[:4] = value
            worker.copy_to(target + 128, host)
        for value, source in ((b"abcd", local), (b"wxyz", global_mem)):
            worker.copy_from(host, source + 128)
            assert bytes(host.shm.buf[:4]) == value
        worker.release_buffer(host)
    finally:
        worker.close()


@pytest.mark.parametrize("blocks", [8, 40, 96])
def test_standard_submission_uses_all_cores(tmp_path, st_platform, blocks):
    _, sizes, _ = fixed_layout()
    compiler = KernelCompiler(st_platform)
    source = Path(__file__).parent / "direct_launch/kernels"
    kernel = compiler.compile_incore(
        str(source / "matmul_add.cpp"),
        core_type="",
        extra_include_dirs=compiler.get_orchestration_include_dirs("host_build_graph"),
    )
    callable_ = ChipCallable.build(
        signature=[D.IN, D.IN, D.OUT, D.OUT, D.IN],
        func_name="aicpu_orchestration_entry",
        binary=compiler.compile_orchestration("host_build_graph", str(source / "matmul_add_orch.cpp")),
        children=[(0, CoreCallable.build(signature=[D.IN, D.IN, D.OUT, D.OUT], binary=kernel))],
    )
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    worker.init()
    try:
        handle = worker.register(callable_)
        a = torch.arange(1024, dtype=torch.float32).reshape(32, 32) % 7 - 3
        b = torch.eye(32, dtype=torch.float32) * 2
        values = [a, b, torch.full((blocks, 32, 32), -1.0), torch.full((blocks, 2), -1.0)]
        args = TaskArgs()
        for index, value in enumerate(values):
            buffer = worker.get_base_addr(coreid=index)
            worker.copy_to(buffer, value)
            args.add_tensor(
                buffer.tensor(shapes=tuple(value.shape), dtype=DataType.FLOAT32),
                TensorArgType.INPUT if index < 2 else TensorArgType.OUTPUT_EXISTING,
            )
        args.add_scalar(blocks)
        config = CallConfig()
        config.output_prefix = str(tmp_path)
        worker.submit(handle, args, config).wait()
        for index in (2, 3):
            worker.copy_from(values[index], worker.get_base_addr(coreid=index))
        expected = torch.stack([a @ b + float(block + 1) for block in range(blocks)])
        torch.testing.assert_close(values[2], expected, rtol=0, atol=0)
        assert values[3].to(torch.int64).tolist() == [[block % len(sizes), 0] for block in range(blocks)]
        global_window = worker.get_base_addr(memory="global")
        global_record = json.loads((tmp_path / "wse_sram.json").read_text())["global_memory"]
        assert global_record["base"] == global_window.base
        assert global_record["bytes"] == global_window.nbytes
        rows = [json.loads(line) for line in (tmp_path / "wse_events.jsonl").read_text().splitlines()]
        assert rows[0]["core_count"] == len(sizes)
        events = rows[1:]
        assert {(e["core"], e["pe"]) for e in events if e["kind"] == 2} == {
            (core, pe) for core in range(len(sizes)) for pe in range(4)
        }
        assert sum(e["kind"] == 6 for e in events) == 1
        assert sorted(e["block"] for e in events if e["kind"] == 5) == list(range(blocks))
    finally:
        worker.close()
