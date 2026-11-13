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
    offsets = [sum(sizes[:core]) for core in range(cores)]
    return capacities, sizes, offsets


def test_standard_worker_fixed_sram_and_empty_run(st_platform):
    assert not any(
        hasattr(_task_interface, name) for name in ("_wse_malloc", "_wse_submit", "_wse_query", "_wse_configure")
    )
    assert RuntimeBuilder(st_platform).get_binaries("host_build_graph").aicpu_path is None
    capacities, sizes, offsets = fixed_layout()
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    worker.init()
    try:
        with pytest.raises(RuntimeError, match="malloc failed"):
            worker.malloc(128)
        buffer = worker.malloc(sum(sizes))
        with pytest.raises(RuntimeError, match="malloc failed"):
            worker.malloc(sum(sizes))
        for core, offset in enumerate(offsets):
            value = torch.tensor([core, core + 1], dtype=torch.int32)
            worker.copy_to(buffer, value, dst_offset=offset)
        for core, offset in enumerate(offsets):
            value = torch.empty(2, dtype=torch.int32)
            worker.copy_from(value, buffer, src_offset=offset)
            assert value.tolist() == [core, core + 1]
        assert worker._chip_worker.committed_device_memory == sum(capacities)
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
        worker.free(buffer)
        with pytest.raises((ValueError, RuntimeError), match="freed|released|stale|closed"):
            worker.copy_from(torch.empty(2, dtype=torch.int32), buffer)
        again = worker.malloc(sum(sizes))
        worker.free(again)
    finally:
        worker.close()


@pytest.mark.parametrize("blocks", [8, 40, 96])
def test_standard_submission_uses_all_cores(tmp_path, st_platform, blocks):
    capacities, sizes, offsets = fixed_layout()
    compiler = KernelCompiler(st_platform)
    source = Path(__file__).parent / "direct_launch/kernels"
    kernel = compiler.compile_incore(
        str(source / "mixed.cpp"),
        core_type="",
        extra_include_dirs=compiler.get_orchestration_include_dirs("host_build_graph"),
    )
    callable_ = ChipCallable.build(
        signature=[D.IN, D.IN, D.OUT, D.OUT, D.IN],
        func_name="aicpu_orchestration_entry",
        binary=compiler.compile_orchestration("host_build_graph", str(source / "mixed_orch.cpp")),
        children=[(0, CoreCallable.build(signature=[D.IN, D.IN, D.OUT, D.OUT], binary=kernel))],
    )
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    worker.init()
    try:
        buffer = worker.malloc(sum(sizes))
        handle = worker.register(callable_)
        a = torch.arange(1024, dtype=torch.float32).reshape(32, 32) % 7 - 3
        b = torch.eye(32, dtype=torch.float32) * 2
        values = [a, b, torch.full((blocks, 32, 32), -1.0), torch.full((blocks, 2), -1.0)]
        args = TaskArgs()
        for index, value in enumerate(values):
            worker.copy_to(buffer, value, dst_offset=offsets[index])
            args.add_tensor(
                buffer.tensor(shapes=tuple(value.shape), dtype=DataType.FLOAT32, byte_offset=offsets[index]),
                TensorArgType.INPUT if index < 2 else TensorArgType.OUTPUT_EXISTING,
            )
        args.add_scalar(blocks)
        config = CallConfig()
        config.output_prefix = str(tmp_path)
        worker.submit(handle, args, config).wait()
        for index in (2, 3):
            worker.copy_from(values[index], buffer, src_offset=offsets[index])
        expected = torch.stack([a @ b + float(block + 1) for block in range(blocks)])
        torch.testing.assert_close(values[2], expected, rtol=0, atol=0)
        assert values[3].to(torch.int64).tolist() == [[block % len(sizes), 0] for block in range(blocks)]
        rows = [json.loads(line) for line in (tmp_path / "wse_events.jsonl").read_text().splitlines()]
        assert rows[0]["core_count"] == len(sizes)
        events = rows[1:]
        assert {(e["core"], e["pe"]) for e in events if e["kind"] == 2} == {
            (core, pe) for core in range(len(sizes)) for pe in range(4)
        }
        assert sum(e["kind"] == 6 for e in events) == 1
        assert sorted(e["block"] for e in events if e["kind"] == 5) == list(range(blocks))
        worker.free(buffer)
    finally:
        worker.close()
