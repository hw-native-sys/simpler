#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Predicate readiness and false-dispatch retirement over fixed SRAM views."""

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
CALLABLE = {
    "orchestration": {
        "source": "kernels/predicate/orchestration/predicated_dispatch_orch.cpp",
        "function_name": "aicpu_orchestration_entry",
        "signature": [D.INOUT, D.INOUT, D.INOUT],
    },
    "incores": [
        {
            "func_id": 0,
            "name": "WRITE_CONST",
            "source": "kernels/predicate/aic/kernel_write_const.cpp",
            "core_type": "",
            "signature": [D.INOUT],
        },
        {
            "func_id": 1,
            "name": "COPY_FIRST",
            "source": "kernels/predicate/aic/kernel_copy_first.cpp",
            "core_type": "",
            "signature": [D.IN, D.INOUT],
        },
        {
            "func_id": 2,
            "name": "CLOBBER",
            "source": "kernels/predicate/aic/kernel_clobber.cpp",
            "core_type": "",
            "signature": [D.INOUT],
        },
        {
            "func_id": 3,
            "name": "WRITE_GATE",
            "source": "kernels/predicate/aic/kernel_write_gate.cpp",
            "core_type": "",
            "signature": [D.INOUT],
        },
    ],
}


@lru_cache
def build_predicate(platform):
    compiler = KernelCompiler(platform)
    here = Path(__file__).parent
    children = []
    for spec in CALLABLE["incores"]:
        code = compiler.compile_incore(
            str(here / spec["source"]),
            core_type=spec["core_type"],
            extra_include_dirs=compiler.get_orchestration_include_dirs("host_build_graph"),
        )
        children.append((spec["func_id"], CoreCallable.build(signature=spec["signature"], binary=code)))
    orch = CALLABLE["orchestration"]
    return ChipCallable.build(
        signature=orch["signature"],
        func_name=orch["function_name"],
        binary=compiler.compile_orchestration("host_build_graph", str(here / orch["source"])),
        children=children,
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_predicate_fixed_sram(tmp_path, st_platform, enabled):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    worker.init()
    try:
        handle = worker.register(build_predicate(st_platform))
        values = [torch.full((16,), -1.0), torch.full((16,), -1.0), torch.full((16,), -1, dtype=torch.int32)]
        windows, args = [], TaskArgs()
        for index, value in enumerate(values):
            window = layout_view(worker, value.nbytes, core=index)
            copy_to(worker, window, value)
            args.add_tensor(
                window.tensor(shapes=(16,), dtype=DataType.INT32 if index == 2 else DataType.FLOAT32),
                TensorArgType.INOUT,
            )
            windows.append(window)
        args.add_scalar(2 if enabled else 1)
        config = CallConfig()
        config.output_prefix = str(tmp_path)
        worker.run(handle, args, config)
        for value, window in zip(values, windows, strict=True):
            copy_from(worker, value, window)
        expected = 999.0 if enabled else 42.0
        assert values[0][0].item() == values[1][0].item() == expected
        assert values[2][0].item() == int(enabled)
        assert all(torch.all(value[1:] == -1) for value in values)
        rows = [json.loads(line) for line in (tmp_path / "wse_events.jsonl").read_text().splitlines()]
        events = rows[1:]
        done = {e["task"]: e["seq"] for e in events if e["kind"] == 6}
        assert len(done) == rows[0]["expected_tasks"] == 4
        skipped = [e for e in events if e["kind"] == 11]
        assert len(skipped) == (0 if enabled else 1)
        assert done[0] < done[2] < done[3]
        assert done[1] < done[2]
        for window in windows:
            release_view(worker, window)
        worker.unregister(handle)
    finally:
        worker.close()
