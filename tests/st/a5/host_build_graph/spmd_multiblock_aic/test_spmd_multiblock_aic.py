#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""An AIC-only task of many blocks, dispatched by the resident AICore scheduler.

Every block stamps its own index and the task's block count into a slot of its
own, so the golden checks that all logical blocks execute exactly once.
The resident marker independently checks the scheduler selected for the run.

`test_spmd_per_block_trace` checks that every logical block has its own AIV
trace record, including when a core reuses its dispatch slots. It enables the
chip swimlane itself so this coverage does not depend on a command-line flag.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    ChipCallable,
    CoreCallable,
    DataType,
    TaskArgs,
    TensorArgType,
)
from simpler.task_interface import ArgDirection as D
from simpler.worker import Worker

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

HERE = os.path.dirname(os.path.abspath(__file__))
RUNTIME = "host_build_graph"
ORCHESTRATION_SOURCE = os.path.join(HERE, "kernels", "orchestration", "spmd_multiblock_aic_orch.cpp")
KERNEL_SOURCE = os.path.join(HERE, "kernels", "aic", "kernel_stamp_block.cpp")

BLOCK_NUM = 8
FLOATS_PER_CACHE_LINE = 16
TRACE_BLOCKS = 16


@pytest.mark.scheduler_mode("resident")
@scene_test(level=2, runtime="host_build_graph")
class TestSpmdMultiblockAic(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spmd_multiblock_aic_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.IN, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aic/kernel_stamp_block.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            # The same source on the other core type: the kernel decides what to
            # write from its block index alone, so nothing in it is AIC-specific.
            {
                "func_id": 1,
                "source": "kernels/aic/kernel_stamp_block.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
        ],
    }

    CASES = [
        # Eight blocks is already past what one cluster holds. The AIV case is
        # the only multi-block AIV coverage there is, and it is claimed by an
        # AIV rather than by a cluster's AIC -- a different path to the same
        # queue.
        {
            "name": "aic_block_num_8",
            "platforms": ["a5sim", "a5"],
            "params": {"blocks": 8, "aiv": 0},
        },
        {
            "name": "aiv_block_num_16",
            "platforms": ["a5sim", "a5"],
            "params": {"blocks": 16, "aiv": 1},
        },
        # More blocks than any topology has cores of one type, so they can only
        # complete in waves. Before worker demand narrowed to sync-start this
        # was rejected at bind.
        {
            "name": "aiv_block_num_128_waves",
            "platforms": ["a5sim", "a5"],
            "manual": ["a5"],
            "params": {"blocks": 128, "aiv": 1},
        },
    ]

    def generate_args(self, params):
        # Each block owns a cache line, and -1 marks a slot no block wrote.
        blocks = params["blocks"]
        out = torch.full((blocks * FLOATS_PER_CACHE_LINE,), -1.0, dtype=torch.float32)
        return TaskArgsBuilder(TensorArg("out", out), Scalar("blocks", blocks), Scalar("aiv", params["aiv"]))

    def compute_golden(self, args, params):
        blocks = params["blocks"]
        for block in range(blocks):
            base = block * FLOATS_PER_CACHE_LINE
            args.out[base] = float(block)
            args.out[base + 1] = float(blocks)


def _build_callable(platform):
    compiler = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root()
    include_dirs = list(compiler.get_orchestration_include_dirs(RUNTIME)) + [
        str(compiler.project_root / "src" / "common")
    ]

    def incore(core_type):
        binary = compiler.compile_incore(
            source_path=KERNEL_SOURCE,
            core_type=core_type,
            pto_isa_root=pto_isa_root,
            extra_include_dirs=include_dirs,
        )
        # Onboard loads raw code, so the ELF wrapper must come off first; sim
        # consumes the ELF as-is. scene_test.py makes the same split.
        if not platform.endswith("sim"):
            binary = extract_text_section(binary)
        return CoreCallable.build(signature=[ArgDirection.INOUT], binary=binary)

    orchestration = compiler.compile_orchestration(runtime_name=RUNTIME, source_path=ORCHESTRATION_SOURCE)
    return ChipCallable.build(
        signature=[ArgDirection.INOUT, ArgDirection.IN, ArgDirection.IN],
        func_name="aicpu_orchestration_entry",
        binary=orchestration,
        children=[(0, incore("aic")), (1, incore("aiv"))],
    )


def _find_swimlane_records(output_prefix: Path) -> Path:
    matches = sorted(output_prefix.rglob("chip_swimlane_records.json"))
    assert matches, f"no chip_swimlane_records.json under {output_prefix}"
    return matches[-1]


@pytest.mark.platforms(["a5sim", "a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
def test_spmd_per_block_trace(st_platform, st_device_ids, tmp_path):
    """Every logical block has its own result and AIV trace record.

    Ordinary SPMD blocks need not be co-resident. A core can finish and claim
    again before another core participates, so there is no distinct-core floor.
    The HbgAdmission unit tests control the claim order to verify visibility
    across directory shards independently of hardware timing.
    """
    blocks = TRACE_BLOCKS
    worker = Worker(level=2, platform=st_platform, runtime=RUNTIME, device_id=int(st_device_ids[0]))
    buffer = None
    try:
        handle = worker.register(_build_callable(st_platform))
        worker.init()

        slots = blocks * FLOATS_PER_CACHE_LINE
        host_out = torch.full((slots,), -1.0, dtype=torch.float32)
        buffer = worker.malloc(host_out.nbytes)
        worker.copy_to(buffer, host_out)

        args = TaskArgs()
        args.add_tensor(buffer.tensor(shapes=(slots,), dtype=DataType.FLOAT32), TensorArgType.INOUT)
        args.add_scalar(blocks)
        args.add_scalar(1)  # AIV, the wider core type on every topology

        output_prefix = tmp_path / "spmd_spread"
        output_prefix.mkdir(parents=True, exist_ok=True)
        config = CallConfig()
        config.enable_chip_swimlane = 3
        config.output_prefix = str(output_prefix)
        worker.run(handle, args, config)

        worker.copy_from(host_out, buffer)
        stamped = host_out[0:slots:FLOATS_PER_CACHE_LINE]
        assert torch.equal(stamped, torch.arange(blocks, dtype=torch.float32)), (
            f"not every block stamped its own slot: {stamped.tolist()}"
        )

        raw = json.loads(_find_swimlane_records(output_prefix).read_text())
        rows = raw["aicore_tasks"]
        assert rows, "no AICore task records: the run produced no per-block trace"
        # One record per block, all of them the same task. Before the trace
        # index carried a block, every block of a task wrote the same cell and
        # this would be a single row.
        assert len(rows) == blocks, f"expected {blocks} per-block records, got {len(rows)}"
        assert len({int(row[1]) for row in rows}) == 1, (
            f"records span more than one task: {sorted({int(row[1]) for row in rows})}"
        )

        core_types = raw["metadata"]["core_types"]
        for row in rows:
            core = int(row[0])
            assert 0 <= core < len(core_types), f"trace names invalid core {core}"
            assert core_types[core] == "aiv", f"AIV block ran on {core_types[core]} core {core}"
    finally:
        # A poisoned run still owns its device memory, so free() would raise and
        # bury whatever actually failed. Closing the worker releases it.
        if buffer is not None:
            try:
                worker.free(buffer)
            except Exception:  # noqa: BLE001
                pass
        worker.close()


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
