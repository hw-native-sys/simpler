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
own, so the golden is sensitive to both. A kernel that wrote one fixed cell
would pass whether all eight blocks ran or one block ran eight times, which is
what the case this replaces actually did: it wrote `out[0] = 42` from two blocks
and could not tell them apart.

The `resident` marker is the other half. Until SPMD reached the resident
scheduler a task like this sent the whole graph to the legacy AICPU scheduler,
and both schedulers compute the same answer -- so a correct result proves
nothing about which one ran.

`test_spmd_blocks_reach_past_one_shard` below adds the assertion neither of
those can make: that the blocks ran on different cores. It turns the chip
swimlane on itself rather than inheriting a command-line flag, because an
assertion that evaporates when a flag is absent guards nothing.
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
# Wide enough that an uneven spread still clears one shard's worth of owners.
SPREAD_BLOCKS = 16

# How far one publisher's work can travel without the SPMD queue: a ready task
# lands in one inbox, and a thief scans only its own shard of
# SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD owners. Blocks reaching more cores
# than that is the property the global queue exists for, and it is the same
# number on every topology -- which is what lets this threshold be a constant
# while the ceiling below comes from the run.
OWNERS_PER_SHARD = 7


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
def test_spmd_blocks_reach_past_one_shard(st_platform, st_device_ids, tmp_path):
    """The blocks of one task reach more cores than one directory shard holds.

    This is what the SPMD queue is for. A ready task published the ordinary way
    lands in a single inbox, and the only cores that can reach it are that
    inbox's owner plus the peers sharing its directory shard -- seven of them.
    A task of many blocks cannot spread that way, which is why its blocks are
    claimed from a queue every core can see instead of being popped whole.

    AIV-only and sixteen blocks wide, for margin: the spread is a floor and not
    an equality, because nothing promises an even one. A core frees a slot and
    claims again while a peer has not claimed at all, so the same graph was
    measured over eight cores once and seven the next time. On the narrower core
    type there are only eight cores to begin with, which leaves no room between
    "spread" and "confined"; on the wider one the margin is most of the type.

    The result assertion the scene-test cases make cannot see any of this:
    blocks each stamping their own slot look identical whether they ran on
    sixteen cores or on one.
    """
    blocks = SPREAD_BLOCKS
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
        aiv_count = sum(1 for kind in core_types if kind == "aiv")
        assert aiv_count > OWNERS_PER_SHARD, (
            f"this topology has {aiv_count} AIV cores, not enough for a shard's worth to be exceeded; "
            "the property is not observable here"
        )
        cores = {int(row[0]) for row in rows}
        assert len(cores) > OWNERS_PER_SHARD, (
            f"{blocks} blocks reached {len(cores)} of {aiv_count} AIV cores, within the {OWNERS_PER_SHARD} "
            f"one directory shard holds: {sorted(cores)}. A per-core queue could have done this much, so this "
            "run does not show the blocks leaving the shard that would have confined them."
        )
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
