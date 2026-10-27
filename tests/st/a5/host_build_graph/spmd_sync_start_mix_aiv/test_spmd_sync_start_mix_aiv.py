#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A sync-start cohort of Mix blocks that have no AIC subtask.

This shape is the one place a cohort can defeat the obvious check. A core must
not hold two blocks of one cohort -- the second would wait in a slot the first
cannot release, because the first is spinning in the kernel's barrier with its
core's whole scheduling loop stopped behind it. The check for "do I already
have one" is a scan of the claiming core's slots, and for a Mix cohort the
claimer is the cluster's AIC.

For every other Mix shape that scan is trivially right: the AIC subtask of a
block goes into the coordinator's own slots, so a coordinator holding a block
can see it there. A Mix task with two AIV subtasks and no AIC is still MIX --
the flag means "more than one subtask", not "has an AIC" -- and its blocks land
entirely on the cluster's Mix slots. A coordinator scanning only its own slots
would find nothing and take a second block of the same cohort.

`test_cohort_blocks_land_on_distinct_clusters` is what sees that. The result
alone cannot: with no barrier in the kernel, two blocks sharing a cluster
simply run one after the other and stamp the same answer.
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
ORCHESTRATION_SOURCE = os.path.join(HERE, "kernels", "orchestration", "spmd_sync_start_mix_aiv_orch.cpp")
# Both subtasks wait for every subtask of every block, so this case tests
# co-residency and not only which clusters the blocks reached.
BARRIER_DIR = os.path.join(HERE, "..", "spmd_cohort_saturation", "kernels", "aiv")
KERNEL_LANE0 = os.path.join(BARRIER_DIR, "kernel_cohort_barrier.cpp")
KERNEL_LANE1 = os.path.join(BARRIER_DIR, "kernel_cohort_barrier_lane1.cpp")
KERNEL_SOURCE = KERNEL_LANE0

FLOATS_PER_CACHE_LINE = 16
LANES_PER_BLOCK = 2
ARRIVED = 1.0
POISON = -99.0
# Measured in spmd_cohort_saturation: about a second before a lane gives up.
SPIN_LIMIT = 200000000
# Every block of a cohort has to be resident at once, so the width is bounded by
# the narrowest topology this runs on: eight blocks need sixteen AIV cores, the
# count of the smallest box here. Worker demand is checked against the same
# number, and a cohort past it is refused at bind rather than run.
COHORT_BLOCKS = 8


@pytest.mark.scheduler_mode("resident")
@scene_test(level=2, runtime="host_build_graph")
class TestSpmdSyncStartMixAiv(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spmd_sync_start_mix_aiv_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.IN, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "../spmd_cohort_saturation/kernels/aiv/kernel_cohort_barrier.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.IN, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 1,
                "source": "../spmd_cohort_saturation/kernels/aiv/kernel_cohort_barrier_lane1.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.IN, D.IN, D.IN, D.IN],
            },
        ],
    }

    CASES = [
        {
            "name": "mix_aiv_cohort_8",
            "platforms": ["a5sim", "a5"],
            "params": {"blocks": COHORT_BLOCKS},
        },
    ]

    def generate_args(self, params):
        lanes = params["blocks"] * LANES_PER_BLOCK
        return TaskArgsBuilder(
            TensorArg("out", torch.full((2 * lanes * FLOATS_PER_CACHE_LINE,), -1.0, dtype=torch.float32)),
            Scalar("blocks", params["blocks"]),
            Scalar("spin_limit", SPIN_LIMIT),
        )

    def compute_golden(self, args, params):
        lanes = params["blocks"] * LANES_PER_BLOCK
        for lane in range(lanes):
            # Arrival half: every subtask reaches its flag.
            args.out[lane * FLOATS_PER_CACHE_LINE] = ARRIVED
            # Result half: each subtask stamps its block, so a block's index
            # appears once per lane. A lane that gave up waiting writes POISON
            # instead, which no golden can contain.
            args.out[(lanes + lane) * FLOATS_PER_CACHE_LINE] = float(lane // LANES_PER_BLOCK)


def _build_callable(platform):
    compiler = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root()
    include_dirs = list(compiler.get_orchestration_include_dirs(RUNTIME)) + [
        str(compiler.project_root / "src" / "common")
    ]

    def incore(source):
        binary = compiler.compile_incore(
            source_path=source,
            core_type="aiv",
            pto_isa_root=pto_isa_root,
            extra_include_dirs=include_dirs,
        )
        if not platform.endswith("sim"):
            binary = extract_text_section(binary)
        return CoreCallable.build(
            signature=[
                ArgDirection.INOUT,
                ArgDirection.IN,
                ArgDirection.IN,
                ArgDirection.IN,
                ArgDirection.IN,
            ],
            binary=binary,
        )

    orchestration = compiler.compile_orchestration(runtime_name=RUNTIME, source_path=ORCHESTRATION_SOURCE)
    return ChipCallable.build(
        signature=[ArgDirection.INOUT, ArgDirection.IN, ArgDirection.IN],
        func_name="aicpu_orchestration_entry",
        binary=orchestration,
        children=[(0, incore(KERNEL_LANE0)), (1, incore(KERNEL_LANE1))],
    )


def _find_swimlane_records(output_prefix: Path) -> Path:
    matches = sorted(output_prefix.rglob("chip_swimlane_records.json"))
    assert matches, f"no chip_swimlane_records.json under {output_prefix}"
    return matches[-1]


@pytest.mark.platforms(["a5sim", "a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
def test_cohort_blocks_land_on_distinct_clusters(st_platform, st_device_ids, tmp_path):
    """No cluster runs two blocks of one cohort.

    Unlike the spread of an ordinary SPMD task, this is exact rather than a
    floor. An ordinary task's blocks are distributed by whoever frees a slot
    first, so the same graph reaches a different number of cores from run to
    run. A cohort's are governed by a rule -- one block per claiming core --
    which either holds or does not.

    Each block occupies both AIV lanes of the cluster that claimed it, so a
    cohort of N blocks that respected the rule covers 2N distinct AIV cores. A
    coordinator that took two blocks would put them on the same two cores in
    different Mix slots, and the count would come up short by two.
    """
    blocks = COHORT_BLOCKS
    worker = Worker(level=2, platform=st_platform, runtime=RUNTIME, device_id=int(st_device_ids[0]))
    buffer = None
    try:
        handle = worker.register(_build_callable(st_platform))
        worker.init()

        lanes = blocks * LANES_PER_BLOCK
        slots = 2 * lanes * FLOATS_PER_CACHE_LINE
        host_out = torch.full((slots,), -1.0, dtype=torch.float32)
        buffer = worker.malloc(host_out.nbytes)
        worker.copy_to(buffer, host_out)

        args = TaskArgs()
        args.add_tensor(buffer.tensor(shapes=(slots,), dtype=DataType.FLOAT32), TensorArgType.INOUT)
        args.add_scalar(blocks)
        args.add_scalar(SPIN_LIMIT)

        output_prefix = tmp_path / "cohort_placement"
        output_prefix.mkdir(parents=True, exist_ok=True)
        config = CallConfig()
        config.enable_chip_swimlane = 3
        config.output_prefix = str(output_prefix)
        worker.run(handle, args, config)

        worker.copy_from(host_out, buffer)
        stamped = host_out[lanes * FLOATS_PER_CACHE_LINE : 2 * lanes * FLOATS_PER_CACHE_LINE : FLOATS_PER_CACHE_LINE]
        poisoned = int((stamped == POISON).sum())
        assert poisoned == 0, (
            f"{poisoned} of the cohort's {lanes} subtasks gave up waiting for the others, so the cohort was never "
            f"resident all at once: {stamped.tolist()}"
        )
        expected = torch.tensor([float(i // LANES_PER_BLOCK) for i in range(lanes)])
        assert torch.equal(stamped, expected), f"not every block/lane stamped its slot: {stamped.tolist()}"

        raw = json.loads(_find_swimlane_records(output_prefix).read_text())
        rows = raw["aicore_tasks"]
        assert rows, "no AICore task records: the run produced no per-block trace"
        subtasks = blocks * LANES_PER_BLOCK
        assert len(rows) == subtasks, f"expected {subtasks} per-subtask records, got {len(rows)}"

        core_types = raw["metadata"]["core_types"]
        aiv_count = sum(1 for kind in core_types if kind == "aiv")
        assert aiv_count >= subtasks, (
            f"this topology has {aiv_count} AIV cores, fewer than the {subtasks} a cohort of {blocks} blocks "
            "needs; such a cohort is refused at bind rather than run, so this case cannot execute here"
        )
        cores = {int(row[0]) for row in rows}
        assert len(cores) == subtasks, (
            f"a cohort of {blocks} blocks covered {len(cores)} AIV cores, not {subtasks}: {sorted(cores)}. "
            "Two blocks sharing a cluster is what this looks like, and it is what a coordinator does when it "
            "cannot see the blocks it already placed on its cluster's Mix slots."
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
