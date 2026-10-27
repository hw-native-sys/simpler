#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A cohort has to assemble while ordinary tasks are taking the same cores.

spmd_cohort_saturation runs cohorts back to back with nothing else in the
graph, so every core a cohort needs is either idle or holding one of its own
blocks. The question here is the other one: a cohort needs every core of its
kind at once, and ordinary work is competing for exactly those cores.

Everything is submitted without dependencies, so it is all ready at the same
moment and the ordinary tasks are already running when the cohorts are
claimed. They occupy their cores rather than returning at once -- one that
finished immediately would free its slot before the claim pass looked, and
there would have been no competition to measure.

This is deliberately small. The alternative was to give spmd_mixed_waves' two
cohort colours real barriers, but a quarter of that palette is cohorts and
each wants the whole machine, so a thousand-task stress case would have become
a strictly serial one -- destroying the only thing it is for, which is many
shapes in flight at once. Its cohorts stay nominal and its docstring says so.
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
from simpler.worker import Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

HERE = os.path.dirname(os.path.abspath(__file__))
RUNTIME = "host_build_graph"
ORCHESTRATION_SOURCE = os.path.join(HERE, "kernels", "orchestration", "spmd_cohort_with_ordinary_orch.cpp")
BUSY_AIC = os.path.join(HERE, "kernels", "aic", "kernel_busy_mark.cpp")
BUSY_AIV = os.path.join(HERE, "kernels", "aiv", "kernel_busy_mark.cpp")
BARRIER_DIR = os.path.join(HERE, "..", "spmd_cohort_saturation", "kernels")
BARRIER_AIC = os.path.join(BARRIER_DIR, "aic", "kernel_cohort_barrier.cpp")
BARRIER_AIV_LANE0 = os.path.join(BARRIER_DIR, "aiv", "kernel_cohort_barrier.cpp")
BARRIER_AIV_LANE1 = os.path.join(BARRIER_DIR, "aiv", "kernel_cohort_barrier_lane1.cpp")
BARRIER_AIV_LANE2 = os.path.join(BARRIER_DIR, "aiv", "kernel_cohort_barrier_lane2.cpp")

FLOATS_PER_CACHE_LINE = 16
# Mirrors the slot map in the orchestration.
MAX_ORDINARY = 256
MAX_LANES_PER_COHORT = 320
COHORT_BASE = MAX_ORDINARY
SLOTS_PER_COHORT = 2 * MAX_LANES_PER_COHORT

RAN = 1.0
ARRIVED = 1.0
POISON = -99.0
UNWRITTEN = -1.0
# Measured in spmd_cohort_saturation: about a second before a lane gives up.
BARRIER_SPIN = 2000000000
# Long enough that the ordinary tasks are still holding cores when the cohorts
# are claimed, and far shorter than the barrier so a correct run assembles well
# inside the wait.
ORDINARY_SPIN = 2000000


def _build_callable(platform):
    compiler = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root()
    include_dirs = list(compiler.get_orchestration_include_dirs(RUNTIME)) + [
        str(compiler.project_root / "src" / "common")
    ]

    def incore(source, core_type, arg_count):
        binary = compiler.compile_incore(
            source_path=source,
            core_type=core_type,
            pto_isa_root=pto_isa_root,
            extra_include_dirs=include_dirs,
        )
        if not platform.endswith("sim"):
            binary = extract_text_section(binary)
        signature = [ArgDirection.INOUT] + [ArgDirection.IN] * (arg_count - 1)
        return CoreCallable.build(signature=signature, binary=binary)

    orchestration = compiler.compile_orchestration(runtime_name=RUNTIME, source_path=ORCHESTRATION_SOURCE)
    return ChipCallable.build(
        signature=[ArgDirection.INOUT] + [ArgDirection.IN] * 4,
        func_name="aicpu_orchestration_entry",
        binary=orchestration,
        children=[
            (0, incore(BARRIER_AIC, "aic", 5)),
            (1, incore(BARRIER_AIV_LANE0, "aiv", 5)),
            (2, incore(BARRIER_AIV_LANE1, "aiv", 5)),
            (3, incore(BARRIER_AIV_LANE2, "aiv", 5)),
            (4, incore(BUSY_AIC, "aic", 3)),
            (5, incore(BUSY_AIV, "aiv", 3)),
        ],
    )


def _find_swimlane_records(output_prefix: Path) -> Path:
    matches = sorted(output_prefix.rglob("chip_swimlane_records.json"))
    assert matches, f"no chip_swimlane_records.json under {output_prefix}"
    return matches[-1]


def _run(worker, handle, cohort_count, ordinary_count, output_prefix=None):
    slots = COHORT_BASE + max(cohort_count, 1) * SLOTS_PER_COHORT
    host_out = torch.full((slots * FLOATS_PER_CACHE_LINE,), UNWRITTEN, dtype=torch.float32)
    buffer = worker.malloc(host_out.nbytes)
    try:
        worker.copy_to(buffer, host_out)
        args = TaskArgs()
        args.add_tensor(
            buffer.tensor(shapes=(slots * FLOATS_PER_CACHE_LINE,), dtype=DataType.FLOAT32), TensorArgType.INOUT
        )
        args.add_scalar(cohort_count)
        args.add_scalar(ordinary_count)
        args.add_scalar(ORDINARY_SPIN)
        args.add_scalar(BARRIER_SPIN)
        config = CallConfig()
        if output_prefix is not None:
            config.enable_chip_swimlane = 3
            config.output_prefix = str(output_prefix)
        worker.run(handle, args, config)
        worker.copy_from(host_out, buffer)
        return host_out
    finally:
        try:
            worker.free(buffer)
        except Exception:  # noqa: BLE001
            pass


def _cohort_shape(cohort, aic_count, aiv_count):
    """(kind name, block count, lanes per block), matching the orchestration."""
    kind = cohort % 3
    if kind == 0:
        return "AIC", aic_count, 1
    if kind == 1:
        return "Mix", aic_count, 3
    return "AIV", aiv_count, 1


@pytest.mark.platforms(["a5sim", "a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
@pytest.mark.parametrize(
    "cohort_count",
    [
        1,
        2,
        pytest.param(
            3,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "Known defect: adding the AIV-only cohort starves the Mix cohort's AIV subtasks. "
                    "A Mix block's AIV subtask sits in that core's Mix slot, an AIV-only cohort's block "
                    "sits in its own dispatch slot, and the executor takes own slots unconditionally "
                    "before it ever scans Mix slots -- so the AIV runs the later cohort first and the "
                    "earlier one is never resident all at once. Confirmed three ways: this barrier, the "
                    "chip swimlane (latest start after earliest end), and the 1/2/3 comparison above. "
                    "strict so that whoever fixes the slot model is told to delete this marker."
                ),
            ),
        ),
    ],
)
def test_cohorts_assemble_against_competing_ordinary_work(
    st_platform, st_device_ids, tmp_path, cohort_count
):
    worker = Worker(level=2, platform=st_platform, runtime=RUNTIME, device_id=int(st_device_ids[0]))
    try:
        handle = worker.register(_build_callable(st_platform))
        worker.init()
        output_prefix = tmp_path / f"probe_{cohort_count}"
        output_prefix.mkdir(parents=True, exist_ok=True)
        # A probe run with no competition, to learn the topology the real run
        # has to size itself against.
        _run(worker, handle, cohort_count=1, ordinary_count=0, output_prefix=output_prefix)
        raw = json.loads(_find_swimlane_records(output_prefix).read_text())
        core_types = raw["metadata"]["core_types"]
        aic_count = sum(1 for kind in core_types if kind == "aic")
        aiv_count = sum(1 for kind in core_types if kind == "aiv")
        assert aic_count > 0 and aiv_count > 0, raw["metadata"]

        # Two ordinary tasks per core, so every core is busy and has another
        # queued behind it when the cohorts are claimed.
        ordinary_count = min(2 * (aic_count + aiv_count), MAX_ORDINARY)
        host_out = _run(worker, handle, cohort_count=cohort_count, ordinary_count=ordinary_count)

        marks = host_out[0 : ordinary_count * FLOATS_PER_CACHE_LINE : FLOATS_PER_CACHE_LINE]
        assert torch.equal(marks, torch.full((ordinary_count,), RAN)), (
            f"an ordinary task never ran, so the cohorts were not actually competing: {marks.tolist()}"
        )

        for cohort in range(cohort_count):
            name, width, lanes_per_block = _cohort_shape(cohort, aic_count, aiv_count)
            lane_count = width * lanes_per_block
            arrive_base = COHORT_BASE + cohort * SLOTS_PER_COHORT
            result_base = arrive_base + MAX_LANES_PER_COHORT

            # Arrivals first, deliberately. A lane that never ran and a lane
            # that ran but timed out are different failures -- the first says
            # the cohort was never fully dispatched, the second says it was
            # dispatched too far apart to overlap -- and checking results
            # first would report both as the same thing.
            arrivals = host_out[
                arrive_base * FLOATS_PER_CACHE_LINE : (arrive_base + lane_count) * FLOATS_PER_CACHE_LINE :
                FLOATS_PER_CACHE_LINE
            ]
            missing = int((arrivals != ARRIVED).sum())
            assert missing == 0, (
                f"cohort {cohort} ({name}, {width} blocks x {lanes_per_block} lanes): {missing} of its lanes never "
                f"ran at all while {ordinary_count} ordinary tasks competed, so the cohort was never fully "
                f"dispatched: {arrivals.tolist()}"
            )

            results = host_out[
                result_base * FLOATS_PER_CACHE_LINE : (result_base + lane_count) * FLOATS_PER_CACHE_LINE :
                FLOATS_PER_CACHE_LINE
            ]
            poisoned = int((results == POISON).sum())
            # Each lane also records how long it waited, in the cache line next
            # to its verdict. Without it a failure cannot be read: a lane that
            # waited out the whole limit says its siblings never came, and one
            # that stopped early says the limit was the problem.
            spins = host_out[
                result_base * FLOATS_PER_CACHE_LINE + 1 :
                (result_base + lane_count) * FLOATS_PER_CACHE_LINE + 1 : FLOATS_PER_CACHE_LINE
            ]
            assert poisoned == 0, (
                f"cohort {cohort} ({name}, {width} blocks x {lanes_per_block} lanes): every lane ran, but "
                f"{poisoned} of them gave up waiting for the others while {ordinary_count} ordinary tasks competed "
                f"for the same cores -- so they were dispatched too far apart to ever overlap. A cohort that "
                f"cannot assemble against ordinary work is a hang in any graph that mixes the two.\n"
                f"  results = {results.tolist()}\n"
                f"  spins   = {[int(s) for s in spins.tolist()]} (limit {BARRIER_SPIN})"
            )
            expected = torch.arange(width, dtype=torch.float32).repeat_interleave(lanes_per_block)
            assert torch.equal(results, expected), (
                f"cohort {cohort} ({name}) did not run every lane exactly once: {results.tolist()}"
            )
    finally:
        worker.close()
