#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A Mix cohort waits out the whole backlog in front of it, not one task.

spmd_cohort_saturation runs cohorts back to back with nothing else in the
graph, so every core a cohort needs is either idle or holding one of its own
blocks. The question here is the other one: what an unrelated queue of
ordinary work does to a cohort that needs those same cores.

A Mix block's AIV subtasks wait in their cores' Mix slots, and the executor
drains a core's own dispatch slots before it ever scans its Mix slots. So the
subtask does not wait for the task that is running -- it waits for every
ordinary task that core can still claim. The delay is the size of the backlog,
and the backlog is unbounded.

That is why the ordinary work is a queue per AIV rather than one task each,
and why each task holds its core for a measured duration rather than a spin
count. An earlier version of this case did neither, and it reproduced nothing
on silicon: two microsecond-long tasks per core leave a Mix subtask waiting
for almost nothing, so the case was measuring whether the delay existed rather
than how long it was. With a backlog the measurement is the same on both
platforms --

    simulation (8 AIC / 16 AIV):  latest start 1,100,044 us
    a5 hardware (32 AIC / 64 AIV): latest start 1,100,078 us

-- about 20 x 50 ms either way, which is the backlog, not the hardware.

The bound between the two readings is what makes this discriminate rather than
merely fail. Today a subtask waits out the whole queue; once a core's slots
are one FIFO queue and a core that owes a Mix subtask stops claiming more, it
waits out only the task already running. BARRIER_WAIT_MS sits between them, so
the case goes green on the fix instead of needing a new number.

Co-residency itself is decided from the swimlane rather than from whether the
barrier timed out -- see the conftest fixture for why.
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
MAX_ORDINARY = 1536
MAX_LANES_PER_COHORT = 320
COHORT_BASE = MAX_ORDINARY
SLOTS_PER_COHORT = 2 * MAX_LANES_PER_COHORT

RAN = 1.0
ARRIVED = 1.0
POISON = -99.0
UNWRITTEN = -1.0
# How long a lane waits for its siblings, in milliseconds; the kernel
# converts it with the platform's own tick rate. See spmd_cohort_saturation
# for why this is a duration and not a spin count.
# How long a Mix cohort's subtasks may wait for each other before the case
# calls them not co-resident.
BARRIER_WAIT_MS = 500
# How long one ordinary task holds its core, and how many of them are queued
# per AIV. The product is what matters, and the two bounds it has to sit
# between are what make this case discriminate rather than merely fail:
#
#   today            a Mix subtask waits out the WHOLE backlog, because a core
#                    drains its own dispatch slots before it ever scans its Mix
#                    slots -- about ORDINARY_PER_AIV * ORDINARY_HOLD_MS
#   once slots are   it waits out the one task already running, because a core
#   one FIFO queue   that owes a Mix subtask stops claiming more -- about
#                    ORDINARY_HOLD_MS
#
# So BARRIER_WAIT_MS is set above the second and below the first: 50 ms of
# unavoidable wait passes, 1000 ms of backlog does not.
ORDINARY_HOLD_MS = 50
ORDINARY_PER_AIV = 20


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
        args.add_scalar(ORDINARY_HOLD_MS)
        args.add_scalar(BARRIER_WAIT_MS)
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


# The Mix cohort is the second one, so a run with only the AIC cohort has
# nothing in a Mix slot to starve and passes. Everything from two cohorts up
# carries the defect; three is kept because it was the shape that first showed
# it, back when the construction needed the AIV-only cohort's own-slot blocks
# to supply the backlog that ordinary work now supplies directly.
_STARVES_THE_MIX_COHORT = pytest.mark.xfail(
    strict=True,
    reason=(
        "Known defect: a Mix cohort's AIV subtasks wait out the entire backlog of ordinary work in "
        "front of them, because a core drains its own dispatch slots before it ever scans its Mix "
        "slots. Measured at about 20 x 50 ms of backlog on both a5sim and a5 silicon, against a 500 ms "
        "bound. The fix is the dual-issue slot model in tests/docs/: one FIFO queue per core, and a "
        "core that owes a Mix subtask stops claiming more work. strict, so that whoever lands it is "
        "told to delete this marker rather than leaving a passing xfail behind."
    ),
)


@pytest.mark.platforms(["a5sim", "a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
@pytest.mark.parametrize(
    "cohort_count",
    [
        1,
        pytest.param(2, marks=_STARVES_THE_MIX_COHORT),
        pytest.param(3, marks=_STARVES_THE_MIX_COHORT),
    ],
)
def test_cohorts_assemble_against_competing_ordinary_work(
    st_platform, st_device_ids, tmp_path, cohort_count, assert_cohorts_were_coresident
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

        # A backlog on the AIVs, not merely one task each. One task would make
        # the Mix subtask wait once and then run; what this case is about is
        # that today it waits for the whole queue, because a core drains its
        # own dispatch slots before it scans its Mix slots. The backlog is
        # what turns that difference into a number.
        ordinary_count = min(ORDINARY_PER_AIV * aiv_count, MAX_ORDINARY)
        real_prefix = tmp_path / f"run_{cohort_count}"
        real_prefix.mkdir(parents=True, exist_ok=True)
        host_out = _run(
            worker, handle, cohort_count=cohort_count, ordinary_count=ordinary_count, output_prefix=real_prefix
        )

        marks = host_out[0 : ordinary_count * FLOATS_PER_CACHE_LINE : FLOATS_PER_CACHE_LINE]
        assert torch.equal(marks, torch.full((ordinary_count,), RAN)), (
            f"an ordinary task never ran, so the cohorts were not actually competing: {marks.tolist()}"
        )

        # Every lane ran: a subtask that was never dispatched is a different
        # failure from one that ran without its siblings, and the swimlane
        # check below would report the first as a missing task rather than as
        # a co-residency verdict.
        for cohort in range(cohort_count):
            name, width, lanes_per_block = _cohort_shape(cohort, aic_count, aiv_count)
            lane_count = width * lanes_per_block
            arrive_base = COHORT_BASE + cohort * SLOTS_PER_COHORT
            arrivals = host_out[
                arrive_base * FLOATS_PER_CACHE_LINE : (arrive_base + lane_count)
                * FLOATS_PER_CACHE_LINE : FLOATS_PER_CACHE_LINE
            ]
            missing = int((arrivals != ARRIVED).sum())
            assert missing == 0, (
                f"cohort {cohort} ({name}, {width} blocks x {lanes_per_block} lanes): {missing} of its lanes never "
                f"ran at all while {ordinary_count} ordinary tasks competed, so the cohort was never fully "
                f"dispatched: {arrivals.tolist()}"
            )

        # And they overlapped. Decided from the run's own record rather than
        # from whether the barrier timed out: the barrier races a wall clock,
        # so its verdict moved with machine load -- this case reported the
        # defect alone and reported it absent after the rest of the suite, on
        # one build. Whether two intervals intersect does not move.
        widths = [
            width * lanes for _, width, lanes in (_cohort_shape(c, aic_count, aiv_count) for c in range(cohort_count))
        ]
        assert_cohorts_were_coresident(
            real_prefix, widths, f"{cohort_count} cohort(s) against {ordinary_count} ordinary tasks"
        )
    finally:
        worker.close()
