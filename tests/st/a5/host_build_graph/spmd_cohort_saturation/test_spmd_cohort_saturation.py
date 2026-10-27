#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Back-to-back cohorts, each as wide as the resource it runs on.

Every other sync-start case writes its slot and returns, which leaves
co-residency unobservable: what sync-start buys is that all of a task's blocks
hold cores at the same instant, and a block that never waits cannot tell
whether that held. Those cases pin placement and nothing else -- if blocks
stopped being co-resident tomorrow they would all still pass.

The blocks here wait for each other, so a cohort that does not assemble cannot
finish. The wait is bounded and a block that gives up writes a poison value, so
a failure arrives as a wrong number in a second rather than as a device-side
timeout forty-five seconds later.

Width is the point. Each cohort needs every core of its kind at once, so a core
still carrying a block of the previous cohort is a core the next one cannot
have. The cohorts are independent, so they are all ready at once and queue
behind a single head: every task boundary is a head handover under maximum
contention, repeated as many times as the case asks for.

Kinds alternate so consecutive cohorts land on different channels -- an AIC
cohort on the cores' own dispatch slots, an AIV cohort on the AIVs' own slots.
Widths come from the run, so there is no fixed golden and this is a standalone
case.
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
ORCHESTRATION_SOURCE = os.path.join(HERE, "kernels", "orchestration", "spmd_cohort_saturation_orch.cpp")
AIC_KERNEL = os.path.join(HERE, "kernels", "aic", "kernel_cohort_barrier.cpp")
AIV_KERNEL = os.path.join(HERE, "kernels", "aiv", "kernel_cohort_barrier.cpp")
AIV_LANE1_KERNEL = os.path.join(HERE, "kernels", "aiv", "kernel_cohort_barrier_lane1.cpp")
AIV_LANE2_KERNEL = os.path.join(HERE, "kernels", "aiv", "kernel_cohort_barrier_lane2.cpp")

FLOATS_PER_CACHE_LINE = 16
# Mirrors kMaxLanesPerCohort / kSlotsPerCohort in the orchestration.
MAX_LANES_PER_COHORT = 320
SLOTS_PER_COHORT = 2 * MAX_LANES_PER_COHORT
ARRIVED = 1.0
POISON = -99.0
UNWRITTEN = -1.0
# How long a block waits for its siblings before giving up. The bound is in
# iterations because a kernel has no portable wall clock, so the value is
# measured rather than derived: the positive control below gave up after 11 ms
# at 200_000 and 1.08 s at this value -- note that is a hundredfold in
# iterations for a hundredfold in time only across the whole range, so the
# relationship is not one an arithmetic guess would get right.
#
# A second is long enough that no correct cohort is ever cut short by a slow
# dispatch, and short enough that a broken one fails in seconds rather than
# waiting out the device's own forty-five second op-execute timeout. It costs
# nothing on the passing path: a cohort that assembles leaves the loop on its
# first pass.
SPIN_LIMIT = 200000000


def _build_callable(platform):
    compiler = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root()
    include_dirs = list(compiler.get_orchestration_include_dirs(RUNTIME)) + [
        str(compiler.project_root / "src" / "common")
    ]

    def incore(source, core_type):
        binary = compiler.compile_incore(
            source_path=source,
            core_type=core_type,
            pto_isa_root=pto_isa_root,
            extra_include_dirs=include_dirs,
        )
        if not platform.endswith("sim"):
            binary = extract_text_section(binary)
        signature = [
            ArgDirection.INOUT,
            ArgDirection.IN,
            ArgDirection.IN,
            ArgDirection.IN,
            ArgDirection.IN,
        ]
        return CoreCallable.build(signature=signature, binary=binary)

    orchestration = compiler.compile_orchestration(runtime_name=RUNTIME, source_path=ORCHESTRATION_SOURCE)
    return ChipCallable.build(
        signature=[ArgDirection.INOUT, ArgDirection.IN, ArgDirection.IN],
        func_name="aicpu_orchestration_entry",
        binary=orchestration,
        children=[
            (0, incore(AIC_KERNEL, "aic")),
            (1, incore(AIV_KERNEL, "aiv")),
            (2, incore(AIV_LANE1_KERNEL, "aiv")),
            (3, incore(AIV_LANE2_KERNEL, "aiv")),
        ],
    )


def _find_swimlane_records(output_prefix: Path) -> Path:
    matches = sorted(output_prefix.rglob("chip_swimlane_records.json"))
    assert matches, f"no chip_swimlane_records.json under {output_prefix}"
    return matches[-1]


def _run(worker, handle, cohort_count, output_prefix=None, oversubscribed_blocks=0):
    slots = max(cohort_count, 1) * SLOTS_PER_COHORT
    host_out = torch.full((slots * FLOATS_PER_CACHE_LINE,), UNWRITTEN, dtype=torch.float32)
    buffer = worker.malloc(host_out.nbytes)
    try:
        worker.copy_to(buffer, host_out)
        args = TaskArgs()
        args.add_tensor(
            buffer.tensor(shapes=(slots * FLOATS_PER_CACHE_LINE,), dtype=DataType.FLOAT32), TensorArgType.INOUT
        )
        args.add_scalar(cohort_count)
        args.add_scalar(SPIN_LIMIT)
        args.add_scalar(oversubscribed_blocks)
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
    """(kind name, block count, lanes per block) for one cohort, as submitted.

    Kept in step with the orchestration by hand. The order -- AIC, Mix, AIV --
    is deliberate: it is the Mix-then-AIV adjacency that lets one AIV hold an
    obligation on its Mix slot while claiming into its own.
    """
    kind = cohort % 3
    if kind == 0:
        return "AIC", aic_count, 1
    if kind == 1:
        return "Mix", aic_count, 3
    return "AIV", aiv_count, 1


def _check(host_out, cohort_count, aic_count, aiv_count):
    """Every cohort assembled, and every lane of it ran exactly once."""
    for cohort in range(cohort_count):
        name, width, lanes_per_block = _cohort_shape(cohort, aic_count, aiv_count)
        lane_count = width * lanes_per_block
        arrive_base = cohort * SLOTS_PER_COHORT
        result_base = arrive_base + MAX_LANES_PER_COHORT

        results = host_out[
            result_base * FLOATS_PER_CACHE_LINE : (result_base + lane_count) * FLOATS_PER_CACHE_LINE :
            FLOATS_PER_CACHE_LINE
        ]
        poisoned = int((results == POISON).sum())
        assert poisoned == 0, (
            f"cohort {cohort} ({name}, {width} blocks x {lanes_per_block} lanes): {poisoned} of its lanes gave up "
            f"waiting for their siblings, so the cohort was never co-resident: {results.tolist()}"
        )
        # Every lane of a block stamps that block's index, so the expected
        # pattern repeats each index `lanes_per_block` times.
        expected = torch.arange(width, dtype=torch.float32).repeat_interleave(lanes_per_block)
        assert torch.equal(results, expected), (
            f"cohort {cohort} ({name}) did not run every lane exactly once: {results.tolist()}"
        )

        arrivals = host_out[
            arrive_base * FLOATS_PER_CACHE_LINE : (arrive_base + lane_count) * FLOATS_PER_CACHE_LINE :
            FLOATS_PER_CACHE_LINE
        ]
        assert torch.equal(arrivals, torch.full((lane_count,), ARRIVED)), (
            f"cohort {cohort} ({name}) has a lane that never arrived: {arrivals.tolist()}"
        )


@pytest.mark.platforms(["a5sim", "a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
def test_the_barrier_actually_fires(st_platform, st_device_ids, tmp_path, capfd):
    """Positive control: a task whose blocks cannot be co-resident must poison.

    Every co-residency assertion in this file rests on the barrier being live.
    A barrier that was optimized away, or whose give-up path stopped reaching
    the output, would make all of them pass for the wrong reason -- and that is
    exactly the failure the rest of the suite was blind to before these cases
    existed.

    More blocks than cores, and no sync-start, is not co-resident by
    construction: the task runs in waves. The first wave waits for siblings
    that cannot start until it finishes, so it must give up. Later waves find
    every flag already set -- an earlier wave arrives before it gives up -- and
    assemble immediately, so the signature is "some poisoned, not all".

    The elapsed time is reported because it is what the spin limit has to be
    read against: the bound is in iterations, and an iteration costs a
    different amount in simulation than on silicon.
    """
    import time

    worker = Worker(level=2, platform=st_platform, runtime=RUNTIME, device_id=int(st_device_ids[0]))
    try:
        handle = worker.register(_build_callable(st_platform))
        worker.init()
        output_prefix = tmp_path / "control"
        output_prefix.mkdir(parents=True, exist_ok=True)
        # One probe run to learn the core count, then the control itself.
        _run(worker, handle, cohort_count=1, output_prefix=output_prefix)
        raw = json.loads(_find_swimlane_records(output_prefix).read_text())
        aic_count = sum(1 for kind in raw["metadata"]["core_types"] if kind == "aic")

        blocks = 2 * aic_count
        started = time.monotonic()
        host_out = _run(worker, handle, cohort_count=0, oversubscribed_blocks=blocks)
        elapsed = time.monotonic() - started

        results = host_out[
            MAX_LANES_PER_COHORT * FLOATS_PER_CACHE_LINE :
            (MAX_LANES_PER_COHORT + blocks) * FLOATS_PER_CACHE_LINE : FLOATS_PER_CACHE_LINE
        ]
        poisoned = int((results == POISON).sum())
        print(f"\n[BARRIER-CONTROL] blocks={blocks} aic={aic_count} poisoned={poisoned} elapsed={elapsed:.2f}s")
        assert poisoned > 0, (
            f"{blocks} blocks on {aic_count} cores without sync-start cannot be co-resident, yet none of them gave "
            f"up waiting: the barrier is not live, so every co-residency assertion in this file is vacuous. "
            f"results={results.tolist()}"
        )
    finally:
        worker.close()


@pytest.mark.platforms(["a5sim", "a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
def test_one_full_width_cohort_assembles(st_platform, st_device_ids, tmp_path):
    """The premise: a cohort whose blocks wait for each other does assemble.

    Separate from the stress case below because it is the control. If blocks of
    one cohort are not co-resident, every multi-cohort assertion would fail for
    a reason that has nothing to do with handover between cohorts.
    """
    worker = Worker(level=2, platform=st_platform, runtime=RUNTIME, device_id=int(st_device_ids[0]))
    try:
        handle = worker.register(_build_callable(st_platform))
        worker.init()
        output_prefix = tmp_path / "one_cohort"
        output_prefix.mkdir(parents=True, exist_ok=True)
        host_out = _run(worker, handle, cohort_count=1, output_prefix=output_prefix)

        raw = json.loads(_find_swimlane_records(output_prefix).read_text())
        core_types = raw["metadata"]["core_types"]
        aic_count = sum(1 for kind in core_types if kind == "aic")
        aiv_count = sum(1 for kind in core_types if kind == "aiv")
        assert aic_count > 0 and aiv_count > 0, raw["metadata"]
        _check(host_out, 1, aic_count, aiv_count)
    finally:
        worker.close()


@pytest.mark.platforms(["a5sim", "a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
@pytest.mark.parametrize("cohort_count", [2, 12])
def test_back_to_back_full_width_cohorts_all_assemble(st_platform, st_device_ids, tmp_path, cohort_count):
    """Many cohorts, each needing every core of its kind, one after another.

    Two is the smallest count that has a handover at all; twelve runs the same
    handover eleven times with the kinds alternating, so a core that carries an
    obligation across a boundary gets many chances to.

    The assertion is that every cohort assembled. A cohort whose blocks are not
    co-resident leaves poison in its result slots, and the cohort after it
    inherits cores that never came free -- so the first failure tends to cascade,
    which is why the message names the cohort index and its kind.
    """
    worker = Worker(level=2, platform=st_platform, runtime=RUNTIME, device_id=int(st_device_ids[0]))
    try:
        handle = worker.register(_build_callable(st_platform))
        worker.init()
        output_prefix = tmp_path / f"cohorts_{cohort_count}"
        output_prefix.mkdir(parents=True, exist_ok=True)
        host_out = _run(worker, handle, cohort_count=cohort_count, output_prefix=output_prefix)

        raw = json.loads(_find_swimlane_records(output_prefix).read_text())
        core_types = raw["metadata"]["core_types"]
        aic_count = sum(1 for kind in core_types if kind == "aic")
        aiv_count = sum(1 for kind in core_types if kind == "aiv")
        _check(host_out, cohort_count, aic_count, aiv_count)
    finally:
        worker.close()
