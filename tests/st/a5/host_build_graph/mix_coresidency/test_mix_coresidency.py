#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A plain Mix task's 1C2V must be running together.

A Mix task is one task spanning an AIC and both AIVs of a cluster, and real Mix
kernels make them wait on each other -- the fused attention kernel has its AIC
publish a flag the AIVs block on and its AIVs publish flags the AIC blocks on.
So a Mix task carries a co-residency requirement at width one: the same thing
sync-start asks for across blocks, asked for across subtasks instead. It is not
conditional on sync-start, and nothing in the submit API lets a caller opt out
of it.

Every existing Mix case writes its slot and returns, so none of them can tell
whether that requirement holds; spmd_mix_block_closes_last deliberately makes
the subtasks finish far apart, which is right for what it pins (the inner level
of completion counting) and blind to this.

The second case is the one with teeth. Placement is atomic but execution is
not: a subtask sits in its slot until its core reaches it, so putting the AIVs
to work first makes the Mix task's AIC subtask start while its AIV subtasks are
still queued behind something else. The requirement has to hold there too.
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
ORCHESTRATION_SOURCE = os.path.join(HERE, "kernels", "orchestration", "mix_coresidency_orch.cpp")
# The barrier the cohort cases use, with its own lane constant per file.
BARRIER_DIR = os.path.join(HERE, "..", "spmd_cohort_saturation", "kernels")
BARRIER_AIC = os.path.join(BARRIER_DIR, "aic", "kernel_cohort_barrier.cpp")
BARRIER_AIV_LANE0 = os.path.join(BARRIER_DIR, "aiv", "kernel_cohort_barrier.cpp")
BARRIER_AIV_LANE1 = os.path.join(BARRIER_DIR, "aiv", "kernel_cohort_barrier_lane1.cpp")
BARRIER_AIV_LANE2 = os.path.join(BARRIER_DIR, "aiv", "kernel_cohort_barrier_lane2.cpp")

FLOATS_PER_CACHE_LINE = 16
# Mirrors the slot map in the orchestration.
MAX_OCCUPY = 160
OCCUPY_ARRIVE_BASE = 0
OCCUPY_BASE = OCCUPY_ARRIVE_BASE + MAX_OCCUPY
ARRIVE_BASE = OCCUPY_BASE + MAX_OCCUPY
RESULT_BASE = ARRIVE_BASE + 8
LANES_PER_BLOCK = 3
TOTAL_SLOTS = RESULT_BASE + 8

ARRIVED = 1.0
POISON = -99.0
UNWRITTEN = -1.0
# How long a lane waits for its siblings, in milliseconds; the kernel
# converts it with the platform's own tick rate. See spmd_cohort_saturation
# for why this is a duration and not a spin count.
BARRIER_WAIT_MS = 1000
# Long enough that the AIVs are still inside it when the Mix task is placed,
# and far shorter than the barrier so a correct run still assembles well
# within the wait.
OCCUPY_SPIN = 2000000


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
        signature=[ArgDirection.INOUT, ArgDirection.IN, ArgDirection.IN, ArgDirection.IN],
        func_name="aicpu_orchestration_entry",
        binary=orchestration,
        children=[
            (0, incore(BARRIER_AIC, "aic", 5)),
            (1, incore(BARRIER_AIV_LANE1, "aiv", 5)),
            (2, incore(BARRIER_AIV_LANE2, "aiv", 5)),
            (3, incore(BARRIER_AIV_LANE0, "aiv", 5)),
        ],
    )


def _find_swimlane_records(output_prefix: Path) -> Path:
    matches = sorted(output_prefix.rglob("chip_swimlane_records.json"))
    assert matches, f"no chip_swimlane_records.json under {output_prefix}"
    return matches[-1]


def _run(worker, handle, occupy_count, output_prefix=None):
    host_out = torch.full((TOTAL_SLOTS * FLOATS_PER_CACHE_LINE,), UNWRITTEN, dtype=torch.float32)
    buffer = worker.malloc(host_out.nbytes)
    try:
        worker.copy_to(buffer, host_out)
        args = TaskArgs()
        args.add_tensor(
            buffer.tensor(shapes=(TOTAL_SLOTS * FLOATS_PER_CACHE_LINE,), dtype=DataType.FLOAT32),
            TensorArgType.INOUT,
        )
        args.add_scalar(occupy_count)
        args.add_scalar(OCCUPY_SPIN)
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


def _assert_mix_was_coresident(host_out, context):
    results = host_out[
        RESULT_BASE * FLOATS_PER_CACHE_LINE : (RESULT_BASE + LANES_PER_BLOCK)
        * FLOATS_PER_CACHE_LINE : FLOATS_PER_CACHE_LINE
    ]
    arrivals = host_out[
        ARRIVE_BASE * FLOATS_PER_CACHE_LINE : (ARRIVE_BASE + LANES_PER_BLOCK)
        * FLOATS_PER_CACHE_LINE : FLOATS_PER_CACHE_LINE
    ]
    assert torch.equal(arrivals, torch.full((LANES_PER_BLOCK,), ARRIVED)), (
        f"{context}: a subtask of the Mix task never ran at all: arrivals={arrivals.tolist()}"
    )
    poisoned = int((results == POISON).sum())
    assert poisoned == 0, (
        f"{context}: {poisoned} of the Mix task's three subtasks gave up waiting for the others, so its 1C2V were "
        f"not running together. A Mix task is a co-residency requirement at width one -- real Mix kernels have the "
        f"AIC and the AIVs block on each other's flags -- so this is a stall or a hang in any kernel that uses the "
        f"shape as intended. results={results.tolist()}"
    )
    assert torch.equal(results, torch.zeros(LANES_PER_BLOCK)), (
        f"{context}: the Mix task has one block, so every subtask should stamp block 0: {results.tolist()}"
    )


@pytest.mark.platforms(["a5sim", "a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
def test_mix_subtasks_run_together(st_platform, st_device_ids):
    """The baseline: nothing else competing, the 1C2V must still co-reside."""
    worker = Worker(level=2, platform=st_platform, runtime=RUNTIME, device_id=int(st_device_ids[0]))
    try:
        handle = worker.register(_build_callable(st_platform))
        worker.init()
        host_out = _run(worker, handle, occupy_count=0)
        _assert_mix_was_coresident(host_out, "a Mix task running alone")
    finally:
        worker.close()


# Each case holds a device for a process of its own, so a simulator run
# carries only the plain co-residency above.
@pytest.mark.platforms(["a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
def test_mix_subtasks_run_together_with_the_aivs_busy(st_platform, st_device_ids, tmp_path):
    """And still when the AIVs reach their subtasks later than the AIC does.

    Placement is atomic; execution is not. Filling every AIV with other work
    first means the Mix task's AIC subtask starts while its AIV subtasks are
    still queued, which is the shape where "placed together" and "runs
    together" come apart. The requirement does not weaken there: a Mix kernel
    whose AIC is already blocking on an AIV flag cannot be rescued by the AIV
    arriving a second later, because the AIC is holding its core while it
    waits.
    """
    worker = Worker(level=2, platform=st_platform, runtime=RUNTIME, device_id=int(st_device_ids[0]))
    try:
        handle = worker.register(_build_callable(st_platform))
        worker.init()
        output_prefix = tmp_path / "busy"
        output_prefix.mkdir(parents=True, exist_ok=True)
        # One occupying task per AIV, so none of them is idle when the Mix task
        # is placed. The count comes from the run: a fixed one would leave AIVs
        # free on a wider box and test nothing there.
        _run(worker, handle, occupy_count=0, output_prefix=output_prefix)
        raw = json.loads(_find_swimlane_records(output_prefix).read_text())
        aiv_count = sum(1 for kind in raw["metadata"]["core_types"] if kind == "aiv")
        assert 0 < aiv_count <= MAX_OCCUPY, raw["metadata"]

        host_out = _run(worker, handle, occupy_count=aiv_count)
        occupied = host_out[
            OCCUPY_BASE * FLOATS_PER_CACHE_LINE : (OCCUPY_BASE + aiv_count)
            * FLOATS_PER_CACHE_LINE : FLOATS_PER_CACHE_LINE
        ]
        assert int((occupied == UNWRITTEN).sum()) == 0, (
            f"an occupying task never ran, so the AIVs were not actually busy: {occupied.tolist()}"
        )
        _assert_mix_was_coresident(host_out, f"a Mix task placed while all {aiv_count} AIVs were busy")
    finally:
        worker.close()
