#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A full-width cohort whose blocks only AIV cores may claim.

Which cores may take a block of an SPMD task is read from the task's own shape,
and every cohort case so far has a shape that admits the claiming core: the AIC
cohort of spmd_cohort_full_width is claimed by AICs, and spmd_sync_start_mix_aiv
is Mix, so its blocks are routed by each cluster's coordinator rather than by
cores deciding for themselves. Neither can see an AIC core wrongly deciding that
an AIV-only shape is one it may run.

This one's mask names AIV alone, and it is as wide as the AIV population, which
is wider than the AIC one. So there are more blocks than there are AICs to
mistake them for their own, and the AICs sit idle beside a queue head they can
see and must refuse -- while the cohort still has to assemble across every AIV
to become resident at all.

The width comes from the run, so there is no fixed golden and this is a
standalone case: it checks what the run reports about itself.
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
ORCHESTRATION_SOURCE = os.path.join(HERE, "kernels", "orchestration", "spmd_cohort_aiv_full_width_orch.cpp")
# The blocks wait for each other, so this case tests co-residency and not only
# which cores the blocks landed on.
KERNEL_SOURCE = os.path.join(HERE, "..", "spmd_cohort_saturation", "kernels", "aiv", "kernel_cohort_barrier.cpp")

FLOATS_PER_CACHE_LINE = 16
# Mirrors SPMD_COHORT_MAX_BLOCKS in the orchestration: wide enough for the AIV
# population of any box this runs on, without the test knowing which one.
MAX_BLOCKS = 160
UNWRITTEN = -1.0
ARRIVED = 1.0
POISON = -99.0
# Measured in spmd_cohort_saturation: about a second before a block gives up.
SPIN_LIMIT = 200000000


def _build_callable(platform):
    compiler = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root()
    include_dirs = list(compiler.get_orchestration_include_dirs(RUNTIME)) + [
        str(compiler.project_root / "src" / "common")
    ]
    binary = compiler.compile_incore(
        source_path=KERNEL_SOURCE,
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=include_dirs,
    )
    if not platform.endswith("sim"):
        binary = extract_text_section(binary)
    orchestration = compiler.compile_orchestration(runtime_name=RUNTIME, source_path=ORCHESTRATION_SOURCE)
    return ChipCallable.build(
        signature=[ArgDirection.INOUT, ArgDirection.IN],
        func_name="aicpu_orchestration_entry",
        binary=orchestration,
        children=[
            (
                0,
                CoreCallable.build(
                    signature=[
                        ArgDirection.INOUT,
                        ArgDirection.IN,
                        ArgDirection.IN,
                        ArgDirection.IN,
                        ArgDirection.IN,
                    ],
                    binary=binary,
                ),
            )
        ],
    )


def _find_swimlane_records(output_prefix: Path) -> Path:
    matches = sorted(output_prefix.rglob("chip_swimlane_records.json"))
    assert matches, f"no chip_swimlane_records.json under {output_prefix}"
    return matches[-1]


@pytest.mark.platforms(["a5sim", "a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
def test_full_width_aiv_cohort_uses_every_aiv_once(st_platform, st_device_ids, tmp_path):
    worker = Worker(level=2, platform=st_platform, runtime=RUNTIME, device_id=int(st_device_ids[0]))
    buffer = None
    try:
        handle = worker.register(_build_callable(st_platform))
        worker.init()

        slots = 2 * MAX_BLOCKS * FLOATS_PER_CACHE_LINE
        host_out = torch.full((slots,), UNWRITTEN, dtype=torch.float32)
        buffer = worker.malloc(host_out.nbytes)
        worker.copy_to(buffer, host_out)

        args = TaskArgs()
        args.add_tensor(buffer.tensor(shapes=(slots,), dtype=DataType.FLOAT32), TensorArgType.INOUT)
        args.add_scalar(SPIN_LIMIT)

        output_prefix = tmp_path / "aiv_full_width"
        output_prefix.mkdir(parents=True, exist_ok=True)
        config = CallConfig()
        config.enable_chip_swimlane = 3
        config.output_prefix = str(output_prefix)
        worker.run(handle, args, config)
        worker.copy_from(host_out, buffer)

        raw = json.loads(_find_swimlane_records(output_prefix).read_text())
        core_types = raw["metadata"]["core_types"]
        aiv_count = sum(1 for kind in core_types if kind == "aiv")
        aic_count = sum(1 for kind in core_types if kind == "aic")
        assert 0 < aiv_count <= MAX_BLOCKS, f"unexpected AIV count {aiv_count}: {raw['metadata']}"
        # The premise of the case: AICs outnumbered by the blocks they must
        # refuse. With no AIC present there is nothing to refuse and this is
        # only a restatement of spmd_cohort_full_width.
        assert aic_count > 0, f"no AIC cores on this box: {raw['metadata']}"

        rows = raw["aicore_tasks"]
        assert len(rows) == aiv_count, (
            f"a cohort as wide as the {aiv_count} AIV cores produced {len(rows)} block records"
        )
        cores = {int(row[0]) for row in rows}
        assert len(cores) == aiv_count, (
            f"the cohort covered {len(cores)} of {aiv_count} AIV cores: {sorted(cores)}. At full width a core "
            "holding two blocks leaves another with none, which is a cohort that can never be resident."
        )

        arrivals = host_out[0 : aiv_count * FLOATS_PER_CACHE_LINE : FLOATS_PER_CACHE_LINE]
        assert torch.equal(arrivals, torch.full((aiv_count,), ARRIVED)), (
            f"a block of the cohort never ran at all: {arrivals.tolist()}"
        )
        stamped = host_out[
            MAX_BLOCKS * FLOATS_PER_CACHE_LINE : (MAX_BLOCKS + aiv_count) * FLOATS_PER_CACHE_LINE :
            FLOATS_PER_CACHE_LINE
        ]
        poisoned = int((stamped == POISON).sum())
        assert poisoned == 0, (
            f"{poisoned} of the cohort's {aiv_count} blocks gave up waiting for their siblings, so the cohort was "
            f"never resident all at once: {stamped.tolist()}"
        )
        assert torch.equal(stamped, torch.arange(aiv_count, dtype=torch.float32)), (
            f"not every block stamped its slot: {stamped.tolist()}"
        )
        assert torch.all(
            host_out[aiv_count * FLOATS_PER_CACHE_LINE : MAX_BLOCKS * FLOATS_PER_CACHE_LINE :
                     FLOATS_PER_CACHE_LINE] == UNWRITTEN
        )
        assert torch.all(
            host_out[(MAX_BLOCKS + aiv_count) * FLOATS_PER_CACHE_LINE :: FLOATS_PER_CACHE_LINE] == UNWRITTEN
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
