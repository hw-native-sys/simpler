#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A cohort as wide as the machine: every core takes exactly one block.

This is the boundary the one-block-per-core rule exists for. Below full width
there is slack -- a core that wrongly took two blocks leaves another core idle
and the cohort still assembles, so nothing shows. At full width there is none:
a second block on any core means some core has none, and the cohort can never
be resident.

The width is read from the run, not fixed, because the boundary is what is
being tested and it sits in a different place on an eight-cluster box than on a
thirty-two-cluster one. Which also means the output cannot be a fixed golden,
so this is a standalone case rather than a scene-test one: it checks what the
run reports about itself.

The blocks hold their cores rather than returning at once, and that is load
bearing: the rule is enforced by a core scanning its own dispatch slots for the
task, and a slot is cleared when its block completes. With a kernel that
returns immediately, whether a core takes a second block depends on whether the
claim round outran the first block -- so the assertion below would be measuring
a race. The AIV cohort of spmd_cohort_aiv_full_width is where that showed:
sixteen blocks landed on nine cores.
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
ORCHESTRATION_SOURCE = os.path.join(HERE, "kernels", "orchestration", "spmd_cohort_full_width_orch.cpp")
KERNEL_SOURCE = os.path.join(HERE, "kernels", "aic", "kernel_stamp_block_resident.cpp")

FLOATS_PER_CACHE_LINE = 16
# The platform's own ceiling on clusters, so the buffer fits any box this runs
# on without the test having to know which one it is.
MAX_BLOCKS = 36
UNWRITTEN = -1.0


def _build_callable(platform):
    compiler = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root()
    include_dirs = list(compiler.get_orchestration_include_dirs(RUNTIME)) + [
        str(compiler.project_root / "src" / "common")
    ]
    binary = compiler.compile_incore(
        source_path=KERNEL_SOURCE,
        core_type="aic",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=include_dirs,
    )
    if not platform.endswith("sim"):
        binary = extract_text_section(binary)
    orchestration = compiler.compile_orchestration(runtime_name=RUNTIME, source_path=ORCHESTRATION_SOURCE)
    return ChipCallable.build(
        signature=[ArgDirection.INOUT],
        func_name="aicpu_orchestration_entry",
        binary=orchestration,
        children=[(0, CoreCallable.build(signature=[ArgDirection.INOUT], binary=binary))],
    )


def _find_swimlane_records(output_prefix: Path) -> Path:
    matches = sorted(output_prefix.rglob("chip_swimlane_records.json"))
    assert matches, f"no chip_swimlane_records.json under {output_prefix}"
    return matches[-1]


@pytest.mark.platforms(["a5sim", "a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
def test_full_width_cohort_uses_every_core_once(st_platform, st_device_ids, tmp_path):
    worker = Worker(level=2, platform=st_platform, runtime=RUNTIME, device_id=int(st_device_ids[0]))
    buffer = None
    try:
        handle = worker.register(_build_callable(st_platform))
        worker.init()

        slots = MAX_BLOCKS * FLOATS_PER_CACHE_LINE
        host_out = torch.full((slots,), UNWRITTEN, dtype=torch.float32)
        buffer = worker.malloc(host_out.nbytes)
        worker.copy_to(buffer, host_out)

        args = TaskArgs()
        args.add_tensor(buffer.tensor(shapes=(slots,), dtype=DataType.FLOAT32), TensorArgType.INOUT)

        output_prefix = tmp_path / "full_width"
        output_prefix.mkdir(parents=True, exist_ok=True)
        config = CallConfig()
        config.enable_chip_swimlane = 3
        config.output_prefix = str(output_prefix)
        worker.run(handle, args, config)
        worker.copy_from(host_out, buffer)

        raw = json.loads(_find_swimlane_records(output_prefix).read_text())
        core_types = raw["metadata"]["core_types"]
        aic_count = sum(1 for kind in core_types if kind == "aic")
        assert 0 < aic_count <= MAX_BLOCKS, f"unexpected AIC count {aic_count}: {raw['metadata']}"

        rows = raw["aicore_tasks"]
        assert len(rows) == aic_count, (
            f"a cohort as wide as the {aic_count} AIC cores produced {len(rows)} block records"
        )
        cores = {int(row[0]) for row in rows}
        assert len(cores) == aic_count, (
            f"the cohort covered {len(cores)} of {aic_count} AIC cores: {sorted(cores)}. At full width a core "
            "holding two blocks leaves another with none, which is a cohort that can never be resident."
        )

        stamped = host_out[0 : aic_count * FLOATS_PER_CACHE_LINE : FLOATS_PER_CACHE_LINE]
        assert torch.equal(stamped, torch.arange(aic_count, dtype=torch.float32)), (
            f"not every block stamped its slot: {stamped.tolist()}"
        )
        # Nothing past the cohort's width ran, which is what says the width
        # came from the topology rather than from a constant that happened to
        # match it.
        assert torch.all(host_out[aic_count * FLOATS_PER_CACHE_LINE :: FLOATS_PER_CACHE_LINE] == UNWRITTEN)
    finally:
        # A poisoned run still owns its device memory, so free() would raise and
        # bury whatever actually failed. Closing the worker releases it.
        if buffer is not None:
            try:
                worker.free(buffer)
            except Exception:  # noqa: BLE001
                pass
        worker.close()
