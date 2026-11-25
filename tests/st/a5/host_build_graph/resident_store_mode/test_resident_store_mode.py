# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Full-width AIV contamination followed by a separate resident invocation."""

import json
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

HERE = Path(__file__).parent
MAX_WORKERS = 320
ROW_SIZE = 32


def _build_callable():
    compiler = KernelCompiler(platform="a5")
    binary = compiler.compile_incore(
        source_path=str(HERE / "kernels/aiv/store_mode.cpp"),
        core_type="aiv",
        pto_isa_root=ensure_pto_isa_root(),
        extra_include_dirs=list(compiler.get_orchestration_include_dirs("host_build_graph")),
    )
    orchestration = compiler.compile_orchestration(
        runtime_name="host_build_graph", source_path=str(HERE / "kernels/orchestration/store_mode_orch.cpp")
    )
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.INOUT],
        func_name="aicpu_orchestration_entry",
        binary=orchestration,
        children=[
            (
                0,
                CoreCallable.build(
                    signature=[ArgDirection.IN, ArgDirection.INOUT, ArgDirection.IN],
                    binary=extract_text_section(binary),
                ),
            )
        ],
    )


def _task_args(buffers, mode):
    args = TaskArgs()
    args.add_tensor(buffers[0].tensor(shapes=(16,), dtype=DataType.FLOAT32), TensorArgType.INPUT)
    args.add_tensor(buffers[1].tensor(shapes=(MAX_WORKERS, ROW_SIZE), dtype=DataType.FLOAT32), TensorArgType.INOUT)
    args.add_scalar(mode)
    return args


@pytest.mark.manual
@pytest.mark.platforms(["a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime("host_build_graph")
def test_resident_entry_resets_prior_store_mode(st_platform, st_device_ids, tmp_path):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=int(st_device_ids[0]))
    buffers = []
    evidence = {"device": int(st_device_ids[0]), "runs": []}
    try:
        handle = worker.register(_build_callable())
        worker.init()
        source = torch.ones(16, dtype=torch.float32)
        target = torch.full((MAX_WORKERS, ROW_SIZE), 4.0, dtype=torch.float32)
        for host in (source, target):
            buffers.append(worker.malloc(host.nbytes))
            worker.copy_to(buffers[-1], host)
        core_sets = []
        for poison in (1, 0):
            target.fill_(4.0)
            target[:, 16:] = -1.0
            worker.copy_to(buffers[1], target)
            args = _task_args(buffers, poison)
            worker.run(handle, args, CallConfig())
            assert worker.scheduler_runtime_mode == 5, "the run must use the calendar scheduler"
            worker.copy_from(target, buffers[1])
            active = target[:, 16] >= 0
            rows = target[active]
            ids = [int(value) for value in rows[:, 16].tolist()]
            evidence["runs"].append({"poison": poison, "physical_cores": ids, "values": rows[:, :16].tolist()})
            (tmp_path / "store_mode_evidence.json").write_text(json.dumps(evidence, indent=2))
            assert len(ids) > 0 and len(set(ids)) == len(ids), "each AIV must execute exactly one block"
            assert torch.all(rows[:, 17] == len(ids)), "the full-width cohort must cover every AIV"
            assert torch.all(rows[:, 18] == poison)
            core_sets.append(set(ids))
            if not poison:
                assert core_sets[0] == core_sets[1], "the verification must reuse all poisoned physical cores"
            expected = 5.0 if poison else 1.0
            assert torch.all(rows[:, :16] == expected), (
                f"expected {expected}, observed {torch.unique(rows[:, :16]).tolist()}; "
                f"evidence: {tmp_path / 'store_mode_evidence.json'}"
            )
    finally:
        try:
            if len(buffers) == 2:
                worker.run(handle, _task_args(buffers, 2), CallConfig())
        finally:
            for buffer in buffers:
                worker.free(buffer)
            worker.close()
