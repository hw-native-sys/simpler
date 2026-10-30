#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A dispatch predicate on a task of many blocks is refused at bind.

The two features never meet. A predicate is read once, at the moment the task
is dispatched, and an SPMD task is dispatched once per block by whichever cores
claim them -- there is no single moment to read it at. The bind walk refuses
the graph rather than pick one.

The refusal is older than SPMD support and is not a choice the resident
scheduler makes: the check sits ahead of any scheduler selection, so the graph
never reaches one. Before SPMD was resident, a multi-block task was handed to
the AICPU scheduler, but only if it got that far, and a predicated one did not.
So this is the same answer it has always given, and the case exists to say so
in one place: the message a caller actually sees names predicates and
early-resolve together and nothing about block counts, which makes it easy to
read as something else.

Both gate values are run because the predicate's value cannot matter. A refusal
that depended on it would mean the gate had been read, which is the thing that
cannot happen here.
"""

import os

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
ORCHESTRATION_SOURCE = os.path.join(HERE, "kernels", "orchestration", "spmd_predicated_orch.cpp")
GATE_SOURCE = os.path.join(HERE, "..", "predicated_dispatch", "kernels", "aic", "kernel_write_gate.cpp")
STAMP_SOURCE = os.path.join(HERE, "..", "spmd_multiblock_aic", "kernels", "aic", "kernel_stamp_block.cpp")

FLOATS_PER_CACHE_LINE = 16
BLOCK_NUM = 4
UNWRITTEN = -1.0


def _build_callable(platform):
    compiler = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root()
    include_dirs = list(compiler.get_orchestration_include_dirs(RUNTIME)) + [
        str(compiler.project_root / "src" / "common")
    ]

    def incore(source, signature):
        binary = compiler.compile_incore(
            source_path=source,
            core_type="aic",
            pto_isa_root=pto_isa_root,
            extra_include_dirs=include_dirs,
        )
        if not platform.endswith("sim"):
            binary = extract_text_section(binary)
        return CoreCallable.build(signature=signature, binary=binary)

    orchestration = compiler.compile_orchestration(runtime_name=RUNTIME, source_path=ORCHESTRATION_SOURCE)
    return ChipCallable.build(
        signature=[ArgDirection.INOUT, ArgDirection.INOUT, ArgDirection.IN, ArgDirection.IN],
        func_name="aicpu_orchestration_entry",
        binary=orchestration,
        children=[
            (0, incore(GATE_SOURCE, [ArgDirection.INOUT, ArgDirection.IN])),
            (1, incore(STAMP_SOURCE, [ArgDirection.INOUT])),
        ],
    )


def _run(worker, handle, block_num, gate_value):
    slots = BLOCK_NUM * FLOATS_PER_CACHE_LINE
    host_out = torch.full((slots,), UNWRITTEN, dtype=torch.float32)
    host_gate = torch.zeros((1,), dtype=torch.int32)
    out_buffer = worker.malloc(host_out.nbytes)
    gate_buffer = worker.malloc(host_gate.nbytes)
    try:
        worker.copy_to(out_buffer, host_out)
        worker.copy_to(gate_buffer, host_gate)
        args = TaskArgs()
        args.add_tensor(gate_buffer.tensor(shapes=(1,), dtype=DataType.INT32), TensorArgType.INOUT)
        args.add_tensor(out_buffer.tensor(shapes=(slots,), dtype=DataType.FLOAT32), TensorArgType.INOUT)
        args.add_scalar(block_num)
        args.add_scalar(gate_value)
        worker.run(handle, args, CallConfig())
        worker.copy_from(host_out, out_buffer)
        return host_out
    finally:
        for buffer in (out_buffer, gate_buffer):
            try:
                worker.free(buffer)
            except Exception:  # noqa: BLE001
                pass


@pytest.mark.platforms(["a5sim", "a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
@pytest.mark.parametrize("gate_value", [1, 0])
def test_predicate_on_a_multi_block_task_is_refused(st_platform, st_device_ids, gate_value, capfd, drain_host_log):
    worker = Worker(level=2, platform=st_platform, runtime=RUNTIME, device_id=int(st_device_ids[0]))
    try:
        handle = worker.register(_build_callable(st_platform))
        worker.init()
        with pytest.raises(RuntimeError):
            _run(worker, handle, block_num=BLOCK_NUM, gate_value=gate_value)
        log = drain_host_log(capfd)
        assert "uses predicate/early-resolve" in log, log
    finally:
        worker.close()


# Each case holds a device for a process of its own, so a simulator run
# carries only the refusal above; the accepted shape stays on hardware.
@pytest.mark.platforms(["a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
@pytest.mark.parametrize(("gate_value", "expected"), [(1, 0.0), (0, UNWRITTEN)])
def test_the_same_predicate_on_one_block_runs(st_platform, st_device_ids, gate_value, expected):
    """The other half: the predicate itself is fine, the block count is not.

    Without this the refusal above is satisfied by a build where predicates do
    not work at all, which is a different defect wearing the same message.
    """
    worker = Worker(level=2, platform=st_platform, runtime=RUNTIME, device_id=int(st_device_ids[0]))
    try:
        handle = worker.register(_build_callable(st_platform))
        worker.init()
        out = _run(worker, handle, block_num=1, gate_value=gate_value)
        assert out[0] == expected, f"gate={gate_value} expected slot 0 to be {expected}, got {out[0]}"
    finally:
        worker.close()
