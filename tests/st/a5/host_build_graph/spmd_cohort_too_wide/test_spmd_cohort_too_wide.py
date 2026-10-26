#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A cohort wider than the machine is refused; an ordinary task that wide runs.

Worker demand is what the topology is checked against, and admitting SPMD
narrowed it to sync-start alone. The two halves of that have to be asserted
together, because each alone is satisfied by the wrong implementation: a demand
that still counted every task would reject both, and one that counted neither
would accept both.

Refusing rather than falling back is the other half. A cohort's blocks wait on
each other inside the kernel, so one wider than the machine cannot be satisfied
by any scheduler -- the AICPU one no more than the resident one. Sending it to
the AICPU scheduler would trade this message, which names the demand and the
core counts, for a forty-five-second device timeout that names nothing.

The width is past what any box here has rather than derived from this one: a
cohort that merely exceeds an eight-cluster box would be accepted on a
thirty-two-cluster one, and the case would quietly stop testing anything.
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
ORCHESTRATION_SOURCE = os.path.join(HERE, "kernels", "orchestration", "spmd_cohort_too_wide_orch.cpp")
KERNEL_SOURCE = os.path.join(
    HERE, "..", "spmd_multiblock_aic", "kernels", "aic", "kernel_stamp_block.cpp"
)

FLOATS_PER_CACHE_LINE = 16
# Past the AIC count of every topology this runs on -- the platform's own
# ceiling is 36 clusters, so nothing can hold 64 blocks at once.
TOO_WIDE = 64


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
        signature=[ArgDirection.INOUT, ArgDirection.IN, ArgDirection.IN],
        func_name="aicpu_orchestration_entry",
        binary=orchestration,
        children=[(0, CoreCallable.build(signature=[ArgDirection.INOUT], binary=binary))],
    )


def _run(worker, handle, sync_start, block_num=TOO_WIDE):
    slots = TOO_WIDE * FLOATS_PER_CACHE_LINE
    host_out = torch.full((slots,), -1.0, dtype=torch.float32)
    buffer = worker.malloc(host_out.nbytes)
    try:
        worker.copy_to(buffer, host_out)
        args = TaskArgs()
        args.add_tensor(buffer.tensor(shapes=(slots,), dtype=DataType.FLOAT32), TensorArgType.INOUT)
        args.add_scalar(block_num)
        args.add_scalar(1 if sync_start else 0)
        worker.run(handle, args, CallConfig())
        worker.copy_from(host_out, buffer)
        return host_out
    finally:
        try:
            worker.free(buffer)
        except Exception:  # noqa: BLE001
            pass


@pytest.mark.platforms(["a5sim", "a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
def test_cohort_wider_than_the_machine_is_refused(st_platform, st_device_ids, capfd, drain_host_log):
    """The refusal comes from orchestration, at submit, naming the limit.

    Two guards can reject this and the earlier one is the one that speaks: the
    shared orchestrator refuses a sync-start task wider than its target core
    type as the task is submitted, before a graph is built at all. The
    topology check the AICore scheduler makes later compares the same
    quantities against the cores that actually handshook, so it stays as a
    backstop for a topology that shrank in between -- not as the message a
    caller sees.
    """
    worker = Worker(level=2, platform=st_platform, runtime=RUNTIME, device_id=int(st_device_ids[0]))
    try:
        handle = worker.register(_build_callable(st_platform))
        worker.init()
        with pytest.raises(RuntimeError):
            _run(worker, handle, sync_start=True)
        log = drain_host_log(capfd)
        assert "require_sync_start block_num=64" in log, log
        assert "REQUIRE_SYNC_START_INVALID" in log, log
    finally:
        worker.close()


@pytest.mark.platforms(["a5sim", "a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
def test_ordinary_task_that_wide_runs_in_waves(st_platform, st_device_ids):
    """The same width, without sync-start, is routine rather than a refusal.

    Its blocks are claimed as cores free up and run in as many waves as it
    takes. Before demand narrowed to sync-start this was rejected by the same
    check the case above still trips.
    """
    worker = Worker(level=2, platform=st_platform, runtime=RUNTIME, device_id=int(st_device_ids[0]))
    try:
        handle = worker.register(_build_callable(st_platform))
        worker.init()
        out = _run(worker, handle, sync_start=False)
        stamped = out[0 : TOO_WIDE * FLOATS_PER_CACHE_LINE : FLOATS_PER_CACHE_LINE]
        assert torch.equal(stamped, torch.arange(TOO_WIDE, dtype=torch.float32)), (
            f"not every block ran: {stamped.tolist()}"
        )
    finally:
        worker.close()


@pytest.mark.platforms(["a5sim", "a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
def test_sync_start_on_a_single_block_is_dropped_not_refused(st_platform, st_device_ids):
    """One block asking for sync-start is an ordinary task, not an error.

    A cohort of one is co-resident the moment it is dispatched, so there is
    nothing for the barrier to wait on and the submit path clears the attribute
    rather than carrying it. The scheduler keeps its own guard against a
    sync-start task of one block, but nothing that submits through this API can
    reach it: the attribute is gone before a graph is built.

    Which is why this asserts the task runs. Asserting the refusal would be
    asserting a path no caller has.
    """
    worker = Worker(level=2, platform=st_platform, runtime=RUNTIME, device_id=int(st_device_ids[0]))
    try:
        handle = worker.register(_build_callable(st_platform))
        worker.init()
        out = _run(worker, handle, sync_start=True, block_num=1)
        assert out[0] == 0.0, f"the single block did not run: {out[0]}"
        assert torch.all(out[FLOATS_PER_CACHE_LINE::FLOATS_PER_CACHE_LINE] == -1.0), (
            "a task of one block stamped more than one slot"
        )
    finally:
        worker.close()
