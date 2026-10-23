#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A dispatch predicate on a Mix task is refused by the a5 HBG resident scheduler.

`create_scheduler_state()` requires a predicated task to carry exactly one
active subtask and one logical block:

    slot.task_attrs.has_predicate() && (active_subtasks != 1 || logical_block_num != 1)

A Mix task has two or three active subtasks, so pairing it with a predicate is
refused while the graph is still being prepared, and the run fails. A single
active subtask with a predicate is accepted and is covered by
`tests/st/a5/host_build_graph/predicated_dispatch/`.

This test pins that boundary. It is a contract test, not an aspiration: the
per-core Scheduler work deliberately keeps the restriction, so any change that
starts accepting a predicated Mix task has to come here and say so rather than
silently turning green. Lifting the restriction is what would make the
"evaluate the predicate once and ship the result with the dispatch" rule
meaningful — today a Mix task never carries a predicate at all.

The worker is private to this test. A deliberate bind failure must not reach the
session-scoped worker pool, whose other users would inherit a poisoned worker.
"""

import functools
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
ORCHESTRATION_SOURCE = os.path.join(HERE, "kernels", "orchestration", "mix_predicated_dispatch_orch.cpp")
AIC_SOURCE = os.path.join(HERE, "kernels", "aic", "mix_sentinel_aic.cpp")
AIV0_SOURCE = os.path.join(HERE, "kernels", "aiv", "mix_sentinel_aiv0.cpp")
AIV1_SOURCE = os.path.join(HERE, "kernels", "aiv", "mix_sentinel_aiv1.cpp")
GATE_SOURCE = os.path.join(HERE, "..", "predicated_dispatch", "kernels", "aic", "kernel_write_gate.cpp")

ELEMS = 16

# 0b011 / 0b101 / 0b110 / 0b111 — every single-lane Mix shape. 0b110 is the one
# where the AIC holds no subtask of its own.
MIX_MASKS = (3, 5, 6, 7)


@functools.cache
def _build_callable(platform: str) -> ChipCallable:
    compiler = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root()
    include_dirs = list(compiler.get_orchestration_include_dirs(RUNTIME)) + [
        str(compiler.project_root / "src" / "common")
    ]

    def incore(source, core_type, signature):
        binary = compiler.compile_incore(
            source_path=source,
            core_type=core_type,
            pto_isa_root=pto_isa_root,
            extra_include_dirs=include_dirs,
        )
        # Onboard loads raw code, so the ELF wrapper must come off first;
        # sim consumes the ELF as-is. scene_test.py:1852 makes the same split.
        if not platform.endswith("sim"):
            binary = extract_text_section(binary)
        return CoreCallable.build(signature=signature, binary=binary)

    three_inout = [ArgDirection.INOUT] * 3
    orchestration = compiler.compile_orchestration(runtime_name=RUNTIME, source_path=ORCHESTRATION_SOURCE)
    return ChipCallable.build(
        signature=[ArgDirection.INOUT] * 4,
        func_name="aicpu_orchestration_entry",
        binary=orchestration,
        children=[
            (0, incore(AIC_SOURCE, "aic", three_inout)),
            (1, incore(AIV0_SOURCE, "aiv", three_inout)),
            (2, incore(AIV1_SOURCE, "aiv", three_inout)),
            (3, incore(GATE_SOURCE, "aic", [ArgDirection.INOUT])),
        ],
    )


@pytest.mark.platforms(["a5sim", "a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
@pytest.mark.parametrize("mask", MIX_MASKS)
@pytest.mark.parametrize("gate", (0, 1))
def test_predicated_mix_task_is_refused(st_platform, st_device_ids, mask, gate, capfd, drain_host_log):
    """Every Mix shape is refused, and the gate value does not change that.

    The refusal happens while the graph is prepared, before the predicate could
    be evaluated, so it cannot depend on whether the predicate would have held.
    """
    worker = Worker(level=2, platform=st_platform, runtime=RUNTIME, device_id=int(st_device_ids[0]))
    buffers = []
    try:
        handle = worker.register(_build_callable(st_platform))
        worker.init()

        for host_value in (
            torch.full((ELEMS,), -1.0, dtype=torch.float32),
            torch.full((ELEMS,), -1.0, dtype=torch.float32),
            torch.full((ELEMS,), -1.0, dtype=torch.float32),
            torch.full((ELEMS,), -1, dtype=torch.int32),
        ):
            buffer = worker.malloc(host_value.nbytes)
            worker.copy_to(buffer, host_value)
            buffers.append((buffer, host_value))

        args = TaskArgs()
        for buffer, host_value in buffers:
            dtype = DataType.INT32 if host_value.dtype == torch.int32 else DataType.FLOAT32
            args.add_tensor(buffer.tensor(shapes=(ELEMS,), dtype=dtype), TensorArgType.INOUT)
        args.add_scalar(gate)
        args.add_scalar(mask)

        config = CallConfig()
        config.aicpu_thread_num = 2
        with pytest.raises(RuntimeError, match=r"failed with code -1000\b"):
            worker.run(handle, args, config)

        log = drain_host_log(capfd)
        assert "uses predicate/early-resolve" in log, log
    finally:
        for buffer, _ in buffers:
            worker.free(buffer)
        worker.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *os.sys.argv[1:]]))
