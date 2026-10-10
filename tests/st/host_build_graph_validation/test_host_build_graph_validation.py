#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Invalid host-build-graph inputs fail with the documented invalid-argument status."""

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
    _host_log_dropped_records,
)
from simpler.worker import Worker

from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

HERE = os.path.dirname(os.path.abspath(__file__))
RUNTIME = "host_build_graph"
ORCHESTRATION_SOURCE = os.path.join(HERE, "kernels", "orchestration", "validation_orch.cpp")
CORE_SOURCE = os.path.join(HERE, "kernels", "core", "kernel_noop.cpp")

CASES = {
    "zero_block_num": 0,
    "mixed_subtask_overflow": 1,
    "unbound_owner_read": 2,
    "unbound_owner_write": 3,
    "sub_task_dependency": 4,
    "read_task_output": 5,
    "write_task_produced_tensor": 6,
    "read_alloc_tensors_output": 7,
}


@functools.cache
def _build_callable(platform: str) -> ChipCallable:
    compiler = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root()
    include_dirs = list(compiler.get_orchestration_include_dirs(RUNTIME)) + [
        str(compiler.project_root / "src" / "common")
    ]

    aic_binary = compiler.compile_incore(
        source_path=CORE_SOURCE,
        core_type="aic",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=include_dirs,
    )
    aiv_binary = compiler.compile_incore(
        source_path=CORE_SOURCE,
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=include_dirs,
    )
    orchestration = compiler.compile_orchestration(runtime_name=RUNTIME, source_path=ORCHESTRATION_SOURCE)
    return ChipCallable.build(
        signature=[ArgDirection.INOUT],
        func_name="aicpu_orchestration_entry",
        binary=orchestration,
        children=[
            (0, CoreCallable.build(signature=[], binary=aic_binary)),
            (1, CoreCallable.build(signature=[], binary=aiv_binary)),
            (2, CoreCallable.build(signature=[], binary=aiv_binary)),
        ],
    )


@pytest.mark.platforms(["a2a3sim", "a5sim"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
@pytest.mark.parametrize("case_name", list(CASES))
def test_invalid_input_reports_code_five(st_platform, st_device_ids, case_name, capfd, drain_host_log):
    worker = Worker(level=2, platform=st_platform, runtime=RUNTIME, device_id=int(st_device_ids[0]))
    buffer = None
    dropped_before = _host_log_dropped_records()
    try:
        handle = worker.register(_build_callable(st_platform))
        worker.init()

        host_value = torch.zeros(1, dtype=torch.int32)
        buffer = worker.malloc(host_value.nbytes)
        worker.copy_to(buffer, host_value)

        args = TaskArgs()
        args.add_tensor(buffer.tensor(shapes=(1,), dtype=DataType.INT32), TensorArgType.INOUT)
        args.add_scalar(CASES[case_name])

        config = CallConfig()
        config.aicpu_thread_num = 2
        with pytest.raises(RuntimeError, match=r"(run_runtime|run) failed with code -5\b"):
            worker.run(handle, args, config)

        log = drain_host_log(capfd)
        assert "orch_error_code=5" in log, log
        assert "INVALID_ARGS" in log, log
        dropped_after = _host_log_dropped_records()
        assert dropped_after == dropped_before, (
            f"host-log drop counter changed during invalid-input rejection: "
            f"before={dropped_before}, after={dropped_after}"
        )
    finally:
        if buffer is not None:
            worker.free(buffer)
        worker.close()
