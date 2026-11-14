# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Kernel validation reaches Host through the ordinary AIC/AIV completion path."""

import re
import time
from pathlib import Path

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, ChipCallable, CoreCallable
from simpler.worker import Worker

from simpler_setup import TaskArgsBuilder, TensorArg
from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.log_config import configure_logging
from simpler_setup.pto_isa import ensure_pto_isa_root
from simpler_setup.scene_test import _build_l2_ref_args

HERE = Path(__file__).parent
RUNTIME = "tensormap_and_ringbuffer"
pytestmark = [
    pytest.mark.platforms(["a2a3", "a2a3sim"]),
    pytest.mark.device_count(1),
    pytest.mark.runtime(RUNTIME),
    pytest.mark.timeout(180),
]


@pytest.mark.parametrize("core_type", ["aic", "aiv"])
@pytest.mark.parametrize("code", [0, 105, 106, 107])
def test_kernel_validation(st_platform, st_device_ids, tmp_path, capfd, core_type, code):
    configure_logging("error")
    compiler = KernelCompiler(st_platform)
    orch = tmp_path / "orchestration.cpp"
    template = (HERE / "kernels/orchestration.cpp.in").read_text()
    orch.write_text(template.replace("CORE_TYPE", core_type).replace("ERROR_CODE", str(code)))
    binary = compiler.compile_incore(
        str(HERE / "kernels/kernel.cpp"),
        core_type=core_type,
        pto_isa_root=ensure_pto_isa_root(),
        extra_include_dirs=compiler.get_orchestration_include_dirs(RUNTIME),
    )
    if not st_platform.endswith("sim"):
        binary = extract_text_section(binary)
    callable_ = ChipCallable.build(
        signature=[D.OUT],
        func_name="aicpu_orchestration_entry",
        binary=compiler.compile_orchestration(RUNTIME, str(orch)),
        children=[(219, CoreCallable.build(signature=[D.OUT], binary=binary))],
    )
    worker = Worker(level=2, platform=st_platform, runtime=RUNTIME, device_id=int(st_device_ids[0]))
    try:
        handle = worker.register(callable_)
        valid_orch = tmp_path / "valid_orchestration.cpp"
        valid_orch.write_text(template.replace("CORE_TYPE", core_type).replace("ERROR_CODE", "0"))
        valid_handle = worker.register(
            ChipCallable.build(
                signature=[D.OUT],
                func_name="aicpu_orchestration_entry",
                binary=compiler.compile_orchestration(RUNTIME, str(valid_orch)),
                children=[(219, CoreCallable.build(signature=[D.OUT], binary=binary))],
            )
        )
        args = TaskArgsBuilder(TensorArg("output", torch.zeros(1, dtype=torch.int32)))
        task_args, _ = _build_l2_ref_args(args, [D.OUT], worker)
        worker.init()
        config = CallConfig()
        config.aicpu_thread_num = 2
        config.enable_chip_swimlane = 0
        started = time.monotonic()
        if code:
            args.output.fill_(77)
            with pytest.raises(RuntimeError) as failure:
                worker.run(handle, task_args, config)
            assert args.output[0].item() == 77
            with pytest.raises(RuntimeError, match="poison"):
                worker.run(valid_handle, task_args, config)
            assert args.output[0].item() == 77
        else:
            for _ in range(2):
                args.output.zero_()
                worker.run(handle, task_args, config)
                assert args.output[0].item() == 1
        assert time.monotonic() - started < 15
    finally:
        started = time.monotonic()
        worker.close()
        assert time.monotonic() - started < 45
    out, err = capfd.readouterr()
    if code:
        host_error = f"orch_error_code={code} sched_error_code={code} runtime_status=-{code}"
        assert host_error in out + err, out + err
        # On hardware the AICPU identity diagnostic goes to device logs.
        if st_platform.endswith("sim"):
            assert re.search(
                rf"Kernel failure: code={code} task=0x[0-9a-f]+ func=219 block=0 core=[0-9]+",
                out + err,
            ), out + err
        print(host_error)
        print(str(failure.value))
    else:
        assert "Kernel failure:" not in out + err
