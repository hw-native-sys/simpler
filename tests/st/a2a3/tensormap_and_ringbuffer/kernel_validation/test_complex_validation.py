# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cooperative MIX, gated, SPMD and deferred failure retirement, without reuse."""

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


@pytest.mark.parametrize(
    ("code", "mode"),
    [
        pytest.param(
            code,
            mode,
            marks=pytest.mark.platforms(["a2a3"] if mode == 4 else ["a2a3sim"] if code == 108 else ["a2a3", "a2a3sim"]),
            id=f"{code}-{name}",
        )
        for code in (0, 106, 108)
        for mode, name in enumerate(("spmd", "mix", "gated", "deferred", "sdma"))
        if code != 108 or mode != 4
    ],
)
def test_complex_validation(st_platform, st_device_ids, tmp_path, capfd, mode, code):
    configure_logging("error")
    compiler = KernelCompiler(st_platform)
    children = []
    for kid, core, wait in ([(219, "aic", 0), (220, "aiv", 1)] if mode == 1 else [(219, "aiv", 0)]) + [(221, "aiv", 0)]:
        kernel = tmp_path / f"kernel_{kid}.cpp"
        kernel.write_text(
            (HERE / "kernels/complex.cpp.in")
            .read_text()
            .replace("CASE_MODE", str(mode))
            .replace("WAIT_LANE", str(wait))
            .replace("CONSUMER", str(int(kid == 221)))
        )
        binary = compiler.compile_incore(
            str(kernel),
            core_type=core,
            pto_isa_root=ensure_pto_isa_root(),
            extra_include_dirs=compiler.get_orchestration_include_dirs(RUNTIME),
        )
        if not st_platform.endswith("sim"):
            binary = extract_text_section(binary)
        children.append((kid, CoreCallable.build(signature=[D.OUT, D.IN] if kid == 221 else [D.OUT], binary=binary)))
    orch = tmp_path / "orchestration.cpp"
    orch.write_text(
        (HERE / "kernels/complex_orchestration.cpp.in")
        .read_text()
        .replace("CASE_MODE", str(mode))
        .replace("ERROR_CODE", str(code))
        .replace("FORCE_CONSUMER", str(int(code == 108)))
    )
    callable_ = ChipCallable.build(
        signature=[D.OUT],
        func_name="aicpu_orchestration_entry",
        binary=compiler.compile_orchestration(RUNTIME, str(orch)),
        children=children,
    )
    worker = Worker(
        level=2, platform=st_platform, runtime=RUNTIME, device_id=int(st_device_ids[0]), enable_sdma=(mode == 4)
    )
    try:
        handle = worker.register(callable_)
        args = TaskArgsBuilder(TensorArg("output", torch.zeros(4096, dtype=torch.float32)))
        task_args, _ = _build_l2_ref_args(args, [D.OUT], worker)
        worker.init()
        config = CallConfig()
        config.aicpu_thread_num = 3
        config.enable_chip_swimlane = 0
        started = time.monotonic()
        if code:
            with pytest.raises(RuntimeError):
                worker.run(handle, task_args, config)
        else:
            worker.run(handle, task_args, config)
            assert args.output[0].item() == 1
        assert time.monotonic() - started < 20
    finally:
        started = time.monotonic()
        worker.close()
        assert time.monotonic() - started < 45
    out, err = capfd.readouterr()
    if code == 108:
        # The exact same consumer entry is deliberately submitted without its
        # failed producer. Prove that the assertion used by cancellation cases
        # rejects execution independently of the first-error CAS/logging.
        with pytest.raises(AssertionError, match="consumer executed"):
            assert_consumer_not_entered(out + err)
        return
    if code:
        if mode == 4 and not st_platform.endswith("sim"):
            # Existing SDMA/RTS recovery can refuse safe copy-back after the
            # AICPU error. Device logs carry the primary code and drain result;
            # this PR preserves failed-run isolation rather than enabling reuse.
            assert "507018" in out + err, out + err
        else:
            assert "runtime_status=-106" in out + err, out + err
        if st_platform.endswith("sim"):
            assert "complete=1" in out + err, out + err
            assert_consumer_not_entered(out + err)
        print("complex retirement", mode, out + err)


def assert_consumer_not_entered(log):
    assert "VALIDATION_DEPENDENT_ENTERED" not in log, "consumer executed after failed producer"
