#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Negative ST for the AICore op-execution timeout chain (regression for PR #718).

Hardware-only: dispatches an AIC kernel that spins forever. The 3-layer
timeout chain (STARS op watchdog ~1 s, AICPU deinit ~1 s, host stream sync
2 s/stream) must reap the hang and surface a ``RuntimeError`` in
single-digit seconds rather than deadlocking. There is no a2a3sim case
because the simulator has no STARS watchdog — a ``while(true)`` kernel
would wedge the sim. a5 has no equivalent timeout mechanism yet; mirror
this test there once it does.
"""

import os
import time

import pytest
from simpler.task_interface import CallConfig, ChipCallable, ChipStorageTaskArgs, CoreCallable
from simpler.worker import Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.log_config import configure_logging
from simpler_setup.pto_isa import ensure_pto_isa_root

HERE = os.path.dirname(os.path.abspath(__file__))
RUNTIME = "tensormap_and_ringbuffer"
ORCH_SRC = os.path.join(HERE, "kernels/orchestration/aicore_op_timeout_orch.cpp")
AIC_SRC = os.path.join(HERE, "kernels/aic/kernel_hang.cpp")
FUNC_AIC_HANG = 0


def _build_chip_callable(platform: str) -> ChipCallable:
    kc = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root()
    inc_dirs = kc.get_orchestration_include_dirs(RUNTIME)

    orch_bytes = kc.compile_orchestration(runtime_name=RUNTIME, source_path=ORCH_SRC)
    aic_bytes = kc.compile_incore(AIC_SRC, core_type="aic", pto_isa_root=pto_isa_root, extra_include_dirs=inc_dirs)
    # Onboard expects the .text section of the AIC ELF; sim consumes the full ELF.
    if not platform.endswith("sim"):
        aic_bytes = extract_text_section(aic_bytes)

    aic_core = CoreCallable.build(signature=[], binary=aic_bytes)
    return ChipCallable.build(
        signature=[],
        func_name="aicpu_orchestration_entry",
        binary=orch_bytes,
        children=[(FUNC_AIC_HANG, aic_core)],
    )


@pytest.mark.platforms(["a2a3"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
@pytest.mark.timeout(60)
def test_aicore_op_timeout_surfaces_as_runtime_error(st_platform, st_device_ids):
    configure_logging("error")

    chip_callable = _build_chip_callable(st_platform)
    worker = Worker(level=2, platform=st_platform, runtime=RUNTIME, device_id=int(st_device_ids[0]))
    cid = worker.register(chip_callable)
    worker.init()
    try:
        config = CallConfig()
        config.block_dim = 1
        # >=2 so the orchestration thread and the scheduler thread don't fight
        # for a single AICPU; smaller configs may not dispatch the AIC task.
        config.aicpu_thread_num = 2

        t0 = time.monotonic()
        # 507046 = ACL_ERROR_RT_STREAM_SYNC_TIMEOUT — what
        # aclrtSynchronizeStreamWithTimeout returns when the AICore stream
        # (carrying the STARS-killed op) doesn't drain within the host's 2 s
        # budget. Observed elapsed on Ascend910 / a2a3 onboard: ~6.3 s.
        with pytest.raises(RuntimeError, match=r"run_prepared failed with code 507046"):
            worker.run(cid, ChipStorageTaskArgs(), config)
        elapsed = time.monotonic() - t0

        # STARS 1 s + AICPU deinit 1 s + host 2 s/stream — observed ~6 s.
        # If this fires, the timeout chain is broken (or absent).
        assert elapsed < 10, f"run() took {elapsed:.1f}s — timeout chain did not fire"
    finally:
        worker.close()
